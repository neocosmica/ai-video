"""CUDA 显存编排：分阶段插载、段间释放。

为什么需要这个
--------------
12 GB 卡上跑不通「所有模型同时驻留」。Mac 端靠 ``scripts/apply_ltx_gemma_patch.py``
把 Gemma 文本塔与 connector 拆成两段、各段用完立刻 ``aggressive_cleanup()``，
把峰值从 ~14 GB 压到「max(单段)」。CUDA 侧等价做法是：

1. 一次只把**当前阶段**的模型搬上 GPU
2. 阶段结束 ``del`` + ``gc.collect()`` + ``torch.cuda.empty_cache()``
3. 需要时可降级到逐层 offload（权重留 CPU，按 forward 逐层上卡）

实测（RTX 4070 12 GB，LTX-2 19B Q2_K 8.82→7.54 GB）：

    载入 transformer          free 3.12 GB
    前向通过（1680 tokens）
    释放                     free 10.78 GB   ← 完全回收
    载入 VAE                  free 8.47 GB

峰值 = max(单段)，不累加。这是整套方案能在 12 GB 上成立的基础。

与 Mac 的对应关系
-----------------
| Mac (MLX)                     | CUDA (PyTorch)                  |
|-------------------------------|----------------------------------|
| ``gc.collect()``              | ``gc.collect()``                 |
| ``mx.clear_cache()``          | ``torch.cuda.empty_cache()``     |
| ``mx.get_active_memory()``    | ``torch.cuda.memory_allocated()``|
| ``mx.get_peak_memory()``      | ``torch.cuda.max_memory_allocated()``|
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Any, Iterator, Callable

#: 单卡保留余量：低于此值就告警，避免下一个阶段直接 OOM
VRAM_SAFETY_GIB = 1.0


def _torch():
    """延迟 import：只有真跑模型才需要。"""
    import torch

    return torch


def empty_cache() -> None:
    """释放 Python 引用与 CUDA 缓存。

    对齐 Mac 的 ``aggressive_cleanup()``：``gc.collect()`` + 清空 allocator 缓存。
    """
    gc.collect()
    try:
        torch = _torch()
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def memory_stats() -> dict[str, float]:
    """当前显存占用（GiB），对齐 Mac 的 ``get_memory_stats()``。"""
    torch = _torch()
    if not torch.cuda.is_available():
        return {"active_gib": 0.0, "peak_gib": 0.0, "free_gib": 0.0, "total_gib": 0.0}
    to_gib = 1.0 / (1024**3)
    free, total = torch.cuda.mem_get_info()
    return {
        "active_gib": round(torch.cuda.memory_allocated() * to_gib, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() * to_gib, 2),
        "free_gib": round(free * to_gib, 2),
        "total_gib": round(total * to_gib, 2),
    }


def reset_peak() -> None:
    """每进入一个阶段前重置峰值统计，便于分阶段报显存。"""
    try:
        torch = _torch()
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def ensure_headroom(min_free_gib: float = VRAM_SAFETY_GIB, *, label: str = "") -> float:
    """上卡前检查余量；不足则先清理并返回新的可用量（不停机）。"""
    empty_cache()
    stats = memory_stats()
    free = stats.get("free_gib", 0.0)
    if free < min_free_gib:
        import sys

        print(
            f"警告：{label or '当前阶段'}可用显存仅 {free:.2f} GiB"
            f"（建议 ≥ {min_free_gib:.2f}）。可能 OOM，继续尝试。",
            file=sys.stderr,
            flush=True,
        )
    return free


@contextmanager
def stage(label: str, *, min_free_gib: float = VRAM_SAFETY_GIB) -> Iterator[dict[str, Any]]:
    """一个阶段的显存上下文：进入前清理，退出时释放并打印显存。

    用法::

        with stage("videos/transformer") as info:
            model = load_transformer(...)       # 上卡
            result = model(...)                 # 跑
        # 离开 with 时自动 del 引用 → 释放

    注意：``with`` 不会自动 ``del`` 你持有的引用，需在块内显式 ``del model``
    或把模型定义在块内（离开作用域后引用计数归零）。
    """
    reset_peak()
    ensure_headroom(min_free_gib, label=label)
    info: dict[str, Any] = {"label": label}
    try:
        yield info
    finally:
        empty_cache()
        stats = memory_stats()
        info.update(stats)
        print(
            f"[显存] {label}: peak={stats['peak_gib']:.2f} GiB "
            f"free={stats['free_gib']:.2f}/{stats['total_gib']:.2f} GiB",
            flush=True,
        )


def release(*objs: Any) -> None:
    """显式丢弃引用并清缓存（在 ``del`` 之后调用）。"""
    for obj in objs:
        del obj
    empty_cache()


def dtype_for_torch():
    """默认计算 dtype：bf16（Ada/Hopper 原生支持，省一半显存）。"""
    torch = _torch()
    return torch.bfloat16


def offload_layers_hook(model: Any) -> None:
    """逐层 offload：权重留 CPU，forward 时按子模块上卡。

    用于「比 12 GB 大的档位」（如 LTX-2 19B Q4_K_M 13 GB）：把 transformer
    权重放主机内存，GPU 上永远只驻留当前一层。代价是 PCIe 来回搬，明显变慢。

    依赖 ``accelerate``；某些 GGUF 参数不适用时该方法会跳过该模块，
    调用方应检查返回值（返回成功挂载的模块数）。
    """
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    torch = _torch()
    device = torch.device("cuda:0")
    hook = AlignDevicesHook(execution_device=device, offload=True)
    mounted = 0
    for _name, module in model.named_children():
        try:
            add_hook_to_module(module, hook, append=True)
            mounted += 1
        except Exception:
            continue
    empty_cache()
    return mounted
