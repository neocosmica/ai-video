"""MLX 显存工具：把 Mac 端原有的清理/统计逻辑集中到这里。

Mac 端原本分散在各处（``ltx_core_mlx.utils.memory`` 里的 ``aggressive_cleanup``
与 ``get_memory_stats``，以及 ``storyboard_gen`` 里的 ``mx.clear_cache()``）。
这里做一层统一封装，让 ``pipeline/backends`` 两侧有对称的接口：

| 能力        | MLX (Mac)                    | CUDA (Windows/Linux)         |
|------------|------------------------------|------------------------------|
| 清理        | gc + ``mx.clear_cache()``    | gc + ``torch.cuda.empty_cache()`` |
| 活跃显存     | ``mx.get_active_memory()``   | ``torch.cuda.memory_allocated()`` |
| 峰值        | ``mx.get_peak_memory()``     | ``torch.cuda.max_memory_allocated()``|
| 缓存        | ``mx.get_cache_memory()``    | ``torch.cuda.memory_reserved()`` |
| 设备可用     | ``mx.metal.is_available()``  | ``torch.cuda.is_available()`` |

行为与原来一致：只在 macOS + MLX 可用时生效，其他地方安静跳过。
"""

from __future__ import annotations

import gc
from typing import Any


def _mx():
    """延迟 import mlx；非 Mac 环境返回 None。"""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx


def empty_cache() -> None:
    """对齐 ``aggressive_cleanup()``：gc + 清 Metal 缓存。"""
    gc.collect()
    mx = _mx()
    if mx is None:
        return
    try:
        mx.clear_cache()
    except Exception:
        pass


def memory_stats() -> dict[str, float]:
    """当前显存（GiB）；无 MLX 时返回全 0。"""
    mx = _mx()
    if mx is None:
        return {"active_gib": 0.0, "peak_gib": 0.0, "cache_gib": 0.0}
    to_gib = 1.0 / (1024**3)
    try:
        return {
            "active_gib": round(mx.get_active_memory() * to_gib, 2),
            "peak_gib": round(mx.get_peak_memory() * to_gib, 2),
            "cache_gib": round(mx.get_cache_memory() * to_gib, 2),
        }
    except Exception:
        return {"active_gib": 0.0, "peak_gib": 0.0, "cache_gib": 0.0}


def reset_peak() -> None:
    """重置峰值统计；MLX 无对应 API 时跳过。"""
    mx = _mx()
    if mx is None:
        return
    for name in ("reset_peak_memory", "reset_peak_memory_stats"):
        fn = getattr(mx, name, None)
        if fn is not None:
            try:
                fn()
                return
            except Exception:
                continue


def metal_available() -> bool:
    """MLX Metal 后端是否可用（Mac 上为 True）。"""
    mx = _mx()
    if mx is None:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False
