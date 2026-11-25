"""推理后端选择：Mac 走 MLX（原有实现），Windows/Linux + NVIDIA 走 CUDA（PyTorch）。

为什么要分叉
------------
Mac 端的模型全部是 MLX 格式（``mlx-community/*-4bit``、``mflux-community/*``、
``dgrauet/ltx-*-mlx-*``），CUDA 读不了这些权重；反过来 PyTorch 权重 Mac 也没有
Metal 加速的等价实现。两边不是「同一份代码换个 flag」，而是**两套权重 + 两套运行时**。

因此策略是：**公共逻辑（项目/分镜/时间轴/混音/组装）留在 ``pipeline/`` 顶层，
只把「跑模型」这一个动作按后端分叉**，这样 Mac 端行为零变化。

选择规则（``config.json`` 的 ``backend`` 字段）
---------------------------------------------
- ``"auto"``（默认）：macOS → mlx；Windows/Linux 且有 CUDA → cuda；其余 → mlx
- ``"mlx"``：强制 MLX（仅 macOS 有意义）
- ``"cuda"``：强制 CUDA（需要 torch + CUDA）

判定 CUDA 是否可用时**不 import torch**（``doctor`` 在未装 torch 的环境也要能跑），
先看 ``nvidia-smi``，真正用到时再 import。
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Literal

BackendName = Literal["mlx", "cuda"]

#: 已知后端名，便于配置校验
KNOWN_BACKENDS = ("auto", "mlx", "cuda")


@dataclass(frozen=True)
class BackendInfo:
    """后端判定结果。"""

    name: BackendName
    reason: str
    cuda_available: bool = False
    nvidia_vram_gb: float | None = None

    @property
    def is_mlx(self) -> bool:
        return self.name == "mlx"

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def _cuda_hardware_present() -> tuple[bool, float | None]:
    """是否有 NVIDIA 卡。不 import torch。"""
    from pipeline import platform as plat

    vram = plat.nvidia_vram_bytes()
    if not vram:
        return False, None
    return True, round(vram / 1024**3, 2)


def _torch_cuda_ready() -> bool:
    """torch 是否装了且 CUDA 可用。只在确实要跑模型时调用。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_backend(config_value: str | None = None) -> BackendInfo:
    """按配置 + 本机能力决定后端。

    ``config_value`` 来自 ``config.json`` 的 ``backend``（默认 ``"auto"``）。
    """
    want = (config_value or os.environ.get("NOVEL_VIDEO_BACKEND") or "auto").strip().lower()
    if want not in KNOWN_BACKENDS:
        raise SystemExit(
            f"未知 backend：{want!r}（可选 {', '.join(KNOWN_BACKENDS)}）"
        )

    has_card, vram_gb = _cuda_hardware_present()

    if want == "mlx":
        if not sys.platform == "darwin":
            # 强制指定就尊重，但提示：MLX 在非 macOS 上不可用
            return BackendInfo(
                name="mlx",
                reason="配置强制 mlx（注意：MLX 仅在 macOS/Apple Silicon 可用）",
                cuda_available=has_card,
                nvidia_vram_gb=vram_gb,
            )
        return BackendInfo(name="mlx", reason="配置强制 mlx", cuda_available=has_card,
                           nvidia_vram_gb=vram_gb)

    if want == "cuda":
        if not has_card:
            raise SystemExit(
                "配置要求 cuda 后端，但本机没找到 NVIDIA 显卡（nvidia-smi 不可用）。"
                "请检查驱动，或把 config.json 的 backend 改成 auto/mlx。"
            )
        return BackendInfo(
            name="cuda",
            reason="配置强制 cuda",
            cuda_available=True,
            nvidia_vram_gb=vram_gb,
        )

    # ---- auto ----
    if sys.platform == "darwin":
        return BackendInfo(
            name="mlx",
            reason="macOS/Apple Silicon：用 MLX",
            cuda_available=has_card,
            nvidia_vram_gb=vram_gb,
        )
    if has_card:
        return BackendInfo(
            name="cuda",
            reason=f"检测到 NVIDIA 显卡（{vram_gb:.1f} GB）：用 CUDA",
            cuda_available=True,
            nvidia_vram_gb=vram_gb,
        )
    return BackendInfo(
        name="mlx",
        reason="未检测到 NVIDIA 显卡，回退 MLX（本机大概率跑不动）",
        cuda_available=False,
        nvidia_vram_gb=None,
    )


def require_cuda(*, what: str = "该阶段") -> None:
    """CUDA 后端要用模型前的硬校验，给出可操作的报错。"""
    try:
        import torch
    except ImportError:
        raise SystemExit(
            f"{what}需要 PyTorch CUDA，但当前环境没有 torch。\n"
            "请运行 scripts/setup-win.ps1，或在现有 venv 里安装：\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            f"{what}需要 CUDA，但 torch.cuda.is_available() 为 False。\n"
            f"torch={torch.__version__}，CUDA={torch.version.cuda}\n"
            "请确认：1) 显卡驱动正常；2) 装的是 CUDA 版 torch 而不是 CPU 版。"
        )


def describe(config_value: str | None = None) -> dict:
    """给 doctor / 网页用的一份后端说明。"""
    info = resolve_backend(config_value)
    detail = {
        "backend": info.name,
        "reason": info.reason,
        "cuda_hardware": info.cuda_available,
        "nvidia_vram_gb": info.nvidia_vram_gb,
    }
    if info.is_cuda:
        detail["torch_cuda_ready"] = _torch_cuda_ready()
    return detail
