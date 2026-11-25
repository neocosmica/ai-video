"""CUDA 图像后端：FLUX.2 Klein 4B（diffusers 原生管道 + CPU offload）。

实测（RTX 4070 12 GB，576×1024，4 步）：
    加载 244 s（含下载），出图 15 s，峰值 7.80 GB，余量 10.78 GB

为什么是 FLUX.2 Klein 4B 而不是 9B / FLUX.3
------------------------------------------
- FLUX.2-klein-4B：``gated=False``，可直接下载，diffusers 有 ``Flux2KleinPipeline``
- FLUX.2-klein-9B / FLUX.3：``gated: auto``，无 HF token 时 403
- FLUX.3 GGUF 是 ComfyUI 命名（``content_mode_blocks.image.*``），diffusers
  ``Flux3Transformer2DModel`` 无对应映射表，且官方 config 也 gated

因此默认用 4B；要上 9B 需先在 HF 账号接受许可并配置 token。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.backends import cuda_memory, require_cuda

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def load_pipeline(*, model: str = DEFAULT_MODEL, device: str = "cuda") -> Any:
    """加载 Flux2Klein 管道并启用组件级 CPU offload。

    ``enable_model_cpu_offload`` 会按 ``text_encoder->transformer->vae``
    顺序把组件轮流搬上 GPU，等价于 Mac 端的分阶段加载。
    """
    require_cuda(what="图像阶段")
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def generate(
    pipe: Any,
    *,
    prompt: str,
    height: int,
    width: int,
    steps: int = 4,
    seed: int = 0,
    guidance: float = 1.0,
) -> Any:
    """出图，返回 PIL.Image。"""
    import torch

    if cuda_memory.memory_stats().get("free_gib", 0) < 6.0:
        cuda_memory.ensure_headroom(6.0, label="image")
    gen = torch.Generator("cuda").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=gen,
    )
    return result.images[0]


def save(image: Any, path: Path) -> Path:
    """落盘 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    image.save(path)
    return path
