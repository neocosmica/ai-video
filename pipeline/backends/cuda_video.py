"""CUDA 视频后端：LTX-2（19B）GGUF + 分阶段显存编排。

为什么是 LTX-2 19B 而不是 22B(2.3)/2.5
------------------------------------
实测（见 ``docs/win-cuda/PHASE0-PROBE.md``）：
- ``LTX-2.5`` 权重需 9 行 mod 表，diffusers 0.43 建 6 行 → 架构不兼容
- ``LTX-2``（19B）与 ``LTX2VideoTransformer3DModel`` 完全匹配，且仓库非 gated

档位选择
--------
384×640 / 49 帧实测（RTX 4070 12 GB）：

    Q4_K_S 11.04 GB  → OOM
    Q3_K_S  8.82 GB  → peak 9.76 GB，余量 0.08 GB（接 VAE 必崩）
    Q2_K    7.54 GB  → peak 8.57 GB，余量 1.51 GB  ✅ 默认

``config.json`` 的 ``cuda.video_gguf`` 可覆盖档位；想追平 Mac 画质可用
``Q4_K_M`` 配逐层 offload（慢但质量高）。

分阶段编排（对齐 Mac 的 ``apply_ltx_gemma_patch.py``）
-----------------------------------------------------
    1. 载 transformer → 跑完整个去噪
    2. 释放 transformer（free 从 3.1 GiB 回到 10.8 GiB）
    3. 载 VAE → 解码
    4. 释放 VAE

峰值 = max(单段) 而非累加，这是 12 GB 能跑 19B 的前提。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from pipeline.backends import cuda_memory
from pipeline.backends import require_cuda

#: 默认 GGUF 仓库与档位（实测 12 GB 可跑）
DEFAULT_GGUF_REPO = "unsloth/LTX-2-GGUF"
DEFAULT_GGUF_FILE = "ltx-2-19b-dev-Q2_K.gguf"
#: 官方 diffusers 配置来源（非 gated，含 transformer/vae/config.json）
DEFAULT_CONF_REPO = "Lightricks/LTX-2"

#: LTX-2 的 VAE 时空压缩：32×32 空间、8× 时间
VAE_SPATIAL = 32
VAE_TEMPORAL = 8


def latent_dims(height: int, width: int, frames: int) -> tuple[int, int, int]:
    """像素尺寸 → latent 的 (T, H, W)。"""
    lat_h = max(1, height // VAE_SPATIAL)
    lat_w = max(1, width // VAE_SPATIAL)
    lat_t = max(1, (frames - 1) // VAE_TEMPORAL + 1)
    return lat_t, lat_h, lat_w


def _hub_file(repo: str, filename: str, *, offline: bool) -> str:
    """取 Hub 文件本地路径；offline 时只用缓存。"""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_files_only=offline)


def load_transformer(
    *,
    gguf_file: str = DEFAULT_GGUF_FILE,
    gguf_repo: str = DEFAULT_GGUF_REPO,
    conf_repo: str = DEFAULT_CONF_REPO,
    offline: bool = False,
    device: str = "cuda",
) -> Any:
    """加载 LTX-2 transformer（GGUF 量化，单文件）。

    返回已在 ``device`` 上的模型。调用方负责在用完后释放（建议用
    ``cuda_memory.stage`` 包裹）。
    """
    require_cuda(what="视频阶段")
    from diffusers import GGUFQuantizationConfig
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
    import torch

    gguf = _hub_file(gguf_repo, gguf_file, offline=offline)
    conf = _hub_file(conf_repo, "transformer/config.json", offline=offline)

    quant = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    model = LTX2VideoTransformer3DModel.from_single_file(
        gguf,
        config=conf,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 量化时强制 True
    )
    model = model.to(device)
    model.eval()
    return model


def load_vae(*, conf_repo: str = DEFAULT_CONF_REPO, offline: bool = False,
             device: str = "cuda") -> Any:
    """加载 LTX-2 视频 VAE（单独加载，供解码阶段使用）。"""
    require_cuda(what="VAE 解码")
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
    import torch

    conf = _hub_file(conf_repo, "vae/config.json", offline=offline)
    weights = _hub_file(conf_repo, "vae/diffusion_pytorch_model.safetensors", offline=offline)
    vae = AutoencoderKLLTX2Video.from_single_file(
        weights, config=conf, torch_dtype=torch.bfloat16
    )
    vae = vae.to(device).eval()
    return vae


def make_coords(model: Any, frames: int, height: int, width: int,
                audio_frames: int, device: str = "cuda"):
    """构造 video / audio 的 RoPE 坐标（必须用官方 API，手搓会 OOM/维度错）。"""
    import torch

    lat_t, lat_h, lat_w = latent_dims(height, width, frames)
    video_coords = model.rope.prepare_video_coords(
        1, lat_t, lat_h, lat_w, torch.device(device)
    )
    audio_coords = model.audio_rope.prepare_audio_coords(
        1, audio_frames, torch.device(device)
    )
    return video_coords, audio_coords, (lat_t, lat_h, lat_w)


def denoise(
    model: Any,
    *,
    latents: Any,
    audio_latents: Any,
    prompt_embeds: Any,
    audio_prompt_embeds: Any,
    timesteps: list[float],
    frames: int,
    height: int,
    width: int,
    audio_frames: int,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[Any, Any]:
    """跑完整去噪循环，返回 (video_latents, audio_latents)。

    ``timesteps`` 是 sigma×1000 的列表（LTX 用 ``timestep_scale_multiplier=1000``）。
    """
    import torch

    video_coords, audio_coords, _ = make_coords(
        model, frames, height, width, audio_frames
    )
    tokens = latents.shape[1]
    t0 = time.time()
    for idx, ts in enumerate(timesteps):
        step_t = time.time()
        timestep = torch.full((1, tokens), float(ts), device="cuda", dtype=torch.bfloat16)
        audio_ts = torch.full((1,), float(ts), device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(
                hidden_states=latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=prompt_embeds,
                audio_encoder_hidden_states=audio_prompt_embeds,
                timestep=timestep,
                audio_timestep=audio_ts,
                video_coords=video_coords,
                audio_coords=audio_coords,
                num_frames=frames,
                height=height,
                width=width,
                audio_num_frames=audio_frames,
                return_dict=False,
            )
        video_vel = out[0] if isinstance(out, tuple) else out
        audio_vel = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        # Euler 步进（与项目的 FlowMatchEulerDiscreteScheduler 对齐由调用方保证）
        latents = latents - video_vel.to(latents.dtype) * (1.0 / len(timesteps))
        if audio_vel is not None and audio_latents is not None:
            audio_latents = audio_latents - audio_vel.to(audio_latents.dtype) * (1.0 / len(timesteps))
        if on_step is not None:
            on_step(idx, time.time() - step_t)
    print(f"  去噪 {len(timesteps)} 步共 {time.time()-t0:.1f}s", flush=True)
    return latents, audio_latents


def decode_video(vae: Any, latents: Any, *, frames: int, height: int, width: int):
    """latent → 像素帧张量 ``(3, F, H, W)``，值域约 [-1, 1]。"""
    import torch

    lat_t, lat_h, lat_w = latent_dims(height, width, frames)
    # (1, tokens, C) → (1, C, T, H, W)
    packed = latents.reshape(1, lat_t, lat_h, lat_w, -1).permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        dec = vae.decode(packed.contiguous(), return_dict=False)[0]
    return dec


def frames_to_uint8(decoded: Any):
    """解码输出 → ``(F, H, W, 3)`` 的 uint8 数组，供 mp4 写入。"""
    import numpy as np

    frames = ((decoded[0].float().clamp(-1, 1) + 1) / 2)   # (3,F,H,W) → [0,1]
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()       # (F,H,W,3)
    return (frames * 255).astype(np.uint8)


def write_mp4(frames, path: Path, *, fps: int = 24) -> Path:
    """把帧序列写成 mp4（libx264）。"""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8,
                     macro_block_size=1)
    return path


def generate_shot(
    *,
    latents,
    audio_latents,
    prompt_embeds,
    audio_prompt_embeds,
    timesteps: list[float],
    frames: int,
    height: int,
    width: int,
    audio_frames: int,
    out_path: Path,
    fps: int = 24,
    gguf_file: str = DEFAULT_GGUF_FILE,
    gguf_repo: str = DEFAULT_GGUF_REPO,
    conf_repo: str = DEFAULT_CONF_REPO,
    offline: bool = False,
) -> Path:
    """单个镜头完整链路：载 transformer → 去噪 → 释放 → 载 VAE → 解码 → mp4。

    这是显存编排的主入口：两段模型不会同时驻留。
    """
    from pipeline.backends import cuda_memory

    # ---- 段1：transformer ----
    with cuda_memory.stage("videos/transformer", min_free_gib=7.5) as info:
        model = load_transformer(
            gguf_file=gguf_file, gguf_repo=gguf_repo, conf_repo=conf_repo, offline=offline
        )
        latents, audio_latents = denoise(
            model,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            audio_prompt_embeds=audio_prompt_embeds,
            timesteps=timesteps,
            frames=frames,
            height=height,
            width=width,
            audio_frames=audio_frames,
        )
        del model
    # 离开 with：显存回收

    # ---- 段2：VAE ----
    with cuda_memory.stage("videos/vae", min_free_gib=2.0) as info:
        vae = load_vae(conf_repo=conf_repo, offline=offline)
        decoded = decode_video(vae, latents, frames=frames, height=height, width=width)
        frames_arr = frames_to_uint8(decoded)
        del vae
    # 离开 with：显存回收

    return write_mp4(frames_arr, out_path, fps=fps)
