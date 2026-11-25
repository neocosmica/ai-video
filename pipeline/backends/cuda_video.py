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


def load_connectors(*, conf_repo: str = DEFAULT_CONF_REPO, offline: bool = False,
                    device: str = "cuda") -> Any:
    """加载 LTX-2 文本 connectors（Gemma 全部层 → transformer 的 prompt_embeds）。

    没有这一步，直接把 Gemma 最后一层喂给 transformer 会得到网格噪声画面
    （实测）。权重约 2.67 GB。
    """
    require_cuda(what="文本 connector 阶段")
    import torch
    from diffusers.models.condition_embedders.condition_embedder_ltx2 import LTX2TextConnectors

    conf = _hub_file(conf_repo, "connectors/config.json", offline=offline)
    weights = _hub_file(conf_repo, "connectors/diffusion_pytorch_model.safetensors",
                        offline=offline)
    from pathlib import Path as _P

    conn = LTX2TextConnectors.from_pretrained(_P(weights).parent, torch_dtype=torch.bfloat16)
    return conn.to(device).eval()


def encode_prompt_with_connectors(
    connectors: Any,
    *,
    hidden: Any,
    mask: Any,
    scale_factor: int = 8,
) -> tuple[Any, Any]:
    """Gemma 全部层 + connectors → (video_embeds, audio_embeds)。"""
    import torch

    h = hidden.to("cuda", dtype=torch.bfloat16)
    m = mask.to("cuda") if mask is not None else torch.ones(h.shape[:2], device="cuda", dtype=torch.long)
    with torch.no_grad():
        out = connectors(h, m, padding_side="left", scale_factor=int(scale_factor))
    # 官方返回 (video_embeds, audio_embeds, ...)
    video = out[0] if isinstance(out, (tuple, list)) else out
    audio = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else video
    del h, m
    return video, audio


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
    steps: int,
    frames: int,
    height: int,
    width: int,
    audio_frames: int,
    conditioning_mask: Any | None = None,
    conditioning_latents: Any | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[Any, Any]:
    """跑完整去噪循环，返回 (video_latents, audio_latents)。

    ``steps`` 是去噪步数；timestep 表由官方 FlowMatchEulerDiscreteScheduler 生成。
    """
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    video_coords, audio_coords, _ = make_coords(
        model, frames, height, width, audio_frames
    )
    tokens = latents.shape[1]

    # Match diffusers LTX2 I2V: token-count shift and Euler sigmas.
    import numpy as np
    from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift

    scheduler = FlowMatchEulerDiscreteScheduler()
    audio_scheduler = FlowMatchEulerDiscreteScheduler()
    sigmas = np.linspace(1.0, 1.0 / int(steps), int(steps))
    mu = calculate_shift(
        tokens,
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    scheduler.set_timesteps(int(steps), device="cuda", sigmas=sigmas, mu=mu)
    audio_scheduler.set_timesteps(int(steps), device="cuda", sigmas=sigmas, mu=mu)
    if conditioning_mask is None:
        conditioning_mask = torch.zeros((latents.shape[0], tokens), device="cuda", dtype=torch.bfloat16)

    t0 = time.time()
    for idx, t in enumerate(scheduler.timesteps):
        step_t = time.time()
        audio_t = audio_scheduler.timesteps[idx]
        # Official I2V timestep mask: conditioned first-frame tokens receive timestep 0.
        timestep = t.expand(latents.shape[0], tokens) * (1 - conditioning_mask)
        audio_ts = audio_t.expand(audio_latents.shape[0])
        with torch.no_grad():
            out = model(
                hidden_states=latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=prompt_embeds,
                audio_encoder_hidden_states=audio_prompt_embeds,
                timestep=timestep.to(torch.bfloat16),
                audio_timestep=audio_ts.to(torch.bfloat16),
                sigma=timestep.to(torch.bfloat16),
                video_coords=video_coords,
                audio_coords=audio_coords,
                num_frames=frames,
                height=height,
                width=width,
                audio_num_frames=audio_frames,
                return_dict=False,
            )
        # LTX-2 transformer returns velocity (v), not x0. The official I2V pipeline
        # converts v<->x0 only when applying guidance; without extra guidance, feed v
        # directly into FlowMatchEulerDiscreteScheduler.step.
        video_velocity = out[0].float() if isinstance(out, tuple) else out.float()
        audio_velocity = out[1].float() if isinstance(out, tuple) and len(out) > 1 else None
        # Packed token order is [T,H,W]. Keep the first latent frame (H*W tokens)
        # conditioned on the source still; only denoise following frames.
        first_frame_tokens = latent_dims(height, width, frames)[1] * latent_dims(height, width, frames)[2]
        pred = scheduler.step(
            video_velocity[:, first_frame_tokens:], t,
            latents[:, first_frame_tokens:].float(), return_dict=False
        )[0].to(latents.dtype)
        latents = torch.cat([latents[:, :first_frame_tokens], pred], dim=1)
        if audio_velocity is not None and audio_latents is not None:
            audio_latents = audio_scheduler.step(
                audio_velocity, audio_t, audio_latents.float(), return_dict=False
            )[0].to(audio_latents.dtype)
        if on_step is not None:
            on_step(idx, time.time() - step_t)
    print(f"  去噪 {int(steps)} 步共 {time.time()-t0:.1f}s", flush=True)
    return latents, audio_latents


def _normalize_latents(z, vae):
    """按 VAE 的 latents_mean/std 归一化（缺失这一步画面会错乱）。"""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1)
    std = vae.latents_std.view(1, -1, 1, 1, 1)
    return (z - mean) * vae.config.scaling_factor / std


def _denormalize_latents(z, vae):
    mean = vae.latents_mean.view(1, -1, 1, 1, 1)
    std = vae.latents_std.view(1, -1, 1, 1, 1)
    return z * std / vae.config.scaling_factor + mean


def _pack_latents(z, patch_size: int = 1, patch_size_t: int = 1):
    """[B,C,F,H,W] → [B, tokens, C]（与 diffusers 官方实现一致）。"""
    b, c, f, h, w = z.shape
    pf, ph, pw = f // patch_size_t, h // patch_size, w // patch_size
    z = z.reshape(b, -1, pf, patch_size_t, ph, patch_size, pw, patch_size)
    return z.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def _unpack_latents(tokens, num_frames: int, height: int, width: int,
                    patch_size: int = 1, patch_size_t: int = 1):
    """[B, tokens, C] → [B,C,F,H,W]。"""
    b = tokens.size(0)
    z = tokens.reshape(b, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    z = z.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return z


def encode_frame_to_latents(vae: Any, image_path: Path, *, frames: int,
                            height: int, width: int) -> Any:
    """把静帧编码成初始 latent（I2V 起点）。

    必须走「VAE encode → normalize → pack」三步，缺 normalize 会让
    画面出现网格状伪影（实测）。返回 ``(1, tokens, C)``。
    """
    import numpy as np
    import torch
    from PIL import Image

    lat_t, lat_h, lat_w = latent_dims(height, width, frames)

    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    frame = torch.from_numpy(arr).permute(2, 0, 1)[None, :, None, :, :].to("cuda", dtype=torch.bfloat16)

    # Official I2V 先编码单帧 image latent，再沿时间轴复制；不要把一张图
    # 复制成 49 帧送入 causal VAE，否则时间压缩窗口错位，首帧重建会出现伪影。
    with torch.no_grad():
        posterior = vae.encode(frame, causal=True, return_dict=False)[0]
        z0 = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
        z0 = z0[:, :, :1, :lat_h, :lat_w]
        z0 = _normalize_latents(z0, vae).to(torch.bfloat16)
        z = z0.repeat(1, 1, lat_t, 1, 1)
        return _pack_latents(z).contiguous()


def decode_video(vae: Any, latents: Any, *, frames: int, height: int, width: int):
    """latent → 像素帧张量 ``(3, F, H, W)``，值域约 [-1, 1]。

    必须走「unpack → denormalize → VAE decode」，与编码对称。
    """
    import torch

    lat_t, lat_h, lat_w = latent_dims(height, width, frames)
    with torch.no_grad():
        z = _unpack_latents(latents.float(), lat_t, lat_h, lat_w)
        z = _denormalize_latents(z, vae).to(torch.bfloat16)
        dec = vae.decode(z.contiguous(), return_dict=False)[0]
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
    steps: int,
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
            steps=steps,
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


# ---------------------------------------------------------------------------
# 命令入口（与 mac 端 pipeline/video.py 对齐）
# ---------------------------------------------------------------------------


def cmd_videos_cuda(settings, args) -> None:
    """CUDA 视频阶段：逐镜生成 clips/<id>.mp4。

    每镜内部再分两段（transformer → 释放 → VAE），保证 12 GB 卡不爆。
    """
    import random

    import torch

    from pipeline.projects import artifacts_sub
    from pipeline.util import load_json, resolve, snap_frames

    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)

    clips = artifacts_sub(out, "clips")
    clips.mkdir(parents=True, exist_ok=True)
    only = getattr(args, "only", None)
    force = getattr(args, "force", False)
    wanted = None
    if only:
        wanted = {x.strip() for x in str(only).replace("，", ",").split(",") if x.strip()}

    width = settings.video_width
    height = settings.video_height
    fps = settings.video_fps
    gguf_file = getattr(settings, "cuda_video_gguf", DEFAULT_GGUF_FILE)
    gguf_repo = getattr(settings, "cuda_video_gguf_repo", DEFAULT_GGUF_REPO)
    conf_repo = getattr(settings, "cuda_conf_repo", DEFAULT_CONF_REPO)

    print(f"视频后端：CUDA / {gguf_file}  {width}x{height} @{fps}fps", flush=True)

    # 文本编码器只加载一次（编码多个 prompt 后释放）
    prompts: list[tuple[str, callable]] = []
    encoded_cache: dict[str, any] = {}

    failures = 0
    for shot in story["shots"]:
        if wanted and str(shot["id"]) not in wanted:
            continue
        dest = clips / f"{shot['id']}.mp4"
        if dest.exists() and not force:
            print(f"已有镜头 {shot['id']}，跳过", flush=True)
            continue

        image_path = artifacts_sub(out, "frames", f"{shot['id']}.png")
        if not image_path.exists():
            reuse = str(shot.get("reuse_frame") or "").strip()
            if reuse:
                image_path = artifacts_sub(out, "frames", f"{reuse}.png")
        if not image_path.exists():
            shot["video_error"] = "缺少静帧"
            failures += 1
            print(f"跳过 {shot['id']}：没有静帧", flush=True)
            continue

        line = str(shot.get("dialogue") or "").strip()
        voice = artifacts_sub(out, "audio", "voice", f"{shot['id']}.wav")
        audio_latents = None
        audio_embeds = None
        if line:
            if not voice.exists():
                shot["video_error"] = "有对白但缺少 wav，请先跑 audio"
                failures += 1
                print(f"跳过 {shot['id']}：缺配音", flush=True)
                continue

        frames = _shot_frames(shot, settings)
        prompt = _prompt_cuda(shot)
        seed = int(shot.get("seed") or random.Random(shot["id"]).randint(0, 2**31))

        print(f"\n=== 镜头 {shot['id']}  {frames} 帧 / {frames/fps:.1f}s ===", flush=True)
        try:
            _generate_shot_cuda(
                settings, shot, image_path, dest,
                prompt=prompt, frames=frames, height=height, width=width, fps=fps,
                seed=seed, has_audio=bool(line),
                gguf_file=gguf_file, gguf_repo=gguf_repo, conf_repo=conf_repo,
            )
            print(f"  -> {dest}", flush=True)
        except Exception as exc:  # noqa: BLE001 — 单镜失败不中断整片
            shot["video_error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            print(f"  失败：{shot['video_error']}", flush=True)

    if failures:
        raise SystemExit(f"{failures} 个镜头失败，见分镜里的 video_error")
    print(f"镜头在 {clips}", flush=True)


def _prompt_cuda(shot: dict) -> str:
    """与 mac 端 ``_prompt`` 同一套措辞（保证两端视觉提示词一致）。"""
    motion = shot.get("motion_prompt") or "Slow cinematic push in"
    if str(shot.get("dialogue") or "").strip():
        return (
            f"{motion}. The on-screen speaker talks naturally with clear lip sync to "
            "the dialogue audio. Keep the face, hair and clothes identical to the first frame. "
            "Stable identity, natural mouth motion, no morphing, no scene change."
        )
    return (
        f"{motion}. Keep the face, hair and clothes identical to the first frame. "
        "Stable identity, natural motion, no morphing, no scene change. "
        "Ambient room tone only, no spoken words."
    )


def _shot_frames(shot: dict, settings) -> int:
    """镜长：分镜里的 video_frames 优先，否则用配置默认。"""
    from pipeline.util import snap_frames

    raw = shot.get("video_frames")
    if raw is None:
        return int(settings.video_frames)
    return snap_frames(int(raw))


def _generate_shot_cuda(
    settings, shot, image_path, dest, *, prompt, frames, height, width, fps, seed,
    has_audio, gguf_file, gguf_repo, conf_repo,
) -> None:
    """单镜：文本编码 → 释放 → transformer 去噪 → 释放 → VAE 解码 → mp4。"""
    import torch

    from pipeline.backends import cuda_memory, cuda_textenc

    lat_t, lat_h, lat_w = latent_dims(height, width, frames)
    tokens = lat_t * lat_h * lat_w
    audio_frames = 41

    # ---- 段1：文本编码（Gemma 全部层）+ connectors ----
    with cuda_memory.stage("videos/text_encoder", min_free_gib=7.0):
        te, tok = cuda_textenc.load(quant="4bit")
        enc = cuda_textenc.encode_and_free(te, tok, prompt=prompt)
        del te, tok
    cuda_memory.empty_cache()

    with cuda_memory.stage("videos/connectors", min_free_gib=3.0):
        conn = load_connectors(conf_repo=conf_repo)
        prompt_embeds, audio_embeds = encode_prompt_with_connectors(
            conn, hidden=enc["hidden"], mask=enc["mask"],
            scale_factor=enc.get("scale_factor", 8),
        )
        del conn
    del enc
    cuda_memory.empty_cache()

    # ---- 段1.5：用 VAE 编码静帧，得到 I2V 的初始 latent ----
    # 不编码静帧的话生成结果就是纯噪声（实测成片是彩色噪点）
    with cuda_memory.stage("videos/vae_encode", min_free_gib=2.0):
        vae_enc = load_vae(conf_repo=conf_repo)
        latents = encode_frame_to_latents(
            vae_enc, Path(image_path), frames=frames, height=height, width=width
        )
        del vae_enc
    cuda_memory.empty_cache()

    # Official I2V initial state: keep the encoded first latent frame; noise the rest.
    conditioning_latents = latents.clone()
    conditioning_mask = torch.zeros((1, latents.shape[1]), device="cuda", dtype=torch.bfloat16)
    first_frame_tokens = lat_h * lat_w
    conditioning_mask[:, :first_frame_tokens] = 1.0
    noise = torch.randn(latents.shape, device="cuda", dtype=torch.bfloat16,
                        generator=torch.Generator("cuda").manual_seed(int(seed)))
    latents = conditioning_latents * conditioning_mask.unsqueeze(-1) + noise * (1 - conditioning_mask.unsqueeze(-1))
    audio_latents = torch.randn(1, audio_frames, 128, device="cuda", dtype=torch.bfloat16)

    # ---- 段2：transformer 去噪 → 释放 ----
    steps = 8
    with cuda_memory.stage("videos/transformer", min_free_gib=7.5):
        model = load_transformer(gguf_file=gguf_file, gguf_repo=gguf_repo,
                                 conf_repo=conf_repo)
        latents, audio_latents = denoise(
            model, latents=latents, audio_latents=audio_latents,
            prompt_embeds=prompt_embeds, audio_prompt_embeds=audio_embeds,
            steps=steps, frames=frames, height=height, width=width,
            audio_frames=audio_frames,
            conditioning_mask=conditioning_mask,
            conditioning_latents=conditioning_latents,
        )
        del model
    del prompt_embeds, audio_embeds, audio_latents
    cuda_memory.empty_cache()

    # ---- 段3：VAE 解码 → mp4 ----
    with cuda_memory.stage("videos/vae", min_free_gib=2.0):
        vae = load_vae(conf_repo=conf_repo)
        decoded = decode_video(vae, latents, frames=frames, height=height, width=width)
        frames_arr = frames_to_uint8(decoded)
        del vae, decoded
    write_mp4(frames_arr, dest, fps=fps)
