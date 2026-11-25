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


# ---------------------------------------------------------------------------
# 命令入口：与 mac 端 pipeline/images.py 的 cmd_images 对齐
# ---------------------------------------------------------------------------


def cmd_images_cuda(settings, args) -> None:
    """CUDA 图像阶段：肖像 + 静帧，产物路径与 mac 端一致。"""
    import json
    import os
    import sys

    from pipeline.projects import artifacts_sub
    from pipeline.util import load_json, resolve

    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)

    chars = story.get("characters") or []
    shots = story["shots"]
    only = getattr(args, "only", None)
    force = getattr(args, "force", False)
    wanted_chars = getattr(args, "characters", None)

    portraits = artifacts_sub(out, "characters")
    frames = artifacts_sub(out, "frames")
    portraits.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    style = story.get("style") or "写实华语短剧，电影感光线"
    model_name = getattr(settings, "cuda_image_model", DEFAULT_MODEL)

    pipe = None
    try:
        # ---- 肖像 ----
        todo_portraits = []
        wanted = None
        if wanted_chars:
            wanted = {c.strip() for c in str(wanted_chars).replace("，", ",").split(",") if c.strip()}
        for char in chars:
            cid = str(char.get("id") or "").strip()
            if not cid:
                continue
            if wanted is not None and cid not in wanted:
                continue
            path = portraits / f"{cid}.png"
            if path.exists() and not force and wanted is None:
                continue
            todo_portraits.append((char, path))
        if todo_portraits:
            if pipe is None:
                cuda_memory.ensure_headroom(7.0, label="images/portrait")
                pipe = load_pipeline(model=model_name)
                print(f"已加载图像管道：{model_name}", flush=True)
            for char, path in todo_portraits:
                prompt = (
                    f"单人肖像，正面半身，看向镜头，中性表情，纯浅灰背景，电影布光，"
                    f"真实皮肤，竖构图。人物：{char.get('name')}，{char.get('appearance')}。"
                    f"风格：{style}。不要文字，不要水印。"
                )
                print(f"  肖像 {char.get('id')} …", flush=True)
                img = generate(pipe, prompt=prompt, height=settings.portrait_height,
                               width=settings.portrait_width,
                               steps=settings.image_steps, seed=_seed_of(char))
                save(img, path)
                print(f"    -> {path}", flush=True)

        # ---- 静帧 ----
        if only:
            ids = {x.strip() for x in str(only).replace("，", ",").split(",") if x.strip()}
            targets = [s for s in shots if s.get("id") in ids]
        else:
            targets = shots
        todo_frames = []
        for shot in targets:
            path = frames / f"{shot['id']}.png"
            if path.exists() and not force:
                continue
            todo_frames.append((shot, path))
        if todo_frames:
            if pipe is None:
                cuda_memory.ensure_headroom(7.0, label="images/frame")
                pipe = load_pipeline(model=model_name)
                print(f"已加载图像管道：{model_name}", flush=True)
            by_id = {c["id"]: c for c in chars}
            for shot, path in todo_frames:
                cast = [by_id[cid] for cid in (shot.get("characters") or []) if cid in by_id][:2]
                who = "；".join(f"{c['name']}：{c.get('appearance')}" for c in cast)
                base = (
                    f"竖屏短剧静帧，9:16，{style}。地点：{shot.get('location') or '未说明'}。"
                    f"画面：{shot.get('image_prompt') or ''}。"
                )
                if who:
                    base += f"人物外观必须保持：{who}。"
                prompt = base + "不要文字，不要字幕，不要水印。"
                print(f"  静帧 {shot['id']} …", flush=True)
                img = generate(pipe, prompt=prompt,
                               height=settings.image_height, width=settings.image_width,
                               steps=settings.image_steps, seed=_seed_of(shot))
                save(img, path)
                print(f"    -> {path}", flush=True)
    finally:
        if pipe is not None:
            del pipe
            cuda_memory.empty_cache()

    print(f"静帧在 {frames}")


def _seed_of(item: dict) -> int:
    """从分镜/角色里取稳定 seed；没有则按 id 派生。"""
    from pipeline.util import stable_seed

    raw = item.get("seed")
    if raw is not None:
        try:
            return int(raw) & 0x7FFFFFFF
        except (TypeError, ValueError):
            pass
    key = str(item.get("id") or item.get("name") or "")
    return stable_seed(key) & 0x7FFFFFFF
