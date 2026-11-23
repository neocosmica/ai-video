from __future__ import annotations

from pathlib import Path

import argparse
import gc
import os
import sys

from pipeline.config import Settings
from pipeline.mirrors import apply_mirrors
from pipeline.projects import artifacts_sub
from pipeline.runtime import ensure_runtime_layout, preflight
from pipeline.util import ROOT, load_json, resolve

apply_mirrors()


def _save_image(result, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if hasattr(result, "save"):
        result.save(path, overwrite=True)
        if path.exists() and path.stat().st_size > 0:
            return
    image = getattr(result, "image", result)
    if hasattr(image, "save"):
        image.save(path)
        return
    raise RuntimeError("图像模型没有返回可保存的图片")


def _flux_dir() -> str:
    root = Path.home() / ".cache/huggingface/hub/models--mflux-community--flux2-klein-4b-mflux-q4/snapshots"
    snaps = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not snaps:
        raise SystemExit("找不到 FLUX.2 Klein 4B 的 4-bit 权重。请先运行 scripts/setup.sh。")
    return str(snaps[0])


def _load_class(kind: str):
    print("加载 mflux / transformers…", flush=True)
    from mflux.models.common.config import ModelConfig

    weights = _flux_dir()
    if kind == "t2i":
        try:
            from mflux.models.flux2.variants import Flux2Klein
        except ImportError:
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        return Flux2Klein, ModelConfig.flux2_klein_4b(), weights
    try:
        from mflux.models.flux2.variants import Flux2KleinEdit
    except ImportError:
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    return Flux2KleinEdit, ModelConfig.flux2_klein_4b(), weights


_BAR_PATCHED = False


def _use_step_bar(desc: str) -> None:
    """给 mflux 去噪循环的 tqdm 加上说明。每张图新建 Config，所以各有一条。"""
    global _BAR_PATCHED
    from tqdm import tqdm

    from mflux.models.common.config.config import Config

    Config._novel_step_desc = desc  # type: ignore[attr-defined]
    if _BAR_PATCHED:
        return

    def time_steps(self):
        if self._time_steps is None:
            label = getattr(Config, "_novel_step_desc", "") or None
            # initial/total 按完整去噪步数计：img2img 从半程起跑时，条也要从 2/4 走到 4/4，
            # 不要变成 1/2→2/2，否则网页上看起来总在 50% 就结束。
            self._time_steps = tqdm(
                range(self.init_time_step, self.num_inference_steps),
                desc=label,
                initial=self.init_time_step,
                total=self.num_inference_steps,
                miniters=1,
                mininterval=0.2,
                file=sys.stderr,
            )
        return self._time_steps

    Config.time_steps = property(time_steps)
    _BAR_PATCHED = True


def _generate(model, *, prompt: str, seed: int, width: int, height: int, steps: int, desc: str, images=None):
    kwargs = {
        "seed": seed,
        "prompt": prompt,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
        "guidance": 1.0,
    }
    if images:
        kwargs["image_paths"] = images
    _use_step_bar(desc)
    try:
        return model.generate_image(**kwargs)
    except TypeError:
        kwargs.pop("guidance", None)
        if images:
            kwargs.pop("image_paths", None)
            kwargs["image_paths"] = images
        return model.generate_image(**kwargs)


def _portrait_prompt(char: dict, style: str) -> str:
    return (
        f"单人肖像，正面半身，看向镜头，中性表情，纯浅灰背景，电影布光，真实皮肤，竖构图。"
        f"人物：{char['name']}，{char['appearance']}。风格：{style}。不要文字，不要水印。"
    )


def _frame_prompt(shot: dict, story: dict, cast: list[dict]) -> str:
    style = story.get("style") or ""
    who = "；".join(f"{c['name']}：{c['appearance']}" for c in cast)
    base = (
        f"竖屏短剧静帧，9:16，{style}。地点：{shot.get('location') or '未说明'}。"
        f"画面：{shot['image_prompt']}。"
    )
    if cast:
        base += f"人物外观必须保持：{who}。"
    return base + "不要文字，不要字幕，不要水印。"


def _edit_prompt(shot: dict, story: dict, cast: list[dict]) -> str:
    names = "，".join(
        f"第{i}张参考图是{c['name']}（{c['appearance']}）" for i, c in enumerate(cast, start=1)
    )
    return (
        f"保持参考图里人物的脸、发型和衣服不变。{names}。"
        f"把他们放进这个竖屏画面：{shot['image_prompt']}。"
        f"地点：{shot.get('location') or '未说明'}。风格：{story.get('style') or ''}。"
        f"不要文字，不要字幕，不要水印。"
    )


def _selected(shots: list[dict], only: str | None) -> list[dict]:
    if not only:
        return shots
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    return [s for s in shots if s["id"] in wanted]


def _cast_of(shot: dict, characters: list[dict]) -> list[dict]:
    by_id = {c["id"]: c for c in characters}
    return [by_id[cid] for cid in shot.get("characters") or [] if cid in by_id][:2]


def run_t2i(story: dict, out: Path, settings: Settings, only: str | None, force: bool) -> bool:
    portraits = artifacts_sub(out, "characters")
    frames = artifacts_sub(out, "frames")
    shots = _selected(story["shots"], only)
    pending_portraits = []
    for char in story.get("characters") or []:
        path = portraits / f"{char['id']}.png"
        if force or not path.exists():
            if only and not any(char["id"] in (s.get("characters") or []) for s in shots):
                continue
            pending_portraits.append((char, path))
    pending_frames = []
    needs_edit = False
    for shot in shots:
        path = frames / f"{shot['id']}.png"
        cast = _cast_of(shot, story.get("characters") or [])
        if cast:
            needs_edit = True
            continue
        if force or not path.exists():
            pending_frames.append((shot, path))
    if not pending_portraits and not pending_frames:
        print("文生图阶段没有新的肖像或空镜。")
        return needs_edit
    cls, config, weights = _load_class("t2i")
    print(f"加载文生图模型 {weights}")
    model = cls(model_config=config, quantize=None, model_path=weights)
    for char, path in pending_portraits:
        print(f"角色参考图 {char['name']} -> {path.name}")
        image = _generate(
            model,
            prompt=_portrait_prompt(char, story.get("style") or ""),
            seed=int(char["seed"]),
            width=settings.portrait_width,
            height=settings.portrait_height,
            steps=settings.image_steps,
            desc=f"肖像 {char['name']}",
        )
        _save_image(image, path)
    for shot, path in pending_frames:
        print(f"空镜 {shot['id']} -> {path.name}")
        image = _generate(
            model,
            prompt=_frame_prompt(shot, story, []),
            seed=int(shot["seed"]),
            width=settings.image_width,
            height=settings.image_height,
            steps=settings.image_steps,
            desc=f"空镜 {shot['id']}",
        )
        _save_image(image, path)
    del model
    gc.collect()
    return needs_edit


def run_edit(story: dict, out: Path, settings: Settings, only: str | None, force: bool) -> None:
    frames = artifacts_sub(out, "frames")
    portraits = artifacts_sub(out, "characters")
    pending = []
    for shot in _selected(story["shots"], only):
        path = frames / f"{shot['id']}.png"
        cast = _cast_of(shot, story.get("characters") or [])
        if not cast or (path.exists() and not force):
            continue
        refs = []
        for char in cast:
            ref = portraits / f"{char['id']}.png"
            if not ref.exists():
                raise SystemExit(f"缺少角色参考图 {ref}，请先完成文生图。")
            refs.append(ref)
        pending.append((shot, path, cast, refs))
    if not pending:
        print("没有需要按参考图重绘的镜头。")
        return
    cls, config, weights = _load_class("edit")
    print(f"加载参考图编辑模型 {weights}")
    model = cls(model_config=config, quantize=None, model_path=weights)
    for shot, path, cast, refs in pending:
        print(f"分镜静帧 {shot['id']}，参考 {'、'.join(c['name'] for c in cast)}")
        image = _generate(
            model,
            prompt=_edit_prompt(shot, story, cast),
            seed=int(shot["seed"]),
            width=settings.image_width,
            height=settings.image_height,
            steps=settings.image_steps,
            desc=f"静帧 {shot['id']}",
            images=refs,
        )
        _save_image(image, path)
    del model
    gc.collect()


def cmd_images(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    ensure_runtime_layout()
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    phase = os.environ.get("NOVEL_IMAGE_PHASE", "t2i")
    if phase != "edit":
        # 只在第一阶段做硬预检；edit 是同进程 exec 续跑，不必再卡一次
        preflight(timeout_s=120.0, soft=False)
        needs_edit = run_t2i(story, out, settings, args.only, args.force)
        if needs_edit:
            env = os.environ.copy()
            env["NOVEL_IMAGE_PHASE"] = "edit"
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            os.execve(sys.executable, [sys.executable, "-m", "pipeline", *sys.argv[1:]], env)
        print(f"静帧在 {artifacts_sub(out, 'frames')}")
        return
    run_edit(story, out, settings, args.only, args.force)
    print(f"静帧在 {artifacts_sub(out, 'frames')}")
