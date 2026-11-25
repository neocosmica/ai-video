from __future__ import annotations

from pathlib import Path

import argparse
import gc
import os
import sys

from pipeline.config import Settings
from pipeline.mirrors import apply_mirrors
from pipeline.motion import (
    end_edit_prompt,
    end_frame_path,
    start_frame_path,
    wants_end_frame,
)
from pipeline.projects import artifacts_sub
from pipeline.runtime import ensure_runtime_layout, preflight
from pipeline.util import ROOT, load_json, resolve, stable_seed

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


def _generate(
    model,
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    desc: str,
    images=None,
    image_strength: float | None = None,
):
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
    if image_strength is not None:
        kwargs["image_strength"] = image_strength
    _use_step_bar(desc)
    try:
        return model.generate_image(**kwargs)
    except TypeError:
        kwargs.pop("guidance", None)
        if image_strength is not None:
            kwargs.pop("image_strength", None)
        if images:
            kwargs.pop("image_paths", None)
            kwargs["image_paths"] = images
            if image_strength is not None:
                kwargs["image_strength"] = image_strength
        try:
            return model.generate_image(**kwargs)
        except TypeError:
            kwargs.pop("image_strength", None)
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
    chosen = [s for s in shots if s["id"] in wanted]
    have = {str(s.get("id") or "") for s in chosen}
    seen_reuse: set[str] = set()
    for shot in shots:
        reuse = str(shot.get("reuse_frame") or "").strip()
        if not reuse or reuse not in wanted or reuse in have or reuse in seen_reuse:
            continue
        if any(str(s.get("id") or "") == reuse for s in shots):
            continue
        # 父镜头已被拆掉，分镜里只剩子镜头；静帧文件仍用父 id
        ghost = dict(shot)
        ghost["id"] = reuse
        ghost.pop("reuse_frame", None)
        chosen.append(ghost)
        seen_reuse.add(reuse)
        have.add(reuse)
    return chosen


def _parse_id_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    ids = {item.strip() for item in str(raw).replace("，", ",").split(",") if item.strip()}
    return ids or None


def _cast_of(shot: dict, characters: list[dict]) -> list[dict]:
    by_id = {c["id"]: c for c in characters}
    return [by_id[cid] for cid in shot.get("characters") or [] if cid in by_id][:2]


def run_t2i(
    story: dict,
    out: Path,
    settings: Settings,
    only: str | None,
    force: bool,
    characters: str | None = None,
) -> bool:
    portraits = artifacts_sub(out, "characters")
    frames = artifacts_sub(out, "frames")
    shots = _selected(story["shots"], only)
    wanted_chars = _parse_id_list(characters)
    # 肖像：
    # - 缺文件 → 补生成（--only 的镜头用到的肖像，即使没写进 --characters）
    # - --characters + --force → 重做点名的这些肖像
    # - 全量 --force（无 --only / --characters）→ 重做全部肖像
    # - 镜头 --only + --force（无 --characters）→ 不动已有肖像
    redo_all_portraits = bool(force) and not only and wanted_chars is None
    pending_portraits = []
    for char in story.get("characters") or []:
        cid = str(char.get("id") or "").strip()
        if not cid:
            continue
        path = portraits / f"{cid}.png"
        in_frame = bool(only) and any(cid in (s.get("characters") or []) for s in shots)
        explicit = wanted_chars is not None and cid in wanted_chars
        if wanted_chars is not None and not explicit:
            # 不在 --characters 里的肖像：这次要画的镜头用到、文件又没有，才补
            if not (in_frame and not path.exists()):
                continue
        elif wanted_chars is None and only and not in_frame:
            continue
        if path.exists():
            if explicit and force:
                pass  # 点名的肖像：强制重做
            elif redo_all_portraits:
                pass
            else:
                continue
        pending_portraits.append((char, path))
    # 仅 --characters、未指定 --only：只做肖像，不进静帧
    portrait_only = wanted_chars is not None and not only
    pending_frames = []
    needs_edit = False
    if not portrait_only:
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
        return needs_edit and not portrait_only
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
    skipped: list[str] = []
    for shot in _selected(story["shots"], only):
        path = frames / f"{shot['id']}.png"
        cast = _cast_of(shot, story.get("characters") or [])
        if not cast or (path.exists() and not force):
            continue
        refs = []
        missing_ref = False
        for char in cast:
            ref = portraits / f"{char['id']}.png"
            if not ref.exists():
                missing_ref = True
                print(f"跳过 {shot['id']}：缺少角色参考图 {ref}")
                break
            refs.append(ref)
        if missing_ref:
            skipped.append(str(shot["id"]))
            continue
        pending.append((shot, path, cast, refs))
    if not pending and not skipped:
        print("没有需要按参考图重绘的镜头。")
        return
    if pending:
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
    if skipped:
        raise SystemExit(
            f"有 {len(skipped)} 个镜头缺少角色参考图（{', '.join(skipped)}），"
            "这些镜头先跳过。先补上对应肖像，再跑这些静帧。"
        )


def _shots_needing_end_frame(story: dict, only: str | None) -> list[dict]:
    return [s for s in _selected(story["shots"], only) if wants_end_frame(s)]


def _end_frame_stale(start: Path, end: Path) -> bool:
    if not end.exists():
        return True
    try:
        return start.stat().st_mtime > end.stat().st_mtime + 0.01
    except OSError:
        return True


def run_end_frames(story: dict, out: Path, settings: Settings, only: str | None, force: bool) -> None:
    """用 Klein Edit 把首帧改成终点姿势，写出 frames/<id>_end.png。"""
    frames = artifacts_sub(out, "frames")
    portraits = artifacts_sub(out, "characters")
    pending = []
    for shot in _shots_needing_end_frame(story, only):
        start = start_frame_path(frames, shot)
        end = end_frame_path(frames, shot)
        if not start.exists():
            print(f"跳过尾帧 {shot['id']}：还没有首帧 {start.name}")
            continue
        if end.exists() and not force and not _end_frame_stale(start, end):
            continue
        cast = _cast_of(shot, story.get("characters") or [])
        refs: list[Path] = [start]
        for char in cast:
            ref = portraits / f"{char['id']}.png"
            if ref.exists():
                refs.append(ref)
        pending.append((shot, start, end, cast, refs))
    if not pending:
        print("没有需要生成的尾帧。")
        return
    cls, config, weights = _load_class("edit")
    print(f"加载尾帧编辑模型 {weights}")
    model = cls(model_config=config, quantize=None, model_path=weights)
    for shot, start, end, cast, refs in pending:
        # Flux2KleinEdit：首帧经 image_paths 做参考 token 条件；不要传 image_strength。
        # Edit 路径 latent 仍是纯噪声，strength 只会抬高 init_time_step、少跑去噪步（曾从 4 步砍成 2 步）。
        print(
            f"尾帧 {shot['id']} <- {start.name}（参考 {len(refs)} 张："
            + "、".join(p.name for p in refs)
            + "）"
        )
        seed = int(shot["seed"]) ^ stable_seed("end", shot["id"], str(shot.get("end_image_prompt") or ""))
        image = _generate(
            model,
            prompt=end_edit_prompt(shot, story, cast),
            seed=seed & 0x7FFFFFFF,
            width=settings.image_width,
            height=settings.image_height,
            steps=settings.image_steps,
            desc=f"尾帧 {shot['id']}",
            images=refs,
        )
        _save_image(image, end)
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
    characters = getattr(args, "characters", None)
    phase = os.environ.get("NOVEL_IMAGE_PHASE", "t2i")
    if phase == "t2i":
        # 只在第一阶段做硬预检；edit / end 是同进程 exec 续跑，不必再卡一次
        preflight(timeout_s=120.0, soft=False)
        needs_edit = run_t2i(
            story, out, settings, args.only, args.force, characters=characters
        )
        # 仅补肖像时不要顺带跑全片尾帧
        portrait_only = bool(characters) and not args.only
        needs_end = (not portrait_only) and bool(_shots_needing_end_frame(story, args.only))
        if needs_edit or needs_end:
            env = os.environ.copy()
            env["NOVEL_IMAGE_PHASE"] = "edit" if needs_edit else "end"
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            os.execve(sys.executable, [sys.executable, "-m", "pipeline", *sys.argv[1:]], env)
        print(f"静帧在 {artifacts_sub(out, 'frames')}")
        return
    if phase == "edit":
        run_edit(story, out, settings, args.only, args.force)
        # 角色首帧刚生成完，同一次 images 里接着做尾帧（再 exec 一次卸模型）
        if _shots_needing_end_frame(story, args.only):
            env = os.environ.copy()
            env["NOVEL_IMAGE_PHASE"] = "end"
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            os.execve(sys.executable, [sys.executable, "-m", "pipeline", *sys.argv[1:]], env)
        print(f"静帧在 {artifacts_sub(out, 'frames')}")
        return
    if phase == "end":
        run_end_frames(story, out, settings, args.only, args.force)
        print(f"静帧在 {artifacts_sub(out, 'frames')}")
        return
    raise SystemExit(f"未知 NOVEL_IMAGE_PHASE={phase!r}")
