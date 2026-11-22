from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.config import Settings
from pipeline.mirrors import child_env
from pipeline.util import dump_json, load_json, resolve, snap, snap_frames


def _uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("找不到 uv。请先运行 scripts/setup.sh")


def _prompt(shot: dict) -> str:
    motion = shot.get("motion_prompt") or "Slow cinematic push in"
    return (
        f"{motion}. Keep the face, hair and clothes identical to the first frame. "
        "Stable identity, natural motion, no morphing, no scene change. "
        "Ambient room tone only, no spoken words."
    )


def _command(
    settings: Settings,
    *,
    prompt: str,
    image: Path,
    output: Path,
    seed: int,
    width: int,
    height: int,
    frames: int,
) -> list[str]:
    cmd = [
        _uv(),
        "run",
        "--directory",
        str(settings.ltx_dir),
        "ltx-2-mlx",
        "generate",
        "--distilled",
        "--model",
        settings.video_model,
        "--prompt",
        prompt,
        "--image",
        str(image),
        "0",
        "1.0",
        "replace",
        "-f",
        str(frames),
        "--frame-rate",
        str(settings.video_fps),
        "-H",
        str(height),
        "-W",
        str(width),
        "--seed",
        str(seed),
        "-o",
        str(output),
        "--no-audio",
    ]
    if settings.video_low_ram:
        cmd.append("--low-ram")
    return cmd


def _run_once(settings: Settings, shot: dict, image: Path, output: Path, log: Path, width: int, height: int, frames: int) -> int:
    env = child_env()
    # 这台系统的 Python 会跳过带隐藏标记的 .pth，可编辑安装因此进不了 sys.path。
    srcs = [
        settings.ltx_dir / "packages" / "ltx-pipelines-mlx" / "src",
        settings.ltx_dir / "packages" / "ltx-core-mlx" / "src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in srcs)
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    cmd = _command(
        settings,
        prompt=_prompt(shot),
        image=image,
        output=output,
        seed=int(shot["seed"]),
        width=width,
        height=height,
        frames=frames,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def cmd_videos(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    if not settings.ltx_dir.exists():
        raise SystemExit(f"还没有视频模型代码：{settings.ltx_dir}。请先运行 scripts/setup.sh")
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    wanted = None
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    failures = 0
    width, height, frames = settings.video_width, settings.video_height, settings.video_frames
    for shot in story["shots"]:
        if wanted and shot["id"] not in wanted:
            continue
        image = out / "frames" / f"{shot['id']}.png"
        dest = clips / f"{shot['id']}.mp4"
        if dest.exists() and not args.force:
            print(f"已有镜头 {shot['id']}，跳过")
            continue
        if not image.exists():
            shot["video_error"] = "缺少静帧"
            failures += 1
            print(f"跳过 {shot['id']}：没有静帧 {image}")
            continue
        print(
            f"生成镜头 {shot['id']}：{width}x{height}，{frames} 帧。"
            "这一步在 16GB 的 M4 Air 上通常要几分钟到几十分钟，机器会发热。"
        )
        code = _run_once(settings, shot, image, dest, out / "logs" / f"{shot['id']}.log", width, height, frames)
        if code < 0 and (width, height, frames) != (320, 576, snap_frames(25)):
            print(f"{shot['id']} 被系统杀掉了，改成 320x576、25 帧再试一次。")
            width, height, frames = 320, snap(576, 64), snap_frames(25)
            if dest.exists():
                dest.unlink()
            code = _run_once(
                settings, shot, image, dest, out / "logs" / f"{shot['id']}-small.log", width, height, frames
            )
        if code != 0 or not dest.exists():
            shot["video_error"] = f"退出码 {code}，详见 logs/{shot['id']}.log"
            failures += 1
            print(f"镜头 {shot['id']} 失败，继续下一个。")
            continue
        shot.pop("video_error", None)
        shot["video"] = str(dest)
        print(f"完成 {dest}")
    dump_json(story_path, story)
    if failures:
        raise SystemExit(f"有 {failures} 个镜头没有生成。成功的镜头仍会在合成阶段用上。")
