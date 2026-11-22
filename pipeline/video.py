from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from pipeline.config import Settings
from pipeline.hf_cache import resolve_snapshot
from pipeline.mirrors import child_env
from pipeline.runtime import (
    ensure_runtime_layout,
    ltx_cli,
    ltx_python,
    main_venv_path,
    preflight,
)
from pipeline.util import dump_json, load_json, resolve, snap, snap_frames

DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"


def _prompt(shot: dict) -> str:
    motion = shot.get("motion_prompt") or "Slow cinematic push in"
    return (
        f"{motion}. Keep the face, hair and clothes identical to the first frame. "
        "Stable identity, natural motion, no morphing, no scene change. "
        "Ambient room tone only, no spoken words."
    )


def _ensure_ltx_patches(settings: Settings) -> None:
    marker = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"
    blocks = (
        settings.ltx_dir
        / "packages"
        / "ltx-pipelines-mlx"
        / "src"
        / "ltx_pipelines_mlx"
        / "utils"
        / "blocks.py"
    )
    encoder = (
        settings.ltx_dir
        / "packages"
        / "ltx-core-mlx"
        / "src"
        / "ltx_core_mlx"
        / "text_encoders"
        / "gemma"
        / "encoders"
        / "base_encoder.py"
    )
    need = any(not path.exists() or marker not in path.read_text(encoding="utf-8") for path in (blocks, encoder))
    if not need:
        return
    patcher = resolve("scripts/apply_ltx_gemma_patch.py")
    if not patcher.exists():
        raise SystemExit(f"LTX 缺少 Gemma 分阶段加载补丁，且找不到 {patcher}")
    print("正在为 vendor/ltx-2-mlx 应用 Gemma 稳定加载补丁…")
    py = main_venv_path() / "bin" / "python"
    if not py.exists():
        py = Path(resolve(".venv")) / "bin" / "python"
    if not py.exists():
        raise SystemExit("找不到主环境 Python，请先运行 scripts/setup.sh")
    env = child_env()
    env.pop("VIRTUAL_ENV", None)
    code = subprocess.call([str(py), str(patcher)], env=env)
    if code != 0:
        raise SystemExit("应用 Gemma 补丁失败")
    for path in (blocks, encoder):
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"补丁后仍缺少标记：{path}")


def _resolve_model_arg(repo_or_path: str, *, label: str) -> str:
    local = resolve_snapshot(repo_or_path)
    if local is not None:
        print(f"使用本地{label}：{local}")
        return str(local)
    print(f"未命中本地缓存，回退 repo id：{repo_or_path}")
    return repo_or_path


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
    model: str,
    gemma: str,
) -> list[str]:
    # 直接跑 venv 里的入口，禁止每镜 `uv run`（会反复解析环境，偶发重建，再触发慢导入）
    cmd = [
        str(ltx_cli(settings.ltx_dir)),
        "generate",
        "--distilled",
        "--model",
        model,
        "--gemma",
        gemma,
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


def _video_env(settings: Settings, *, offline: bool) -> dict[str, str]:
    env = child_env()
    if offline:
        # 本地 snapshot 齐备时禁止 hub 联网刷新，避免卡在 snapshot_download
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    # 短剧 prompt 很短；1024 会放大 Gemma 激活显存
    env.setdefault("LTX2_GEMMA_MAX_LENGTH", "256")
    # 可编辑包 + 防止隐藏 .pth 被跳过：显式 PYTHONPATH
    srcs = [
        settings.ltx_dir / "packages" / "ltx-pipelines-mlx" / "src",
        settings.ltx_dir / "packages" / "ltx-core-mlx" / "src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in srcs)
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    # 子进程勿继承主项目 VIRTUAL_ENV，否则会指到错误解释器
    env.pop("VIRTUAL_ENV", None)
    return env


def _run_once(
    settings: Settings,
    shot: dict,
    image: Path,
    output: Path,
    log: Path,
    width: int,
    height: int,
    frames: int,
    *,
    model: str,
    gemma: str,
    offline: bool,
) -> int:
    env = _video_env(settings, offline=offline)
    cmd = _command(
        settings,
        prompt=_prompt(shot),
        image=image,
        output=output,
        seed=int(shot["seed"]),
        width=width,
        height=height,
        frames=frames,
        model=model,
        gemma=gemma,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        print(f"启动 LTX CLI（日志 {log.name}）…", flush=True)
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def _wait_memory_headroom(min_free_pct: float = 25.0, timeout_s: float = 90.0) -> None:
    """粗略等统一内存喘口气，降低 Gemma/connector 连续加载被 jetsam 杀掉的概率。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.check_output(["memory_pressure"], text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return
        free = None
        for line in out.splitlines():
            if "System-wide memory free percentage" in line:
                # e.g. "System-wide memory free percentage: 35%"
                raw = line.rsplit(":", 1)[-1].strip().rstrip("%")
                try:
                    free = float(raw)
                except ValueError:
                    free = None
                break
        if free is None or free >= min_free_pct:
            return
        print(f"内存空闲约 {free:.0f}%，等待回收后再加载 Gemma…")
        time.sleep(5)


def cmd_videos(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    if not settings.ltx_dir.exists():
        raise SystemExit(f"还没有视频模型代码：{settings.ltx_dir}。请先运行 scripts/setup.sh")
    ensure_runtime_layout(ltx_dir=settings.ltx_dir)
    preflight(ltx_dir=settings.ltx_dir, python=ltx_python(settings.ltx_dir), soft=False)
    _ensure_ltx_patches(settings)
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
    model = _resolve_model_arg(settings.video_model, label="LTX 权重")
    gemma = _resolve_model_arg(DEFAULT_GEMMA, label="Gemma")
    offline = Path(model).exists() and Path(gemma).exists()
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
        _wait_memory_headroom()
        code = _run_once(
            settings,
            shot,
            image,
            dest,
            out / "logs" / f"{shot['id']}.log",
            width,
            height,
            frames,
            model=model,
            gemma=gemma,
            offline=offline,
        )
        # 被系统杀掉（负退出码）或 Gemma/connector 峰值 OOM：等内存后重试一次
        if code != 0 and (code < 0 or not dest.exists()):
            print(f"{shot['id']} 失败（退出码 {code}），回收内存后重试一次…")
            if dest.exists():
                dest.unlink()
            _wait_memory_headroom(min_free_pct=30.0, timeout_s=120.0)
            time.sleep(3)
            code = _run_once(
                settings,
                shot,
                image,
                dest,
                out / "logs" / f"{shot['id']}-retry.log",
                width,
                height,
                frames,
                model=model,
                gemma=gemma,
                offline=offline,
            )
        if code < 0 and (width, height, frames) != (320, 576, snap_frames(25)):
            print(f"{shot['id']} 仍被系统杀掉，改成 320x576、25 帧再试一次。")
            width, height, frames = 320, snap(576, 64), snap_frames(25)
            if dest.exists():
                dest.unlink()
            code = _run_once(
                settings,
                shot,
                image,
                dest,
                out / "logs" / f"{shot['id']}-small.log",
                width,
                height,
                frames,
                model=model,
                gemma=gemma,
                offline=offline,
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
