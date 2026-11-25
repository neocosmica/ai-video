from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

from pipeline.audio import pad_wav_to_duration
from pipeline.config import Settings
from pipeline.hf_cache import resolve_snapshot
from pipeline.mirrors import child_env
from pipeline.projects import artifacts_sub
from pipeline.runtime import (
    ensure_runtime_layout,
    ltx_cli,
    ltx_python,
    main_venv_path,
    preflight,
)
from pipeline.progress import run_tee
from pipeline.util import dump_json, load_json, resolve, snap, snap_frames

# 环境类错误：同一批再跑下一镜也必然挂，应立即终止并写清原因
_ENV_ERROR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"Bad CPU type|Errno 86", re.I),
        "ffmpeg 架构与本机不符（Bad CPU type）。请重跑 ./scripts/setup.sh 换 arm64 版，"
        "或安装 Rosetta：softwareupdate --install-rosetta",
    ),
    (
        re.compile(r"ffmpeg not found|找不到 ffmpeg|No such file.*ffmpeg", re.I),
        "找不到可用的 ffmpeg。请先 ./scripts/setup.sh 或 brew install ffmpeg",
    ),
    (
        re.compile(r"ffprobe not found|找不到 ffprobe", re.I),
        "找不到可用的 ffprobe。请先 ./scripts/setup.sh 或 brew install ffmpeg",
    ),
)

DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"

# LTX distilled 两段 Denoising 的默认步数（8 步半分辨率 + 3 步精修）。
# 交给进度条先定死分母，第一段跑完只到 8/11，不会先冲到 100% 再掉回来。
DENOISE_PLAN = (8, 3)


def _prompt(shot: dict, *, with_speech: bool) -> str:
    motion = shot.get("motion_prompt") or "Slow cinematic push in"
    if with_speech:
        return (
            f"{motion}. The on-screen speaker talks naturally with clear lip sync to the dialogue audio. "
            "Keep the face, hair and clothes identical to the first frame. "
            "Stable identity, natural mouth motion, no morphing, no scene change."
        )
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


def _shot_frames(shot: dict, settings: Settings) -> int:
    raw = shot.get("video_frames")
    if raw is None:
        return int(settings.video_frames)
    return snap_frames(int(raw))


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
    audio: Path | None,
) -> list[str]:
    # 直接跑 venv 里的入口，禁止每镜 `uv run`
    cli = str(ltx_cli(settings.ltx_dir))
    common_tail = [
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
    ]
    if audio is not None:
        cmd = [cli, "a2v", "--distilled", "--audio", str(audio), *common_tail]
    else:
        cmd = [cli, "generate", "--distilled", *common_tail, "--no-audio"]
    # 固定两段步数，让终端进度条的分母（8 + 3）与实际一致
    cmd += ["--stage1-steps", str(DENOISE_PLAN[0]), "--stage2-steps", str(DENOISE_PLAN[1])]
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
    # LTX 用 shutil.which("ffmpeg")：把已校验可用的 bin 目录放最前，避开坏的 tools/
    ff_dir = str(Path(settings.ffmpeg).resolve().parent)
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = os.pathsep.join(
        [ff_dir, local_bin, env.get("PATH", "")]
    )
    # 子进程勿继承主项目 VIRTUAL_ENV，否则会指到错误解释器
    env.pop("VIRTUAL_ENV", None)
    return env


def _read_log_tail(path: Path, nbytes: int = 12288) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > nbytes:
        data = data[-nbytes:]
    return data.decode("utf-8", errors="replace")


def _classify_video_failure(log: Path, code: int) -> tuple[str, bool]:
    """返回 (用户可见原因, 是否环境类需整批终止)。"""
    text = _read_log_tail(log)
    for pattern, message in _ENV_ERROR_PATTERNS:
        if pattern.search(text):
            return message, True
    # 子进程还没写出日志就因 exec 失败退出时，偶发信息在 stderr 已进 log 头
    if code != 0 and not text.strip():
        return f"退出码 {code}，日志为空", False
    # 截一句最像报错的行
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if re.search(r"Error|Exception|OSError|Traceback|failed|找不到", s, re.I):
            short = s if len(s) <= 180 else s[:177] + "…"
            return f"{short}（退出码 {code}）", False
    return f"退出码 {code}，详见 {log.name}", False


def _prepare_a2v_audio(out: Path, shot: dict, voice: Path, frames: int, fps: int) -> Path:
    """按镜头帧数裁/补对白，满足 A2V「音频不少于画面时长」。"""
    need = frames / max(1, fps) + 0.05
    dest = artifacts_sub(out, "audio", "a2v", f"{shot['id']}.wav")
    return pad_wav_to_duration(voice, dest, need)


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
    audio: Path | None,
) -> int:
    env = _video_env(settings, offline=offline)
    cmd = _command(
        settings,
        prompt=_prompt(shot, with_speech=audio is not None),
        image=image,
        output=output,
        seed=int(shot["seed"]),
        width=width,
        height=height,
        frames=frames,
        model=model,
        gemma=gemma,
        audio=audio,
    )
    mode = "a2v" if audio is not None else "i2v"
    shot_id = str(shot.get("id") or "shot")
    return run_tee(
        cmd,
        log=log,
        env=env,
        progress=f"{shot_id} {mode}",
        plan=DENOISE_PLAN,
        desc="Denoising",
    )


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
    from pipeline.media_bins import binary_runnable

    ff_ok, ff_detail = binary_runnable(settings.ffmpeg)
    fp_ok, fp_detail = binary_runnable(settings.ffprobe)
    if not ff_ok or not fp_ok:
        raise SystemExit(
            f"视频环境错误（ffmpeg 不可用，已终止）：{ff_detail if not ff_ok else fp_detail}"
        )
    note = (settings.budget_detail or {}).get("ffmpeg")
    if note:
        print(f"ffmpeg：{note}", flush=True)
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
    clips = artifacts_sub(out, "clips")
    clips.mkdir(parents=True, exist_ok=True)
    failures = 0
    width, height = settings.video_width, settings.video_height
    model = _resolve_model_arg(settings.video_model, label="LTX 权重")
    gemma = _resolve_model_arg(DEFAULT_GEMMA, label="Gemma")
    offline = Path(model).exists() and Path(gemma).exists()
    for shot in story["shots"]:
        if wanted and shot["id"] not in wanted:
            continue
        image = artifacts_sub(out, "frames", f"{shot['id']}.png")
        if not image.exists():
            # 台词被自动拆分出的子镜头没有自己的静帧，复用父镜头那张
            reuse = str(shot.get("reuse_frame") or "").strip()
            if reuse:
                image = artifacts_sub(out, "frames", f"{reuse}.png")
        dest = clips / f"{shot['id']}.mp4"
        if dest.exists() and not args.force:
            print(f"已有镜头 {shot['id']}，跳过")
            continue
        if not image.exists():
            shot["video_error"] = "缺少静帧"
            failures += 1
            print(f"跳过 {shot['id']}：没有静帧 {image}")
            continue
        line = str(shot.get("dialogue") or "").strip()
        voice = artifacts_sub(out, "audio", "voice", f"{shot['id']}.wav")
        audio_for_ltx: Path | None = None
        frames = _shot_frames(shot, settings)
        if line:
            if not voice.exists():
                shot["video_error"] = "有对白但缺少 wav，请先跑 audio"
                failures += 1
                print(f"跳过 {shot['id']}：请先 ./scripts/run.sh audio --out {out}")
                continue
            audio_for_ltx = _prepare_a2v_audio(out, shot, voice, frames, settings.video_fps)
        def _fail_shot(log_path: Path, exit_code: int) -> None:
            nonlocal failures
            reason, is_env = _classify_video_failure(log_path, exit_code)
            shot["video_error"] = reason
            failures += 1
            dump_json(story_path, story)  # 立刻落盘，UI 能读到失败原因
            if is_env:
                print(f"镜头 {shot['id']} 环境错误，终止本批：{reason}")
                raise SystemExit(f"视频环境错误（已终止，避免重复失败）：{reason}")
            print(f"镜头 {shot['id']} 失败：{reason}；继续下一个。")

        _wait_memory_headroom()
        log_path = artifacts_sub(out, "logs", f"{shot['id']}.log")
        code = _run_once(
            settings,
            shot,
            image,
            dest,
            log_path,
            width,
            height,
            frames,
            model=model,
            gemma=gemma,
            offline=offline,
            audio=audio_for_ltx,
        )
        if code != 0 or not dest.exists():
            reason, is_env = _classify_video_failure(log_path, code)
            if is_env:
                _fail_shot(log_path, code)
                continue
        # 被系统杀掉（负退出码）或 Gemma/connector 峰值 OOM：等内存后重试一次
        if code != 0 and (code < 0 or not dest.exists()):
            print(f"{shot['id']} 失败（退出码 {code}），回收内存后重试一次…")
            if dest.exists():
                dest.unlink()
            _wait_memory_headroom(min_free_pct=30.0, timeout_s=120.0)
            time.sleep(3)
            log_path = artifacts_sub(out, "logs", f"{shot['id']}-retry.log")
            code = _run_once(
                settings,
                shot,
                image,
                dest,
                log_path,
                width,
                height,
                frames,
                model=model,
                gemma=gemma,
                offline=offline,
                audio=audio_for_ltx,
            )
            if code != 0 or not dest.exists():
                reason, is_env = _classify_video_failure(log_path, code)
                if is_env:
                    _fail_shot(log_path, code)
                    continue
        small = (320, snap(576, 64), snap_frames(25))
        if code < 0 and (width, height, frames) != small:
            print(f"{shot['id']} 仍被系统杀掉，改成 320x576、25 帧再试一次。")
            width, height, frames = small
            if dest.exists():
                dest.unlink()
            if audio_for_ltx is not None and voice.exists():
                audio_for_ltx = _prepare_a2v_audio(out, shot, voice, frames, settings.video_fps)
            log_path = artifacts_sub(out, "logs", f"{shot['id']}-small.log")
            code = _run_once(
                settings,
                shot,
                image,
                dest,
                log_path,
                width,
                height,
                frames,
                model=model,
                gemma=gemma,
                offline=offline,
                audio=audio_for_ltx,
            )
        if code != 0 or not dest.exists():
            _fail_shot(log_path, code)
            continue
        shot.pop("video_error", None)
        shot["video"] = str(dest)
        shot["video_frames"] = frames
        shot["video_mode"] = "a2v" if audio_for_ltx is not None else "i2v"
    dump_json(story_path, story)
    if failures:
        raise SystemExit(f"有 {failures} 个镜头没有生成。成功的镜头仍会在合成阶段用上。")
