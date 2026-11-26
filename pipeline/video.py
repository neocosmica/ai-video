from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from pipeline.audio import pad_wav_to_duration
from pipeline.backends import open_video
from pipeline.config import Settings
from pipeline.projects import artifacts_sub
from pipeline.runtime import ensure_runtime_layout, ltx_python, preflight
from pipeline.motion import (
    end_frame_path,
    start_frame_path,
    video_conditioning,
)
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


def _shot_frames(shot: dict, settings: Settings) -> int:
    raw = shot.get("video_frames")
    if raw is None:
        return int(settings.video_frames)
    return snap_frames(int(raw))



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
    runtime = open_video(settings)
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
    frames_dir = artifacts_sub(out, "frames")
    for shot in story["shots"]:
        if wanted and shot["id"] not in wanted:
            continue
        image = start_frame_path(frames_dir, shot)
        end_image = end_frame_path(frames_dir, shot)
        if not end_image.exists():
            end_image = None
        cond = video_conditioning(shot, has_end_frame=bool(end_image))
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
                print(f"跳过 {shot['id']}：有对白但没有配音 {voice}")
                continue
            audio_for_ltx = _prepare_a2v_audio(out, shot, voice, frames, settings.video_fps)
        if cond.use_end_frame and end_image:
            from pipeline.motion import END_FRAME_MIN_MAE, frame_mae

            mae = frame_mae(image, end_image)
            if mae < END_FRAME_MIN_MAE:
                print(
                    f"警告：{shot['id']} 首尾帧过近（mae={mae:.1f}<{END_FRAME_MIN_MAE:g}），"
                    "双帧引导几乎拉不开动作；建议重跑尾帧或改 end_image_prompt"
                )
            print(
                f"{shot['id']} {cond.motion_level}：首帧锁 {cond.start_strength:g}/{cond.start_binding}，"
                f"尾帧 {end_image.name} {cond.end_strength:g}/{cond.end_binding}，"
                f"mae={mae:.1f}，步数 {cond.denoise_plan[0]}+{cond.denoise_plan[1]}"
            )
        else:
            print(
                f"{shot['id']} {cond.motion_level}：首帧锁 {cond.start_strength:g}/{cond.start_binding}，"
                f"步数 {cond.denoise_plan[0]}+{cond.denoise_plan[1]}"
            )

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
        run_kw = dict(
            audio=audio_for_ltx,
            end_image=end_image,
            cond=cond,
        )
        code = runtime.render(
            shot,
            image,
            dest,
            log_path,
            width,
            height,
            frames,
            **run_kw,
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
            code = runtime.render(
                shot,
                image,
                dest,
                log_path,
                width,
                height,
                frames,
                **run_kw,
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
                run_kw["audio"] = audio_for_ltx
            log_path = artifacts_sub(out, "logs", f"{shot['id']}-small.log")
            code = runtime.render(
                shot,
                image,
                dest,
                log_path,
                width,
                height,
                frames,
                **run_kw,
            )
        if code != 0 or not dest.exists():
            _fail_shot(log_path, code)
            continue
        shot.pop("video_error", None)
        shot["video"] = str(dest)
        shot["video_frames"] = frames
        shot["video_mode"] = "a2v" if audio_for_ltx is not None else "i2v"
        shot["motion_level"] = cond.motion_level
        if end_image is not None:
            shot["end_frame"] = str(end_image)
    dump_json(story_path, story)
    if failures:
        raise SystemExit(f"有 {failures} 个镜头没有生成。成功的镜头仍会在合成阶段用上。")
