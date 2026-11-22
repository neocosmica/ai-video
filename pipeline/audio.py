from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.config import Settings
from pipeline.hf_cache import resolve_snapshot
from pipeline.mirrors import child_env
from pipeline.runtime import ensure_runtime_layout, musicgen_python, musicgen_venv_path
from pipeline.progress import run_tee
from pipeline.util import dump_json, frames_for_duration, load_json, resolve, stable_seed
from pipeline.voices import (
    default_music_prompt,
    emotion_instruct,
    infer_gender,
    pick_voice,
)


def _character_map(story: dict) -> dict[str, dict]:
    out = {}
    for char in story.get("characters") or []:
        out[char["id"]] = char
        out[char["name"]] = char
    return out


def _speaker_profile(shot: dict, characters: dict[str, dict]) -> dict:
    speaker = str(shot.get("speaker") or "").strip()
    char = characters.get(speaker) or {}
    gender = infer_gender(char.get("appearance", ""), char.get("gender") or shot.get("gender"))
    voice = pick_voice(gender, char.get("voice") or shot.get("voice"), char.get("appearance", ""))
    emotion = shot.get("emotion") or char.get("emotion") or "calm"
    return {
        "speaker": speaker,
        "gender": gender,
        "voice": voice,
        "emotion": emotion,
        "instruct": emotion_instruct(str(emotion)),
    }


def _save_mono_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    sf.write(str(path), arr, sample_rate)


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def pad_wav_to_duration(src: Path, dest: Path, duration_s: float) -> Path:
    """A2V 要求音频不少于镜头时长；不足则尾部补静音。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    data, sample_rate = sf.read(str(src), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    need = max(1, int(math.ceil(max(0.0, float(duration_s)) * max(1, int(sample_rate)))))
    if arr.shape[0] < need:
        arr = np.concatenate([arr, np.zeros(need - arr.shape[0], dtype=np.float32)])
    elif arr.shape[0] > need:
        # 镜头被 max_frames 截断时，A2V 只吃前 need 采样；成片混音仍用原 wav
        arr = arr[:need]
    _save_mono_wav(dest, arr, int(sample_rate))
    return dest


def _generate_dialogue(settings: Settings, story: dict, out: Path, force: bool) -> list[dict]:
    from mlx_audio.tts.utils import load_model

    voice_dir = out / "audio" / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    characters = _character_map(story)
    jobs = []
    for shot in story["shots"]:
        line = str(shot.get("dialogue") or "").strip()
        if not line:
            continue
        dest = voice_dir / f"{shot['id']}.wav"
        profile = _speaker_profile(shot, characters)
        jobs.append((shot, dest, profile, line))
        if dest.exists() and not force:
            print(f"已有对白 {shot['id']}，跳过")

    pending = [item for item in jobs if force or not item[1].exists()]
    if not pending:
        return [
            {
                "shot_id": shot["id"],
                "path": str(dest),
                **profile,
                "text": line,
            }
            for shot, dest, profile, line in jobs
            if dest.exists()
        ]

    tts_dir = resolve_snapshot(settings.tts_model)
    if tts_dir is None:
        raise SystemExit(f"找不到 TTS 权重 {settings.tts_model}。请先运行 scripts/setup.sh，不要在成片时再下载。")
    # 传本地目录，mlx_audio 就不会再 snapshot_download
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"加载 TTS：{tts_dir}")
    model = load_model(str(tts_dir))
    results = []
    for shot, dest, profile, line in jobs:
        if dest.exists() and not force:
            results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": line})
            continue
        print(f"对白 {shot['id']} · {profile['voice']} · {profile['emotion']}：{line}")
        chunks = []
        sample_rate = 24000
        # 条的总长按字数估，留足余量；说到 EOS 就停，避免默认 2048 步看起来几乎不动
        max_tokens = min(2048, max(128, len(line) * 8))
        for result in model.generate_custom_voice(
            text=line,
            speaker=profile["voice"],
            language="Chinese",
            instruct=profile["instruct"],
            verbose=True,
            max_tokens=max_tokens,
        ):
            chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        if not chunks:
            raise SystemExit(f"TTS 没有产出音频：{shot['id']}")
        audio = np.concatenate(chunks)
        _save_mono_wav(dest, audio, sample_rate)
        results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": line})
    return results


def apply_shot_timings(settings: Settings, story: dict, out: Path) -> None:
    """按对白 wav 写 audio_duration / video_frames，供 A2V 与配乐估长。"""
    voice_dir = out / "audio" / "voice"
    for shot in story["shots"]:
        voice = voice_dir / f"{shot['id']}.wav"
        line = str(shot.get("dialogue") or "").strip()
        if line and voice.exists():
            duration = wav_duration(voice)
            frames = frames_for_duration(
                duration,
                settings.video_fps,
                min_frames=settings.video_frames_min,
                max_frames=settings.video_frames_max,
            )
            shot["audio_duration"] = round(duration, 3)
            shot["video_frames"] = frames
        else:
            shot.pop("audio_duration", None)
            shot["video_frames"] = int(settings.video_frames)


def _estimate_timeline(settings: Settings, out: Path, story: dict) -> float:
    total = 0.0
    fps = max(1, settings.video_fps)
    voice_dir = out / "audio" / "voice"
    for shot in story["shots"]:
        frames = shot.get("video_frames")
        if frames:
            total += float(frames) / fps
            continue
        voice = voice_dir / f"{shot['id']}.wav"
        if voice.exists():
            total += max(0.1, wav_duration(voice))
            continue
        clip = out / "clips" / f"{shot['id']}.mp4"
        if clip.exists():
            proc = subprocess.run(
                [
                    str(settings.ffprobe),
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(clip),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            try:
                total += max(0.1, float(proc.stdout.strip()))
                continue
            except ValueError:
                pass
        total += settings.video_frames / fps
    return max(2.0, total)


def _generate_music(settings: Settings, story: dict, out: Path, force: bool) -> Path:
    music_dir = out / "audio"
    music_dir.mkdir(parents=True, exist_ok=True)
    dest = music_dir / "bgm.wav"
    prompt = str(story.get("music_prompt") or "").strip() or default_music_prompt(
        str(story.get("style") or ""), str(story.get("title") or "")
    )
    seed = int(story.get("music_seed") or stable_seed("bgm", prompt, story.get("title", "")))
    duration = min(30.0, max(8.0, _estimate_timeline(settings, out, story) + 2.0))
    if dest.exists() and not force:
        print(f"已有配乐，跳过：{dest}")
        return dest
    if not settings.musicgen_dir.exists():
        raise SystemExit(f"还没有 MusicGen 代码：{settings.musicgen_dir}。请先运行 scripts/setup.sh")
    script = settings.musicgen_dir / "generate.py"
    if not script.exists():
        raise SystemExit(f"MusicGen 源码不完整：缺少 {script}。请重新运行 scripts/setup.sh")

    ensure_runtime_layout(musicgen_dir=settings.musicgen_dir)
    py = musicgen_python(settings.musicgen_dir)
    model_dir = resolve_snapshot(settings.music_model)
    t5_dir = resolve_snapshot("t5-base")
    if (
        model_dir is None
        or not (model_dir / "state_dict.bin").is_file()
        or not (model_dir / "compression_state_dict.bin").is_file()
    ):
        raise SystemExit(
            f"找不到 MusicGen 权重 {settings.music_model}（需要 state_dict.bin 与 compression_state_dict.bin）。"
            "请先运行 scripts/setup.sh。"
        )
    if t5_dir is None or not (t5_dir / "model.safetensors").is_file():
        raise SystemExit("找不到 MusicGen 文本编码器 t5-base。请先运行 scripts/setup.sh。")
    print(f"配乐约 {duration:.0f}s：{prompt}")
    env = child_env()
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    # 直调 Caches 里的 Python + 仓库 cwd；禁止 uv run（会反复解析/偶发重建）
    env["UV_PROJECT_ENVIRONMENT"] = str(musicgen_venv_path())
    env.pop("VIRTUAL_ENV", None)
    # 权重目录已在本地。T5 仍按检查点里的名字 t5-base 加载，离线后只会读缓存。
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    cmd = [
        str(py),
        "generate.py",
        prompt,
        "-m",
        str(model_dir),
        "-d",
        str(int(round(duration))),
        "-o",
        str(dest),
        "--no-open",
    ]
    _ = seed  # MusicGen-MLX CLI 暂不暴露 seed；保留变量便于以后接上
    log = out / "logs" / "bgm.log"
    code = run_tee(cmd, log=log, env=env, cwd=settings.musicgen_dir, progress="配乐")
    if code != 0 or not dest.exists():
        raise SystemExit(f"配乐失败，详见 {log}")
    return dest


def cmd_audio(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    ensure_runtime_layout(musicgen_dir=settings.musicgen_dir)
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    voices = _generate_dialogue(settings, story, out, args.force)
    apply_shot_timings(settings, story, out)
    music = _generate_music(settings, story, out, args.force)
    manifest = {
        "tts_model": settings.tts_model,
        "music_model": settings.music_model,
        "music": str(music),
        "music_prompt": str(story.get("music_prompt") or default_music_prompt(story.get("style", ""), story.get("title", ""))),
        "voices": voices,
        "timeline_s": round(_estimate_timeline(settings, out, story), 2),
    }
    dump_json(out / "audio" / "manifest.json", manifest)
    # 把推断出的音色写回人物表，方便下次改分镜
    by_name = {c["name"]: c for c in story.get("characters") or []}
    for item in voices:
        char = by_name.get(item["speaker"])
        if not char:
            continue
        char.setdefault("gender", item["gender"])
        char.setdefault("voice", item["voice"])
    if not story.get("music_prompt"):
        story["music_prompt"] = manifest["music_prompt"]
    dump_json(story_path, story)
    timed = sum(1 for shot in story["shots"] if shot.get("audio_duration") is not None)
    print(
        f"音频就绪：对白 {len(voices)} 条（其中 {timed} 镜写入 video_frames），"
        f"配乐 {music}，估长约 {manifest['timeline_s']}s"
    )
