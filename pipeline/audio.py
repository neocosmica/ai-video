from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.config import Settings
from pipeline.mirrors import child_env
from pipeline.util import dump_json, load_json, resolve, stable_seed
from pipeline.voices import (
    default_music_prompt,
    emotion_instruct,
    infer_gender,
    pick_voice,
)


def _uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("找不到 uv。请先运行 scripts/setup.sh")


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

    print(f"加载 TTS：{settings.tts_model}")
    model = load_model(settings.tts_model)
    results = []
    for shot, dest, profile, line in jobs:
        if dest.exists() and not force:
            results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": line})
            continue
        print(f"对白 {shot['id']} · {profile['voice']} · {profile['emotion']}：{line}")
        chunks = []
        sample_rate = 24000
        for result in model.generate_custom_voice(
            text=line,
            speaker=profile["voice"],
            language="Chinese",
            instruct=profile["instruct"],
        ):
            chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        if not chunks:
            raise SystemExit(f"TTS 没有产出音频：{shot['id']}")
        audio = np.concatenate(chunks)
        _save_mono_wav(dest, audio, sample_rate)
        results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": line})
    return results


def _estimate_timeline(settings: Settings, out: Path, story: dict) -> float:
    total = 0.0
    for shot in story["shots"]:
        clip = out / "clips" / f"{shot['id']}.mp4"
        if not clip.exists():
            total += settings.video_frames / max(1, settings.video_fps)
            continue
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
        except ValueError:
            total += settings.video_frames / max(1, settings.video_fps)
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

    print(f"生成配乐（MusicGen small，约 {duration:.0f}s）：{prompt}")
    env = child_env()
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    # 该仓库的 console script 依赖 cwd 包布局，直接跑 generate.py 更稳。
    cmd = [
        _uv(),
        "run",
        "--directory",
        str(settings.musicgen_dir),
        "python",
        "generate.py",
        prompt,
        "-m",
        settings.music_model,
        "-d",
        str(int(round(duration))),
        "-o",
        str(dest),
        "--no-open",
    ]
    _ = seed  # MusicGen-MLX CLI 暂不暴露 seed；保留变量便于以后接上
    log = out / "logs" / "bgm.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0 or not dest.exists():
        raise SystemExit(f"配乐失败，详见 {log}")
    return dest


def cmd_audio(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    voices = _generate_dialogue(settings, story, out, args.force)
    music = _generate_music(settings, story, out, args.force)
    manifest = {
        "tts_model": settings.tts_model,
        "music_model": settings.music_model,
        "music": str(music),
        "music_prompt": str(story.get("music_prompt") or default_music_prompt(story.get("style", ""), story.get("title", ""))),
        "voices": voices,
    }
    dump_json(out / "audio" / "manifest.json", manifest)
    # 把推断出的音色写回人物表，方便 Agent 下次改
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
    print(f"音频就绪：对白 {len(voices)} 条，配乐 {music}")
