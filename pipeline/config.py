from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.util import ROOT, resolve, snap, snap_frames


@dataclass
class Settings:
    image_model: str = "flux2-klein-4b"
    image_quantize: int = 4
    image_steps: int = 4
    image_width: int = 576
    image_height: int = 1024
    portrait_width: int = 768
    portrait_height: int = 1024
    video_model: str = "dgrauet/ltx-2.3-mlx-q4"
    video_width: int = 384
    video_height: int = 640
    video_frames: int = 49
    video_frames_min: int = 25
    video_frames_max: int = 97
    video_fps: int = 24
    video_low_ram: bool = True
    max_shots: int = 8
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    music_model: str = "facebook/musicgen-small"
    voice_volume: float = 1.35
    music_volume: float = 0.22
    ltx_dir: Path = ROOT / "vendor" / "ltx-2-mlx"
    musicgen_dir: Path = ROOT / "vendor" / "musicgen-mlx"
    ffmpeg: Path = ROOT / "tools" / "ffmpeg"
    ffprobe: Path = ROOT / "tools" / "ffprobe"

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        cfg_path = path or (ROOT / "config.json")
        data: dict = {}
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}
        picked = {k: v for k, v in data.items() if k in known}
        settings = cls(**picked)
        settings.image_width = snap(settings.image_width, 16)
        settings.image_height = snap(settings.image_height, 16)
        settings.portrait_width = snap(settings.portrait_width, 16)
        settings.portrait_height = snap(settings.portrait_height, 16)
        settings.video_width = snap(settings.video_width, 64)
        settings.video_height = snap(settings.video_height, 64)
        settings.video_frames = snap_frames(settings.video_frames)
        settings.video_frames_min = snap_frames(settings.video_frames_min)
        settings.video_frames_max = snap_frames(settings.video_frames_max)
        if settings.video_frames_min > settings.video_frames_max:
            settings.video_frames_min, settings.video_frames_max = (
                settings.video_frames_max,
                settings.video_frames_min,
            )
        settings.ltx_dir = Path(settings.ltx_dir)
        settings.musicgen_dir = Path(settings.musicgen_dir)
        settings.ffmpeg = Path(settings.ffmpeg)
        settings.ffprobe = Path(settings.ffprobe)
        if not settings.ltx_dir.is_absolute():
            settings.ltx_dir = resolve(settings.ltx_dir)
        if not settings.musicgen_dir.is_absolute():
            settings.musicgen_dir = resolve(settings.musicgen_dir)
        if not settings.ffmpeg.is_absolute():
            settings.ffmpeg = resolve(settings.ffmpeg)
        if not settings.ffprobe.is_absolute():
            settings.ffprobe = resolve(settings.ffprobe)
        return settings
