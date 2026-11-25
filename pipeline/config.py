from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    # 单镜帧数上限：有对白时镜头跟着 TTS 实际时长走，这里只决定「一镜最多能装多久」。
    # video_frames_auto=True 时由本机总内存查 token 预算表、再按分辨率现算覆盖。
    video_frames_max: int = 241
    video_frames_ceiling: int = 241
    video_frames_auto: bool = True
    # None = 用 pipeline.budget.TOKEN_BUDGET_TABLE；config 里可写成 [[16, 3120], ...]
    video_token_budget_table: list[list[float]] | None = None
    video_fps: int = 24
    video_low_ram: bool = True
    max_shots: int = 16
    # 小说→分镜：Qwen3-4B Instruct 4bit ≈ 3.3 GB，16 GB 统一内存可跑
    llm_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    # 短篇分镜 JSON 通常 1–2k tokens；过大只会拖慢首跑
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    music_model: str = "facebook/musicgen-small"
    # 配乐一次生成的最长秒数；成片多长就生成多长，避免 -stream_loop 在中间留硬接缝
    music_max_seconds: float = 180.0
    voice_volume: float = 1.35
    # 配乐：先 loudnorm 到 music_target_lufs，再乘 music_volume 作相对微调。
    # 对白期间用逐帧表达式做闪避（不是全程固定衰减），空镜段配乐能正常露出。
    music_volume: float = 0.5
    music_target_lufs: float = -20.0
    # 对白期间配乐的压低倍数（0.3 ≈ 下沉 10 dB），空镜段恢复原电平
    music_duck_gain: float = 0.3
    # 闪避窗口在对白前后各多留的秒数，避免压/放正好切在字头上
    music_duck_pad: float = 0.12
    # 混音总线限幅（0.89 ≈ -1 dBFS），防止人声叠配乐冲过满刻度削波
    mix_limit: float = 0.89
    ltx_dir: Path = ROOT / "vendor" / "ltx-2-mlx"
    musicgen_dir: Path = ROOT / "vendor" / "musicgen-mlx"
    ffmpeg: Path = ROOT / "tools" / "ffmpeg"
    ffprobe: Path = ROOT / "tools" / "ffprobe"
    # resolve_frame_cap 的明细，供 doctor / budget 打印；不入库
    budget_detail: dict[str, Any] = field(default_factory=dict, repr=False)
    # ---- 推理后端（见 pipeline/backends）----
    # auto：macOS→mlx，Windows/Linux 有 NVIDIA→cuda
    backend: str = "auto"
    # CUDA 后端专用参数（Mac 端忽略）
    cuda_video_gguf: str = "ltx-2-19b-dev-Q2_K.gguf"
    cuda_video_gguf_repo: str = "unsloth/LTX-2-GGUF"
    cuda_conf_repo: str = "Lightricks/LTX-2"
    cuda_image_model: str = "black-forest-labs/FLUX.2-klein-4B"
    cuda_llm_model: str = "Qwen/Qwen3.5-4B"
    cuda_llm_quant: str = "4bit"
    cuda_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def token_budget_table(self) -> tuple[tuple[float, int], ...] | None:
        raw = self.video_token_budget_table
        if not raw:
            return None
        return tuple((float(row[0]), int(row[1])) for row in raw if len(row) >= 2)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        cfg_path = path or (ROOT / "config.json")
        data: dict = {}
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}
        picked = {k: v for k, v in data.items() if k in known and k != "budget_detail"}
        settings = cls(**picked)
        settings.image_width = snap(settings.image_width, 16)
        settings.image_height = snap(settings.image_height, 16)
        settings.portrait_width = snap(settings.portrait_width, 16)
        settings.portrait_height = snap(settings.portrait_height, 16)
        # 分辨率必须先 snap：video_frames_auto 依赖 width/height
        settings.video_width = snap(settings.video_width, 64)
        settings.video_height = snap(settings.video_height, 64)
        settings.video_frames = snap_frames(settings.video_frames)
        settings.video_frames_min = snap_frames(settings.video_frames_min)
        settings.video_frames_ceiling = snap_frames(settings.video_frames_ceiling)
        settings.video_frames_max = snap_frames(settings.video_frames_max)

        if settings.video_frames_auto:
            from pipeline.budget import resolve_frame_cap

            frames, detail = resolve_frame_cap(
                height=settings.video_height,
                width=settings.video_width,
                fps=settings.video_fps,
                min_frames=settings.video_frames_min,
                ceiling=settings.video_frames_ceiling,
                table=settings.token_budget_table(),
            )
            settings.video_frames_max = frames
            settings.budget_detail = detail

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
