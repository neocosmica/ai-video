"""对白与配乐：MLX 上的 Qwen3-TTS 与 MusicGen。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pipeline.backends.catalog import StageModel, weight_by_id
from pipeline.config import Settings
from pipeline.voices import (
    VOICE_MODE_CLONE,
    VOICE_MODE_DESIGN,
    VOICE_MODE_PRESET,
    design_instruct,
)


class MlxSpeech:
    def __init__(self, spec: StageModel, *, mode: str = VOICE_MODE_PRESET):
        if spec.runtime != "mlx_tts":
            raise SystemExit(f"对白运行时不是 mlx_tts：{spec.runtime}")
        from pipeline.hf_cache import resolve_snapshot

        tts_dir = resolve_snapshot(spec.repo)
        if tts_dir is None:
            raise SystemExit(
                f"找不到 TTS 权重 {spec.label}（{spec.repo}）。"
                "请先运行 scripts/setup.sh，或在网页里点「安装依赖」。"
            )
        self.spec = spec
        self.mode = mode
        # 传本地目录，mlx_audio 就不会再 snapshot_download
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"加载 TTS（{mode}）：{tts_dir}")
        from mlx_audio.tts.utils import load_model

        self._model = load_model(str(tts_dir))

    def synthesize(
        self,
        *,
        text: str,
        speaker: str = "",
        instruct: str = "",
        ref_audio: str | Path | None = None,
        ref_text: str = "",
        voice_design: str = "",
    ) -> tuple[np.ndarray, int]:
        """合成一句台词，返回波形与采样率。"""
        chunks: list[np.ndarray] = []
        sample_rate = 24000
        # 条的总长按字数估，留足余量；说到 EOS 就停，避免默认 2048 步看起来几乎不动
        max_tokens = min(2048, max(128, len(text) * 8))
        mode = self.mode
        if mode == VOICE_MODE_CLONE:
            ref_path = Path(ref_audio) if ref_audio else None
            if ref_path is None or not ref_path.is_file():
                raise SystemExit(
                    f"参考音频克隆需要有效的参考 wav（当前：{ref_audio!r}）。"
                    "请在角色卡上传参考音频。"
                )
            ref_line = (ref_text or "").strip()
            if not ref_line:
                raise SystemExit(
                    "参考音频克隆需要填写「参考音频里说的原文」（voice_ref_text）。"
                )
            for result in self._model.generate(
                text=text,
                ref_audio=str(ref_path),
                ref_text=ref_line,
                verbose=True,
                max_tokens=max_tokens,
            ):
                chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
                sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        elif mode == VOICE_MODE_DESIGN:
            design = design_instruct(voice_design, instruct)
            for result in self._model.generate_voice_design(
                text=text,
                language="Chinese",
                instruct=design,
                verbose=True,
                max_tokens=max_tokens,
            ):
                chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
                sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        else:
            if not speaker:
                raise SystemExit("预设配音缺少 speaker（如 Serena / Dylan）")
            for result in self._model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language="Chinese",
                instruct=instruct or "用自然对白的语气说",
                verbose=True,
                max_tokens=max_tokens,
            ):
                chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
                sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        if not chunks:
            raise SystemExit(f"TTS 没有产出音频：{text}")
        return np.concatenate(chunks), sample_rate

    def close(self) -> None:
        self._model = None


def render_music(
    settings: Settings,
    spec: StageModel,
    *,
    prompt: str,
    seconds: float,
    seed: int,
    dest: Path,
    log: Path,
) -> None:
    if spec.runtime != "musicgen":
        raise SystemExit(f"配乐运行时不是 musicgen：{spec.runtime}")
    from pipeline.hf_cache import resolve_snapshot
    from pipeline.mirrors import child_env
    from pipeline.progress import run_tee
    from pipeline.runtime import ensure_runtime_layout, musicgen_python, musicgen_venv_path

    if not settings.musicgen_dir.exists():
        raise SystemExit(f"还没有 MusicGen 代码：{settings.musicgen_dir}。请先运行 scripts/setup.sh")
    script = settings.musicgen_dir / "generate.py"
    if not script.exists():
        raise SystemExit(f"MusicGen 源码不完整：缺少 {script}。请重新运行 scripts/setup.sh")

    ensure_runtime_layout(musicgen_dir=settings.musicgen_dir)
    py = musicgen_python(settings.musicgen_dir)
    model_dir = resolve_snapshot(spec.repo)
    t5_repo = weight_by_id(spec, "t5").repo
    t5_dir = resolve_snapshot(t5_repo)
    if (
        model_dir is None
        or not (model_dir / "state_dict.bin").is_file()
        or not (model_dir / "compression_state_dict.bin").is_file()
    ):
        raise SystemExit(
            f"找不到配乐权重 {spec.label}（{spec.repo}，需要 state_dict.bin 与 compression_state_dict.bin）。"
            "请先运行 scripts/setup.sh，或在网页里点「安装依赖」。"
        )
    if t5_dir is None or not (t5_dir / "model.safetensors").is_file():
        raise SystemExit(f"找不到配乐文本编码器 {t5_repo}。请先运行 scripts/setup.sh。")
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
        str(int(round(seconds))),
        "-o",
        str(dest),
        "--no-open",
        "--seed",
        str(seed),
    ]
    print(f"配乐模型：{spec.label}，seed={seed}")
    code = run_tee(
        cmd,
        log=log,
        env=env,
        cwd=settings.musicgen_dir,
        progress="配乐",
        desc="MusicGen",
    )
    if code != 0 or not dest.exists():
        raise SystemExit(f"配乐失败，详见 {log}")
