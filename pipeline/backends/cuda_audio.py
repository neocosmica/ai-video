"""CUDA 音频后端：对白 TTS（Qwen3-TTS）+ 配乐（MusicGen）。

TTS 必须跑在**独立 venv**
-------------------------
``qwen-tts==0.1.1`` 钉死 ``transformers==4.57.3``，而 diffusers 的 LTX2 管道
需要 ``transformers>=5``（``Gemma4ForConditionalGeneration`` 等）。两者装不到一起：

    transformers 5.24 → import qwen_tts 即崩：
      TypeError: check_model_inputs() missing 1 required positional argument: 'func'

因此 TTS 走子进程 + 独立解释器（与 Mac 端 ``venvs/{main,ltx,musicgen}`` 并存的
思路一致）。调用方式：

    python -m pipeline.backends.cuda_audio tts --text "..." --speaker Serena --out x.wav

配乐则在主环境跑（``transformers`` 的 ``MusicgenForConditionalGeneration``
两边都兼容）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

#: 默认 TTS venv 位置（相对项目根）
DEFAULT_TTS_VENV = Path("venvs") / "tts"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_MUSIC_MODEL = "facebook/musicgen-small"

TTS_ENV_VAR = "NOVEL_VIDEO_TTS_PYTHON"


def tts_python(root: Path | None = None) -> Path:
    """定位 TTS 独立解释器。

    查找顺序：``NOVEL_VIDEO_TTS_PYTHON`` 环境变量 → ``<root>/venvs/tts`` →
    ``<root>/../venv-tts``（探针时的手工布局）。
    """
    from pipeline.platform import python_in_venv
    from pipeline.util import ROOT

    override = (os.environ.get(TTS_ENV_VAR) or "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(f"{TTS_ENV_VAR} 指向的解释器不存在：{path}")
        return path

    base = root or ROOT
    for candidate in (base / DEFAULT_TTS_VENV, base.parent / "venv-tts"):
        exe = python_in_venv(candidate) if candidate.name == "tts" else (
            candidate / "Scripts" / "python.exe"
        )
        if exe.exists():
            return exe
    raise SystemExit(
        "找不到 TTS 独立环境。\n"
        "TTS 依赖 transformers==4.57.3，与主环境的 transformers>=5 冲突，必须单独装。\n"
        "请运行 scripts/setup-win.ps1（会建 venvs/tts），或用环境变量指定：\n"
        f"  set {TTS_ENV_VAR}=<venv>/Scripts/python.exe"
    )


def list_speakers(py: Path) -> list[str]:
    """列出 TTS 支持的音色（子进程查询，避免污染主环境）。"""
    code = (
        "import json,sys;"
        "from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel;"
        "m=Qwen3TTSModel.from_pretrained('" + DEFAULT_TTS_MODEL + "',device_map='cpu');"
        "print(json.dumps(m.get_supported_speakers() or []))"
    )
    try:
        out = subprocess.check_output([str(py), "-c", code], text=True, timeout=300,
                                      stderr=subprocess.DEVNULL)
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return []


def synth_line(
    *,
    text: str,
    speaker: str,
    language: str = "Chinese",
    model: str = DEFAULT_TTS_MODEL,
    out_wav: Path,
    py: Path | None = None,
    instruct: str | None = None,
) -> Path:
    """合成一句对白到 ``out_wav``（24 kHz 单声道 PCM16）。

    走子进程调独立 venv，主环境不 import qwen_tts。
    """
    interpreter = py or tts_python()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        out_wav.unlink()

    payload = {
        "text": text,
        "speaker": speaker,
        "language": language or "Chinese",
        "model": model,
        "out": str(out_wav),
        "instruct": instruct,
    }
    code = (
        "import json,sys;"
        "payload=json.loads(sys.stdin.read());"
        "import torch;"
        "from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel;"
        "m=Qwen3TTSModel.from_pretrained(payload['model'],device_map='cuda:0',torch_dtype=torch.bfloat16);"
        "kwargs={'text':payload['text'],'speaker':payload['speaker'],'language':payload['language']};"
        "kwargs.update({'instruct':payload['instruct']} if payload.get('instruct') else {});"
        "wavs,sr=m.generate_custom_voice(**kwargs);"
        "import soundfile as sf, numpy as np;"
        "sf.write(payload['out'], np.asarray(wavs[0]), sr);"
        "print('OK', sr, len(wavs[0]))"
    )
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [str(interpreter), "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0 or not out_wav.exists():
        raise SystemExit(
            f"TTS 合成失败（{interpreter}）：\n"
            + (proc.stderr or proc.stdout or "")[-1500:]
        )
    return out_wav


# ---------------------------------------------------------------------------
# 配乐：MusicGen（主环境，无版本冲突）
# ---------------------------------------------------------------------------


def load_music_model(*, model: str = DEFAULT_MUSIC_MODEL, device: str = "cuda"):
    """加载 MusicGen，返回 ``(model, processor)``。"""
    from pipeline.backends import require_cuda
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    require_cuda(what="配乐阶段")
    processor = AutoProcessor.from_pretrained(model)
    estimator = MusicgenForConditionalGeneration.from_pretrained(
        model, dtype=torch.bfloat16
    ).to(device)
    estimator.eval()
    return estimator, processor


def generate_music(
    model: Any,
    processor: Any,
    *,
    prompt: str,
    seconds: float,
    seed: int | None = None,
    guidance: float = 3.0,
    out_wav: Path,
) -> Path:
    """按文本生成配乐 wav（32 kHz 单声道）。

    MusicGen 每个 token ≈ 1/50 s，据此把秒数换算成 ``max_new_tokens``。
    """
    import numpy as np
    import soundfile as sf
    import torch

    tokens = max(64, int(round(seconds * 50)))
    inputs = processor(text=[prompt], padding=True, return_tensors="pt",
                       sampling_rate=16000).to(model.device)
    gen_kwargs: dict[str, Any] = {
        "do_sample": True,
        "guidance_scale": float(guidance),
        "max_new_tokens": tokens,
    }
    if seed is not None:
        gen_kwargs["generator"] = torch.Generator(model.device).manual_seed(int(seed))
    with torch.no_grad():
        audio = model.generate(**inputs, **gen_kwargs)
    wav = audio[0, 0].float().cpu().numpy()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, np.asarray(wav), 32000)
    return out_wav


# ---------------------------------------------------------------------------
# CLI（供子进程入口）
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.backends.cuda_audio")
    sub = parser.add_subparsers(dest="cmd", required=True)
    tts = sub.add_parser("tts", help="合成一句对白")
    tts.add_argument("--text", required=True)
    tts.add_argument("--speaker", default="Serena")
    tts.add_argument("--language", default="Chinese")
    tts.add_argument("--out", required=True)
    tts.add_argument("--instruct", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "tts":
        path = synth_line(text=args.text, speaker=args.speaker, language=args.language,
                          out_wav=Path(args.out), instruct=args.instruct)
        print(path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
