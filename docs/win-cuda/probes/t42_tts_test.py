"""Qwen3-TTS 推理测试（若已安装）。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
try:
    import qwen_tts
    print("[ok] qwen_tts 已安装", flush=True)
except ImportError as e:
    print("[skip] qwen_tts 未安装:", e, flush=True)
    sys.exit(0)

print("\n[1] 探索 API", flush=True)
print("  dir:", [n for n in dir(qwen_tts) if not n.startswith("_")][:20], flush=True)
print("\n[2] 尝试加载模型", flush=True)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print("  Qwen3TTSModel OK", flush=True)
import inspect
print("  from_pretrained sig:", inspect.signature(Qwen3TTSModel.from_pretrained) if hasattr(Qwen3TTSModel,"from_pretrained") else "n/a", flush=True)
