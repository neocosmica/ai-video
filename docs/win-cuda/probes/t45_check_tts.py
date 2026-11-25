"""检查 qwen-tts 是否装上 + 探索 API。"""
import os, sys, io, inspect
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"
try:
    import qwen_tts
    print("[OK] qwen_tts installed", flush=True)
    print("  version:", getattr(qwen_tts,"__version__","?"), flush=True)
    print("  top:", [n for n in dir(qwen_tts) if not n.startswith("_")][:15], flush=True)
except ImportError:
    print("[MISSING] qwen_tts", flush=True)
    raise SystemExit(0)

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print("\n[2] Qwen3TTSModel 方法:", [n for n in dir(Qwen3TTSModel) if not n.startswith("_") and callable(getattr(Qwen3TTSModel,n))][:12], flush=True)
for m in ["from_pretrained","generate","generate_custom_voice"]:
    if hasattr(Qwen3TTSModel, m):
        try: print("  %s: %s" % (m, inspect.signature(getattr(Qwen3TTSModel,m))), flush=True)
        except Exception: pass
