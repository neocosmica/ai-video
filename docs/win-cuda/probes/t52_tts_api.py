"""查 Qwen3TTSModel 正确 API 签名。"""
import sys, io, inspect
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print("[1] from_pretrained 签名:", flush=True)
print("  ", inspect.signature(Qwen3TTSModel.from_pretrained), flush=True)
print("\n[2] __init__ 签名:", flush=True)
print("  ", inspect.signature(Qwen3TTSModel.__init__), flush=True)
print("\n[3] 所有公开方法:", flush=True)
for n in dir(Qwen3TTSModel):
    if not n.startswith("_"):
        try:
            o=getattr(Qwen3TTSModel,n)
            if callable(o): print("  %-28s %s" % (n, inspect.signature(o)), flush=True)
        except Exception: print("  ", n, flush=True)
