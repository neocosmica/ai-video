"""Qwen3-TTS 实测推理（隔离 venv，CUDA）。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
print("[1] 环境", flush=True)
print("  torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
import transformers; print("  transformers", transformers.__version__, flush=True)

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print("\n[2] 加载 Qwen3-TTS-12Hz-1.7B-CustomVoice", flush=True)
t0=time.time()
try:
    m = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda", dtype=torch.bfloat16)
    print("  加载 %.0fs VRAM=%.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/2**30), flush=True)
except Exception as e:
    print("  [FAIL load]", type(e).__name__, flush=True); print("  "+str(e)[:400], flush=True); raise SystemExit(1)

print("\n[3] 合成中文对白", flush=True)
import inspect
gen = None
for name in ["generate_custom_voice","generate","synthesize"]:
    if hasattr(m, name):
        print("  方法 %s: %s" % (name, inspect.signature(getattr(m,name))), flush=True)
        gen = getattr(m, name); break
if gen is None:
    print("  [FAIL] 找不到生成方法", flush=True); raise SystemExit(1)

t1=time.time()
try:
    text = "你没事吧？有没有被威胁？没有做傻事吧？"
    wavs, sr = gen(text=text, speaker="Serena", language="Chinese")
    print("  [OK] %.1fs sr=%d 长度=%s" % (time.time()-t1, sr, len(wavs[0]) if wavs else 0), flush=True)
    import soundfile as sf, numpy as np
    out = r"C:/Users/annas/probes/tts_test.wav"
    data = np.asarray(wavs[0])
    sf.write(out, data, sr)
    print("  保存:", out, "%.2fs 音频" % (len(data)/sr), flush=True)
    print("  [VRAM] peak=%.2fGB" % (torch.cuda.max_memory_allocated()/2**30), flush=True)
except Exception as e:
    print("  [FAIL gen]", type(e).__name__, flush=True); print("  "+str(e)[:500], flush=True)
