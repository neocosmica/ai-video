"""MusicGen small + CUDA 实测。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

print("[1] 加载 MusicGen small", flush=True)
from transformers import MusicgenForConditionalGeneration, AutoProcessor
t0=time.time()
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", dtype=torch.bfloat16).to("cuda")
model.eval()
print("  %.0fs VRAM=%.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/2**30), flush=True)
torch.cuda.reset_peak_memory_stats()

print("[2] 生成 10s 配乐", flush=True)
text = "Warm emotional orchestral ballad, soft piano and strings, no vocals, cinematic"
inputs = proc(text=[text], padding=True, return_tensors="pt", sampling_rate=16000).to("cuda")
t1=time.time()
try:
    with torch.no_grad():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=256)
    torch.cuda.synchronize()
    print("  [OK] %.0fs" % (time.time()-t1), flush=True)
    sr = 32000
    wav = audio[0,0].float().cpu().numpy()
    print("  采样: %d 点 = %.2fs" % (len(wav), len(wav)/sr), flush=True)
    import soundfile as sf
    out = r"C:/Users/annas/probes/musicgen_test.wav"
    sf.write(out, wav, sr)
    print("  保存:", out, flush=True)
    print("  [VRAM] peak=%.2fGB" % (torch.cuda.max_memory_allocated()/2**30), flush=True)
except Exception as e:
    print("  [FAIL]", type(e).__name__, flush=True); print(str(e)[:300], flush=True)
