"""FLUX.2 Klein 4B（非 gated，diffusers 原生管道）+ bnb4bit 出图测试。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

print("[1] 加载 Flux2KleinPipeline（bf16 先测可行性）", flush=True)
from diffusers import Flux2KleinPipeline
import diffusers
print("  diffusers", diffusers.__version__, flush=True)
t0=time.time()
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
print("  加载 %.0fs" % (time.time()-t0), flush=True)
torch.cuda.reset_peak_memory_stats()

print("[2] 逐组件上卡（offload 序列）", flush=True)
pipe.enable_model_cpu_offload()
print("  offload 已启用, seq:", getattr(pipe,"model_cpu_offload_seq",None), flush=True)

print("[3] 出图 576x1024", flush=True)
t1=time.time()
try:
    img = pipe(prompt="竖屏短剧静帧，中近景，雨夜便利店，暖光，电影感，写实华人面孔",
               height=1024, width=576, num_inference_steps=4, guidance_scale=1.0,
               generator=torch.Generator("cuda").manual_seed(42)).images[0]
    torch.cuda.synchronize()
    print("  [OK] %.0fs" % (time.time()-t1), flush=True)
    out = r"C:/Users/annas/probes/flux2_test.png"
    img.save(out)
    print("  保存:", out, img.size, flush=True)
    print("  [VRAM] peak=%.2fGB" % (torch.cuda.max_memory_allocated()/2**30), flush=True)
    free,tot=torch.cuda.mem_get_info(); print("  free=%.2fGB" % (free/2**30), flush=True)
except Exception as e:
    print("  [FAIL]", type(e).__name__, flush=True); print(str(e)[:400], flush=True)
