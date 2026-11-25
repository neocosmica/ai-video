"""诊断 Gemma3 4bit 释放不干净的原因。"""
import os, sys, io, torch, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def stats(tag):
    free,tot=torch.cuda.mem_get_info()
    print("  [%s] free=%.2fGB alloc=%.2fGB reserved=%.2fGB" % (
        tag, free/2**30, torch.cuda.memory_allocated()/2**30,
        torch.cuda.memory_reserved()/2**30), flush=True)

stats("baseline")
from transformers import Gemma3ForConditionalGeneration, BitsAndBytesConfig
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
te = Gemma3ForConditionalGeneration.from_pretrained(
    "Lightricks/LTX-2", subfolder="text_encoder", quantization_config=qcfg,
    device_map={"":0}, dtype=torch.bfloat16, low_cpu_mem_usage=True)
stats("loaded")

print("\n[释放尝试]", flush=True)
del te
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
stats("del+empty_cache")

# accelerate hook 可能持有引用
print("\n[检查残留]", flush=True)
import psutil, os as _os
print("  RSS=%.2fGB" % (psutil.Process(_os.getpid()).memory_info().rss/2**30), flush=True)
# 找所有还持有 CUDA 张量的对象
import gc as _gc
n_tensor=0
for obj in _gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            n_tensor+=1
    except Exception: pass
print("  仍在 GPU 的张量数:", n_tensor, flush=True)
