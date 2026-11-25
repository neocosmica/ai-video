"""纯 ASCII 输出，避免 GBK 编码掩盖异常。"""
import os, torch, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTHONUTF8"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

from huggingface_hub import hf_hub_download
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q4_K_S.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
print("[info] gguf:", os.path.getsize(g)/2**30, "GB", flush=True)

torch.cuda.reset_peak_memory_stats()
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
t1=time.time()
try:
    tr = LTX2VideoTransformer3DModel.from_single_file(
        g, config=c, quantization_config=qcfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print("[OK] loaded", f"{time.time()-t1:.1f}s", flush=True)
    print("[info] params", f"{sum(p.numel() for p in tr.parameters())/1e9:.2f}B", flush=True)
    import bitsandbytes as bnb
    n=sum(1 for m in tr.modules() if isinstance(m, bnb.nn.Linear4bit))
    print("[info] 4bit layers:", n, flush=True)
    free,tot=torch.cuda.mem_get_info()
    print(f"[VRAM] peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB reserved={torch.cuda.memory_reserved()/2**30:.2f}GB free={free/2**30:.2f}GB", flush=True)
except Exception as e:
    print("[FAIL]", type(e).__name__, flush=True)
    print(str(e)[:600], flush=True)
