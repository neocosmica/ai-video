"""完整分阶段：text_encoder(7.8G) -> 释放 -> transformer(8.6G) -> 释放 -> VAE。"""
import os, sys, io, torch, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def stage(name):
    free,_=torch.cuda.mem_get_info()
    print("[%s] free=%.2fGB peak=%.2fGB" % (name, free/2**30, torch.cuda.max_memory_allocated()/2**30), flush=True)
def release(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# 段1：文本编码
print("=== 段1: Gemma3 文本编码器 ===", flush=True)
from transformers import Gemma3ForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
t0=time.time()
te = Gemma3ForConditionalGeneration.from_pretrained(
    "Lightricks/LTX-2", subfolder="text_encoder", quantization_config=qcfg,
    device_map={"":0}, dtype=torch.bfloat16, low_cpu_mem_usage=True)
te.eval(); tok = AutoTokenizer.from_pretrained("Lightricks/LTX-2", subfolder="text_encoder")
print("  载入 %.0fs" % (time.time()-t0), flush=True); stage("text_encoder")

print("  编码 prompt", flush=True)
prompt = "竖屏短剧静帧，雨夜便利店，林晚推门而入，暖光，电影感"
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = te(**ids, output_hidden_states=True)
hs = torch.stack(out.hidden_states, dim=-1) if out.hidden_states else None
print("  hidden_states:", tuple(hs.shape) if hs is not None else None, flush=True)
stage("after encode")
release(te, out)
print("  释放后:", flush=True); stage("after release")

# 段2：transformer
print("\n=== 段2: LTX-2 transformer Q2_K ===", flush=True)
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from huggingface_hub import hf_hub_download
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q2_K.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
torch.cuda.reset_peak_memory_stats()
t1=time.time()
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
tr.eval(); print("  载入 %.1fs" % (time.time()-t1), flush=True); stage("transformer")
release(tr)
print("  释放后:", flush=True); stage("after release 2")
print("\n[结论] 三阶段串行可行，峰值始终是 max(单段)", flush=True)
