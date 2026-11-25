"""正确分阶段：文本编码后立刻物化 embeds、丢弃 hidden_states，再释放。"""
import os, sys, io, torch, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def stage(t):
    free,_=torch.cuda.mem_get_info()
    print("  [%s] free=%.2fGB peak=%.2fGB" % (t, free/2**30, torch.cuda.max_memory_allocated()/2**30), flush=True)

print("=== 段1: 文本编码（正确释放）===", flush=True)
from transformers import Gemma3ForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
te = Gemma3ForConditionalGeneration.from_pretrained(
    "Lightricks/LTX-2", subfolder="text_encoder", quantization_config=qcfg,
    device_map={"":0}, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
tok = AutoTokenizer.from_pretrained("Lightricks/LTX-2", subfolder="text_encoder")
stage("text_encoder loaded")

ids = tok("雨夜便利店，林晚推门而入", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = te(**ids, output_hidden_states=True)
# 关键：把需要的 embeds 精简后拷到 CPU 或小张量，然后坚决丢掉所有引用
embeds = out.hidden_states[-1].detach().to(torch.bfloat16)
embeds_cpu = embeds.cpu()           # ← 放到 CPU，GPU 上不再留
stage("after encode")

del out, embeds, ids, te, tok
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
stage("after release")     # 期望 free ≈ 10.8GB

print("\n=== 段2: transformer Q2_K（应能装上）===", flush=True)
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from huggingface_hub import hf_hub_download
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q2_K.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
torch.cuda.reset_peak_memory_stats()
t=time.time()
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
print("  载入 %.1fs" % (time.time()-t), flush=True)
stage("transformer loaded")
del tr; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
stage("after release 2")
print("\n[OK] 分阶段成立：峰值 = max(单段)，不累加", flush=True)
