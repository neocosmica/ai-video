"""测试 Gemma3 文本编码器：bf16 体积 vs 4bit，选可行方案。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

print("[1] 4bit 加载 Gemma3 文本编码器（LTX-2 用）", flush=True)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
REPO="Lightricks/LTX-2"
sub="text_encoder"
t0=time.time()
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
try:
    from transformers import Gemma3ForConditionalGeneration
    m = Gemma3ForConditionalGeneration.from_pretrained(
        REPO, subfolder=sub, quantization_config=qcfg,
        device_map={"":0}, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print("  [OK] %.0fs VRAM=%.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/2**30), flush=True)
    print("  params=%.2fB" % (sum(p.numel() for p in m.parameters())/1e9), flush=True)
    tok = AutoTokenizer.from_pretrained(REPO, subfolder=sub)
    print("  tokenizer OK", flush=True)
except Exception as e:
    print("  [FAIL]", type(e).__name__, flush=True); print("  "+str(e)[:400], flush=True)
