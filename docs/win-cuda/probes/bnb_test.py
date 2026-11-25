"""验证 bitsandbytes 4bit 在 RTX 4070 上是否可用（修正版）。"""
import torch, time, os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
print("torch", torch.__version__)
import bitsandbytes as bnb
print("bnb", bnb.__version__)

# 1. 直接构造 4bit 层（正确写法：先给 bf16 权重，再 .cuda() 触发量化）
from bitsandbytes.nn import Linear4bit
lin = Linear4bit(4096, 4096, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
lin = lin.cuda()
x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
y = lin(x)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"4bit matmul OK: out={tuple(y.shape)} {y.dtype} {dt*1000:.1f}ms")

# 权重是否真被量化（nf4 下 weight 应是 uint8）
print(f"weight dtype={lin.weight.dtype} (uint8 表示已量化)")
print(f"VRAM used={(torch.cuda.memory_allocated()/2**30):.3f}GB")

# 释放再测：4bit vs bf16 同样矩阵的显存差
del lin, x, y
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

lin2 = Linear4bit(8192, 8192, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
print(f"\n8192x8192 4bit: VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB "
      f"(bf16 需 {8192*8192*2/2**30:.2f}GB)")
del lin2
torch.cuda.empty_cache()

# 2. gguf
try:
    import gguf
    print("\ngguf OK:", getattr(gguf, "__version__", "?"))
    sizes = gguf.GGML_QUANT_SIZES
    print("Q3_K_M:", sizes.get(gguf.GGMLQuantizationType.Q3_K_M) if hasattr(gguf,'GGMLQuantizationType') else "n/a")
except Exception as e:
    print("gguf FAIL", type(e).__name__, str(e)[:150])

# 3. transformers 4bit 真模型
print("\n--- transformers 4bit 加载 Qwen3.5-2B ---")
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
try:
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B", quantization_config=cfg,
        device_map={"": 0}, dtype=torch.bfloat16)
    print("加载成功 VRAM=%.2fGB" % (torch.cuda.memory_allocated()/2**30))
    del m; torch.cuda.empty_cache()
except Exception as e:
    print("失败:", type(e).__name__, str(e)[:400])
