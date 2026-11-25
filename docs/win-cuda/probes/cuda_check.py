"""验证 CUDA 可用性 + 基础算力 + 显存读数。"""
import torch, time

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}  VRAM={props.total_memory/2**30:.2f} GB  cc={props.major}.{props.minor}")

dev = torch.device("cuda:0")
print("\n--- 基础算力测试 ---")
# 1. 小矩阵
a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
print(f"bf16 1024 matmul x20: {dt:.3f}s  ({20/dt:.1f} it/s)")

# 2. 大矩阵（更接近真实负载）
a2 = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b2 = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
cc = a2 @ b2
torch.cuda.synchronize()
dt = time.time() - t0
tflops = 2 * 4096**3 / dt / 1e12
print(f"bf16 4096 matmul single: {dt*1000:.1f}ms  {tflops:.1f} TFLOPS")

print("\n--- 显存读数 ---")
free, total = torch.cuda.mem_get_info()
print(f"free={free/2**30:.2f}GB total={total/2**30:.2f}GB")
print(f"allocated={torch.cuda.memory_allocated()/2**30:.2f}GB reserved={torch.cuda.memory_reserved()/2**30:.2f}GB")

print("\n--- 量化后端可用性 ---")
for mod in ["bitsandbytes", "diffusers", "transformers", "accelerate", "gguf"]:
    try:
        m = __import__(mod)
        print(f"  {mod}: {getattr(m,'__version__','?')}")
    except ImportError:
        print(f"  {mod}: 未安装")
