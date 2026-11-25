"""不下载权重，只验证 diffusers 0.43 是否具备 LTX2 / Flux2 管道与所需类。"""
import diffusers, inspect
print("diffusers", diffusers.__version__)

print("\n=== 可用 pipeline 类 ===")
for name in ["LTX2Pipeline","LTX2Image2VideoPipeline","LTX2ConditionPipeline",
             "Flux2KleinPipeline","Flux2Pipeline","FluxPipeline"]:
    ok = hasattr(diffusers, name)
    print(f"  {name:28s} {'OK' if ok else 'MISSING'}")

print("\n=== LTX2 相关模型类 ===")
import diffusers.models as dm
for name in ["LTX2Transformer2DModel","AutoencoderKLLTX2Video","AutoencoderKLLTX2Audio",
             "LTX2TextConnectors","LTX2LatentUpsampler","LTX2Vocoder","Flux2Transformer2DModel"]:
    ok = hasattr(dm, name)
    print(f"  {name:32s} {'OK' if ok else 'MISSING'}")

print("\n=== 是否有 flux3 支持 ===")
for mod in ["diffusers.pipelines.flux3","diffusers.models.transformers.transformer_flux3"]:
    try:
        __import__(mod); print(f"  {mod}: OK")
    except ImportError:
        print(f"  {mod}: MISSING")

print("\n=== LTX2Pipeline.__call__ 关键参数 ===")
if hasattr(diffusers,"LTX2Pipeline"):
    sig = inspect.signature(diffusers.LTX2Pipeline.__call__)
    keys = [p for p in sig.parameters if p not in ("self","args","kwargs")]
    print("  参数:", keys)
    print("  model_cpu_offload_seq:", diffusers.LTX2Pipeline.model_cpu_offload_seq)

print("\n=== GGUF 量化配置导出 ===")
from diffusers import PipelineQuantizationConfig, GGUFQuantizationConfig
print("  PipelineQuantizationConfig OK")
print("  GGUFQuantizationConfig OK")
