"""深挖：LTX2 与 flux3 的实际可用类名和管道情况。"""
import diffusers, inspect, pkgutil

print("=== diffusers.models.transformers 里 flux3/ltx2 相关 ===")
import diffusers.models.transformers as t
for m in pkgutil.iter_modules(t.__path__):
    if "flux" in m.name or "ltx" in m.name:
        print("  module:", m.name)

print("\n=== transformer_flux3 导出的类 ===")
try:
    import diffusers.models.transformers.transformer_flux3 as tf3
    classes = [n for n in dir(tf3) if not n.startswith("_") and n[0].isupper()]
    print("  ", classes)
except Exception as e:
    print("  ERR", e)

print("\n=== flux3 在顶层 models 是否可见 ===")
import diffusers.models as dm
for n in dir(dm):
    if "flux3" in n.lower() or "Flux3" in n:
        print("  ", n)

print("\n=== LTX2 实际类名（models/transformers） ===")
try:
    import diffusers.models.transformers.transformer_ltx2 as tl2
    classes = [n for n in dir(tl2) if n[0].isupper() and not n.startswith("_")]
    print("  ", classes)
except Exception as e:
    print("  ERR", str(e)[:100])

print("\n=== pipelines/ltx2 支持的管道 ===")
import diffusers.pipelines.ltx2 as p2
print("  __all__:", getattr(p2, "__all__", [n for n in dir(p2) if n[0].isupper()]))

print("\n=== LatentUpsampler / Vocoder 真实名字 ===")
import diffusers.models as dm2
for n in dir(dm2):
    if "Upsample" in n or "Vocoder" in n or "upsampler" in n.lower() or "vocoder" in n.lower():
        print("  ", n)

print("\n=== index2video 相关 ===")
for n in dir(diffusers):
    if "Video" in n and ("2" in n or "Image" in n):
        print("  ", n)
