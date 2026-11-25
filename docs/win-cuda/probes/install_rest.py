"""阶段0-A：安装 diffusers/transformers/gguf/bnb 等依赖。"""
import subprocess, pathlib, time

PY = pathlib.Path(r"C:/Users/annas/avwin/.venv/Scripts/python.exe")
TUNA = "https://pypi.tuna.tsinghua.edu.cn/simple/"

def run(cmd, timeout=3600, label=""):
    print(f"\n>>> {label}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    print(f"[{dt:.0f}s rc={p.returncode}]", flush=True)
    tail = (p.stdout or "")[-1200:] + (p.stderr or "")[-600:]
    print(tail, flush=True)
    return p.returncode

pkgs = [
    ("基础", ["diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub",
                "sentencepiece", "protobuf", "soundfile", "scipy", "einops", "tqdm", "pillow"]),
    ("量化", ["bitsandbytes", "gguf", "optimum"]),
]
for label, lst in pkgs:
    rc = run([str(PY), "-m", "pip", "install", *lst, "-i", TUNA],
             timeout=3600, label=f"install {label}")
    print(f"{label} rc={rc}")

print("\n=== 版本核对 ===")
code = '''
import importlib
for m in ["torch","diffusers","transformers","accelerate","bitsandbytes","gguf","safetensors"]:
    try:
        mod=importlib.import_module(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e: print(f"  {m}: FAIL {e}")
'''
run([str(PY), "-c", code], timeout=120, label="versions")
