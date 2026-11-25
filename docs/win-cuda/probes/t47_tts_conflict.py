"""确认 qwen-tts 与 transformers 版本冲突的具体情况。"""
import subprocess, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY=r"C:/Users/annas/avwin/.venv/Scripts/python.exe"

print("[1] qwen-tts 声明的依赖", flush=True)
p=subprocess.run([PY,"-c","""
import importlib.metadata as md
for req in md.requires('qwen-tts'):
    print(' ', req)
"""], capture_output=True,text=True,encoding="utf-8",errors="replace")
print(p.stdout[-800:] if p.stdout else p.stderr[-400:], flush=True)

print("\n[2] 当前 transformers / torch 版本", flush=True)
p=subprocess.run([PY,"-c","import transformers,torch;print(transformers.__version__, torch.__version__)"],
                 capture_output=True,text=True,encoding="utf-8",errors="replace")
print(" ", (p.stdout or "").strip(), flush=True)

print("\n[3] 冲突函数签名", flush=True)
p=subprocess.run([PY,"-c","""
import inspect
from transformers.modeling_utils import check_model_inputs
print('  check_model_inputs:', inspect.signature(check_model_inputs))
"""], capture_output=True,text=True,encoding="utf-8",errors="replace")
print((p.stdout or "")[-500:], (p.stderr or "")[-300:], flush=True)
