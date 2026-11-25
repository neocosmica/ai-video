"""qwen-tts 装不动（依赖 gradio 等重包），改测纯推理所需的最小依赖。"""
import subprocess, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY=r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
T="https://pypi.tuna.tsinghua.edu.cn/simple/"

# 只装推理必需，跳过 gradio
print("[1] 装 qwen-tts --no-deps + 补最小依赖", flush=True)
for args, label in [
    (["install","--no-deps","qwen-tts","-i",T], "qwen-tts no-deps"),
    (["install","librosa","sox","einops","-i",T], "audio deps"),
]:
    t0=time.time()
    p=subprocess.run([PY,"-m","pip"]+args, capture_output=True, text=True, timeout=900,
                     encoding="utf-8", errors="replace")
    print("  [%s] rc=%d %.0fs" % (label, p.returncode, time.time()-t0), flush=True)
    print("  ", ((p.stdout or "")[-300:]+(p.stderr or "")[-300:]).replace("\n"," ")[:400], flush=True)

print("\n[2] 验证导入", flush=True)
p=subprocess.run([PY,"-c","import qwen_tts; print('OK', qwen_tts.__file__)"],
                 capture_output=True,text=True,encoding="utf-8",errors="replace")
print("  rc=%d out=%s err=%s" % (p.returncode,(p.stdout or "")[-200:],(p.stderr or "")[-300:]), flush=True)
