"""给 TTS venv 装 CUDA torch（重试），然后实测 TTS 推理。"""
import subprocess, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY = r"C:/Users/annas/avwin/venv-tts/Scripts/python.exe"
PT="https://download.pytorch.org/whl/cu128"

def run(cmd,label,timeout=3000):
    print("\n>>> %s" % label, flush=True); t0=time.time()
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,encoding="utf-8",errors="replace")
    print("  [%.0fs rc=%d]" % (time.time()-t0,p.returncode), flush=True)
    print("  "+((p.stdout or "")[-350:]+(p.stderr or "")[-350:]).replace("\n"," ")[:600], flush=True)
    return p.returncode

ok=False
for attempt in range(3):
    rc=run([PY,"-m","pip","install","--force-reinstall","--no-deps","torch","--index-url",PT], "torch cu128 (try %d)"%(attempt+1))
    if rc==0:
        rc2=run([PY,"-c","import torch;print('torch',torch.__version__,torch.cuda.is_available())"], "验证 cuda")
        if rc2==0:
            p=subprocess.run([PY,"-c","import torch;print(torch.cuda.is_available())"],capture_output=True,text=True)
            if "True" in (p.stdout or ""):
                ok=True; break
if not ok:
    print("[WARN] CUDA torch 未装成功，TTS 将在 CPU 上验证", flush=True)
run([PY,"-c","import torch,torchaudio;print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"],"最终版本")
