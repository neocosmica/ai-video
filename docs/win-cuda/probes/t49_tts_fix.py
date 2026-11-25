"""补装 torchaudio(CUDA) 并测 TTS。"""
import subprocess, sys, io, time, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY = r"C:/Users/annas/avwin/venv-tts/Scripts/python.exe"
T="https://pypi.tuna.tsinghua.edu.cn/simple/"
PT="https://download.pytorch.org/whl/cu128"
def run(cmd,label,timeout=2400):
    print("\n>>> %s" % label, flush=True); t0=time.time()
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,encoding="utf-8",errors="replace")
    print("  [%.0fs rc=%d]" % (time.time()-t0,p.returncode), flush=True)
    print("  "+((p.stdout or "")[-400:]+(p.stderr or "")[-400:]).replace("\n"," ")[:600], flush=True)
    return p.returncode

# 先看现有 torch 是否 CUDA
run([PY,"-c","import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"], "检查 torch")
# torchaudio 走清华（版本需匹配 torch 2.15）
run([PY,"-m","pip","install","torchaudio","-i",T],"torchaudio 清华")
run([PY,"-c","import torch,torchaudio;print('torch',torch.__version__,'audio',torchaudio.__version__,'cuda',torch.cuda.is_available())"],"验证")
run([PY,"-c","import qwen_tts; print('qwen_tts import OK')"],"qwen_tts 导入")
