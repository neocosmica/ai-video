"""安装 flash-linear-attention + causal_conv1d，恢复 Qwen3.5 线性注意力加速。"""
import subprocess, pathlib, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY = pathlib.Path(r"C:/Users/annas/avwin/.venv/Scripts/python.exe")
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple/"

def run(cmd, timeout=1800, label=""):
    print("\n>>> %s" % label, flush=True)
    t0=time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    print("[%.0fs rc=%d]" % (time.time()-t0, p.returncode), flush=True)
    print(((p.stdout or "")[-1200:] + (p.stderr or "")[-600:]), flush=True)
    return p.returncode

# causal_conv1d 需编译；先试预编译 wheel
run([str(PY),"-m","pip","install","causal_conv1d","-i",TUNA], label="causal_conv1d")
run([str(PY),"-m","pip","install","flash-linear-attention","-i",TUNA], label="flash-linear-attention")

run([str(PY),"-c","import causal_conv1d; print('causal_conv1d OK', getattr(causal_conv1d,'__version__','?'))"],
    timeout=120, label="verify causal_conv1d")
run([str(PY),"-c","import fla; print('fla OK', getattr(fla,'__version__','?'))"],
    timeout=120, label="verify fla")
