"""尝试安装 triton（Windows 有社区版），这是 Qwen 加速的前提。"""
import subprocess, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY=r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
T="https://pypi.tuna.tsinghua.edu.cn/simple/"

def run(args, label, timeout=1800):
    print("\n>>> %s" % label, flush=True)
    t0=time.time()
    p=subprocess.run([PY,"-m","pip"]+args, capture_output=True, text=True,
                     timeout=timeout, encoding="utf-8", errors="replace")
    out=(p.stdout or "")+(p.stderr or "")
    print("[%.0fs rc=%d]" % (time.time()-t0, p.returncode), flush=True)
    print(out[-900:], flush=True)
    return p.returncode

# 1. 官方 triton（Windows 无 wheel，预期失败）
run(["install","triton","-i",T], "triton 官方")
# 2. 社区 windows 版
run(["install","triton-windows","-i",T], "triton-windows")

print("\n=== 验证 ===", flush=True)
p=subprocess.run([PY,"-c","import triton; print('triton', triton.__version__)"],
                 capture_output=True,text=True,encoding="utf-8",errors="replace")
print("rc=",p.returncode,(p.stdout or "")[-300:],(p.stderr or "")[-400:], flush=True)
