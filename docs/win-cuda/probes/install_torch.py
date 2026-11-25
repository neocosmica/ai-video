"""阶段0-A：建 venv 并安装 torch cu128。分批执行，每步报告状态。"""
import subprocess, sys, os, pathlib, time

VENV = pathlib.Path(r"C:/Users/annas/avwin/.venv")
PY = VENV / "Scripts" / "python.exe"

def run(cmd, timeout=3600, label=""):
    print(f"\n>>> {label or ' '.join(str(c) for c in cmd[:4])}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-800:]
    print(f"[{dt:.0f}s rc={p.returncode}]", flush=True)
    print(tail, flush=True)
    return p.returncode

if not PY.exists():
    print("创建 venv...", flush=True)
    VENV.parent.mkdir(parents=True, exist_ok=True)
    rc = run([sys.executable, "-m", "venv", str(VENV)], label="venv create")
    if rc != 0:
        sys.exit("venv 创建失败")

# 升级 pip 并装 torch（走清华源拉通用包，cu128 单独指 index）
run([str(PY), "-m", "pip", "install", "-U", "pip", "-q",
     "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], timeout=300, label="pip upgrade")

print("\n开始装 torch cu128（约 2.5GB，耐心等）...", flush=True)
rc = run([str(PY), "-m", "pip", "install", "torch", "torchvision", "torchaudio",
          "--index-url", "https://download.pytorch.org/whl/cu128"],
         timeout=5400, label="torch cu128 install")
print("torch install rc =", rc)
