"""Qwen3-TTS 官方 PyTorch 包 + CUDA 实测。"""
import os, sys, io, torch, time, subprocess, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PY=r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
print("[1] 安装 qwen-tts", flush=True)
p=subprocess.run([PY,"-m","pip","install","qwen-tts","-i","https://pypi.tuna.tsinghua.edu.cn/simple/"],
                 capture_output=True,text=True,timeout=1800,encoding="utf-8",errors="replace")
print("  rc=%d" % p.returncode, flush=True)
print(((p.stdout or "")[-500:]+(p.stderr or "")[-400:]), flush=True)
