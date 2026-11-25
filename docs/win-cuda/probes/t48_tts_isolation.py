"""验证：TTS 用独立 venv（transformers 4.57）是否可行——按 Mac 端 LTX/MusicGen 分环境思路。"""
import subprocess, sys, io, time, pathlib, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = pathlib.Path(r"C:/Users/annas/avwin")
TTSVENV = BASE / "venv-tts"
PY = TTSVENV / "Scripts" / "python.exe"
SYS_PY = r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
T = "https://pypi.tuna.tsinghua.edu.cn/simple/"
PT = "https://download.pytorch.org/whl/cu128"

def run(cmd, label, timeout=2400):
    print("\n>>> %s" % label, flush=True)
    t0=time.time()
    p=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    print("  [%.0fs rc=%d]" % (time.time()-t0, p.returncode), flush=True)
    print("  " + ((p.stdout or "")[-400:]+(p.stderr or "")[-400:]).replace("\n"," ")[:700], flush=True)
    return p.returncode

if not PY.exists():
    run([SYS_PY,"-m","venv",str(TTSVENV)], "建 TTS venv")
run([str(PY),"-m","pip","install","-U","pip","-q","-i",T], "pip 升级")
run([str(PY),"-m","pip","install","torch","torchaudio","--index-url",PT], "torch cu128 (tts)")
run([str(PY),"-m","pip","install","transformers==4.57.3","accelerate==1.12.0",
     "soundfile","librosa","einops","onnxruntime","sox","numpy<3","-i",T], "tts deps")
run([str(PY),"-m","pip","install","--no-deps","qwen-tts","-i",T], "qwen-tts no-deps")
run([str(PY),"-c","import transformers,qwen_tts;print('transformers',transformers.__version__);print('qwen_tts OK')"], "验证导入")
