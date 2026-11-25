"""验收补测：网页「连续多阶段」分支用的 `pipeline run --from storyboard` 全链路。

实测：rc=0 / 640s / 七阶段分隔行齐备 / final.mp4 4.46s 106帧 384x640@24fps。
"""
import os, sys, io, pathlib, tarfile, subprocess, time, shutil, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = pathlib.Path(r"C:/Users/annas/avwin/repo")
PY = r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
SRC = r"C:/Users/annas/probes/src.tgz"
if pathlib.Path(SRC).exists():
    with tarfile.open(SRC) as tf: tf.extractall(REPO)
PROJ = pathlib.Path(r"C:/Users/annas/avwin/projects/runfrom")
if PROJ.exists(): shutil.rmtree(PROJ)
PROJ.mkdir(parents=True)
(PROJ/"novel.txt").write_text(
 """雨下得很大。林晚推开便利店的玻璃门，收银的年轻人抬头，两个人都停住了。
是周屿。三年前说好一起离开这座城的人，现在穿着绿围裙站在收银台后面。
「你没事吧？」林晚的声音有点抖。
「我没事。」周屿笑了笑，「一点伤都没有。」""", encoding="utf-8")
env=os.environ.copy(); env.update({"HF_ENDPOINT":"https://hf-mirror.com","PYTHONUTF8":"1",
 "PYTHONPATH":str(REPO),"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
cmd=[PY,"-m","pipeline","run","--from","storyboard","--until","assemble",
     "--novel",str(PROJ/"novel.txt"),"--out",str(PROJ),"--force"]
print("[1] %s" % " ".join(cmd[-8:]), flush=True)
t0=time.time()
r=subprocess.run(cmd,cwd=str(REPO),capture_output=True,text=True,
                 encoding="utf-8",errors="replace",env=env)
print("   rc=%d  %.0fs" % (r.returncode, time.time()-t0), flush=True)
out=(r.stdout or "")+(r.stderr or "")
seps=[l.strip() for l in out.splitlines() if l.strip().startswith("==")]
print("   阶段分隔:", seps, flush=True)
tail=[l.strip() for l in out.splitlines() if l.strip()][-10:]
for l in tail: print("   ", l[:150], flush=True)
print("\n[2] final.mp4:", (PROJ/"final.mp4").exists(), flush=True)
if (PROJ/"final.mp4").exists():
    print("   size %d KB" % ((PROJ/"final.mp4").stat().st_size//1024), flush=True)
