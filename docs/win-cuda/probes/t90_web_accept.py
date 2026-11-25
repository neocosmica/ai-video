"""网页控制台端到端验收（Windows/CUDA）：起服务 -> API 入队 -> 启动 -> 产出 final.mp4。

用法（开发机）：
  python docs/win-cuda/probes/t90_web_accept.py

判据：/api/queue/start 拉起的任务最终 status 为 done，且项目目录出现 final.mp4。
"""
import os, sys, io, pathlib, tarfile, time, json, subprocess, urllib.request, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = pathlib.Path(r"C:/Users/annas/avwin/repo")
PY = r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
# 源码包由本地 tar 上传到开发机后再解包（见 docs/win-cuda/README.md）
SRC_TGZ = r"C:/Users/annas/probes/src.tgz"
if pathlib.Path(SRC_TGZ).exists():
    with tarfile.open(SRC_TGZ) as tf:
        tf.extractall(REPO)

PROJ = pathlib.Path(r"C:/Users/annas/avwin/projects/webdemo")
if PROJ.exists():
    shutil.rmtree(PROJ)
PROJ.mkdir(parents=True)
(PROJ / "novel.txt").write_text(
 """雨下得很大。林晚推开便利店的玻璃门，收银的年轻人抬头，两个人都停住了。
是周屿。三年前说好一起离开这座城的人，现在穿着绿围裙站在收银台后面。
「你没事吧？」林晚的声音有点抖。
「我没事。」周屿笑了笑，「一点伤都没有。」""", encoding="utf-8")

env = os.environ.copy()
env.update({"HF_ENDPOINT":"https://hf-mirror.com","PYTHONUTF8":"1",
            "PYTHONPATH":str(REPO),"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
PORT = 18761
BASE = "http://127.0.0.1:%d" % PORT

def post(path, payload=None, timeout=60):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

print("[1] 启动网页服务 :%d" % PORT, flush=True)
log = open(r"C:/Users/annas/probes/web_srv.log", "w", encoding="utf-8")
proc = subprocess.Popen([PY, "-m", "pipeline", "web", "--port", str(PORT)],
                        cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=env)
ready = False
for _ in range(120):
    time.sleep(1)
    try:
        get("/api/state", timeout=5); ready = True; break
    except Exception:
        if proc.poll() is not None: break
print("   ready=%s pid=%s" % (ready, proc.pid), flush=True)
if not ready:
    print(open(r"C:/Users/annas/probes/web_srv.log", encoding="utf-8", errors="replace").read()[-1500:])
    raise SystemExit(1)

try:
    print("\n[2] 规划（target=final，含全部前置与后置）", flush=True)
    plan = post("/api/artifact/plan", {"kind": "final", "out": str(PROJ)})
    print("   plan ok=%s target=%s" % (plan.get("ok"), (plan.get("target") or {}).get("label")), flush=True)
    print("   前置 %d 项，后置 %d 项" % (len(plan.get("prereqs") or []), len(plan.get("posts") or [])), flush=True)
    if not plan.get("ok"):
        raise SystemExit("规划失败：%s" % plan.get("error"))

    print("\n[3] 入队（目标 + 全部后置）", flush=True)
    posts = [{"kind": p["kind"], "id": p.get("id") or ""} for p in (plan.get("posts") or [])]
    add = post("/api/queue/add", {"kind": "final", "out": str(PROJ), "posts": posts})
    items = (add.get("queue") or {}).get("items") or add.get("items") or []
    print("   队列 %d 项：" % len(items), flush=True)
    for it in items:
        print("     - %-8s %-10s force=%s" % (it.get("kind"), it.get("id") or "-", it.get("force")), flush=True)

    print("\n[4] 启动任务", flush=True)
    start = post("/api/queue/start", {"out": str(PROJ)})
    print("   start:", json.dumps(start, ensure_ascii=False)[:260], flush=True)

    print("\n[5] 轮询直到成片", flush=True)
    t0 = time.time(); last = ""
    for _ in range(3600):
        time.sleep(5)
        st = get("/api/state")
        jobs = st.get("jobs") or []
        if jobs:
            j = jobs[0]
            line = "%s | %s | %s" % (j.get("status"), j.get("stage"),
                                     {k: v for k, v in (j.get("stage_status") or {}).items()})
            if line != last:
                print("   %6.0fs %s" % (time.time()-t0, line), flush=True); last = line
            if j.get("status") in ("done", "failed", "error", "stopped"):
                break
        if (PROJ / "final.mp4").exists() and not jobs:
            break
    print("\n[6] 结果", flush=True)
    for p in sorted(PROJ.rglob("*")):
        if p.is_file():
            print("   %-32s %6d KB" % (p.relative_to(PROJ), p.stat().st_size//1024), flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=20)
    except Exception: proc.kill()
    log.close()
