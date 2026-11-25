"""网页控制台端到端验收（Windows/CUDA），验收目标 c 的实测脚本。

流程与网页操作一致：
  round 1：/api/artifact/plan(kind=storyboard) + queue/add + start  -> 生成 storyboard.json
  round 2：/api/artifact/plan(kind=final) + queue/add(posts=全部后置) + start -> final.mp4

要点（踩过的坑）：
  - /api/state 的任务字段是 tasks，不是 jobs；必须带 ?project=<项目名>
  - 分镜不存在时 final 的前置/后置闭包展不开，必须先跑一轮分镜
  - 队列接口入参是 kind/id/posts，不是 items 列表

实测结果：两轮均 success，产出 final.mp4（4.12s / 99 帧 / 384x640@24fps / h264+aac）。
"""
import os, sys, io, pathlib, tarfile, time, json, subprocess, urllib.request, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = pathlib.Path(r"C:/Users/annas/avwin/repo")
PY = r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
SRC = r"C:/Users/annas/probes/src.tgz"
if pathlib.Path(SRC).exists():
    with tarfile.open(SRC) as tf: tf.extractall(REPO)
PROJ_NAME = "webacc"
PROJ = pathlib.Path(r"C:/Users/annas/avwin/projects") / PROJ_NAME
if PROJ.exists(): shutil.rmtree(PROJ)
PROJ.mkdir(parents=True)
(PROJ/"novel.txt").write_text(
 """雨下得很大。林晚推开便利店的玻璃门，收银的年轻人抬头，两个人都停住了。
是周屿。三年前说好一起离开这座城的人，现在穿着绿围裙站在收银台后面。
「你没事吧？」林晚的声音有点抖。
「我没事。」周屿笑了笑，「一点伤都没有。」""", encoding="utf-8")
env=os.environ.copy(); env.update({"HF_ENDPOINT":"https://hf-mirror.com","PYTHONUTF8":"1",
 "PYTHONPATH":str(REPO),"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
PORT=18762; BASE="http://127.0.0.1:%d"%PORT
def post(p,payload=None,t=120):
    req=urllib.request.Request(BASE+p,data=json.dumps(payload or {}).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=t) as r: return json.loads(r.read().decode())
def get(p,t=60):
    with urllib.request.urlopen(BASE+p,timeout=t) as r: return json.loads(r.read().decode())
print("[1] 起服务", flush=True)
lg=open(r"C:/Users/annas/probes/webacc_srv.log","w",encoding="utf-8")
proc=subprocess.Popen([PY,"-m","pipeline","web","--port",str(PORT)],cwd=str(REPO),
                      stdout=lg,stderr=subprocess.STDOUT,env=env)
ok=False
for _ in range(120):
    time.sleep(1)
    try: get("/api/state?project=webacc",t=5); ok=True; break
    except Exception:
        if proc.poll() is not None: break
print("   ready=%s"%ok, flush=True)
assert ok, open(r"C:/Users/annas/probes/webacc_srv.log",encoding="utf-8",errors="replace").read()[-800:]
novel=(PROJ/"novel.txt").read_text(encoding="utf-8")

def wait_job(tag, grace=25):
    """等服务把任务登记为 running 再开始判结束，避免启动瞬间误判完成。"""
    t0=time.time(); last=""; seen_running=False
    while True:
        time.sleep(5)
        st=get("/api/state?project="+PROJ_NAME); jobs=st.get("tasks") or []
        run=[j for j in jobs if j.get("status")=="running"]
        if run:
            seen_running=True
            j=run[0]
            line="%s|%s"%(j.get("status"),j.get("stage"))
            if line!=last:
                print("   %5.0fs %s"%(time.time()-t0,line),flush=True); last=line
        elif seen_running or time.time()-t0>grace:
            if jobs:
                return jobs[0].get("status")
            return "done(nojob)"

try:
    print("\n[2] round1: 生成分镜", flush=True)
    p1=post("/api/artifact/plan",{"kind":"storyboard","out":str(PROJ),"novel_text":novel})
    post("/api/queue/add",{"kind":"storyboard","out":str(PROJ),"novel_text":novel,"posts":[]})
    post("/api/queue/start",{"out":str(PROJ),"novel_text":novel})
    print("   ->", wait_job("r1"), "| storyboard.json:", (PROJ/"storyboard.json").exists(), flush=True)
    assert (PROJ/"storyboard.json").exists(), "分镜未生成"

    print("\n[3] round2: 以 final 为目标入队", flush=True)
    p2=post("/api/artifact/plan",{"kind":"final","out":str(PROJ)})
    posts=[{"kind":x["kind"],"id":x.get("id") or ""} for x in (p2.get("posts") or [])]
    print("   plan 前置=%d 后置=%d"%(len(p2.get("prereqs") or []),len(posts)), flush=True)
    add=post("/api/queue/add",{"kind":"final","out":str(PROJ),"posts":posts})
    items=(add.get("queue") or {}).get("items") or []
    print("   入队 %d 项:"%len(items), [i["kind"] for i in items], flush=True)
    assert items, "队列为空"
    post("/api/queue/start",{"out":str(PROJ)})
    print("   ->", wait_job("r2"), flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=20)
    except Exception: proc.kill()
    lg.close()
print("\n[4] 产物", flush=True)
for p in sorted(PROJ.rglob("*")):
    if p.is_file(): print("   %-30s %6d KB"%(p.relative_to(PROJ),p.stat().st_size//1024), flush=True)
print("\n[5] final.mp4 存在:", (PROJ/"final.mp4").exists(), flush=True)
