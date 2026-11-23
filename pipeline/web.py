"""本地网页控制台：填写小说片段、起流水线任务、看进度、重跑单个阶段。

零新增依赖，只用 Python 标准库：

    ./scripts/run.sh web --port 8790          # 浏览器打开 http://127.0.0.1:8790

职责边界（与 AGENTS.md 一致）：本地不加载聊天 LLM，分镜仍由人工/Agent 写好
`projects/<片名>/storyboard.json`；本界面负责填写小说、维护分镜、起任务、
看实时进度、对失败步骤单独重跑。

每个项目一个独立文件夹 `projects/<片名>/`（小说、分镜、产物都在里面）；
任务历史也属于项目：写在 `<项目>/work/webui/state.json`，重启服务不丢，
打开网页第一步是选项目，之后看到的任务、镜头产物都是这个项目的。
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline.mirrors import apply_mirrors

apply_mirrors()

from pipeline.progress_model import (
    STAGE_LABELS as _PM_STAGE_LABELS,
    STAGE_WEIGHTS as _PM_STAGE_WEIGHTS,
    STAGES as _PM_STAGES,
    ProgressEngine,
)
from pipeline.projects import (
    PROJECTS_DIRNAME,
    artifacts_dir,
    artifacts_sub,
    novel_candidates,
    project_dir,
)
from pipeline.util import ROOT, resolve

WEB_DIR = ROOT / "webui"
# 任务历史跟项目走：<项目>/work/webui/state.json；历史任务是只读快照，
# 所以服务重启后哪怕项目文件夹被挪走，也只是少了几条历史记录而已。
STATE_DIRNAME = "webui"
STATE_FILENAME = "state.json"
LOG_TAIL_LINES = 4000  # 内存里每任务保留的日志行数
HISTORY = 30  # 每个项目 state.json 里最多保留多少条任务记录

# 与 pipeline.progress_model / pipeline.run 保持一致
STAGES = _PM_STAGES
STAGE_LABELS = dict(_PM_STAGE_LABELS)
STAGE_WEIGHTS = dict(_PM_STAGE_WEIGHTS)


# ---------------------------------------------------------------------------
# 进度：委托 ProgressEngine（见 pipeline/progress_model.py）
# ---------------------------------------------------------------------------


class _EngineTracker:
    """兼容旧 tracker.feed / snapshot / finalize 调用面。"""

    def __init__(self, engine: ProgressEngine) -> None:
        self.engine = engine

    def feed(self, line: str, stage: str | None = None) -> None:
        self.engine.feed_line(line, stage)

    def snapshot(self) -> dict:
        return self.engine.snapshot()

    def finalize(self, *, success: bool) -> None:
        self.engine.finalize(success=success)

    def set_stage_pct(self, stage: str, pct: float) -> None:
        # 旧接口：阶段百分比由引擎按步骤汇总，忽略外部写入
        pass


class _NullTracker:
    """历史任务占位。"""

    def snapshot(self) -> dict:
        return {
            "seq": 0,
            "overall_pct": 0.0,
            "current_step_id": None,
            "stages": [],
            "steps": [],
            "items": [],
            "stage_pct": {},
            "current": None,
            "status": "pending",
            "task_id": "",
        }

    def feed(self, *_args) -> None:
        pass

    def finalize(self, *, success: bool) -> None:
        pass


class Job:
    """一条流水线子进程：日志流 + 结束回调。阶段粒度的状态由调用方给 detail。"""

    def __init__(
        self,
        task_id: str,
        *,
        name: str,
        novel: str,
        out: str,
        stages: list[str],
        only: str | None,
        cmd: list[str],
        log_path: Path,
        detail: str = "",
    ) -> None:
        self.id = task_id
        self.name = name
        self.novel = novel
        self.out = out
        self.stages = stages
        self.only = only
        self.cmd = cmd
        self.log_path = log_path
        self.detail = detail
        self.status = "running"
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.subscribers: list[queue.Queue] = []
        self.stages = list(stages) or list(STAGES)
        self.stage_status: dict[str, str] = {s: "pending" for s in self.stages}
        self.stage_status[self.stages[0]] = "running"
        self.pid: int | None = None
        # 细进度只走 /api/progress SSE，不在每步 bump 时刷全局 change（避免列表狂刷新）
        self.engine = ProgressEngine(out, stages=list(self.stages), task_id=task_id)
        self.tracker = _EngineTracker(self.engine)
        self.returncode: int | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None

    # -- 序列化 ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            prog = self.tracker.snapshot()
            # 阶段状态以引擎为准（含 pct）；任务级 status 仍用 self.stage_status 兜底
            eng_stages = prog.get("stages") or []
            if eng_stages:
                stages = [
                    {
                        "key": s["key"],
                        "label": s.get("label") or STAGE_LABELS.get(s["key"], s["key"]),
                        "status": s.get("status") or self.stage_status.get(s["key"], "pending"),
                        "pct": s.get("pct", 0),
                        "weight": s.get("weight"),
                    }
                    for s in eng_stages
                ]
                for s in stages:
                    self.stage_status[s["key"]] = s["status"]
            else:
                stages = [
                    {"key": s, "label": STAGE_LABELS.get(s, s), "status": self.stage_status.get(s, "pending")}
                    for s in self.stages
                ]
            return {
                "id": self.id,
                "name": self.name,
                "novel": self.novel,
                "out": self.out,
                "only": self.only,
                "detail": self.detail,
                "status": self.status,
                "stages": stages,
                "current_stage": prog.get("current_stage")
                or next((s["key"] for s in stages if s["status"] == "running"), None),
                "progress": prog,
                "returncode": self.returncode,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log_path": str(self.log_path),
                "media_base": self.out,
                "pid": self.pid,
                "source": getattr(self, "source", "web"),
            }

    # -- 日志 --------------------------------------------------------------
    def log_lines(self, after: int) -> tuple[list[str], int]:
        with self.lock:
            return list(self.log[after:]), len(self.log)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.log.append(line)
            if len(self.log) > LOG_TAIL_LINES:
                del self.log[: len(self.log) - LOG_TAIL_LINES]
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(line)
            except Exception:
                pass

    def subscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subscribers.append(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    # -- 阶段游标：解析子进程输出的 "== stage ==" 分隔行 ------------------
    def note_stage(self, stage: str, status: str) -> None:
        with self.lock:
            if stage in self.stage_status:
                self.stage_status[stage] = status

    def current_stage_safe(self) -> str | None:
        with self.lock:
            for s in self.stages:
                if self.stage_status.get(s) == "running":
                    return s
        return None

    # -- 执行 --------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from pipeline.progress import run_tee

        env = dict(__import__("os").environ)
        env["PYTHONUNBUFFERED"] = "1"
        sep = ";" if sys.platform == "win32" else ":"
        env["PYTHONPATH"] = str(ROOT) + sep + env.get("PYTHONPATH", "")
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.started_at = time.time()
            self.append_log(f"$ {shlex.join(self.cmd)}")

            last_persist = [0.0]

            def on_line(text: str) -> None:
                self.append_log(text)
                parse_stage_progress(text, self)
                # 引擎内部节流推送；这里只喂行
                self.engine.feed_line(text, self.current_stage_safe())
                now = time.time()
                if now - last_persist[0] >= 2.0:
                    last_persist[0] = now
                    _persist()

            def on_start(pid: int) -> None:
                self.pid = pid
                _persist()

            code = run_tee(self.cmd, log=self.log_path, env=env, cwd=ROOT, on_line=on_line, on_start=on_start)
            self.returncode = code
            self.status = "success" if code == 0 else "failed"
            self.engine.finalize(success=(code == 0))
            with self.lock:
                for s, st in self.stage_status.items():
                    if st == "running":
                        self.stage_status[s] = "done" if code == 0 else "failed"
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
            self.append_log(f"[webui] 任务异常：{exc}")
        finally:
            self.finished_at = time.time()
            self.append_log(f"[webui] 任务结束：{self.status}（退出码 {self.returncode}）")
            notify_change()
            _persist()  # 终态落盘：否则重启后完成的任务会被误标成「中断」

    def stop(self) -> None:
        """向进程组发 SIGTERM（run_tee 的子进程是独立 PTY 会话，用组信号才收得住）。"""
        import os
        import signal

        pid = self.pid
        if not pid:
            self.append_log("[webui] 任务还没拿到进程号，无法停止")
            return
        try:
            if sys.platform == "win32":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            self.append_log("[webui] 已请求停止…")
        except ProcessLookupError:
            pass
        except Exception as exc:
            self.append_log(f"[webui] 停止失败：{exc}")


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_CHANGE_QUEUES: list[queue.Queue] = []
# 当前网页打开的项目名下的任务 id：清空记录时只清这些，别的项目不受影响
CURRENT_PROJECT_JOB_IDS: set[str] = set()


def notify_change() -> None:
    with _LOCK:
        subs = list(_CHANGE_QUEUES)
    for q in subs:
        try:
            q.put_nowait("change")
        except Exception:
            pass


def state_path_for(out: str) -> Path:
    """某个项目的任务历史文件：<项目>/work/webui/state.json。"""
    return artifacts_dir(out) / STATE_DIRNAME / STATE_FILENAME


def job_out(job: Job) -> str:
    """任务的归属项目文件夹（新建任务在 __init__ 里也存了 out）。"""
    return getattr(job, "out", "") or ""


def project_of_job(job: Job) -> str:
    """任务归属的项目名（projects/<片名> 的 <片名>；旧布局任务取相对路径）。"""
    out = job_out(job).replace("\\", "/").strip("/")
    if out == f"{PROJECTS_DIRNAME}/{STATE_DIRNAME}":
        return ""  # 旧的全局历史不挂到任何项目下
    if out.startswith(f"{PROJECTS_DIRNAME}/"):
        return out[len(PROJECTS_DIRNAME) + 1:].split("/")[0]
    return out


def jobs_for_project(name: str) -> list[Job]:
    """某个项目名下的任务（内存里的；新任务启动后项目文件夹一定存在）。"""
    prefix = f"{PROJECTS_DIRNAME}/{name}"
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    if not name:
        return [j for j in jobs if not project_of_job(j)]
    return [
        j for j in jobs
        if project_of_job(j) == name
        # 旧布局 output/<片名> 的任务与同名新布局项目互认
        or (j.out == f"output/{name}")
    ]


def get_job(task_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(task_id)


def jobs_outside_projects() -> list[Job]:
    """不属于任何项目的任务（老的全局历史、旧布局无同名项目等）。"""
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    known = {item["name"] for item in _all_projects()["projects"]}
    known.add(STATE_DIRNAME)
    return [
        j for j in jobs
        if not project_of_job(j) and j.out != f"{PROJECTS_DIRNAME}/{STATE_DIRNAME}"
        and Path(j.out or "").name not in known
    ]


def clear_finished_tasks() -> dict:
    """清空当前打开项目的非运行中任务记录（产物文件不动）；运行中的保留。"""
    with _LOCK:
        running = [j for j in _JOBS.values() if j.status == "running"]
        keep_ids = {j.id for j in running}
        removed = [
            j for jid, j in _JOBS.items()
            if jid not in keep_ids and (jid in CURRENT_PROJECT_JOB_IDS or not CURRENT_PROJECT_JOB_IDS)
        ]
        for j in removed:
            del _JOBS[j.id]
    # 空桶也要重写落盘，否则下次选项目时旧 state.json 又把记录带回来
    outs = {j.out for j in removed}
    for out in outs:
        try:
            path = state_path_for(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"tasks": {}}, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass
    _persist()
    return {"ok": True, "kept": len(running)}


def register_job(job: Job) -> None:
    with _LOCK:
        _JOBS[job.id] = job
        CURRENT_PROJECT_JOB_IDS.add(job.id)
    _persist()


def load_project_tasks(name: str) -> list[Job]:
    """读某项目的 state.json → 恢复成只读历史任务，返回该项目的全部任务。

    服务重启后内存是空的，网页第一次选某个项目时按需恢复它的历史；
    运行中的旧记录：若 pid 仍活着则挂载继续盯进度，否则标成「中断」。
    另外会扫描 CLI produce / live.json，把外部任务挂进列表。
    """
    from pipeline.live import pid_alive

    path = state_path_for(f"{PROJECTS_DIRNAME}/{name}")
    restored: list[Job] = []
    dirty = False
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for task_id, snap in (data.get("tasks") or {}).items():
            if get_job(task_id):
                continue  # 内存里已经有了（例如刚重启后仍在跑的任务）
            if snap.get("status") == "running":
                if pid_alive(snap.get("pid")):
                    # 已有同 out / 同 pid 的挂载（例如启动时扫到的 CLI），不要再挂一份
                    dup = next(
                        (
                            j
                            for j in _JOBS.values()
                            if j.status == "running"
                            and (
                                (snap.get("pid") and j.pid == snap.get("pid"))
                                or (snap.get("out") and job_out(j) == snap.get("out"))
                            )
                        ),
                        None,
                    )
                    if dup:
                        continue
                    # 网页服务重启后子进程还在：挂载盯盘进度，不标中断
                    job = _AttachedJob.from_snapshot(snap)
                    with _LOCK:
                        _JOBS[task_id] = job
                        CURRENT_PROJECT_JOB_IDS.add(job.id)
                    job.start_watch()
                    restored.append(job)
                    continue
                snap["status"] = "interrupted"
                snap["error"] = "服务重启，任务中断；请重跑需要的阶段"
                for stage in snap.get("stages") or []:
                    if stage.get("status") == "running":
                        stage["status"] = "failed"
                dirty = True
            try:
                created = float(snap.get("created_at") or 0)
            except (TypeError, ValueError):
                created = 0.0
            job = _RestoredJob(snap, created)
            with _LOCK:
                _JOBS[task_id] = job
                CURRENT_PROJECT_JOB_IDS.add(job.id)
            restored.append(job)
    attach_external_jobs(name)
    if dirty:
        _persist()  # 把「中断」标记立刻写回 state.json
    return jobs_for_project(name)


def attach_external_jobs(name: str | None = None) -> list[Job]:
    """发现 CLI produce / live.json，挂成可刷新进度的任务。

    ``name`` 为空时扫描全部；指定项目名则只挂该项目。
    """
    from pipeline.live import (
        build_live_snapshot,
        discover_produce_processes,
        pid_alive,
        project_name_from_out,
        read_live,
        stable_cli_task_id,
    )

    attached: list[Job] = []
    procs = discover_produce_processes()
    by_out = {p["out"]: p for p in procs}

    # live.json：进程扫描不到时也能靠文件发现（刚写完、或 ps 权限问题）
    outs: set[str] = set(by_out)
    projects_root = ROOT / PROJECTS_DIRNAME
    if projects_root.is_dir():
        for child in projects_root.iterdir():
            if not child.is_dir():
                continue
            if name and child.name != name:
                continue
            rel = f"{PROJECTS_DIRNAME}/{child.name}"
            live = read_live(rel)
            if live and (live.get("status") == "running" or pid_alive(live.get("pid"))):
                outs.add(rel)

    for out in outs:
        proj = project_name_from_out(out)
        if name and proj != name:
            continue
        # 已有同 out 的运行中任务（网页自己起的）就别再挂一份
        existing = [
            j for j in jobs_for_project(proj)
            if job_out(j) == out and j.status == "running"
        ]
        if existing:
            continue
        task_id = stable_cli_task_id(out)
        if get_job(task_id) and get_job(task_id).status == "running":
            continue
        snap = build_live_snapshot(out, proc=by_out.get(out), live=read_live(out))
        if snap.get("status") != "running":
            continue
        snap["id"] = task_id
        job = _AttachedJob.from_snapshot(snap)
        with _LOCK:
            _JOBS[task_id] = job
            CURRENT_PROJECT_JOB_IDS.add(job.id)
        job.start_watch()
        attached.append(job)
        notify_change()
    return attached


def migrate_legacy_state() -> None:
    """旧的全局 projects/webui/state.json → 各项目 work/webui/state.json。

    按 snap["out"] 归属分发；一次迁移完把旧文件挪成 legacy.json 留底。
    """
    legacy_path = ROOT / PROJECTS_DIRNAME / STATE_DIRNAME / STATE_FILENAME
    if not legacy_path.exists():
        return
    try:
        data = json.loads(legacy_path.read_text(encoding="utf-8"))
    except Exception:
        return
    moved = 0
    for _task_id, snap in (data.get("tasks") or {}).items():
        out = str(snap.get("out") or "").strip()
        if not out or out == f"{PROJECTS_DIRNAME}/{STATE_DIRNAME}":
            continue
        path = state_path_for(out)
        try:
            bucket = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tasks": {}}
            bucket.setdefault("tasks", {})
            bucket["tasks"][snap.get("id") or uuid.uuid4().hex[:12]] = snap
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(bucket, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
            moved += 1
        except Exception:
            continue
    if moved:
        legacy_path.replace(legacy_path.with_name("legacy.json"))
    elif legacy_path.exists():
        legacy_path.replace(legacy_path.with_name("legacy.json"))


def _persist() -> None:
    """把任务历史写回各自项目的 <项目>/work/webui/state.json。"""
    try:
        buckets: dict[str, dict] = {}
        with _LOCK:
            jobs = list(_JOBS.values())
        for job in jobs:
            out = job_out(job)
            if not out:
                continue
            buckets.setdefault(out, {})[job.id] = job.snapshot()
        for out, tasks in buckets.items():
            path = state_path_for(out)
            if len(tasks) > HISTORY:
                keep = sorted(tasks.items(), key=lambda kv: kv[1].get("created_at", 0), reverse=True)[:HISTORY]
                tasks = dict(keep)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass


class _RestoredJob(Job):
    """历史任务：只读快照，不能再 start。"""

    def __init__(self, snap: dict, created_at: float) -> None:
        stages_raw = snap.get("stages") or []
        if stages_raw and isinstance(stages_raw[0], str):
            stage_keys = list(stages_raw)
            stage_status = dict(snap.get("stage_status") or {s: "pending" for s in stage_keys})
        else:
            stage_keys = [s.get("key") for s in stages_raw]
            stage_status = {s.get("key"): s.get("status", "pending") for s in stages_raw}
        self.__dict__.update(
            {
                "id": snap.get("id") or uuid.uuid4().hex[:12],
                "name": snap.get("name") or "",
                "novel": snap.get("novel") or "",
                "out": snap.get("out") or "",
                "only": snap.get("only"),
                "detail": snap.get("detail") or "",
                "status": snap.get("status") or "interrupted",
                "stages": stage_keys,
                "stage_status": stage_status,
                "returncode": snap.get("returncode"),
                "error": snap.get("error"),
                "created_at": created_at,
                "started_at": snap.get("started_at"),
                "finished_at": snap.get("finished_at"),
                "log_path": Path(snap.get("log_path") or ROOT / "output"),
                "lock": threading.Lock(),
                "log": [],
                "subscribers": [],
                "cmd": [],
                "pid": snap.get("pid"),
                "pgid": snap.get("pgid"),
                "source": snap.get("source") or "web",
                "tracker": _NullTracker(),
            }
        )
        # 历史任务若 snap 里带了 progress，塞进可快照的 tracker
        self.engine = None
        prog = snap.get("progress")
        if isinstance(prog, dict) and (
            prog.get("items") or prog.get("stage_pct") or prog.get("steps") or prog.get("overall_pct") is not None
        ):
            self.tracker = _FrozenTracker(prog)

    def start(self) -> None:  # pragma: no cover
        raise RuntimeError("历史任务不能重新启动，请新建任务")


class _FrozenTracker:
    """历史任务只读 progress 快照（完整 snapshot 形状）。"""

    def __init__(self, snap: dict) -> None:
        base = _NullTracker().snapshot()
        base.update(snap or {})
        self._snap = base

    def snapshot(self) -> dict:
        return dict(self._snap)

    def feed(self, *_args) -> None:
        pass

    def finalize(self, *, success: bool) -> None:
        pass


class _AttachedJob(Job):
    """挂载外部 produce（CLI 或网页重启后仍活着的子进程）：盯 pid + Engine.observe_disk。"""

    def __init__(self, snap: dict) -> None:
        _RestoredJob.__init__(self, snap, float(snap.get("created_at") or time.time()))
        self.status = "running"
        self.source = snap.get("source") or "cli"
        self.detail = snap.get("detail") or "CLI produce（网页自动挂载）"
        self._watch_started = False
        stages = [s["key"] if isinstance(s, dict) else s for s in (snap.get("stages") or self.stages)]
        self.stages = [s for s in stages if s in STAGE_LABELS] or list(STAGES)
        self.engine = ProgressEngine(self.out, stages=list(self.stages), task_id=self.id)
        self.tracker = _EngineTracker(self.engine)
        self.refresh_from_disk()

    @classmethod
    def from_snapshot(cls, snap: dict) -> "_AttachedJob":
        return cls(snap)

    def start(self) -> None:  # pragma: no cover
        raise RuntimeError("挂载任务不能 start，只监视已有进程")

    def start_watch(self) -> None:
        if self._watch_started:
            return
        self._watch_started = True
        threading.Thread(target=self._watch, daemon=True).start()

    def refresh_from_disk(self) -> None:
        from pipeline.live import read_live

        live = read_live(self.out) or {}
        cur = live.get("current_stage")
        if cur and cur in self.stage_status:
            self.engine.set_stage(cur, "running")
        self.engine.observe_disk()
        prog = self.engine.snapshot()
        with self.lock:
            for s in prog.get("stages") or []:
                self.stage_status[s["key"]] = s.get("status") or "pending"
            if live.get("pid"):
                self.pid = live["pid"]

    def _watch(self) -> None:
        from pipeline.live import pid_alive

        while self.status == "running":
            alive = pid_alive(self.pid)
            if not alive:
                final = resolve(self.out) / "final.mp4"
                ok = final.exists()
                self.status = "success" if ok else "failed"
                self.returncode = 0 if ok else 1
                self.error = None if ok else "外部 produce 已结束（未见成片）"
                self.finished_at = time.time()
                self.engine.finalize(success=ok)
                with self.lock:
                    for s, st in list(self.stage_status.items()):
                        if st == "running":
                            self.stage_status[s] = "done" if ok else "failed"
                self.append_log(f"[webui] 外部任务结束：{self.status}")
                notify_change()
                _persist()
                return
            try:
                self.refresh_from_disk()
            except Exception as exc:
                self.append_log(f"[webui] 刷新外部进度失败：{exc}")
            time.sleep(0.5)  # 挂载任务 0.5s 扫盘，配合引擎节流

    def stop(self) -> None:
        import os
        import signal

        pid = self.pid
        pgid = getattr(self, "pgid", None) or pid
        if not pid:
            self.append_log("[webui] 外部任务没有进程号，无法停止")
            return
        try:
            if sys.platform == "win32":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
            else:
                try:
                    os.killpg(int(pgid), signal.SIGTERM)
                except Exception:
                    os.kill(int(pid), signal.SIGTERM)
            self.append_log("[webui] 已请求停止外部任务…")
        except ProcessLookupError:
            pass
        except Exception as exc:
            self.append_log(f"[webui] 停止失败：{exc}")


# ---------------------------------------------------------------------------
# 任务命令构造
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    slug = re.sub(r"[\s/\\]+", "-", (title or "").strip()).strip("-")
    return slug or "untitled"


def _single_stage_cmd(
    stage: str,
    *,
    out: Path,
    novel: str | None = None,
    storyboard: str | None = None,
    only: str | None = None,
    force: bool = False,
) -> list[str]:
    """单阶段命令；CLI 子命令一次只能一个，多阶段靠 run / bash && 串。"""
    cmd = [sys.executable, "-m", "pipeline", stage, "--out", str(out)]
    if stage == "screenplay":
        # screenplay 只校验：外部分镜 + 小说路径用于对白覆盖检查
        if novel:
            cmd += ["--novel", str(resolve(novel))]
        if storyboard:
            cmd += ["--storyboard", str(resolve(storyboard))]
    if only and stage in {"images", "audio", "videos"}:
        cmd += ["--only", str(only)]
    if force:
        cmd.append("--force")
    return cmd


def build_pipeline_cmd(
    *,
    stages: list[str],
    novel: str | None,
    out: str,
    storyboard: str | None,
    only: str | None,
    force: bool,
) -> tuple[list[str], Path]:
    """拼出 CLI 能接受的串跑命令。

    CLI 的子命令一次只能一个（argparse subparsers），不能写成
    ``pipeline screenplay images audio …``。所以：

    - 勾选的阶段是 STAGES 的连续段（含全选 / 单阶段）→ 等价
      ``run --from <首个> --until <末个>``：由 run 负责安装分镜、串跑、打
      ``== stage ==`` 分隔行（webui 的阶段游标靠它推进）；
    - 中间跳了阶段 → 用 ``bash -c`` 把单阶段命令 ``&&`` 起来，
      语义与逐个调用 CLI 完全一致。

    返回 ``(argv, 日志路径)``；argv 可能是 ``["/bin/bash", "-c", "…"]``。
    """
    out_dir = resolve(out)
    selected = [s for s in STAGES if s in set(stages)]
    if not selected:
        selected = list(STAGES)
    start_idx = STAGES.index(selected[0])
    contiguous = selected == list(STAGES[start_idx : STAGES.index(selected[-1]) + 1])
    if contiguous:
        cmd = [sys.executable, "-m", "pipeline", "run", "--from", selected[0]]
        if selected[-1] != STAGES[-1]:
            cmd += ["--until", selected[-1]]
        if novel:
            cmd += ["--novel", str(resolve(novel))]
        cmd += ["--out", str(out_dir)]
        if storyboard:
            cmd += ["--storyboard", str(resolve(storyboard))]
        if only:
            cmd += ["--only", str(only)]
        if force:
            cmd.append("--force")
    else:
        parts = []
        for stage in selected:
            parts.append(
                shlex.join(
                    _single_stage_cmd(
                        stage,
                        out=out_dir,
                        novel=novel,
                        storyboard=storyboard,
                        only=only,
                        force=force,
                    )
                )
            )
        cmd = ["/bin/bash", "-c", " && ".join(parts)]
    log_path = artifacts_sub(out_dir, "logs", f"webui-{time.strftime('%Y%m%d-%H%M%S')}.log")
    return cmd, log_path


def parse_stage_progress(line: str, job: Job) -> None:
    """子进程打印 ``== stage ==`` 分隔行时推进阶段游标。"""
    m = re.match(r"^==\s*(\w+)\s*==$", line.strip())
    if not m:
        return
    stage = m.group(1)
    if stage not in job.stage_status:
        return
    now_idx = job.stages.index(stage)
    with job.lock:
        for s, st in job.stage_status.items():
            idx = job.stages.index(s)
            if idx < now_idx and st in ("running", "pending"):
                job.stage_status[s] = "done"
        job.stage_status[stage] = "running"
    engine = getattr(job, "engine", None)
    if engine is not None:
        engine.set_stage(stage, "running")
    notify_change()


# ---------------------------------------------------------------------------
# 项目文件
# ---------------------------------------------------------------------------


def list_input_files() -> dict:
    """旧布局的 input/ 文件（分镜编辑器的备选清单；新项目不再写这里）。"""
    novels: list[dict] = []
    storyboards: list[dict] = []
    input_dir = ROOT / "input"
    if input_dir.is_dir():
        for path in sorted(input_dir.glob("*.txt")):
            novels.append({"name": path.stem, "path": str(path.relative_to(ROOT)).replace("\\", "/")})
        for path in sorted(input_dir.glob("*.json")):
            storyboards.append({"name": path.stem, "path": str(path.relative_to(ROOT)).replace("\\", "/")})
    return {"novels": novels, "storyboards": storyboards}


def _project_file_options() -> dict:
    """小说 / 分镜备选清单：各项目里的 novel.txt 与 storyboard.json（含旧 input/）。"""
    from pipeline.projects import list_projects as scan_projects

    novels = list_input_files()["novels"]
    storyboards = list_input_files()["storyboards"]
    for item in scan_projects():
        if item.get("novel"):
            novels.append({"name": f"{item['name']}/novel", "path": item["novel"]})
        if item.get("has_storyboard"):
            storyboards.append(
                {"name": f"{item['name']}/storyboard", "path": f"{item['path']}/storyboard.json"}
            )
    return {"novels": novels, "storyboards": storyboards}


def _all_projects() -> dict:
    """项目清单 + 每个项目的任务统计，供项目选择页一次拉全。"""
    from pipeline.live import discover_produce_processes, project_name_from_out
    from pipeline.projects import list_projects as scan_projects

    # 先挂一次外部任务，统计才准
    try:
        attach_external_jobs(None)
    except Exception:
        pass

    projects = scan_projects()
    mem_running: dict[str, int] = {}
    with _LOCK:
        for j in _JOBS.values():
            if j.status != "running":
                continue
            proj = project_of_job(j)
            if proj:
                mem_running[proj] = mem_running.get(proj, 0) + 1
    # 进程扫描兜底（内存还没挂上时项目卡也能亮「运行中」）
    for proc in discover_produce_processes():
        proj = project_name_from_out(proc["out"])
        if proj and proj not in mem_running:
            mem_running[proj] = 1

    for item in projects:
        path = state_path_for(item["path"])
        done = running = failed = 0
        last = None
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            for snap in (data.get("tasks") or {}).values():
                status = snap.get("status")
                if status == "running":
                    running += 1
                elif status == "failed" or status == "interrupted":
                    failed += 1
                else:
                    done += 1
                created = snap.get("created_at") or 0
                if created > (last or 0):
                    last = created
        # 磁盘上可能还没有 running 记录（CLI 没写 state.json），用内存/进程补
        live_run = mem_running.get(item["name"], 0)
        if live_run > running:
            running = live_run
        item["tasks"] = {"done": done, "running": running, "failed": failed, "total": done + running + failed}
        item["last_activity"] = last
    return {"projects": projects}


def create_project_api(data: dict) -> dict:
    """网页上新建项目文件夹。"""
    from pipeline.projects import create_project

    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError("项目名不能为空")
    title = str(data.get("title") or "").strip()
    directory = create_project(name, title=title)
    return {
        "name": directory.name,
        "path": f"{PROJECTS_DIRNAME}/{directory.name}",
        "novel": f"{PROJECTS_DIRNAME}/{directory.name}/novel.txt",
        "storyboard": f"{PROJECTS_DIRNAME}/{directory.name}/storyboard.json",
    }


_SAFE_PATH = re.compile(r"^[\w\-./\u4e00-\u9fff]+$")


def safe_path(rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or not _SAFE_PATH.match(rel):
        raise ValueError(f"不接受的路径：{rel!r}")
    path = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"路径越界：{rel}")
    return path


def read_text(rel: str) -> str:
    return safe_path(rel).read_text(encoding="utf-8")


def write_text(rel: str, content: str) -> None:
    path = safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_storyboard(rel: str) -> dict:
    return json.loads(safe_path(rel).read_text(encoding="utf-8"))


def write_storyboard(rel: str, data: dict) -> None:
    path = safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_storyboard(path_str: str, *, strict: bool) -> dict:
    """借用 pipeline.screenplay 的逻辑做秒级校验（不加载模型）。"""
    src = safe_path(path_str)
    from pipeline.config import Settings
    from pipeline.projects import strip_storyboard_suffix
    from pipeline.screenplay import (
        assert_dialogue_coverage,
        dialogue_char_limit,
        normalize_storyboard,
        report_dialogue_budget,
    )
    from pipeline.util import load_json, snap_frames

    settings = Settings.load()
    raw = load_json(src)
    novel_path: Path | None = None
    linked = str(raw.get("novel") or "").strip()
    if linked:
        cand = safe_path(linked)
        if cand.exists():
            novel_path = cand
    if novel_path is None:
        # 优先同一项目文件夹里的小说；兼容旧布局 input/<片名>_novel.txt
        for cand in novel_candidates(src):
            novel_path = cand
            break
    if novel_path is None:
        stem = strip_storyboard_suffix(src.stem)
        for cand_name in (f"{PROJECTS_DIRNAME}/{stem}/novel.txt", f"input/{stem}_novel.txt", f"input/{stem}.txt"):
            cand = ROOT / cand_name
            if cand.exists():
                novel_path = cand
                break
    fps = max(1, settings.video_fps)
    max_seconds = snap_frames(settings.video_frames_max) / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    normalized = normalize_storyboard(
        raw,
        novel_path=novel_path,
        max_shots=settings.max_shots,
        dialogue_limit=None,
        max_seconds=max_seconds,
    )
    shots = normalized["shots"]
    if novel_path is not None:
        assert_dialogue_coverage(novel_path.read_text(encoding="utf-8"), shots)
    import contextlib
    import io

    buf = io.StringIO()
    strict_error: str | None = None
    try:
        with contextlib.redirect_stdout(buf):
            report_dialogue_budget(shots, limit, max_seconds, strict=strict)
    except SystemExit as exc:
        if not strict:
            raise
        strict_error = str(exc)
    return {
        "ok": strict_error is None,
        "strict_error": strict_error,
        "shots": len(shots),
        "characters": len(normalized.get("characters") or []),
        "dialogue_lines": normalized.get("dialogue_lines", 0),
        "limit_chars": limit,
        "max_seconds": round(max_seconds, 1),
        "novel": str(novel_path) if novel_path else "",
        "report": buf.getvalue(),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def budget_info() -> dict:
    """单镜上限（复用 budget 命令逻辑，秒级、不加载模型）。"""
    from pipeline.config import Settings

    settings = Settings.load()
    detail = settings.budget_detail
    if not detail:
        from pipeline.budget import resolve_frame_cap

        _, detail = resolve_frame_cap(
            height=settings.video_height,
            width=settings.video_width,
            fps=settings.video_fps,
            min_frames=settings.video_frames_min,
            ceiling=settings.video_frames_ceiling,
            table=settings.token_budget_table(),
        )
    return {
        "frames": settings.video_frames_max,
        "seconds": detail.get("seconds"),
        "dialogue_chars": detail.get("dialogue_chars"),
        "total_gib": detail.get("total_gib"),
        "fps": settings.video_fps,
        "resolution": f"{settings.video_width}x{settings.video_height}",
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "NovelVideoWeb/0.2"

    def log_message(self, fmt: str, *args) -> None:  # 安静
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _json_error(self, exc: Exception, status: int = 500) -> None:
        self._json({"error": f"{type(exc).__name__}: {exc}"}, status)

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw.strip() else {}

    def _static_file(self, name: str, ctype: str) -> None:
        path = WEB_DIR / name
        if not path.exists():
            self._json({"error": f"missing {name}"}, 404)
            return
        self._send(path.read_bytes(), ctype)

    _MEDIA_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".json": "application/json",
        ".log": "text/plain; charset=utf-8",
    }

    def _serve_media(self, rel: str) -> None:
        """只读地暴露项目目录下的产物（projects/ 与旧 input/、output/）。"""
        try:
            path = safe_path(rel)
        except (ValueError, FileNotFoundError):
            self._json({"error": "bad path"}, 400)
            return
        if not path.is_file():
            self._json({"error": "not found"}, 404)
            return
        ctype = self._MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        status = 200
        start, end = 0, size - 1
        if range_header and range_header.startswith("bytes="):
            try:
                first, last = range_header[6:].split("-", 1)
                start = int(first) if first else 0
                end = int(last) if last else size - 1
                end = min(end, size - 1)
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- GET ---------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        path, _, _query = self.path.partition("?")
        if path.startswith("/media/"):
            self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        try:
            if path in ("", "/"):
                self._static_file("index.html", "text/html; charset=utf-8")
            elif path == "/app.js":
                self._static_file("app.js", "application/javascript; charset=utf-8")
            elif path == "/style.css":
                self._static_file("style.css", "text/css; charset=utf-8")
            elif path == "/api/media":
                self._json(self.api_media(query))
            elif path == "/api/state":
                self._json(self.api_state(query))
            elif path == "/api/budget":
                self._json(budget_info())
            elif path == "/api/events":
                self._stream_events()
            elif path.startswith("/api/log/"):
                self._api_log(path.rsplit("/", 1)[-1], query)
            elif path.startswith("/api/log-stream/"):
                self._stream_job_log(path.rsplit("/", 1)[-1])
            elif path.startswith("/api/progress/"):
                self._stream_progress(path.rsplit("/", 1)[-1])
            elif path.startswith("/media/"):
                self._serve_media(path[len("/media/"):])
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._json_error(exc)

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path, _, _query = self.path.partition("?")
        try:
            data = self._body_json()
            if path == "/api/task":
                self._json(self.create_task(data))
            elif path == "/api/task/stop":
                self._json(self.stop_task(data))
            elif path == "/api/tasks/clear":
                self._json(clear_finished_tasks())
            elif path == "/api/project/select":
                self._json(self.select_project(data))
            elif path == "/api/projects/new":
                self._json(create_project_api(data))
            elif path == "/api/file/read":
                rel = str(data.get("path") or "")
                self._json({"path": rel, "content": read_text(rel)})
            elif path == "/api/file/write":
                rel = str(data.get("path") or "")
                write_text(rel, str(data.get("content") or ""))
                self._json({"path": rel, "ok": True})
            elif path == "/api/storyboard/read":
                rel = str(data.get("path") or "")
                self._json({"path": rel, "storyboard": read_storyboard(rel)})
            elif path == "/api/storyboard/write":
                rel = str(data.get("path") or "")
                write_storyboard(rel, data.get("storyboard") or {})
                self._json({"path": rel, "ok": True})
            elif path == "/api/storyboard/validate":
                self._json(validate_storyboard(str(data.get("path") or ""), strict=bool(data.get("strict"))))
            elif path == "/api/shots/rerun":
                self._json(self.rerun_shots(data))
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError as exc:
            self._json_error(exc, 404)
        except json.JSONDecodeError as exc:
            self._json_error(exc, 400)
        except Exception as exc:
            self._json_error(exc)

    # -- API ---------------------------------------------------------------
    def api_state(self, query: str) -> dict:
        """选了项目 → 返回该项目的一切（任务、项目卡片、小说/分镜备选）。"""
        from urllib.parse import parse_qs, unquote

        params = parse_qs(query)
        name = unquote(params.get("project", [""])[0]).strip()
        if name:
            tasks = load_project_tasks(name)
        else:
            # 项目选择页也要发现外部 produce，卡片才能显示「运行中」
            try:
                attach_external_jobs(None)
            except Exception:
                pass
            tasks = jobs_outside_projects()
        file_options = _project_file_options()
        return {
            "project": name,
            "projects": _all_projects()["projects"],
            "novels": file_options["novels"],
            "storyboards": file_options["storyboards"],
            "tasks": [j.snapshot() for j in tasks],
            "server_time": time.time(),
        }

    def select_project(self, data: dict) -> dict:
        """记住网页当前打开的项目：清空记录只清它名下的任务。"""
        name = str(data.get("name") or "").strip()
        with _LOCK:
            keep = {j.id for j in _JOBS.values() if j.status == "running"}
            CURRENT_PROJECT_JOB_IDS.clear()
            CURRENT_PROJECT_JOB_IDS.update(keep)
        if name:
            load_project_tasks(name)
        return {"ok": True, "project": name}

    def create_task(self, data: dict) -> dict:
        title = str(data.get("title") or "").strip()
        name = str(data.get("name") or "").strip() or _slugify(title)
        # 新任务一律落在项目文件夹 projects/<片名>/
        out = str(data.get("out") or "").strip() or f"{PROJECTS_DIRNAME}/{_slugify(name)}"
        novel = str(data.get("novel") or "").strip()
        storyboard = str(data.get("storyboard") or "").strip()
        only = str(data.get("only") or "").strip() or None
        force = bool(data.get("force"))
        stages = [s for s in (data.get("stages") or []) if s in STAGES]
        if not stages:
            stages = list(STAGES)
        # 前端勾选顺序不可靠；统一按流水线顺序传给 CLI（run --from 依赖它）
        stages = [s for s in STAGES if s in set(stages)]

        # 粘贴的小说片段先落盘到项目文件夹
        novel_text = str(data.get("novel_text") or "").strip()
        if novel_text:
            novel = novel or f"{project_dir(name).relative_to(ROOT)}/novel.txt"
            write_text(novel, novel_text.rstrip() + "\n")
        # 粘贴的分镜 JSON 先落盘到项目文件夹
        sb_data = data.get("storyboard_data")
        if isinstance(sb_data, dict) and sb_data.get("shots"):
            storyboard = storyboard or f"{project_dir(name).relative_to(ROOT)}/storyboard.json"
            write_storyboard(storyboard, sb_data)

        # --force：启动前先清产物（bash 串跑路径不会进 cmd_run 的清理逻辑）
        if force:
            from pipeline.projects import clear_artifacts

            removed = clear_artifacts(out, stages=stages, only=only)
            if removed:
                print(f"[webui] --force 已清除 {len(removed)} 项产物：{out}", flush=True)

        cmd, log_path = build_pipeline_cmd(
            stages=stages, novel=novel, out=out, storyboard=storyboard, only=only, force=force
        )
        job = Job(
            uuid.uuid4().hex[:12],
            name=name or out,
            novel=novel,
            out=out,
            stages=stages,
            only=only,
            cmd=cmd,
            log_path=log_path,
            detail=title,
        )
        register_job(job)
        job.start()
        notify_change()
        return job.snapshot()

    def stop_task(self, data: dict) -> dict:
        job = get_job(str(data.get("id") or ""))
        if not job:
            raise FileNotFoundError("task not found")
        job.stop()
        return {"ok": True}

    def rerun_shots(self, data: dict) -> dict:
        """对指定镜头重跑所选阶段（前端传什么阶段就跑什么）。"""
        shots_raw = data.get("shots") or []
        shots = ",".join(sorted({str(s).strip() for s in shots_raw if str(s).strip()}))
        if not shots:
            raise ValueError("没给出镜头 id")
        out = str(data.get("out") or "").strip()
        if not out:
            raise ValueError("没给输出目录")
        storyboard = str(data.get("storyboard") or "").strip()
        stages = [s for s in (data.get("stages") or []) if s in STAGES]
        if not stages:
            stages = ["images", "videos"]  # 兼容旧前端：缺省图文都重做
        novel = str(data.get("novel") or "").strip()
        from pipeline.projects import clear_artifacts

        clear_artifacts(out, stages=stages, only=shots)
        cmd, log_path = build_pipeline_cmd(
            stages=stages, novel=novel or None, out=out, storyboard=storyboard, only=shots, force=True
        )
        job = Job(
            uuid.uuid4().hex[:12],
            name=f"重跑镜头 {shots}",
            novel=novel,
            out=out,
            stages=stages,
            only=shots,
            cmd=cmd,
            log_path=log_path,
            detail="部分镜头重跑",
        )
        register_job(job)
        job.start()
        notify_change()
        return job.snapshot()

    def api_media(self, query: str) -> dict:
        """列出某输出目录下已生成的静帧 / 视频 / 配音文件（相对路径 + 修改时间）。"""
        from urllib.parse import unquote

        params = {}
        for pair in query.split("&"):
            if "=" in pair:
                key, _, value = pair.partition("=")
                params[key] = unquote(value)
        out_rel = params.get("out", "")
        if not out_rel:
            raise ValueError("缺少 out 参数")
        base = safe_path(out_rel)

        def _scan(rel_dir: str, exts: tuple[str, ...]) -> list[dict]:
            d = artifacts_sub(base, rel_dir)
            if not d.is_dir():
                return []
            found = []
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    rel = f"{out_rel.rstrip('/')}/work/{rel_dir}/{f.name}".replace("\\", "/")
                    found.append(
                        {
                            "id": f.stem,
                            "path": rel,
                            "url": "/media/" + rel,
                            "mtime": int(f.stat().st_mtime),
                            "size": f.stat().st_size,
                        }
                    )
            return found

        final_path = base / "final.mp4"
        final = None
        if final_path.is_file():
            rel = f"{out_rel.rstrip('/')}/final.mp4".replace("\\", "/")
            st = final_path.stat()
            final = {
                "path": rel,
                "url": "/media/" + rel,
                "mtime": int(st.st_mtime),
                "size": st.st_size,
            }
        return {
            "frames": _scan("frames", (".png", ".jpg", ".jpeg", ".webp")),
            "clips": _scan("clips", (".mp4",)),
            "voices": _scan("audio/voice", (".wav",)),
            "final": final,
        }

    def _api_log(self, task_id: str, query: str) -> None:
        job = get_job(task_id)
        if not job:
            raise FileNotFoundError("task not found")
        after = 0
        for pair in query.split("&"):
            if pair.startswith("after="):
                after = int(pair[6:] or 0)
        lines, total = job.log_lines(after)
        # 附带最新进度：前端 1.2s 轮询就能推整体条，不必等 4s 全量 refresh
        snap = job.snapshot()
        self._json(
            {
                "lines": lines,
                "total": total,
                "status": job.status,
                "progress": snap.get("progress"),
                "stages": snap.get("stages"),
                "current_stage": snap.get("current_stage"),
            }
        )

    # -- SSE ---------------------------------------------------------------
    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        # SSE 长连接：不要 Connection: close，否则浏览器会反复重连，顶栏「已连接」狂闪
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse_event(self, payload: str) -> None:
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_events(self) -> None:
        """任务列表级变更（新建/结束）；细进度走 /api/progress/<id>。"""
        self._sse_start()
        q: queue.Queue = queue.Queue()
        with _LOCK:
            _CHANGE_QUEUES.append(q)
        try:
            self._sse_event(json.dumps({"event": "hello"}))
            while True:
                try:
                    q.get(timeout=15.0)
                    self._sse_event(json.dumps({"event": "change"}))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _LOCK:
                if q in _CHANGE_QUEUES:
                    _CHANGE_QUEUES.remove(q)

    def _sse_progress_event(self, snap: dict) -> None:
        seq = int(snap.get("seq") or 0)
        payload = json.dumps({"event": "progress", "progress": snap}, ensure_ascii=False)
        self.wfile.write(f"id: {seq}\nevent: progress\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_progress(self, task_id: str) -> None:
        """按任务推送进度快照；支持 Last-Event-ID 跳过已见 seq。"""
        job = get_job(task_id)
        if not job:
            raise FileNotFoundError("task not found")
        last_id = 0
        raw = self.headers.get("Last-Event-ID") or self.headers.get("Last-Event-Id") or ""
        try:
            last_id = int(str(raw).strip() or "0")
        except ValueError:
            last_id = 0
        self._sse_start()
        engine = getattr(job, "engine", None)
        q: queue.Queue | None = queue.Queue() if engine is not None else None
        try:
            # 连接后立刻推当前快照
            snap = job.tracker.snapshot() if hasattr(job, "tracker") else {}
            self._sse_progress_event(snap)
            last_id = max(last_id, int(snap.get("seq") or 0))
            if engine is None:
                # 历史任务：推一次终态后结束
                self.wfile.write(b"event: end\ndata: {}\n\n")
                self.wfile.flush()
                return
            assert q is not None
            engine.subscribe(q)
            while job.status == "running":
                try:
                    snap = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if int(snap.get("seq") or 0) <= last_id:
                    continue
                last_id = int(snap.get("seq") or 0)
                self._sse_progress_event(snap)
            # 终态再推一次
            final = job.tracker.snapshot()
            if int(final.get("seq") or 0) > last_id:
                self._sse_progress_event(final)
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if engine is not None and q is not None:
                engine.unsubscribe(q)

    def _stream_job_log(self, task_id: str) -> None:
        job = get_job(task_id)
        if not job:
            raise FileNotFoundError("task not found")
        self._sse_start()
        q: queue.Queue = queue.Queue()
        lines, _ = job.log_lines(0)
        for line in lines:
            self._sse_event(json.dumps({"line": line}, ensure_ascii=False))
        if job.status != "running":
            self._sse_event(json.dumps({"event": "end"}))
            return
        job.subscribe(q)
        try:
            while True:
                try:
                    line = q.get(timeout=15.0)
                except queue.Empty:
                    if job.status != "running":
                        break
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._sse_event(json.dumps({"line": line}, ensure_ascii=False))
            self._sse_event(json.dumps({"event": "end"}))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            job.unsubscribe(q)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def serve(host: str, port: int) -> None:
    migrate_legacy_state()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    if not (WEB_DIR / "index.html").exists():
        raise SystemExit(f"缺少网页文件：{WEB_DIR / 'index.html'}")
    # 启动时立刻挂上已在跑的 CLI produce，避免用户打开网页看不到进度
    try:
        attached = attach_external_jobs(None)
        if attached:
            print(f"已挂载外部任务 {len(attached)} 个：{', '.join(j.out for j in attached)}", flush=True)
    except Exception as exc:
        print(f"挂载外部任务失败（可忽略）：{exc}", flush=True)
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"小说成片控制台：http://{host}:{port}  （Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def cmd_web(args: argparse.Namespace) -> None:
    serve(args.host, args.port)
