"""本地网页控制台：填写小说片段、起流水线任务、看进度、重跑单个阶段。

零新增依赖，只用 Python 标准库：

    ./scripts/run.sh                          # 默认起网页；等同 ./scripts/run.sh web
    ./scripts/run.sh --port 8790              # 浏览器打开 http://127.0.0.1:8790

职责边界：小说可经本机 MLX 聊天模型写成分镜（storyboard 阶段）；也可手写/
Agent 维护项目文件夹里的 `storyboard.json`。本界面负责填小说、改分镜、起任务、
看实时进度、对失败步骤单独重跑。

每个项目一个独立文件夹，默认在 ``~/AI短视频项目/<片名>/``（小说、分镜、产物都在里面）；
任务历史也属于项目：写在 `<项目>/work/webui/state.json`，重启服务不丢，
打开网页第一步是选项目，之后看到的任务、镜头产物都是这个项目的。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from pipeline.mirrors import apply_mirrors

apply_mirrors()

from pipeline.media_bins import server_pid_path, support_dir
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
    display_path,
    migrate_repo_projects,
    novel_candidates,
    project_dir,
    project_path_str,
    projects_dir,
)
from pipeline.util import ROOT, cite_path, resolve, under_allowed_root

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
    """一条流水线子进程：日志流 + 结束回调。阶段粒度的状态由调用方给 detail。

    status=queued 时只入列表、不拉起进程；点「启动任务」后再 start。
    """

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
        status: str = "running",
        queue_clear: list[dict] | None = None,
        characters: str | None = None,
        image_parts: str | None = None,
    ) -> None:
        self.id = task_id
        self.name = name
        self.novel = novel
        self.out = out
        self.only = only
        self.characters = characters
        self.image_parts = str(image_parts or "both").strip().lower() or "both"
        self.cmd = list(cmd or [])
        self.log_path = log_path
        self.detail = detail
        self.status = status if status in {"queued", "running"} else "running"
        self.queue_clear = list(queue_clear or [])  # 真正 start 时再清的产物节点
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.subscribers: list[queue.Queue] = []
        # 首帧+尾帧会排两次 images 命令；进度游标/步骤表只保留一个 images
        uniq_stages: list[str] = []
        seen_stages: set[str] = set()
        for s in list(stages or []) or list(STAGES):
            if s and s not in seen_stages:
                uniq_stages.append(s)
                seen_stages.add(s)
        self.stages = uniq_stages or list(STAGES)
        self.stage_status: dict[str, str] = {s: "pending" for s in self.stages}
        if self.status == "running" and self.stages:
            self.stage_status[self.stages[0]] = "running"
        self.pid: int | None = None
        # 细进度只走 /api/progress SSE，不在每步 bump 时刷全局 change（避免列表狂刷新）
        self.engine = ProgressEngine(
            out,
            stages=list(self.stages),
            task_id=task_id,
            only=only,
            characters=characters,
            image_parts=self.image_parts,
        )
        self.tracker = _EngineTracker(self.engine)
        self.returncode: int | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.stop_requested = False

    @classmethod
    def from_queued_snapshot(cls, snap: dict) -> "Job":
        """从 state.json 恢复可再 start 的待跑任务。"""
        stages_raw = snap.get("stages") or []
        if stages_raw and isinstance(stages_raw[0], str):
            stage_keys = list(stages_raw)
        else:
            stage_keys = [s.get("key") for s in stages_raw if isinstance(s, dict) and s.get("key")]
        stage_keys = [s for s in stage_keys if s] or list(STAGES)
        cmd = snap.get("cmd") or []
        if not isinstance(cmd, list) or not cmd:
            raise ValueError("待跑任务缺少可执行命令，请重新添加")
        job = cls(
            snap.get("id") or uuid.uuid4().hex[:12],
            name=str(snap.get("name") or ""),
            novel=str(snap.get("novel") or ""),
            out=str(snap.get("out") or ""),
            stages=stage_keys,
            only=snap.get("only"),
            characters=snap.get("characters"),
            image_parts=snap.get("image_parts"),
            cmd=[str(x) for x in cmd],
            log_path=Path(snap.get("log_path") or ROOT / "output"),
            detail=str(snap.get("detail") or ""),
            status="queued",
            queue_clear=[n for n in (snap.get("queue_clear") or []) if isinstance(n, dict)],
        )
        try:
            job.created_at = float(snap.get("created_at") or job.created_at)
        except (TypeError, ValueError):
            pass
        return job

    # -- 序列化 ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            prog = self.tracker.snapshot()
            # 阶段状态以引擎为准（含 pct）；任务级 status 仍用 self.stage_status 兜底
            eng_stages = prog.get("stages") or []
            if eng_stages and self.status != "queued":
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
            # 任务已终态时，禁止阶段再带 running（否则前端会一直闪）
            if self.status in {"interrupted", "failed", "success"}:
                terminal_running = "done" if self.status == "success" else "failed"
                for s in stages:
                    if s.get("status") == "running":
                        s["status"] = terminal_running
                        self.stage_status[s["key"]] = terminal_running
                prog = {**prog, "status": self.status, "current_stage": None, "current": None}
                if self.status == "interrupted":
                    prog["current"] = "已中断"
                elif self.status == "failed":
                    prog["current"] = "失败"
                elif self.status == "success":
                    prog["current"] = "完成"
                    prog["overall_pct"] = 100.0
            return {
                "id": self.id,
                "name": self.name,
                "novel": self.novel,
                "out": self.out,
                "only": self.only,
                "characters": getattr(self, "characters", None),
                "image_parts": getattr(self, "image_parts", "both"),
                "detail": self.detail,
                "status": self.status,
                "stages": stages,
                "current_stage": None
                if self.status in {"interrupted", "failed", "success", "queued"}
                else (
                    prog.get("current_stage")
                    or next((s["key"] for s in stages if s["status"] == "running"), None)
                ),
                "progress": prog if self.status != "queued" else {
                    **prog,
                    "status": "queued",
                    "overall_pct": 0.0,
                },
                "returncode": self.returncode,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log_path": str(self.log_path),
                "media_base": self.out,
                "pid": self.pid,
                "source": getattr(self, "source", "web"),
                # 待跑 / 中断都留 cmd，服务重启后可退回待跑再点启动
                "cmd": list(self.cmd) if self.status in {"queued", "interrupted"} else [],
                "queue_clear": list(self.queue_clear) if self.status in {"queued", "interrupted"} else [],
            }

    # -- 日志 --------------------------------------------------------------
    def _disk_log_lines(self) -> list[str]:
        """读 log_path 尾部。网页重启后内存 log 会空，挂载任务靠这个回填终端。"""
        path = getattr(self, "log_path", None)
        if not path:
            return []
        path = Path(path)
        if not path.is_file():
            return []
        try:
            # 只读末尾，避免超大日志拖垮接口
            data = path.read_bytes()
            if len(data) > 512_000:
                data = data[-512_000:]
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return []
        lines = [ln.rstrip("\r") for ln in text.splitlines() if ln.strip()]
        if len(lines) > LOG_TAIL_LINES:
            lines = lines[-LOG_TAIL_LINES:]
        return lines

    def log_lines(self, after: int) -> tuple[list[str], int]:
        after = max(0, int(after or 0))
        with self.lock:
            mem = list(self.log)
        disk = self._disk_log_lines()
        # 挂载/重启后内存往往只有几条 [webui]，磁盘才是完整终端；取更长的那份
        lines = disk if len(disk) > len(mem) else mem
        return lines[after:], len(lines)

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
        """queued → 先清约定产物再真正跑；已是 running 则直接拉起线程。"""
        with self.lock:
            if self.status == "queued":
                clear_nodes = list(self.queue_clear or [])
                self.status = "running"
                self.stage_status = {s: "pending" for s in self.stages}
                if self.stages:
                    self.stage_status[self.stages[0]] = "running"
                self.queue_clear = []
            elif self.status != "running":
                raise RuntimeError(f"任务状态 {self.status} 不能启动")
            else:
                clear_nodes = []
            out = self.out
        if clear_nodes:
            try:
                from pipeline.artifact_plan import clear_nodes as _clear

                removed = _clear(out, clear_nodes)
                if removed:
                    self.append_log(f"[webui] 启动前已清除 {len(removed)} 项产物")
            except Exception as exc:
                self.append_log(f"[webui] 启动前清产物失败：{exc}")
        try:
            from pipeline.process_guard import KIND_DOWNLOAD, KIND_SETUP, sweep_stale

            killed = sweep_stale(kinds=(KIND_DOWNLOAD, KIND_SETUP), keep_pids=(os.getpid(),))
            if killed:
                self.append_log(f"[webui] 启动前已清理残留进程：{', '.join(str(pid) for pid in killed)}")
        except Exception as exc:
            self.append_log(f"[webui] 启动前清理残留进程失败：{exc}")
        _persist()
        notify_change()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from pipeline.progress import run_tee

        env = dict(__import__("os").environ)
        env["PYTHONUNBUFFERED"] = "1"
        sep = ";" if sys.platform == "win32" else ":"
        env["PYTHONPATH"] = str(ROOT) + sep + env.get("PYTHONPATH", "")
        try:
            if self.stop_requested:
                self.status = "interrupted"
                self.finished_at = time.time()
                self.engine.finalize(success=False)
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.started_at = time.time()
            self.append_log(f"$ {shlex.join(self.cmd)}")

            last_persist = [0.0]

            def on_line(text: str) -> None:
                self.append_log(text)
                parse_stage_progress(text, self)
                # 引擎内部节流推送；这里只喂行
                self.engine.feed_line(text, self.current_stage_safe())
                try:
                    from pipeline.run_queue import prune_queue_by_steps

                    snap = self.engine.snapshot()
                    if prune_queue_by_steps(self.out, snap.get("steps") or []):
                        notify_change()
                except Exception:
                    pass
                now = time.time()
                if now - last_persist[0] >= 2.0:
                    last_persist[0] = now
                    _persist()

            def on_start(pid: int) -> None:
                self.pid = pid
                _persist()

            code = run_tee(self.cmd, log=self.log_path, env=env, cwd=ROOT, on_line=on_line, on_start=on_start)
            self.returncode = code
            # 已被 stop_all 退回 queued：不要再改成 interrupted/failed
            if self.status == "queued":
                pass
            elif self.stop_requested or self.status == "interrupted":
                self.status = "interrupted"
                self.engine.finalize(success=False)
                with self.lock:
                    for s, st in self.stage_status.items():
                        if st == "running":
                            self.stage_status[s] = "failed"
            else:
                self.status = "success" if code == 0 else "failed"
                self.engine.finalize(success=(code == 0))
                with self.lock:
                    for s, st in self.stage_status.items():
                        if st == "running":
                            self.stage_status[s] = "done" if code == 0 else "failed"
        except Exception as exc:
            self.error = str(exc)
            if self.status != "queued":
                self.status = "interrupted" if self.stop_requested else "failed"
            self.append_log(f"[webui] 任务异常：{exc}")
        finally:
            parked = self.status == "queued"
            if not parked:
                self.finished_at = time.time()
            self.append_log(
                f"[webui] 任务结束：{self.status}（退出码 {self.returncode}）"
                if not parked
                else f"[webui] 进程已停，保持待跑（退出码 {self.returncode}）"
            )
            notify_change()
            _persist()
            disarm_queue(self.out)
            # 成功：待跑项清空；失败/中断：复位 pending 可再启
            try:
                from pipeline.run_queue import clear_queue, load_queue, reset_statuses, save_queue

                if self.status == "success":
                    clear_queue(self.out)
                elif not parked:
                    q = load_queue(self.out)
                    reset_statuses(q, "pending")
                    save_queue(self.out, q)
                notify_change()
            except Exception as exc:
                print(f"[webui] 同步队列状态失败：{exc}", flush=True)

    def park_as_queued(self) -> bool:
        """把已停下的任务重新挂回待跑（保留 cmd，可再点启动）。"""
        with self.lock:
            if not self.cmd:
                return False
            self.status = "queued"
            self.stop_requested = False
            self.error = None
            self.returncode = None
            self.pid = None
            self.started_at = None
            self.finished_at = None
            self.stage_status = {s: "pending" for s in self.stages}
            # 已跑过一轮：再启动时按 cmd 里的 --force 重做即可
            self.queue_clear = []
        try:
            # 新进度引擎，避免沿用已 finalize 的旧状态
            self.engine = ProgressEngine(
                self.out,
                stages=list(self.stages),
                task_id=self.id,
                only=self.only,
                characters=getattr(self, "characters", None),
                image_parts=getattr(self, "image_parts", "both"),
            )
            self.tracker = _EngineTracker(self.engine)
        except Exception:
            pass
        self.append_log("[webui] 已退回待跑，点「启动任务」继续")
        return True

    def stop(self) -> None:
        """请求停止：状态保持 running 到工作线程真正退出，避免新旧进程重叠。

        进程组先 SIGTERM；若 10 秒后仍未退出，再 SIGKILL，防止互斥被卡死。
        """
        import os
        import signal

        with self.lock:
            if self.status != "running":
                return
            self.stop_requested = True
            self.error = "用户停止"
            for s, st in self.stage_status.items():
                if st == "running":
                    self.stage_status[s] = "failed"
        disarm_queue(self.out)
        try:
            if self.engine is not None:
                self.engine.finalize(success=False)
        except Exception:
            pass
        pid = self.pid
        if not pid:
            self.append_log("[webui] 已请求停止（进程尚未拉起，启动后立即退出）")
            notify_change()
            _persist()
            return

        def _terminate() -> None:
            try:
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                self.append_log("[webui] 已请求停止…")
            except ProcessLookupError:
                self.append_log("[webui] 进程已结束")
            except Exception as exc:
                self.append_log(f"[webui] 停止失败：{exc}")

        _terminate()
        notify_change()
        _persist()

        def _force_kill() -> None:
            from pipeline.live import pid_alive

            for _ in range(20):
                time.sleep(0.5)
                with self.lock:
                    gone = self.status != "running" or self.pid != pid
                if gone or not pid_alive(pid):
                    return
            try:
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
                else:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                self.append_log("[webui] 进程未响应 SIGTERM，已强制结束")
                notify_change()
            except Exception as exc:
                self.append_log(f"[webui] 强制结束失败：{exc}")

        threading.Thread(target=_force_kill, daemon=True).start()


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_CHANGE_QUEUES: list[queue.Queue] = []
# 当前网页打开的项目名下的任务 id：清空记录时只清这些，别的项目不受影响
CURRENT_PROJECT_JOB_IDS: set[str] = set()
# 用户点过「启动任务」的 out：只有 armed 才会在上一项成功后自动续跑待跑队列。
# 绝不能仅因「列表里有 queued」或「别的任务碰巧跑完」就开跑。
_QUEUE_ARMED: set[str] = set()
_START_LOCK = threading.Lock()


def _running_jobs_locked() -> list[Job]:
    # 停止时不再提前把状态改成 interrupted：工作线程真正退出后才离开 running，
    # 这样新任务不会和还没退净的旧进程重叠。不能用 pid 存活兜底——PID 会被复用。
    return [j for j in _JOBS.values() if j.status == "running"]


def _ensure_no_running_job() -> None:
    try:
        attach_external_jobs(None)
    except Exception:
        pass
    with _LOCK:
        running = _running_jobs_locked()
    if running:
        raise ValueError("已有任务在跑，请先停止")


def _norm_out(out: str) -> str:
    """队列 armed 集合用的规范化路径（保留绝对路径前导 /）。"""
    text = (out or "").replace("\\", "/")
    if text.startswith("/"):
        return text.rstrip("/") or "/"
    return text.strip("/")


def arm_queue(out: str) -> None:
    out_n = _norm_out(out)
    if out_n:
        _QUEUE_ARMED.add(out_n)


def disarm_queue(out: str) -> None:
    _QUEUE_ARMED.discard(_norm_out(out))


def queue_is_armed(out: str) -> bool:
    return _norm_out(out) in _QUEUE_ARMED


def notify_change() -> None:
    with _LOCK:
        subs = list(_CHANGE_QUEUES)
    for q in subs:
        try:
            q.put_nowait("change")
        except Exception:
            pass


_ACTIVE_ATTACH_AT = 0.0


def _with_version(payload: dict) -> dict:
    from pipeline.version import read_version

    payload["version"] = read_version()
    return payload


def active_run_info() -> dict:
    """菜单栏 / 打开网页用：当前是否有生成任务、总进度%、所属项目。

    多任务在跑时取最近启动的一条。
    """
    global _ACTIVE_ATTACH_AT
    now = time.time()
    # 轻量轮询别每次全盘挂载；隔几秒补一次外部 produce
    if now - _ACTIVE_ATTACH_AT >= 3.0:
        _ACTIVE_ATTACH_AT = now
        try:
            attach_external_jobs(None)
        except Exception:
            pass
    with _LOCK:
        running = [j for j in _JOBS.values() if j.status == "running"]
    if not running:
        return _with_version({
            "running": False,
            "pct": None,
            "project": None,
            "out": None,
            "task_id": None,
            "current": None,
            "detail": None,
        })
    running.sort(key=lambda j: float(getattr(j, "started_at", None) or getattr(j, "created_at", 0) or 0), reverse=True)
    job = running[0]
    snap = job.snapshot()
    prog = snap.get("progress") or {}
    try:
        pct = float(prog.get("overall_pct") if prog.get("overall_pct") is not None else 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    pct = max(0.0, min(100.0, pct))
    return _with_version({
        "running": True,
        "pct": round(pct, 1),
        "project": project_of_job(job) or None,
        "out": snap.get("out") or job_out(job) or None,
        "task_id": snap.get("id"),
        "current": prog.get("current") or snap.get("current_stage"),
        "detail": snap.get("detail") or snap.get("name"),
    })


def state_path_for(out: str) -> Path:
    """某个项目的任务历史文件：<项目>/work/webui/state.json。"""
    return artifacts_dir(out) / STATE_DIRNAME / STATE_FILENAME


def job_out(job: Job) -> str:
    """任务的归属项目文件夹（新建任务在 __init__ 里也存了 out）。"""
    return getattr(job, "out", "") or ""


def project_of_job(job: Job) -> str:
    """任务归属的项目名（文件夹名；旧相对路径 / 绝对路径都能认）。"""
    from pipeline.live import project_name_from_out

    out = job_out(job).replace("\\", "/").strip()
    if out in {f"{PROJECTS_DIRNAME}/{STATE_DIRNAME}", STATE_DIRNAME}:
        return ""  # 旧的全局历史不挂到任何项目下
    return project_name_from_out(out)


def jobs_for_project(name: str) -> list[Job]:
    """某个项目名下的任务（内存里的；新任务启动后项目文件夹一定存在）。"""
    abs_out = project_path_str(name) if name else ""
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    if not name:
        return [j for j in jobs if not project_of_job(j)]
    return [
        j for j in jobs
        if project_of_job(j) == name
        # 旧布局 output/<片名> / projects/<片名> 与同名新布局项目互认
        or (j.out in {f"output/{name}", f"{PROJECTS_DIRNAME}/{name}", abs_out})
        or resolve(j.out or "") == Path(abs_out)
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
        if not project_of_job(j)
        and j.out not in {f"{PROJECTS_DIRNAME}/{STATE_DIRNAME}", STATE_DIRNAME}
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
        if job.status == "running" and _running_jobs_locked():
            raise ValueError("已有任务在跑，请先停止")
        _JOBS[job.id] = job
        CURRENT_PROJECT_JOB_IDS.add(job.id)
    _persist()


def queued_jobs_for_out(out: str) -> list[Job]:
    """某输出目录下按创建时间排序的待跑任务。"""
    out_n = _norm_out(out)
    with _LOCK:
        jobs = [
            j for j in _JOBS.values()
            if j.status == "queued" and _norm_out(job_out(j)) == out_n
        ]
    return sorted(jobs, key=lambda j: j.created_at)


def pump_queued_jobs(out: str) -> Job | None:
    """同项目没有运行中任务、且已 armed 时，启动最早一条待跑。返回启动的任务或 None。"""
    out_n = _norm_out(out)
    if not out_n:
        return None
    if not queue_is_armed(out_n):
        return None
    with _LOCK:
        if _running_jobs_locked():
            return None
        queued = sorted(
            [
                j for j in _JOBS.values()
                if j.status == "queued" and _norm_out(job_out(j)) == out_n
            ],
            key=lambda j: j.created_at,
        )
        nxt = queued[0] if queued else None
        if not nxt:
            return None
        # 在全局锁内抢占，避免并发 pump 双开
        clear_nodes = list(nxt.queue_clear or [])
        nxt.status = "running"
        nxt.stage_status = {s: "pending" for s in nxt.stages}
        if nxt.stages:
            nxt.stage_status[nxt.stages[0]] = "running"
        nxt.queue_clear = []
    if clear_nodes:
        try:
            from pipeline.artifact_plan import clear_nodes as _clear

            removed = _clear(out_n, clear_nodes)
            if removed:
                nxt.append_log(f"[webui] 启动前已清除 {len(removed)} 项产物")
        except Exception as exc:
            nxt.append_log(f"[webui] 启动前清产物失败：{exc}")
    _persist()
    notify_change()
    threading.Thread(target=nxt._run, daemon=True).start()
    return nxt


def start_queued_tasks(out: str) -> dict:
    """启动指定 out 的待跑任务；全局已有任务时拒绝启动。"""
    out_n = _norm_out(out)
    if not out_n:
        raise ValueError("请指定项目 out")
    _ensure_no_running_job()
    queued = queued_jobs_for_out(out_n)
    if not queued:
        raise ValueError("任务列表里没有待跑项；请先通过单跑对话框添加")
    # 显式点启动才 armed；之后同 out 上一项成功才会续跑
    with _START_LOCK:
        with _LOCK:
            running = _running_jobs_locked()
        if running:
            raise ValueError("已有任务在跑，请先停止")
        arm_queue(out_n)
        started = pump_queued_jobs(out_n)
    return {
        "ok": True,
        "started": started.snapshot() if started else None,
        "queued": max(0, len(queued) - (1 if started else 0)),
        "message": f"已启动：{started.name}" if started else "未能启动",
    }


def _drop_task_from_disk(out: str, task_id: str) -> None:
    """从 state.json 删掉指定任务（合并持久化不会自动删）。"""
    path = state_path_for(out)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = dict(data.get("tasks") or {})
        if task_id not in tasks:
            return
        del tasks[task_id]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"[webui] 从磁盘移除任务失败：{exc}", flush=True)


def cancel_queued_task(task_id: str) -> dict:
    """从列表移除一条待跑任务（未开始的）。"""
    job = get_job(task_id)
    if not job:
        raise FileNotFoundError("task not found")
    if job.status != "queued":
        raise ValueError("只能移除待跑任务")
    out = job_out(job)
    with _LOCK:
        _JOBS.pop(job.id, None)
        CURRENT_PROJECT_JOB_IDS.discard(job.id)
    if out:
        _drop_task_from_disk(out, task_id)
    if out and not queued_jobs_for_out(out):
        disarm_queue(out)
    _persist()
    notify_change()
    return {"ok": True, "id": task_id}


def stop_pipeline_keep_queue(out: str) -> dict:
    """杀掉本项目在跑的历史任务；待跑队列内容不变，状态全部复位 pending。"""
    from pipeline.run_queue import annotate_removable, load_queue, reset_statuses, save_queue

    out_n = _norm_out(out)
    if not out_n:
        raise ValueError("请指定项目 out")
    disarm_queue(out_n)
    with _LOCK:
        running = [
            j for j in _JOBS.values()
            if _norm_out(job_out(j)) == out_n and j.status == "running"
        ]
    stopped: list[str] = []
    for job in running:
        job.stop()
        stopped.append(job.id)
    queue = load_queue(out_n)
    reset_statuses(queue, "pending")
    save_queue(out_n, queue)
    _persist()
    notify_change()
    return {
        "ok": True,
        "out": out_n,
        "stopped": stopped,
        "queue": annotate_removable(out_n, queue),
        "message": (
            f"已停止 {len(stopped)} 个运行中任务；待跑队列仍有 {len(queue.get('items') or [])} 项"
            if stopped
            else "当前没有在跑任务"
        ),
    }


def load_project_tasks(name: str) -> list[Job]:
    """读某项目的 state.json → 恢复成只读历史任务，返回该项目的全部任务。

    服务重启后内存是空的，网页第一次选某个项目时按需恢复它的历史；
    运行中的旧记录：若 pid 仍活着则挂载继续盯进度，否则标成「中断」。
    另外会扫描 CLI produce / live.json，把外部任务挂进列表。
    """
    from pipeline.live import pid_alive

    path = state_path_for(project_path_str(name))
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
            if snap.get("status") in {"queued", "interrupted"} and snap.get("cmd"):
                # 中断但留着 cmd：服务重启后当作待跑，点启动即可继续
                try:
                    qsnap = dict(snap)
                    qsnap["status"] = "queued"
                    job = Job.from_queued_snapshot(qsnap)
                except Exception as exc:
                    if snap.get("status") == "queued":
                        snap = dict(snap)
                        snap["status"] = "interrupted"
                        snap["error"] = f"待跑任务无法恢复：{exc}"
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
                        continue
                    # 普通中断无 cmd 可恢复：走下面只读历史
                    pass
                else:
                    if snap.get("status") == "interrupted":
                        dirty = True  # 写回 queued
                    with _LOCK:
                        _JOBS[task_id] = job
                        CURRENT_PROJECT_JOB_IDS.add(job.id)
                    restored.append(job)
                    continue
            if snap.get("status") == "queued":
                # 无 cmd 的待跑：降级历史
                snap = dict(snap)
                snap["status"] = "interrupted"
                snap["error"] = "待跑任务缺少命令，请重新添加"
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
                continue
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
    projects_root = projects_dir()
    if projects_root.is_dir():
        for child in projects_root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if name and child.name != name:
                continue
            abs_out = str(child.resolve())
            live = read_live(abs_out)
            if live and (live.get("status") == "running" or pid_alive(live.get("pid"))):
                outs.add(abs_out)

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
        with _LOCK:
            # 不挂载第二个外部进程，以便其他任务入口能一致拒绝启动。
            if _running_jobs_locked():
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
    """把任务历史写回各自项目的 <项目>/work/webui/state.json。

    与磁盘合并写入：内存里的同 id 覆盖磁盘；磁盘上其它历史保留。
    避免服务刚启动、内存尚未 load 全量时，一次入队把整份历史冲成只剩一条。
    """
    try:
        with _LOCK:
            jobs = list(_JOBS.values())
        mem_by_out: dict[str, dict] = {}
        for job in jobs:
            out = job_out(job)
            if not out:
                continue
            mem_by_out.setdefault(out, {})[job.id] = job.snapshot()
        for out, mem_tasks in mem_by_out.items():
            path = state_path_for(out)
            disk_tasks: dict = {}
            if path.exists():
                try:
                    disk_tasks = dict((json.loads(path.read_text(encoding="utf-8")) or {}).get("tasks") or {})
                except Exception:
                    disk_tasks = {}
            merged = dict(disk_tasks)
            merged.update(mem_tasks)
            if len(merged) > HISTORY:
                keep = sorted(
                    merged.items(),
                    key=lambda kv: kv[1].get("created_at", 0),
                    reverse=True,
                )[:HISTORY]
                merged = dict(keep)
            path.parent.mkdir(parents=True, exist_ok=True)
            # 唯一临时名：并发 _persist（启动/停止/进度）共用 state.tmp 会互相踩丢文件
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps({"tasks": merged}, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
    except Exception as exc:
        print(f"[webui] 持久化任务失败：{exc}", flush=True)


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
                "characters": snap.get("characters"),
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
        characters = getattr(self, "characters", None)
        image_parts = getattr(self, "image_parts", None) or "both"
        if not characters or image_parts == "both":
            try:
                from pipeline.run_queue import load_queue, queue_image_parts, queue_portrait_ids

                q = load_queue(self.out)
                if not characters:
                    characters = queue_portrait_ids(q)
                    self.characters = characters
                if image_parts == "both" and (q.get("items") or []):
                    image_parts = queue_image_parts(q)
                    self.image_parts = image_parts
            except Exception:
                pass
        self.engine = ProgressEngine(
            self.out,
            stages=list(self.stages),
            task_id=self.id,
            only=self.only,
            characters=characters,
            image_parts=image_parts,
        )
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
                try:
                    from pipeline.run_queue import clear_queue, load_queue, reset_statuses, save_queue

                    if ok:
                        clear_queue(self.out)
                    else:
                        q = load_queue(self.out)
                        reset_statuses(q, "pending")
                        save_queue(self.out, q)
                    notify_change()
                except Exception as exc:
                    print(f"[webui] 同步队列状态失败：{exc}", flush=True)
                return
            try:
                self.refresh_from_disk()
                from pipeline.run_queue import prune_queue_by_steps

                snap = self.engine.snapshot()
                if prune_queue_by_steps(self.out, snap.get("steps") or []):
                    notify_change()
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
    characters: str | None = None,
    end_only: bool = False,
    skip_end: bool = False,
) -> list[str]:
    """单阶段命令；CLI 子命令一次只能一个，多阶段靠 run / bash && 串。"""
    cmd = [sys.executable, "-m", "pipeline", stage, "--out", str(out)]
    if stage in {"storyboard", "screenplay"}:
        # storyboard：小说→分镜；screenplay：校验（可带外部分镜）
        if novel:
            cmd += ["--novel", str(resolve(novel))]
        if storyboard and stage == "screenplay":
            cmd += ["--storyboard", str(resolve(storyboard))]
    if only and stage in {"images", "audio", "videos"}:
        cmd += ["--only", str(only)]
    if characters and stage == "images":
        cmd += ["--characters", str(characters)]
    if stage == "images" and end_only:
        cmd.append("--end-only")
    elif stage == "images" and skip_end:
        cmd.append("--skip-end")
    if force:
        cmd.append("--force")
    return cmd


def build_artifact_groups_cmd(
    *,
    out: str,
    groups: list[dict],
    novel: str | None = None,
) -> tuple[list[str], Path, list[str]]:
    """按 execution_groups 拼 bash && 串跑；返回 (argv, log, stage_keys)。"""
    out_dir = resolve(out)
    parts: list[str] = []
    stage_keys: list[str] = []
    for g in groups:
        stage = g["stage"]
        stage_keys.append(stage)
        chars = g.get("characters") or []
        parts.append(
            shlex.join(
                _single_stage_cmd(
                    stage,
                    out=out_dir,
                    novel=novel,
                    only=g.get("only"),
                    force=bool(g.get("force")),
                    characters=",".join(chars) if chars else None,
                    end_only=bool(g.get("end_only")),
                    skip_end=bool(g.get("skip_end")),
                )
            )
        )
    if not parts:
        raise ValueError("没有可执行的阶段")
    # 打印阶段分隔，便于进度引擎识别
    script_parts = []
    for g, part in zip(groups, parts):
        script_parts.append(f"echo '== {g['stage']} =='")
        script_parts.append(part)
    cmd = ["/bin/bash", "-c", " && ".join(script_parts)]
    log_path = artifacts_sub(out_dir, "logs", f"webui-{time.strftime('%Y%m%d-%H%M%S')}.log")
    return cmd, log_path, stage_keys


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
        done = running = failed = queued = 0
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
                elif status == "queued":
                    queued += 1
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
        item["tasks"] = {
            "done": done,
            "running": running,
            "failed": failed,
            "queued": queued,
            "total": done + running + failed + queued,
        }
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
    abs_dir = str(directory.resolve())
    return {
        "name": directory.name,
        "path": abs_dir,
        "display_path": display_path(directory),
        "novel": str((directory / "novel.txt").resolve()),
        "storyboard": str((directory / "storyboard.json").resolve()),
    }


_SAFE_PATH = re.compile(r"^[\w\-./\u4e00-\u9fff~]+$")


def safe_path(rel: str) -> Path:
    """解析媒体/文本路径：允许运行根、用户项目根；兼容旧 projects/ 前缀。"""
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or not _SAFE_PATH.match(rel):
        raise ValueError(f"不接受的路径：{rel!r}")
    # 允许 ~/…
    if rel.startswith("~/"):
        path = Path(rel).expanduser().resolve()
    else:
        path = resolve(rel)
    if not under_allowed_root(path):
        raise ValueError(f"路径越界：{rel}")
    return path


def read_text(rel: str) -> str:
    path = safe_path(rel)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path.name}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件不是 UTF-8 文本：{path.name}") from exc


def write_text(rel: str, content: str) -> None:
    path = safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def api_voice_ref_upload(data: dict) -> dict:
    """把角色参考音频写到项目 ``work/voices/<id>_ref.wav``，并回写分镜字段。"""
    import base64
    import binascii

    from pipeline.projects import artifacts_sub

    out_rel = str(data.get("out") or data.get("project") or "").strip()
    char_id = str(data.get("character_id") or data.get("id") or "").strip()
    if not out_rel or not char_id:
        raise ValueError("需要 out 与 character_id")
    if not re.match(r"^[\w\-]+$", char_id):
        raise ValueError(f"不接受的角色 id：{char_id!r}")
    payload = data.get("content_base64") or data.get("base64") or ""
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("需要 content_base64（参考音频）")
    # 允许 data URL
    raw_b64 = payload.strip()
    if "," in raw_b64 and raw_b64.lower().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("参考音频 base64 无效") from exc
    if len(blob) < 256:
        raise ValueError("参考音频太短")
    if len(blob) > 12 * 1024 * 1024:
        raise ValueError("参考音频请控制在 12 MB 以内")
    base = safe_path(out_rel)
    if not base.is_dir():
        raise ValueError(f"项目不存在：{out_rel}")
    voices_dir = artifacts_sub(base, "voices")
    voices_dir.mkdir(parents=True, exist_ok=True)
    # 后缀以客户端为准，默认 wav
    name = str(data.get("filename") or "ref.wav")
    suffix = Path(name).suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        suffix = ".wav"
    dest = voices_dir / f"{char_id}_ref{suffix}"
    dest.write_bytes(blob)
    rel = str(dest.relative_to(base)).replace("\\", "/")
    ref_text = data.get("ref_text")
    # 可选：直接改 storyboard.json 里的角色卡
    sb_path = base / "storyboard.json"
    updated = False
    if sb_path.is_file() and bool(data.get("update_storyboard", True)):
        from pipeline.util import dump_json, load_json

        story = load_json(sb_path)
        for char in story.get("characters") or []:
            if not isinstance(char, dict):
                continue
            if str(char.get("id") or "") != char_id:
                continue
            char["voice"] = "clone"
            char["voice_mode"] = "clone"
            char["voice_ref"] = rel
            if ref_text is not None:
                char["voice_ref_text"] = str(ref_text).strip()
            updated = True
            break
        if updated:
            dump_json(sb_path, story)
    return {
        "ok": True,
        "path": rel,
        "character_id": char_id,
        "bytes": len(blob),
        "storyboard_updated": updated,
        "voice_ref_text": None if ref_text is None else str(ref_text).strip(),
    }


def _storyboard_read_error(data: object) -> str | None:
    from pipeline.artifact_plan import _shape_error

    return _shape_error(data)


def read_storyboard(rel: str) -> dict:
    """读分镜。

    文件不存在时 ``exists=False``，不抛错（网页据此显示空列表）。
    文件在但不是合法分镜时带 ``error``，网页要显示损坏原因，不要当成「还没有分镜」。
    """
    path = safe_path(rel)
    if not path.is_file():
        return {"path": rel, "exists": False, "storyboard": None, "mtime": None}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": rel,
            "exists": True,
            "storyboard": None,
            "mtime": mtime,
            "error": f"分镜文件损坏，不是合法 JSON（{exc.msg}）",
        }
    err = _storyboard_read_error(data)
    if err:
        return {"path": rel, "exists": True, "storyboard": None, "mtime": mtime, "error": err}
    return {"path": rel, "exists": True, "storyboard": data, "mtime": mtime}


def write_storyboard(rel: str, data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("分镜必须是 JSON 对象")
    # 去掉网页内部标记（_missing / _corrupt），避免写进源文件
    payload = {k: v for k, v in data.items() if not str(k).startswith("_")}
    err = _storyboard_read_error(payload)
    if err:
        raise ValueError(err)
    path = safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def materialize_web_draft(
    out: str,
    *,
    storyboard_data: dict | None = None,
    novel_text: str | None = None,
) -> tuple[str | None, str | None]:
    """把网页暂存落到 ``work/webui/run_*``，供本次任务读；不写项目源 novel.txt。

    分镜会 ``install_storyboard(..., force=True)`` 进 ``out/storyboard.json``（管线运行态），
    与用户点「保存分镜」写源文件是两回事——保存仍只走显式写盘接口。
    返回 ``(storyboard_rel, novel_rel)``，未提供的一侧为 None。
    """
    from pipeline.run import install_storyboard

    base = resolve(out)
    webui = artifacts_sub(base, "webui")
    webui.mkdir(parents=True, exist_ok=True)
    def atomic_write(path: Path, content: str) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    sb_rel: str | None = None
    novel_rel: str | None = None
    if isinstance(storyboard_data, dict):
        # 剥掉网页内部标记；空壳（无镜头）不装进运行态，避免假分镜让成片依赖算空
        payload = {
            k: v for k, v in storyboard_data.items() if not str(k).startswith("_")
        }
        shots = payload.get("shots") or []
        has_shot = isinstance(shots, list) and any(
            isinstance(s, dict) and str(s.get("id") or "").strip() for s in shots
        )
        chars = payload.get("characters") or []
        has_char = isinstance(chars, list) and any(
            isinstance(c, dict) and str(c.get("id") or "").strip() for c in chars
        )
        run_sb = webui / "run_storyboard.json"
        atomic_write(run_sb, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        dest = base / "storyboard.json"
        # 有镜头、有角色，或项目里已经有分镜（正在改标题/配乐）才装进运行态。
        # 纯空壳不落盘，避免「还没有分镜」被写成假文件，成片依赖被算空。
        if has_shot or has_char or dest.is_file():
            draft_rel = cite_path(run_sb)
            install_storyboard(out, draft_rel, force=True)
            sb_rel = cite_path(dest)
        else:
            sb_rel = None
    if novel_text is not None:
        run_novel = webui / "run_novel.txt"
        atomic_write(run_novel, str(novel_text).rstrip() + "\n")
        novel_rel = cite_path(run_novel)
    return sb_rel, novel_rel


def validate_storyboard(
    path_str: str,
    *,
    strict: bool,
    storyboard: dict | None = None,
    novel_text: str | None = None,
) -> dict:
    """借用 pipeline.screenplay 的逻辑做秒级校验（不加载模型、不写盘）。

    ``storyboard`` / ``novel_text`` 若传入，则用内存内容校验，不读/写磁盘上的对应文件。
    """
    from pipeline.config import Settings
    from pipeline.projects import strip_storyboard_suffix
    from pipeline.screenplay import (
        CHARS_PER_SECOND,
        assert_dialogue_coverage,
        dialogue_budget_violations,
        dialogue_char_limit,
        normalize_storyboard,
    )
    from pipeline.util import snap_frames

    settings = Settings.load()
    src: Path | None = None
    if path_str:
        src = safe_path(path_str)
    raw = storyboard if isinstance(storyboard, dict) else None
    if raw is None:
        if src is None or not src.is_file():
            # 还没有分镜文件：网页应显示空列表，不算校验失败
            return _validate_skip_empty(
                settings,
                source="disk",
                novel_source="none",
                reason="missing",
            )
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _validate_fail(
                settings,
                f"分镜文件损坏，不是合法 JSON（{exc.msg}）",
            )
    shape = _storyboard_read_error(raw)
    if shape:
        return _validate_fail(settings, shape)

    shots_in = raw.get("shots") or []
    if not isinstance(shots_in, list) or not shots_in:
        # 空壳分镜（无镜头）不跑硬校验
        return _validate_skip_empty(
            settings,
            source="memory" if isinstance(storyboard, dict) else "disk",
            novel_source="memory" if novel_text is not None else "none",
            reason="empty",
            characters=len((raw or {}).get("characters") or []) if isinstance(raw, dict) else 0,
        )

    novel_path: Path | None = None
    novel_body = novel_text
    if novel_body is None:
        linked = str(raw.get("novel") or "").strip()
        if linked:
            cand = safe_path(linked)
            if cand.exists():
                novel_path = cand
        if novel_path is None and src is not None:
            for cand in novel_candidates(src):
                novel_path = cand
                break
        if novel_path is None and src is not None:
            stem = strip_storyboard_suffix(src.stem)
            for cand in (
                project_dir(stem) / "novel.txt",
                ROOT / f"input/{stem}_novel.txt",
                ROOT / f"input/{stem}.txt",
            ):
                if cand.exists():
                    novel_path = cand
                    break
        if novel_path is not None:
            novel_body = novel_path.read_text(encoding="utf-8")

    fps = max(1, settings.video_fps)
    max_frames = snap_frames(settings.video_frames_max)
    max_seconds = max_frames / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    detail = settings.budget_detail or {}
    try:
        normalized = normalize_storyboard(
            raw,
            novel_path=novel_path,
            max_shots=settings.max_shots,
            dialogue_limit=None,
            max_seconds=max_seconds,
        )
        shots = normalized["shots"]
        if novel_body:
            assert_dialogue_coverage(novel_body, shots)
    except SystemExit as exc:
        # screenplay 用 SystemExit 报错；网页转成 JSON，别把 HTTP 线程干死
        msg = str(exc).strip() or "校验未通过"
        return _validate_fail(
            settings,
            msg,
            shots=len(shots_in),
            characters=len(raw.get("characters") or []),
            novel=str(novel_path) if novel_path else "",
            source="memory" if isinstance(storyboard, dict) else "disk",
            novel_source="memory" if novel_text is not None else ("disk" if novel_path else "none"),
        )

    rows, violations = dialogue_budget_violations(
        shots, limit, max_seconds, strict=strict
    )
    report_lines = []
    if rows:
        report_lines.append(
            f"对白时长预估（语速 {CHARS_PER_SECOND} 字/秒，仅供参考；镜头会跟着配音长度走）："
        )
        report_lines.extend(rows)

    strict_error: str | None = None
    if strict and violations:
        strict_error = (
            f"{len(violations)} 个镜头对白超过单镜额度 {limit} 字"
            f"（约 {round(max_seconds, 1)}s）。不要删字，按建议拆成相邻镜头。"
        )

    result = {
        "ok": strict_error is None,
        "strict_error": strict_error,
        "violations": violations if strict else [],
        "shots": len(shots),
        "characters": len(normalized.get("characters") or []),
        "dialogue_lines": normalized.get("dialogue_lines", 0),
        "limit_chars": limit,
        "max_seconds": round(max_seconds, 1),
        "max_frames": max_frames,
        "fps": fps,
        "chars_per_second": CHARS_PER_SECOND,
        "resolution": f"{settings.video_width}x{settings.video_height}",
        "total_gib": detail.get("total_gib"),
        "tier_gib": detail.get("tier_gib"),
        "novel": str(novel_path) if novel_path else "",
        "report": "\n".join(report_lines),
        # 标明校验用的是请求体还是磁盘，避免前端误以为落盘了
        "source": "memory" if isinstance(storyboard, dict) else "disk",
        "novel_source": "memory" if novel_text is not None else ("disk" if novel_path else "none"),
    }
    return result


def _validate_fail(
    settings,
    message: str,
    *,
    shots: int = 0,
    characters: int = 0,
    novel: str = "",
    source: str = "disk",
    novel_source: str = "none",
) -> dict:
    """分镜读不出来或覆盖检查失败：返回 ok=False，别把异常抛出 HTTP 线程。"""
    from pipeline.screenplay import CHARS_PER_SECOND, dialogue_char_limit
    from pipeline.util import snap_frames

    fps = max(1, settings.video_fps)
    max_frames = snap_frames(settings.video_frames_max)
    max_seconds = max_frames / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    detail = settings.budget_detail or {}
    msg = str(message or "").strip() or "校验未通过"
    return {
        "ok": False,
        "strict_error": msg,
        "violations": [],
        "shots": shots,
        "characters": characters,
        "dialogue_lines": 0,
        "limit_chars": limit,
        "max_seconds": round(max_seconds, 1),
        "max_frames": max_frames,
        "fps": fps,
        "chars_per_second": CHARS_PER_SECOND,
        "resolution": f"{settings.video_width}x{settings.video_height}",
        "total_gib": detail.get("total_gib"),
        "tier_gib": detail.get("tier_gib"),
        "novel": novel,
        "report": msg,
        "source": source,
        "novel_source": novel_source,
    }


def _validate_skip_empty(
    settings,
    *,
    source: str,
    novel_source: str,
    reason: str,
    characters: int = 0,
) -> dict:
    """无分镜 / 无镜头：跳过硬校验，返回 ok。"""
    from pipeline.screenplay import CHARS_PER_SECOND, dialogue_char_limit
    from pipeline.util import snap_frames

    fps = max(1, settings.video_fps)
    max_frames = snap_frames(settings.video_frames_max)
    max_seconds = max_frames / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    detail = settings.budget_detail or {}
    return {
        "ok": True,
        "strict_error": None,
        "violations": [],
        "shots": 0,
        "characters": characters,
        "dialogue_lines": 0,
        "limit_chars": limit,
        "max_seconds": round(max_seconds, 1),
        "max_frames": max_frames,
        "fps": fps,
        "chars_per_second": CHARS_PER_SECOND,
        "resolution": f"{settings.video_width}x{settings.video_height}",
        "total_gib": detail.get("total_gib"),
        "tier_gib": detail.get("tier_gib"),
        "novel": "",
        "report": "",
        "skipped": reason,
        "source": source,
        "novel_source": novel_source,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def api_models() -> dict:
    """各阶段可选模型，以及 config.json 里当前选中的那条。"""
    from pipeline.backends import selection_view
    from pipeline.config import Settings

    return selection_view(Settings.load())


def api_models_space(data: dict | None = None) -> dict:
    """当前选择里还没下载的权重，和缓存盘剩余空间。不够时 ``ok`` 为假，调用方不要开始下载。"""
    from dataclasses import replace

    from pipeline.backends.catalog import STAGE_FIELDS, STAGES, download_specs, resolve_stage_model
    from pipeline.config import Settings
    from pipeline.disk_space import plan_weight_space

    settings = Settings.load()
    raw = None
    if isinstance(data, dict):
        candidate = data.get("models") if isinstance(data.get("models"), dict) else data.get("keep")
        if "models" in data and not isinstance(data.get("models"), dict) and data.get("models") is not None:
            raise ValueError("models 需要是阶段到模型 id 的对象")
        if "keep" in data and not isinstance(data.get("keep"), dict) and data.get("keep") is not None:
            raise ValueError("keep 需要是阶段到模型 id 的对象")
        raw = candidate if isinstance(candidate, dict) else None
    if raw:
        unknown = [str(key) for key in raw if str(key) not in STAGE_FIELDS]
        if unknown:
            known = "、".join(STAGES)
            raise ValueError(f"未知阶段 {'、'.join(unknown)}（可选 {known}）")
        overlay: dict[str, str] = {}
        for stage, value in raw.items():
            try:
                resolve_stage_model(str(stage), str(value))
            except SystemExit as exc:
                raise ValueError(str(exc)) from exc
            overlay[STAGE_FIELDS[str(stage)]] = str(value)
        settings = replace(settings, **overlay)
    try:
        specs = download_specs(settings)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    return plan_weight_space(specs)


def api_models_save(data: dict) -> dict:
    """把网页上的阶段选择写回 config.json。只改模型字段（及可选外部 API）。"""
    from pipeline.backends import save_selection, selection_view
    from pipeline.config import Settings
    from pipeline.util import ROOT

    raw = data.get("models") if isinstance(data.get("models"), dict) else data.get("selections")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("需要 models：阶段 → 模型 id")
    api = data.get("storyboard_api") if isinstance(data.get("storyboard_api"), dict) else {}
    extra: dict = {}
    if "base" in api:
        extra["api_base"] = str(api.get("base") or "")
    # key 省略 = 不改已有密钥；传 "" 表示清空
    if "key" in api:
        extra["api_key_value"] = "" if api.get("key") is None else str(api.get("key"))
    try:
        save_selection(
            ROOT / "config.json",
            {str(key): str(value) for key, value in raw.items()},
            **extra,
        )
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    return selection_view(Settings.load())


def _model_keep(data: dict | None) -> dict[str, str] | None:
    if not isinstance(data, dict) or not isinstance(data.get("keep"), dict):
        return None
    return {str(key): str(value) for key, value in data["keep"].items()}


def _assert_model_cache_idle() -> None:
    """安装或生成还在写权重时，不删缓存。"""
    from pipeline.bootstrap import setup_status

    status = setup_status()
    if status.get("status") in {"installing", "checking"}:
        raise ValueError("正在安装或检查依赖，请等它结束后再清理模型。")
    if active_run_info().get("running"):
        raise ValueError("有生成任务在跑，请等它结束后再清理模型。")


def api_models_unused(data: dict | None = None, *, delete: bool = False) -> dict:
    """列出 hub 里的全部模型。``delete`` 为真时只删请求里点名的仓库。"""
    from pipeline.config import Settings
    from pipeline.model_cache import delete_selected_caches, list_hub_caches

    settings = Settings.load()
    keep = _model_keep(data)
    if not delete:
        return list_hub_caches(settings, keep=keep)
    if not (isinstance(data, dict) and data.get("confirm") is True):
        raise ValueError("清理模型需要确认")
    repos = data.get("repos")
    if not isinstance(repos, list) or not any(str(item).strip() for item in repos):
        raise ValueError("先选择要删除的模型")
    _assert_model_cache_idle()
    result = delete_selected_caches([str(item) for item in repos])
    if result["errors"] and result["freed_bytes"] == 0:
        detail = "；".join(f"{item['repo']}：{item['error']}" for item in result["errors"])
        raise ValueError(f"没能清理：{detail}")
    result.update(list_hub_caches(settings, keep=keep))
    return result


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
        # ValueError / FileNotFoundError 的文案已经是给用户看的，不再套异常类名
        if isinstance(exc, (ValueError, FileNotFoundError)):
            msg = str(exc).strip() or "请求无效"
        else:
            msg = f"{type(exc).__name__}: {exc}"
        self._json({"error": msg}, status)

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
        """只读地暴露项目目录下的产物（用户项目根与运行根）。"""
        from urllib.parse import unquote

        try:
            path = safe_path(unquote(rel))
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
            elif path == "/api/menubar":
                self._json(active_run_info())
            elif path == "/api/health":
                self._json({"ok": True, "pid": os.getpid(), "root": str(ROOT)})
            elif path == "/api/budget":
                self._json(budget_info())
            elif path == "/api/models":
                self._json(api_models())
            elif path == "/api/models/unused":
                self._json(api_models_unused())
            elif path == "/api/models/space":
                self._json(api_models_space())
            elif path == "/api/setup/status":
                from pipeline.bootstrap import setup_status

                # 只返回检查结果。下载要等用户点「安装依赖」。
                self._json(setup_status())
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
            elif path == "/api/voice-ref/upload":
                self._json(api_voice_ref_upload(data))
            elif path == "/api/storyboard/read":
                rel = str(data.get("path") or "")
                self._json(read_storyboard(rel))
            elif path == "/api/storyboard/write":
                rel = str(data.get("path") or "")
                write_storyboard(rel, data.get("storyboard") or {})
                self._json({"path": rel, "ok": True})
            elif path == "/api/storyboard/validate":
                sb_payload = data.get("storyboard")
                # novel_text 键存在则用网页正文（含空串）；不要回落读磁盘
                novel_text = data["novel_text"] if "novel_text" in data else None
                if novel_text is not None:
                    novel_text = str(novel_text)
                result = validate_storyboard(
                    str(data.get("path") or ""),
                    strict=bool(data.get("strict")),
                    storyboard=sb_payload if isinstance(sb_payload, dict) else None,
                    novel_text=novel_text,
                )
                if not result.get("ok"):
                    # 400：让前端走失败分支；error 给人看，其余字段仍带上
                    payload = {
                        **result,
                        "error": result.get("strict_error") or result.get("report") or "校验未通过",
                    }
                    self._json(payload, 400)
                else:
                    self._json(result)
            elif path == "/api/shots/rerun":
                self._json(self.rerun_shots(data))
            elif path == "/api/artifact/plan":
                self._json(self.artifact_plan(data))
            elif path in ("/api/artifact/rerun", "/api/queue/add"):
                self._json(self.queue_add(data))
            elif path == "/api/queue/remove":
                self._json(self.queue_remove(data))
            elif path == "/api/queue/clear":
                self._json(self.queue_clear(data))
            elif path in ("/api/tasks/start", "/api/queue/start"):
                self._json(self.queue_start(data))
            elif path == "/api/task/cancel":
                self._json(self.queue_remove(data))
            elif path == "/api/models":
                self._json(api_models_save(data))
            elif path == "/api/models/unused":
                self._json(api_models_unused(data, delete=bool(data.get("confirm"))))
            elif path == "/api/models/space":
                self._json(api_models_space(data))
            elif path == "/api/setup/ensure":
                from pipeline.bootstrap import ensure_async

                self._json(ensure_async(force=bool(data.get("force"))))
            elif path == "/api/setup/recheck":
                from pipeline.bootstrap import recheck_async

                self._json(recheck_async())
            elif path == "/api/setup/cancel":
                from pipeline.bootstrap import cancel_async

                self._json(cancel_async())
            elif path == "/api/shutdown":
                self._json(request_web_shutdown())
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError as exc:
            self._json_error(exc, 404)
        except ValueError as exc:
            # 含 JSONDecodeError：参数/分镜不合法，用 400 而不是 500
            self._json_error(exc, 400)
        except Exception as exc:
            self._json_error(exc)

    # -- API ---------------------------------------------------------------
    def api_state(self, query: str) -> dict:
        """选了项目 → 返回该项目的一切（待跑队列、历史任务、项目卡片…）。"""
        from urllib.parse import parse_qs, unquote

        from pipeline.run_queue import annotate_removable, load_queue

        params = parse_qs(query)
        name = unquote(params.get("project", [""])[0]).strip()
        queue = {"items": [], "updated_at": 0}
        if name:
            tasks = load_project_tasks(name)
            out = project_path_str(name)
            try:
                queue = annotate_removable(out, load_queue(out))
            except Exception:
                queue = load_queue(out)
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
            "queue": queue,
            "active_run": active_run_info(),
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
        with _START_LOCK:
            return self._create_task_locked(data)

    def _create_task_locked(self, data: dict) -> dict:
        _ensure_no_running_job()
        title = str(data.get("title") or "").strip()
        name = str(data.get("name") or "").strip() or _slugify(title)
        # 新任务一律落在用户项目根下的项目文件夹
        out = str(data.get("out") or "").strip() or project_path_str(_slugify(name) or name)
        novel = str(data.get("novel") or "").strip()
        storyboard = str(data.get("storyboard") or "").strip()
        only = str(data.get("only") or "").strip() or None
        force = bool(data.get("force"))
        stages = [s for s in (data.get("stages") or []) if s in STAGES]
        if not stages:
            stages = list(STAGES)
        # 前端勾选顺序不可靠；统一按流水线顺序传给 CLI（run --from 依赖它）
        stages = [s for s in STAGES if s in set(stages)]

        # 网页暂存：落到 work/webui/run_*，不写项目源 novel.txt；分镜装进运行态 storyboard.json
        sb_data = data.get("storyboard_data")
        has_novel_text = "novel_text" in data
        novel_text = data.get("novel_text") if has_novel_text else None
        if isinstance(sb_data, dict) or has_novel_text:
            sb_rel, novel_rel = materialize_web_draft(
                out,
                storyboard_data=sb_data if isinstance(sb_data, dict) else None,
                novel_text=None if novel_text is None else str(novel_text),
            )
            if sb_rel:
                storyboard = sb_rel
            if novel_rel:
                novel = novel_rel

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
        """停止本项目在跑的历史任务；待跑队列保留并复位为 pending。"""
        out = str(data.get("out") or "").strip()
        if not out:
            job = get_job(str(data.get("id") or ""))
            if not job:
                raise FileNotFoundError("task not found")
            out = job_out(job)
        return stop_pipeline_keep_queue(out)

    def rerun_shots(self, data: dict) -> dict:
        """对指定镜头重跑所选阶段（前端传什么阶段就跑什么）。"""
        with _START_LOCK:
            return self._rerun_shots_locked(data)

    def _rerun_shots_locked(self, data: dict) -> dict:
        _ensure_no_running_job()
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

    def artifact_plan(self, data: dict) -> dict:
        """单跑计划：前向/后向传递闭包，供确认对话框。"""
        from pipeline.artifact_plan import plan_rerun

        out = str(data.get("out") or "").strip()
        if not out:
            raise ValueError("没给输出目录")
        # 规划前先装网页暂存，使依赖图与当前编辑一致；不写项目源 novel.txt
        sb_data = data.get("storyboard_data")
        has_novel_text = "novel_text" in data
        if isinstance(sb_data, dict) or has_novel_text:
            materialize_web_draft(
                out,
                storyboard_data=sb_data if isinstance(sb_data, dict) else None,
                novel_text=None if not has_novel_text else str(data.get("novel_text") or ""),
            )
        kind = str(data.get("kind") or "").strip()
        nid = str(data.get("id") or "").strip()
        return plan_rerun(out, kind=kind, id=nid)

    def queue_add(self, data: dict) -> dict:
        """对话框确认 → 加入左边待跑队列（去重 + 补缺失前置）。"""
        from pipeline.run_queue import add_from_plan

        out = str(data.get("out") or "").strip()
        if not out:
            raise ValueError("没给输出目录")
        sb_data = data.get("storyboard_data")
        has_novel_text = "novel_text" in data
        if isinstance(sb_data, dict) or has_novel_text:
            materialize_web_draft(
                out,
                storyboard_data=sb_data if isinstance(sb_data, dict) else None,
                novel_text=None if not has_novel_text else str(data.get("novel_text") or ""),
            )
        result = add_from_plan(
            out,
            kind=str(data.get("kind") or "").strip(),
            id=str(data.get("id") or "").strip(),
            posts=data.get("posts") or [],
        )
        if not result.get("ok"):
            raise ValueError(result.get("error") or "无法加入队列")
        notify_change()
        return result

    def queue_remove(self, data: dict) -> dict:
        from pipeline.run_queue import remove_item

        out = str(data.get("out") or "").strip()
        key = str(data.get("key") or data.get("id") or "").strip()
        if not out or not key:
            raise ValueError("需要 out 与 key")
        result = remove_item(out, key)
        if not result.get("ok"):
            raise ValueError(result.get("error") or "无法删除")
        notify_change()
        return result

    def queue_clear(self, data: dict) -> dict:
        from pipeline.run_queue import clear_queue

        out = str(data.get("out") or "").strip()
        if not out:
            raise ValueError("没给输出目录")
        # 在跑时不允许清空？允许——用户可清队列，不影响已拉起的进程；或禁止
        result = clear_queue(out)
        notify_change()
        return result

    def queue_start(self, data: dict) -> dict:
        """点启动：按待跑队列拉起一条历史任务。"""
        with _START_LOCK:
            return self._queue_start_locked(data)

    def _queue_start_locked(self, data: dict) -> dict:
        _ensure_no_running_job()
        from pipeline.artifact_plan import clear_for_force_rerun, queue_block_reason
        from pipeline.run_queue import (
            annotate_removable,
            build_groups_from_queue,
            expand_missing_prereqs,
            load_queue,
            queue_image_parts,
            queue_only_shots,
            queue_portrait_ids,
            reset_statuses,
            save_queue,
        )

        out = str(data.get("out") or "").strip()
        if not out:
            raise ValueError("没给输出目录")
        _ensure_no_running_job()
        sb_data = data.get("storyboard_data")
        has_novel_text = "novel_text" in data
        novel_rel = None
        if isinstance(sb_data, dict) or has_novel_text:
            _, novel_rel = materialize_web_draft(
                out,
                storyboard_data=sb_data if isinstance(sb_data, dict) else None,
                novel_text=None if not has_novel_text else str(data.get("novel_text") or ""),
            )
        queue = load_queue(out)
        if not (queue.get("items") or []):
            raise ValueError("待跑队列为空；请先通过 ↻ 添加")
        blocked = queue_block_reason(out, list(queue.get("items") or []))
        if blocked:
            raise ValueError(blocked)
        expanded = expand_missing_prereqs(out, list(queue.get("items") or []))
        if [it.get("key") for it in expanded] != [it.get("key") for it in queue.get("items") or []]:
            queue["items"] = expanded
            save_queue(out, queue)
            notify_change()

        groups = build_groups_from_queue(out, queue)
        if not groups:
            raise ValueError("队列里没有可执行项（产物可能已齐全）")
        # 强制项 + 其后向闭包：未勾选的后置也会清掉，避免陈旧产物留盘
        force_items = [
            {"kind": it["kind"], "id": it.get("id") or ""}
            for it in queue["items"]
            if it.get("force")
        ]
        if force_items:
            clear_for_force_rerun(out, force_items)
        novel = novel_rel or str(data.get("novel") or "").strip()
        if not novel:
            project_novel = resolve(out) / "novel.txt"
            if project_novel.is_file() and project_novel.read_text(encoding="utf-8").strip():
                novel = cite_path(project_novel)
        cmd, log_path, stage_keys = build_artifact_groups_cmd(
            out=out, groups=groups, novel=novel or None
        )
        only = queue_only_shots(queue)
        characters = queue_portrait_ids(queue)
        image_parts = queue_image_parts(queue)
        n = len(queue["items"])
        labels = "、".join(it.get("label") or it["key"] for it in queue["items"][:4])
        if n > 4:
            labels += f" 等 {n} 项"
        job = Job(
            uuid.uuid4().hex[:12],
            name=f"启动 · {n} 项",
            novel=novel,
            out=out,
            stages=stage_keys,
            only=only,
            characters=characters,
            image_parts=image_parts,
            cmd=cmd,
            log_path=log_path,
            detail=labels,
            status="running",
        )
        reset_statuses(queue, "running")
        save_queue(out, queue)
        register_job(job)
        job.start()
        notify_change()
        snap = job.snapshot()
        snap["queue"] = annotate_removable(out, load_queue(out))
        snap["ok"] = True
        snap["message"] = f"已启动：{job.name}"
        return snap

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

        from urllib.parse import quote

        def _media_entry(file_path: Path, *, entry_id: str | None = None) -> dict:
            abs_path = str(file_path.resolve())
            url = "/media/" + quote(abs_path, safe="/")
            st = file_path.stat()
            item = {
                "path": abs_path,
                "url": url,
                "mtime": int(st.st_mtime),
                "size": st.st_size,
            }
            if entry_id is not None:
                item["id"] = entry_id
            return item

        def _scan(rel_dir: str, exts: tuple[str, ...]) -> list[dict]:
            d = artifacts_sub(base, rel_dir)
            if not d.is_dir():
                return []
            found = []
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    found.append(_media_entry(f, entry_id=f.stem))
            return found

        final_path = base / "final.mp4"
        final = _media_entry(final_path) if final_path.is_file() else None
        music = None
        for name in ("bgm.wav", "music.wav"):
            path = artifacts_sub(base, "audio", name)
            if not path.is_file():
                continue
            music = _media_entry(path, entry_id="bgm")
            break
        return {
            "frames": _scan("frames", (".png", ".jpg", ".jpeg", ".webp")),
            "clips": _scan("clips", (".mp4",)),
            "voices": _scan("audio/voice", (".wav",)),
            "characters": _scan("characters", (".png", ".jpg", ".jpeg", ".webp")),
            "music": music,
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
            # queued 也挂着：点启动后同一条 SSE 能接到 running；被移出列表则结束
            while job.status in {"queued", "running"} and get_job(task_id) is job:
                if job.status == "queued":
                    try:
                        time.sleep(1.0)
                    except Exception:
                        break
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    # 刚被启动：推一次快照再进 running 循环
                    if job.status == "running":
                        snap = job.tracker.snapshot()
                        if int(snap.get("seq") or 0) > last_id:
                            last_id = int(snap.get("seq") or 0)
                            self._sse_progress_event(snap)
                    continue
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


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pidfile() -> int:
    path = server_pid_path()
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def _health_ok(host: str, port: int) -> bool:
    try:
        with urlopen(f"http://{host}:{port}/api/health", timeout=1.0) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        return bool(data.get("ok"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        pass
    # 旧实例可能还没有 /api/health：根路径能开就算活着
    try:
        with urlopen(f"http://{host}:{port}/", timeout=1.0) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 500
    except (URLError, TimeoutError, OSError):
        return False


def _claim_port_or_exit(host: str, port: int) -> None:
    """端口预检：已有健康实例则提示；被占用则立刻失败，避免启动器盲等 45 秒。"""
    if not _port_in_use(host, port):
        return
    old_pid = _read_pidfile()
    if _health_ok(host, port):
        hint = f"（PID {old_pid}）" if old_pid and _pid_alive(old_pid) else ""
        raise SystemExit(
            f"控制台已在运行{hint}：http://{host}:{port}/ 。请用菜单栏打开网页，或先退出旧实例。"
        )
    if old_pid and _pid_alive(old_pid):
        raise SystemExit(
            f"端口 {port} 被占用（PID {old_pid}），但健康检查失败。"
            f"请退出旧实例（kill {old_pid}）或用 --port 换端口。"
        )
    raise SystemExit(
        f"端口 {port} 被占用（Address already in use）。请退出占用进程或用 --port 换端口。"
    )


def _write_pidfile() -> None:
    path = server_pid_path()
    try:
        support_dir().mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"写 pidfile 失败（可忽略）：{exc}", flush=True)
        return

    def _clear() -> None:
        try:
            if path.exists() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_clear)


def _stop_all_jobs_for_exit() -> None:
    with _LOCK:
        jobs = list(_JOBS.values())
    for job in jobs:
        try:
            if getattr(job, "status", None) == "running":
                job.stop()
        except Exception:
            pass


def cleanup_project_on_exit() -> None:
    """菜单栏退出 / Ctrl+C：停安装、停任务，再清本仓库相关进程。"""
    try:
        from pipeline.bootstrap import cancel_async

        cancel_async()
    except Exception as exc:
        print(f"停止安装失败（可忽略）：{exc}", flush=True)
    try:
        _stop_all_jobs_for_exit()
    except Exception as exc:
        print(f"停止任务失败（可忽略）：{exc}", flush=True)
    try:
        from pipeline.process_guard import shutdown_project_processes

        killed = shutdown_project_processes(keep_pids=(os.getpid(),), include_web=False)
        if killed:
            print(f"已结束相关进程：{', '.join(str(pid) for pid in killed)}", flush=True)
    except Exception as exc:
        print(f"清理相关进程失败（可忽略）：{exc}", flush=True)


_SHUTDOWN_ONCE = threading.Event()


def request_web_shutdown() -> dict:
    """给菜单栏「退出」用：先回包，再在后台收尾并结束本进程。"""
    import signal

    print(f"[webui] /api/shutdown 被调用 pid={os.getpid()}", flush=True)

    if _SHUTDOWN_ONCE.is_set():
        return {"ok": True, "status": "stopping"}
    _SHUTDOWN_ONCE.set()

    def _run() -> None:
        time.sleep(0.15)
        try:
            cleanup_project_on_exit()
        finally:
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                os._exit(0)

    threading.Thread(target=_run, name="webui-shutdown", daemon=True).start()
    return {"ok": True, "status": "stopping"}


def serve(host: str, port: int) -> None:
    # 脱离启动器/终端进程组，避免 launcher 退出或外层 shell 收尾时把后端连坐杀掉；
    # 否则菜单栏会因 checkParent 发现父进程已死而跟着退出，看起来像「助手自己挂了」。
    try:
        os.setsid()
    except OSError:
        try:
            os.setpgrp()
        except OSError:
            pass
    _claim_port_or_exit(host, port)
    try:
        from pipeline.process_guard import KIND_DOWNLOAD, KIND_SETUP, sweep_stale

        killed = sweep_stale(kinds=(KIND_DOWNLOAD, KIND_SETUP), keep_pids=(os.getpid(),))
        if killed:
            print(f"已清理残留进程：{', '.join(str(pid) for pid in killed)}", flush=True)
    except Exception as exc:
        print(f"启动前清理残留进程失败（可忽略）：{exc}", flush=True)
    try:
        moved = migrate_repo_projects()
        if moved:
            print(f"已迁移项目到 {display_path(projects_dir())}：{', '.join(moved)}", flush=True)
    except Exception as exc:
        print(f"项目迁移跳过：{exc}", flush=True)
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
    # 打开即检查。缺依赖只提示，等用户点「安装依赖」再下载。
    try:
        from pipeline.bootstrap import recheck_async

        report = recheck_async()
        if report.get("ready"):
            print("依赖与模型：已就绪", flush=True)
        else:
            print(report.get("message") or "依赖未齐，打开网页后点「安装依赖」", flush=True)
    except Exception as exc:
        print(f"依赖检查失败（网页仍可开）：{exc}", flush=True)
    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48 or "Address already in use" in str(exc):
            raise SystemExit(
                f"端口 {port} 被占用（Address already in use）。请退出旧实例或用 --port 换端口。"
            ) from exc
        raise
    _write_pidfile()
    url = f"http://{host}:{port}"
    print(f"小说成片控制台：{url}  （菜单栏退出 / Ctrl+C 停止）", flush=True)

    from pipeline.menubar import start_menubar

    menubar = start_menubar(url)
    if menubar is not None:
        # HTTP 在后台线程；主线程盯菜单栏。用户点「退出」会 SIGTERM 本进程。
        http_thread = threading.Thread(target=server.serve_forever, name="webui-http", daemon=True)
        http_thread.start()
        print("菜单栏已就绪：打开网页 / 退出", flush=True)

        stopping = False

        def _sigterm(_signum, _frame) -> None:
            nonlocal stopping
            stopping = True
            mb_pid = getattr(menubar, "pid", None) if menubar is not None else None
            mb_alive = (
                menubar.poll() is None if menubar is not None else False
            )
            print(
                f"[webui] 收到 SIGTERM pid={os.getpid()} "
                f"menubar_pid={mb_pid} menubar_alive={mb_alive}",
                flush=True,
            )
            raise KeyboardInterrupt

        prev_term = None
        try:
            import signal

            prev_term = signal.signal(signal.SIGTERM, _sigterm)
            while not stopping:
                try:
                    code = menubar.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    continue
                if code is None:
                    continue
                # helper 自己挂了（非用户退出）：重启，别把网页服务拖死
                if stopping:
                    break
                print(
                    f"菜单栏助手已退出（code={code}），正在重启…",
                    flush=True,
                )
                menubar = start_menubar(url)
                if menubar is None:
                    print("菜单栏助手无法重启，网页服务继续运行（Ctrl+C 停止）", flush=True)
                    while not stopping:
                        time.sleep(0.5)
                    break
        except KeyboardInterrupt:
            stopping = True
        finally:
            if not _SHUTDOWN_ONCE.is_set():
                cleanup_project_on_exit()
            if prev_term is not None:
                try:
                    import signal

                    signal.signal(signal.SIGTERM, prev_term)
                except Exception:
                    pass
            if menubar is not None and menubar.poll() is None:
                menubar.terminate()
                try:
                    menubar.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    menubar.kill()
            server.shutdown()
            server.server_close()
        return

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if not _SHUTDOWN_ONCE.is_set():
            cleanup_project_on_exit()
        server.server_close()


def cmd_web(args: argparse.Namespace) -> None:
    serve(args.host, args.port)
