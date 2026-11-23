"""本地网页控制台：填写小说片段、起流水线任务、看进度、重跑单个阶段。

零新增依赖，只用 Python 标准库：

    ./scripts/run.sh web --port 8790          # 浏览器打开 http://127.0.0.1:8790

职责边界（与 AGENTS.md 一致）：本地不加载聊天 LLM，分镜仍由人工/Agent 写好
`input/<片名>_storyboard.json`；本界面负责填写小说、维护分镜、起任务、
看实时进度、对失败步骤单独重跑。

任务记录写入 `output/webui/state.json`，重启服务不丢历史。
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

from pipeline.util import ROOT, resolve

WEB_DIR = ROOT / "webui"
STATE_DIR = ROOT / "output" / "webui"
STATE_PATH = STATE_DIR / "state.json"
LOG_TAIL_LINES = 4000  # 内存里每任务保留的日志行数
HISTORY = 30  # state.json 里最多保留多少条任务记录

# 与 pipeline.run.STAGES 保持一致；单独在这里声明是为了 import 轻量
STAGES = ("screenplay", "images", "audio", "videos", "music", "assemble")
STAGE_LABELS = {
    "screenplay": "校验分镜",
    "images": "静帧生成",
    "audio": "对白配音",
    "videos": "视频生成",
    "music": "配乐生成",
    "assemble": "合成混音",
}
# 总进度条里各阶段的权重（videos 最重）
STAGE_WEIGHTS = {
    "screenplay": 0.02,
    "images": 0.18,
    "audio": 0.10,
    "videos": 0.60,
    "music": 0.05,
    "assemble": 0.05,
}

# 子进程终端条（tqdm）解析：`s003 a2v:  45%|████▌     | 5/11 [00:16<00:20,  2.30s/it]`
_TQDM_LINE = re.compile(
    r"^(?P<desc>.+?):\s+\d+%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<]*)<(?P<eta>[^\],]*)"
)
# 进度条 desc → 正在处理的条目 id
_DESC_ITEMS = (
    (re.compile(r"^(?P<id>\S+)\s+(?:a2v|i2v)$"), "{id}"),
    (re.compile(r"^静帧\s+(?P<id>\S+)"), "{id}"),
    (re.compile(r"^空镜\s+(?P<id>\S+)"), "{id}"),
    (re.compile(r"^肖像\s+(?P<name>.+)$"), "肖像 {name}"),
    (re.compile(r"^(?P<name>配乐)$"), "配乐"),
)
# 行首日志 → 条目开始 / 完成 / 失败
_ITEM_START = (
    (re.compile(r"^角色参考图\s+(.+?)\s*->"), "images"),
    (re.compile(r"^空镜\s+(\S+)\s*->"), "images"),
    (re.compile(r"^分镜静帧\s+(\S+)"), "images"),
    (re.compile(r"^对白\s+(\S+)\s*·"), "audio"),
    (re.compile(r"^整理\s+(\S+)"), "assemble"),
)
_ITEM_DONE = (
    re.compile(r"^已有镜头\s+(\S+)，跳过"),
)
_ITEM_FAIL = (
    re.compile(r"^镜头\s+(\S+)\s+失败"),
    re.compile(r"^跳过\s+(\S+)[:：]"),
)


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------


class ProgressTracker:
    """解析子进程输出的 tqdm 行，维护每镜进度 + 阶段内合成进度。

    进度来源与终端一致（``pipeline.progress._ShotBar`` 看到的同一批行）：
    - 视频每镜：LTX 两段 Denoising 合并条（8+3 步），解析 ``desc: n/total``；
    - 图像每张 / TTS 每句 / 配乐：只标「进行中」，模型不吐可解析的步数；
    - 失败/被杀：标记该镜失败。
    """

    def __init__(self, job: "Job") -> None:
        self.job = job
        self.lock = threading.Lock()
        # item_id -> {"label", "pct", "step", "total", "eta_s", "status", "desc"}
        self.items: dict[str, dict] = {}
        self.stage_pct: dict[str, float] = {}
        self.current_desc: str | None = None

    # -- 主入口：每收到一行调用一次 ---------------------------------------
    def feed(self, line: str, stage: str | None) -> None:
        with self.lock:
            self._feed_locked(line, stage)

    def _feed_locked(self, line: str, stage: str | None) -> None:
        m = _TQDM_LINE.match(line.strip())
        if m:
            desc = m.group("desc").strip()
            n, total = int(m.group("n")), int(m.group("total"))
            eta_s = _parse_hms(m.group("eta"))
            item_id, label = self._item_for_desc(desc)
            if item_id is not None:
                item = self.items.setdefault(
                    item_id,
                    {"label": label, "pct": 0.0, "step": 0, "total": total, "eta_s": None, "status": "running"},
                )
                item.update(
                    {
                        "label": label,
                        "step": n,
                        "total": total,
                        "pct": round(100.0 * n / max(1, total), 1),
                        "eta_s": eta_s,
                        "status": "running" if n < total else "done",
                    }
                )
                self.current_desc = desc
            return
        # 阶段级整行完成消息
        for pattern in _ITEM_DONE:
            mm = pattern.match(line.strip())
            if mm:
                item_id = f"shot:{mm.group(1)}"
                item = self.items.setdefault(
                    item_id,
                    {"label": mm.group(1), "pct": 0.0, "step": 0, "total": 1, "eta_s": None, "status": "running"},
                )
                item["status"] = "done"
                item["pct"] = 100.0
                return
        for pattern in _ITEM_FAIL:
            mm = pattern.match(line.strip())
            if mm:
                item_id = f"shot:{mm.group(1)}"
                item = self.items.setdefault(
                    item_id,
                    {"label": mm.group(1), "pct": 0.0, "step": 0, "total": 1, "eta_s": None, "status": "running"},
                )
                item["status"] = "failed"
                item["pct"] = item.get("pct") or 0.0
                return
        for pattern, stage_key in _ITEM_START:
            mm = pattern.match(line.strip())
            if mm:
                name = mm.group(1)
                # assemble 的「整理 s001」与视频阶段的镜头条目同名，加前缀避免混同
                label = f"整理 {name}" if stage_key == "assemble" else name
                item_id = f"{stage_key}:{mm.group(1)}"
                self.items.setdefault(
                    item_id,
                    {"label": label, "pct": 0.0, "step": 0, "total": 1, "eta_s": None, "status": "running"},
                )
                return

    def finalize(self, *, success: bool) -> None:
        """任务结束时落定所有条目：否则「整理 sXXX」这类只有开始、没有完成行的
        条目会永远停在 running，任务结束后还挂着一条假进度。"""
        with self.lock:
            for item in self.items.values():
                if item["status"] == "running":
                    item["status"] = "done" if success else "failed"
                    item["pct"] = 100.0
                    item["eta_s"] = None

    def _item_for_desc(self, desc: str) -> tuple[str | None, str]:
        for pattern, template in _DESC_ITEMS:
            mm = pattern.match(desc)
            if mm:
                label = template.format(**{k: mm.group(k) for k in mm.groupdict()})
                return f"shot:{label}" if template.startswith("{id}") else f"misc:{label}", label
        return None, desc

    # -- 快照 --------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            items = [dict(v, id=k) for k, v in self.items.items()]
            return {
                "items": items,
                "stage_pct": dict(self.stage_pct),
                "current": self.current_desc,
            }

    def set_stage_pct(self, stage: str, pct: float) -> None:
        with self.lock:
            self.stage_pct[stage] = round(max(0.0, min(100.0, pct)), 1)


def _parse_hms(text: str) -> float | None:
    """`00:20` / `01:02:03` / `20s` → 秒。"""
    text = (text or "").strip()
    m = re.match(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$", text)
    if not m:
        return None
    h, mi, s = (g or "0" for g in m.groups())
    return int(h) * 3600 + int(mi) * 60 + float(s)


class _NullTracker:
    """历史任务占位：snapshot() 的 progress 恒为空。"""

    def snapshot(self) -> dict:
        return {"items": [], "stage_pct": {}, "current": None}

    def feed(self, *_args) -> None:
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
        self.tracker = ProgressTracker(self)
        self.returncode: int | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None

    # -- 序列化 ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
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
                "current_stage": next((s["key"] for s in stages if s["status"] == "running"), None),
                "progress": self.tracker.snapshot(),
                "returncode": self.returncode,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log_path": str(self.log_path),
                "media_base": self.out,
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

            def on_line(text: str) -> None:
                self.append_log(text)
                parse_stage_progress(text, self)
                self.tracker.feed(text, self.current_stage_safe())

            def on_start(pid: int) -> None:
                self.pid = pid

            code = run_tee(self.cmd, log=self.log_path, env=env, cwd=ROOT, on_line=on_line, on_start=on_start)
            self.returncode = code
            self.status = "success" if code == 0 else "failed"
            self.tracker.finalize(success=(code == 0))
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


def notify_change() -> None:
    with _LOCK:
        subs = list(_CHANGE_QUEUES)
    for q in subs:
        try:
            q.put_nowait("change")
        except Exception:
            pass


def get_job(task_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(task_id)


def all_jobs() -> list[Job]:
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)


def clear_finished_tasks() -> dict:
    """清掉非运行中的历史任务记录（产物文件不动）；运行中的保留。"""
    with _LOCK:
        running = [j for j in _JOBS.values() if j.status == "running"]
        _JOBS.clear()
        for job in running:
            _JOBS[job.id] = job
    _persist()
    return {"ok": True, "kept": len(running)}


def register_job(job: Job) -> None:
    with _LOCK:
        _JOBS[job.id] = job
    _persist()


def _persist() -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tasks = {j.id: j.snapshot() for j in all_jobs()}
        if len(tasks) > HISTORY:
            keep = sorted(tasks.items(), key=lambda kv: kv[1].get("created_at", 0), reverse=True)[:HISTORY]
            tasks = dict(keep)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        pass


def restore_state() -> None:
    """重启服务后，把 state.json 里的历史任务标记为中断。"""
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    for task_id, snap in (data.get("tasks") or {}).items():
        if snap.get("status") == "running":
            snap["status"] = "interrupted"
            snap["error"] = "服务重启，任务中断；请重跑需要的阶段"
            for stage in snap.get("stages") or []:
                if stage.get("status") == "running":
                    stage["status"] = "failed"
        try:
            created = float(snap.get("created_at") or 0)
        except (TypeError, ValueError):
            created = 0.0
        with _LOCK:
            _JOBS[task_id] = _RestoredJob(snap, created)


class _RestoredJob(Job):
    """历史任务：只读快照，不能再 start。"""

    def __init__(self, snap: dict, created_at: float) -> None:
        self.__dict__.update(
            {
                "id": snap.get("id") or uuid.uuid4().hex[:12],
                "name": snap.get("name") or "",
                "novel": snap.get("novel") or "",
                "out": snap.get("out") or "",
                "only": snap.get("only"),
                "detail": snap.get("detail") or "",
                "status": snap.get("status") or "interrupted",
                "stages": [s.get("key") for s in (snap.get("stages") or [])],
                "stage_status": {s.get("key"): s.get("status", "pending") for s in (snap.get("stages") or [])},
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
                "pid": None,
                "tracker": _NullTracker(),
            }
        )

    def start(self) -> None:  # pragma: no cover
        raise RuntimeError("历史任务不能重新启动，请新建任务")


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
    log_path = out_dir / "logs" / f"webui-{time.strftime('%Y%m%d-%H%M%S')}.log"
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
    notify_change()


# ---------------------------------------------------------------------------
# 项目文件
# ---------------------------------------------------------------------------


def list_projects() -> dict:
    novels: list[dict] = []
    storyboards: list[dict] = []
    input_dir = ROOT / "input"
    if input_dir.is_dir():
        for path in sorted(input_dir.glob("*.txt")):
            novels.append({"name": path.stem, "path": str(path.relative_to(ROOT)).replace("\\", "/")})
        for path in sorted(input_dir.glob("*.json")):
            storyboards.append({"name": path.stem, "path": str(path.relative_to(ROOT)).replace("\\", "/")})
    outputs: list[dict] = []
    output_dir = ROOT / "output"
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir()):
            if not child.is_dir() or child.name == "webui":
                continue
            outputs.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(ROOT)).replace("\\", "/"),
                    "has_final": (child / "final.mp4").exists(),
                    "has_storyboard": (child / "storyboard.json").exists(),
                }
            )
    return {"novels": novels, "storyboards": storyboards, "outputs": outputs}


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
        stem = re.sub(r"_?storyboard$", "", src.stem)
        for cand_name in (f"input/{stem}_novel.txt", f"input/{stem}.txt"):
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
        """只读地暴露项目目录下的产物（output/ 与 input/）。"""
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
                self._json(
                    {
                        "projects": list_projects(),
                        "tasks": [j.snapshot() for j in all_jobs()],
                        "server_time": time.time(),
                    }
                )
            elif path == "/api/projects":
                pass  # included in /api/state
            elif path.startswith("/media/"):
                self._serve_media(path[len("/media/"):])
            elif path == "/api/budget":
                self._json(budget_info())
            elif path == "/api/events":
                self._stream_events()
            elif path.startswith("/api/log/"):
                self._api_log(path.rsplit("/", 1)[-1], query)
            elif path.startswith("/api/log-stream/"):
                self._stream_job_log(path.rsplit("/", 1)[-1])
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
    def create_task(self, data: dict) -> dict:
        title = str(data.get("title") or "").strip()
        name = str(data.get("name") or "").strip() or _slugify(title)
        out = str(data.get("out") or "").strip() or f"output/{_slugify(name)}"
        novel = str(data.get("novel") or "").strip()
        storyboard = str(data.get("storyboard") or "").strip()
        only = str(data.get("only") or "").strip() or None
        force = bool(data.get("force"))
        stages = [s for s in (data.get("stages") or []) if s in STAGES]
        if not stages:
            stages = list(STAGES)
        # 前端勾选顺序不可靠；统一按流水线顺序传给 CLI（run --from 依赖它）
        stages = [s for s in STAGES if s in set(stages)]

        # 粘贴的小说片段先落盘
        novel_text = str(data.get("novel_text") or "").strip()
        if novel_text:
            novel = novel or f"input/{name}_novel.txt"
            write_text(novel, novel_text.rstrip() + "\n")
        # 粘贴的分镜 JSON 先落盘
        sb_data = data.get("storyboard_data")
        if isinstance(sb_data, dict) and sb_data.get("shots"):
            storyboard = storyboard or f"input/{name}_storyboard.json"
            write_storyboard(storyboard, sb_data)

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
            d = base / rel_dir
            if not d.is_dir():
                return []
            found = []
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    rel = f"{out_rel.rstrip('/')}/{rel_dir}/{f.name}".replace("\\", "/")
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

        return {
            "frames": _scan("frames", (".png", ".jpg", ".jpeg", ".webp")),
            "clips": _scan("clips", (".mp4",)),
            "voices": _scan("audio/voice", (".wav",)),
            "final": (base / "final.mp4").exists(),
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
        self._json({"lines": lines, "total": total, "status": job.status})

    # -- SSE ---------------------------------------------------------------
    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse_event(self, payload: str) -> None:
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_events(self) -> None:
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
    restore_state()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    if not (WEB_DIR / "index.html").exists():
        raise SystemExit(f"缺少网页文件：{WEB_DIR / 'index.html'}")
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
