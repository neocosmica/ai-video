"""本仓库相关进程的查找与清理。

``download-one`` 等子进程开了独立 session，杀网页进程带不走它们。
启动后台任务前、菜单栏点退出时，都按命令行扫一遍并清掉。
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from typing import Any, Iterable

from pipeline.util import ROOT

KIND_DOWNLOAD = "download"
KIND_WEB = "web"
KIND_PRODUCE = "produce"
KIND_SETUP = "setup"
KIND_MENUBAR = "menubar"

_ROOT = str(ROOT.resolve())
_PRODUCE_CMD = re.compile(r"(?:python\S*\s+-m\s+pipeline|pipeline)\s+(produce|run)\b")
_PS_LINE = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$")


def _classify(cmd: str) -> str | None:
    text = cmd or ""
    if "pipeline.bootstrap download-one" in text or (
        "pipeline.bootstrap" in text and "download-one" in text
    ):
        return KIND_DOWNLOAD
    if re.search(r"(?:python\S*\s+-m\s+pipeline|pipeline)\s+web\b", text):
        return KIND_WEB
    if _PRODUCE_CMD.search(text):
        return KIND_PRODUCE
    if "menubar-helper" in text and (
        _ROOT in text or "api/menubar" in text or "AI短视频" in text or "novel-video" in text
    ):
        return KIND_MENUBAR
    if "scripts/setup.sh" in text and (_ROOT in text or "novel-video" in text):
        return KIND_SETUP
    if _ROOT in text and re.search(r"-m\s+pipeline\b", text):
        if " download-one" in text:
            return KIND_DOWNLOAD
        if re.search(r"\bweb\b", text):
            return KIND_WEB
        if re.search(r"\b(produce|run)\b", text):
            return KIND_PRODUCE
    return None


def list_project_processes() -> list[dict[str, Any]]:
    """扫本机，找出和本仓库有关的进程。"""
    try:
        raw = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    found: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _PS_LINE.match(line)
        if not match:
            continue
        pid, ppid, pgid, cmd = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            match.group(4),
        )
        kind = _classify(cmd)
        if not kind:
            continue
        found.append(
            {
                "pid": pid,
                "ppid": ppid,
                "pgid": pgid,
                "cmd": cmd,
                "kind": kind,
            }
        )
    return found


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


def _send(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    except OSError:
        return


def _send_pg(pgid: int, sig: int) -> None:
    if pgid <= 1:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    except OSError:
        return


def kill_processes(
    procs: Iterable[dict[str, Any]],
    *,
    keep_pids: Iterable[int] = (),
    grace_s: float = 1.5,
) -> list[int]:
    """先 SIGTERM 进程组，再 SIGKILL。返回实际动手杀过的 pid。"""
    keep = {int(pid) for pid in keep_pids if int(pid) > 0}
    keep.add(os.getpid())
    targets = []
    for proc in procs:
        pid = int(proc.get("pid") or 0)
        if pid <= 0 or pid in keep or not _pid_alive(pid):
            continue
        targets.append(proc)
    if not targets:
        return []

    for proc in targets:
        pid = int(proc["pid"])
        pgid = int(proc.get("pgid") or 0)
        if pgid > 1:
            _send_pg(pgid, signal.SIGTERM)
        _send(pid, signal.SIGTERM)

    deadline = time.time() + max(0.2, grace_s)
    while time.time() < deadline:
        if not any(_pid_alive(int(proc["pid"])) for proc in targets):
            break
        time.sleep(0.1)

    for proc in targets:
        pid = int(proc["pid"])
        if not _pid_alive(pid):
            continue
        pgid = int(proc.get("pgid") or 0)
        if pgid > 1:
            _send_pg(pgid, signal.SIGKILL)
        _send(pid, signal.SIGKILL)
    return [int(proc["pid"]) for proc in targets]


def sweep_stale(
    *,
    kinds: Iterable[str] = (KIND_DOWNLOAD,),
    keep_pids: Iterable[int] = (),
) -> list[int]:
    """启动后台任务前：清掉仍活着的同类残留。"""
    wanted = set(kinds)
    procs = [proc for proc in list_project_processes() if proc.get("kind") in wanted]
    return kill_processes(procs, keep_pids=keep_pids)


def shutdown_project_processes(
    *,
    keep_pids: Iterable[int] = (),
    include_web: bool = False,
) -> list[int]:
    """退出时清掉本仓库相关进程。默认不动别的网页实例。"""
    skip = set() if include_web else {KIND_WEB}
    procs = [proc for proc in list_project_processes() if proc.get("kind") not in skip]
    return kill_processes(procs, keep_pids=keep_pids, grace_s=2.0)


def selftest_process_guard() -> None:
    root = _ROOT
    assert _classify(f"python -m pipeline.bootstrap download-one org/x") == KIND_DOWNLOAD
    assert _classify(f"{root}/.venv/bin/python -m pipeline web --port 8790") == KIND_WEB
    assert _classify("uv run python -m pipeline produce --project demo") == KIND_PRODUCE
    assert _classify(f"{root}/scripts/macos/menubar-helper --url http://127.0.0.1:8790/") == KIND_MENUBAR
    assert _classify("/bin/zsh /tmp/other/scripts/setup.sh") is None
    assert _classify(f"/bin/zsh {root}/scripts/setup.sh") == KIND_SETUP
    assert _classify("ollama serve") is None
