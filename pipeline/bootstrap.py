"""本机依赖与模型就绪检查 / 补装。

供 ``scripts/setup.sh``、网页启动与 ``doctor`` 共用同一份清单。
分镜语言模型（Qwen3-4B / mlx-lm）也在这里，避免只装进网页却忘了 setup。
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pipeline.mirrors import apply_mirrors, child_env
from pipeline.util import ROOT

apply_mirrors()

def weight_specs() -> list[dict[str, Any]]:
    """当前 config 选中的各阶段权重。没选中的档位不下载。"""
    from pipeline.backends.catalog import download_specs
    from pipeline.config import Settings

    return download_specs(Settings.load())


class InstallCancelled(Exception):
    """用户点了停止安装。"""


@dataclass
class SetupState:
    status: str = "unknown"  # unknown|ready|checking|installing|failed|need_install|need_space
    pct: float = 0.0
    message: str = ""
    error: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    disk: dict[str, Any] | None = None
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "ready": self.status == "ready",
                "pct": round(float(self.pct), 1),
                "message": self.message,
                "error": self.error,
                "items": list(self.items),
                "disk": dict(self.disk) if isinstance(self.disk, dict) else None,
                "updated_at": self.updated_at,
            }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.updated_at = time.time()


_STATE = SetupState()
_WORKER: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()
_CANCEL = threading.Event()
_CHILD: Any = None
_CHILD_LOCK = threading.Lock()


def _set_child(child: Any) -> None:
    global _CHILD
    with _CHILD_LOCK:
        _CHILD = child


def _clear_child(child: Any) -> None:
    global _CHILD
    with _CHILD_LOCK:
        if _CHILD is child:
            _CHILD = None


def _stop_popen(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def _kill_child() -> None:
    with _CHILD_LOCK:
        child = _CHILD
    if child is None:
        return
    try:
        if isinstance(child, subprocess.Popen):
            _stop_popen(child)
        elif hasattr(child, "is_alive"):
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
    except Exception:
        traceback.print_exc()
    finally:
        _clear_child(child)


def _raise_if_cancelled() -> None:
    if _CANCEL.is_set():
        raise InstallCancelled()


def _mark_stopped(items: list[dict[str, Any]] | None = None) -> None:
    report_items = list(items) if items is not None else list(_STATE.items)
    if not report_items:
        report_items = probe(refresh_state=False)["items"]
    missing_n = len([it for it in report_items if not it["ok"]])
    message = (
        f"安装已停止。还缺 {missing_n} 项。点「安装依赖」可继续。"
        if missing_n
        else "安装已停止。"
    )
    _STATE.update(
        status="need_install",
        pct=0.0,
        message=message,
        error="",
        items=report_items,
        disk=None,
    )


def _weight_ok(spec: dict[str, Any]) -> tuple[bool, str]:
    from pipeline.hf_cache import resolve_snapshot, snapshot_ready

    snap = resolve_snapshot(str(spec["repo"]))
    return snapshot_ready(
        snap,
        required_files=tuple(spec.get("required_files") or ()),
        weight_globs=tuple(spec.get("weight_globs") or ()),
    )


def _pkg_items() -> list[dict[str, Any]]:
    from pipeline.config import Settings
    from pipeline.runtime import ltx_cli, main_venv_path, musicgen_venv_path

    settings = Settings.load()
    items: list[dict[str, Any]] = []

    def add(iid: str, label: str, ok: bool, detail: str = "") -> None:
        items.append({"id": iid, "kind": "pkg", "label": label, "ok": ok, "detail": detail})

    main_py = main_venv_path() / "bin" / "python"
    add("venv-main", "主环境 Python", main_py.exists(), str(main_py) if main_py.exists() else "请跑 setup")

    mlx_ok = False
    mlx_detail = ""
    try:
        import mlx_lm  # noqa: F401

        mlx_ok = True
        mlx_detail = getattr(mlx_lm, "__version__", "installed")
    except Exception as exc:
        mlx_detail = str(exc)
    add("mlx-lm", "分镜依赖 mlx-lm", mlx_ok, mlx_detail)

    try:
        import mflux  # noqa: F401

        add("mflux", "文生图依赖 mflux", True, "installed")
    except Exception as exc:
        add("mflux", "文生图依赖 mflux", False, str(exc))

    from pipeline.media_bins import binary_runnable

    # Settings.load 已做过包内→系统回退；这里再试跑，堵住「文件在但不可执行」假阳性
    ff_ok, ff_detail = binary_runnable(settings.ffmpeg)
    fp_ok, fp_detail = binary_runnable(settings.ffprobe)
    if ff_ok and fp_ok:
        note = str((settings.budget_detail or {}).get("ffmpeg") or ff_detail)
        add("ffmpeg", "ffmpeg", True, note)
    else:
        add("ffmpeg", "ffmpeg", False, ff_detail if not ff_ok else fp_detail)

    try:
        cli = ltx_cli(settings.ltx_dir)
        add("ltx-cli", "LTX CLI", True, str(cli))
    except SystemExit as exc:
        add("ltx-cli", "LTX CLI", False, str(exc))

    gen_py = settings.musicgen_dir / "generate.py"
    mg_py = musicgen_venv_path() / "bin" / "python"
    add(
        "musicgen",
        "MusicGen 环境",
        gen_py.is_file() and mg_py.exists(),
        str(gen_py) if gen_py.is_file() else "缺 generate.py",
    )

    marker = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"
    blocks = (
        settings.ltx_dir
        / "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/blocks.py"
    )
    patched = blocks.is_file() and marker in blocks.read_text(encoding="utf-8", errors="ignore")
    add("ltx-patch", "LTX Gemma 分阶段补丁", patched, "已打" if patched else "未打（请 setup）")
    return items


def _weight_items() -> list[dict[str, Any]]:
    from pipeline.disk_space import format_bytes

    items: list[dict[str, Any]] = []
    for spec in weight_specs():
        ok, detail = _weight_ok(spec)
        size = int(spec.get("approx_bytes") or 0)
        label = spec["label"]
        if size:
            label = f"{label} · {format_bytes(size)}"
        items.append(
            {
                "id": f"weight-{spec['id']}",
                "kind": "weight",
                "label": label,
                "ok": ok,
                "detail": detail,
                "repo": spec["repo"],
            }
        )
    return items


def probe(*, refresh_state: bool = True) -> dict[str, Any]:
    """快速检查（不下载）。"""
    items = _pkg_items() + _weight_items()
    missing = [it for it in items if not it["ok"]]
    ready = not missing
    payload = {
        "status": "ready" if ready else "need_install",
        "ready": ready,
        "pct": 100.0 if ready else 0.0,
        "message": "依赖与模型已就绪" if ready else f"还缺 {len(missing)} 项。点「安装依赖」后再下载。",
        "error": "",
        "items": items,
        "missing": [it["id"] for it in missing],
    }
    if refresh_state and _STATE.status not in {"installing", "checking"}:
        _STATE.update(
            status="ready" if ready else "need_install",
            pct=payload["pct"],
            message=payload["message"],
            error="",
            items=items,
            disk=None,
        )
    return payload


def setup_status() -> dict[str, Any]:
    snap = _STATE.snapshot()
    if snap["status"] in {"installing", "checking", "need_space"}:
        return snap
    if snap["status"] in {"ready", "need_install", "failed"} and snap["items"]:
        return snap
    return probe(refresh_state=True)


def _uv_bin() -> str:
    return shutil.which("uv") or str(Path.home() / ".local/bin/uv")


def _run_uv_sync(set_msg: Callable[[str, float], None]) -> None:
    uv = _uv_bin()
    if not Path(uv).exists() and not shutil.which("uv"):
        raise RuntimeError("找不到 uv，请先安装或跑 scripts/setup.sh")
    set_msg("正在同步 Python 依赖（含 mlx-lm）…", max(_STATE.pct, 2.0))
    env = child_env()
    cache = Path.home() / "Library" / "Caches" / "novel-video" / "venvs" / "main"
    env["UV_PROJECT_ENVIRONMENT"] = str(cache)
    cmd = [uv, "sync", "--index-url", env.get("UV_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")]
    # 尽量沿用 setup 的 Python 选择；不强制版本以免打断已有环境
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _set_child(proc)
    chunks: list[str] = []
    try:
        assert proc.stdout is not None
        while True:
            _raise_if_cancelled()
            line = proc.stdout.readline()
            if line:
                chunks.append(line)
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        code = proc.wait()
    finally:
        _clear_child(proc)
    if code != 0:
        err = "".join(chunks).strip() or f"uv sync 失败（{code}）"
        raise RuntimeError(err[-2000:])
    # 确认分镜依赖
    try:
        import importlib

        importlib.invalidate_caches()
        import mlx_lm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"uv sync 后仍缺 mlx-lm：{exc}") from exc


def _missing_weight_specs() -> list[dict[str, Any]]:
    return [spec for spec in weight_specs() if not _weight_ok(spec)[0]]


def _hold_for_disk(specs: list[dict[str, Any]], items: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """空间不够就停在下载前，并留下网页能直接显示的说明。"""
    if not specs:
        return None
    from pipeline.disk_space import plan_weight_space

    plan = plan_weight_space(specs)
    if plan["ok"]:
        return None
    _STATE.update(
        status="need_space",
        pct=0.0,
        message=plan["message"],
        error="",
        items=list(items) if items is not None else list(_STATE.items),
        disk=plan,
    )
    return setup_status()


_TQDM_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
_TQDM_TAIL = re.compile(r"(\?it/s\]|\?B/s\]|it/s\]|B/s\]|s/it\]|s/file\])")


def terminal_progress_line(line: str) -> str:
    """终端最后一行原样用。警告若粘在进度条后面，只留进度条这一截。"""
    text = (line or "").strip()
    if not text:
        return ""
    if "%" not in text:
        return text
    match = _TQDM_TAIL.search(text)
    if match and match.end() < len(text):
        return text[: match.end()].strip()
    return text


def pct_from_tqdm_line(line: str) -> float | None:
    """从终端 tqdm 行里抠百分比，只给进度条用；文案仍用原行。"""
    match = _TQDM_PCT.search(line or "")
    if not match:
        return None
    return min(100.0, max(0.0, float(match.group(1))))


def _download_one_cli(argv: list[str]) -> int:
    """子进程入口：在伪终端里跑 snapshot_download，让 tqdm 原样刷进度。"""
    apply_mirrors()
    from huggingface_hub import snapshot_download

    if not argv:
        print("usage: python -m pipeline.bootstrap download-one <repo> [--allow PATTERN]...", file=sys.stderr)
        return 2
    repo = argv[0]
    patterns: list[str] = []
    i = 1
    while i < len(argv):
        if argv[i] == "--allow" and i + 1 < len(argv):
            patterns.append(argv[i + 1])
            i += 2
            continue
        print(f"unknown arg: {argv[i]}", file=sys.stderr)
        return 2
    kwargs: dict[str, Any] = {"repo_id": repo}
    if patterns:
        kwargs["allow_patterns"] = patterns
    snapshot_download(**kwargs)
    return 0


def overall_download_pct(local_pct: float, *, index: int, count: int) -> float:
    """多项下载时，进度条按总进度走；单项内的百分比折进当前这一份。"""
    n = max(1, int(count))
    i = min(max(1, int(index)), n)
    local = min(100.0, max(0.0, float(local_pct))) / 100.0
    return ((i - 1) + local) / n * 100.0


def _download_weight(
    spec: dict[str, Any],
    set_msg: Callable[[str, float], None],
    *,
    index: int,
    count: int,
) -> None:
    import pty

    from pipeline.progress import _prepare_pty, _ProgressRelay

    label = str(spec["label"])
    where = f"（{index}/{count}）" if count > 1 else ""
    task = f"{label}{where}"

    def publish(detail: str, local_pct: float) -> None:
        text = (detail or "").strip() or "准备下载…"
        set_msg(f"{task} · {text}", overall_download_pct(local_pct, index=index, count=count))

    publish("准备下载…", 0.0)
    _raise_if_cancelled()
    try:
        from pipeline.process_guard import KIND_DOWNLOAD, sweep_stale

        sweep_stale(kinds=(KIND_DOWNLOAD,), keep_pids=(os.getpid(),))
    except Exception:
        pass

    cmd = [sys.executable, "-m", "pipeline.bootstrap", "download-one", str(spec["repo"])]
    for pattern in spec.get("allow_patterns") or ():
        cmd.extend(["--allow", str(pattern)])
    env = child_env()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    env.pop("TQDM_DISABLE", None)

    master, slave = pty.openpty()
    try:
        _prepare_pty(slave)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=slave,
            cwd=str(ROOT),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        os.close(slave)
        os.close(master)
        raise
    os.close(slave)
    _set_child(proc)
    relay = _ProgressRelay()
    last_pct = 0.0

    def on_line(line: str) -> None:
        nonlocal last_pct
        text = terminal_progress_line(line)
        if not text:
            return
        found = pct_from_tqdm_line(text)
        if found is not None:
            last_pct = found
        publish(text, last_pct)

    try:
        while True:
            _raise_if_cancelled()
            ready, _, _ = select.select([master], [], [], 0.4)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    for line in relay.feed(chunk):
                        on_line(line)
                elif proc.poll() is not None:
                    break
            if proc.poll() is not None:
                while True:
                    more, _, _ = select.select([master], [], [], 0)
                    if not more:
                        break
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    for line in relay.feed(chunk):
                        on_line(line)
                break
        for line in relay.finish():
            on_line(line)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            _stop_popen(proc)
        _clear_child(proc)
        try:
            os.close(master)
        except OSError:
            pass
    _raise_if_cancelled()
    if code != 0:
        raise RuntimeError(f"{label} 下载失败（退出码 {code}）")
    ok, detail = _weight_ok(spec)
    if not ok:
        raise RuntimeError(f"{label} 下载后仍不可用：{detail}")
    publish("已就绪", 100.0)


def _install_loop(*, force: bool = False) -> None:
    def set_msg(message: str, pct: float) -> None:
        _STATE.update(status="installing", message=message, pct=min(99.9, max(0.0, pct)), error="")

    items: list[dict[str, Any]] = []
    try:
        _raise_if_cancelled()
        _STATE.update(status="checking", message="正在检查依赖与模型…", pct=1.0, error="")
        report = probe(refresh_state=False)
        items = report["items"]
        _STATE.update(items=items)
        missing = [it for it in items if not it["ok"]]
        # 自动检查说齐了就停。手动「安装依赖」带 force：判断可能失误，仍按清单再拉一遍。
        if not missing and not force:
            _STATE.update(
                status="ready", pct=100.0, message="依赖与模型已就绪", items=items, error="", disk=None
            )
            return

        # 缺权重时先看磁盘。setup.sh 和 snapshot_download 都会往缓存里写，空间不够就不要启动。
        if _hold_for_disk(_missing_weight_specs(), items):
            return

        if not missing and force:
            set_msg("检查显示已就绪，仍重新拉取依赖与模型（已有文件会跳过）…", 3.0)
        else:
            set_msg(f"发现缺 {len(missing)} 项，开始补装…", 3.0)
        _raise_if_cancelled()

        pkg_missing = [it for it in missing if it["kind"] == "pkg"]
        # 包/环境缺：先 uv sync；更重的 LTX/MusicGen/ffmpeg 仍建议完整 setup
        need_full_setup = any(
            it["id"] in {"ltx-cli", "musicgen", "ffmpeg", "ltx-patch", "venv-main"}
            for it in pkg_missing
        )
        if need_full_setup:
            set_msg("环境不完整，正在跑 scripts/setup.sh（含依赖与模型）…", 5.0)
            env = child_env()
            proc = subprocess.Popen(
                ["/bin/zsh", str(ROOT / "scripts" / "setup.sh")],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _set_child(proc)
            try:
                assert proc.stdout is not None
                lines = 0
                while True:
                    _raise_if_cancelled()
                    line = proc.stdout.readline()
                    if line:
                        lines += 1
                        text = line.strip()
                        if text:
                            # setup 很长：用行数做粗进度，卡在 5–90
                            pct = min(90.0, 5.0 + lines * 0.35)
                            set_msg(text[:160], pct)
                        continue
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
                code = proc.wait()
            finally:
                _clear_child(proc)
            if code != 0:
                raise RuntimeError(f"setup.sh 退出码 {code}")
        else:
            if force or any(it["id"] in {"mlx-lm", "mflux"} for it in pkg_missing):
                _run_uv_sync(set_msg)

            # force 会再问一遍仓库，已齐的文件会被跳过；真正要下的只计还没齐的。
            weight_missing = list(weight_specs()) if force else _missing_weight_specs()
            if _hold_for_disk(_missing_weight_specs(), items):
                return
            n = max(1, len(weight_missing))
            for i, spec in enumerate(weight_missing):
                _raise_if_cancelled()
                _download_weight(spec, set_msg, index=i + 1, count=n)

        final = probe(refresh_state=False)
        if not final["ready"]:
            still = ", ".join(it["label"] for it in final["items"] if not it["ok"])
            raise RuntimeError(f"补装后仍缺：{still}")
        _STATE.update(
            status="ready",
            pct=100.0,
            message="依赖与模型已就绪",
            items=final["items"],
            error="",
            disk=None,
        )
    except InstallCancelled:
        _mark_stopped(items or None)
    except Exception as exc:
        _STATE.update(
            status="failed",
            message="依赖安装失败",
            error=f"{exc}",
            pct=max(_STATE.pct, 0.0),
        )
        traceback.print_exc()


def _begin_install(report: dict[str, Any], *, force: bool) -> dict[str, Any]:
    global _WORKER
    _CANCEL.clear()
    try:
        from pipeline.process_guard import KIND_DOWNLOAD, KIND_SETUP, sweep_stale

        killed = sweep_stale(kinds=(KIND_DOWNLOAD, KIND_SETUP), keep_pids=(os.getpid(),))
        if killed:
            print(f"已清理残留安装进程：{', '.join(str(pid) for pid in killed)}", flush=True)
    except Exception as exc:
        print(f"清理残留安装进程失败（可忽略）：{exc}", flush=True)
    _STATE.update(
        status="installing",
        message="准备安装…",
        pct=1.0,
        error="",
        disk=None,
        items=report["items"],
    )
    _WORKER = threading.Thread(
        target=_install_loop,
        kwargs={"force": force},
        name="novel-video-bootstrap",
        daemon=True,
    )
    _WORKER.start()
    return setup_status()


def cancel_async() -> dict[str, Any]:
    """停止正在进行的安装。已下完的文件保留，下次可续下。"""
    _CANCEL.set()
    _kill_child()
    try:
        from pipeline.process_guard import KIND_DOWNLOAD, KIND_SETUP, sweep_stale

        sweep_stale(kinds=(KIND_DOWNLOAD, KIND_SETUP), keep_pids=(os.getpid(),))
    except Exception:
        pass
    worker = None
    with _WORKER_LOCK:
        worker = _WORKER
    if worker and worker.is_alive():
        worker.join(timeout=4)
    with _WORKER_LOCK:
        still = bool(_WORKER and _WORKER.is_alive())
    if still:
        _STATE.update(status="installing", message="正在停止安装…", error="")
        return setup_status()
    snap = setup_status()
    if snap["status"] in {"installing", "checking"}:
        _mark_stopped()
    return setup_status()


def ensure_async(*, force: bool = False) -> dict[str, Any]:
    """后台补装；已在装则返回当前进度。"""
    with _WORKER_LOCK:
        if _WORKER and _WORKER.is_alive():
            return setup_status()
        report = probe(refresh_state=False)
        if not force and report["ready"]:
            _STATE.update(
                status="ready",
                pct=100.0,
                message=report["message"],
                error="",
                items=report["items"],
                disk=None,
            )
            return setup_status()
        held = _hold_for_disk(_missing_weight_specs(), report["items"])
        if held is not None:
            return held
        return _begin_install(report, force=force)


def recheck_async() -> dict[str, Any]:
    """按刚保存的模型重新检查。不沿用上一次的就绪或空间不足结论。"""
    with _WORKER_LOCK:
        if _WORKER and _WORKER.is_alive():
            return setup_status()
        report = probe(refresh_state=False)
        if report["ready"]:
            _STATE.update(
                status="ready",
                pct=100.0,
                message="依赖与模型已就绪",
                error="",
                items=report["items"],
                disk=None,
            )
            return setup_status()
        held = _hold_for_disk(_missing_weight_specs(), report["items"])
        if held is not None:
            return held
        missing_n = len([it for it in report["items"] if not it["ok"]])
        _STATE.update(
            status="need_install",
            pct=0.0,
            message=f"还缺 {missing_n} 项。点「安装依赖」后再下载。",
            error="",
            items=report["items"],
            disk=None,
        )
        return setup_status()


def install_weights_cli() -> int:
    """供 setup.sh 调用：按当前各阶段模型拉齐权重（含分镜 LLM）。"""
    apply_mirrors()
    from huggingface_hub import snapshot_download

    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    from pipeline.disk_space import plan_weight_space

    plan = plan_weight_space(_missing_weight_specs())
    if not plan["ok"]:
        print(plan["message"], flush=True)
        return 1
    for spec in weight_specs():
        print(f"下载 {spec['label']} ← {spec['repo']}", flush=True)
        kwargs: dict[str, Any] = {"repo_id": spec["repo"]}
        if spec.get("allow_patterns"):
            kwargs["allow_patterns"] = spec["allow_patterns"]
        snapshot_download(**kwargs)
        ok, detail = _weight_ok(spec)
        print(f"  → {'OK' if ok else '缺'} {detail}", flush=True)
        if not ok:
            return 1
    print("weights ready", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m pipeline.bootstrap [probe|ensure|install-weights|download-one]")
        return 0
    cmd = args[0]
    if cmd == "probe":
        print(json.dumps(probe(), ensure_ascii=False, indent=2))
        return 0
    if cmd == "ensure":
        ensure_async(force=True)
        while True:
            st = setup_status()
            print(f"[{st['status']}] {st['pct']}% {st['message']}", flush=True)
            if st["status"] in {"ready", "failed", "need_space"}:
                return 0 if st["status"] == "ready" else 1
            time.sleep(1.0)
    if cmd == "install-weights":
        return install_weights_cli()
    if cmd == "download-one":
        return _download_one_cli(args[1:])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


def selftest_recheck() -> None:
    """保存后的重检只出结论。空间不够或仍缺依赖时都不能启动下载。"""
    import pipeline.disk_space as disk_space

    saved = _STATE.snapshot()
    old_worker = _WORKER
    old_probe = probe
    old_missing = _missing_weight_specs
    old_plan = disk_space.plan_weight_space
    old_begin = _begin_install
    started = {"n": 0}
    missing = [{"id": "weight-x", "kind": "weight", "label": "测试模型", "ok": False, "detail": "", "repo": "org/x"}]

    def fake_probe(*, refresh_state: bool = True) -> dict[str, Any]:
        return {
            "status": "need_install",
            "ready": False,
            "pct": 0.0,
            "message": "还缺 1 项",
            "error": "",
            "items": missing,
            "missing": ["weight-x"],
        }

    def no_begin(report: dict[str, Any], *, force: bool) -> dict[str, Any]:
        started["n"] += 1
        return {"status": "installing", "ready": False, "started": True, "force": force}

    try:
        globals()["_WORKER"] = None
        globals()["probe"] = fake_probe
        globals()["_missing_weight_specs"] = lambda: [{"id": "x", "label": "测试模型", "repo": "org/x", "approx_bytes": 10**12}]
        globals()["_begin_install"] = no_begin
        disk_space.plan_weight_space = lambda specs, **kwargs: {
            "ok": False,
            "message": "磁盘空间不够，下载尚未开始",
            "items": list(specs),
            "need_bytes": 10**12,
            "free_bytes": 0,
        }
        short = recheck_async()
        assert short["status"] == "need_space", short
        assert started["n"] == 0, started
        assert "磁盘空间不够" in short["message"]

        disk_space.plan_weight_space = lambda specs, **kwargs: {
            "ok": True,
            "message": "",
            "items": [],
            "need_bytes": 0,
            "free_bytes": 10**12,
        }
        _STATE.update(status="ready", message="旧的就绪结论", disk=None, items=[])
        waiting = recheck_async()
        assert waiting["status"] == "need_install", waiting
        assert started["n"] == 0
        assert "安装依赖" in waiting["message"]

        def ready_probe(*, refresh_state: bool = True) -> dict[str, Any]:
            item = {**missing[0], "ok": True}
            return {
                "status": "ready",
                "ready": True,
                "pct": 100.0,
                "message": "依赖与模型已就绪",
                "error": "",
                "items": [item],
                "missing": [],
            }

        globals()["probe"] = ready_probe
        _STATE.update(status="need_space", message="旧的空间不足", disk={"ok": False}, items=missing)
        ready = recheck_async()
        assert ready["status"] == "ready", ready
        assert ready["ready"] is True
        assert ready.get("disk") in (None, {})
        assert started["n"] == 0
    finally:
        globals()["probe"] = old_probe
        globals()["_missing_weight_specs"] = old_missing
        globals()["_begin_install"] = old_begin
        globals()["_WORKER"] = old_worker
        disk_space.plan_weight_space = old_plan
        _STATE.update(
            status=saved["status"],
            pct=saved["pct"],
            message=saved["message"],
            error=saved["error"],
            items=saved["items"],
            disk=saved["disk"],
        )


def selftest_download_progress() -> None:
    """进度条走总进度；下面一行前缀任务名，细节用终端原行。"""
    line = "Fetching 12 files:  45%|████      | 5/12 [00:12<00:15,  2.10s/it]"
    assert pct_from_tqdm_line(line) == 45.0
    assert pct_from_tqdm_line("Downloading bytes: 12.5%|█| 100M/800M") == 12.5
    assert pct_from_tqdm_line("还没有百分比") is None
    glued = line + "You have used a large amount of egress over the last 30 days."
    assert terminal_progress_line(glued) == line
    assert terminal_progress_line("Reconstructing:   0%| | 10.5MB / 29.8GB") == "Reconstructing:   0%| | 10.5MB / 29.8GB"
    assert overall_download_pct(0, index=1, count=2) == 0.0
    assert overall_download_pct(100, index=1, count=2) == 50.0
    assert overall_download_pct(50, index=2, count=2) == 75.0
    assert overall_download_pct(100, index=1, count=1) == 100.0


def selftest_cancel() -> None:
    """停止安装要立刻记下取消，并杀掉正在跑的子进程。"""
    saved = _STATE.snapshot()
    old_worker = _WORKER
    old_child = None
    with _CHILD_LOCK:
        old_child = _CHILD
    _CANCEL.clear()
    try:
        globals()["_WORKER"] = None
        _set_child(None)
        stopped = cancel_async()
        assert stopped["status"] in {"ready", "need_install", "need_space", "failed", "unknown"}, stopped

        proc = subprocess.Popen(
            ["/bin/zsh", "-c", "sleep 30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _set_child(proc)
        _CANCEL.set()
        _kill_child()
        assert proc.poll() is not None
        _mark_stopped([{"id": "x", "ok": False, "label": "测试"}])
        snap = setup_status()
        assert snap["status"] == "need_install", snap
        assert "安装已停止" in snap["message"]
    finally:
        _CANCEL.clear()
        globals()["_WORKER"] = old_worker
        with _CHILD_LOCK:
            globals()["_CHILD"] = old_child
        _STATE.update(
            status=saved["status"],
            pct=saved["pct"],
            message=saved["message"],
            error=saved["error"],
            items=saved["items"],
            disk=saved["disk"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
