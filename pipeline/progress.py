"""把子进程输出实时打到终端，并复制进日志。

LTX、MusicGen 自己会画 tqdm。stdout 若只进日志文件，进度条就只写在文件里。
这里给子进程一个伪终端，让 tqdm 按交互终端来画，再把字节原样抄到当前终端。
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
from pathlib import Path


def run_tee(
    cmd: list[str],
    *,
    log: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    """运行命令。stdout/stderr 同时出现在终端和 log 里，返回退出码。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy() if env is None else dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    master, slave = pty.openpty()
    try:
        _copy_winsize(slave)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=slave,
            env=child_env,
            cwd=str(cwd) if cwd is not None else None,
            close_fds=True,
        )
    except Exception:
        os.close(slave)
        os.close(master)
        raise
    os.close(slave)
    _set_nonblocking(master)
    try:
        with log.open("wb") as handle:
            header = ""
            if cwd is not None:
                header += f"cwd={cwd}\n"
            header += " ".join(cmd) + "\n\n"
            handle.write(header.encode("utf-8", errors="replace"))
            handle.flush()
            _pump(master, proc, handle)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        os.close(master)
    return proc.returncode if proc.returncode is not None else proc.wait()


def _copy_winsize(slave: int) -> None:
    if not sys.stdout.isatty():
        return
    try:
        import fcntl
        import struct
        import termios

        packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        fcntl.ioctl(slave, termios.TIOCSWINSZ, packed)
    except OSError:
        return


def _set_nonblocking(fd: int) -> None:
    import fcntl

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _pump(master: int, proc: subprocess.Popen, handle) -> None:
    while True:
        status = _read_available(master, handle)
        if status == "data":
            continue
        if status == "eof":
            return
        if proc.poll() is not None:
            while _read_available(master, handle) == "data":
                pass
            return
        select.select([master], [], [], 0.2)


def _read_available(master: int, handle) -> str:
    try:
        chunk = os.read(master, 4096)
    except BlockingIOError:
        return "wait"
    except OSError:
        return "eof"
    if not chunk:
        return "eof"
    handle.write(chunk)
    handle.flush()
    out = getattr(sys.stdout, "buffer", None)
    if out is not None:
        out.write(chunk)
        out.flush()
    else:
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    return "data"
