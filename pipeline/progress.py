"""把子进程输出实时打到终端，并复制进日志。

LTX、MusicGen 用 tqdm 在同一行上回车刷新。伪终端若再设成非阻塞，
macOS 会把这个标志传到子进程，进度那一行经常写不出来。
这里按行把每次刷新写到终端和日志，这样不等整段结束也能看到进度。
"""

from __future__ import annotations

import codecs
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_MIN_GAP = 0.25


class _ProgressRelay:
    """把 tqdm 的回车刷新收成一行一行。普通换行立刻输出。"""

    def __init__(self) -> None:
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""
        self._held = ""
        self._last = ""
        self._last_t = 0.0
        self._last_was_bar = False

    def feed(self, data: bytes) -> bytes:
        out: list[str] = []
        text = self._dec.decode(data)
        for ch in text:
            if ch == "\r":
                self._arm()
                self._take(out, force=False)
                continue
            if ch == "\n":
                self._arm()
                self._take(out, force=True)
                continue
            self._buf += ch
        return "".join(out).encode("utf-8")

    def poll(self) -> bytes:
        out: list[str] = []
        self._take(out, force=False)
        self._take_buf(out, force=False)
        return "".join(out).encode("utf-8")

    def finish(self) -> bytes:
        out: list[str] = []
        tail = self._dec.decode(b"", final=True)
        self._buf += tail
        self._arm()
        self._take(out, force=True)
        return "".join(out).encode("utf-8")

    def _arm(self) -> None:
        line = _ANSI.sub("", self._buf).strip()
        self._buf = ""
        if line:
            self._held = line

    def _take(self, out: list[str], *, force: bool) -> None:
        if not self._held or self._held == self._last:
            self._held = ""
            return
        now = time.monotonic()
        if not force and self._last_was_bar and now - self._last_t < _MIN_GAP:
            return
        out.append(self._held + "\n")
        self._last = self._held
        self._last_t = now
        self._last_was_bar = not force
        self._held = ""

    def _take_buf(self, out: list[str], *, force: bool) -> None:
        """当前行还没遇到回车时，也把它当作一条进度发出去。"""
        line = _ANSI.sub("", self._buf).strip()
        if not line or line == self._last:
            return
        now = time.monotonic()
        if not force and self._last_was_bar and now - self._last_t < _MIN_GAP:
            return
        out.append(line + "\n")
        self._last = line
        self._last_t = now
        self._last_was_bar = True


class _LiveStream:
    """给本进程的 tqdm 用：每次刷新都换成新的一行。"""

    def __init__(self, raw) -> None:
        self.raw = raw
        self.relay = _ProgressRelay()

    def write(self, data: str) -> int:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        self._emit(self.relay.feed(payload))
        self._emit(self.relay.poll())
        return len(data)

    def flush(self) -> None:
        self._emit(self.relay.poll())
        flush = getattr(self.raw, "flush", None)
        if flush is not None:
            flush()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.raw.fileno()

    def _emit(self, payload: bytes) -> None:
        if not payload:
            return
        text = payload.decode("utf-8", errors="replace")
        self.raw.write(text)
        flush = getattr(self.raw, "flush", None)
        if flush is not None:
            flush()


_BARS_INSTALLED = False


def install_live_bars() -> None:
    """本进程里的 tqdm（文生图、对白）每次刷新都另起一行。"""
    global _BARS_INSTALLED
    if _BARS_INSTALLED:
        return
    from tqdm import std as tqdm_std

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (OSError, ValueError):
                pass

    original = tqdm_std.tqdm.__init__

    def __init__(self, *args, **kwargs):
        file = kwargs.get("file", None)
        if file is None:
            kwargs["file"] = _LiveStream(sys.stderr)
        elif not isinstance(file, _LiveStream) and file in (sys.stderr, sys.stdout):
            kwargs["file"] = _LiveStream(file)
        kwargs["miniters"] = 1
        interval = kwargs.get("mininterval", 0.1)
        if interval is None or interval > 0.3:
            kwargs["mininterval"] = 0.2
        if kwargs.get("ncols") is None:
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
            kwargs["ncols"] = min(100, max(40, columns))
        original(self, *args, **kwargs)

    tqdm_std.tqdm.__init__ = __init__
    _BARS_INSTALLED = True


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
        _prepare_pty(slave)
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


def _prepare_pty(slave: int) -> None:
    """给子进程一个有宽度的终端。宽度为 0 时 tqdm 画不出进度条。"""
    import fcntl
    import struct
    import termios

    rows, cols = 40, 80
    if sys.stdout.isatty():
        try:
            packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
            got_rows, got_cols, _, _ = struct.unpack("HHHH", packed)
            if got_rows and got_cols:
                rows, cols = got_rows, got_cols
        except OSError:
            pass
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        return
    try:
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
    except termios.error:
        return


def _pump(master: int, proc: subprocess.Popen, handle) -> None:
    relay = _ProgressRelay()
    while True:
        _emit(relay.poll(), handle)
        ready, _, _ = select.select([master], [], [], 0.2)
        if not ready:
            if proc.poll() is not None:
                _drain(master, relay, handle)
                return
            continue
        try:
            chunk = os.read(master, 65536)
        except OSError:
            if proc.poll() is not None:
                _emit(relay.finish(), handle)
                return
            continue
        if not chunk:
            _emit(relay.finish(), handle)
            return
        _emit(relay.feed(chunk), handle)


def _drain(master: int, relay: _ProgressRelay, handle) -> None:
    while True:
        ready, _, _ = select.select([master], [], [], 0)
        if not ready:
            break
        try:
            chunk = os.read(master, 65536)
        except OSError:
            break
        if not chunk:
            break
        _emit(relay.feed(chunk), handle)
    _emit(relay.finish(), handle)


def _emit(payload: bytes, handle) -> None:
    if not payload:
        return
    handle.write(payload)
    handle.flush()
    # 先清掉文本层里还没写出的 print，再写原始字节，避免进度插到旧输出前面。
    sys.stdout.flush()
    out = getattr(sys.stdout, "buffer", None)
    if out is not None:
        out.write(payload)
        out.flush()
    else:
        sys.stdout.write(payload.decode("utf-8", errors="replace"))
        sys.stdout.flush()
