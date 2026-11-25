"""子进程输出进日志；终端只保留必要的进度条。

视频每个分镜把 LTX 两段 Denoising（以及其它阶段条）合成一条进度。
本机有 TTY 时，本进程 tqdm 仍用同行刷新，避免刷屏。
"""

from __future__ import annotations

import codecs
import os
import re
import select
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Denoising:  12%|…| 1/8 [00:02<00:18,  2.65s/it]
_TQDM = re.compile(
    r"^(?P<desc>.+?):\s+(?P<pct>\d+)%.*? (?P<n>\d+)/(?P<total>\d+)\b"
)
_PHASE_START = re.compile(r"^\[(?P<label>[^\]]+)\]\s*\.\.\.\s*$")
_ERRORISH = re.compile(
    r"^(Traceback\b|\w*Error:|\w*Exception:|SystemExit:|FATAL)",
    re.IGNORECASE,
)


class _ProgressRelay:
    """把 PTY 字节拆成完整行（\\r / \\n 都算一行结束）。"""

    def __init__(self) -> None:
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""

    def feed(self, data: bytes) -> list[str]:
        lines: list[str] = []
        for ch in self._dec.decode(data):
            if ch in "\r\n":
                line = _ANSI.sub("", self._buf).strip()
                self._buf = ""
                if line:
                    lines.append(line)
                continue
            self._buf += ch
        return lines

    def finish(self) -> list[str]:
        self._buf += self._dec.decode(b"", final=True)
        line = _ANSI.sub("", self._buf).strip()
        self._buf = ""
        return [line] if line else []


class StageMerge:
    """把连续多段 n/total 进度合成一条绝对进度。

    给了 ``plan``（各段预期步数，例如 LTX distilled 的 8 + 3）就按步数加权，
    分母从一开始固定为 ``sum(plan)``、段与段首尾相接：第一段跑完停在 8/11，
    第二段从 8/11 接着走到 11/11，不会先冲 100% 再掉回来。
    没给 ``plan`` 时退回旧行为（遇到新的一段才把上一段并入偏移），分母会随之变大。
    """

    def __init__(self, plan: Sequence[int] | None = None) -> None:
        self.plan = [max(1, int(step)) for step in plan] if plan else []
        self.segments = 0
        self.offset = 0
        self.last_n = 0
        self.last_total = 0
        self.seen = False
        self.n = 0
        self.total = sum(self.plan)

    def _new_segment(self, n: int, total: int) -> bool:
        """同一段时间里 n 只会涨；倒退或从 0 重新数就是新的一段。"""
        if n < self.last_n:
            return True
        return self.last_n > 0 and total != self.last_total and n == 0

    def update(self, n: int, total: int) -> tuple[int, int]:
        if total <= 0:
            return self.n, self.total
        if self.seen and self._new_segment(n, total):
            self.segments += 1
            self.offset += self.last_total
        self.seen = True
        self.last_n = n
        self.last_total = total
        if self.plan:
            # 按计划步数给每段划一段固定区间；计划外的额外段吃掉剩下的区间
            self.total = sum(self.plan)
            base = sum(self.plan[: self.segments])
            span = self.plan[self.segments] if self.segments < len(self.plan) else self.total - base
            step = base + round(min(1.0, n / total) * max(0, span))
            self.n = max(self.n, min(step, self.total))
            return self.n, self.total
        self.n = self.offset + n
        self.total = self.offset + total
        return self.n, self.total


class _ShotBar:
    """终端上只画一条合并后的进度；细节只进日志。

    ``plan`` 是各段预期步数（LTX distilled 为 8 + 3），给了分母就从头固定；
    ``desc`` 限定只吃哪种 tqdm（默认全吃）。日志里加载权重等杂条不参与进度。
    """

    def __init__(
        self,
        label: str,
        *,
        plan: Sequence[int] | None = None,
        desc: str | None = None,
    ) -> None:
        from tqdm import tqdm

        self.merge = StageMerge(plan)
        self.desc = desc
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        self.bar = tqdm(
            total=max(1, self.merge.total),
            desc=label,
            unit="step",
            file=sys.stderr,
            dynamic_ncols=False,
            ncols=min(100, max(40, cols)),
            miniters=1,
            mininterval=0.2,
            leave=True,
        )
        self.bar.set_postfix_str("准备", refresh=True)

    def _owns(self, line: str) -> bool:
        if not self.desc:
            return True
        match = _TQDM.match(line)
        return bool(match) and match.group("desc").strip().startswith(self.desc)

    def handle(self, line: str) -> None:
        if _ERRORISH.match(line) or line.startswith("  File "):
            from tqdm import tqdm

            tqdm.write(line, file=sys.stderr)
            return
        match = _TQDM.match(line)
        if match and self._owns(line):
            n = int(match.group("n"))
            total = int(match.group("total"))
            done, overall = self.merge.update(n, total)
            if self.bar.total != overall:
                self.bar.total = max(1, overall)
            self.bar.n = min(done, self.bar.total)
            self.bar.set_postfix_str("", refresh=False)
            self.bar.refresh()
            return
        phase = _PHASE_START.match(line)
        if phase:
            label = phase.group("label").strip()
            if label.startswith("estimate"):
                return
            short = label.split("(")[0].strip()
            if len(short) > 18:
                short = short[:18]
            self.bar.set_postfix_str(short, refresh=True)

    def close(self, *, success: bool = True) -> None:
        # 真跑完才补满；失败/被杀不要假装 100%
        if success and self.merge.seen:
            self.bar.n = self.bar.total
        self.bar.set_postfix_str("", refresh=False)
        self.bar.close()


class _LiveStream:
    """非 TTY 时把 tqdm 回车刷新改成换行，方便落日志。"""

    def __init__(self, raw) -> None:
        self.raw = raw
        self._buf = ""

    def write(self, data: str) -> int:
        text = data if isinstance(data, str) else data.decode("utf-8", errors="replace")
        for ch in text:
            if ch == "\r":
                line = _ANSI.sub("", self._buf).rstrip()
                self._buf = ""
                if line:
                    self.raw.write(line + "\n")
                continue
            if ch == "\n":
                line = _ANSI.sub("", self._buf).rstrip()
                self._buf = ""
                self.raw.write((line + "\n") if line else "\n")
                continue
            self._buf += ch
        flush = getattr(self.raw, "flush", None)
        if flush is not None:
            flush()
        return len(data)

    def flush(self) -> None:
        if self._buf:
            line = _ANSI.sub("", self._buf).rstrip()
            self._buf = ""
            if line:
                self.raw.write(line + "\n")
        flush = getattr(self.raw, "flush", None)
        if flush is not None:
            flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.raw.fileno()


_BARS_INSTALLED = False


def install_live_bars() -> None:
    """TTY：同行刷新；非 TTY：改成换行，避免进度条在日志里叠成一团。"""
    global _BARS_INSTALLED
    if _BARS_INSTALLED:
        return
    from tqdm import std as tqdm_std

    tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    original = tqdm_std.tqdm.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("miniters", 1)
        if kwargs.get("mininterval", 0.1) is None or kwargs.get("mininterval", 0.1) > 0.3:
            kwargs["mininterval"] = 0.2
        if not tty:
            file = kwargs.get("file")
            if file is None:
                kwargs["file"] = _LiveStream(sys.stderr)
            elif not isinstance(file, _LiveStream) and file in (sys.stderr, sys.stdout):
                kwargs["file"] = _LiveStream(file)
        elif kwargs.get("ncols") is None:
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
    progress: str | None = None,
    plan: Sequence[int] | None = None,
    desc: str | None = None,
    on_line=None,
    on_start=None,
) -> int:
    """运行命令。完整输出进 log；若给了 progress，终端只显示一条合并进度条。

    ``plan``：各段步数（分母一开始就固定）；``desc``：只统计这种 tqdm。
    ``on_start(pid)``：子进程起来后回调一次；``on_line(line)``：每收到一行
    去掉 ANSI 后的完整行回调一次（webui 用它推进阶段游标）。
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy() if env is None else dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    use_pty = hasattr(os, "openpty") and sys.platform != "win32"
    if use_pty:
        try:
            import pty  # Windows 无此模块，但上面已用 use_pty 拦住

            master, slave = pty.openpty()
        except OSError:
            # 某些容器/受限环境没有可用 PTY；退回普通管道
            use_pty = False
    if not use_pty:
        return _run_tee_plain(cmd, log=log, env=child_env, cwd=cwd, on_line=on_line, on_start=on_start)
    try:
        _prepare_pty(slave)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=slave,
            env=child_env,
            cwd=str(cwd) if cwd is not None else None,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        os.close(slave)
        os.close(master)
        raise
    if on_start is not None:
        try:
            on_start(proc.pid)
        except Exception:
            pass
    os.close(slave)
    shot: _ShotBar | None = _ShotBar(progress, plan=plan, desc=desc) if progress else None
    try:
        with log.open("wb") as handle:
            header = ""
            if cwd is not None:
                header += f"cwd={cwd}\n"
            header += " ".join(cmd) + "\n\n"
            handle.write(header.encode("utf-8", errors="replace"))
            handle.flush()
            _pump(master, proc, handle, shot=shot, on_line=on_line)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if shot is not None:
            shot.close(success=proc.returncode == 0)
        os.close(master)
    return proc.returncode if proc.returncode is not None else proc.wait()


def _run_tee_plain(
    cmd: list[str],
    *,
    log: Path,
    env: dict[str, str],
    cwd: Path | None = None,
    on_line=None,
    on_start=None,
) -> int:
    """无 PTY 环境（受限容器）的退化版：不画进度条，只落日志 + 回调行。"""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    if on_start is not None:
        try:
            on_start(proc.pid)
        except Exception:
            pass
    relay = _ProgressRelay()
    with log.open("wb") as handle:
        header = ""
        if cwd is not None:
            header += f"cwd={cwd}\n"
        header += " ".join(cmd) + "\n\n"
        handle.write(header.encode("utf-8", errors="replace"))
        handle.flush()
        assert proc.stdout is not None
        for raw in iter(lambda: proc.stdout.readline(), b""):
            handle.write(raw)
            handle.flush()
            for line in relay.feed(raw):
                if on_line is not None:
                    try:
                        on_line(line)
                    except Exception:
                        pass
                _emit_lines([line])
        for line in relay.finish():
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass
    proc.wait()
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


def _pump(master: int, proc: subprocess.Popen, handle, *, shot: _ShotBar | None, on_line=None) -> None:
    relay = _ProgressRelay()
    while True:
        ready, _, _ = select.select([master], [], [], 0.2)
        if not ready:
            if proc.poll() is not None:
                _drain(master, relay, handle, shot=shot, on_line=on_line)
                return
            continue
        try:
            chunk = os.read(master, 65536)
        except OSError:
            if proc.poll() is not None:
                _finish(relay, handle, shot=shot, on_line=on_line)
                return
            continue
        if not chunk:
            _finish(relay, handle, shot=shot, on_line=on_line)
            return
        _consume(chunk, relay, handle, shot=shot, on_line=on_line)


def _drain(master: int, relay: _ProgressRelay, handle, *, shot: _ShotBar | None, on_line=None) -> None:
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
        _consume(chunk, relay, handle, shot=shot, on_line=on_line)
    _finish(relay, handle, shot=shot, on_line=on_line)


def _consume(chunk: bytes, relay: _ProgressRelay, handle, *, shot: _ShotBar | None, on_line=None) -> None:
    handle.write(chunk)
    handle.flush()
    lines = relay.feed(chunk)
    if not lines:
        return
    if on_line is not None:
        try:
            for line in lines:
                on_line(line)
        except Exception:
            pass
    if shot is not None:
        for line in lines:
            shot.handle(line)
        return
    _emit_lines(lines)


def _finish(relay: _ProgressRelay, handle, *, shot: _ShotBar | None, on_line=None) -> None:
    lines = relay.finish()
    if not lines:
        return
    if on_line is not None:
        try:
            for line in lines:
                on_line(line)
        except Exception:
            pass
    if shot is not None:
        for line in lines:
            shot.handle(line)
        return
    _emit_lines(lines)


def _emit_lines(lines: list[str]) -> None:
    if not lines:
        return
    sys.stdout.flush()
    text = "".join(line + "\n" for line in lines)
    out = getattr(sys.stdout, "buffer", None)
    if out is not None:
        out.write(text.encode("utf-8", errors="replace"))
        out.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
