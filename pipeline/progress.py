"""子进程输出进日志；终端只保留必要的进度条。

视频每个分镜把 LTX 两段 Denoising（以及其它阶段条）合成一条进度。
本机有 TTY 时，本进程 tqdm 仍用同行刷新，避免刷屏。
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
from collections.abc import Sequence
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Denoising:  12%|…| 1/8 [00:02<00:18,  2.65s/it]
_TQDM = re.compile(
    r"^(?P<desc>.+?):\s+(?P<pct>\d+)%.*? (?P<n>\d+)/(?P<total>\d+)\b"
)
_TQDM_ETA = re.compile(r"\[(?P<elapsed>[^<]*)<(?P<eta>[^\],]*)")
_PHASE_START = re.compile(r"^\[(?P<label>[^\]]+)\]\s*\.\.\.\s*$")
_PHASE_DONE = re.compile(
    r"^\[(?P<label>[^\]]+)\]\s+done in\s+(?P<sec>\d+(?:\.\d+)?)s\b",
    re.I,
)
_ERRORISH = re.compile(
    r"^(Traceback\b|\w*Error:|\w*Exception:|SystemExit:|FATAL)",
    re.IGNORECASE,
)

# 加载/编解码在进度条里的权重（相对 denoise 步）；冷启动 Gemma/DiT/decode 很慢，不能只占 1 格
VIDEO_LOAD_PRE_WEIGHT = 10
VIDEO_LOAD_POST_WEIGHT = 8
VIDEO_LOAD_PRE_COUNT = 3   # text encoder / encode prompt / transformer
VIDEO_LOAD_POST_COUNT = 2  # decoders / decode video
# 兼容旧名
VIDEO_LOAD_PRE = VIDEO_LOAD_PRE_COUNT
VIDEO_LOAD_POST = VIDEO_LOAD_POST_COUNT

# 各阶段缺实测时的默认耗时（秒），用于软爬与 ETA
_PRE_EXPECT_S = (22.0, 6.0, 18.0)
_POST_EXPECT_S = (8.0, 14.0)
_DENOISE_STEP_EXPECT_S = 4.5


def classify_load_phase(label: str) -> str | None:
    """返回 ``pre`` / ``post``；无关或 estimate 返回 None。"""
    low = (label or "").strip().lower()
    if not low or low.startswith("estimate"):
        return None
    if any(
        s in low
        for s in (
            "loading text encoder",
            "encoding prompt",
            "loading transformer",
            "loading dit",
            "encoding nag",
        )
    ):
        return "pre"
    if any(s in low for s in ("loading decoder", "decoding video", "decoding hdr")):
        return "post"
    return None


def video_progress_plan(denoise: Sequence[int] | None) -> list[int]:
    """加载权重 + 去噪段 + 收尾权重 → ShotBar / StageMerge 完整 plan。"""
    mid = [max(1, int(x)) for x in (denoise or ())]
    plan: list[int] = []
    if VIDEO_LOAD_PRE_WEIGHT > 0:
        plan.append(VIDEO_LOAD_PRE_WEIGHT)
    plan.extend(mid)
    if VIDEO_LOAD_POST_WEIGHT > 0:
        plan.append(VIDEO_LOAD_POST_WEIGHT)
    return plan or [1]


def parse_duration_seconds(text: str | float | int | None) -> float | None:
    """把时长统一解析成**秒**（内部计算只用秒）。

    接受：
    - 数字 / 数字字符串 → 秒
    - tqdm 风格 ``SS`` / ``M:SS`` / ``H:MM:SS``
      （两段一定是分:秒，``1:00`` = 60 秒，绝不是 1 小时）
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        val = float(text)
        return val if val >= 0 else None
    raw = str(text).strip()
    if not raw or raw in {"?", "??", "-"}:
        return None
    # 纯秒（可带小数）
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            # M:SS —— 常见 tqdm 剩余时间；旧正则曾把 1:00 误当成 1 小时
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None
    return None


# 兼容旧名
_parse_hms = parse_duration_seconds


def format_duration(seconds: float | int | None) -> str:
    """秒 → 终端/日志用的易读串（``0:45`` / ``3:20`` / ``1:02:03``）。计算请用秒，不要反解析这串。"""
    if seconds is None:
        return "?"
    sec = max(0, int(round(float(seconds))))
    if sec < 60:
        return f"0:{sec:02d}"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


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
        """同一段时间里 n 只会涨；倒退、换 total、或从 0 重数 → 新段。

        加载段跑满后去噪可能直接跳到 2/12（不见 0/12），也要切段。
        """
        if n < self.last_n:
            return True
        if self.last_total > 0 and total != self.last_total:
            return True
        return False

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

    分母 = 加载权重 + 去噪步 + 收尾权重。加载 ``[phase]...`` 期间按期望耗时软爬，
    结束 ``done in Xs`` 时咬合到该段；ETA 按剩余加载/去噪/收尾估算，避免 tqdm 乱跳。
    """

    def __init__(
        self,
        label: str,
        *,
        plan: Sequence[int] | None = None,
        desc: str | None = None,
    ) -> None:
        from tqdm import tqdm

        self.denoise_plan = [max(1, int(x)) for x in (plan or ())]
        self.full_plan = video_progress_plan(self.denoise_plan)
        self.pre_weight = VIDEO_LOAD_PRE_WEIGHT if VIDEO_LOAD_PRE_WEIGHT > 0 else 0
        self.post_weight = VIDEO_LOAD_POST_WEIGHT if VIDEO_LOAD_POST_WEIGHT > 0 else 0
        self.pre_count = VIDEO_LOAD_PRE_COUNT
        self.post_count = VIDEO_LOAD_POST_COUNT
        self.merge = StageMerge(self.full_plan)
        self.desc = desc
        self._pre_done = 0
        self._post_done = 0
        self._in_denoise = False
        self._denoise_finished = False
        self._seen_phases: set[str] = set()
        self._phase_kind: str | None = None
        self._phase_t0 = time.time()
        self._phase_expect = 8.0
        self._phase_postfix = "准备"
        self._pre_expect = list(_PRE_EXPECT_S[: self.pre_count]) + [12.0] * max(
            0, self.pre_count - len(_PRE_EXPECT_S)
        )
        self._post_expect = list(_POST_EXPECT_S[: self.post_count]) + [10.0] * max(
            0, self.post_count - len(_POST_EXPECT_S)
        )
        self._denoise_eta_s: float | None = None
        self._denoise_n = 0
        self._denoise_total = sum(self.denoise_plan) or 1
        self._t0 = time.time()
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        self.bar = tqdm(
            total=max(1, self.merge.total),
            desc=label,
            unit="step",
            file=sys.stderr,
            dynamic_ncols=False,
            ncols=min(100, max(40, cols)),
            miniters=1,
            mininterval=0.15,
            leave=True,
        )
        # 一开始就进入「准备」软爬，进度条不是 0
        self._phase_kind = "boot"
        self._phase_expect = 6.0
        self._phase_t0 = time.time()
        self._paint(self._pre_units(0.08), self.merge.total, postfix="准备")

    def _owns(self, line: str) -> bool:
        if not self.desc:
            return True
        match = _TQDM.match(line)
        return bool(match) and match.group("desc").strip().startswith(self.desc)

    def _pre_units(self, frac_phases: float) -> int:
        if self.pre_weight <= 0:
            return 0
        return int(min(self.pre_weight, max(0, round(frac_phases * self.pre_weight))))

    def _post_units(self, frac_phases: float) -> int:
        if self.post_weight <= 0:
            return 0
        base = self.pre_weight + sum(self.denoise_plan)
        return base + int(min(self.post_weight, max(0, round(frac_phases * self.post_weight))))

    def _eta_s(self) -> float:
        """剩余秒数。去噪子条的 ETA 若已被解析成离谱大数，改用预设步时。"""
        now = time.time()
        rem = 0.0
        # 单镜合理上限：加载 + 去噪 + 收尾经验值（秒），避免写成 60:27 再被下游当成 3627s
        denoise_left = max(0, self._denoise_total - self._denoise_n)
        preset_denoise = denoise_left * _DENOISE_STEP_EXPECT_S
        preset_pre = sum(self._pre_expect[self._pre_done :]) if not self._in_denoise else 0.0
        preset_post = sum(self._post_expect[self._post_done :])
        hard_cap = max(90.0, preset_pre + preset_denoise + preset_post + 30.0)

        if self._phase_kind in {"boot", "pre"} and not self._in_denoise:
            rem += max(0.0, self._phase_expect - (now - self._phase_t0))
            for i in range(self._pre_done, self.pre_count):
                if self._phase_kind == "pre" and i == self._pre_done:
                    continue
                rem += self._pre_expect[i] if i < len(self._pre_expect) else 12.0
        if not self._denoise_finished:
            if self._in_denoise and self._denoise_eta_s is not None:
                # 子 tqdm 剩余若 > 预设×2，多半是 M:SS/速率尖刺，改信预设
                if self._denoise_eta_s <= max(60.0, preset_denoise * 2.5):
                    rem += max(0.0, self._denoise_eta_s)
                else:
                    rem += preset_denoise
            else:
                rem += preset_denoise
        if self._post_done < self.post_count:
            if self._phase_kind == "post":
                rem += max(0.0, self._phase_expect - (now - self._phase_t0))
                for i in range(self._post_done + 1, self.post_count):
                    rem += self._post_expect[i] if i < len(self._post_expect) else 10.0
            else:
                for i in range(self._post_done, self.post_count):
                    rem += self._post_expect[i] if i < len(self._post_expect) else 10.0
        return min(hard_cap, max(0.0, rem))

    def _paint(self, done: int, overall: int, *, postfix: str | None = None) -> None:
        done = max(int(self.bar.n), min(int(done), int(overall)))
        if self.bar.total != overall:
            self.bar.total = max(1, overall)
        self.bar.n = min(done, self.bar.total)
        # 内部始终用秒；写入进度条的 HMS 仅供人读，下游应再 parse_duration_seconds
        eta = self._eta_s()
        elapsed = format_duration(time.time() - self._t0)
        self.bar.bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} ["
            + elapsed
            + "<"
            + format_duration(eta)
            + "{postfix}]"
        )
        if postfix is not None:
            self._phase_postfix = postfix
        # tqdm 会自己在 {postfix} 前加 ", "；这里只放文案
        self.bar.set_postfix_str(self._phase_postfix or "", refresh=False)
        self.bar.refresh()
        self.merge.n = max(self.merge.n, self.bar.n)
        self.merge.total = self.bar.total
        self.merge.seen = True

    def tick(self) -> None:
        """静默加载期间由 pump 定时调用，进度条不要卡死。"""
        if self._phase_kind not in {"boot", "pre", "post"}:
            return
        if self._phase_kind == "boot" and self._pre_done > 0:
            return
        elapsed = time.time() - self._phase_t0
        frac = min(0.9, elapsed / max(0.8, self._phase_expect))
        if self._phase_kind in {"boot", "pre"} and not self._in_denoise:
            # boot 算第 0 段的开头；正式 pre 从 _pre_done 起算当前段
            if self._phase_kind == "boot":
                units = self._pre_units(frac / max(1, self.pre_count))
            else:
                units = self._pre_units((self._pre_done + frac) / max(1, self.pre_count))
            self._paint(units, self.merge.total, postfix=self._phase_postfix)
            return
        if self._phase_kind == "post":
            units = self._post_units((self._post_done + frac) / max(1, self.post_count))
            self._paint(units, self.merge.total, postfix=self._phase_postfix)

    def _begin_phase(self, kind: str, *, postfix: str, expect: float) -> None:
        self._phase_kind = kind
        self._phase_t0 = time.time()
        self._phase_expect = max(1.0, expect)
        self._phase_postfix = postfix
        self.tick()

    def _finish_pre_phase(self, *, sec: float | None = None) -> None:
        idx = min(self._pre_done, max(0, self.pre_count - 1))
        if sec is not None and sec > 0.5 and idx < len(self._pre_expect):
            # 平滑更新期望，供后续 ETA
            self._pre_expect[idx] = 0.6 * self._pre_expect[idx] + 0.4 * sec
        self._pre_done = min(self.pre_count, self._pre_done + 1)
        self._phase_kind = None
        units = self._pre_units(self._pre_done / max(1, self.pre_count))
        # 咬合到当前已完成加载格，并写入 StageMerge 首段
        if self.pre_weight > 0:
            self.merge.update(units, self.pre_weight)
        self._paint(units, self.merge.total, postfix=self._phase_postfix)

    def _finish_post_phase(self, *, sec: float | None = None) -> None:
        idx = min(self._post_done, max(0, self.post_count - 1))
        if sec is not None and sec > 0.5 and idx < len(self._post_expect):
            self._post_expect[idx] = 0.6 * self._post_expect[idx] + 0.4 * sec
        self._post_done = min(self.post_count, self._post_done + 1)
        self._phase_kind = None
        units = self._post_units(self._post_done / max(1, self.post_count))
        if self.post_weight > 0:
            # 末段：用 post 计数推进
            self.merge.update(self._post_done, self.post_count)
        self._paint(units, self.merge.total, postfix=self._phase_postfix)

    def handle(self, line: str) -> None:
        if _ERRORISH.match(line) or line.startswith("  File "):
            from tqdm import tqdm

            tqdm.write(line, file=sys.stderr)
            return
        done_m = _PHASE_DONE.match(line)
        if done_m:
            label = done_m.group("label").strip()
            kind = classify_load_phase(label)
            try:
                sec = float(done_m.group("sec"))
            except ValueError:
                sec = None
            if kind == "pre" and not self._in_denoise:
                self._finish_pre_phase(sec=sec)
            elif kind == "post":
                self._finish_post_phase(sec=sec)
            return
        match = _TQDM.match(line)
        if match and self._owns(line):
            n = int(match.group("n"))
            total = int(match.group("total"))
            if not self._in_denoise:
                # 去噪开始：先咬合加载段
                if self.pre_weight > 0 and self._pre_done < self.pre_count:
                    self._pre_done = self.pre_count
                    self.merge.update(self.pre_weight, self.pre_weight)
                self._in_denoise = True
                self._phase_kind = None
            self._denoise_n = max(self._denoise_n, n)
            self._denoise_total = max(self._denoise_total, total)
            eta_m = _TQDM_ETA.search(line)
            if eta_m:
                parsed = parse_duration_seconds(eta_m.group("eta"))
                if parsed is not None:
                    self._denoise_eta_s = parsed  # 秒
            done, overall = self.merge.update(n, total)
            if n >= total and total > 0:
                self._denoise_finished = True
                self._denoise_eta_s = 0.0
            self._paint(done, overall, postfix="")
            return
        phase = _PHASE_START.match(line)
        if phase:
            label = phase.group("label").strip()
            if label.startswith("estimate"):
                return
            short = label.split("(")[0].strip()
            if len(short) > 18:
                short = short[:18]
            kind = classify_load_phase(label)
            key = label.lower()
            if kind == "pre" and not self._in_denoise and key not in self._seen_phases:
                self._seen_phases.add(key)
                expect = (
                    self._pre_expect[self._pre_done]
                    if self._pre_done < len(self._pre_expect)
                    else 12.0
                )
                self._begin_phase("pre", postfix=short, expect=expect)
            elif kind == "post" and key not in self._seen_phases:
                self._seen_phases.add(key)
                if self._in_denoise and not self._denoise_finished:
                    self._denoise_finished = True
                    self._denoise_eta_s = 0.0
                expect = (
                    self._post_expect[self._post_done]
                    if self._post_done < len(self._post_expect)
                    else 10.0
                )
                self._begin_phase("post", postfix=short, expect=expect)
            else:
                self._phase_postfix = short
                self.bar.set_postfix_str(f", {short}", refresh=True)

    def close(self, *, success: bool = True) -> None:
        if success and self.merge.seen:
            self.bar.n = self.bar.total
        self.bar.set_postfix_str("", refresh=False)
        self.bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
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
            # 加载阶段无新输出时也要软爬进度 / 刷新 ETA
            if shot is not None:
                try:
                    shot.tick()
                except Exception:
                    pass
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
