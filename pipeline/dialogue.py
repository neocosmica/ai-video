"""台词切分：把一句长台词按标点切到每段都能在单镜时长内念完。

这里只做纯文本/时长逻辑，不依赖模型，便于自检。
"""

from __future__ import annotations

import re

# 在这些字符后面断句（标点留在前一段）
_HARD_BREAK = "。！？；…!?;"
_SOFT_BREAK = "，、：,:"

_SPOKEN_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)


def spoken_chars(text: str) -> int:
    """只数会被念出来的字：去掉空白、标点和引号。"""
    return len(_SPOKEN_STRIP.sub("", text or ""))


def split_clauses(text: str) -> list[str]:
    """按标点切成小句，标点跟着前一句，原文一个字都不丢。"""
    out: list[str] = []
    buf = ""
    for ch in text or "":
        buf += ch
        if ch in _HARD_BREAK or ch in _SOFT_BREAK:
            if buf.strip():
                out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf)
    return out or ([text] if str(text or "").strip() else [])


def _force_cut(piece: str, max_chars: int) -> list[str]:
    """没有标点可断时按字数硬切，保证每段都不超过上限。"""
    max_chars = max(1, int(max_chars))
    out: list[str] = []
    rest = piece
    while spoken_chars(rest) > max_chars:
        # 按「会被念出来的字」计数，标点不占额度
        count = 0
        cut = 0
        for index, ch in enumerate(rest):
            if not _SPOKEN_STRIP.match(ch):
                continue
            count += 1
            if count > max_chars:
                cut = index
                break
        if cut <= 0:
            cut = max_chars
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest.strip():
        out.append(rest)
    return out


def plan_segments(text: str, max_chars: int, *, overshoot: float = 1.4) -> list[str]:
    """把台词切成若干段，每段去标点后接近 max_chars 字。

    优先在标点处断开；单个小句只要没超出 ``max_chars * overshoot`` 就整体保留，
    避免把「的样子」切成「的样」+「子」。真超太多（无标点长句）才按字数硬切。
    """
    max_chars = max(2, int(max_chars))
    soft_cap = max(max_chars, int(round(max_chars * max(1.0, overshoot))))
    pieces: list[str] = []
    for clause in split_clauses(text):
        chars = spoken_chars(clause)
        if chars > soft_cap:
            pieces.extend(_force_cut(clause, max_chars))
        elif clause.strip():
            pieces.append(clause)

    groups: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif spoken_chars(current + piece) <= max_chars:
            current += piece
        else:
            groups.append(current)
            current = piece
    if current.strip():
        groups.append(current)

    # 极端情况：拼回去必须和原文一致（只差首尾空白）
    if spoken_chars("".join(groups)) != spoken_chars(text):
        return [text]
    return groups or [text]


def split_text_by_ratios(text: str, ratios: list[float]) -> list[str] | None:
    """按累计时长比例把台词切开，切点落在标点边界上。

    ratios 是各段结束位置占总长的比例（升序，最后一项应为 1）。
    用于让字幕分段与音频切点大致对齐；切不开时返回 None。
    """
    clauses = split_clauses(text)
    if len(clauses) < len(ratios):
        return None
    total = spoken_chars(text)
    if total <= 0:
        return None
    groups: list[str] = []
    start = 0
    acc = 0
    for ratio in ratios[:-1]:
        target = total * float(ratio)
        best_index = start
        best_gap = None
        running = acc
        for index in range(start, len(clauses) - 1):
            running += spoken_chars(clauses[index])
            gap = abs(running - target)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = index + 1
        # 至少给后面每段留一个小句
        remaining = len(ratios) - len(groups) - 1
        best_index = min(best_index, len(clauses) - remaining)
        if best_index <= start:
            return None
        groups.append("".join(clauses[start:best_index]))
        acc += spoken_chars("".join(clauses[start:best_index]))
        start = best_index
    groups.append("".join(clauses[start:]))
    if any(not group.strip() for group in groups):
        return None
    if spoken_chars("".join(groups)) != total:
        return None
    return groups


def halve(text: str) -> list[str]:
    """实在切不开时，找中间最近的标点对半切；没有标点就按字数切。"""
    clauses = split_clauses(text)
    if len(clauses) >= 2:
        mid = len(clauses) / 2
        acc = 0
        for index, clause in enumerate(clauses):
            acc += spoken_chars(clause)
            if acc >= mid:
                left = "".join(clauses[: index + 1])
                right = "".join(clauses[index + 1 :])
                if left.strip() and right.strip():
                    return [left, right]
                break
    total = spoken_chars(text)
    return _force_cut(text, max(1, total // 2)) if total > 2 else [text]


def plan_by_measured_rate(text: str, duration_s: float, max_seconds: float) -> list[str]:
    """按整句的实测语速，反推「一个镜头最多能装多少字」，再据此切分文本。

    注意：这只是**文本切分**的辅助工具，用来提前规划字幕/分镜边界。
    它不会、也不应该用来调速——配音始终按原文一次合成，语速由内容和情绪决定。
    """
    chars = spoken_chars(text)
    if chars <= 0 or duration_s <= 0:
        return [text]
    if duration_s <= max_seconds:
        return [text]
    chars_per_second = chars / duration_s
    # 留 8% 余量：估算本身有误差
    max_chars = max(2, int(max_seconds * chars_per_second * 0.92))
    return plan_segments(text, max_chars)
