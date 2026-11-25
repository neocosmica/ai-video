"""按本机总内存查经验 token 预算表，再按当前分辨率现算单镜最长帧数。

只读总内存配置（不读 wired / 空闲页 / swap），同一台机器任何时候都得到
同一个上限，分镜可稳定复现。经验值表存的是 video token 预算（与分辨率
无关）；帧数由 token 预算 + config 里的分辨率即时换算。
"""

from __future__ import annotations

import os
from typing import Any

from pipeline.util import snap_frames

# 总内存 GiB → 单镜可承受的 video token 预算（stage2 = 全分辨率）
# 16 GiB → 3120 = 97 帧 @ 384x640；32 GiB → 6720 = 217 帧 @ 384x640
TOKEN_BUDGET_TABLE: tuple[tuple[float, int], ...] = (
    (8.0, 960),  # = 25 帧 @ 384x640
    (16.0, 3120),  # = 97 帧 @ 384x640
    (32.0, 6720),  # = 217 帧 @ 384x640
    (64.0, 7440),  # = 241 帧 @ 384x640（硬顶）
)

# ETA 经验锚点：token 预算 → 两段去噪合计秒数（粗略估计）
_ETA_ANCHORS: tuple[tuple[int, float], ...] = (
    (3120, 120.0),
    (6720, 360.0),
)

CHARS_PER_SECOND = 2.4


def total_memory_bytes() -> int:
    """本机物理内存字节数。

    统一走 ``pipeline.platform``：macOS/Linux 用 sysconf，Windows 用
    ``GlobalMemoryStatusEx``（不引入 psutil）。Mac 行为与之前一致。
    """
    from pipeline.platform import total_memory_bytes as _total

    return _total()


def gib(bytes_: int | float) -> float:
    return float(bytes_) / (1024**3)


def latent_frames(frames: int) -> int:
    """像素帧数 → latent 帧数：(F - 1) // 8 + 1。"""
    return max(1, (int(frames) - 1) // 8 + 1)


def tokens_per_latent_frame(height: int, width: int) -> int:
    """stage2（全分辨率）每 latent 帧的 video token 数。"""
    return 4 * max(1, int(height) // 64) * max(1, int(width) // 64)


def video_tokens(frames: int, height: int, width: int) -> int:
    """给定像素帧数与分辨率，算出 stage2 video token 数。"""
    return latent_frames(frames) * tokens_per_latent_frame(height, width)


def lookup_token_budget(
    total_bytes: int,
    *,
    table: tuple[tuple[float, int], ...] = TOKEN_BUDGET_TABLE,
) -> tuple[int, dict[str, Any]]:
    """单调阶梯：取 <= 本机总内存的最高档；低于最低档用最低档并标记 underflow。"""
    total_gib = gib(total_bytes)
    rows = sorted(table, key=lambda row: row[0])
    if not rows:
        raise SystemExit("token 预算表为空")
    chosen = rows[0]
    exact = False
    for gib_cap, tokens in rows:
        if total_gib + 1e-9 >= gib_cap:
            chosen = (gib_cap, tokens)
            exact = abs(total_gib - gib_cap) < 0.05
        else:
            break
    underflow = total_gib + 1e-9 < rows[0][0]
    off_table = not exact and not underflow and chosen[0] < rows[-1][0]
    return int(chosen[1]), {
        "total_gib": round(total_gib, 2),
        "tier_gib": chosen[0],
        "token_budget": int(chosen[1]),
        "exact_tier": exact,
        "underflow": underflow,
        "off_table": off_table,
        "table": [(g, t) for g, t in rows],
    }


def frames_for_budget(
    tokens: int,
    height: int,
    width: int,
    *,
    min_frames: int = 25,
    ceiling: int = 241,
) -> tuple[int, dict[str, Any]]:
    """token 预算 + 分辨率 → 像素帧数（8k+1，宁短勿长，夹在 [min, ceiling]）。"""
    per = tokens_per_latent_frame(height, width)
    latent = max(1, int(tokens) // per)
    frames = 8 * (latent - 1) + 1
    lo = snap_frames(min_frames)
    hi = snap_frames(ceiling)
    if lo > hi:
        lo, hi = hi, lo
    capped = max(lo, min(hi, frames))
    return capped, {
        "tokens_per_latent": per,
        "latent_frames": latent,
        "raw_frames": frames,
        "frames": capped,
        "hit_ceiling": frames > hi,
        "hit_floor": frames < lo,
    }


def estimate_denoise_seconds(tokens: int) -> float:
    """按经验锚点线性插值粗略估去噪耗时（秒）。"""
    if tokens <= _ETA_ANCHORS[0][0]:
        t0, s0 = _ETA_ANCHORS[0]
        return max(30.0, s0 * tokens / max(1, t0))
    if tokens >= _ETA_ANCHORS[-1][0]:
        t1, s1 = _ETA_ANCHORS[-1]
        return s1 * tokens / max(1, t1)
    (t0, s0), (t1, s1) = _ETA_ANCHORS[0], _ETA_ANCHORS[1]
    ratio = (tokens - t0) / max(1, t1 - t0)
    return s0 + ratio * (s1 - s0)


def dialogue_char_budget(frames: int, fps: int) -> int:
    """单镜台词字数额度：帧数 / fps × 实测语速。"""
    seconds = max(1.0, float(frames) / max(1, int(fps)))
    return max(4, int(seconds * CHARS_PER_SECOND))


def resolve_frame_cap(
    *,
    total_bytes: int | None = None,
    height: int = 640,
    width: int = 384,
    fps: int = 24,
    min_frames: int = 25,
    ceiling: int = 241,
    table: tuple[tuple[float, int], ...] | None = None,
) -> tuple[int, dict[str, Any]]:
    """查 token 预算表并按分辨率现算单镜最长帧数。

    ``total_bytes`` 为 None 时读本机总内存；自检时传入显式值，不碰系统。
    """
    mem = int(total_bytes) if total_bytes is not None else total_memory_bytes()
    budget, tier = lookup_token_budget(mem, table=table or TOKEN_BUDGET_TABLE)
    frames, geom = frames_for_budget(
        budget,
        height,
        width,
        min_frames=min_frames,
        ceiling=ceiling,
    )
    seconds = float(frames) / max(1, int(fps))
    eta = estimate_denoise_seconds(budget)
    detail: dict[str, Any] = {
        **tier,
        **geom,
        "height": int(height),
        "width": int(width),
        "fps": int(fps),
        "ceiling": snap_frames(ceiling),
        "min_frames": snap_frames(min_frames),
        "seconds": round(seconds, 2),
        "dialogue_chars": dialogue_char_budget(frames, fps),
        "eta_seconds": round(eta, 1),
        "stage2_tokens": video_tokens(frames, height, width),
    }
    return frames, detail


def format_budget_report(detail: dict[str, Any]) -> str:
    """把 resolve_frame_cap 的明细打成可读报告。"""
    lines: list[str] = []
    total = detail["total_gib"]
    tier = detail["tier_gib"]
    budget = detail["token_budget"]
    if detail.get("underflow"):
        lines.append(f"本机总内存：{total:.1f} GB → 低于最低档，按 {tier:g} GB 档处理，token 预算 {budget}")
    elif detail.get("off_table"):
        lines.append(
            f"本机总内存：{total:.1f} GB → 不在表中，按 {tier:g} GB 档处理，token 预算 {budget}"
        )
    else:
        lines.append(f"本机总内存：{total:.1f} GB → 命中「{tier:g} GB」档，token 预算 {budget}")

    w, h, fps = detail["width"], detail["height"], detail["fps"]
    per = detail["tokens_per_latent"]
    frames = detail["frames"]
    seconds = detail["seconds"]
    ceiling = detail["ceiling"]
    lines.append("")
    lines.append(
        f"分辨率 {w}x{h} @{fps}fps（每 latent 帧 {per} tokens），--low-ram + q4"
    )
    ceiling_note = f"（硬顶 {ceiling} 帧 / {ceiling / max(1, fps):.1f} s）"
    if detail.get("hit_ceiling"):
        lines.append(f"单镜上限：{frames} 帧 ≈ {seconds:.1f} s{ceiling_note} ← 已撞硬顶")
    else:
        lines.append(f"单镜上限：{frames} 帧 ≈ {seconds:.1f} s{ceiling_note}")
    eta_min = detail["eta_seconds"] / 60.0
    lines.append(
        f"  该镜头 stage2 约 {detail['stage2_tokens']} video tokens，"
        f"去噪预估 ~{eta_min:.0f} min（粗略估计）"
    )
    lines.append(
        f"单镜台词额度：约 {detail['dialogue_chars']} 字"
        f"（语速 {CHARS_PER_SECOND} 字/秒）"
    )
    lines.append("")
    lines.append("token 预算表（与分辨率无关，帧数由当前分辨率现算）：")
    for gib_cap, tokens in detail["table"]:
        demo_frames, demo = frames_for_budget(
            tokens,
            h,
            w,
            min_frames=detail["min_frames"],
            ceiling=ceiling,
        )
        demo_s = demo_frames / max(1, fps)
        mark = "（硬顶）" if demo.get("hit_ceiling") else ""
        lines.append(
            f"  {gib_cap:4g} GB → {tokens:5d} tokens → "
            f"{w}x{h} 下 {demo_frames:3d} 帧 / {demo_s:.1f} s{mark}"
        )
    if (w, h) == (384, 640):
        alt, _ = frames_for_budget(
            budget, 768, 512, min_frames=detail["min_frames"], ceiling=ceiling
        )
        lines.append("")
        lines.append(
            f"若把分辨率改成 512x768，本机同样 {budget} tokens 只能装 {alt} 帧 "
            f"（{(alt / max(1, fps)):.1f} s）。"
        )
    lines.append("")
    lines.append("注：解码会自动切块，不参与上限判定。")
    return "\n".join(lines)
