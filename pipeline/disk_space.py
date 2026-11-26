"""下载新权重之前，看 Hugging Face 缓存所在的磁盘还剩多少。

只估算还没齐的仓库。已经能用的权重不算进本次下载。
空间不够时给出给人看的说明，调用方应停在下载开始之前。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

from pipeline.hf_cache import cache_nbytes, hub_dir, locate_repo_dir, resolve_snapshot, snapshot_ready

# 下完仍给系统和成片留出的空闲。
RESERVE_BYTES = 1024**3
# 目录里的体积是约数，再留一成，避免估小了才开始下。
SLACK = 1.10
# 自定义仓库没有约数时，至少按这个量提醒。
UNKNOWN_BYTES = 8 * 1024**3


def format_bytes(n: int) -> str:
    v = max(0, int(n))
    if v < 1024:
        return f"{v} B"
    units = ("KB", "MB", "GB", "TB")
    x = v / 1024
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    digits = 0 if x >= 10 or i == 0 else 1
    return f"{x:.{digits}f} {units[i]}"


def disk_free_bytes(path: Path | None = None) -> int:
    cursor = path or hub_dir()
    while not cursor.exists():
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return int(shutil.disk_usage(cursor).free)


def _as_dict(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    as_dict = getattr(spec, "as_dict", None)
    if callable(as_dict):
        data = as_dict()
        if isinstance(data, dict):
            return data
    raise TypeError(f"无法读取权重说明：{spec!r}")


def plan_weight_space(
    specs: Iterable[Any],
    *,
    hub: Path | None = None,
    free: int | None = None,
) -> dict[str, Any]:
    """还没齐的权重要占多少空间。``ok`` 为假时不要开始下载。"""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in specs:
        spec = _as_dict(raw)
        repo = str(spec.get("repo") or "").strip()
        if not repo or repo in seen:
            continue
        seen.add(repo)
        snap = resolve_snapshot(repo, hub=hub)
        ready, _detail = snapshot_ready(
            snap,
            required_files=tuple(spec.get("required_files") or ()),
            weight_globs=tuple(spec.get("weight_globs") or ()),
        )
        if ready:
            continue
        folder = locate_repo_dir(repo, hub)
        cached = cache_nbytes(folder) if folder is not None else 0
        approx = max(0, int(spec.get("approx_bytes") or 0))
        unknown = approx <= 0
        if unknown:
            need = 0
        else:
            gap = approx - cached
            if gap <= 0:
                # 约数小于已有残片：再留一成，避免残缺文件被当成已经够了
                gap = max(approx // 10, 1)
            need = gap
        items.append(
            {
                "label": str(spec.get("label") or repo),
                "repo": repo,
                "approx_bytes": approx,
                "cached_bytes": cached,
                "need_bytes": need,
                "unknown": unknown,
            }
        )

    need = sum(int(item["need_bytes"]) for item in items)
    unknown_extra = UNKNOWN_BYTES if any(item["unknown"] for item in items) else 0
    required = 0 if not items else int(need * SLACK) + unknown_extra + RESERVE_BYTES
    free_bytes = disk_free_bytes(hub or hub_dir()) if free is None else int(free)
    shortfall = max(0, required - free_bytes)
    ok = shortfall == 0
    return {
        "ok": ok,
        "free_bytes": free_bytes,
        "need_bytes": need + unknown_extra,
        "required_bytes": required,
        "shortfall_bytes": shortfall,
        "reserve_bytes": RESERVE_BYTES if items else 0,
        "message": "" if ok else _short_message(items, free_bytes, shortfall),
        "items": items,
    }


def _short_message(items: list[dict[str, Any]], free_bytes: int, shortfall: int) -> str:
    parts: list[str] = []
    for item in items:
        label = str(item["label"])
        if item["unknown"]:
            parts.append(f"{label}（体积未知，按 {format_bytes(UNKNOWN_BYTES)} 预留）")
        else:
            parts.append(f"{label}约 {format_bytes(int(item['need_bytes']))}")
    listed = "、".join(parts)
    return (
        f"磁盘空间不够，下载尚未开始。还需要 {listed}，"
        f"当前剩余 {format_bytes(free_bytes)}。"
        f"请先腾出至少 {format_bytes(shortfall)}（含 {format_bytes(RESERVE_BYTES)} 余量），"
        f"或清理未使用的模型后再安装。"
    )


def selftest_disk_space() -> None:
    """临时 hub：缺的计入，齐的跳过，空间不够时明确说还没开始下载。"""
    import tempfile

    from pipeline.hf_cache import repo_dir_name

    def plant(hub: Path, repo: str, payload: bytes, *, ready: bool) -> None:
        folder = hub / repo_dir_name(repo)
        blobs = folder / "blobs"
        blobs.mkdir(parents=True)
        blob = blobs / "weight"
        blob.write_bytes(payload)
        if not ready:
            return
        snap = folder / "snapshots" / "rev"
        snap.mkdir(parents=True)
        link = snap / "model.safetensors"
        link.symlink_to(blob)
        refs = folder / "refs"
        refs.mkdir()
        (refs / "main").write_text("rev", encoding="utf-8")

    missing = {
        "label": "测试模型",
        "repo": "org/missing",
        "required_files": (),
        "weight_globs": ("model*.safetensors",),
        "approx_bytes": 10_000,
    }
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp)
        empty = plan_weight_space([], hub=hub, free=0)
        assert empty["ok"] and empty["need_bytes"] == 0 and empty["items"] == []
        assert empty["message"] == ""

        tight = plan_weight_space([missing, dict(missing)], hub=hub, free=0)
        assert len(tight["items"]) == 1
        assert tight["need_bytes"] == 10_000
        assert tight["ok"] is False
        assert "尚未开始" in tight["message"]
        assert "测试模型" in tight["message"]
        assert tight["required_bytes"] >= 10_000 + RESERVE_BYTES

        required = int(tight["required_bytes"])
        assert plan_weight_space([missing], hub=hub, free=required)["ok"] is True
        short = plan_weight_space([missing], hub=hub, free=required - 1)
        assert short["ok"] is False and short["shortfall_bytes"] == 1

        plant(hub, "org/missing", b"1234", ready=False)
        partial = plan_weight_space([missing], hub=hub, free=required)
        assert partial["items"][0]["cached_bytes"] == 4
        assert partial["items"][0]["need_bytes"] == 9_996

        plant(hub, "org/ready", b"full-weight-bytes", ready=True)
        ready_spec = {**missing, "repo": "org/ready", "label": "已齐"}
        done = plan_weight_space([ready_spec], hub=hub, free=0)
        assert done["ok"] is True and done["items"] == []

        stub = {**missing, "repo": "org/stub", "approx_bytes": 100}
        plant(hub, "org/stub", b"x" * 150, ready=False)
        leftover = plan_weight_space([stub], hub=hub, free=0)
        assert leftover["items"][0]["need_bytes"] == 10

        unknown = {**missing, "repo": "org/custom", "label": "自定义", "approx_bytes": 0}
        floor = UNKNOWN_BYTES + RESERVE_BYTES
        assert plan_weight_space([unknown], hub=hub, free=floor)["ok"] is True
        low = plan_weight_space([unknown], hub=hub, free=floor - 1)
        assert low["ok"] is False and "体积未知" in low["message"]
