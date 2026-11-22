from __future__ import annotations

from pathlib import Path


def resolve_snapshot(repo_id: str) -> Path | None:
    """把 HuggingFace repo id 解析到本机 hub 缓存里的 snapshot 目录。

    不调用 huggingface_hub，避免镜像/配额下 snapshot_download 卡住。
    """
    repo = repo_id.strip().strip("/")
    if not repo:
        return None
    local = Path(repo)
    if local.exists():
        return local.resolve()

    cache_name = "models--" + repo.replace("/", "--")
    hub = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    snaps = hub / "snapshots"
    if not snaps.is_dir():
        return None

    ref = hub / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        snap = snaps / rev
        if snap.is_dir():
            return snap.resolve()

    ranked = sorted(
        (p for p in snaps.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return ranked[0].resolve() if ranked else None
