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


def file_ready(path: Path) -> bool:
    """真实文件且非空。断掉的符号链接、0 字节残片算没有。"""
    if not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def iter_weight_files(snap: Path, pattern: str) -> list[Path]:
    """snapshot 里匹配的权重。``model*.safetensors`` 不含 ``*.index.json``。"""
    if not snap.is_dir():
        return []
    return [path for path in sorted(snap.glob(pattern)) if file_ready(path)]


def snapshot_ready(
    snap: Path | None,
    *,
    required_files: tuple[str, ...] = (),
    weight_globs: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """缓存目录能不能当成本地模型用。缺权重时返回原因，避免 mlx 只看到空目录。"""
    if snap is None:
        return False, "未下载"
    for name in required_files:
        if not file_ready(snap / name):
            return False, f"缺 {name}"
    for pattern in weight_globs:
        if not iter_weight_files(snap, pattern):
            return False, f"缺 {pattern}"
    return True, str(snap)


def ensure_snapshot(
    repo_id: str,
    *,
    required_files: tuple[str, ...] = (),
    weight_globs: tuple[str, ...] = (),
    label: str = "模型",
) -> Path:
    """返回齐备的 snapshot。目录在但缺 safetensors 时补下载，不把残缺目录交给加载器。"""
    from pipeline.mirrors import apply_mirrors

    apply_mirrors()
    repo = repo_id.strip().strip("/")

    def ready(snap: Path | None) -> tuple[bool, str]:
        return snapshot_ready(
            snap,
            required_files=required_files,
            weight_globs=weight_globs,
        )

    snap = resolve_snapshot(repo)
    ok, detail = ready(snap)
    if ok and snap is not None:
        return snap
    if snap is not None:
        print(f"{label}缓存不完整（{detail}），正在补下：{repo}", flush=True)
    else:
        print(f"正在下载{label}：{repo}", flush=True)
    from huggingface_hub import snapshot_download

    downloaded = Path(snapshot_download(repo))
    ok, detail = ready(downloaded)
    if not ok:
        raise SystemExit(
            f"{label}仍不可用：{detail}\n"
            f"目录：{downloaded}\n"
            "请检查网络后重开，或删掉该 Hugging Face 缓存目录再下载。"
        )
    return downloaded
