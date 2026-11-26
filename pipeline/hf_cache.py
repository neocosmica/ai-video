from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def hub_dir() -> Path:
    """本机 Hugging Face hub 缓存。与 huggingface_hub 的默认位置一致。"""
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dir_name(repo: str) -> str:
    """``org/name`` → ``models--org--name``。拒绝 ``..``，避免清缓存时走出 hub。"""
    text = repo.strip().strip("/")
    parts = text.split("/")
    if not text or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"非法仓库名 {repo!r}")
    return "models--" + text.replace("/", "--")


def repo_from_cache_dir(name: str) -> str | None:
    """``models--org--name`` → ``org/name``。对不上往返规则的目录名不当成仓库。"""
    prefix = "models--"
    if not name.startswith(prefix) or "/" in name or "\\" in name:
        return None
    rest = name[len(prefix) :]
    if not rest:
        return None
    repo = rest.replace("--", "/", 1)
    try:
        if repo_dir_name(repo) != name:
            return None
    except ValueError:
        return None
    return repo


def list_model_dirs(hub: Path | None = None) -> list[tuple[str, Path]]:
    """hub 里全部模型目录。跳过符号链接、数据集和锁目录。"""
    root = hub or hub_dir()
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    found: list[tuple[str, Path]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        repo = repo_from_cache_dir(child.name)
        if repo is None or child.is_symlink() or not child.is_dir():
            continue
        try:
            resolved = child.resolve()
        except OSError:
            continue
        if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
            continue
        found.append((repo, child))
    found.sort(key=lambda item: item[0])
    return found


def resolve_snapshot(repo_id: str, *, hub: Path | None = None) -> Path | None:
    """把 HuggingFace repo id 解析到本机 hub 缓存里的 snapshot 目录。

    不调用 huggingface_hub，避免镜像/配额下 snapshot_download 卡住。
    ``hub`` 只给测试用；平时走 ``hub_dir()``。
    """
    repo = repo_id.strip().strip("/")
    if not repo:
        return None
    if hub is None:
        local = Path(repo)
        if local.exists():
            return local.resolve()
    try:
        cache_name = repo_dir_name(repo)
    except ValueError:
        return None

    cache = (hub or hub_dir()) / cache_name
    snaps = cache / "snapshots"
    if not snaps.is_dir():
        return None

    ref = cache / "refs" / "main"
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


def locate_repo_dir(repo: str, hub: Path | None = None) -> Path | None:
    """hub 里这个仓库的缓存目录。没有、或是指向外部的符号链接，则返回 None。"""
    try:
        name = repo_dir_name(repo)
    except ValueError:
        return None
    root = (hub or hub_dir()).resolve()
    candidate = root / name
    if not candidate.exists() or candidate.is_symlink() or not candidate.is_dir():
        return None
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        return None
    return candidate


def cache_nbytes(root: Path) -> int:
    """目录占的磁盘字节。符号链接不跟进去，硬链接只算一次。"""
    total = 0
    seen: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept: list[str] = []
        for name in dirnames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


def remove_repo_cache(repo: str, hub: Path | None = None) -> int:
    """删掉这个仓库的 hub 目录（含下载锁）。返回释放的字节数。"""
    root = (hub or hub_dir()).resolve()
    path = locate_repo_dir(repo, root)
    if path is None:
        return 0
    nbytes = cache_nbytes(path)
    shutil.rmtree(path)
    lock = root / ".locks" / path.name
    if lock.exists() and not lock.is_symlink() and lock.is_dir():
        resolved = lock.resolve()
        if resolved != root and resolved.is_relative_to(root):
            shutil.rmtree(lock)
    return nbytes


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


def _approx_bytes(repo: str) -> int:
    """目录里这份仓库的磁盘预估。自定义仓库没有约数时返回 0。"""
    from pipeline.backends.catalog import STAGES, models_for

    for stage in STAGES:
        for model in models_for(stage):
            for item in model.downloads:
                if item.repo == repo and item.approx_bytes > 0:
                    return item.approx_bytes
    return 0


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
    from pipeline.disk_space import plan_weight_space

    space = plan_weight_space(
        [
            {
                "repo": repo,
                "label": label,
                "approx_bytes": _approx_bytes(repo),
                "required_files": required_files,
                "weight_globs": weight_globs,
            }
        ]
    )
    if not space["ok"]:
        raise SystemExit(space["message"])
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
