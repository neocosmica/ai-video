from __future__ import annotations

"""本机运行时护栏：重型 venv 落在 ~/Library/Caches，项目内只留符号链接。"""

import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline.util import ROOT

CACHE_ENV = "NOVEL_VIDEO_CACHE"
DEFAULT_CACHE = Path.home() / "Library" / "Caches" / "novel-video"


def cache_root() -> Path:
    raw = os.environ.get(CACHE_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_CACHE


def venv_root() -> Path:
    return cache_root() / "venvs"


def main_venv_path() -> Path:
    return venv_root() / "main"


def ltx_venv_path() -> Path:
    return venv_root() / "ltx"


def musicgen_venv_path() -> Path:
    return venv_root() / "musicgen"


def clear_uf_hidden(path: Path) -> None:
    if not path.exists():
        return
    subprocess.call(
        ["chflags", "-R", "nohidden", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _is_symlink_to(link: Path, target: Path) -> bool:
    if not link.is_symlink():
        return False
    try:
        return link.resolve() == target.resolve()
    except OSError:
        return False


def _venv_usable(path: Path) -> bool:
    return (path / "bin" / "python").exists()


def ensure_venv_symlink(link: Path, target: Path, *, label: str) -> Path:
    """保证项目内 .venv 指向 ~/Library/Caches/novel-video/venvs/。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_symlink_to(link, target) and _venv_usable(target):
        clear_uf_hidden(target)
        return target

    if link.is_symlink():
        link.unlink(missing_ok=True)

    if link.exists() and link.is_dir() and not link.is_symlink():
        if _venv_usable(target):
            stale = link.with_name(f"{link.name}.stale-{int(time.time())}")
            print(f"{label}：Caches 已有可用环境，移走旧目录 → {stale.name}")
            link.rename(stale)
        else:
            print(f"{label}：迁移 {link} → {target}")
            if target.exists():
                raise SystemExit(f"{label}：目标已存在但不可用：{target}。请删后重跑 setup.sh")
            link.rename(target)

    if not _venv_usable(target):
        raise SystemExit(
            f"{label} 在 {target} 不可用（缺少 bin/python）。"
            "请运行 scripts/setup.sh。"
        )

    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
        print(f"{label}：{link} → {target}")
    elif not _is_symlink_to(link, target):
        raise SystemExit(f"{label} 路径异常：{link} 不是指向 {target} 的符号链接")

    clear_uf_hidden(target)
    return target


def ensure_runtime_layout(
    *,
    ltx_dir: Path | None = None,
    musicgen_dir: Path | None = None,
    require_main: bool = True,
) -> dict[str, Path]:
    """幂等：主/LTX/MusicGen venv 都落到 Library/Caches，项目里只留符号链接。"""
    paths: dict[str, Path] = {}
    if require_main or (ROOT / ".venv").exists() or _venv_usable(main_venv_path()):
        paths["main"] = ensure_venv_symlink(ROOT / ".venv", main_venv_path(), label="主 venv")
    if ltx_dir is not None and ((ltx_dir / ".venv").exists() or _venv_usable(ltx_venv_path())):
        paths["ltx"] = ensure_venv_symlink(ltx_dir / ".venv", ltx_venv_path(), label="LTX venv")
    if musicgen_dir is not None and (
        (musicgen_dir / ".venv").exists() or _venv_usable(musicgen_venv_path())
    ):
        paths["musicgen"] = ensure_venv_symlink(
            musicgen_dir / ".venv", musicgen_venv_path(), label="MusicGen venv"
        )
    if "main" in paths:
        os.environ.setdefault("UV_PROJECT_ENVIRONMENT", str(main_venv_path()))
    return paths


def ltx_cli(ltx_dir: Path) -> Path:
    """直接调用 LTX venv 里的控制台脚本，避免每镜 `uv run` 触发环境解析/重建。"""
    ensure_runtime_layout(ltx_dir=ltx_dir)
    candidates = [
        ltx_venv_path() / "bin" / "ltx-2-mlx",
        ltx_dir / ".venv" / "bin" / "ltx-2-mlx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        f"找不到 ltx-2-mlx 可执行文件。请运行 scripts/setup.sh 安装到 {ltx_venv_path()}"
    )


def ltx_python(ltx_dir: Path) -> Path:
    ensure_runtime_layout(ltx_dir=ltx_dir)
    for path in (ltx_venv_path() / "bin" / "python", ltx_dir / ".venv" / "bin" / "python"):
        if path.exists():
            return path
    raise SystemExit(f"找不到 LTX Python：{ltx_venv_path() / 'bin' / 'python'}")


def musicgen_python(musicgen_dir: Path) -> Path:
    """MusicGen venv 的 Python。generate.py 依赖仓库 cwd，不要靠 console script。"""
    ensure_runtime_layout(musicgen_dir=musicgen_dir)
    for path in (
        musicgen_venv_path() / "bin" / "python",
        musicgen_dir / ".venv" / "bin" / "python",
    ):
        if path.exists():
            return path
    raise SystemExit(
        f"找不到 MusicGen Python：{musicgen_venv_path() / 'bin' / 'python'}。"
        "请运行 scripts/setup.sh。"
    )


def timed_import_transformers(python: str | Path, *, timeout_s: float = 120.0) -> float:
    """子进程计时 import transformers；超时或失败抛错。"""
    code = (
        "import time; t=time.time(); import transformers; "
        "print(f'{time.time()-t:.3f}', transformers.__version__)"
    )
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"import transformers 超过 {timeout_s:.0f}s 仍未完成。"
            f"请确认 venv 在 {venv_root()}（运行 scripts/setup.sh）。"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "import transformers 失败：\n"
            + (proc.stderr or proc.stdout or f"exit {proc.returncode}")
        )
    line = (proc.stdout or "").strip().splitlines()[-1]
    seconds = float(line.split()[0])
    wall = time.time() - t0
    return min(seconds, wall)


def preflight(
    *,
    ltx_dir: Path | None = None,
    python: str | Path | None = None,
    timeout_s: float = 120.0,
    soft: bool = False,
) -> None:
    """图像/视频重型阶段前调用：布局检查 + transformers 导入冒烟。"""
    ensure_runtime_layout(ltx_dir=ltx_dir)
    if python is not None:
        py = str(python)
    else:
        main_py = main_venv_path() / "bin" / "python"
        py = str(main_py if main_py.exists() else sys.executable)
    print(f"预检：用 {py} 计时 import transformers …", flush=True)
    try:
        seconds = timed_import_transformers(py, timeout_s=timeout_s)
    except RuntimeError as exc:
        if soft:
            print(f"警告：{exc}")
            return
        raise SystemExit(str(exc)) from exc
    print(f"预检通过：transformers {seconds:.1f}s", flush=True)
    if seconds > 20:
        print(
            f"警告：transformers 导入仍偏慢（{seconds:.1f}s）。"
            "若反复出现，确认 .venv 指向 Library/Caches 且环境完整。"
        )
