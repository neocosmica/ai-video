"""ffmpeg / ffprobe 可执行校验与回退。

包内 ``tools/ffmpeg`` 若架构不对（x86_64 打到 arm64 机），仅 ``exists()``
会假阳性「已就绪」，真正跑视频时才 ``Bad CPU type``。这里统一：试跑
``-version`` + 架构提示，失败则回退 ``PATH`` 上的 Homebrew 原生版。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "AI短视频"


def server_pid_path() -> Path:
    return support_dir() / "server.pid"


def binary_runnable(path: Path, *, timeout: float = 10.0) -> tuple[bool, str]:
    """真正试跑；Errno 86（Bad CPU type）会落在 OSError。"""
    if not path.exists():
        return False, f"缺 {path}"
    if not os.access(path, os.X_OK):
        return False, f"{path} 不可执行"
    try:
        r = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        err = str(exc)
        if "Bad CPU type" in err or "Errno 86" in err or getattr(exc, "errno", None) == 86:
            host = os.uname().machine
            return (
                False,
                f"{path} 架构与本机（{host}）不符（Bad CPU type）。"
                "请换成 arm64 版，或安装 Rosetta：softwareupdate --install-rosetta",
            )
        return False, f"{path} 不可用：{err}"
    except subprocess.TimeoutExpired:
        return False, f"{path} -version 超时"
    if r.returncode == 0:
        arch = _lipo_arch(path)
        host = os.uname().machine
        if arch and arch != host:
            return (
                False,
                f"{path} 是 {arch}，本机 {host}（未装 Rosetta 会 Bad CPU type）",
            )
        note = f"{path}" + (f"（{arch}）" if arch else "")
        return True, note
    err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace").strip()
    if "Bad CPU type" in err or "Errno 86" in err:
        return False, f"{path} 架构与本机不符：{err[:200]}"
    return False, f"{path} 不可用：{err[:200] or f'exit {r.returncode}'}"


def _lipo_arch(path: Path) -> str:
    try:
        r = subprocess.run(
            ["/usr/bin/lipo", "-archs", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if r.returncode != 0:
        return ""
    parts = (r.stdout or "").split()
    return parts[0] if parts else ""


def resolve_ffmpeg_bins(
    bundled_ffmpeg: Path,
    bundled_ffprobe: Path,
) -> tuple[Path, Path, str]:
    """优先包内 tools/；不可用则回退 ``shutil.which``。

    返回 (ffmpeg, ffprobe, detail)。detail 供 doctor / 日志注明用了哪个。
    """
    ff_ok, ff_detail = binary_runnable(bundled_ffmpeg)
    fp_ok, fp_detail = binary_runnable(bundled_ffprobe)
    if ff_ok and fp_ok:
        return bundled_ffmpeg, bundled_ffprobe, f"包内 {ff_detail}"

    sys_ff = shutil.which("ffmpeg")
    sys_fp = shutil.which("ffprobe")
    # which 可能仍命中坏的 tools/（PATH 里 tools 在前）；显式跳过已判定失败的路径
    if sys_ff and Path(sys_ff).resolve() == bundled_ffmpeg.resolve():
        sys_ff = _which_skipping(bundled_ffmpeg, "ffmpeg")
    if sys_fp and Path(sys_fp).resolve() == bundled_ffprobe.resolve():
        sys_fp = _which_skipping(bundled_ffprobe, "ffprobe")

    if sys_ff and sys_fp:
        alt_ff, alt_fp = Path(sys_ff), Path(sys_fp)
        a_ok, a_detail = binary_runnable(alt_ff)
        b_ok, b_detail = binary_runnable(alt_fp)
        if a_ok and b_ok:
            why = ff_detail if not ff_ok else fp_detail
            return (
                alt_ff,
                alt_fp,
                f"回退系统 {a_detail}（包内不可用：{why}）",
            )

    # 都不可用：仍返回包内路径，让 probe 报红
    detail = ff_detail if not ff_ok else fp_detail
    if sys_ff or sys_fp:
        detail += f"；系统 which 亦不可用（ff={sys_ff!r} fp={sys_fp!r}）"
    return bundled_ffmpeg, bundled_ffprobe, detail


def _which_skipping(bad: Path, name: str) -> str | None:
    """在 PATH 里找 name，跳过与 bad 同 inode/路径的项。"""
    bad_res = bad.resolve()
    for raw in (os.environ.get("PATH") or "").split(os.pathsep):
        if not raw:
            continue
        cand = Path(raw) / name
        try:
            if not cand.is_file() or not os.access(cand, os.X_OK):
                continue
            if cand.resolve() == bad_res:
                continue
        except OSError:
            continue
        return str(cand)
    return None
