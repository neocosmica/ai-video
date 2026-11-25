"""跨平台差异集中处理：Mac（Apple Silicon / MLX）与 Windows·Linux（NVIDIA / CUDA）。

设计原则
--------
1. **只读探测，不做决策**：这里只回答「本机是什么」，选后端由 ``pipeline.backends`` 决定。
2. **不改变 Mac 行为**：所有 Windows 分支都在平台判断之后，macOS 走原有逻辑。
3. 尽量少依赖：Windows 内存探测用 ctypes，避免引入 psutil。

对应关系（Mac → Windows）
-------------------------
| 能力            | macOS                          | Windows                    |
|----------------|--------------------------------|----------------------------|
| 总内存          | os.sysconf(SC_PHYS_PAGES)      | GlobalMemoryStatusEx       |
| 可用内存百分比   | memory_pressure                | GlobalMemoryStatusEx       |
| venv 解释器      | .venv/bin/python               | .venv/Scripts/python.exe   |
| 控制台脚本       | .venv/bin/ltx-2-mlx            | .venv/Scripts/ltx-2-mlx.exe|
| 杀进程树         | os.killpg(pid, SIGTERM)        | taskkill /F /T /PID        |
| 壳命令链         | /bin/bash -c "a && b"          | 分阶段直起子进程            |
| 字幕字体目录      | /System/Library/Fonts/Supplemental | C:\\Windows\\Fonts      |
| 字幕字体名        | Songti SC                      | Microsoft YaHei            |
| 重型 venv 落点    | ~/Library/Caches/novel-video   | 原地（Windows 建符号链接需特权）|
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# venv / 解释器路径
# ---------------------------------------------------------------------------


def venv_bin_dir(venv_root: Path) -> Path:
    """venv 内可执行文件目录：POSIX 是 ``bin``，Windows 是 ``Scripts``。"""
    return venv_root / ("Scripts" if IS_WIN else "bin")


def python_in_venv(venv_root: Path) -> Path:
    """venv 内的解释器路径（不检查是否存在）。"""
    return venv_bin_dir(venv_root) / ("python.exe" if IS_WIN else "python")


def console_script(venv_root: Path, name: str) -> Path:
    """venv 内的控制台脚本路径，Windows 自动补 ``.exe``。"""
    if IS_WIN and not name.lower().endswith(".exe"):
        name = name + ".exe"
    return venv_bin_dir(venv_root) / name


# ---------------------------------------------------------------------------
# 内存
# ---------------------------------------------------------------------------


def total_memory_bytes() -> int:
    """本机物理内存字节数。

    macOS/Linux 用 ``os.sysconf``；Windows 用 ``GlobalMemoryStatusEx``（ctypes，
    不引入 psutil）。失败时抛 ``SystemExit``，与 Mac 端行为一致。
    """
    if IS_WIN:
        return _win_memory()["total"]
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError) as exc:
        raise SystemExit(f"无法读取本机总内存：{exc}") from exc
    if pages <= 0 or page_size <= 0:
        raise SystemExit("无法读取本机总内存（SC_PHYS_PAGES / SC_PAGE_SIZE）")
    return int(pages) * int(page_size)


def free_memory_percent() -> float | None:
    """当前可用物理内存百分比；拿不到返回 ``None``。

    macOS 走 ``memory_pressure``（原逻辑）；Windows 用 ``GlobalMemoryStatusEx``。
    """
    if IS_WIN:
        mem = _win_memory()
        total = mem["total"]
        return (mem["avail"] / total * 100.0) if total else None
    if IS_MAC:
        return _mac_free_percent()
    # Linux：读 /proc/meminfo
    try:
        info = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    total = avail = None
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            total = float(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail = float(line.split()[1])
    if not total or avail is None:
        return None
    return avail / total * 100.0


def _mac_free_percent() -> float | None:
    try:
        out = subprocess.check_output(
            ["memory_pressure"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "System-wide memory free percentage" in line:
            raw = line.rsplit(":", 1)[-1].strip().rstrip("%")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _win_memory() -> dict[str, int]:
    """Windows 物理内存总量 / 可用量（字节）。"""
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise SystemExit("无法读取本机总内存（GlobalMemoryStatusEx 失败）")
    return {"total": int(stat.ullTotalPhys), "avail": int(stat.ullAvailPhys)}


# ---------------------------------------------------------------------------
# GPU（NVIDIA）
# ---------------------------------------------------------------------------


def nvidia_vram_bytes() -> int | None:
    """用 ``nvidia-smi`` 查第一块 NVIDIA 卡的显存总量（字节）。

    不 import torch：``doctor`` 在未装 torch 的环境也要能报硬件。
    没有 nvidia-smi 或执行失败返回 ``None``。
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = (out or "").strip().splitlines()
    if not line:
        return None
    try:
        mib = float(line[0].strip())
    except ValueError:
        return None
    return int(mib * 1024 * 1024)


# ---------------------------------------------------------------------------
# 进程
# ---------------------------------------------------------------------------


def kill_process_tree(pid: int, *, pgid: int | None = None) -> None:
    """终止一个子进程及其整棵进程树。

    Windows：``taskkill /F /T /PID``（``/T`` 连带子进程）。
    POSIX：优先杀进程组，退化到单 pid。
    """
    if IS_WIN:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(int(pid))])
        return
    import signal

    try:
        target = int(pgid) if pgid else os.getpgid(int(pid))
        os.killpg(target, signal.SIGTERM)
    except ProcessLookupError:
        raise
    except Exception:
        os.kill(int(pid), signal.SIGTERM)


# ---------------------------------------------------------------------------
# 字体（字幕）
# ---------------------------------------------------------------------------


def font_dir() -> Path:
    """ffmpeg ``fontsdir=`` 用的字体目录。"""
    if IS_WIN:
        win = Path(os.environ.get("WINDIR", r"C:\Windows"))
        return win / "Fonts"
    if IS_MAC:
        return Path("/System/Library/Fonts/Supplemental")
    return Path("/usr/share/fonts/truetype")


def subtitle_font_name() -> str:
    """ASS 字幕的 ``Fontname``。"""
    if IS_WIN:
        return "Microsoft YaHei"
    if IS_MAC:
        return "Songti SC"
    return "Noto Sans CJK SC"


# ---------------------------------------------------------------------------
# 用户目录
# ---------------------------------------------------------------------------


def user_cache_dir() -> Path:
    """重型依赖（venv、权重链接）的落点。"""
    if IS_WIN:
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "novel-video"
    if IS_MAC:
        return Path.home() / "Library" / "Caches" / "novel-video"
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "novel-video"


def app_support_dir(app_name: str = "AI短视频") -> Path:
    """日志等非缓存产物的落点。"""
    if IS_WIN:
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / app_name
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / app_name
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / app_name


# ---------------------------------------------------------------------------
# 汇总（给 doctor / 网页用）
# ---------------------------------------------------------------------------


def describe() -> dict:
    """一眼看清当前平台与可用的加速后端。"""
    info = {
        "platform": sys.platform,
        "is_windows": IS_WIN,
        "is_mac": IS_MAC,
        "is_linux": IS_LINUX,
        "python": sys.version.split()[0],
        "total_memory_gb": None,
        "nvidia_vram_gb": None,
        "has_nvidia_smi": shutil.which("nvidia-smi") is not None,
        "font_dir": str(font_dir()),
        "subtitle_font": subtitle_font_name(),
        "cache_dir": str(user_cache_dir()),
        "app_support_dir": str(app_support_dir()),
    }
    try:
        info["total_memory_gb"] = round(total_memory_bytes() / 1024**3, 2)
    except SystemExit:
        pass
    vram = nvidia_vram_bytes()
    if vram:
        info["nvidia_vram_gb"] = round(vram / 1024**3, 2)
    return info
