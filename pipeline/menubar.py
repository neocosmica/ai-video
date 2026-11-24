"""macOS 菜单栏图标：挂在网页控制台进程上。

用仓内编译好的 ``scripts/macos/menubar-helper``（Swift / AppKit），
菜单项：打开网页、复制地址、退出。退出会 SIGTERM 本进程，``serve`` 收尾停 HTTP。

非 Darwin、缺二进制、或 ``NOVEL_VIDEO_NO_MENUBAR=1`` 时直接跳过。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pipeline.util import ROOT

HELPER = ROOT / "scripts" / "macos" / "menubar-helper"
# 仅用专用模板图（黑白/透明）；彩色 app 图标做 template 会糊成一团，交给 helper 用 SF Symbol
ICON_TEMPLATE = ROOT / "scripts" / "macos" / "menubar-template.png"


def menubar_enabled() -> bool:
    if sys.platform != "darwin":
        return False
    if os.environ.get("NOVEL_VIDEO_NO_MENUBAR", "").strip() in ("1", "true", "yes"):
        return False
    return HELPER.is_file() and os.access(HELPER, os.X_OK)


def start_menubar(url: str, *, title: str = "AI短视频") -> subprocess.Popen | None:
    """拉起菜单栏助手；失败返回 None（调用方回退到纯 HTTP）。"""
    if not menubar_enabled():
        return None
    cmd = [str(HELPER), "--url", url, "--title", title, "--parent-pid", str(os.getpid())]
    if ICON_TEMPLATE.is_file():
        cmd.extend(["--icon", str(ICON_TEMPLATE)])
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"菜单栏助手启动失败（可忽略）：{exc}", flush=True)
        return None
