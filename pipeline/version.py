"""应用版本号。纯文本 ``VERSION``，打包时末位加一。"""

from __future__ import annotations

from pathlib import Path

from pipeline.util import ROOT


def read_version(root: Path | None = None) -> str:
    path = (root or ROOT) / "VERSION"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def bump_version_text(current: str) -> str:
    """末位数字加一。空串从 1.0.0 起算，加一后是 1.0.1。"""
    text = (current or "").strip() or "1.0.0"
    i = len(text)
    while i > 0 and text[i - 1].isdigit():
        i -= 1
    if i == len(text):
        return f"{text}.1"
    return f"{text[:i]}{int(text[i:]) + 1}"


def bump_version_file(root: Path | None = None) -> str:
    root = root or ROOT
    path = root / "VERSION"
    new = bump_version_text(read_version(root))
    path.write_text(new + "\n", encoding="utf-8")
    return new
