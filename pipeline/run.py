from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.util import resolve


# 先对白/配乐，再 A2V 条件生成视频（口型跟音频）
STAGES = ("screenplay", "images", "audio", "videos", "assemble")


def install_storyboard(out: str | Path, storyboard: str | None) -> None:
    """把外部分镜拷进 out/storyboard.json。Agent 只写这一份，后续全靠脚本。"""
    dest_dir = resolve(out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "storyboard.json"
    if not storyboard:
        if not dest.exists():
            raise SystemExit(
                f"还没有分镜：{dest}\n"
                "先由 Agent 写好 storyboard.json，或传 --storyboard <路径>。"
            )
        return
    src = resolve(storyboard)
    if not src.exists():
        raise SystemExit(f"找不到分镜：{src}")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        print(f"已放入分镜：{src} → {dest}")
    else:
        print(f"分镜已在输出目录：{dest}")


def cmd_run(args: argparse.Namespace) -> None:
    start = getattr(args, "start", "screenplay")
    if start not in STAGES:
        raise SystemExit(f"--from 只能是 {', '.join(STAGES)}")
    out = resolve(args.out)
    if start == "screenplay" or getattr(args, "storyboard", None):
        install_storyboard(out, getattr(args, "storyboard", None))
    failed_videos = False
    for stage in STAGES[STAGES.index(start) :]:
        cmd = [sys.executable, "-m", "pipeline", stage, "--out", str(out)]
        if args.config:
            cmd += ["--config", str(resolve(args.config))]
        if args.force:
            cmd.append("--force")
        if stage == "screenplay":
            # 分镜 JSON 已写好；这里只校验。小说路径用于对白覆盖检查。
            if getattr(args, "novel", None):
                cmd += ["--novel", str(resolve(args.novel))]
            if getattr(args, "max_shots", None):
                cmd += ["--max-shots", str(args.max_shots)]
        if getattr(args, "only", None) and stage in {"images", "videos"}:
            cmd += ["--only", args.only]
        print(f"\n== {stage} ==")
        code = subprocess.call(cmd)
        if code != 0 and stage == "videos":
            failed_videos = True
            continue
        if code != 0:
            raise SystemExit(code)
    final = out / "final.mp4"
    if final.exists():
        print(f"\n成片：{final}")
    if failed_videos:
        raise SystemExit("视频阶段有镜头失败。能用的镜头已经合成进成片。")


def cmd_produce(args: argparse.Namespace) -> None:
    """一键：分镜 → 校验 → 图 → 对白配乐 → A2V 视频 → 成片。不再需要 Agent。"""
    args.start = "screenplay"
    cmd_run(args)
