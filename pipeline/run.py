from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline.util import resolve


STAGES = ("screenplay", "images", "videos", "audio", "assemble")


def cmd_run(args: argparse.Namespace) -> None:
    if args.start not in STAGES:
        raise SystemExit(f"--from 只能是 {', '.join(STAGES)}")
    out = resolve(args.out)
    failed_videos = False
    for stage in STAGES[STAGES.index(args.start) :]:
        cmd = [sys.executable, "-m", "pipeline", stage, "--out", str(out)]
        if args.config:
            cmd += ["--config", str(resolve(args.config))]
        if args.force:
            cmd.append("--force")
        if stage == "screenplay":
            # 分镜 JSON 由 Cursor Agent 写好；这里只校验。小说路径可选，用来写回元数据。
            if args.novel:
                cmd += ["--novel", str(resolve(args.novel))]
            if args.max_shots:
                cmd += ["--max-shots", str(args.max_shots)]
        if args.only and stage in {"images", "videos"}:
            cmd += ["--only", args.only]
        print(f"\n== {stage} ==")
        code = subprocess.call(cmd)
        if code != 0 and stage == "videos":
            failed_videos = True
            continue
        if code != 0:
            raise SystemExit(code)
    if failed_videos:
        raise SystemExit("视频阶段有镜头失败。能用的镜头已经合成进成片。")
