from __future__ import annotations

import argparse
import atexit
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.util import resolve

from pipeline.projects import artifacts_sub

# 对白先于视频（A2V 口型要音频条件）；配乐放在视频之后，此时成片长度已经确定
STAGES = ("screenplay", "images", "audio", "videos", "music", "assemble")


def _install_live_hooks(out: Path, stages: list[str]) -> None:
    """CLI produce/run 写 work/webui/live.json，网页后开也能挂上进度。"""
    from pipeline.live import clear_live, mark_stage, write_live

    write_live(
        out,
        {
            "status": "running",
            "stages": stages,
            "stage_status": {s: ("running" if s == stages[0] else "pending") for s in stages},
            "current_stage": stages[0] if stages else None,
            "source": "cli",
            "detail": "CLI produce",
        },
    )

    def _atexit() -> None:
        # 正常路径会 clear_live；异常退出时尽量标失败，避免网页一直显示运行中
        from pipeline.live import read_live

        cur = read_live(out)
        if cur and cur.get("status") == "running":
            clear_live(out, status="failed", returncode=1)

    atexit.register(_atexit)


def install_storyboard(out: str | Path, storyboard: str | None, *, force: bool = False) -> None:
    """把外部分镜拷进 out/storyboard.json。Agent 只写这一份，后续全靠脚本。

    分镜是「源文件」，留在项目顶层；过程产物进 out/work/。
    输出目录里已有分镜时默认不覆盖：audio 阶段可能按台词长度自动拆过镜，
    那份才是最新的；要覆盖请显式传 --force。
    """
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
    if src.resolve() == dest.resolve():
        print(f"分镜已在输出目录：{dest}")
        return
    if dest.exists() and not force:
        print(
            f"输出目录已有分镜，保留不覆盖：{dest}\n"
            "（audio 阶段可能已按台词长度自动拆过镜；确实要用源文件覆盖请加 --force）"
        )
        return
    shutil.copy2(src, dest)
    print(f"已放入分镜：{src} → {dest}")


def cmd_run(args: argparse.Namespace) -> None:
    start = getattr(args, "start", "screenplay")
    if start not in STAGES:
        raise SystemExit(f"--from 只能是 {', '.join(STAGES)}")
    until = getattr(args, "until", None)
    if until is not None and until not in STAGES:
        raise SystemExit(f"--until 只能是 {', '.join(STAGES)}")
    out = resolve(args.out)
    if start == "screenplay" or getattr(args, "storyboard", None):
        install_storyboard(
            out,
            getattr(args, "storyboard", None),
            force=bool(getattr(args, "force", False)),
        )
    failed_videos = False
    end_idx = STAGES.index(until) + 1 if until else len(STAGES)
    selected = list(STAGES[STAGES.index(start) : end_idx])
    from pipeline.live import clear_live, mark_stage

    _install_live_hooks(out, selected)
    exit_code = 0
    try:
        for stage in selected:
            mark_stage(out, stage, selected)
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
            if getattr(args, "only", None) and stage in {"images", "audio", "videos"}:
                cmd += ["--only", args.only]
            print(f"\n== {stage} ==")
            code = subprocess.call(cmd)
            if code != 0 and stage == "videos":
                failed_videos = True
                continue
            if code != 0:
                exit_code = code
                clear_live(out, status="failed", returncode=code)
                raise SystemExit(code)
        final = out / "final.mp4"
        if final.exists():
            print(f"\n成片：{final}")
        else:
            work_final = artifacts_sub(out, "final.mp4")
            if work_final.exists():
                print(f"\n成片（旧位置，建议移到 {final}）：{work_final}")
        if failed_videos:
            exit_code = 1
            clear_live(out, status="failed", returncode=1)
            raise SystemExit("视频阶段有镜头失败。能用的镜头已经合成进成片。")
        clear_live(out, status="success", returncode=0)
    except SystemExit:
        raise
    except Exception:
        clear_live(out, status="failed", returncode=exit_code or 1)
        raise


def cmd_produce(args: argparse.Namespace) -> None:
    """一键：分镜 → 校验 → 图 → 对白配乐 → A2V 视频 → 成片。不再需要 Agent。"""
    args.start = "screenplay"
    cmd_run(args)
