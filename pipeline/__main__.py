from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.mirrors import apply_mirrors

apply_mirrors()

from pipeline.assemble import cmd_assemble
from pipeline.audio import cmd_audio
from pipeline.config import Settings
from pipeline.images import cmd_images
from pipeline.run import STAGES, cmd_run
from pipeline.screenplay import cmd_screenplay
from pipeline.util import ROOT, chunk_novel, extract_json
from pipeline.video import cmd_videos


def cmd_doctor(_: argparse.Namespace) -> None:
    settings = Settings.load()
    print(f"项目目录：{ROOT}")
    print("分镜：Cursor Agent 写 storyboard.json（本机不加载聊天 LLM）")
    print(f"文生图：{settings.image_model}，{settings.image_quantize}-bit，{settings.image_width}x{settings.image_height}")
    print(
        f"图生视频：{settings.video_model}，{settings.video_width}x{settings.video_height}，"
        f"{settings.video_frames} 帧 / {settings.video_fps} fps"
    )
    print("提示词编码器：LTX 内置 Gemma 3 12B 4-bit（不是聊天用的 VLM）")
    print(f"对白 TTS：{settings.tts_model}")
    print(f"配乐：{settings.music_model}")
    print(f"HF 镜像：{os.environ.get('HF_ENDPOINT')}")
    print(f"PyPI 镜像：{os.environ.get('UV_INDEX_URL')}")
    print(f"禁用 XET：{os.environ.get('HF_HUB_DISABLE_XET')}")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    print(f"uv：{uv}（{'存在' if Path(uv).exists() or shutil.which('uv') else '缺失'}）")
    print(f"ffmpeg：{settings.ffmpeg}（{'存在' if settings.ffmpeg.exists() else '缺失'}）")
    print(f"LTX 代码：{settings.ltx_dir}（{'存在' if settings.ltx_dir.exists() else '缺失'}）")
    print(f"MusicGen 代码：{settings.musicgen_dir}（{'存在' if settings.musicgen_dir.exists() else '缺失'}）")
    try:
        import mlx.core as mx

        print(f"MLX Metal：{mx.metal.is_available()}")
    except Exception as exc:
        print(f"MLX 还不可用：{exc}")
    if settings.ffmpeg.exists():
        subprocess.run([str(settings.ffmpeg), "-version"], check=False)


def cmd_selftest(_: argparse.Namespace) -> None:
    parts = chunk_novel("第一段。\n\n第二段。")
    assert parts == ["第一段。\n\n第二段。"], parts
    long = "甲" * 1000 + "\n\n" + "乙" * 1000
    chunks = chunk_novel(long, limit=1200)
    assert len(chunks) == 2, chunks
    data = extract_json("说明\n```json\n{\"shots\": []}\n```")
    assert data == {"shots": []}
    print("selftest ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把小说本地做成竖屏短剧")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser, novel: bool = False) -> None:
        p.add_argument("--config", default=None, help="配置文件，默认项目里的 config.json")
        p.add_argument("--out", default="output/demo", help="这一部片子的输出目录")
        p.add_argument("--force", action="store_true", help="重做已有的分镜、静帧和镜头")
        if novel:
            p.add_argument("--novel", default="input/novel.txt")
            p.add_argument("--max-shots", type=int, default=None)

    screenplay = sub.add_parser("screenplay", help="校验 Agent 写好的分镜 JSON")
    add_common(screenplay, novel=True)
    screenplay.set_defaults(func=cmd_screenplay)

    images = sub.add_parser("images", help="角色参考图和分镜静帧")
    add_common(images)
    images.add_argument("--only", default=None, help="只做这些镜头，例如 s001,s002")
    images.set_defaults(func=cmd_images)

    videos = sub.add_parser("videos", help="按静帧生成镜头视频")
    add_common(videos)
    videos.add_argument("--only", default=None)
    videos.set_defaults(func=cmd_videos)

    audio = sub.add_parser("audio", help="本地生成对白和配乐")
    add_common(audio)
    audio.set_defaults(func=cmd_audio)

    assemble = sub.add_parser("assemble", help="拼接镜头、混音并烧录字幕")
    add_common(assemble)
    assemble.set_defaults(func=cmd_assemble)

    run = sub.add_parser("run", help="从小说一路做到成片")
    add_common(run, novel=True)
    run.add_argument("--from", dest="start", default="screenplay", choices=STAGES)
    run.add_argument("--only", default=None, help="只重做列出的镜头，用于 images 和 videos")
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser("doctor", help="检查本机环境")
    doctor.set_defaults(func=cmd_doctor)
    test = sub.add_parser("selftest", help="检查不依赖模型的逻辑")
    test.set_defaults(func=cmd_selftest)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
