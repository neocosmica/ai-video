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
from pipeline.run import STAGES, cmd_produce, cmd_run
from pipeline.screenplay import cmd_screenplay
from pipeline.screenplay import extract_novel_dialogues
from pipeline.util import ROOT, chunk_novel, extract_json
from pipeline.video import cmd_videos


def cmd_doctor(_: argparse.Namespace) -> None:
    from pipeline.runtime import (
        cache_root,
        ltx_cli,
        main_venv_path,
        ltx_venv_path,
        musicgen_venv_path,
        timed_import_transformers,
    )

    settings = Settings.load()
    print(f"项目目录：{ROOT}")
    print(f"venv 缓存根：{cache_root()}")
    for label, path in (
        ("主 .venv", ROOT / ".venv"),
        ("LTX .venv", settings.ltx_dir / ".venv"),
        ("MusicGen .venv", settings.musicgen_dir / ".venv"),
    ):
        if path.is_symlink():
            print(f"{label}：符号链接 → {path.resolve()}")
        elif path.exists():
            print(f"{label}：实目录（应迁到 Caches）{path}")
        else:
            print(f"{label}：缺失")
    print(f"主 venv 目标：{main_venv_path()}")
    print(f"LTX venv 目标：{ltx_venv_path()}")
    print(f"MusicGen venv 目标：{musicgen_venv_path()}")
    print("分镜：Agent 只写 storyboard.json；成片用 produce 一键跑完（本机不加载聊天 LLM）")
    print(f"文生图：{settings.image_model}，{settings.image_quantize}-bit，{settings.image_width}x{settings.image_height}")
    print(
        f"视频：{settings.video_model}，{settings.video_width}x{settings.video_height}，"
        f"默认 {settings.video_frames} 帧 / {settings.video_fps} fps；"
        f"有对白走 a2v（{settings.video_frames_min}–{settings.video_frames_max} 帧跟 TTS）"
    )
    print("提示词编码器：LTX 内置 Gemma 3 12B 4-bit（不是聊天用的 VLM）")
    print(f"对白 TTS：{settings.tts_model}（先于视频，供 A2V 条件）")
    print(f"配乐：{settings.music_model}")
    print(f"HF 镜像：{os.environ.get('HF_ENDPOINT')}")
    print(f"PyPI 镜像：{os.environ.get('UV_INDEX_URL')}")
    print(f"禁用 XET：{os.environ.get('HF_HUB_DISABLE_XET')}")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    print(f"uv：{uv}（{'存在' if Path(uv).exists() or shutil.which('uv') else '缺失'}）")
    print(f"ffmpeg：{settings.ffmpeg}（{'存在' if settings.ffmpeg.exists() else '缺失'}）")
    print(f"LTX 代码：{settings.ltx_dir}（{'存在' if settings.ltx_dir.exists() else '缺失'}）")
    from pipeline.hf_cache import resolve_snapshot

    ltx_snap = resolve_snapshot(settings.video_model)
    if ltx_snap is None:
        print(f"LTX 权重缓存：未找到 {settings.video_model}")
    else:
        for name in ("audio_vae.safetensors", "vocoder.safetensors"):
            path = ltx_snap / name
            print(f"LTX A2V 依赖 {name}：{'有' if path.exists() else '缺（对白镜会失败，请补拉或重跑 setup）'}")
    print(f"MusicGen 代码：{settings.musicgen_dir}（{'存在' if settings.musicgen_dir.exists() else '缺失'}）")
    gen_py = settings.musicgen_dir / "generate.py"
    print(f"MusicGen generate.py：{'存在' if gen_py.exists() else '缺失'}")
    patcher = ROOT / "scripts" / "apply_ltx_gemma_patch.py"
    print(f"LTX Gemma 补丁脚本：{'存在' if patcher.exists() else '缺失'}")
    marker = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"
    blocks = (
        settings.ltx_dir
        / "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/blocks.py"
    )
    if blocks.exists():
        patched = marker in blocks.read_text(encoding="utf-8")
        print(f"LTX Gemma 分阶段补丁：{'已打' if patched else '未打（请重跑 setup.sh）'}")
    else:
        print("LTX Gemma 分阶段补丁：vendor 未就绪")
    try:
        cli = ltx_cli(settings.ltx_dir)
        print(f"LTX CLI：{cli}（存在）")
    except SystemExit as exc:
        print(f"LTX CLI：缺失（{exc}）")
    try:
        import mlx.core as mx

        print(f"MLX Metal：{mx.metal.is_available()}")
    except Exception as exc:
        print(f"MLX 还不可用：{exc}")
    py = main_venv_path() / "bin" / "python"
    if py.exists():
        try:
            seconds = timed_import_transformers(py, timeout_s=90.0)
            print(f"transformers 导入：{seconds:.2f}s（>20s 仍偏慢）")
        except Exception as exc:
            print(f"transformers 导入失败：{exc}")
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
    lines = extract_novel_dialogues('甲说：“你好。”乙回「在呢」。又说：“你好。”')
    assert lines == ["你好。", "在呢"], lines
    from pipeline.hf_cache import resolve_snapshot
    from pipeline.runtime import main_venv_path, musicgen_venv_path, ltx_venv_path

    assert resolve_snapshot("") is None
    assert (ROOT / "scripts" / "apply_ltx_gemma_patch.py").exists(), "缺少 LTX 补丁脚本"
    assert (ROOT / "input" / "demo_storyboard.json").exists(), "缺少演示分镜"
    assert (ROOT / "input" / "novel.txt").exists(), "缺少演示小说"
    from pipeline.run import STAGES, install_storyboard
    from pipeline.util import frames_for_duration
    import tempfile

    assert STAGES == ("screenplay", "images", "audio", "videos", "assemble"), STAGES
    assert frames_for_duration(2.0, 24, min_frames=25, max_frames=97) == 49
    assert frames_for_duration(0.2, 24, min_frames=25, max_frames=97) == 25
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "storyboard.json"
        install_storyboard(tmp, str(ROOT / "input" / "demo_storyboard.json"))
        assert dest.exists(), "produce 应能把分镜拷进输出目录"
    for label, path in (
        ("main", main_venv_path()),
        ("ltx", ltx_venv_path()),
        ("musicgen", musicgen_venv_path()),
    ):
        # setup 之后应有；纯逻辑自检时允许尚未安装
        if path.exists() and not (path / "bin" / "python").exists():
            raise AssertionError(f"{label} venv 损坏：{path}")
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

    screenplay = sub.add_parser("screenplay", help="校验已写好的分镜 JSON")
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

    def add_storyboard(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--storyboard",
            default=None,
            help="外部分镜 JSON；会拷进 --out/storyboard.json（演示默认可用 input/demo_storyboard.json）",
        )

    produce = sub.add_parser(
        "produce",
        help="一键成片：分镜 → 校验 → 图 → 对白配乐 → A2V 视频 → final.mp4（不再需要 Agent）",
    )
    add_common(produce, novel=True)
    add_storyboard(produce)
    produce.add_argument("--only", default=None, help="只重做列出的镜头，用于 images 和 videos")
    produce.set_defaults(func=cmd_produce, start="screenplay")

    run = sub.add_parser("run", help="按阶段串跑（默认从 screenplay 起，等同 produce）")
    add_common(run, novel=True)
    add_storyboard(run)
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
