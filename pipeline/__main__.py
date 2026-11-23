from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.mirrors import apply_mirrors

apply_mirrors()

from pipeline.assemble import cmd_assemble
from pipeline.audio import cmd_audio, cmd_music
from pipeline.config import Settings
from pipeline.images import cmd_images
from pipeline.run import STAGES, cmd_produce, cmd_run
from pipeline.screenplay import cmd_screenplay
from pipeline.screenplay import (
    assert_dialogue_coverage,
    count_spoken_chars,
    dialogue_char_limit,
    extract_novel_dialogues,
)
from pipeline.util import ROOT, chunk_novel, extract_json, resolve
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
    if settings.video_frames_auto and settings.budget_detail:
        d = settings.budget_detail
        print(
            f"单镜上限（auto）：{settings.video_frames_max} 帧 ≈ {d.get('seconds', '?')} s"
            f"（总内存 {d.get('total_gib', '?')} GB → {d.get('tier_gib', '?')} GB 档，"
            f"token 预算 {d.get('token_budget', '?')}，"
            f"台词约 {d.get('dialogue_chars', '?')} 字）"
        )
    elif settings.video_frames_auto:
        print(f"单镜上限（auto）：{settings.video_frames_max} 帧（硬顶 {settings.video_frames_ceiling}）")
    else:
        print(f"单镜上限（手动）：{settings.video_frames_max} 帧（video_frames_auto=false）")
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


def cmd_budget(args: argparse.Namespace) -> None:
    """只读总内存配置 + 当前分辨率，打印单镜上限（不加载模型）。"""
    settings = Settings.load(resolve(args.config) if args.config else None)
    detail = settings.budget_detail
    if not detail:
        from pipeline.budget import format_budget_report, resolve_frame_cap

        _, detail = resolve_frame_cap(
            height=settings.video_height,
            width=settings.video_width,
            fps=settings.video_fps,
            min_frames=settings.video_frames_min,
            ceiling=settings.video_frames_ceiling,
            table=settings.token_budget_table(),
        )
    else:
        from pipeline.budget import format_budget_report

    print(format_budget_report(detail))
    if getattr(args, "out", None):
        out = resolve(args.out)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n输出目录：{out}（本命令不写文件，只供后续 produce 使用）")


def cmd_selftest(_: argparse.Namespace) -> None:
    import tempfile

    parts = chunk_novel("第一段。\n\n第二段。")
    assert parts == ["第一段。\n\n第二段。"], parts
    long = "甲" * 1000 + "\n\n" + "乙" * 1000
    chunks = chunk_novel(long, limit=1200)
    assert len(chunks) == 2, chunks
    data = extract_json("说明\n```json\n{\"shots\": []}\n```")
    assert data == {"shots": []}
    lines = extract_novel_dialogues('甲说：“你好。”乙回「在呢」。又说：“你好。”')
    assert lines == ["你好。", "在呢"], lines
    # 一句台词按标点拆到相邻镜头，拼回去仍算覆盖
    assert_dialogue_coverage("她说：“我要把它找回来。上传那天，我少拷了一段。”", [
        {"dialogue": "我要把它找回来。"},
        {"dialogue": "上传那天，我少拷了一段。"},
    ])
    assert count_spoken_chars("我要把它找回来。上传那天，我少拷了一段。") == 17
    assert dialogue_char_limit(121, 24) == 12

    # 时长预估：字数 → 秒数，写分镜时靠它自查
    from pipeline.screenplay import estimate_speech_seconds

    assert abs(estimate_speech_seconds("十二个字的台词刚好可以念完") - 13 / 2.4) < 1e-6
    assert estimate_speech_seconds("") == 0.0
    assert dialogue_char_limit(145, 24) == 14

    # 内存预算：只读总内存配置的经验 token 表 + 按分辨率现算帧数
    from pipeline.budget import (
        frames_for_budget,
        resolve_frame_cap,
        video_tokens,
    )
    from pipeline.util import snap_frames

    assert video_tokens(49, 384, 640) == 1680
    assert video_tokens(217, 384, 640) == 6720
    gib = 1024**3
    # 两个经验值必须原样命中（384x640）
    f16, d16 = resolve_frame_cap(total_bytes=16 * gib, height=640, width=384)
    f32, d32 = resolve_frame_cap(total_bytes=32 * gib, height=640, width=384)
    assert f16 == 97 and d16["token_budget"] == 3120, (f16, d16)
    assert f32 == 217 and d32["token_budget"] == 6720, (f32, d32)
    # 表边界与硬顶
    f8, _ = resolve_frame_cap(total_bytes=8 * gib, height=640, width=384)
    f64, _ = resolve_frame_cap(total_bytes=64 * gib, height=640, width=384)
    f128, _ = resolve_frame_cap(total_bytes=128 * gib, height=640, width=384)
    assert f8 == 25, f8
    assert f64 == 241 and f128 == 241, (f64, f128)
    # 表外向下取档
    f24, d24 = resolve_frame_cap(total_bytes=24 * gib, height=640, width=384)
    f48, d48 = resolve_frame_cap(total_bytes=48 * gib, height=640, width=384)
    assert f24 == 97 and d24["off_table"] and d24["tier_gib"] == 16.0, (f24, d24)
    assert f48 == 217 and d48["off_table"] and d48["tier_gib"] == 32.0, (f48, d48)
    # 分辨率现算：高分辨率变短，低分辨率撞硬顶
    f512, _ = resolve_frame_cap(total_bytes=32 * gib, height=768, width=512)
    f256, _ = resolve_frame_cap(total_bytes=32 * gib, height=448, width=256)
    assert f512 == 129 and f512 < 217, f512
    assert f256 == 241, f256
    assert snap_frames(f16) == f16 and snap_frames(f32) == f32
    assert snap_frames(f512) == f512
    # 纯函数性：同一输入多次一致
    f16b, _ = resolve_frame_cap(total_bytes=16 * gib, height=640, width=384)
    assert f16b == f16
    # frames_for_budget 宁短勿长
    raw, geom = frames_for_budget(3119, 640, 384)  # 差一点不够 13 latent
    assert raw == 89 and geom["latent_frames"] == 12, (raw, geom)

    # 长台词自动拆段：每段都变短，拼回去一字不差
    from pipeline.dialogue import halve, plan_by_measured_rate, plan_segments, spoken_chars

    long_line = "他是我父亲。他死在地球那天，我在火星。我只想再看一次他唱歌的样子。"
    pieces = plan_segments(long_line, 12)
    assert len(pieces) > 1 and spoken_chars("".join(pieces)) == spoken_chars(long_line), pieces
    assert max(spoken_chars(p) for p in pieces) <= 17, pieces  # 允许在小句边界略微超出
    measured = plan_by_measured_rate(long_line, 4.94, 4.0)
    assert len(measured) > 1 and spoken_chars("".join(measured)) == spoken_chars(long_line)
    assert plan_by_measured_rate("短句。", 1.0, 4.0) == ["短句。"]
    assert len(halve(long_line)) == 2
    assert spoken_chars("".join(halve(long_line))) == spoken_chars(long_line)

    # 假 TTS：直接按字数给时长，验证「念不完 → 自动拆镜」闭环
    import numpy as np

    from pipeline import audio as audio_mod

    class _FakeTTS:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_custom_voice(self, *, text, speaker, language, instruct, verbose, max_tokens):
            self.calls.append(text)

            class _R:
                audio = np.full(int(24000 * (spoken_chars(text) / 2.0)), 0.5, dtype=np.float32)
                sample_rate = 24000

            return [_R()]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake = _FakeTTS()
        segments = audio_mod._speak_with_fit(
            fake,
            shot_id="s001",
            line=long_line,
            profile={"voice": "Serena", "instruct": "用平静自然的语气说"},
            max_seconds=10.0,
            scratch=tmp_path,
        )
        assert len(segments) >= 1, segments
        # 关键约束：台词只念一次，绝不为了凑时长重念或改语速
        assert fake.calls == [long_line], f"台词被重复合成或改写了：{fake.calls}"
        assert spoken_chars("".join(text for text, _, _ in segments)) == spoken_chars(long_line)
        # 拆段时每段都不该超过上限（切不开则允许整句放长）
        if len(segments) > 1:
            assert all(dur <= 10.0 + 1e-3 for _, _, dur in segments), segments

    # ---- 混音回归：人声不被无故削弱、闪避只在有对白的时段生效 ----
    # 之前出过两个 bug：(1) aformat 做 mono→stereo 会乘 0.707，白掉 3 dB；
    # (2) 只要有对白就全程把配乐乘 0.35，空镜段也被压掉、整条片听不见配乐。
    import struct as _struct
    import wave as _wave
    import math as _math

    from pipeline.assemble import _mix_soundtrack
    from pipeline.config import Settings as _Settings

    def _rms_at(path: Path, start: float, dur: float) -> float:
        with _wave.open(str(path)) as wav:
            rate = wav.getframerate()
            wav.setpos(int(start * rate))
            raw = wav.readframes(int(dur * rate))
        vals = _struct.unpack(f"<{len(raw) // 2}h", raw)
        if not vals:
            return -120.0
        return 20 * _math.log10(_math.sqrt(sum(v * v for v in vals) / len(vals)) / 32768.0 + 1e-12)

    def _write_sine(path: Path, seconds: float, *, rate: int = 48000) -> None:
        with _wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(
                b"".join(
                    _struct.pack("<h", int(12000 * _math.sin(2 * _math.pi * 220 * i / rate)))
                    for i in range(int(seconds * rate))
                )
            )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        settings_mix = _Settings.load()
        settings_mix.ffmpeg = ROOT / "tools" / "ffmpeg"
        settings_mix.ffprobe = ROOT / "tools" / "ffprobe"
        if settings_mix.ffmpeg.exists():
            music = tmp_path / "bgm.wav"
            _write_sine(music, 12.0)
            # 用一条静音轨当「对白」：它能触发闪避，但不贡献任何声音，
            # 这样测到的就是纯配乐曲线，不会被真对白盖住。
            silent = tmp_path / "silent.wav"
            with _wave.open(str(silent), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(b"\x00\x00" * int(6.0 * 48000))
            out2 = tmp_path / "o2"
            out2.mkdir()
            mixed_ducked = _mix_soundtrack(
                settings_mix,
                out=out2,
                total=12.0,
                voice_cues=[(0.0, silent, 6.0)],
                music=music,
            )
            flat2 = tmp_path / "flat2.wav"
            subprocess.run(
                [
                    str(settings_mix.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(mixed_ducked), "-ac", "1", "-ar", "48000", str(flat2),
                ],
                check=True,
            )
            ducked = _rms_at(flat2, 1.0, 3.0)   # 0-6s 是对白窗，配乐该压低
            open_ = _rms_at(flat2, 7.0, 3.0)   # 6.5s 之后是空镜，配乐该恢复
            assert open_ > ducked + 5.0, (
                f"闪避没生效：对白窗 {ducked:.1f} dB，空镜 {open_:.1f} dB"
            )
            # 空镜段必须能听到配乐（早期 bug 会掉到 -50 dB 量级）
            assert open_ > -35.0, f"空镜段配乐几乎听不见：{open_:.1f} dB"

    from pipeline.hf_cache import resolve_snapshot
    from pipeline.runtime import main_venv_path, musicgen_venv_path, ltx_venv_path

    assert resolve_snapshot("") is None
    assert (ROOT / "scripts" / "apply_ltx_gemma_patch.py").exists(), "缺少 LTX 补丁脚本"
    assert (ROOT / "input" / "demo_storyboard.json").exists(), "缺少演示分镜"
    assert (ROOT / "input" / "novel.txt").exists(), "缺少演示小说"
    from pipeline.run import STAGES, install_storyboard
    from pipeline.util import frames_for_duration

    assert STAGES == ("screenplay", "images", "audio", "videos", "music", "assemble"), STAGES
    # 配乐必须排在 videos 之后：那时镜头帧数已定，配乐长度才准确
    assert STAGES.index("music") > STAGES.index("videos")
    assert frames_for_duration(2.0, 24, min_frames=25, max_frames=97) == 49
    assert frames_for_duration(0.2, 24, min_frames=25, max_frames=97) == 25
    from pipeline.progress import StageMerge, _ProgressRelay

    relay = _ProgressRelay()
    assert relay.feed("开始\n".encode()) == ["开始"]
    assert relay.feed(b"\rDenoising:  12%|x| 1/8 [00:02<00:18,  2.65s/it]\r") == [
        "Denoising:  12%|x| 1/8 [00:02<00:18,  2.65s/it]"
    ]
    merge = StageMerge()
    assert merge.update(0, 8) == (0, 8)
    assert merge.update(8, 8) == (8, 8)
    assert merge.update(0, 3) == (8, 11)
    assert merge.update(3, 3) == (11, 11)
    # 已知两段步数时，分母一开始就是 11：第一段跑完只到 73%，不会先到 100%
    planned = StageMerge((8, 3))
    assert planned.total == 11
    assert planned.update(0, 8) == (0, 11)
    assert planned.update(4, 8) == (4, 11)
    assert planned.update(8, 8) == (8, 11)
    assert planned.update(0, 3) == (8, 11)
    assert planned.update(3, 3) == (11, 11)
    # 实际段数比计划多时也不回退百分比（多出的段吃掉剩余区间）
    over = StageMerge((2,))
    assert over.update(2, 2) == (2, 2)
    assert over.update(0, 2) == (2, 2)
    assert over.update(2, 2) == (2, 2)
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

    def add_storyboard(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--storyboard",
            default=None,
            help="外部分镜 JSON；会拷进 --out/storyboard.json（演示默认可用 input/demo_storyboard.json）",
        )

    screenplay = sub.add_parser(
        "screenplay",
        help="校验分镜并打印每个对白镜头的预估时长（写分镜时自查，不跑成片）",
    )
    add_common(screenplay, novel=True)
    add_storyboard(screenplay)
    screenplay.add_argument(
        "--strict",
        action="store_true",
        help="对白超过单镜额度时失败，并给出可照抄的拆镜方案（不许删字）",
    )
    screenplay.set_defaults(func=cmd_screenplay)

    images = sub.add_parser("images", help="角色参考图和分镜静帧")
    add_common(images)
    images.add_argument("--only", default=None, help="只做这些镜头，例如 s001,s002")
    images.set_defaults(func=cmd_images)

    videos = sub.add_parser("videos", help="按静帧生成镜头视频")
    add_common(videos)
    videos.add_argument("--only", default=None)
    videos.set_defaults(func=cmd_videos)

    audio = sub.add_parser("audio", help="本地生成对白（TTS）")
    add_common(audio)
    audio.set_defaults(func=cmd_audio)

    music = sub.add_parser(
        "music",
        help="生成配乐；放在 videos 之后，按已渲染镜头长度定长（也可单独重跑）",
    )
    add_common(music)
    music.set_defaults(func=cmd_music)

    assemble = sub.add_parser("assemble", help="拼接镜头、混音并烧录字幕")
    add_common(assemble)
    assemble.set_defaults(func=cmd_assemble)

    produce = sub.add_parser(
        "produce",
        help="一键成片：分镜 → 校验 → 图 → 对白 → A2V 视频 → 配乐 → final.mp4（不再需要 Agent）",
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

    budget = sub.add_parser(
        "budget",
        help="按本机总内存查 token 预算表，现算单镜最长帧数 / 台词额度（不加载模型）",
    )
    budget.add_argument("--config", default=None, help="配置文件，默认项目里的 config.json")
    budget.add_argument("--out", default=None, help="片子输出目录（可选，仅打印路径）")
    budget.set_defaults(func=cmd_budget)

    doctor = sub.add_parser("doctor", help="检查本机环境")
    doctor.set_defaults(func=cmd_doctor)
    test = sub.add_parser("selftest", help="检查不依赖模型的逻辑")
    test.set_defaults(func=cmd_selftest)
    return parser


def main() -> None:
    from pipeline.progress import install_live_bars

    install_live_bars()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
