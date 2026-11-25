from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.mirrors import apply_mirrors

apply_mirrors()

from pipeline.assemble import cmd_assemble
from pipeline.audio import cmd_audio, cmd_music
from pipeline.config import Settings
from pipeline.images import cmd_images
from pipeline.projects import cmd_projects
from pipeline.run import STAGES, cmd_produce, cmd_run
from pipeline.screenplay import cmd_screenplay
from pipeline.screenplay import (
    assert_dialogue_coverage,
    count_spoken_chars,
    dialogue_char_limit,
    extract_novel_dialogues,
)
from pipeline.storyboard_gen import DEFAULT_LLM_MODEL, cmd_storyboard
from pipeline.util import ROOT, chunk_novel, cite_path, extract_json, resolve
from pipeline.video import cmd_videos
from pipeline.web import cmd_web


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
    print(f"分镜 LLM：{settings.llm_model}（小说→storyboard.json；约 3–4 GB，适合 16 GB Mac）")
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
    print("提示词编码器：LTX 内置 Gemma 3 12B 4-bit（视频阶段用，不是分镜聊天）")
    print(f"对白 TTS：{settings.tts_model}（先于视频，供 A2V 条件）")
    print(f"配乐：{settings.music_model}")
    print(f"HF 镜像：{os.environ.get('HF_ENDPOINT')}")
    print(f"PyPI 镜像：{os.environ.get('UV_INDEX_URL')}")
    print(f"禁用 XET：{os.environ.get('HF_HUB_DISABLE_XET')}")
    # ---- 平台与推理后端 ----
    from pipeline import platform as plat
    from pipeline.backends import describe as describe_backend

    pinfo = plat.describe()
    print(f"平台：{pinfo['platform']}  Python {pinfo['python']}")
    print(f"总内存：{pinfo['total_memory_gb']} GB")
    if pinfo["nvidia_vram_gb"]:
        print(f"NVIDIA 显存：{pinfo['nvidia_vram_gb']} GB（nvidia-smi 可用）")
    elif pinfo["has_nvidia_smi"]:
        print("NVIDIA：有 nvidia-smi 但读不到显存")
    else:
        print("NVIDIA：未检测到 nvidia-smi")
    b = describe_backend(getattr(settings, "backend", None))
    print(f"推理后端：{b['backend']}（{b['reason']}）")
    if b["backend"] == "cuda":
        print(f"  torch CUDA 就绪：{b.get('torch_cuda_ready', False)}")
    print(f"缓存目录：{pinfo['cache_dir']}")
    print(f"字幕字体：{pinfo['subtitle_font']}（{pinfo['font_dir']}）")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    print(f"uv：{uv}（{'存在' if Path(uv).exists() or shutil.which('uv') else '缺失'}）")
    print(f"ffmpeg：{settings.ffmpeg}（{'存在' if settings.ffmpeg.exists() else '缺失'}）")
    if sys.platform == "darwin":
        from pipeline.menubar import HELPER, menubar_enabled

        print(
            f"菜单栏助手：{HELPER}"
            f"（{'可用' if menubar_enabled() else '缺二进制，跑 scripts/macos/build-menubar-helper.sh'}）"
        )
    print(f"LTX 代码：{settings.ltx_dir}（{'存在' if settings.ltx_dir.exists() else '缺失'}）")
    from pipeline.hf_cache import resolve_snapshot, snapshot_ready

    llm_snap = resolve_snapshot(settings.llm_model or DEFAULT_LLM_MODEL)
    llm_ok, llm_detail = snapshot_ready(llm_snap, weight_globs=("model*.safetensors",))
    if llm_ok:
        print(f"分镜 LLM 权重：有 {llm_snap}")
    elif llm_snap is not None:
        print(f"分镜 LLM 权重：缓存残缺（{llm_detail}）{llm_snap}")
    else:
        print("分镜 LLM 权重：未下载（setup.sh 会拉）")
    try:
        import mlx_lm  # noqa: F401

        print("mlx-lm：已安装（主 venv）")
    except ImportError:
        print("mlx-lm：缺失（请 uv sync / 重跑 setup.sh）")
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
    if getattr(args, "project", None):
        from pipeline.projects import project_dir

        print(f"\n项目文件夹：{project_dir(args.project)}（本命令不写文件，只供后续 produce 使用）")
    elif getattr(args, "out", None):
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

    # ---- assemble：合规流零重编码 / 探测失败必须硬报错 ----
    from pipeline.assemble import _duration, _normalize, _streams_info

    if (ROOT / "tools" / "ffprobe").exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 1) 合规流（分辨率/帧率/像素格式全对、无音轨）→ 走流复制分支
            good = tmp_path / "good.mp4"
            subprocess.run(
                [
                    str(ROOT / "tools" / "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=gray:s=384x640:r=24:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(good),
                ],
                check=True,
            )
            settings_asm = _Settings.load()
            settings_asm.ffmpeg = ROOT / "tools" / "ffmpeg"
            settings_asm.ffprobe = ROOT / "tools" / "ffprobe"
            info = _streams_info(settings_asm.ffprobe, good)
            assert info["width"] == 384 and info["height"] == 640, info
            assert not info["has_audio"] and abs(info["fps"] - 24.0) < 0.1, info
            copied = tmp_path / "copied.mp4"
            _normalize(settings_asm, good, copied)
            assert copied.exists()
            # 流复制不重编码，体积应与源接近（重编码会差数倍）
            src_size = good.stat().st_size
            dst_size = copied.stat().st_size
            assert 0.5 < dst_size / max(1, src_size) < 2.0, (src_size, dst_size)
            # 2) 带音轨的流 → 必须重编码剥轨
            withaudio = tmp_path / "withaudio.mp4"
            subprocess.run(
                [
                    str(ROOT / "tools" / "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=gray:s=384x640:r=24:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(withaudio),
                ],
                check=True,
            )
            stripped = tmp_path / "stripped.mp4"
            _normalize(settings_asm, withaudio, stripped)
            assert not _streams_info(settings_asm.ffprobe, stripped)["has_audio"]
            # 3) 探测失败必须抛 SystemExit，不许静默回退假时长
            broken = tmp_path / "broken.mp4"
            broken.write_bytes(b"not a video")
            try:
                _duration(settings_asm.ffprobe, broken)
                raise AssertionError("探测失败应抛 SystemExit，而不是回退默认时长")
            except SystemExit:
                pass

    # ---- MusicGen 进度条：分块生成分母不再越换越大 ----
    import importlib.util as _ilu

    _mg_path = ROOT / "vendor" / "musicgen-mlx" / "audiocraft_mlx" / "models" / "musicgen.py"
    if _mg_path.exists():
        spec = _ilu.spec_from_file_location("_mg_musicgen_selftest", _mg_path)
        mg_mod = _ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mg_mod)
        except Exception:
            mg_mod = None
        if mg_mod is not None and hasattr(mg_mod, "_StepBar"):
            bar = mg_mod._StepBar("MusicGen")
            updates: list[tuple[int, int]] = []

            class _FakeTqdm:
                def __init__(self, *, total, desc, unit):
                    updates.append(("new", total))

                def refresh(self):
                    pass

                def close(self):
                    pass

            original_tqdm = mg_mod.tqdm
            mg_mod.tqdm = _FakeTqdm
            try:
                bar.update(500, 1503)
                assert updates == [("new", 1503)], updates
                bar.update(1000, 1503)
                # 分母试图变大（分块生成的下一块更长）：必须被忽略，不换条
                bar.update(0, 2106)
                assert updates == [("new", 1503)], updates
                # 正常推进仍在同一条上
                bar.update(1200, 1503)
                assert updates == [("new", 1503)], updates
            finally:
                mg_mod.tqdm = original_tqdm

    from pipeline.hf_cache import iter_weight_files, resolve_snapshot, snapshot_ready
    from pipeline.runtime import main_venv_path, musicgen_venv_path, ltx_venv_path

    assert resolve_snapshot("") is None
    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp)
        (bare / "config.json").write_text("{}", encoding="utf-8")
        (bare / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        assert iter_weight_files(bare, "model*.safetensors") == []
        ok, detail = snapshot_ready(bare, weight_globs=("model*.safetensors",))
        assert not ok and "model*.safetensors" in detail, detail
        (bare / "model.safetensors").write_bytes(b"x")
        dangling = bare / "model-00001-of-00002.safetensors"
        dangling.symlink_to(bare / "missing-blob")
        names = {path.name for path in iter_weight_files(bare, "model*.safetensors")}
        assert names == {"model.safetensors"}, names
        ok, _ = snapshot_ready(bare, weight_globs=("model*.safetensors",))
        assert ok
    assert (ROOT / "scripts" / "apply_ltx_gemma_patch.py").exists(), "缺少 LTX 补丁脚本"
    from pipeline.projects import project_dir, projects_dir

    demo_dir = project_dir("demo")
    assert (demo_dir / "storyboard.json").exists(), f"缺少演示分镜：{demo_dir}"
    assert (demo_dir / "novel.txt").exists(), f"缺少演示小说：{demo_dir}"
    from pipeline.run import STAGES, install_storyboard
    from pipeline.util import frames_for_duration

    assert STAGES == (
        "storyboard",
        "screenplay",
        "images",
        "audio",
        "videos",
        "music",
        "assemble",
    ), STAGES
    assert STAGES.index("storyboard") < STAGES.index("screenplay")
    # 串跑仍 music 在 videos 后；数据上配乐只依赖 audio 写入的 video_frames，不依赖 clips
    assert STAGES.index("music") > STAGES.index("audio")
    assert STAGES.index("music") > STAGES.index("videos")
    from pipeline.storyboard_gen import (
        _ensure_missing_dialogues,
        _scrub_fake_dialogues,
        _split_long_dialogues,
    )

    scrubbed = _scrub_fake_dialogues(
        {
            "shots": [
                {"id": "s001", "dialogue": "雨下得很大。", "speaker": "林晚"},
                {"id": "s002", "dialogue": "你不是去了南方吗？", "speaker": "林晚"},
            ]
        },
        '林晚说：“你不是去了南方吗？”',
    )
    assert scrubbed["shots"][0]["dialogue"] == ""
    assert scrubbed["shots"][1]["dialogue"] == "你不是去了南方吗？"

    split = _split_long_dialogues(
        {
            "shots": [
                {
                    "id": "s001",
                    "image_prompt": "近景",
                    "motion_prompt": "Locked-off",
                    "dialogue": "他是我父亲。他死在地球那天，我在火星。我只想再看一次他唱歌的样子。",
                }
            ]
        },
        12,
    )
    assert len(split["shots"]) >= 2, split
    assert "".join(s["dialogue"] for s in split["shots"]).replace(" ", "") == (
        "他是我父亲。他死在地球那天，我在火星。我只想再看一次他唱歌的样子。"
    )
    patched = _ensure_missing_dialogues(
        {"characters": [{"id": "a", "name": "甲"}], "shots": [{"id": "s001", "image_prompt": "x", "dialogue": ""}]},
        '甲说：“你好。”',
    )
    assert any("你好。" in str(s.get("dialogue")) for s in patched["shots"]), patched
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

    # 进度引擎：权重和、步内上升、过期 leave 不压下一镜
    from pipeline.progress_model import STAGE_WEIGHTS, ProgressEngine, build_plan

    demo_steps = build_plan(demo_dir)
    assert abs(sum(s["weight"] for s in demo_steps) - sum(STAGE_WEIGHTS.values())) < 1e-3
    eng = ProgressEngine(str(demo_dir), task_id="selftest")
    eng.feed_line("分镜静帧 s001")
    eng.feed_line("静帧 s001:  50%|████| 2/4 [00:20<00:20, 10s/it]")
    snap1 = eng.snapshot()
    assert snap1["overall_pct"] < 20, snap1["overall_pct"]
    s001 = next(s for s in snap1["steps"] if s["id"] == "images:s001")
    assert s001["pct"] == 50.0 and s001["status"] == "running"
    eng.feed_line("静帧 s001: 100%|████| 4/4 [00:40<00:00, 10s/it]")
    eng.feed_line("分镜静帧 s002")
    eng.feed_line("静帧 s002:  50%|████| 2/4 [00:20<00:20, 10s/it]")
    o_mid = eng.snapshot()["overall_pct"]
    eng.feed_line("静帧 s001: 100%|████| 4/4 [00:40<00:00, 10s/it]")  # 过期 leave
    snap2 = eng.snapshot()
    s002 = next(s for s in snap2["steps"] if s["id"] == "images:s002")
    assert s002["status"] == "running" and s002["pct"] >= 50.0
    assert snap2["overall_pct"] >= o_mid

    # 混音成片：整理切走后须激活 mux；成功收尾不得留下 pending 步却 overall 100%
    eng.feed_line("整理 s001")
    eng.feed_line("混音成片")
    mux = next(s for s in eng.snapshot()["steps"] if s["id"] == "assemble:mux")
    assert mux["status"] == "running", mux
    eng.feed_line(f"成片：{demo_dir}/final.mp4，约 1.0 秒")
    mux = next(s for s in eng.snapshot()["steps"] if s["id"] == "assemble:mux")
    assert mux["status"] == "done" and mux["pct"] == 100.0, mux
    eng2 = ProgressEngine(str(demo_dir), task_id="selftest-finalize")
    eng2.feed_line("整理 s001")
    eng2.finalize(success=True)
    left = [s for s in eng2.snapshot()["steps"] if s["status"] == "pending"]
    assert not left, left
    assert eng2.snapshot()["overall_pct"] == 100.0

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "storyboard.json"
        install_storyboard(tmp, str(demo_dir / "storyboard.json"))
        assert dest.exists(), "produce 应能把分镜拷进输出目录"

    # 项目文件夹约定：--project 补默认路径、命名守卫、清单与 novel 猜测
    from pipeline.projects import (
        apply_project_defaults,
        artifacts_dir,
        artifacts_sub,
        check_name,
        has_work_dir,
        list_projects,
        novel_candidates,
        project_path_str,
    )

    abs_demo = project_path_str("demo")
    ns = argparse.Namespace(project="demo", out=None, novel=None, storyboard=None)
    apply_project_defaults(ns)
    assert ns.out == abs_demo, ns.out
    assert ns.novel == str((demo_dir / "novel.txt").resolve()), ns.novel
    assert ns.storyboard == str((demo_dir / "storyboard.json").resolve()), ns.storyboard
    ns2 = argparse.Namespace(project="demo", out="output/demo", novel=None, storyboard=None)
    apply_project_defaults(ns2)
    assert ns2.out == abs_demo, ns2.out
    ns3 = argparse.Namespace(project="demo", out="somewhere/else", novel=None, storyboard=None)
    apply_project_defaults(ns3)
    assert ns3.out == "somewhere/else", ns3.out
    # 过程产物收进 <项目>/work/；顶层只留小说、分镜、成片
    assert artifacts_dir("projects/x").name == "work", artifacts_dir("projects/x")
    assert artifacts_sub("projects/x", "audio", "voice") == artifacts_dir("projects/x") / "audio" / "voice"
    assert artifacts_dir(Path("/abs/out")) == Path("/abs/out/work")
    assert (demo_dir / "novel.txt").exists() and (demo_dir / "storyboard.json").exists()
    assert (demo_dir / "final.mp4").exists()
    assert has_work_dir(demo_dir), "demo 项目应有 work/ 过程产物目录"
    assert (demo_dir / "work" / "frames").is_dir()
    for stray in ("frames", "clips", "audio", "logs", "normalized"):
        assert not (demo_dir / stray).exists(), f"demo 顶层不应再有过程产物：{stray}/"
    try:
        check_name("../escape")
        raise AssertionError("项目名不应接受路径分隔符")
    except SystemExit:
        pass
    cands = novel_candidates(demo_dir / "storyboard.json")
    assert cands and cands[0] == demo_dir / "novel.txt", cands
    names = {item["name"] for item in list_projects()}
    assert "demo" in names and "scifi" in names, names
    assert projects_dir().name == "AI短视频项目" or "NOVEL_VIDEO_PROJECTS" in __import__("os").environ

    # 网页暂存落在 ~/AI短视频项目，不在 App runtime / 仓库根下面。
    # relative_to(ROOT) 会抛 “is not in the subpath”，入队对话框就显示「无法添加」。
    inside = ROOT / "webui" / "index.html"
    assert cite_path(inside) == "webui/index.html", cite_path(inside)
    assert resolve(cite_path(inside)) == inside.resolve()
    from pipeline.web import materialize_web_draft

    with tempfile.TemporaryDirectory() as outside:
        proj = Path(outside) / "反向救赎的时空书店"
        proj.mkdir()
        sb_rel, novel_rel = materialize_web_draft(
            str(proj),
            storyboard_data={"shots": [{"id": "s001", "dialogue": "你好"}]},
            novel_text="她说：「你好」。",
        )
        assert novel_rel and Path(novel_rel).is_absolute(), novel_rel
        assert resolve(novel_rel) == (proj / "work" / "webui" / "run_novel.txt").resolve()
        assert "你好" in resolve(novel_rel).read_text(encoding="utf-8")
        assert sb_rel and resolve(sb_rel) == (proj / "storyboard.json").resolve()
        assert (proj / "storyboard.json").is_file()
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
    parser = argparse.ArgumentParser(
        description="把小说本地做成竖屏短剧（无子命令时默认启动网页控制台）",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="网页监听地址（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8790,
        help="网页监听端口（默认 8790）",
    )
    # 无参数 / 只有 --host/--port → 起网页；管线仍用 produce / run / 各阶段子命令
    parser.set_defaults(func=cmd_web)
    sub = parser.add_subparsers(dest="cmd", required=False)

    def add_common(p: argparse.ArgumentParser, novel: bool = False) -> None:
        p.add_argument("--config", default=None, help="配置文件，默认项目里的 config.json")
        p.add_argument(
            "--project",
            default=None,
            help="项目名：等价于 --out ~/AI短视频项目/<名>，novel/storyboard 也默认取该项目文件夹",
        )
        p.add_argument("--out", default=None, help="这一部片子的输出目录（默认 ~/AI短视频项目/<片名>）")
        p.add_argument("--force", action="store_true", help="重做已有的分镜、静帧和镜头")
        if novel:
            p.add_argument("--novel", default=None, help="小说路径（默认 <项目文件夹>/novel.txt）")
            p.add_argument("--max-shots", type=int, default=None)

    def add_storyboard(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--storyboard",
            default=None,
            help="外部分镜 JSON；会拷进 --out/storyboard.json（缺省取 <项目文件夹>/storyboard.json）",
        )

    storyboard = sub.add_parser(
        "storyboard",
        help="本机 LLM 从小说生成 storyboard.json（Qwen3-4B 4bit，适合 16 GB Mac）",
    )
    add_common(storyboard, novel=True)
    storyboard.set_defaults(func=cmd_storyboard)

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
    images.add_argument(
        "--characters",
        default=None,
        help="只（重）做这些角色肖像，例如 linwan,zhouyu；可与 --only 组合",
    )
    images.set_defaults(func=cmd_images)

    videos = sub.add_parser("videos", help="按静帧生成镜头视频")
    add_common(videos)
    videos.add_argument("--only", default=None)
    videos.set_defaults(func=cmd_videos)

    audio = sub.add_parser("audio", help="本地生成对白（TTS）")
    add_common(audio)
    audio.add_argument(
        "--only",
        default=None,
        help="只重新合成这些镜头的对白，例如 s004,s007；其余镜头复用已有配音",
    )
    audio.set_defaults(func=cmd_audio)

    music = sub.add_parser(
        "music",
        help="生成配乐；按分镜 video_frames 计划片长定长（对白后即可，不依赖 clips）",
    )
    add_common(music)
    music.set_defaults(func=cmd_music)

    assemble = sub.add_parser("assemble", help="拼接镜头、混音并烧录字幕")
    add_common(assemble)
    assemble.set_defaults(func=cmd_assemble)

    produce = sub.add_parser(
        "produce",
        help="一键成片：小说→分镜 → 校验 → 图 → 对白 → A2V → 配乐 → final.mp4",
    )
    add_common(produce, novel=True)
    add_storyboard(produce)
    produce.add_argument("--only", default=None, help="只重做列出的镜头，用于 images、audio 和 videos")
    produce.set_defaults(func=cmd_produce, start="storyboard")

    run = sub.add_parser("run", help="按阶段串跑（默认从 storyboard 起，等同 produce）")
    add_common(run, novel=True)
    add_storyboard(run)
    run.add_argument("--from", dest="start", default="storyboard", choices=STAGES)
    run.add_argument(
        "--until",
        default=None,
        choices=STAGES,
        help="跑到哪个阶段为止（含）；缺省一路跑到 assemble。例：--from images --until audio",
    )
    run.add_argument("--only", default=None, help="只重做列出的镜头，用于 images、audio 和 videos")
    run.set_defaults(func=cmd_run)

    budget = sub.add_parser(
        "budget",
        help="按本机总内存查 token 预算表，现算单镜最长帧数 / 台词额度（不加载模型）",
    )
    budget.add_argument("--config", default=None, help="配置文件，默认项目里的 config.json")
    budget.add_argument("--project", default=None, help="项目名：打印 ~/AI短视频项目/<片名> 供后续 produce 使用")
    budget.add_argument("--out", default=None, help="片子输出目录（可选，仅打印路径）")
    budget.set_defaults(func=cmd_budget)

    projects = sub.add_parser(
        "projects",
        help="列出 ~/AI短视频项目/ 下的全部项目（--new 新建一个项目文件夹）",
    )
    projects.add_argument("--new", default=None, help="新建项目文件夹 ~/AI短视频项目/<片名>/")
    projects.add_argument("--title", default=None, help="项目标题（写进 project.json）")
    projects.set_defaults(func=cmd_projects)

    doctor = sub.add_parser("doctor", help="检查本机环境")
    doctor.set_defaults(func=cmd_doctor)
    web = sub.add_parser(
        "web",
        help="启动本地网页控制台（默认入口，等同于不传子命令）",
    )
    # SUPPRESS：勿盖掉父级已解析的 --host/--port（如「--port 9000 web」）
    web.add_argument("--host", default=argparse.SUPPRESS, help="网页监听地址（默认 127.0.0.1）")
    web.add_argument("--port", type=int, default=argparse.SUPPRESS, help="网页监听端口（默认 8790）")
    web.set_defaults(func=cmd_web)
    test = sub.add_parser("selftest", help="检查不依赖模型的逻辑")
    test.set_defaults(func=cmd_selftest)
    return parser


def main() -> None:
    from pipeline.progress import install_live_bars

    install_live_bars()
    parser = build_parser()
    args = parser.parse_args()
    from pipeline.projects import apply_project_defaults

    apply_project_defaults(args)
    args.func(args)


if __name__ == "__main__":
    main()
