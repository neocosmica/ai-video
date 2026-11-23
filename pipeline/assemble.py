from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline.config import Settings
from pipeline.util import ass_escape, ass_time, ffmpeg_concat_line, load_json, resolve

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT_NAME = "Songti SC"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise SystemExit(f"命令失败：{' '.join(cmd[:6])}\n{tail}")


def _duration(ffprobe: Path, clip: Path) -> float:
    proc = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        return max(0.1, float(proc.stdout.strip()))
    except ValueError:
        return 2.0


def _has_audio(ffprobe: Path, clip: Path) -> bool:
    proc = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def _wav_seconds(settings: Settings, path: Path) -> float:
    proc = subprocess.run(
        [
            str(settings.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError:
        return 0.0


def _normalize(settings: Settings, src: Path, dest: Path) -> None:
    scale = (
        f"scale={settings.video_width}:{settings.video_height}:force_original_aspect_ratio=decrease,"
        f"pad={settings.video_width}:{settings.video_height}:(ow-iw)/2:(oh-ih)/2,fps={settings.video_fps}"
    )
    if _has_audio(settings.ffprobe, src):
        _run(
            [
                str(settings.ffmpeg),
                "-y",
                "-i",
                str(src),
                "-vf",
                scale,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ]
        )
        return
    _run(
        [
            str(settings.ffmpeg),
            "-y",
            "-i",
            str(src),
            "-vf",
            scale,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
    )


def _font_size(width: int) -> int:
    return max(18, min(36, width // 16))


def _wrap_cue(text: str, width: int, fontsize: int) -> str:
    budget = max(8, (width - 48) // fontsize)
    if len(text) <= budget:
        return text
    lines = []
    rest = text
    while rest:
        if len(rest) <= budget:
            lines.append(rest)
            break
        cut = budget
        for mark in "，。！？、；：":
            found = rest.rfind(mark, 0, budget + 1)
            if found >= budget // 2:
                cut = found + 1
                break
        lines.append(rest[:cut])
        rest = rest[cut:]
    return "\n".join(lines)


def _write_ass(path: Path, cues: list[tuple[float, float, str]], settings: Settings) -> None:
    fontsize = _font_size(settings.video_width)
    events = []
    for start, end, text in cues:
        if not text.strip() or end <= start:
            continue
        wrapped = _wrap_cue(text, settings.video_width, fontsize)
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{ass_escape(wrapped)}")
    body = "\n".join(events)
    path.write_text(
        f"""[Script Info]
ScriptType: v4.00+
PlayResX: {settings.video_width}
PlayResY: {settings.video_height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,24,24,56,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{body}
""",
        encoding="utf-8",
    )


def _mix_soundtrack(
    settings: Settings,
    *,
    out: Path,
    total: float,
    voice_cues: list[tuple[float, Path, float]],
    music: Path | None,
) -> Path:
    """把对白按时间轴铺开，配乐循环；对白出现时配乐下沉，空镜让配乐露出，输出 stereo aac。

    voice_cues: (start_s, wav, max_duration_s) — max_duration 对齐镜头长度，避免串镜。

    配乐不做全程固定衰减（否则空镜段也被一起压掉、整条片几乎听不见配乐），
    而是按已知的对白时间轴分段设音量：说话时乘 music_duck_gain，空镜段恢复满电平。
    """
    mixed = out / "soundtrack.m4a"
    inputs: list[str] = [
        "-f",
        "lavfi",
        "-t",
        f"{total:.3f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
    ]
    filters: list[str] = []
    voice_labels: list[str] = []
    for index, (start, wav, max_dur) in enumerate(voice_cues):
        inputs += ["-i", str(wav)]
        delay_ms = max(0, int(round(start * 1000)))
        trim = max(0.05, float(max_dur))
        filters.append(
            f"[{index + 1}:a]aresample=48000,"
            f"atrim=0:{trim:.3f},asetpts=PTS-STARTPTS,"
            f"volume={settings.voice_volume},"
            # 用 pan 复制声道：aformat 做 mono→stereo 会乘 0.707，白白削掉 3 dB
            f"pan=stereo|c0=c0|c1=c0,"
            f"adelay={delay_ms}|{delay_ms}[v{index}]"
        )
        voice_labels.append(f"[v{index}]")

    parts = ["[0:a]"]
    if music and music.exists():
        inputs += ["-stream_loop", "-1", "-t", f"{total:.3f}", "-i", str(music)]
        music_idx = 1 + len(voice_cues)
        # 基础电平：先 loudnorm 统一到参考响度（MusicGen 原始输出常只有 -28 LUFS），再乘 music_volume
        chain = [
            f"[{music_idx}:a]aresample=48000",
            # mono 源用 pan 复制声道：aformat 做 mono→stereo 会乘 0.707，白白削掉 3 dB
            "pan=stereo|c0=c0|c1=c0",
            f"loudnorm=I={settings.music_target_lufs}:TP=-2:LRA=20",
            f"volume={settings.music_volume}",
        ]
        # 对白闪避：按已知对白时间轴逐帧求值，说话时配乐下沉、空镜段恢复满电平。
        # 注意必须用 eval=frame，默认 eval=once 只算第一帧，等于全程一个音量。
        if voice_cues:
            windows = []
            for start, _wav, max_dur in voice_cues:
                left = max(0.0, float(start) - settings.music_duck_pad)
                right = min(float(total), float(start) + float(max_dur) + settings.music_duck_pad)
                if right > left:
                    windows.append((left, right))
            if windows:
                expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in windows)
                chain.append(
                    f"volume='if({expr},{settings.music_duck_gain},1.0)':eval=frame"
                )
        chain.append(f"afade=t=in:st=0:d=0.8")
        chain.append(f"afade=t=out:st={max(0.0, total - 1.2):.3f}:d=1.2")
        filters.append(",".join(chain) + "[m]")
        parts.append("[m]")
    parts.extend(voice_labels)
    # normalize=0：混音只做相加，不按路数除；否则人声会随镜头数变轻。
    # 末尾加限幅器兜底，防止人声叠配乐在某些字上冲过 0 dBFS 削波。
    filters.append(
        f"{''.join(parts)}amix=inputs={len(parts)}:normalize=0:dropout_transition=0,"
        f"alimiter=limit={settings.mix_limit}:level=0[aout]"
    )
    cmd = [
        str(settings.ffmpeg),
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[aout]",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(mixed),
    ]
    _run(cmd)
    return mixed


def cmd_assemble(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    if not settings.ffmpeg.exists() or not settings.ffprobe.exists():
        raise SystemExit("找不到 ffmpeg / ffprobe。请先运行 scripts/setup.sh")
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    clips = []
    for shot in story["shots"]:
        path = out / "clips" / f"{shot['id']}.mp4"
        if path.exists():
            clips.append((shot, path))
    if not clips:
        raise SystemExit(f"没有可合成的镜头。请先看 {out / 'logs'}")

    normal_dir = out / "normalized"
    normal_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    cursor = 0.0
    cues = []
    voice_cues: list[tuple[float, Path, float]] = []
    for shot, path in clips:
        dest = normal_dir / f"{shot['id']}.mp4"
        print(f"整理 {shot['id']}")
        _normalize(settings, path, dest)
        duration = _duration(settings.ffprobe, dest)
        line = shot.get("dialogue") or ""
        speaker = shot.get("speaker") or ""
        if line:
            text = f"{speaker}：{line}" if speaker else line
            cues.append((cursor, cursor + duration, text))
            voice = out / "audio" / "voice" / f"{shot['id']}.wav"
            if voice.exists():
                # 成片仍用 TTS 原轨（A2V mp4 里的音会被 normalize 剥掉）；裁到本镜时长
                voice_cues.append((cursor, voice, duration))
            else:
                print(f"警告：{shot['id']} 有对白但没有旁白文件，请先跑 audio 阶段")
        cursor += duration
        normalized.append(dest)

    listing = out / "concat.txt"
    listing.write_text("\n".join(ffmpeg_concat_line(p) for p in normalized) + "\n", encoding="utf-8")
    joined = out / "joined.mp4"
    _run([str(settings.ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])

    music = out / "audio" / "bgm.wav"
    if music.exists():
        have = _wav_seconds(settings, music)
        if have + 0.05 < cursor:
            print(
                f"提示：配乐 {have:.1f}s 短于成片 {cursor:.1f}s，会循环补齐（中间可能听到接缝）。"
                "建议跑一次 music 阶段按成片长度重新生成。"
            )
    soundtrack = _mix_soundtrack(
        settings,
        out=out,
        total=cursor,
        voice_cues=voice_cues,
        music=music if music.exists() else None,
    )

    subs = out / "subs.ass"
    _write_ass(subs, cues, settings)
    final = out / "final.mp4"
    fonts = f"ass=subs.ass:fontsdir={FONT_DIR}"
    _run(
        [
            str(settings.ffmpeg),
            "-y",
            "-i",
            "joined.mp4",
            "-i",
            "soundtrack.m4a",
            "-vf",
            fonts,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "final.mp4",
        ],
        cwd=out,
    )
    if not final.exists():
        raise SystemExit("合成没有产出 final.mp4")
    print(f"成片：{final}，约 {cursor:.1f} 秒，{len(normalized)} 个镜头，对白 {len(voice_cues)} 条，配乐 {'有' if music.exists() else '无'}。")
