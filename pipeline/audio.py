from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.backends import open_speech, render_music
from pipeline.config import Settings
from pipeline.dialogue import plan_segments, spoken_chars, split_text_by_ratios
from pipeline.projects import artifacts_sub
from pipeline.runtime import ensure_runtime_layout
from pipeline.util import dump_json, frames_for_duration, load_json, resolve, snap_frames, stable_seed
from pipeline.voices import (
    default_music_prompt,
    emotion_instruct,
    infer_gender,
    pick_voice,
)


def _character_map(story: dict) -> dict[str, dict]:
    out = {}
    for char in story.get("characters") or []:
        out[char["id"]] = char
        out[char["name"]] = char
    return out


def _speaker_profile(shot: dict, characters: dict[str, dict]) -> dict:
    speaker = str(shot.get("speaker") or "").strip()
    char = characters.get(speaker) or {}
    gender = infer_gender(char.get("appearance", ""), char.get("gender") or shot.get("gender"))
    voice = pick_voice(gender, char.get("voice") or shot.get("voice"), char.get("appearance", ""))
    emotion = shot.get("emotion") or char.get("emotion") or "calm"
    return {
        "speaker": speaker,
        "gender": gender,
        "voice": voice,
        "emotion": emotion,
        "instruct": emotion_instruct(str(emotion)),
    }


def silence_cuts(arr: np.ndarray, sample_rate: int, *, min_gap: float = 0.12) -> list[tuple[int, int]]:
    """返回音频里可切的静音区间（采样下标），用于把一句连续语音切成几段。"""
    sr = max(1, int(sample_rate))
    hop = max(1, int(0.02 * sr))
    count = arr.shape[0] // hop
    if count < 3:
        return []
    frames = arr[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    speech = float(np.percentile(rms, 90))
    threshold = max(0.008, 0.18 * speech)
    quiet = rms <= threshold
    cuts: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet:
            if start is None:
                start = index
            continue
        if start is not None:
            if (index - start) * hop >= min_gap * sr:
                cuts.append((start * hop, index * hop))
            start = None
    if start is not None and (count - start) * hop >= min_gap * sr:
        cuts.append((start * hop, count * hop))
    return cuts


def split_wav_at_cuts(
    src: Path,
    dests: list[Path],
    cuts: list[tuple[int, int]],
    target_seconds: float,
    *,
    sample_rate: int,
    head_pad: int = 0,
) -> list[float]:
    """按候选静音区间把一条 wav 切成 len(dests) 段，返回各段时长。

    段数由 ``target_seconds`` 决定，切点优先选最接近均分位置的静音区间，
    因此语调连贯（本来就是一次合成），且一个字都不会丢。
    """
    data, sr = sf.read(str(src), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    total = arr.shape[0]
    if len(dests) <= 1 or not cuts:
        dests[0].parent.mkdir(parents=True, exist_ok=True)
        _save_mono_wav(dests[0], arr[head_pad:], int(sr))
        return [float(total - head_pad) / max(1, int(sr))]

    # 每段的理想边界
    want = [total * (index + 1) / len(dests) for index in range(len(dests) - 1)]
    chosen: list[int] = []
    pool = list(cuts)
    for boundary in want:
        if not pool:
            break
        best = min(pool, key=lambda cut: abs((cut[0] + cut[1]) / 2 - boundary))
        chosen.append((best[0] + best[1]) // 2)
        pool = [cut for cut in pool if cut[0] > best[1]]
    chosen = sorted(set(chosen))
    if len(chosen) != len(dests) - 1:
        return []

    edges = [head_pad, *chosen, total]
    durations: list[float] = []
    for index, dest in enumerate(dests):
        chunk = arr[edges[index] : edges[index + 1]]
        dest.parent.mkdir(parents=True, exist_ok=True)
        _save_mono_wav(dest, chunk, int(sr))
        durations.append(float(chunk.shape[0]) / max(1, int(sr)))
    return durations


def _save_mono_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    sf.write(str(path), arr, sample_rate)


def trim_silence_wav(
    path: Path,
    *,
    pad_head: float = 0.08,
    pad_tail: float = 0.16,
    floor: float = 0.010,
    run_frames: int = 3,
    report_pad: bool = False,
) -> float | tuple[float, int]:
    """裁掉 TTS 首尾静音，只留一点呼吸余量；返回裁后时长。

    Qwen3-TTS 常在句首尾留 1 秒以上空白（有时还带一点底噪/口水音），不裁会让
    A2V 镜头白拖、口型等半天。用 20ms 窗 RMS 相对语音电平判定，并要求连续
    多帧才算说话，避免把零星噪声当语音；只从两端向内收，句中停顿不动。

    ``report_pad=True`` 时返回 (裁后时长, 被裁掉的开头采样数)，供按静音切段用。
    """
    data, sample_rate = sf.read(str(path), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    sr = max(1, int(sample_rate))

    def _result(seconds: float, pad: int = 0):
        return (seconds, pad) if report_pad else seconds

    if arr.size == 0:
        return _result(0.0)

    hop = max(1, int(0.02 * sr))
    count = arr.shape[0] // hop
    if count < run_frames + 1:
        return _result(float(arr.shape[0]) / sr)
    frames = arr[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    # 语音电平取 90 分位；阈值取它的一小截，绝对下限防止整段静音时乱裁
    speech = float(np.percentile(rms, 90))
    threshold = max(floor, 0.25 * speech)
    loud = rms > threshold

    first: int | None = None
    last: int | None = None
    run = 0
    for index, flag in enumerate(loud):
        if flag:
            run += 1
            if run >= run_frames:
                if first is None:
                    first = index - run_frames + 1
                last = index
        else:
            run = 0
    if first is None or last is None:
        return _result(float(arr.shape[0]) / sr)

    start = max(0, int(first * hop - pad_head * sr))
    end = min(arr.shape[0], int((last + 1) * hop + pad_tail * sr))
    trimmed = arr[start:end]
    if trimmed.shape[0] < arr.shape[0]:
        _save_mono_wav(path, trimmed, sr)
    return _result(float(trimmed.shape[0]) / sr, start)


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def pad_wav_to_duration(src: Path, dest: Path, duration_s: float) -> Path:
    """A2V 要求音频不少于镜头时长；不足则尾部补静音。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    data, sample_rate = sf.read(str(src), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    need = max(1, int(math.ceil(max(0.0, float(duration_s)) * max(1, int(sample_rate)))))
    if arr.shape[0] < need:
        arr = np.concatenate([arr, np.zeros(need - arr.shape[0], dtype=np.float32)])
    elif arr.shape[0] > need:
        # 镜头被 max_frames 截断时，A2V 只吃前 need 采样；成片混音仍用原 wav
        arr = arr[:need]
    _save_mono_wav(dest, arr, int(sample_rate))
    return dest


def _speak_with_fit(
    speech,
    *,
    shot_id: str,
    line: str,
    profile: dict,
    max_seconds: float,
    scratch: Path,
) -> list[tuple[str, Path, float]]:
    """合成整句台词，并把它切成每段都放得进单镜时长的若干段。

    **台词只合成一次**，语速、语气完全由原文和 emotion 决定，绝不为了凑时长去调语速
    或重念（重念会换一次 take，同一句前后语调就不一致了）。超长时只做两件事：

    1. 在句中静音处切开（同一段音频切的，语调天然连贯）——首选；
    2. 找不到足够静音点时（一口气念完），把整句原样放进一个更长的镜头。

    返回 [(分段文本, wav 路径, 时长)]；第 2 种情况的时长可以超过 max_seconds。
    """
    audio, sample_rate = speech.synthesize(
        text=line, speaker=profile["voice"], instruct=profile["instruct"]
    )
    whole = scratch / f"{shot_id}-whole.wav"
    _save_mono_wav(whole, audio, sample_rate)
    trimmed = trim_silence_wav(whole, report_pad=True)
    duration, head_pad = trimmed if isinstance(trimmed, tuple) else (trimmed, 0)
    if duration <= max_seconds:
        return [(line, whole, duration)]

    # 需要几段才够：按实测总长向上取整
    count = max(2, int(math.ceil(duration / max_seconds)))
    data, sr = sf.read(str(whole), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    cuts = silence_cuts(arr, int(sr))
    dests = [scratch / f"{shot_id}-{index:02d}.wav" for index in range(1, count + 1)]
    durations = split_wav_at_cuts(
        whole,
        dests,
        cuts,
        max_seconds,
        sample_rate=int(sr),
        head_pad=head_pad,
    )
    if durations and all(value <= max_seconds + 1e-3 for value in durations):
        total = sum(durations) or 1.0
        ratios: list[float] = []
        running = 0.0
        for value in durations:
            running += value
            ratios.append(running / total)
        texts = split_text_by_ratios(line, ratios) or plan_segments(
            line, max(2, math.ceil(spoken_chars(line) / len(durations)))
        )
        if len(texts) == len(durations):
            print(
                f"镜头 {shot_id} 的台词 {duration:.2f}s 超过 {max_seconds:.2f}s，"
                f"已在静音处切成 {len(durations)} 段（同一次配音，语调连贯）："
                f"{' / '.join(t.strip() for t in texts)}"
            )
            return list(zip(texts, dests, durations))

    # 一口气念完、没有可用静音点：保持原样，让镜头跟着配音变长。
    # 宁可这一镜久一点，也不重念或改语速去迁就时长。
    print(
        f"镜头 {shot_id} 的台词 {duration:.2f}s，句中没有可切的停顿；"
        f"保持原样、不重念，镜头按 {duration:.2f}s 生成。"
    )
    return [(line, whole, duration)]


def _generate_dialogue(
    settings: Settings, story: dict, out: Path, force: bool, only: str | None = None
) -> list[dict]:
    """生成对白，必要时把镜头拆成多个子镜头，保证每句都念得完。

    拆分结果按原镜头顺序写回 story（子镜头复用父镜头的静帧），
    分镜文件随后会落盘，因此文生图阶段不用重跑。

    ``only``：逗号分隔的镜头 id。只重新合成这些镜头的对白。
    其余镜头已有 wav 就留着；还没配音的镜头这次跳过并提示，不因此中断
    （网页按镜头重做配音走的就是这条路径）。缺的配音留给之后单独补，
    或去掉 ``--only`` 再跑一遍。``--only`` 不自动拆镜。
    """
    voice_dir = artifacts_sub(out, "audio", "voice")
    voice_dir.mkdir(parents=True, exist_ok=True)
    scratch = artifacts_sub(out, "audio", "scratch")
    characters = _character_map(story)
    max_seconds = snap_frames(settings.video_frames_max) / max(1, settings.video_fps)
    wanted: set[str] | None = None
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
    jobs = []
    for shot in story["shots"]:
        line = str(shot.get("dialogue") or "").strip()
        if not line:
            continue
        if wanted is not None and str(shot["id"]) not in wanted:
            continue
        jobs.append((shot, voice_dir / f"{shot['id']}.wav", _speaker_profile(shot, characters), line))

    def _reusable(dest: Path) -> bool:
        """已有 wav 只有在不超单镜时长时才复用。

        旧产物（比如上限还是 97 帧时生成的）可能比现在的镜头上限长，
        直接复用会重演「念不完」，所以这里按当前上限复核一遍。
        """
        if not dest.exists():
            return False
        if force:
            return False
        return trim_silence_wav(dest) <= max_seconds + 1e-3

    if wanted is not None:
        missing = _dialogue_ids_missing_voice(story, voice_dir, wanted)
        if missing:
            chosen = [str(s["id"]) for s in story["shots"] if str(s["id"]) in wanted]
            label = ", ".join(chosen) if chosen else ", ".join(sorted(wanted))
            print(
                f"--only 只合成 {label}。"
                f"镜头 {', '.join(missing)} 有对白但还没有配音，这次先留着；"
                "要补上就再跑这些镜头，或去掉 --only 跑一遍 audio。"
            )
        # --only 模式只重念指定镜头，绝不自动拆镜改写 shots（拆镜会动整个分镜，
        # 和「只重做这几个镜头」的预期相悖）；要拆镜请不带 --only 跑。
        force = force or True

    pending = [item for item in jobs if not _reusable(item[1])]
    if not pending:
        return [
            {"shot_id": shot["id"], "path": str(dest), **profile, "text": line}
            for shot, dest, profile, line in jobs
            if dest.exists()
        ]

    speech = open_speech(settings)
    try:
        return _dialogue_jobs(
            speech,
            story,
            jobs,
            voice_dir=voice_dir,
            scratch=scratch,
            max_seconds=max_seconds,
            force=force,
            wanted=wanted,
        )
    finally:
        speech.close()


def _dialogue_jobs(
    speech,
    story: dict,
    jobs,
    *,
    voice_dir: Path,
    scratch: Path,
    max_seconds: float,
    force: bool,
    wanted: set[str] | None,
) -> list[dict]:
    results: list[dict] = []
    # 每个对白镜头的处理结果：单段就是原镜头，多段则是一串子镜头
    by_shot: dict[str, list[dict]] = {}
    split_count = 0
    for shot, dest, profile, line in jobs:
        if dest.exists() and not force:
            # 能走到这里说明长度复核通过（见 _reusable）
            results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": line})
            by_shot[str(shot["id"])] = [shot]
            continue
        if dest.exists():
            # 旧 wav 超长：删掉重念，让 _speak_with_fit 按当前上限切段
            dest.unlink()
        print(f"对白 {shot['id']} · {profile['voice']} · {profile['emotion']}：{line}")
        segments = _speak_with_fit(
            speech,
            shot_id=str(shot["id"]),
            line=line,
            profile=profile,
            max_seconds=max_seconds,
            scratch=scratch,
        )
        if len(segments) == 1:
            text, wav, _ = segments[0]
            wav.replace(dest)
            shot["dialogue"] = text
            results.append({"shot_id": shot["id"], "path": str(dest), **profile, "text": text})
            by_shot[str(shot["id"])] = [shot]
            continue
        # 一句台词放不进一个镜头：拆成相邻子镜头，各段独立配音
        if wanted is not None:
            raise SystemExit(
                f"镜头 {shot['id']} 的台词超过单镜上限，需要拆镜；"
                "--only 模式不改分镜，请不带 --only 跑 audio（或先在分镜里手工拆好）。"
            )
        children: list[dict] = []
        for offset, (text, wav, _) in enumerate(segments, start=1):
            child = _split_shot(shot, offset, text)
            child_wav = voice_dir / f"{child['id']}.wav"
            wav.replace(child_wav)
            children.append(child)
            results.append({"shot_id": child["id"], "path": str(child_wav), **profile, "text": text})
        by_shot[str(shot["id"])] = children
        split_count += len(children)
        if dest.exists():
            dest.unlink()

    if split_count:
        rebuilt: list[dict] = []
        for shot in story["shots"]:
            rebuilt.extend(by_shot.get(str(shot["id"]), [shot]))
        story["shots"] = rebuilt
        story["auto_split_dialogue"] = split_count
        print(
            f"台词太长，已自动拆成 {split_count} 个子镜头（复用同一静帧，不需要重跑文生图）。"
            f"分镜现在 {len(rebuilt)} 镜。"
        )
    return results


def _split_shot(parent: dict, offset: int, text: str) -> dict:
    """由父镜头派生第 offset 段子镜头（offset 从 1 起）。

    子镜头沿用父镜头的人物、场景、情绪和静帧，并记下 ``reuse_frame``：
    视频阶段直接复用父镜头那张静帧，所以拆镜不用重跑文生图。
    """
    parent_id = str(parent["id"])
    child = dict(parent)
    # 用 _p1/_p2 这种明确后缀，避免和分镜里本来就有的 s007b 之类 id 撞车
    child["id"] = f"{parent_id}_p{offset}"
    child["dialogue"] = text
    child["reuse_frame"] = parent_id
    # 静帧相同但运镜要有差异，否则几段视频会像同一段复制
    child["seed"] = stable_seed("split", parent_id, str(offset), text)
    for key in ("audio_duration", "audio_overflow", "video_frames", "video", "video_error"):
        child.pop(key, None)
    return child


def _dialogue_ids_missing_voice(story: dict, voice_dir: Path, wanted: set[str]) -> list[str]:
    """``--only`` 范围之外、有对白但还没有 wav 的镜头 id（按分镜顺序）。"""
    missing: list[str] = []
    for shot in story.get("shots") or []:
        sid = str(shot.get("id") or "")
        line = str(shot.get("dialogue") or "").strip()
        if not sid or not line or sid in wanted:
            continue
        if not (voice_dir / f"{sid}.wav").is_file():
            missing.append(sid)
    return missing


def music_block_reason(story: dict, out: Path) -> str | None:
    """配乐依赖全部对白镜的 wav 来定长。缺了就返回原因，齐了返回 None。"""
    missing = _dialogue_ids_missing_voice(story, artifacts_sub(out, "audio", "voice"), set())
    if not missing:
        return None
    return (
        f"配乐按全部对白镜的实测时长定长。这些镜头有对白、还没有配音：{', '.join(missing)}。"
        "先跑这些镜头的 audio，再跑 music。"
    )


def _merge_voice_records(story: dict, previous: list, fresh: list) -> list:
    """把本轮合成的配音并进已有清单；wav 已经不在的旧记录丢掉。"""
    by_id: dict[str, dict] = {}
    for item in previous or []:
        sid = str(item.get("shot_id") or "")
        if sid:
            by_id[sid] = item
    fresh_ids: set[str] = set()
    for item in fresh or []:
        sid = str(item.get("shot_id") or "")
        if not sid:
            continue
        by_id[sid] = item
        fresh_ids.add(sid)
    ordered = [str(shot.get("id") or "") for shot in story.get("shots") or []]
    ordered += [sid for sid in by_id if sid not in ordered]
    merged: list[dict] = []
    for sid in ordered:
        item = by_id.get(sid)
        if not item:
            continue
        path = Path(str(item.get("path") or ""))
        if sid not in fresh_ids and not path.is_file():
            continue
        merged.append(item)
    return merged


def apply_shot_timings(settings: Settings, story: dict, out: Path) -> None:
    """按对白 wav / 空镜默认，提前写齐每镜 video_frames（镜头计划时长）。

    有对白且已有 wav：时长跟着配音走（向上取 8k+1 格），不删字压短。
    有对白但还没有 wav：不动 video_frames，避免 --only 补几镜时把其余台词镜写成空镜时长。
    空镜：用 settings.video_frames。
    视频阶段服从这个数；配乐也按各镜 video_frames 加总定长，不依赖 clips。
    """
    voice_dir = artifacts_sub(out, "audio", "voice")
    fps = max(1, settings.video_fps)
    cap_frames = snap_frames(settings.video_frames_max)
    for shot in story["shots"]:
        voice = voice_dir / f"{shot['id']}.wav"
        line = str(shot.get("dialogue") or "").strip()
        if line and voice.exists():
            # 已有 wav 可能带首尾静音（老产物），这里再裁一次；对已裁过的文件幂等
            duration = trim_silence_wav(voice)
            shot["audio_duration"] = round(duration, 3)
            shot["video_frames"] = frames_for_duration(
                duration,
                fps,
                min_frames=settings.video_frames_min,
                max_frames=max(cap_frames, snap_frames(int(math.ceil(duration * fps)) + 1)),
            )
        elif line:
            shot.pop("audio_duration", None)
        else:
            shot.pop("audio_duration", None)
            shot["video_frames"] = int(settings.video_frames)
        shot.pop("audio_overflow", None)


def ensure_video_frames(settings: Settings, story: dict, out: Path, *, persist: Path | None = None) -> float:
    """保证分镜每镜都有 video_frames，返回计划片长（秒）。

    供 music 等阶段在渲视频前就能定 BGM 长度；若写了新字段可 persist 回 storyboard.json。
    """
    before = [(s.get("id"), s.get("video_frames"), s.get("audio_duration")) for s in story.get("shots") or []]
    apply_shot_timings(settings, story, out)
    after = [(s.get("id"), s.get("video_frames"), s.get("audio_duration")) for s in story.get("shots") or []]
    if persist is not None and before != after:
        dump_json(persist, story)
    timeline, _ = _timeline_seconds(settings, out, story)
    return timeline


def assert_no_overflow(settings: Settings, story: dict) -> None:
    """兜底：确认没有镜头的画面比自己的配音短。

    镜头时长现在是跟着配音走的（frames_for_duration 向上取格），所以正常情况
    不会触发。真触发了说明 video_frames 是旧产物或算错了，必须报错——
    绝不允许静默截掉半句台词，这是最初「台词读不完」的根因。
    """
    fps = max(1, settings.video_fps)
    bad = [
        (
            str(shot.get("id") or "?"),
            float(shot["audio_duration"]),
            float(shot["video_frames"]) / fps,
        )
        for shot in story["shots"]
        if shot.get("audio_duration") is not None and shot.get("video_frames")
        and float(shot["video_frames"]) / fps + 1e-3 < float(shot["audio_duration"])
    ]
    if not bad:
        return
    listed = "\n".join(
        f"- {sid}：对白 {dur:.2f}s，镜头只有 {frames_s:.2f}s" for sid, dur, frames_s in bad
    )
    raise SystemExit(
        f"有镜头的画面比配音短，继续下去台词会被截断：\n{listed}\n"
        "请用 --force 重跑 audio 让 video_frames 按实测配音重算。"
    )


def _timeline_seconds(settings: Settings, out: Path, story: dict) -> tuple[float, str]:
    """成片计划时长（秒）：只认分镜 video_frames，不依赖已渲 clips。

    video_frames 由 audio（对白实测）或空镜默认提前写入；视频阶段服从该帧数。
    若个别镜尚未写入，临时用 wav / 默认帧估算，并在说明里标出。
    """
    fps = max(1, settings.video_fps)
    shots = story.get("shots") or []
    if shots and all(s.get("video_frames") for s in shots):
        frames_total = sum(int(s["video_frames"]) for s in shots)
        return max(2.0, frames_total / fps), "video_frames（分镜计划时长）"

    total = 0.0
    missing = 0
    voice_dir = artifacts_sub(out, "audio", "voice")
    for shot in shots:
        frames = shot.get("video_frames")
        if frames:
            total += float(frames) / fps
            continue
        missing += 1
        line = str(shot.get("dialogue") or "").strip()
        voice = voice_dir / f"{shot.get('id')}.wav"
        if line and voice.exists():
            total += max(0.1, wav_duration(voice))
            continue
        total += settings.video_frames / fps
    if missing:
        return max(2.0, total), f"部分估算（{missing}/{len(shots)} 镜尚无 video_frames）"
    return max(2.0, total), "video_frames（分镜计划时长）"


def _estimate_timeline(settings: Settings, out: Path, story: dict) -> float:
    return _timeline_seconds(settings, out, story)[0]


def _generate_music(settings: Settings, story: dict, out: Path, force: bool) -> Path:
    music_dir = artifacts_sub(out, "audio")
    music_dir.mkdir(parents=True, exist_ok=True)
    dest = music_dir / "bgm.wav"
    prompt = str(story.get("music_prompt") or "").strip() or default_music_prompt(
        str(story.get("style") or ""), str(story.get("title") or "")
    )
    seed = int(story.get("music_seed") or stable_seed("bgm", prompt, story.get("title", "")))
    # 缺省时把算出的 seed 写回内存，便于 UI / 下次复现
    if story.get("music_seed") is None:
        story["music_seed"] = seed
    timeline, source = _timeline_seconds(settings, out, story)
    # 配乐要比成片长一点：留出尾部淡出的余量，也让 -stream_loop 不需要真的绕回
    wanted = min(float(settings.music_max_seconds), max(8.0, timeline + 2.0))
    if dest.exists() and not force:
        have = wav_duration(dest)
        # 关键：旧配乐比成片短就必须重做，否则合成时只能靠循环拼接，中间留硬接缝
        if have + 0.05 >= timeline:
            print(f"已有配乐 {have:.1f}s ≥ 成片 {timeline:.1f}s，跳过：{dest}")
            return dest
        print(
            f"已有配乐只有 {have:.1f}s，短于成片 {timeline:.1f}s，重新生成："
            f"{have:.1f}s → {wanted:.0f}s（时间轴依据：{source}）"
        )
    print(f"配乐约 {wanted:.0f}s（计划片长 {timeline:.1f}s，依据 {source}）：{prompt}")
    render_music(
        settings,
        prompt=prompt,
        seconds=wanted,
        seed=seed,
        dest=dest,
        log=artifacts_sub(out, "logs", "bgm.log"),
    )
    return dest


def cmd_audio(args: argparse.Namespace) -> None:
    """只做对白（TTS），并写齐各镜 video_frames（供视频服从、配乐定长）。"""
    settings = Settings.load(resolve(args.config) if args.config else None)
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    voices = _generate_dialogue(settings, story, out, args.force, getattr(args, "only", None))
    apply_shot_timings(settings, story, out)
    # 兜底：自动拆镜之后仍有超长对白就不许继续，避免成片悄悄被截台词
    assert_no_overflow(settings, story)
    timeline, source = _timeline_seconds(settings, out, story)
    manifest = _load_manifest(out)
    voices = _merge_voice_records(story, manifest.get("voices") or [], voices)
    manifest.update(
        {
            "tts_model": settings.tts_model,
            "music_model": settings.music_model,
            "music_prompt": str(story.get("music_prompt") or default_music_prompt(story.get("style", ""), story.get("title", ""))),
            "voices": voices,
            "timeline_s": round(timeline, 2),
            "timeline_source": source,
        }
    )
    dump_json(artifacts_sub(out, "audio", "manifest.json"), manifest)
    # 把推断出的音色写回人物表，方便下次改分镜
    by_name = {c["name"]: c for c in story.get("characters") or []}
    for item in voices:
        char = by_name.get(item["speaker"])
        if not char:
            continue
        char.setdefault("gender", item["gender"])
        char.setdefault("voice", item["voice"])
    if not story.get("music_prompt"):
        story["music_prompt"] = manifest["music_prompt"]
    dump_json(story_path, story)
    timed = sum(1 for shot in story["shots"] if shot.get("audio_duration") is not None)
    print(
        f"对白就绪：{len(voices)} 条（其中 {timed} 镜按配音写入 video_frames），"
        f"计划片长 {manifest['timeline_s']}s（{source}）；配乐可按此定长，不必等视频"
    )


def _load_manifest(out: Path) -> dict:
    path = artifacts_sub(out, "audio", "manifest.json")
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            pass
    return {}


def cmd_music(args: argparse.Namespace) -> None:
    """按分镜 video_frames 计划片长生成配乐（不依赖 clips 已渲完）。"""
    settings = Settings.load(resolve(args.config) if args.config else None)
    ensure_runtime_layout(musicgen_dir=settings.musicgen_dir)
    out = resolve(args.out)
    story_path = out / "storyboard.json"
    if not story_path.exists():
        raise SystemExit(f"还没有分镜：{story_path}")
    story = load_json(story_path)
    blocked = music_block_reason(story, out)
    if blocked:
        raise SystemExit(blocked)
    # 对白 wav 已齐：按实测写入 video_frames，再定配乐长度
    ensure_video_frames(settings, story, out, persist=story_path)
    timeline, source = _timeline_seconds(settings, out, story)
    music = _generate_music(settings, story, out, args.force)
    have = wav_duration(music)
    # music_seed 可能是本轮补上的：写回分镜，方便 UI 读取与复现
    if story.get("music_seed") is not None:
        dump_json(story_path, story)
    manifest = _load_manifest(out)
    manifest.update(
        {
            "music": str(music),
            "music_seconds": round(have, 3),
            "music_model": settings.music_model,
            "music_seed": story.get("music_seed"),
            "timeline_s": round(timeline, 3),
            "timeline_source": source,
        }
    )
    dump_json(artifacts_sub(out, "audio", "manifest.json"), manifest)
    if have + 0.05 < timeline:
        print(
            f"提示：配乐 {have:.1f}s 仍短于成片 {timeline:.1f}s（受 music_max_seconds="
            f"{settings.music_max_seconds:.0f} 限制），合成时会循环补齐。"
        )
    print(f"配乐就绪：{music}（{have:.1f}s，成片 {timeline:.1f}s，依据 {source}）")
