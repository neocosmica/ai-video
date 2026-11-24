"""校验已写好的分镜 JSON，并打印每个对白镜头的预估时长（写分镜时用它自查）。

不跑成片、不加载模型，秒级返回：

    ./scripts/run.sh screenplay --out projects/<片名> --novel projects/<片名>/novel.txt

或直接用项目名（novel / storyboard 自动取该文件夹）：

    ./scripts/run.sh screenplay --project <片名>

超长台词只提示、不报错：audio 阶段会自动在静音处切段。想少一次返工就照提示先拆好。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from pipeline.config import Settings
from pipeline.util import dump_json, has_cjk, load_json, resolve, snap_frames, stable_seed


SCHEMA_HINT = """分镜可由本机 LLM 从小说生成，也可手写/Agent 写入 projects/<片名>/storyboard.json。

生成：
  ./scripts/run.sh storyboard --project <片名>
  # 或一键：./scripts/run.sh produce --project <片名>（含 storyboard → screenplay → …）

最小结构：
{
  "title": "短剧标题",
  "style": "一句中文画面风格",
  "music_prompt": "English instrumental BGM description, no vocals",
  "music_seed": 123456789,
  "characters": [
    {
      "id": "拼音id",
      "name": "中文名",
      "gender": "female|male",
      "voice": "Serena|Vivian|Dylan|Uncle_Fu|Eric",
      "appearance": "年龄性别发型衣服，勿编造原文没有的配饰"
    }
  ],
  "shots": [
    {
      "characters": ["拼音id或中文名"],
      "location": "地点",
      "image_prompt": "中文画面：景别、地点、动作、光线，不要写对白字",
      "motion_prompt": "English camera/motion only",
      "speaker": "说话人中文名，没有就空",
      "dialogue": "对白，没有就空",
      "emotion": "calm|warm|sad|tense|bitter|nostalgic|surprised|whisper 或中文情绪"
    }
  ]
}

音色：女 Serena/Vivian，男 Dylan/Uncle_Fu/Eric。情绪写在镜头上，影响 TTS instruct。
约束：竖屏短剧，每镜约 2 秒（空镜）；motion_prompt 与 music_prompt 必须是英文。
台词长度：**按小说原样写，不要为了时长改写、删字或压短**。配音语速和语气由原文与 emotion 决定，
镜头时长跟着配音走；一句话太长就按语义在标点处分成相邻几个镜头，让运镜跟得上，但文字不许变。
对白硬性要求：小说里所有引号对白（「」 “” ""）必须原样出现在某个镜头的 dialogue 里，
允许跨相邻镜头连续覆盖，禁止漏句删字。
写分镜前先拿单镜上限（只读总内存配置，秒级）：
./scripts/run.sh budget --out projects/xxx
写完用 --strict 硬校验（超额度会给出可照抄的拆法，不许删字）：
./scripts/run.sh screenplay --strict --project xxx
然后一句话成片：
./scripts/run.sh produce --project xxx
（audio 阶段按实测配音定镜头长度，必要时在句中停顿处自动切段，绝不静默截台词。）
"""

# 中文弯引号、直角引号、直引号
_DIALOGUE_RE = re.compile(r"[「“\"]([^」”\"]+)[」”\"]")


def extract_novel_dialogues(text: str) -> list[str]:
    """抽出小说里的引号对白，保序去重。"""
    found: list[str] = []
    seen: set[str] = set()
    for match in _DIALOGUE_RE.finditer(text or ""):
        line = match.group(1).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        found.append(line)
    return found


def _coverage_blob(shots: list[dict]) -> str:
    # 不加分隔符：同一句台词允许按标点拆到相邻镜头里，拼回来仍算覆盖
    return _spoken_key("".join(str(shot.get("dialogue") or "") for shot in shots))


_SPOKEN_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)


def _spoken_key(text: str) -> str:
    return _SPOKEN_STRIP.sub("", text or "")


def count_spoken_chars(text: str) -> int:
    """只数会被念出来的字：去掉空白、标点和引号。"""
    return len(_spoken_key(text))


# Qwen3-TTS 中文实测语速（含句中停顿），单位：字/秒。
# 用它把「多少个字」换算成「大概念几秒」，写分镜时就能预判会不会超。
CHARS_PER_SECOND = 2.4


def estimate_speech_seconds(text: str, chars_per_second: float = CHARS_PER_SECOND) -> float:
    """按实测语速估这句台词要念多久。"""
    return count_spoken_chars(text) / max(0.1, float(chars_per_second))


def dialogue_char_limit(
    max_frames: int,
    fps: int,
    chars_per_second: float = CHARS_PER_SECOND,
) -> int:
    """单镜台词字数上限：按实测中文 TTS 语速和单镜最长帧数估。

    这是「不用拆镜」的安全额度；超过也不报错，audio 阶段会自动拆。
    """
    seconds = max(1.0, float(max_frames) / max(1, int(fps)))
    return max(4, int(seconds * chars_per_second))


def _split_shot_label(sid: str, index: int) -> str:
    if index == 0:
        return sid
    if index < 26:
        return f"{sid}{'abcdefghijklmnopqrstuvwxyz'[index]}"
    return f"{sid}_p{index + 1}"


def dialogue_budget_violations(
    shots: list[dict],
    limit: int,
    max_seconds: float,
    *,
    tolerance: float = 1.15,
    strict: bool = False,
) -> tuple[list[str], list[dict]]:
    """返回 (预估行, 超额度镜头结构化列表)。"""
    from pipeline.dialogue import plan_segments

    soft_cap = max_seconds * tolerance
    rows: list[str] = []
    violations: list[dict] = []
    for shot in shots:
        line = str(shot.get("dialogue") or "").strip()
        if not line:
            continue
        seconds = estimate_speech_seconds(line)
        chars = count_spoken_chars(line)
        sid = str(shot.get("id") or "?")
        if seconds > soft_cap or (strict and chars > limit):
            pieces = plan_segments(line, limit)
            piece_list = pieces if len(pieces) > 1 else [line]
            plan = " / ".join(piece.strip() for piece in piece_list)
            violations.append(
                {
                    "id": sid,
                    "chars": chars,
                    "seconds": round(seconds, 1),
                    "limit": limit,
                    "max_seconds": round(max_seconds, 1),
                    "splits": [
                        {"id": _split_shot_label(sid, i), "dialogue": piece.strip()}
                        for i, piece in enumerate(piece_list)
                    ],
                }
            )
            rows.append(
                f"  ~ {sid}：{chars} 字 ≈ {seconds:.1f}s（镜头参考 {max_seconds:.1f}s / {limit} 字）"
                f"→ 若想分镜可断在：{plan}"
            )
        else:
            rows.append(f"    {sid}：{chars} 字 ≈ {seconds:.1f}s")
    return rows, violations


def report_dialogue_budget(
    shots: list[dict],
    limit: int,
    max_seconds: float,
    *,
    tolerance: float = 1.15,
    strict: bool = False,
) -> None:
    """打印每个对白镜头的预估时长，供写分镜时有个时间概念。

    默认只提示、不失败。``strict=True`` 时：超额度镜头给出可照抄的拆法并以退出码 1 失败，
    强迫写分镜时就把长对白拆成相邻多镜（台词原样，不许删字）。

    ``tolerance``：略微超出（默认 15% 以内）不提示，避免为 0.2 秒把句子拆碎。
    """
    rows, violations = dialogue_budget_violations(
        shots, limit, max_seconds, tolerance=tolerance, strict=strict
    )
    if not rows:
        return
    print(
        f"对白时长预估（语速 {CHARS_PER_SECOND} 字/秒，仅供参考；镜头会跟着配音长度走）："
    )
    print("\n".join(rows))
    if not violations:
        return
    if strict:
        blocks: list[str] = []
        for item in violations:
            sid = item["id"]
            blocks.append(
                f"{sid}：{item['chars']} 字 ≈ {item['seconds']}s，超过单镜额度 {item['limit']} 字"
            )
            blocks.append("  请拆成相邻镜头，台词原样不要删字：")
            for part in item["splits"]:
                blocks.append(f"    {part['id']}  → {part['dialogue']}")
        raise SystemExit(
            "分镜有镜头对白超过单镜额度（--strict）。"
            "**不要删字压短**，按下面断点拆成相邻镜头，每镜换景别/角度：\n"
            + "\n".join(blocks)
        )
    print(
        f"提示：上面 {len(violations)} 个镜头对白较长。**不需要为时长改写台词**——配音按原文一次念完，"
        "镜头时长跟着配音走；audio 阶段还会在句中停顿处自动切段拆镜（同一次配音，语调连贯）。"
        "只有想控制运镜节奏时，才按上面的断点自己分镜。"
    )


def assert_dialogue_coverage(novel_text: str, shots: list[dict]) -> None:
    """分镜必须覆盖小说全部引号对白，否则直接失败，避免静默丢台词。"""
    expected = extract_novel_dialogues(novel_text)
    if not expected:
        return
    blob = _coverage_blob(shots)
    missing = [line for line in expected if _spoken_key(line) not in blob]
    if missing:
        listed = "\n".join(f"- {line}" for line in missing)
        raise SystemExit(
            "分镜漏了对白。下面这些原文引号台词没有出现在任何镜头的 dialogue 里：\n"
            f"{listed}\n"
            "请改 storyboard.json：一句重要对白一个镜头，speaker 写中文名。"
        )


def re_id(value: str) -> str:
    cleaned = re.sub(r"[^\w]+", "", value, flags=re.UNICODE).strip("_")
    return cleaned[:32]


def _speaker_name(token, characters: list[dict]) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    for char in characters:
        if text == char["id"] or text == char["name"]:
            return char["name"]
    return text


def _resolve_cast(tokens: list, characters: list[dict]) -> list[str]:
    ids = []
    for token in tokens or []:
        text = str(token).strip()
        for char in characters:
            if text == char["id"] or text == char["name"]:
                if char["id"] not in ids:
                    ids.append(char["id"])
                break
    return ids


def _normalize_characters(chars: list[dict]) -> list[dict]:
    out = []
    used = set()
    for index, raw in enumerate(chars, start=1):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        appearance = str(raw.get("appearance") or "").strip()
        if not name:
            continue
        cid = re_id(str(raw.get("id") or name)) or f"c{index}"
        base = cid
        n = 2
        while cid in used:
            cid = f"{base}{n}"
            n += 1
        used.add(cid)
        gender = str(raw.get("gender") or "").strip().lower()
        if gender not in {"female", "male"}:
            from pipeline.voices import infer_gender

            gender = infer_gender(appearance, gender)
        voice = str(raw.get("voice") or "").strip()
        item = {
            "id": cid,
            "name": name,
            "gender": gender,
            "appearance": appearance or name,
            "seed": int(raw["seed"]) if raw.get("seed") is not None else stable_seed("portrait", name, appearance),
        }
        if voice:
            item["voice"] = voice
        out.append(item)
    return out


def normalize_storyboard(
    data: dict,
    *,
    novel_path: Path | None,
    max_shots: int,
    dialogue_limit: int | None = None,
    max_seconds: float | None = None,
    strict: bool = False,
) -> dict:
    if not isinstance(data, dict):
        raise SystemExit("storyboard.json 必须是一个 JSON 对象")
    characters = _normalize_characters(data.get("characters") or [])
    shots_in = data.get("shots") or []
    if not isinstance(shots_in, list) or not shots_in:
        raise SystemExit("storyboard.json 里至少要有 1 个镜头")
    shots: list[dict] = []
    for index, raw in enumerate(shots_in, start=1):
        if len(shots) >= max_shots:
            break
        if not isinstance(raw, dict):
            continue
        image_prompt = str(raw.get("image_prompt") or "").strip()
        motion = str(raw.get("motion_prompt") or "").strip()
        if not image_prompt:
            continue
        if has_cjk(motion):
            raise SystemExit(
                f"镜头 {raw.get('id') or index} 的 motion_prompt 必须是英文，当前是：{motion}"
            )
        shot_id = str(raw.get("id") or f"s{index:03d}").strip() or f"s{index:03d}"
        emotion = str(raw.get("emotion") or "").strip()
        shot = {
            "id": shot_id,
            "characters": _resolve_cast(raw.get("characters") or [], characters),
            "location": str(raw.get("location") or "").strip(),
            "image_prompt": image_prompt,
            "motion_prompt": motion or "Slow cinematic push in, subtle natural movement.",
            "speaker": _speaker_name(raw.get("speaker"), characters),
            "dialogue": str(raw.get("dialogue") or "").strip(),
            "seed": int(raw["seed"]) if raw.get("seed") is not None else stable_seed(
                shot_id, image_prompt, motion
            ),
        }
        if emotion:
            shot["emotion"] = emotion
        elif shot["dialogue"]:
            shot["emotion"] = "calm"
        for key in ("video", "video_error", "reuse_frame"):
            if key in raw:
                shot[key] = raw[key]
        shots.append(shot)
    if not shots:
        raise SystemExit("没有可用镜头：每个镜头都需要非空的 image_prompt")
    from pipeline.voices import default_music_prompt

    title = str(data.get("title") or (novel_path.stem if novel_path else "untitled")).strip()
    style = str(data.get("style") or "写实华语短剧，自然肤色，电影感光线，浅景深").strip()
    music_prompt = str(data.get("music_prompt") or "").strip() or default_music_prompt(style, title)
    if has_cjk(music_prompt):
        raise SystemExit("music_prompt 必须是英文，方便 MusicGen 理解")
    novel_file = novel_path
    if novel_file is None:
        candidate = str(data.get("novel") or "").strip()
        if candidate:
            novel_file = Path(candidate)
    if novel_file and novel_file.exists():
        assert_dialogue_coverage(novel_file.read_text(encoding="utf-8"), shots)
    if dialogue_limit is not None:
        report_dialogue_budget(
            shots,
            dialogue_limit,
            float(max_seconds or 0.0),
            strict=strict,
        )
    spoken = sum(1 for shot in shots if shot.get("dialogue"))
    music_seed = (
        int(data["music_seed"])
        if data.get("music_seed") is not None
        else stable_seed("bgm", music_prompt, title)
    )
    return {
        "title": title,
        "style": style,
        "music_prompt": music_prompt,
        "music_seed": music_seed,
        "novel": str(novel_path) if novel_path else str(data.get("novel") or ""),
        "max_shots": max_shots,
        "dialogue_char_limit": dialogue_limit,
        "shot_max_seconds": round(float(max_seconds), 2) if max_seconds else None,
        "truncated": bool(data.get("truncated", False)) or len(shots_in) > max_shots,
        "dialogue_lines": spoken,
        "characters": characters,
        "shots": shots,
    }


def validate_or_explain(
    story_path: Path,
    novel_path: Path | None,
    settings: Settings,
    force: bool,
    *,
    strict: bool = False,
) -> Path:
    if not story_path.exists():
        raise SystemExit(
            f"还没有分镜文件：{story_path}\n\n{SCHEMA_HINT}"
        )
    raw = load_json(story_path)
    fps = max(1, settings.video_fps)
    max_seconds = snap_frames(settings.video_frames_max) / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    normalized = normalize_storyboard(
        raw,
        novel_path=novel_path,
        max_shots=settings.max_shots,
        dialogue_limit=limit,
        max_seconds=max_seconds,
        strict=strict,
    )
    if force or raw != normalized:
        dump_json(story_path, normalized)
        print(
            f"已校验分镜：{story_path}，{len(normalized['shots'])} 镜，"
            f"{len(normalized['characters'])} 个角色，对白 {normalized.get('dialogue_lines', 0)} 条。"
        )
    else:
        print(
            f"分镜已就绪：{story_path}，{len(normalized['shots'])} 镜，"
            f"对白 {normalized.get('dialogue_lines', 0)} 条。"
        )
    return story_path


def cmd_screenplay(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    if args.max_shots:
        settings.max_shots = args.max_shots
    out = resolve(args.out)
    novel = resolve(args.novel) if args.novel else None
    if novel and not novel.exists():
        raise SystemExit(f"找不到小说：{novel}")
    out.mkdir(parents=True, exist_ok=True)
    # 若未显式传小说，仍尝试用 storyboard 里的 novel 字段做对白覆盖检查
    story_path = out / "storyboard.json"
    src = getattr(args, "storyboard", None)
    if src:
        candidate = resolve(src)
        if not candidate.exists():
            raise SystemExit(f"找不到分镜：{candidate}")
        if candidate.resolve() != story_path.resolve():
            if story_path.exists() and not getattr(args, "force", False):
                print(f"输出目录已有分镜，保留不覆盖：{story_path}")
            else:
                story_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, story_path)
                print(f"已放入分镜：{candidate} → {story_path}")
    if (novel is None or not novel.exists()) and story_path.exists():
        linked = str(load_json(story_path).get("novel") or "").strip()
        if linked:
            novel = Path(linked)
    # 还没有？项目文件夹里就放着小说（novel.txt 优先），自动找同一部的
    if (novel is None or not novel.exists()) and story_path.exists():
        from pipeline.projects import novel_candidates

        for cand in novel_candidates(story_path):
            novel = cand
            break
    validate_or_explain(
        story_path,
        novel if novel and novel.exists() else None,
        settings,
        args.force,
        strict=bool(getattr(args, "strict", False)),
    )
