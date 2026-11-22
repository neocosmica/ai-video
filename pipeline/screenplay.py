from __future__ import annotations

import argparse
import re
from pathlib import Path

from pipeline.config import Settings
from pipeline.util import dump_json, has_cjk, load_json, resolve, stable_seed


SCHEMA_HINT = """分镜不由本机 LLM 生成。请让 Cursor Agent 阅读小说后写入 storyboard.json。

最小结构：
{
  "title": "短剧标题",
  "style": "一句中文画面风格",
  "music_prompt": "English instrumental BGM description, no vocals",
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
约束：竖屏短剧，每镜约 2 秒；能少就少；motion_prompt 与 music_prompt 必须是英文。
写好后：./scripts/run.sh screenplay --out output/demo
"""


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


def normalize_storyboard(data: dict, *, novel_path: Path | None, max_shots: int) -> dict:
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
        for key in ("video", "video_error"):
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
    return {
        "title": title,
        "style": style,
        "music_prompt": music_prompt,
        "novel": str(novel_path) if novel_path else str(data.get("novel") or ""),
        "max_shots": max_shots,
        "truncated": bool(data.get("truncated", False)) or len(shots_in) > max_shots,
        "characters": characters,
        "shots": shots,
    }


def validate_or_explain(story_path: Path, novel_path: Path | None, settings: Settings, force: bool) -> Path:
    if not story_path.exists():
        raise SystemExit(
            f"还没有分镜文件：{story_path}\n\n{SCHEMA_HINT}"
        )
    raw = load_json(story_path)
    normalized = normalize_storyboard(raw, novel_path=novel_path, max_shots=settings.max_shots)
    if force or raw != normalized:
        dump_json(story_path, normalized)
        print(f"已校验分镜：{story_path}，{len(normalized['shots'])} 镜，{len(normalized['characters'])} 个角色。")
    else:
        print(f"分镜已就绪：{story_path}，{len(normalized['shots'])} 镜。")
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
    validate_or_explain(out / "storyboard.json", novel if novel and novel.exists() else None, settings, args.force)
