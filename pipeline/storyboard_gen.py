"""用本机聊天模型把小说写成 storyboard.json。

多轮分步：① 角色与片头 → ② 镜头大纲（对白）→ ③ 分批补画面/运镜。
模型由 ``pipeline.backends`` 按配置加载。生成后交给 screenplay 规范化与对白校验。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline.backends import open_storyboard
from pipeline.backends.catalog import default_model
from pipeline.config import Settings
from pipeline.dialogue import plan_segments
from pipeline.screenplay import (
    count_spoken_chars,
    dialogue_char_limit,
    extract_novel_dialogues,
)
from pipeline.util import (
    chunk_novel,
    dump_json,
    extract_json,
    has_cjk,
    resolve,
    snap_frames,
)
from pipeline.voices import (
    VOICE_MODE_CLONE,
    default_music_prompt,
    default_voice_design,
    infer_gender,
    pick_voice,
    voice_mode,
)

DEFAULT_LLM_MODEL = default_model("storyboard").repo

_SYSTEM_BASE = """你是华语竖屏短剧分镜编剧。
思考结束后只输出一个 JSON 对象，不要 markdown，不要解释。
台词必须照小说引号原文，禁止删字、改写、压短。"""

_EFFORT_HINT = {
    "low": "reasoning_effort=low：用极少步骤想完就输出；不要复述约束、不要英文长篇、不要起草完整逐镜字段。",
    "medium": "reasoning_effort=medium：简短规划即可，不要逐镜起草完整字段。",
    "high": "reasoning_effort=high：可稍细规划，但仍不要输出思考以外的解释。",
}

# 每步生成额度（思考额度统一用 settings.llm_think_max_tokens）
_STEP_GEN_MAX = {
    # 单步天花板仍受 llm_max_tokens（8192）约束；略抬高以容纳 think+JSON
    "cast": 3072,
    "outline": 4096,
    "enrich": 4096,
}
_ENRICH_BATCH = 4


def _system_prompt(effort: str) -> str:
    key = str(effort or "low").strip().lower()
    hint = _EFFORT_HINT.get(key, _EFFORT_HINT["low"])
    return f"{_SYSTEM_BASE}\n{hint}"


def _spoken_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE)


def _quote_block(quoted_lines: list[str]) -> str:
    if not quoted_lines:
        return "本段没有引号台词：所有镜头 dialogue 必须是 \"\"。\n"
    listed = "\n".join(f"- {line}" for line in quoted_lines)
    return (
        "本段必须原样覆盖的引号台词（dialogue 只能用这些字，可拆到相邻镜；"
        "禁止把旁白写进 dialogue）：\n"
        f"{listed}\n"
    )


def _chunk_note(chunk_index: int, chunk_total: int) -> str:
    if chunk_total <= 1:
        return ""
    return f"\n这是小说第 {chunk_index + 1}/{chunk_total} 段，只根据本段写；角色可重复出现。\n"


def _complete_json(
    llm: Any,
    *,
    label: str,
    system: str,
    user: str,
    max_tokens: int,
    think_max_tokens: int,
    temperature: float,
    retries: int,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        print(f"  {label}：尝试 {attempt + 1}/{retries}", flush=True)
        try:
            raw = llm.complete(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature if attempt == 0 else min(temperature, 0.2),
                think_max_tokens=think_max_tokens,
            )
            data = extract_json(raw)
            return data
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"    失败：{exc}", flush=True)
    raise ValueError(f"{label}失败：{last_err}")


def _step_cast(
    llm: Any,
    novel: str,
    *,
    settings: Settings,
    retries: int,
    temperature: float,
    system: str,
) -> dict:
    gen_max = _STEP_GEN_MAX["cast"]
    user = f"""根据小说提取片头与角色卡。只输出 JSON。

约束：
- 外貌只取原文可见特征
- gender：female/male；voice：女 Serena/Vivian，男 Dylan/Uncle_Fu/Eric（也可后改成 clone/design）
- voice_design：英文音色描述（给 VoiceDesign 文字造声用），根据外貌与气质写，例如
  "A soft-spoken woman in her twenties, slightly breathy, southern Mandarin tint"
  每个角色都要写，即使当前 voice 是预设音色
- music_prompt 必须英文、无人声器乐

结构：
{{
  "title": "短剧标题",
  "style": "写实华语短剧，自然肤色，电影感光线，浅景深",
  "music_prompt": "Sparse melancholic piano, soft rain, instrumental only, no vocals",
  "characters": [
    {{
      "id": "linwan",
      "name": "林晚",
      "gender": "female",
      "voice": "Serena",
      "voice_design": "A soft-spoken woman in her twenties, slightly breathy, calm Mandarin delivery",
      "appearance": "二十多岁女性，短发，深色风衣"
    }}
  ]
}}

小说：
<<<
{novel}
>>>
"""
    data = _complete_json(
        llm,
        label="① 角色与片头",
        system=system,
        user=user,
        max_tokens=min(gen_max, settings.llm_max_tokens),
        think_max_tokens=settings.llm_think_max_tokens,
        temperature=temperature,
        retries=retries,
    )
    chars = data.get("characters")
    if not isinstance(chars, list) or not chars:
        raise ValueError("角色步骤缺少 characters")
    return data


def _step_outline(
    llm: Any,
    novel: str,
    *,
    characters: list[dict],
    quoted_lines: list[str],
    max_shots: int,
    dialogue_limit: int,
    max_seconds: float,
    settings: Settings,
    retries: int,
    temperature: float,
    system: str,
    chunk_index: int = 0,
    chunk_total: int = 1,
) -> list[dict]:
    gen_max = _STEP_GEN_MAX["outline"]
    cast_json = json.dumps(characters, ensure_ascii=False, indent=2)
    user = f"""根据小说与角色卡，只写镜头大纲（不要 image_prompt / motion_prompt）。只输出 JSON。
{_chunk_note(chunk_index, chunk_total)}
硬约束：
- 最多 {max_shots} 个镜头；id 用 s001、s002…
- 单镜对白约 {dialogue_limit} 字 / {max_seconds:.1f} 秒；超长按标点拆镜，台词不许删字
- dialogue 只能是引号内台词；叙述镜头 dialogue 与 speaker 都必须 ""
- emotion：calm/warm/sad/tense/bitter/nostalgic/surprised/whisper
- beat：中文一句话说明本镜在干什么（给下一步写画面用）
- characters 填角色 id 数组
{_quote_block(quoted_lines)}
已有角色：
{cast_json}

结构：
{{
  "shots": [
    {{
      "id": "s001",
      "characters": ["linwan"],
      "location": "街角便利店门口",
      "speaker": "",
      "dialogue": "",
      "emotion": "calm",
      "beat": "林晚推门走进雨夜便利店"
    }},
    {{
      "id": "s002",
      "characters": ["linwan"],
      "location": "便利店柜台前",
      "speaker": "林晚",
      "dialogue": "引号原句",
      "emotion": "surprised",
      "beat": "林晚近景开口"
    }}
  ]
}}

小说：
<<<
{novel}
>>>
"""
    data = _complete_json(
        llm,
        label="② 镜头大纲",
        system=system,
        user=user,
        max_tokens=min(gen_max, settings.llm_max_tokens),
        think_max_tokens=settings.llm_think_max_tokens,
        temperature=temperature,
        retries=retries,
    )
    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("大纲步骤缺少 shots")
    return [s for s in shots if isinstance(s, dict)][:max_shots]


def _step_enrich_batch(
    llm: Any,
    novel: str,
    *,
    characters: list[dict],
    shots_batch: list[dict],
    settings: Settings,
    retries: int,
    temperature: float,
    system: str,
) -> list[dict]:
    gen_max = _STEP_GEN_MAX["enrich"]
    user = f"""为下列镜头大纲补画面与运镜字段。保持 id/dialogue/speaker/location/emotion/characters 原样。只输出 JSON。

约束：
- image_prompt 中文：景别/地点/动作/光线；人脸清晰；有对白时嘴微张；每镜换景别或角度
- motion_prompt 必须英文
- motion_level：subtle（对白近景）/ normal（空镜轻运镜）/ dynamic（明确肢体或镜头动作）
- 动作镜（dynamic）必须写 end_image_prompt：同一机位/景别，只写终点姿势差；禁止「已走进室内换景」；对白近景可省略 end_image_prompt
- 外貌只取角色卡与原文可见特征

角色：
{json.dumps(characters, ensure_ascii=False, indent=2)}

待补全镜头大纲：
{json.dumps(shots_batch, ensure_ascii=False, indent=2)}

结构：
{{
  "shots": [
    {{
      "id": "s001",
      "characters": ["linwan"],
      "location": "地点",
      "image_prompt": "中景，……",
      "motion_prompt": "She pushes the door open and steps inside",
      "motion_level": "dynamic",
      "end_image_prompt": "同一机位，门已推开，人物侧入门内半步",
      "speaker": "",
      "dialogue": "",
      "emotion": "calm"
    }}
  ]
}}

小说（供核对场景，勿改台词）：
<<<
{novel}
>>>
"""
    data = _complete_json(
        llm,
        label=f"③ 补画面（{shots_batch[0].get('id', '?')}…）",
        system=system,
        user=user,
        max_tokens=min(gen_max, settings.llm_max_tokens),
        think_max_tokens=settings.llm_think_max_tokens,
        temperature=temperature,
        retries=retries,
    )
    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("补画面步骤缺少 shots")
    return [s for s in shots if isinstance(s, dict)]


def _merge_enrich(outline: list[dict], enriched: list[dict]) -> list[dict]:
    """按 id 合并；缺补的镜保留大纲并填保守默认画面。"""
    by_id = {
        str(s.get("id") or "").strip(): s
        for s in enriched
        if str(s.get("id") or "").strip()
    }
    out: list[dict] = []
    for raw in outline:
        sid = str(raw.get("id") or "").strip()
        base = dict(raw)
        base.pop("beat", None)
        got = by_id.get(sid)
        if got:
            for key in (
                "image_prompt",
                "motion_prompt",
                "motion_level",
                "end_image_prompt",
            ):
                if got.get(key):
                    base[key] = got[key]
            # 大纲字段以大纲为准，防止模型改台词
            for key in ("id", "characters", "location", "speaker", "dialogue", "emotion"):
                if key in raw:
                    base[key] = raw[key]
        if not str(base.get("image_prompt") or "").strip():
            beat = str(raw.get("beat") or "").strip()
            base["image_prompt"] = (
                f"中景，{beat}，电影感光线，浅景深"
                if beat
                else "中景，场景交代，电影感光线，浅景深"
            )
        if not str(base.get("motion_prompt") or "").strip():
            base["motion_prompt"] = (
                "Locked-off camera, natural lip motion, subtle expression"
                if str(base.get("dialogue") or "").strip()
                else "Slow cinematic push in, subtle natural movement"
            )
        out.append(base)
    return out


def _generate_chunk(
    llm: Any,
    novel: str,
    *,
    settings: Settings,
    max_shots: int,
    dialogue_limit: int,
    max_seconds: float,
    quoted_lines: list[str],
    retries: int,
    temperature: float,
    chunk_index: int,
    chunk_total: int,
    cast_seed: dict | None = None,
) -> dict:
    """单段小说：角色 → 大纲 → 分批补画面。"""
    system = _system_prompt(str(getattr(settings, "llm_reasoning_effort", "low") or "low"))
    if cast_seed and isinstance(cast_seed.get("characters"), list) and cast_seed["characters"]:
        cast = {
            "title": cast_seed.get("title") or "",
            "style": cast_seed.get("style") or "",
            "music_prompt": cast_seed.get("music_prompt") or "",
            "characters": list(cast_seed["characters"]),
        }
        print("  ① 角色与片头：复用上一段", flush=True)
    else:
        cast = _step_cast(
            llm,
            novel,
            settings=settings,
            retries=retries,
            temperature=temperature,
            system=system,
        )
    outline = _step_outline(
        llm,
        novel,
        characters=list(cast.get("characters") or []),
        quoted_lines=quoted_lines,
        max_shots=max_shots,
        dialogue_limit=dialogue_limit,
        max_seconds=max_seconds,
        settings=settings,
        retries=retries,
        temperature=temperature,
        system=system,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
    )
    enriched: list[dict] = []
    for i in range(0, len(outline), _ENRICH_BATCH):
        batch = outline[i : i + _ENRICH_BATCH]
        try:
            enriched.extend(
                _step_enrich_batch(
                    llm,
                    novel,
                    characters=list(cast.get("characters") or []),
                    shots_batch=batch,
                    settings=settings,
                    retries=retries,
                    temperature=temperature,
                    system=system,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    补画面批次失败，用大纲兜底：{exc}", flush=True)
    shots = _merge_enrich(outline, enriched)
    return {
        "title": cast.get("title") or "",
        "style": cast.get("style") or "",
        "music_prompt": cast.get("music_prompt") or "",
        "characters": cast.get("characters") or [],
        "shots": shots,
    }


def _merge_storyboards(parts: list[dict], *, max_shots: int) -> dict:
    if not parts:
        raise SystemExit("模型没有返回任何分镜")
    base = dict(parts[0])
    chars: list[dict] = []
    seen_char: set[str] = set()
    shots: list[dict] = []
    for part in parts:
        for raw in part.get("characters") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            cid = str(raw.get("id") or name).strip()
            key = cid or name
            if not key or key in seen_char:
                continue
            seen_char.add(key)
            chars.append(raw)
        for raw in part.get("shots") or []:
            if isinstance(raw, dict):
                shots.append(raw)
    # 重新编号，避免跨段 id 冲突
    for index, shot in enumerate(shots[:max_shots], start=1):
        shot["id"] = f"s{index:03d}"
    base["characters"] = chars
    base["shots"] = shots[:max_shots]
    if len(shots) > max_shots:
        base["truncated"] = True
    return base


def _split_long_dialogues(data: dict, limit: int) -> dict:
    """超额度对白按标点拆成相邻镜；台词原样，换景别提示写进 image_prompt 后缀。"""
    shots_in = list(data.get("shots") or [])
    if not shots_in or limit <= 0:
        return data
    out: list[dict] = []
    for raw in shots_in:
        if not isinstance(raw, dict):
            continue
        line = str(raw.get("dialogue") or "").strip()
        if not line or count_spoken_chars(line) <= limit:
            out.append(raw)
            continue
        pieces = plan_segments(line, limit)
        if len(pieces) <= 1:
            out.append(raw)
            continue
        base_id = str(raw.get("id") or f"s{len(out) + 1:03d}")
        for i, piece in enumerate(pieces):
            shot = dict(raw)
            if i == 0:
                shot["id"] = base_id
            elif i < 26:
                shot["id"] = f"{base_id}{'abcdefghijklmnopqrstuvwxyz'[i]}"
            else:
                shot["id"] = f"{base_id}_p{i + 1}"
            shot["dialogue"] = piece.strip()
            img = str(shot.get("image_prompt") or "").strip()
            if i > 0 and img and "换角度" not in img:
                shot["image_prompt"] = f"{img}，换角度近景"
            motion = str(shot.get("motion_prompt") or "").strip()
            if i > 0:
                shot["motion_prompt"] = (
                    "Slight reframing, locked-off camera, subtle facial micro-movement"
                    if not motion
                    else motion
                )
            out.append(shot)
    data = dict(data)
    data["shots"] = out
    return data


def _scrub_fake_dialogues(data: dict, novel_text: str) -> dict:
    """清掉不在引号台词里的旁白 dialogue，避免 TTS 念旁白。"""
    expected = extract_novel_dialogues(novel_text)
    if not expected:
        # 整部没有引号对白：全部清空
        shots = []
        for raw in data.get("shots") or []:
            if not isinstance(raw, dict):
                continue
            shot = dict(raw)
            if str(shot.get("dialogue") or "").strip():
                shot["dialogue"] = ""
                shot["speaker"] = ""
            shots.append(shot)
        data = dict(data)
        data["shots"] = shots
        return data
    keys = [_spoken_key(line) for line in expected]
    cleared = 0
    shots = []
    for raw in data.get("shots") or []:
        if not isinstance(raw, dict):
            continue
        shot = dict(raw)
        line = str(shot.get("dialogue") or "").strip()
        if not line:
            shots.append(shot)
            continue
        key = _spoken_key(line)
        ok = any(key and (key in ek or ek in key) for ek in keys)
        if not ok:
            shot["dialogue"] = ""
            shot["speaker"] = ""
            cleared += 1
        shots.append(shot)
    if cleared:
        print(f"提示：清掉 {cleared} 条旁白误入的 dialogue。", flush=True)
    data = dict(data)
    data["shots"] = shots
    return data


def _ensure_missing_dialogues(data: dict, novel_text: str) -> dict:
    """模型漏了引号台词时，补对白镜头，避免校验直接失败。"""
    expected = extract_novel_dialogues(novel_text)
    if not expected:
        return data
    blob = _spoken_key(
        "".join(
            str(s.get("dialogue") or "")
            for s in (data.get("shots") or [])
            if isinstance(s, dict)
        )
    )
    missing = [line for line in expected if _spoken_key(line) not in blob]
    if not missing:
        return data
    shots = list(data.get("shots") or [])
    chars = [c for c in (data.get("characters") or []) if isinstance(c, dict)]
    default_speaker = str(chars[0].get("name") or "") if chars else ""
    default_cid = str(chars[0].get("id") or "") if chars else ""
    start = len(shots) + 1
    for offset, line in enumerate(missing):
        sid = f"s{start + offset:03d}"
        shots.append(
            {
                "id": sid,
                "characters": [default_cid] if default_cid else [],
                "location": "原场景",
                "image_prompt": "中近景，说话人面部清晰，嘴微张，浅景深，电影感光线",
                "motion_prompt": "Locked-off camera, natural lip motion, subtle expression",
                "speaker": default_speaker,
                "dialogue": line,
                "emotion": "calm",
            }
        )
    data = dict(data)
    data["shots"] = shots
    print(f"提示：模型漏了 {len(missing)} 条引号对白，已补镜头。", flush=True)
    return data


def _polish_fields(data: dict, *, title_fallback: str) -> dict:
    data = dict(data)
    title = str(data.get("title") or title_fallback).strip() or title_fallback
    style = str(data.get("style") or "写实华语短剧，自然肤色，电影感光线，浅景深").strip()
    music = str(data.get("music_prompt") or "").strip()
    if not music or has_cjk(music):
        music = default_music_prompt(style, title)
    chars_out: list[dict] = []
    for raw in data.get("characters") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        appearance = str(raw.get("appearance") or name).strip()
        gender = infer_gender(appearance, str(raw.get("gender") or ""))
        voice = pick_voice(gender, str(raw.get("voice") or "") or None, appearance)
        mode = voice_mode({"voice": voice, "voice_mode": raw.get("voice_mode")})
        design = str(raw.get("voice_design") or "").strip()
        if not design:
            design = default_voice_design(gender, appearance, name)
        item = {
            "id": str(raw.get("id") or name).strip() or name,
            "name": name,
            "gender": gender,
            "voice": voice,
            "voice_mode": mode,
            "voice_design": design,
            "appearance": appearance,
        }
        if mode == VOICE_MODE_CLONE:
            ref = str(raw.get("voice_ref") or "").strip()
            ref_text = str(raw.get("voice_ref_text") or "").strip()
            if ref:
                item["voice_ref"] = ref
            if ref_text:
                item["voice_ref_text"] = ref_text
        chars_out.append(item)
    shots_out: list[dict] = []
    for index, raw in enumerate(data.get("shots") or [], start=1):
        if not isinstance(raw, dict):
            continue
        image_prompt = str(raw.get("image_prompt") or "").strip()
        if not image_prompt:
            continue
        motion = str(raw.get("motion_prompt") or "").strip()
        if not motion or has_cjk(motion):
            motion = "Slow cinematic push in, subtle natural movement."
        shot = {
            "id": str(raw.get("id") or f"s{index:03d}").strip() or f"s{index:03d}",
            "characters": list(raw.get("characters") or []),
            "location": str(raw.get("location") or "").strip(),
            "image_prompt": image_prompt,
            "motion_prompt": motion,
            "speaker": str(raw.get("speaker") or "").strip(),
            "dialogue": str(raw.get("dialogue") or "").strip(),
            "emotion": str(raw.get("emotion") or ("calm" if raw.get("dialogue") else "")).strip(),
        }
        level = str(raw.get("motion_level") or "").strip().lower()
        if level in {"subtle", "normal", "dynamic"}:
            shot["motion_level"] = level
        end_prompt = str(raw.get("end_image_prompt") or "").strip()
        if end_prompt:
            shot["end_image_prompt"] = end_prompt
        if not shot["emotion"]:
            shot.pop("emotion", None)
        shots_out.append(shot)
    data["title"] = title
    data["style"] = style
    data["music_prompt"] = music
    data["characters"] = chars_out
    data["shots"] = shots_out
    return data


def generate_storyboard_dict(
    novel_text: str,
    settings: Settings,
    *,
    title_fallback: str = "untitled",
    retries: int = 2,
) -> dict:
    fps = max(1, settings.video_fps)
    max_seconds = snap_frames(settings.video_frames_max) / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    chunks = chunk_novel(novel_text, limit=1600)
    # 多段时每段镜头配额均分，至少 4
    per_chunk_shots = max(4, settings.max_shots // max(1, len(chunks)))
    temperature = float(settings.llm_temperature)

    llm = open_storyboard(settings)
    parts: list[dict] = []
    cast_seed: dict | None = None
    try:
        for index, chunk in enumerate(chunks):
            quoted = extract_novel_dialogues(chunk)
            max_shots = min(
                settings.max_shots,
                per_chunk_shots if len(chunks) > 1 else settings.max_shots,
            )
            print(
                f"生成分镜：段 {index + 1}/{len(chunks)}（角色→大纲→画面，最多 {max_shots} 镜）",
                flush=True,
            )
            try:
                data = _generate_chunk(
                    llm,
                    chunk,
                    settings=settings,
                    max_shots=max_shots,
                    dialogue_limit=limit,
                    max_seconds=max_seconds,
                    quoted_lines=quoted,
                    retries=retries,
                    temperature=temperature,
                    chunk_index=index,
                    chunk_total=len(chunks),
                    cast_seed=cast_seed,
                )
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"分镜生成失败：{exc}") from exc
            if not data.get("shots"):
                raise SystemExit("分镜生成失败：没有镜头")
            cast_seed = {
                "title": data.get("title"),
                "style": data.get("style"),
                "music_prompt": data.get("music_prompt"),
                "characters": data.get("characters"),
            }
            parts.append(data)
    finally:
        llm.close()

    merged = _merge_storyboards(parts, max_shots=settings.max_shots)
    merged = _polish_fields(merged, title_fallback=title_fallback)
    merged = _scrub_fake_dialogues(merged, novel_text)
    merged = _ensure_missing_dialogues(merged, novel_text)
    merged = _split_long_dialogues(merged, limit)
    merged = _polish_fields(merged, title_fallback=title_fallback)
    merged["novel"] = ""  # 由 cmd 写入真实路径
    return merged


def cmd_storyboard(args: argparse.Namespace) -> None:
    settings = Settings.load(resolve(args.config) if args.config else None)
    if args.max_shots:
        settings.max_shots = args.max_shots
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    story_path = out / "storyboard.json"
    novel = resolve(args.novel) if args.novel else out / "novel.txt"
    if not novel.exists():
        raise SystemExit(f"找不到小说：{novel}")

    # 已有分镜且未 --force：跳过（与 images/audio 幂等一致）
    if story_path.exists() and not args.force:
        print(f"已有分镜，跳过生成：{story_path}（要重写请加 --force）")
        return

    novel_text = novel.read_text(encoding="utf-8")
    if not novel_text.strip():
        raise SystemExit(f"小说是空的：{novel}")

    print(f"小说 → 分镜：{novel} → {story_path}", flush=True)
    print(f"模型：{settings.llm_model}", flush=True)
    data = generate_storyboard_dict(
        novel_text,
        settings,
        title_fallback=out.name or novel.stem,
    )
    data["novel"] = str(novel)
    dump_json(story_path, data)
    n_shots = len(data.get("shots") or [])
    n_chars = len(data.get("characters") or [])
    n_dlg = sum(1 for s in data.get("shots") or [] if str(s.get("dialogue") or "").strip())
    print(f"已写分镜：{story_path}，{n_shots} 镜，{n_chars} 个角色，对白 {n_dlg} 条。")
    print("下一步会由 screenplay 做规范化与硬校验。")


__all__ = [
    "DEFAULT_LLM_MODEL",
    "cmd_storyboard",
    "generate_storyboard_dict",
]
