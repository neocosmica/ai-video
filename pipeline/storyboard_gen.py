"""用本机 MLX 聊天模型把小说写成 storyboard.json。

默认模型：mlx-community/Qwen3-4B-Instruct-2507-4bit（约 3.3 GB，16 GB 统一内存可跑）。
生成后交给 screenplay 做规范化与对白覆盖校验。
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

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
from pipeline.voices import default_music_prompt, infer_gender, pick_voice

DEFAULT_LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_SYSTEM = """你是华语竖屏短剧分镜编剧。只输出一个 JSON 对象，不要 markdown，不要解释。
字段与约束严格按用户提示；台词必须照小说引号原文，禁止删字、改写、压短。"""


def _spoken_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE)


def _build_user_prompt(
    novel: str,
    *,
    max_shots: int,
    dialogue_limit: int,
    max_seconds: float,
    quoted_lines: list[str],
    chunk_index: int = 0,
    chunk_total: int = 1,
) -> str:
    chunk_note = ""
    if chunk_total > 1:
        chunk_note = f"\n这是小说第 {chunk_index + 1}/{chunk_total} 段，只根据本段写镜头；角色可重复出现。\n"
    if quoted_lines:
        listed = "\n".join(f"- {line}" for line in quoted_lines)
        quote_block = (
            "本段必须原样覆盖的引号台词（dialogue 只能用这些字，可拆到相邻镜；"
            "禁止把旁白写进 dialogue）：\n"
            f"{listed}\n"
        )
    else:
        quote_block = "本段没有引号台词：所有镜头 dialogue 必须是 \"\"。\n"
    return f"""把下面小说改成竖屏短剧分镜 JSON。只输出 JSON。

硬约束：
- 最多 {max_shots} 个镜头
- 单镜对白约 {dialogue_limit} 字 / {max_seconds:.1f} 秒；超长按标点拆镜，台词不许删字
- dialogue 只能是引号内台词；叙述镜头 dialogue 与 speaker 都必须 ""
- 禁止把旁白（如「雨下得很大」「她推开门」）写进 dialogue
- motion_prompt 与 music_prompt 必须英文；music_prompt 无人声
- gender：female/male；voice：女 Serena/Vivian，男 Dylan/Uncle_Fu/Eric
- emotion：calm/warm/sad/tense/bitter/nostalgic/surprised/whisper
- image_prompt 中文景别/地点/动作/光线；人脸清晰；有对白时嘴微张
- 外貌只取原文可见特征；每镜换景别或角度
{chunk_note}{quote_block}
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
      "appearance": "二十多岁女性，短发，深色风衣"
    }}
  ],
  "shots": [
    {{
      "id": "s001",
      "characters": ["linwan"],
      "location": "地点",
      "image_prompt": "中景，……",
      "motion_prompt": "Locked-off camera, subtle natural movement",
      "speaker": "",
      "dialogue": "",
      "emotion": "calm"
    }},
    {{
      "id": "s002",
      "characters": ["linwan"],
      "location": "地点",
      "image_prompt": "中近景，嘴微张，面部清晰",
      "motion_prompt": "Locked-off camera, natural lip motion",
      "speaker": "林晚",
      "dialogue": "引号原句",
      "emotion": "surprised"
    }}
  ]
}}

小说：
<<<
{novel}
>>>
"""


def _load_mlx(model_id: str):
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise SystemExit(
            "缺少 mlx-lm。请重跑 ./scripts/setup.sh，或在主环境安装：uv sync"
        ) from exc

    print(f"加载分镜 LLM：{model_id}", flush=True)
    # 优先本机 hub 缓存。快照目录可能只有 config/tokenizer，没有 model*.safetensors；
    # 直接 load 会 FileNotFoundError: No safetensors found。缺了就补下再加载。
    from pipeline.hf_cache import ensure_snapshot

    path = ensure_snapshot(
        model_id,
        weight_globs=("model*.safetensors",),
        label="分镜语言模型",
    )
    try:
        model, tokenizer = load(str(path))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"分镜模型权重打不开：{exc}\n"
            f"可删掉该目录后重开，让它重新下载：{path}"
        ) from exc
    return model, tokenizer, generate, make_sampler


def _chat_once(
    model,
    tokenizer,
    generate,
    make_sampler,
    *,
    user: str,
    max_tokens: int,
    temp: float,
) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temp)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=True,
    )


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
        item = {
            "id": str(raw.get("id") or name).strip() or name,
            "name": name,
            "gender": gender,
            "voice": voice,
            "appearance": appearance,
        }
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
    model_id = (settings.llm_model or DEFAULT_LLM_MODEL).strip()
    fps = max(1, settings.video_fps)
    max_seconds = snap_frames(settings.video_frames_max) / fps
    limit = dialogue_char_limit(settings.video_frames_max, fps)
    chunks = chunk_novel(novel_text, limit=1600)
    # 多段时每段镜头配额均分，至少 4
    per_chunk_shots = max(4, settings.max_shots // max(1, len(chunks)))

    model, tokenizer, generate, make_sampler = _load_mlx(model_id)
    parts: list[dict] = []
    try:
        for index, chunk in enumerate(chunks):
            quoted = extract_novel_dialogues(chunk)
            user = _build_user_prompt(
                chunk,
                max_shots=min(settings.max_shots, per_chunk_shots if len(chunks) > 1 else settings.max_shots),
                dialogue_limit=limit,
                max_seconds=max_seconds,
                quoted_lines=quoted,
                chunk_index=index,
                chunk_total=len(chunks),
            )
            last_err: Exception | None = None
            data: dict | None = None
            for attempt in range(max(1, retries)):
                print(
                    f"生成分镜：段 {index + 1}/{len(chunks)}，尝试 {attempt + 1}/{retries}",
                    flush=True,
                )
                try:
                    raw_text = _chat_once(
                        model,
                        tokenizer,
                        generate,
                        make_sampler,
                        user=user,
                        max_tokens=settings.llm_max_tokens,
                        temp=settings.llm_temperature if attempt == 0 else 0.2,
                    )
                    data = extract_json(raw_text)
                    if not isinstance(data.get("shots"), list) or not data["shots"]:
                        raise ValueError("JSON 缺少 shots")
                    break
                except Exception as exc:  # noqa: BLE001 — 重试 JSON/生成失败
                    last_err = exc
                    print(f"  失败：{exc}", flush=True)
                    data = None
            if data is None:
                raise SystemExit(f"分镜生成失败：{last_err}")
            parts.append(data)
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

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
    from pipeline.backends import resolve_backend

    if resolve_backend(settings.backend).is_cuda:
        from pipeline.backends.cuda_storyboard import cmd_storyboard_cuda

        cmd_storyboard_cuda(settings, args)
        return
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
    print(f"模型：{settings.llm_model}（约 3–4 GB，适合 16 GB Mac）", flush=True)
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
