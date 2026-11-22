from __future__ import annotations

import re

# Qwen3-TTS CustomVoice 中文预设。按性别/气质挑，情绪另用 instruct。
FEMALE_VOICES = {
    "vivian": "Vivian",  # 偏亮、年轻
    "serena": "Serena",  # 温柔
}
MALE_VOICES = {
    "uncle_fu": "Uncle_Fu",  # 沉稳
    "dylan": "Dylan",  # 北京口音，偏日常男声
    "eric": "Eric",  # 四川口音，偏松弛
}

EMOTION_HINTS = {
    "calm": "用平静自然的语气说",
    "warm": "用温柔亲近的语气说",
    "sad": "用低沉略带伤感的语气说",
    "tense": "用紧张克制的语气说",
    "angry": "用压抑而生气的语气说",
    "happy": "用轻快愉快的语气说",
    "surprised": "用惊讶的语气说",
    "whisper": "用接近耳语、很轻的语气说",
    "bitter": "用苦涩、欲言又止的语气说",
    "nostalgic": "用怀念、轻声的语气说",
}


def infer_gender(appearance: str, explicit: str | None = None) -> str:
    text = f"{explicit or ''} {appearance or ''}".lower()
    if any(token in text for token in ("女", "female", "woman", "girl", "她")):
        return "female"
    if any(token in text for token in ("男", "male", "man", "boy", "他")):
        return "male"
    return "female"


def pick_voice(gender: str, voice: str | None = None, appearance: str = "") -> str:
    if voice:
        key = re.sub(r"[^a-z_]", "", voice.lower().replace("-", "_"))
        for table in (FEMALE_VOICES, MALE_VOICES):
            if key in table:
                return table[key]
            for name in table.values():
                if key == name.lower().replace("-", "_"):
                    return name
    look = appearance or ""
    if gender == "male":
        if any(word in look for word in ("温和", "温柔", "年轻", "少年")):
            return "Dylan"
        if any(word in look for word in ("年长", "沉稳", "大叔", "经理")):
            return "Uncle_Fu"
        return "Dylan"
    if any(word in look for word in ("温柔", "柔", "安静", "文静")):
        return "Serena"
    return "Vivian"


def emotion_instruct(emotion: str | None, fallback: str = "用自然对白的语气说") -> str:
    if not emotion:
        return fallback
    key = emotion.strip().lower()
    if key in EMOTION_HINTS:
        return EMOTION_HINTS[key]
    # 允许 Agent 直接写中文情绪指令
    if re.search(r"[\u4e00-\u9fff]", emotion):
        if "语气" in emotion:
            return emotion
        return f"用{emotion}的语气说"
    return f"Speak in a {emotion} tone."


def default_music_prompt(style: str, title: str) -> str:
    base = style or title or "cinematic short drama"
    return (
        f"Instrumental underscore for a Chinese vertical short drama titled {title or 'untitled'}. "
        f"Mood matches: {base}. Soft piano and warm ambient pads, rainy night, melancholic reunion, "
        "no vocals, no lyrics, cinematic, loopable, gentle dynamics."
    )
