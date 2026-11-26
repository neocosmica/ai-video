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

# 角色配音模式：预设音色 / Base 参考音频克隆 / VoiceDesign 文字造声
VOICE_MODE_PRESET = "preset"
VOICE_MODE_CLONE = "clone"
VOICE_MODE_DESIGN = "design"
VOICE_MODES = (VOICE_MODE_PRESET, VOICE_MODE_CLONE, VOICE_MODE_DESIGN)

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


def normalize_voice_key(voice: str | None) -> str:
    return re.sub(r"[^a-z_]", "", (voice or "").lower().replace("-", "_"))


def voice_mode(char_or_voice) -> str:
    """从角色卡或 voice 字段推断配音模式。"""
    if isinstance(char_or_voice, dict):
        raw_mode = str(char_or_voice.get("voice_mode") or "").strip().lower()
        if raw_mode in VOICE_MODES:
            return raw_mode
        voice = char_or_voice.get("voice")
    else:
        voice = char_or_voice
    key = normalize_voice_key(str(voice or ""))
    if key == VOICE_MODE_CLONE:
        return VOICE_MODE_CLONE
    if key == VOICE_MODE_DESIGN:
        return VOICE_MODE_DESIGN
    return VOICE_MODE_PRESET


def pick_voice(gender: str, voice: str | None = None, appearance: str = "") -> str:
    if voice:
        key = normalize_voice_key(voice)
        if key in (VOICE_MODE_CLONE, VOICE_MODE_DESIGN):
            return key
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


def default_voice_design(gender: str, appearance: str = "", name: str = "") -> str:
    """VoiceDesign 用的英文音色描述；分镜没写时按外貌兜底。"""
    look = (appearance or name or "a Chinese drama character").strip()
    # 去掉过长外貌，避免 instruct 喧宾夺主
    look = re.sub(r"\s+", " ", look)
    if len(look) > 120:
        look = look[:117].rstrip() + "..."
    if gender == "male":
        return (
            f"A natural Mandarin male speaking voice matching this character: {look}. "
            "Clear diction, cinematic short-drama delivery, no singing."
        )
    return (
        f"A natural Mandarin female speaking voice matching this character: {look}. "
        "Clear diction, cinematic short-drama delivery, no singing."
    )


def design_instruct(voice_design: str, emotion_line: str) -> str:
    """VoiceDesign 的 instruct = 音色描述 + 本镜情绪。"""
    design = (voice_design or "").strip()
    mood = (emotion_line or "").strip()
    if design and mood:
        return f"{design} {mood}"
    return design or mood or "A natural Mandarin speaking voice."


def default_music_prompt(style: str, title: str) -> str:
    base = style or title or "cinematic short drama"
    return (
        f"Instrumental underscore for a Chinese vertical short drama titled {title or 'untitled'}. "
        f"Mood matches: {base}. Soft piano and warm ambient pads, rainy night, melancholic reunion, "
        "no vocals, no lyrics, cinematic, loopable, gentle dynamics."
    )


def companion_tts_id(selected_tts_id: str, mode: str) -> str:
    """按当前对白档位，解析 clone/design 要用的具体模型 id。"""
    mode = (mode or VOICE_MODE_PRESET).strip().lower()
    selected = (selected_tts_id or "").strip() or "qwen3-tts-1.7b-customvoice-8bit"
    if mode == VOICE_MODE_PRESET:
        return selected
    bit = "bf16" if "bf16" in selected.lower() else "8bit"
    if mode == VOICE_MODE_CLONE:
        return f"qwen3-tts-1.7b-base-{bit}"
    if mode == VOICE_MODE_DESIGN:
        return f"qwen3-tts-1.7b-voicedesign-{bit}"
    return selected
