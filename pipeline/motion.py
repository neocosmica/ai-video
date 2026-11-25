"""镜头运动档位与首尾帧约定（16G：仍用 Klein Edit + LTX distilled）。

- subtle：对白近景，仅首帧强锁
- normal：空镜/轻运镜，仅首帧，锁稍松
- dynamic：有明确动作；若有尾帧文件则首尾引导，否则只松首帧锁
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MOTION_LEVELS = ("subtle", "normal", "dynamic")

# motion_prompt / image_prompt 里出现这些 → 倾向 dynamic，并值得做尾帧
_DYNAMIC_RE = re.compile(
    r"\b("
    r"walk|walking|run|running|turn|turns|turning|swing|swings|swinging|"
    r"raise|raises|raising|wave|waves|waving|point|points|pointing|"
    r"open|opens|opening|close|closes|closing|push(?!\s*in)|pull(?!\s*back)|"
    r"throw|throws|kick|kicks|nod|nods|shake|shakes|sit|sits|stand|stands|"
    r"reach|reaches|grab|grabs|pick|picks|put|puts|hand\s+over|"
    r"pan\b|tilt\b|dolly|track|orbit|crane|whip"
    r")\b|"
    r"(挥|走|跑|转|推门|开门|关门|举手|挥手|点头|摇头|坐下|站起|扔|踢|拿起|放下|递|"
    r"扫过|摇移|俯仰|跟拍)",
    re.I,
)

_SUBTLE_MOTION_RE = re.compile(
    r"locked-?off|lip\s*motion|subtle\s+(natural\s+)?(movement|expression)|"
    r"微动|嘴型|口型|细微",
    re.I,
)


def end_frame_filename(shot_id: str) -> str:
    return f"{shot_id}_end.png"


def frame_stem(shot: dict) -> str:
    """静帧文件 id：子镜头复用父镜时用 reuse_frame。"""
    reuse = str(shot.get("reuse_frame") or "").strip()
    if reuse:
        return reuse
    return str(shot.get("id") or "").strip()


def start_frame_path(frames_dir: Path, shot: dict) -> Path:
    return frames_dir / f"{frame_stem(shot)}.png"


def end_frame_path(frames_dir: Path, shot: dict) -> Path:
    return frames_dir / end_frame_filename(frame_stem(shot))


def looks_dynamic(text: str) -> bool:
    return bool(_DYNAMIC_RE.search(text or ""))


def normalize_motion_level(raw: str | None) -> str | None:
    level = str(raw or "").strip().lower()
    if level in MOTION_LEVELS:
        return level
    aliases = {
        "low": "subtle",
        "soft": "subtle",
        "micro": "subtle",
        "medium": "normal",
        "mid": "normal",
        "high": "dynamic",
        "action": "dynamic",
        "strong": "dynamic",
    }
    return aliases.get(level)


def infer_motion_level(shot: dict) -> str:
    explicit = normalize_motion_level(shot.get("motion_level"))
    if explicit:
        return explicit
    end_prompt = str(shot.get("end_image_prompt") or "").strip()
    if end_prompt:
        return "dynamic"
    dialogue = str(shot.get("dialogue") or "").strip()
    motion = str(shot.get("motion_prompt") or "")
    image = str(shot.get("image_prompt") or "")
    blob = f"{motion} {image}"
    if looks_dynamic(blob):
        return "dynamic"
    if dialogue and _SUBTLE_MOTION_RE.search(motion):
        return "subtle"
    if dialogue:
        return "subtle"
    return "normal"


def wants_end_frame(shot: dict) -> bool:
    """是否应生成/使用尾帧（真尾帧，不是把首帧再贴一次）。

    只认运动档位 dynamic；normal/subtle 即便写了 end_image_prompt 也不做尾帧。
    """
    if str(shot.get("reuse_frame") or "").strip():
        # 子镜头复用父镜文件；尾帧也跟父 id，不单独生成
        return False
    return infer_motion_level(shot) == "dynamic"


def fallback_end_image_prompt(shot: dict) -> str:
    """未手写 end_image_prompt 时，从画面/动作里挤一句终点姿势。"""
    explicit = str(shot.get("end_image_prompt") or "").strip()
    if explicit:
        return explicit
    image = str(shot.get("image_prompt") or "").strip()
    motion = str(shot.get("motion_prompt") or "").strip()
    if motion:
        return (
            f"同一机位、景别与光线，人物已完成该动作后的瞬间："
            f"{motion}。手臂/躯干/道具位置必须与动作开始时明显不同"
        )
    if image and looks_dynamic(image):
        return (
            f"同一机位与光线，人物完成动作后的瞬间（姿势已变，不是动作开始态）：{image}"
        )
    return (
        "同一机位与光线，人物完成一个清晰可见的动作后的姿势；"
        "手臂或躯干相对动作开始时有明显位移，面部与服装不变"
    )


def end_edit_prompt(shot: dict, story: dict, cast: list[dict], *, stronger: bool = False) -> str:
    """Klein Edit：以首帧为参考改出终点姿势。stronger=近复制重试时加强姿势差。"""
    end_desc = fallback_end_image_prompt(shot)
    style = story.get("style") or ""
    who = "；".join(f"{c['name']}：{c['appearance']}" for c in cast)
    parts = [
        "第1张参考图是本镜首帧。",
        "保持同一机位、景别、背景、光线、色调、脸、发型与服装。",
        "必须改出终点姿势：手臂、躯干或道具相对参考图要有肉眼可辨的位移；禁止几乎原样复制参考姿势。",
        f"终点画面：{end_desc}。",
    ]
    if stronger:
        parts.append(
            "上一版几乎没动姿势，这次加大动作幅度："
            "门/手/身体至少有一项明显挪位，仍留在同一取景里，不要切到室内全景。"
        )
    if cast:
        parts.append(f"人物外观必须保持：{who}。")
    if style:
        parts.append(f"风格：{style}。")
    parts.append("不要文字，不要字幕，不要水印。")
    return "".join(parts)


def frame_mae(a: Path, b: Path) -> float:
    """两张 RGB 图的平均绝对差（0–255）。用于判断尾帧是否近复制首帧。"""
    from PIL import Image, ImageChops, ImageStat

    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB")
    if ia.size != ib.size:
        ib = ib.resize(ia.size, Image.Resampling.LANCZOS)
    st = ImageStat.Stat(ImageChops.difference(ia, ib))
    return float(sum(st.mean) / 3.0)


# Klein Edit 4 步时常只改雨丝/噪点；低于此值视为姿势没拉开
END_FRAME_MIN_MAE = 22.0


@dataclass(frozen=True)
class VideoConditioning:
    motion_level: str
    start_strength: float
    start_binding: str
    end_strength: float | None
    end_binding: str | None
    denoise_plan: tuple[int, int]
    use_end_frame: bool


def video_conditioning(shot: dict, *, has_end_frame: bool) -> VideoConditioning:
    level = infer_motion_level(shot)
    if level == "subtle":
        return VideoConditioning(
            motion_level=level,
            start_strength=1.0,
            start_binding="replace",
            end_strength=None,
            end_binding=None,
            denoise_plan=(8, 3),
            use_end_frame=False,
        )
    if level == "normal":
        # normal 若已有真尾帧，轻用尾锚；否则只松首帧
        if has_end_frame:
            return VideoConditioning(
                motion_level=level,
                start_strength=0.85,
                start_binding="replace",
                end_strength=0.5,
                end_binding="guide",
                denoise_plan=(8, 3),
                use_end_frame=True,
            )
        return VideoConditioning(
            motion_level=level,
            start_strength=0.85,
            start_binding="replace",
            end_strength=None,
            end_binding=None,
            denoise_plan=(8, 3),
            use_end_frame=False,
        )
    # dynamic
    if has_end_frame:
        return VideoConditioning(
            motion_level=level,
            start_strength=0.7,
            start_binding="guide",
            end_strength=0.45,
            end_binding="guide",
            denoise_plan=(12, 4),
            use_end_frame=True,
        )
    return VideoConditioning(
        motion_level=level,
        start_strength=0.7,
        start_binding="guide",
        end_strength=None,
        end_binding=None,
        denoise_plan=(12, 4),
        use_end_frame=False,
    )


def video_prompt(shot: dict, *, with_speech: bool, cond: VideoConditioning) -> str:
    motion = shot.get("motion_prompt") or "Slow cinematic push in"
    if with_speech:
        identity = (
            "Keep the face, hair and clothes consistent with the first frame. "
            "Natural lip sync to the dialogue audio, no morphing."
        )
        if cond.motion_level == "subtle":
            return f"{motion}. The on-screen speaker talks naturally. {identity} No scene change."
        return (
            f"{motion}. The on-screen speaker talks naturally with clear lip sync. "
            f"{identity}"
        )
    if cond.use_end_frame:
        return (
            f"{motion}. Animate continuously from the first-frame pose toward the last-frame pose. "
            "Keep face, hair and clothes consistent. Natural body motion, no morphing, no cut. "
            "Ambient room tone only, no spoken words."
        )
    if cond.motion_level == "dynamic":
        return (
            f"{motion}. Clear visible body and camera motion while keeping face, hair and clothes "
            "consistent with the first frame. No morphing. Ambient room tone only, no spoken words."
        )
    return (
        f"{motion}. Keep the face, hair and clothes identical to the first frame. "
        "Stable identity, natural motion, no morphing, no scene change. "
        "Ambient room tone only, no spoken words."
    )


def selftest_motion() -> None:
    subtle = {"dialogue": "你好", "motion_prompt": "Locked-off camera, natural lip motion", "image_prompt": "近景"}
    assert infer_motion_level(subtle) == "subtle"
    assert not wants_end_frame(subtle)

    dyn = {
        "id": "s003",
        "dialogue": "",
        "motion_prompt": "He swings the driver through impact",
        "image_prompt": "中景，挥杆",
    }
    assert infer_motion_level(dyn) == "dynamic"
    assert wants_end_frame(dyn)

    tagged = {"motion_level": "normal", "end_image_prompt": "手已举起", "dialogue": ""}
    assert infer_motion_level(tagged) == "normal"
    assert not wants_end_frame(tagged)
    assert wants_end_frame({"motion_level": "dynamic", "end_image_prompt": "手已举起"})

    child = {**dyn, "id": "s003_p1", "reuse_frame": "s003"}
    assert not wants_end_frame(child)
    assert end_frame_filename("s003") == "s003_end.png"

    cond = video_conditioning(dyn, has_end_frame=True)
    assert cond.use_end_frame and cond.start_binding == "guide" and cond.denoise_plan == (12, 4)
    cond2 = video_conditioning(subtle, has_end_frame=False)
    assert cond2.start_strength == 1.0 and not cond2.use_end_frame
