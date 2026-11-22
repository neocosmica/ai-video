from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def snap(n: int, base: int) -> int:
    return max(base, (int(n) // base) * base)


def snap_frames(n: int) -> int:
    k = max(1, round((int(n) - 1) / 8))
    return 8 * k + 1


def has_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text or "") is not None


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<think>", "").replace("</think>", "").strip()


def extract_json(text: str) -> dict:
    cleaned = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON")
    blob = cleaned[start : end + 1]
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    return data


CHAPTER_RE = re.compile(
    r"(?m)^[ \t]*((?:第[0-9零〇一二三四五六七八九十百千两]+[章回节卷部].*)|(?:Chapter\s+\d+.*))$"
)


def split_len(text: str, limit: int) -> list[str]:
    paras = re.split(r"\n\s*\n", text.strip())
    out: list[str] = []
    buf = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 2 <= limit:
            buf = f"{buf}\n\n{para}"
            continue
        if buf:
            out.append(buf)
        if len(para) <= limit:
            buf = para
            continue
        for i in range(0, len(para), limit):
            out.append(para[i : i + limit])
        buf = ""
    if buf:
        out.append(buf)
    return out or [text.strip()]


def chunk_novel(text: str, limit: int = 1800) -> list[str]:
    text = text.strip()
    if not text:
        raise ValueError("小说是空的")
    if len(text) <= limit:
        return [text]
    parts = CHAPTER_RE.split(text)
    if len(parts) <= 2:
        return split_len(text, limit)
    pieces: list[str] = []
    pre = parts[0].strip()
    if pre:
        pieces.append(pre)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        pieces.append(f"{title}\n{body}".strip())
    out: list[str] = []
    for piece in pieces:
        out.extend(split_len(piece, limit) if len(piece) > limit else [piece])
    return out


def ffmpeg_concat_line(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'"


def ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    cs = int(round(seconds * 100))
    s, cs = divmod(cs, 100)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
