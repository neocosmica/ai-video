from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """相对路径相对运行根；``projects/<名>/…`` 映射到用户项目根；绝对路径原样。"""
    if isinstance(path, Path):
        if path.is_absolute():
            return path
        text = path.as_posix()
    else:
        text = str(path or "").strip().replace("\\", "/")
        if not text:
            return ROOT
        p = Path(text)
        if p.is_absolute():
            return p
    # 旧相对前缀 projects/<片名>/… → ~/AI短视频项目/<片名>/…
    if text == "projects" or text.startswith("projects/"):
        from pipeline.projects import projects_dir

        rest = "" if text == "projects" else text[len("projects/") :]
        base = projects_dir()
        return (base / rest).resolve() if rest else base
    return (ROOT / text).resolve() if text else ROOT


def cite_path(path: str | Path) -> str:
    """交给 ``resolve()`` 的路径字符串。

    仓库里的文件用相对运行根的路径。打包进 ``AI短视频.app`` 之后，
    项目在 ``~/AI短视频项目``，不在 ``Contents/Resources/runtime`` 下面，
    ``Path.relative_to(ROOT)`` 会抛 ``ValueError: ... is not in the subpath of ...``。
    这时返回绝对路径，``resolve()`` 会原样打开。
    """
    p = path if isinstance(path, Path) else Path(str(path))
    try:
        p = p.resolve()
    except OSError:
        p = Path(str(p))
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def allowed_data_roots() -> list[Path]:
    """safe_path 允许落在其下的根：运行根 + 用户项目根 + 旧仓库 projects/。"""
    roots = [ROOT.resolve()]
    try:
        from pipeline.projects import legacy_repo_projects_dir, projects_dir

        roots.append(projects_dir().resolve())
        legacy = legacy_repo_projects_dir()
        if legacy.exists():
            roots.append(legacy.resolve())
    except Exception:
        pass
    # Application Support 日志等
    support = Path.home() / "Library" / "Application Support" / "AI短视频"
    roots.append(support.resolve())
    # 去重保序
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def under_allowed_root(path: Path) -> bool:
    resolved = path.resolve()
    for root in allowed_data_roots():
        if resolved == root or root in resolved.parents:
            return True
    return False


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


def frames_for_duration(
    duration_s: float,
    fps: int,
    *,
    min_frames: int = 25,
    max_frames: int = 121,
) -> int:
    """按对白秒数算 LTX 帧数（8k+1），夹在 min/max 之间。

    取格必须是「向上」，否则取整后会比音频短几十毫秒，台词尾巴还是会被切掉。
    """
    fps = max(1, int(fps))
    lo = snap_frames(min_frames)
    hi = snap_frames(max_frames)
    if lo > hi:
        lo, hi = hi, lo
    raw = max(lo, int(math.ceil(float(duration_s) * fps)))
    if raw >= hi:
        return hi
    # 向上对齐到 8k+1 网格
    frames = 8 * int(math.ceil((raw - 1) / 8)) + 1
    return max(lo, min(hi, frames))


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
