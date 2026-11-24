"""产物单跑依赖：直接边 + 前向/后向传递闭包。

单跑只认数据依赖，不认 produce 的固定串跑顺序。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

from pipeline.projects import artifacts_sub
from pipeline.util import load_json, resolve

KINDS = ("storyboard", "portrait", "frame", "voice", "clip", "music", "final")
KIND_LABELS = {
    "storyboard": "分镜",
    "portrait": "肖像",
    "frame": "静帧",
    "voice": "配音",
    "clip": "视频",
    "music": "配乐",
    "final": "成片",
}
# 拓扑序：祖先靠前
KIND_ORDER = {k: i for i, k in enumerate(KINDS)}


def _node_key(kind: str, nid: str = "") -> str:
    return f"{kind}:{nid or ''}"


def _parse_key(key: str) -> tuple[str, str]:
    kind, _, nid = key.partition(":")
    return kind, nid


def _load_story(out: str | Path, *, optional: bool = False) -> dict:
    path = resolve(out) / "storyboard.json"
    if not path.is_file():
        if optional:
            return {}
        raise FileNotFoundError(f"还没有分镜：{path}")
    return load_json(path)


def _novel_ready(out: str | Path) -> bool:
    """项目 novel.txt 或网页暂存 run_novel.txt 有正文即可。"""
    base = resolve(out)
    for path in (base / "novel.txt", artifacts_sub(base, "webui", "run_novel.txt")):
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return True
    return False


def _shot_map(story: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for s in story.get("shots") or []:
        sid = str(s.get("id") or "").strip()
        if sid:
            out[sid] = s
    return out


def _char_map(story: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in story.get("characters") or []:
        cid = str(c.get("id") or "").strip()
        if cid:
            out[cid] = c
    return out


def _has_dialogue(shot: dict | None) -> bool:
    return bool(str((shot or {}).get("dialogue") or "").strip())


def _cast_ids(shot: dict | None) -> list[str]:
    return [str(x).strip() for x in ((shot or {}).get("characters") or []) if str(x).strip()]


def artifact_path(out: str | Path, kind: str, nid: str = "") -> Path:
    base = resolve(out)
    if kind == "storyboard":
        return base / "storyboard.json"
    if kind == "portrait":
        return artifacts_sub(base, "characters", f"{nid}.png")
    if kind == "frame":
        return artifacts_sub(base, "frames", f"{nid}.png")
    if kind == "voice":
        return artifacts_sub(base, "audio", "voice", f"{nid}.wav")
    if kind == "clip":
        return artifacts_sub(base, "clips", f"{nid}.mp4")
    if kind == "music":
        return artifacts_sub(base, "audio", "bgm.wav")
    if kind == "final":
        return base / "final.mp4"
    raise ValueError(f"未知产物类型：{kind}")


def node_label(story: dict, kind: str, nid: str = "") -> str:
    prefix = KIND_LABELS.get(kind, kind)
    if kind == "storyboard":
        return "从小说生成分镜"
    if kind == "portrait":
        ch = _char_map(story).get(nid) or {}
        name = str(ch.get("name") or nid).strip() or nid
        return f"{prefix} {name}"
    if kind in {"frame", "voice", "clip"}:
        return f"{prefix} {nid}"
    return prefix


def _exists(out: str | Path, kind: str, nid: str = "") -> bool:
    path = artifact_path(out, kind, nid)
    return path.is_file() and path.stat().st_size > 0


def direct_pre(story: dict, out: str | Path, kind: str, nid: str = "") -> list[tuple[str, str]]:
    """一层前置。只在最前插入分镜；其余边与原先一致。"""
    shots = _shot_map(story)
    if kind == "storyboard":
        return []
    # 其余产物都先依赖 storyboard.json（缺则生成 / 已有则跳过）
    sb: list[tuple[str, str]] = [("storyboard", "")]
    if kind == "portrait":
        return sb
    if kind == "frame":
        shot = shots.get(nid)
        if not shot:
            return sb
        return sb + [("portrait", cid) for cid in _cast_ids(shot)]
    if kind == "voice":
        return sb
    if kind == "clip":
        shot = shots.get(nid)
        pre: list[tuple[str, str]] = sb + [("frame", nid)]
        if _has_dialogue(shot):
            pre.append(("voice", nid))
        return pre
    if kind == "music":
        # 配乐只认计划片长：对白镜要有 voice（写入 video_frames）；空镜用默认帧，无产物前置
        return sb + [("voice", sid) for sid, shot in shots.items() if _has_dialogue(shot)]
    if kind == "final":
        # 有 music_prompt 就必须先有配乐；以前只在 bgm 已存在时才挂边，缺文件时对话框会漏掉 BGM
        pre = sb + [("clip", sid) for sid in shots]
        if str(story.get("music_prompt") or "").strip():
            pre.append(("music", ""))
        return pre
    raise ValueError(f"未知产物类型：{kind}")


def direct_post(story: dict, kind: str, nid: str = "") -> list[tuple[str, str]]:
    """一层后继。"""
    shots = _shot_map(story)
    if kind == "storyboard":
        # 重写分镜会使现有角色图 / 镜头产物过时；对话框里这些后置默认不勾
        posts: list[tuple[str, str]] = []
        for cid in _char_map(story):
            posts.append(("portrait", cid))
        for sid, shot in shots.items():
            posts.append(("frame", sid))
            if _has_dialogue(shot):
                posts.append(("voice", sid))
            posts.append(("clip", sid))
        if shots or str(story.get("music_prompt") or "").strip():
            posts.append(("music", ""))
        if shots:
            posts.append(("final", ""))
        return posts
    if kind == "portrait":
        return [
            ("frame", sid)
            for sid, shot in shots.items()
            if nid in _cast_ids(shot)
        ]
    if kind == "frame":
        # 换静帧不影响计划片长 → 不牵动 music
        return [("clip", nid)]
    if kind == "voice":
        shot = shots.get(nid)
        if not _has_dialogue(shot):
            return []
        # 配音变 → 视频要重做；video_frames 变 → 配乐建议重做
        return [("clip", nid), ("music", "")]
    if kind == "clip":
        # 服从既有 video_frames，改 clip 内容不强制重配乐
        return [("final", "")]
    if kind == "music":
        return [("final", "")]
    if kind == "final":
        return []
    raise ValueError(f"未知产物类型：{kind}")


def _reachable(
    story: dict,
    out: str | Path,
    start: tuple[str, str],
    *,
    forward: bool,
) -> list[tuple[str, str]]:
    """从 start 出发的传递闭包（不含 start）。forward=False 走前置边。"""
    seen: set[str] = set()
    order: list[tuple[str, str]] = []
    q: deque[tuple[str, str]] = deque()
    sk, sid = start
    for nxt in (direct_post(story, sk, sid) if forward else direct_pre(story, out, sk, sid)):
        q.append(nxt)
    while q:
        kind, nid = q.popleft()
        key = _node_key(kind, nid)
        if key in seen:
            continue
        seen.add(key)
        order.append((kind, nid))
        nbrs = direct_post(story, kind, nid) if forward else direct_pre(story, out, kind, nid)
        for nxt in nbrs:
            nk = _node_key(*nxt)
            if nk not in seen:
                q.append(nxt)
    return order


def _topo_sort(nodes: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(nodes, key=lambda n: (KIND_ORDER.get(n[0], 99), n[1] or ""))


def validate_target(story: dict, kind: str, nid: str = "", *, out: str | Path | None = None) -> str | None:
    """非法目标返回错误文案，合法返回 None。"""
    if kind not in KINDS:
        return f"未知产物类型：{kind}"
    if kind == "storyboard":
        if out is not None and not _novel_ready(out):
            return "还没有小说正文：请先在上方填写并保存，或确保 ~/AI短视频项目/<片>/novel.txt 有内容"
        return None
    shots = _shot_map(story)
    chars = _char_map(story)
    if kind == "portrait":
        if not nid or nid not in chars:
            return f"分镜里没有角色 id「{nid}」"
        return None
    if kind in {"frame", "voice", "clip"}:
        if not nid or nid not in shots:
            return f"分镜里没有镜头「{nid}」"
        if kind == "voice" and not _has_dialogue(shots.get(nid)):
            return f"镜头 {nid} 无对白，不能重跑配音"
        return None
    if kind in {"music", "final"}:
        return None
    return f"未知产物类型：{kind}"


def plan_rerun(out: str | Path, *, kind: str, id: str = "") -> dict:
    """计算单跑计划：前向闭包 + 目标 + 后向闭包。"""
    kind = str(kind or "").strip()
    nid = str(id or "").strip()
    # optional：缺 storyboard.json 时仍可规划，前置里出现「从小说生成分镜」
    story = _load_story(out, optional=True)
    err = validate_target(story, kind, nid, out=out)
    if err:
        return {
            "ok": False,
            "error": err,
            "prereqs": [],
            "target": None,
            "posts": [],
        }

    target = {
        "kind": kind,
        "id": nid,
        "label": node_label(story, kind, nid),
        "path": str(artifact_path(out, kind, nid).relative_to(resolve(out)))
        if kind != "final"
        else "final.mp4",
    }

    prereq_nodes = _topo_sort(_reachable(story, out, (kind, nid), forward=False))
    post_nodes = _topo_sort(_reachable(story, out, (kind, nid), forward=True))

    prereqs = []
    for pk, pid in prereq_nodes:
        ready = _exists(out, pk, pid)
        prereqs.append(
            {
                "kind": pk,
                "id": pid,
                "label": node_label(story, pk, pid),
                "status": "ready" if ready else "missing",
                "path": str(artifact_path(out, pk, pid)),
            }
        )

    # 从小说生成分镜会牵出整条后置链，默认不勾；其他目标的后置默认全勾
    suggest_posts = kind != "storyboard"
    posts = []
    for pk, pid in post_nodes:
        posts.append(
            {
                "kind": pk,
                "id": pid,
                "label": node_label(story, pk, pid),
                "suggested": suggest_posts,
                "path": str(artifact_path(out, pk, pid)),
            }
        )

    return {
        "ok": True,
        "error": None,
        "prereqs": prereqs,
        "target": target,
        "posts": posts,
        "summary": _summary(target, prereqs, posts),
    }


def _summary(target: dict, prereqs: list[dict], posts: list[dict]) -> str:
    miss = [p["label"] for p in prereqs if p["status"] == "missing"]
    parts = [f"目标：{target['label']}（将重做）"]
    if miss:
        parts.append("缺失前置将生成：" + "、".join(miss))
    ready = [p["label"] for p in prereqs if p["status"] == "ready"]
    if ready:
        parts.append("已有前置将跳过：" + "、".join(ready))
    if posts:
        chosen = [p["label"] for p in posts if p.get("suggested")]
        if len(chosen) == len(posts):
            parts.append("后置默认全选（可取消）：" + "、".join(chosen))
        elif not chosen:
            parts.append("后置默认全不选（可勾选）：" + "、".join(p["label"] for p in posts))
        else:
            parts.append("后置默认勾选：" + "、".join(chosen))
    return "；".join(parts)


def clear_nodes(out: str | Path, nodes: Iterable[dict | tuple[str, str]]) -> list[str]:
    """删除指定产物文件，返回相对路径列表。"""
    base = resolve(out)
    removed: list[str] = []
    for node in nodes:
        if isinstance(node, dict):
            kind, nid = node.get("kind", ""), node.get("id", "") or ""
        else:
            kind, nid = node
        path = artifact_path(base, kind, nid)
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(base))
        except ValueError:
            rel = str(path)
        path.unlink()
        removed.append(rel)
        # 配音连带 a2v 中间件
        if kind == "voice" and nid:
            for extra in (
                artifacts_sub(base, "audio", "a2v", f"{nid}.wav"),
                artifacts_sub(base, "audio", "a2v", f"{nid}.mp4"),
            ):
                if extra.is_file():
                    try:
                        removed.append(str(extra.relative_to(base)))
                    except ValueError:
                        removed.append(str(extra))
                    extra.unlink()
    return removed


def execution_groups(
    *,
    prereqs: list[dict],
    target: dict,
    posts: list[dict],
) -> list[dict]:
    """把要跑的节点收成 CLI 阶段组（按阶段串跑）。

    前置：仅 status=missing；目标与勾选后置：一律 force。
    """
    run_nodes: list[dict] = []
    for p in prereqs:
        if p.get("status") == "missing":
            run_nodes.append({**p, "force": False})
    run_nodes.append({**target, "force": True})
    for p in posts:
        run_nodes.append({**p, "force": True})

    # 按 kind 聚合
    portraits: list[str] = []
    frames: list[str] = []
    voices: list[str] = []
    clips: list[str] = []
    want_storyboard = False
    want_music = False
    want_final = False
    force_portraits = False
    force_frames = False
    force_voices = False
    force_clips = False

    for n in run_nodes:
        kind = n["kind"]
        nid = n.get("id") or ""
        force = bool(n.get("force"))
        if kind == "storyboard":
            want_storyboard = True
        elif kind == "portrait":
            portraits.append(nid)
            force_portraits = force_portraits or force
        elif kind == "frame":
            frames.append(nid)
            force_frames = force_frames or force
        elif kind == "voice":
            voices.append(nid)
            force_voices = force_voices or force
        elif kind == "clip":
            clips.append(nid)
            force_clips = force_clips or force
        elif kind == "music":
            want_music = True
        elif kind == "final":
            want_final = True

    def uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    groups: list[dict] = []
    portraits = uniq(portraits)
    frames = uniq(frames)
    voices = uniq(voices)
    clips = uniq(clips)

    if want_storyboard:
        groups.append({"stage": "storyboard", "force": True})
        groups.append({"stage": "screenplay", "force": True})
    if portraits or frames:
        groups.append(
            {
                "stage": "images",
                "characters": portraits,
                "only": ",".join(frames) if frames else None,
                "force": force_portraits or force_frames,
                "force_characters": force_portraits,
                "force_frames": force_frames,
            }
        )
    if voices:
        groups.append(
            {
                "stage": "audio",
                "only": ",".join(voices),
                "force": force_voices,
            }
        )
    if clips:
        groups.append(
            {
                "stage": "videos",
                "only": ",".join(clips),
                "force": force_clips,
            }
        )
    if want_music:
        groups.append({"stage": "music", "force": True})
    if want_final:
        groups.append({"stage": "assemble", "force": True})
    return groups
