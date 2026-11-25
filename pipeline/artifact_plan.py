"""产物单跑依赖：直接边 + 前向/后向传递闭包。

单跑只认数据依赖，不认 produce 的固定串跑顺序。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable

from pipeline.projects import artifacts_sub
from pipeline.util import resolve

KINDS = ("storyboard", "portrait", "frame", "end_frame", "voice", "clip", "music", "final")
KIND_LABELS = {
    "storyboard": "分镜",
    "portrait": "肖像",
    "frame": "静帧",
    "end_frame": "尾帧",
    "voice": "配音",
    "clip": "视频",
    "music": "配乐",
    "final": "成片",
}
# 拓扑序：祖先靠前
KIND_ORDER = {k: i for i, k in enumerate(KINDS)}

# 分镜阶段会整份重写镜头表。和静帧/配音/视频/配乐/成片放在同一次启动里，
# 后面的步骤仍按「旧镜头 id」排队，生成完对不上。
STORYBOARD_EXCLUSIVE = (
    "从小说生成分镜会重写镜头列表。请先只跑分镜，完成后再添加静帧、配音、视频、配乐或成片。"
)


def _node_key(kind: str, nid: str = "") -> str:
    return f"{kind}:{nid or ''}"


def _parse_key(key: str) -> tuple[str, str]:
    kind, _, nid = key.partition(":")
    return kind, nid


def _shape_error(data: object) -> str | None:
    """分镜顶层结构不对时返回给人看的原因。"""
    if not isinstance(data, dict):
        return "分镜文件损坏：顶层必须是 JSON 对象"
    if "shots" in data and data.get("shots") is not None and not isinstance(data.get("shots"), list):
        return "分镜格式不对：shots 必须是数组"
    if (
        "characters" in data
        and data.get("characters") is not None
        and not isinstance(data.get("characters"), list)
    ):
        return "分镜格式不对：characters 必须是数组"
    return None


def _load_story(out: str | Path, *, optional: bool = False) -> dict:
    path = resolve(out) / "storyboard.json"
    if not path.is_file():
        if optional:
            return {}
        raise FileNotFoundError(f"还没有分镜：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"分镜文件损坏，不是合法 JSON（{exc.msg}）") from exc
    err = _shape_error(data)
    if err:
        raise ValueError(err)
    return data


def _storyboard_on_disk(out: str | Path | None) -> bool:
    return out is not None and (resolve(out) / "storyboard.json").is_file()


def _novel_ready(out: str | Path) -> bool:
    """项目 novel.txt 或网页暂存正文有内容即可。兼容旧版固定文件名。"""
    base = resolve(out)
    webui = artifacts_sub(base, "webui")
    candidates = [base / "novel.txt", webui / "run_novel.txt"]
    for path in candidates:
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return True
    return False


def _shot_map(story: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    raw = story.get("shots") if isinstance(story, dict) else None
    if not isinstance(raw, list):
        return out
    for s in raw:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or "").strip()
        if sid:
            out[sid] = s
    return out


def _char_map(story: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    raw = story.get("characters") if isinstance(story, dict) else None
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if cid:
            out[cid] = c
    return out


def _has_dialogue(shot: dict | None) -> bool:
    return bool(str((shot or {}).get("dialogue") or "").strip())


def _cast_ids(shot: dict | None) -> list[str]:
    return [str(x).strip() for x in ((shot or {}).get("characters") or []) if str(x).strip()]


def _frame_owner_id(shot: dict | None, nid: str) -> str:
    """静帧/尾帧文件 id：子镜头复用父镜。"""
    return str((shot or {}).get("reuse_frame") or "").strip() or nid


def _shot_wants_end(story: dict, nid: str) -> bool:
    from pipeline.motion import wants_end_frame

    shots = _shot_map(story)
    shot = shots.get(nid)
    owner_id = _frame_owner_id(shot, nid)
    owner = shots.get(owner_id) or shot or {}
    return wants_end_frame(owner)


def artifact_path(out: str | Path, kind: str, nid: str = "") -> Path:
    base = resolve(out)
    if kind == "storyboard":
        return base / "storyboard.json"
    if kind == "portrait":
        return artifacts_sub(base, "characters", f"{nid}.png")
    if kind == "frame":
        return artifacts_sub(base, "frames", f"{nid}.png")
    if kind == "end_frame":
        return artifacts_sub(base, "frames", f"{nid}_end.png")
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
    if kind in {"frame", "end_frame", "voice", "clip"}:
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
    if kind == "end_frame":
        shot = shots.get(nid)
        if not shot:
            return sb
        # 尾帧只依赖本镜首帧（文件 id）；肖像已由首帧链路保证
        frame_id = _frame_owner_id(shot, nid)
        return sb + [("frame", frame_id)]
    if kind == "voice":
        return sb
    if kind == "clip":
        shot = shots.get(nid)
        # 自动拆出来的子镜头没有自己的静帧，依赖父镜头那张
        frame_id = _frame_owner_id(shot, nid)
        pre: list[tuple[str, str]] = sb + [("frame", frame_id)]
        if _shot_wants_end(story, nid):
            pre.append(("end_frame", frame_id))
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
            if str(shot.get("reuse_frame") or "").strip():
                # 子镜头没有独立静帧/尾帧文件
                if _has_dialogue(shot):
                    posts.append(("voice", sid))
                posts.append(("clip", sid))
                continue
            posts.append(("frame", sid))
            if _shot_wants_end(story, sid):
                posts.append(("end_frame", sid))
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
            if nid in _cast_ids(shot) and not str(shot.get("reuse_frame") or "").strip()
        ]
    if kind == "frame":
        # 换首帧 → 建议重做尾帧（若需要）和视频；不影响计划片长 → 不牵动 music
        posts = []
        if _shot_wants_end(story, nid):
            posts.append(("end_frame", nid))
        posts.append(("clip", nid))
        # 复用该静帧的子镜头视频也过时
        for sid, shot in shots.items():
            if str(shot.get("reuse_frame") or "").strip() == nid:
                posts.append(("clip", sid))
        return posts
    if kind == "end_frame":
        posts = [("clip", nid)]
        for sid, shot in shots.items():
            if str(shot.get("reuse_frame") or "").strip() == nid:
                posts.append(("clip", sid))
        return posts
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


def _clips_covered_for_final(
    out: str | Path,
    story: dict,
    *,
    target_kind: str,
    target_id: str,
    post_nodes: list[tuple[str, str]],
) -> set[str]:
    """本次目标 + 默认会跑的 clip 后置 + 磁盘已有视频 → 成片能覆盖到的镜头。"""
    shots = _shot_map(story)
    covered = {sid for sid in shots if _exists(out, "clip", sid)}
    if target_kind == "clip" and target_id:
        covered.add(target_id)
    for pk, pid in post_nodes:
        if pk == "clip" and pid:
            covered.add(pid)
    return covered


def final_post_suggestion(
    out: str | Path,
    story: dict,
    *,
    target_kind: str,
    target_id: str,
    post_nodes: list[tuple[str, str]],
) -> tuple[bool, str]:
    """单镜重跑时，成片是否该默认勾选。

    成片要齐全部镜头视频（有 music_prompt 时还要配乐）。缺别的镜时默认勾上，
    对话框只显示「视频 s003 + 成片」，看起来能跑、实际会失败或偷偷拉全片重渲。
    """
    shots = _shot_map(story)
    if not shots:
        return False, "分镜里还没有镜头"
    covered = _clips_covered_for_final(
        out,
        story,
        target_kind=target_kind,
        target_id=target_id,
        post_nodes=post_nodes,
    )
    missing = [sid for sid in shots if sid not in covered]
    if missing:
        shown = "、".join(missing[:6])
        more = f" 等 {len(missing)} 镜" if len(missing) > 6 else ""
        return False, f"成片需要全部镜头视频，还缺 {shown}{more}"
    if str(story.get("music_prompt") or "").strip():
        music_coming = (
            target_kind == "music"
            or any(pk == "music" for pk, _ in post_nodes)
            or _exists(out, "music")
        )
        if not music_coming:
            return False, "成片还缺配乐（分镜写了 music_prompt）"
    return True, ""


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


def _need_storyboard_first(kind: str, nid: str = "") -> str:
    """分镜文件还不存在时，镜头级产物无法靠「先生成分镜」补齐（镜头 id 未知）。"""
    label = KIND_LABELS.get(kind, kind)
    who = f"{label} {nid}".strip() if nid else label
    if kind in {"music", "final"}:
        return (
            f"还没有分镜，不能生成{label}。"
            "请先点「从小说生成分镜」，或点 ＋ 手写镜头。"
            "镜头确定之后再生成这一项，缺失的静帧、配音和视频会自动补上。"
        )
    return (
        f"还没有分镜，不能生成{who}。"
        "请先从小说生成分镜，或点 ＋ 手写镜头。"
    )


def _need_shots(kind: str) -> str:
    label = KIND_LABELS.get(kind, kind)
    return (
        f"分镜里还没有镜头，不能生成{label}。"
        "请先添加镜头，或从小说重新生成分镜。"
    )


def validate_target(story: dict, kind: str, nid: str = "", *, out: str | Path | None = None) -> str | None:
    """非法目标返回错误文案，合法返回 None。"""
    if kind not in KINDS:
        return f"未知产物类型：{kind}"
    if not isinstance(story, dict):
        return "分镜文件损坏：顶层必须是 JSON 对象"
    if kind == "storyboard":
        if out is not None and not _novel_ready(out):
            return "还没有小说正文。请先在上方填写小说，再生成分镜。"
        return None
    on_disk = _storyboard_on_disk(out)
    shots = _shot_map(story)
    chars = _char_map(story)
    if kind == "portrait":
        if not nid or nid not in chars:
            if out is not None and not on_disk:
                return _need_storyboard_first(kind, nid)
            return f"分镜里没有角色 id「{nid}」"
        return None
    if kind in {"frame", "end_frame", "voice", "clip"}:
        if not nid or nid not in shots:
            if out is not None and not on_disk:
                return _need_storyboard_first(kind, nid)
            return f"分镜里没有镜头「{nid}」"
        if kind == "voice" and not _has_dialogue(shots.get(nid)):
            return f"镜头 {nid} 无对白，不能重跑配音"
        if kind == "end_frame":
            shot = shots.get(nid) or {}
            if str(shot.get("reuse_frame") or "").strip():
                return f"镜头 {nid} 复用静帧，不能单独生成尾帧"
            if not _shot_wants_end(story, nid):
                return f"镜头 {nid} 不需要尾帧（运动档位不是 dynamic）"
        if kind == "frame" and str((shots.get(nid) or {}).get("reuse_frame") or "").strip():
            return f"镜头 {nid} 复用静帧，请重跑父镜静帧"
        return None
    if kind in {"music", "final"}:
        # 镜头列表还未知时，不能把「生成分镜」和配乐/成片排进同一次：
        # 静帧、配音、视频都要等分镜写出镜头 id 才能补。
        if not shots:
            if out is not None and not on_disk:
                if not _novel_ready(out):
                    label = KIND_LABELS.get(kind, kind)
                    return (
                        f"还没有小说，也没有分镜，不能生成{label}。"
                        "请先在上方填写小说并生成分镜，或点 ＋ 手写镜头。"
                    )
                return _need_storyboard_first(kind)
            return _need_shots(kind)
        return None
    return f"未知产物类型：{kind}"


def queue_block_reason(out: str | Path, items: list[dict]) -> str | None:
    """启动前再查一遍当前文件：坏分镜、镜头对不上、或分镜与后置步骤绑在同一次。"""
    try:
        story = _load_story(out, optional=True)
    except ValueError as exc:
        return str(exc)
    kinds = [str((it or {}).get("kind") or "") for it in items or []]
    if "storyboard" in kinds and any(k and k != "storyboard" for k in kinds):
        return STORYBOARD_EXCLUSIVE
    for it in items or []:
        if not isinstance(it, dict):
            continue
        err = validate_target(
            story,
            str(it.get("kind") or ""),
            str(it.get("id") or ""),
            out=out,
        )
        if err:
            label = str(it.get("label") or "").strip()
            return f"{label}：{err}" if label else err
    return None


def plan_rerun(out: str | Path, *, kind: str, id: str = "") -> dict:
    """计算单跑计划：前向闭包 + 目标 + 后向闭包。"""
    kind = str(kind or "").strip()
    nid = str(id or "").strip()
    # optional：缺 storyboard.json 时仍可规划；文件在但不是合法分镜则直接拒绝
    try:
        story = _load_story(out, optional=True)
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "prereqs": [],
            "target": None,
            "posts": [],
        }
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

    # 从小说生成分镜会牵出整条后置链，默认不勾且不可选（须先单独跑分镜）。
    # 成片：条件不齐时禁用勾选（勾上必然缺镜/缺配乐，或会误拉全片重渲）。
    suggest_posts = kind != "storyboard"
    final_ok, final_hint = final_post_suggestion(
        out,
        story,
        target_kind=kind,
        target_id=nid,
        post_nodes=post_nodes,
    )
    posts = []
    for pk, pid in post_nodes:
        suggested = suggest_posts
        enabled = suggest_posts
        hint = ""
        if kind == "storyboard":
            enabled = False
            hint = "请先只生成分镜，完成后再单独添加后置产物"
        elif pk == "final":
            suggested = suggest_posts and final_ok
            enabled = suggested
            if not enabled:
                hint = final_hint or "成片条件未齐，不能勾选"
        posts.append(
            {
                "kind": pk,
                "id": pid,
                "label": node_label(story, pk, pid),
                "suggested": suggested,
                "enabled": enabled,
                "hint": hint,
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
        if target.get("kind") == "storyboard":
            parts.append("请先只生成分镜。分镜写完后，再单独添加静帧、配音、视频、配乐或成片")
        elif len(chosen) == len(posts):
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

    return _groups_from_kind_nodes(run_nodes)


def _groups_from_kind_nodes(nodes: Iterable[dict]) -> list[dict]:
    """把产物节点收成 CLI 阶段组；首帧与尾帧拆成两次 images。"""
    portraits: list[str] = []
    frames: list[str] = []
    end_frames: list[str] = []
    voices: list[str] = []
    clips: list[str] = []
    want_storyboard = False
    want_music = False
    want_final = False
    force_portraits = False
    force_frames = False
    force_ends = False
    force_voices = False
    force_clips = False

    for n in nodes:
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
        elif kind == "end_frame":
            end_frames.append(nid)
            force_ends = force_ends or force
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
    portraits, frames, end_frames, voices, clips = map(
        uniq, (portraits, frames, end_frames, voices, clips)
    )

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
                # 队列里首帧与尾帧分开；整包 produce 不走这里，仍会自动衔尾帧
                "skip_end": True,
            }
        )
    if end_frames:
        groups.append(
            {
                "stage": "images",
                "only": ",".join(end_frames),
                "force": force_ends,
                "end_only": True,
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


def selftest_web_edges() -> None:
    """缺文件、空项目、坏分镜、不按顺序单跑：规划必须拒绝或把缺失前置补全。"""
    import shutil

    from pipeline.projects import artifacts_sub
    from pipeline.run_queue import add_from_plan
    from pipeline.util import ROOT, cite_path
    from pipeline.web import (
        materialize_web_draft,
        read_storyboard,
        read_text,
        validate_storyboard,
        write_storyboard,
    )

    root = ROOT / "output" / "_web_edge_selftest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    def pre(plan: dict) -> set[tuple[str, str, str]]:
        assert plan.get("ok"), plan
        return {(p["kind"], p.get("id") or "", p.get("status")) for p in plan["prereqs"]}

    story = {
        "title": "边角",
        "style": "写实",
        "music_prompt": "soft piano, no vocals",
        "characters": [{"id": "lin", "name": "林", "gender": "female"}],
        "shots": [
            {
                "id": "s001",
                "characters": ["lin"],
                "image_prompt": "近景",
                "motion_prompt": "Slow push in",
                "motion_level": "subtle",
                "dialogue": "你好",
                "speaker": "林",
            },
            {
                "id": "s002",
                "characters": [],
                "image_prompt": "空镜",
                "motion_prompt": "Locked off",
                "motion_level": "normal",
                "dialogue": "",
            },
            {
                "id": "s001_p1",
                "characters": ["lin"],
                "reuse_frame": "s001",
                "image_prompt": "近景",
                "motion_prompt": "Locked off",
                "motion_level": "subtle",
                "dialogue": "后半",
                "speaker": "林",
            },
            {
                "id": "s003",
                "characters": ["lin"],
                "image_prompt": "中景推门",
                "motion_prompt": "She pushes the door open",
                "motion_level": "dynamic",
                "end_image_prompt": "同一机位，门已推开半扇",
                "dialogue": "",
            },
        ],
    }

    try:
        empty = root / "empty"
        empty.mkdir()
        music = plan_rerun(empty, kind="music")
        assert not music["ok"] and "没有小说" in music["error"] and "没有分镜" in music["error"], music
        final = plan_rerun(empty, kind="final")
        assert not final["ok"] and "没有小说" in final["error"], final
        assert not plan_rerun(empty, kind="frame", id="s001")["ok"]
        assert "还没有分镜" in plan_rerun(empty, kind="clip", id="s001")["error"]
        assert "还没有分镜" in plan_rerun(empty, kind="portrait", id="lin")["error"]
        assert not plan_rerun(empty, kind="storyboard")["ok"]
        (empty / "novel.txt").write_text("  \n\t", encoding="utf-8")
        assert not plan_rerun(empty, kind="storyboard")["ok"]
        (empty / "novel.txt").write_text("她说：「你好」。\n", encoding="utf-8")
        assert plan_rerun(empty, kind="storyboard")["ok"]
        still = plan_rerun(empty, kind="final")
        assert not still["ok"] and "还没有分镜" in still["error"], still
        assert "缺失前置" not in (still.get("summary") or "")

        blank = root / "blank"
        blank.mkdir()
        sb_rel, _novel_rel = materialize_web_draft(
            str(blank),
            storyboard_data={"title": "空", "shots": [], "characters": [], "_missing": True},
            novel_text="  ",
        )
        assert sb_rel is None and not (blank / "storyboard.json").exists()
        assert not (blank / "work" / "webui" / "run_novel.txt").read_text(encoding="utf-8").strip()
        refused = add_from_plan(blank, kind="final")
        assert not refused["ok"] and not (refused.get("queue") or {}).get("items"), refused

        hand = root / "hand"
        hand.mkdir()
        materialize_web_draft(
            str(hand),
            storyboard_data={
                "title": "手写",
                "characters": [{"id": "lin", "name": "林"}],
                "shots": [{
                    "id": "s001",
                    "characters": ["lin"],
                    "dialogue": "你好",
                    "image_prompt": "近景",
                }],
            },
            novel_text="",
        )
        assert (hand / "storyboard.json").is_file()
        handed = pre(plan_rerun(hand, kind="clip", id="s001"))
        assert ("storyboard", "", "ready") in handed
        assert ("frame", "s001", "missing") in handed
        assert ("voice", "s001", "missing") in handed
        assert ("portrait", "lin", "missing") in handed

        film = root / "film"
        film.mkdir()
        (film / "novel.txt").write_text("她说：「你好」。\n", encoding="utf-8")
        (film / "storyboard.json").write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
        clip = pre(plan_rerun(film, kind="clip", id="s001"))
        assert ("frame", "s001", "missing") in clip
        assert ("voice", "s001", "missing") in clip
        assert ("portrait", "lin", "missing") in clip
        assert ("storyboard", "", "ready") in clip
        # 其它镜还没视频：成片可列后置，但默认不勾
        clip_plan = plan_rerun(film, kind="clip", id="s001")
        final_post = next(p for p in clip_plan["posts"] if p["kind"] == "final")
        assert final_post["suggested"] is False
        assert final_post.get("enabled") is False
        assert "还缺" in (final_post.get("hint") or "")
        blocked = add_from_plan(
            film,
            kind="clip",
            id="s001",
            posts=[{"kind": "final", "id": ""}],
        )
        assert not blocked["ok"] and "还缺" in (blocked.get("error") or ""), blocked
        silent = pre(plan_rerun(film, kind="clip", id="s002"))
        assert ("frame", "s002", "missing") in silent
        assert not any(item[0] == "voice" for item in silent)
        assert not any(item[0] == "end_frame" for item in silent)
        child = pre(plan_rerun(film, kind="clip", id="s001_p1"))
        assert ("frame", "s001", "missing") in child
        assert ("voice", "s001_p1", "missing") in child
        assert not any(item[1] == "s001_p1" and item[0] == "frame" for item in child)
        dyn = pre(plan_rerun(film, kind="clip", id="s003"))
        assert ("frame", "s003", "missing") in dyn
        assert ("end_frame", "s003", "missing") in dyn
        end_plan = plan_rerun(film, kind="end_frame", id="s003")
        assert end_plan["ok"], end_plan
        assert ("frame", "s003", "missing") in pre(end_plan)
        assert "不需要尾帧" in plan_rerun(film, kind="end_frame", id="s002")["error"]
        assert "复用静帧" in plan_rerun(film, kind="end_frame", id="s001_p1")["error"]
        frame_posts = {
            (p["kind"], p.get("id") or "") for p in plan_rerun(film, kind="frame", id="s003")["posts"]
        }
        assert ("end_frame", "s003") in frame_posts and ("clip", "s003") in frame_posts
        assert "无对白" in plan_rerun(film, kind="voice", id="s002")["error"]
        assert "没有镜头" in plan_rerun(film, kind="clip", id="s999")["error"]
        assert "未知产物" in plan_rerun(film, kind="bogus")["error"]
        whole = pre(plan_rerun(film, kind="final"))
        for sid in ("s001", "s002", "s001_p1", "s003"):
            assert ("clip", sid, "missing") in whole
        assert ("music", "", "missing") in whole
        assert ("voice", "s002", "missing") not in whole
        bare = dict(story)
        bare["music_prompt"] = ""
        (film / "storyboard.json").write_text(json.dumps(bare, ensure_ascii=False), encoding="utf-8")
        no_bgm = pre(plan_rerun(film, kind="final"))
        assert not any(item[0] == "music" for item in no_bgm)
        (film / "storyboard.json").write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")

        frame = artifacts_sub(film, "frames", "s002.png")
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"")
        assert ("frame", "s002", "missing") in pre(plan_rerun(film, kind="clip", id="s002"))
        frame.write_bytes(b"png")
        assert ("frame", "s002", "ready") in pre(plan_rerun(film, kind="clip", id="s002"))
        voice = artifacts_sub(film, "audio", "voice", "s001.wav")
        voice.parent.mkdir(parents=True, exist_ok=True)
        voice.write_bytes(b"")
        assert ("voice", "s001", "missing") in pre(plan_rerun(film, kind="music"))

        # 其它镜视频齐了 + 有配乐 → 重跑一镜时成片才默认勾选
        for sid in ("s001", "s002", "s001_p1", "s003"):
            cpath = artifacts_sub(film, "clips", f"{sid}.mp4")
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_bytes(b"mp4")
        bgm = artifacts_sub(film, "audio", "bgm.wav")
        bgm.parent.mkdir(parents=True, exist_ok=True)
        bgm.write_bytes(b"wav")
        ready_final = next(
            p for p in plan_rerun(film, kind="clip", id="s002")["posts"] if p["kind"] == "final"
        )
        assert ready_final["suggested"] is True
        assert ready_final.get("enabled") is True
        assert not (ready_final.get("hint") or "")

        added = add_from_plan(film, kind="clip", id="s002", posts=[])
        assert added["ok"], added
        keys = {(it["kind"], it.get("id") or "") for it in added["queue"]["items"]}
        assert ("clip", "s002") in keys
        assert ("frame", "s002") not in keys  # 静帧已有，不入队
        assert ("final", "") not in keys  # 未勾选成片，不入队
        assert not any(it[0] == "voice" for it in keys)

        exclusive = add_from_plan(
            film,
            kind="storyboard",
            posts=[{"kind": "frame", "id": "s002"}],
        )
        assert not exclusive["ok"] and "只跑分镜" in exclusive["error"], exclusive
        # 拒绝入队时不要把后置写进队列
        assert ("frame", "s002") not in {
            (it["kind"], it.get("id") or "") for it in exclusive["queue"]["items"]
        }
        regen = plan_rerun(film, kind="storyboard")
        assert regen["ok"] and "请先只生成分镜" in regen["summary"], regen

        assert queue_block_reason(film, [
            {"kind": "storyboard", "id": "", "label": "分镜"},
            {"kind": "final", "id": "", "label": "成片"},
        ]) == STORYBOARD_EXCLUSIVE
        assert queue_block_reason(film, [{"kind": "clip", "id": "s001", "label": "视频 s001"}]) is None
        assert "没有镜头" in (queue_block_reason(
            film, [{"kind": "clip", "id": "s404", "label": "视频 s404"}]
        ) or "")
        (film / "storyboard.json").unlink()
        gone = queue_block_reason(film, [{"kind": "final", "id": "", "label": "成片"}]) or ""
        assert "还没有分镜" in gone, gone
        (film / "novel.txt").unlink()
        neither = queue_block_reason(film, [{"kind": "music", "id": "", "label": "配乐"}]) or ""
        assert "没有小说" in neither and "没有分镜" in neither, neither

        bad = root / "bad"
        bad.mkdir()
        (bad / "novel.txt").write_text("正文\n", encoding="utf-8")
        (bad / "storyboard.json").write_text("{", encoding="utf-8")
        broken = plan_rerun(bad, kind="final")
        assert not broken["ok"] and "损坏" in broken["error"], broken
        assert "损坏" in (queue_block_reason(bad, [{"kind": "final", "id": ""}]) or "")
        (bad / "storyboard.json").write_text("[]", encoding="utf-8")
        assert "顶层" in plan_rerun(bad, kind="music")["error"]
        (bad / "storyboard.json").write_text('{"shots": "x"}', encoding="utf-8")
        assert "shots" in plan_rerun(bad, kind="frame", id="s001")["error"]
        (bad / "storyboard.json").write_bytes(b"")
        assert "损坏" in plan_rerun(bad, kind="final")["error"]

        rel = cite_path(bad / "storyboard.json")
        info = read_storyboard(rel)
        assert info.get("error") and "损坏" in info["error"], info
        missing = read_storyboard(cite_path(bad / "nope.json"))
        assert missing["exists"] is False and "error" not in missing
        (bad / "storyboard.json").write_text("[]", encoding="utf-8")
        assert "顶层" in read_storyboard(cite_path(bad / "storyboard.json"))["error"]
        checked = validate_storyboard(cite_path(bad / "storyboard.json"), strict=True)
        assert checked["ok"] is False and "顶层" in (checked.get("strict_error") or ""), checked
        write_storyboard(cite_path(bad / "storyboard.json"), {
            "title": "写回",
            "shots": [],
            "characters": [],
            "_missing": True,
            "_corrupt": "x",
        })
        saved = json.loads((bad / "storyboard.json").read_text(encoding="utf-8"))
        assert "_missing" not in saved and "_corrupt" not in saved and saved["title"] == "写回"
        try:
            read_text(cite_path(bad / "missing.txt"))
            raise AssertionError("缺文件应报错")
        except FileNotFoundError as exc:
            assert "找不到文件" in str(exc)
        (bad / "novel.txt").write_bytes(b"\xff\xfe")
        try:
            read_text(cite_path(bad / "novel.txt"))
            raise AssertionError("非 UTF-8 应报错")
        except ValueError as exc:
            assert "UTF-8" in str(exc)

        shell = root / "shell"
        shell.mkdir()
        (shell / "storyboard.json").write_text("{}\n", encoding="utf-8")
        assert "还没有镜头" in plan_rerun(shell, kind="final")["error"]
        assert "还没有镜头" in plan_rerun(shell, kind="music")["error"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
