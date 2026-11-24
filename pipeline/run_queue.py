"""项目待跑队列：左边「生成过程」里要启动的产物清单。

与历史任务（点一次启动算一条 Job）分离；队列持久化在
``projects/<片>/work/webui/queue.json``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from pipeline.artifact_plan import (
    _exists,
    _node_key,
    _parse_key,
    _reachable,
    _topo_sort,
    direct_pre,
    plan_rerun,
)
from pipeline.projects import artifacts_dir
from pipeline.util import load_json, resolve

QUEUE_FILENAME = "queue.json"


def queue_path(out: str | Path) -> Path:
    return artifacts_dir(out) / "webui" / QUEUE_FILENAME


def _empty() -> dict:
    return {"items": [], "updated_at": time.time()}


def load_queue(out: str | Path) -> dict:
    path = queue_path(out)
    if not path.is_file():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return _empty()
    cleaned = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip()
        nid = str(raw.get("id") or "").strip()
        if not kind:
            continue
        key = str(raw.get("key") or _node_key(kind, nid))
        cleaned.append(
            {
                "key": key,
                "kind": kind,
                "id": nid,
                "label": str(raw.get("label") or key),
                "force": bool(raw.get("force", True)),
                "status": str(raw.get("status") or "pending"),
            }
        )
    return {"items": cleaned, "updated_at": float(data.get("updated_at") or 0)}


def save_queue(out: str | Path, queue: dict) -> dict:
    path = queue_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(queue.get("items") or [])
    payload = {"items": items, "updated_at": time.time()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload


def _story(out: str | Path) -> dict:
    return load_json(resolve(out) / "storyboard.json")


def dependents_of(out: str | Path, items: list[dict], key: str) -> list[str]:
    """队列里依赖 key 的其它项（有则不可删 key）。"""
    story = _story(out)
    kind, nid = _parse_key(key)
    dependants: list[str] = []
    for it in items:
        if it["key"] == key:
            continue
        pre = _reachable(story, out, (it["kind"], it.get("id") or ""), forward=False)
        if (kind, nid) in pre:
            dependants.append(it["key"])
            continue
        if (kind, nid) in direct_pre(story, out, it["kind"], it.get("id") or ""):
            dependants.append(it["key"])
    return dependants


def can_remove(out: str | Path, items: list[dict], key: str) -> tuple[bool, str]:
    deps = dependents_of(out, items, key)
    if not deps:
        return True, ""
    by = {it["key"]: it.get("label") or it["key"] for it in items}
    return False, "被依赖，不能删：" + "、".join(by.get(k, k) for k in deps)


def reset_statuses(queue: dict, status: str = "pending") -> dict:
    for it in queue.get("items") or []:
        it["status"] = status
    return queue


def queue_key_for_step(step_id: str) -> str | None:
    """进度步骤 id → 队列项 key；对不上则 None。"""
    sid = str(step_id or "").strip()
    if not sid:
        return None
    if sid.startswith("images:portrait:"):
        return f"portrait:{sid.split(':', 2)[-1]}"
    if sid.startswith("images:") and not sid.startswith("images:_"):
        return f"frame:{sid.split(':', 1)[1]}"
    if sid.startswith("audio:") and not sid.startswith("audio:_"):
        return f"voice:{sid.split(':', 1)[1]}"
    if sid.startswith("videos:") and not sid.startswith("videos:_"):
        return f"clip:{sid.split(':', 1)[1]}"
    if sid.startswith("music:"):
        return "music:"
    if sid == "assemble:mux":
        return "final:"
    return None


def drop_keys(queue: dict, keys: Iterable[str]) -> list[str]:
    """从队列去掉指定 key，返回实际删掉的 key 列表。"""
    drop = {str(k) for k in keys if k}
    if not drop:
        return []
    before = list(queue.get("items") or [])
    kept = [it for it in before if it.get("key") not in drop]
    removed = [str(it.get("key")) for it in before if it.get("key") in drop]
    queue["items"] = kept
    return removed


def prune_queue_by_steps(out: str | Path, steps: Iterable[dict]) -> dict | None:
    """按已完成进度步骤剔除对应队列项；有改动则写盘并返回队列，否则 None。"""
    done_keys: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict) or step.get("status") != "done":
            continue
        key = queue_key_for_step(str(step.get("id") or ""))
        if key:
            done_keys.append(key)
    if not done_keys:
        return None
    queue = load_queue(out)
    removed = drop_keys(queue, done_keys)
    if not removed:
        return None
    save_queue(out, queue)
    return queue


def add_from_plan(
    out: str | Path,
    *,
    kind: str,
    id: str = "",
    posts: list[dict] | None = None,
) -> dict:
    """目标 + 缺失前置 + 勾选后置入队；去重；已有可升 force。"""
    plan = plan_rerun(out, kind=str(kind or "").strip(), id=str(id or "").strip())
    if not plan.get("ok"):
        return {"ok": False, "error": plan.get("error") or "无法规划", "queue": load_queue(out)}

    target = plan["target"]
    allowed = {(p["kind"], p.get("id") or "") for p in plan.get("posts") or []}
    selected_posts: list[dict] = []
    for raw in posts or []:
        if not isinstance(raw, dict):
            continue
        pk = str(raw.get("kind") or "").strip()
        pid = str(raw.get("id") or "").strip()
        if (pk, pid) not in allowed:
            continue
        label = next(
            (p["label"] for p in plan["posts"] if p["kind"] == pk and (p.get("id") or "") == pid),
            pk,
        )
        selected_posts.append({"kind": pk, "id": pid, "label": label})

    to_add: list[dict] = []
    for p in plan.get("prereqs") or []:
        if p.get("status") != "missing":
            continue
        to_add.append(
            {
                "key": _node_key(p["kind"], p.get("id") or ""),
                "kind": p["kind"],
                "id": p.get("id") or "",
                "label": p.get("label") or p["kind"],
                "force": False,
                "status": "pending",
            }
        )
    to_add.append(
        {
            "key": _node_key(target["kind"], target.get("id") or ""),
            "kind": target["kind"],
            "id": target.get("id") or "",
            "label": target.get("label") or target["kind"],
            "force": True,
            "status": "pending",
        }
    )
    for p in selected_posts:
        to_add.append(
            {
                "key": _node_key(p["kind"], p.get("id") or ""),
                "kind": p["kind"],
                "id": p.get("id") or "",
                "label": p.get("label") or p["kind"],
                "force": True,
                "status": "pending",
            }
        )

    queue = load_queue(out)
    by_key = {it["key"]: dict(it) for it in queue["items"]}
    added, upgraded = [], []
    for node in to_add:
        key = node["key"]
        if key in by_key:
            if node["force"] and not by_key[key].get("force"):
                by_key[key]["force"] = True
                upgraded.append(by_key[key]["label"])
            continue
        by_key[key] = dict(node)
        added.append(node["label"])

    ordered = _topo_sort([(it["kind"], it.get("id") or "") for it in by_key.values()])
    queue["items"] = [by_key[_node_key(k, i)] for k, i in ordered if _node_key(k, i) in by_key]
    save_queue(out, queue)
    return {
        "ok": True,
        "error": None,
        "queue": annotate_removable(out, queue),
        "added": added,
        "upgraded": upgraded,
        "summary": plan.get("summary") or "",
    }


def remove_item(out: str | Path, key: str) -> dict:
    queue = load_queue(out)
    items = queue["items"]
    key = str(key or "").strip()
    if not any(it["key"] == key for it in items):
        return {"ok": False, "error": "队列里没有这项", "queue": annotate_removable(out, queue)}
    ok, reason = can_remove(out, items, key)
    if not ok:
        return {"ok": False, "error": reason, "queue": annotate_removable(out, queue)}
    queue["items"] = [it for it in items if it["key"] != key]
    save_queue(out, queue)
    return {"ok": True, "error": None, "queue": annotate_removable(out, queue)}


def clear_queue(out: str | Path) -> dict:
    queue = _empty()
    save_queue(out, queue)
    return {"ok": True, "queue": annotate_removable(out, queue)}


def annotate_removable(out: str | Path, queue: dict) -> dict:
    items = queue.get("items") or []
    try:
        out_items = []
        for it in items:
            ok, reason = can_remove(out, items, it["key"])
            out_items.append({**it, "removable": ok, "block_reason": "" if ok else reason})
        return {**queue, "items": out_items}
    except Exception:
        return {
            **queue,
            "items": [{**it, "removable": True, "block_reason": ""} for it in items],
        }


def _groups_from_nodes(nodes: Iterable[dict]) -> list[dict]:
    portraits: list[str] = []
    frames: list[str] = []
    voices: list[str] = []
    clips: list[str] = []
    want_music = False
    want_final = False
    force_portraits = force_frames = force_voices = force_clips = False

    for n in nodes:
        kind = n["kind"]
        nid = n.get("id") or ""
        force = bool(n.get("force"))
        if kind == "portrait":
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
    portraits, frames, voices, clips = map(uniq, (portraits, frames, voices, clips))
    if portraits or frames:
        groups.append(
            {
                "stage": "images",
                "characters": portraits,
                "only": ",".join(frames) if frames else None,
                "force": force_portraits or force_frames,
            }
        )
    if voices:
        groups.append({"stage": "audio", "only": ",".join(voices), "force": force_voices})
    if clips:
        groups.append({"stage": "videos", "only": ",".join(clips), "force": force_clips})
    if want_music:
        groups.append({"stage": "music", "force": True})
    if want_final:
        groups.append({"stage": "assemble", "force": True})
    return groups


def build_groups_from_queue(out: str | Path, queue: dict | None = None) -> list[dict]:
    queue = queue or load_queue(out)
    nodes = []
    for it in queue.get("items") or []:
        force = bool(it.get("force"))
        if not force and _exists(out, it["kind"], it.get("id") or ""):
            continue
        nodes.append(
            {
                "kind": it["kind"],
                "id": it.get("id") or "",
                "force": force,
                "label": it.get("label"),
            }
        )
    return _groups_from_nodes(nodes)


def queue_only_shots(queue: dict) -> str | None:
    """进度引擎 / 历史任务的 only：只列队列里出现过的镜头。

    即使队列里有 music/final，也要用镜头子集缩步骤表，
    否则 live-items 会把分镜里全部 s00x 都画出来。
    队列完全没有镜头级项（只配乐/成片）时返回 None。
    """
    seen: list[str] = []
    for it in queue.get("items") or []:
        if it.get("kind") in {"frame", "voice", "clip"} and it.get("id"):
            if it["id"] not in seen:
                seen.append(it["id"])
    return ",".join(seen) if seen else None
