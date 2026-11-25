"""成片进度真相源：完整步骤表 + 步内观测 + 权重 + 总进度 + progress.json。

网页 Job（on_line）与 AttachedJob（扫盘）都喂同一 ProgressEngine；
快照经 SSE /api/progress/<id> 推送，并原子写入 work/webui/progress.json。
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable

from pipeline.projects import artifacts_sub
from pipeline.util import resolve

STAGES = ("storyboard", "screenplay", "images", "audio", "videos", "music", "assemble")
STAGE_LABELS = {
    "storyboard": "生成分镜",
    "screenplay": "校验分镜",
    "images": "静帧生成",
    "audio": "对白配音",
    "videos": "视频生成",
    "music": "配乐生成",
    "assemble": "合成混音",
}
STAGE_WEIGHTS = {
    "storyboard": 0.06,
    "screenplay": 0.02,
    "images": 0.17,
    "audio": 0.10,
    "videos": 0.55,
    "music": 0.05,
    "assemble": 0.05,
}

PROGRESS_FILENAME = "progress.json"
DENOISE_PLAN = (8, 3)

_TQDM = re.compile(r"^(?P<desc>.+?):\s+(?P<pct>\d+)%.*? (?P<n>\d+)/(?P<total>\d+)\b")
_TQDM_ETA = re.compile(r"\[(?P<elapsed>[^<]*)<(?P<eta>[^\],]*)")

# tqdm desc → step_id 模板
_DESC_MAP = (
    (re.compile(r"^(?P<id>\S+)\s+(?:a2v|i2v)$"), "videos:{id}"),
    (re.compile(r"^静帧\s+(?P<id>\S+)"), "images:{id}"),
    (re.compile(r"^空镜\s+(?P<id>\S+)"), "images:{id}"),
    (re.compile(r"^尾帧\s+(?P<id>\S+)"), "images:{id}:end"),
    (re.compile(r"^肖像\s+(?P<name>.+)$"), "portrait:{name}"),
    (re.compile(r"^(?:配乐|MusicGen)$"), "music:bgm"),
    (re.compile(r"^Denoising$"), "videos:_denoise"),  # 原始段，靠 current 镜合并
)

_ITEM_START = (
    (re.compile(r"^角色参考图\s+(.+?)\s*->"), "portrait"),
    (re.compile(r"^空镜\s+(\S+)\s*->"), "images"),
    (re.compile(r"^分镜静帧\s+(\S+)"), "images"),
    (re.compile(r"^尾帧\s+(\S+)\s*<-"), "images_end"),
    (re.compile(r"^对白\s+(\S+)\s*·"), "audio"),
    (re.compile(r"^整理\s+(\S+)"), "assemble"),
)
_STORYBOARD_START = re.compile(r"^(?:加载分镜 LLM|小说\s*→\s*分镜|生成分镜：)")
_STORYBOARD_DONE = re.compile(r"^(?:已写分镜|已有分镜，跳过生成)")
_MUX_START = re.compile(r"^混音成片\b")
_MUX_DONE = re.compile(r"^成片：")
_ITEM_DONE = (re.compile(r"^已有镜头\s+(\S+)，跳过"),)
_ITEM_FAIL = (
    re.compile(r"^镜头\s+(\S+)\s+失败"),
    re.compile(r"^跳过\s+(\S+)[:：]"),
)
_STAGE_SEP = re.compile(r"^==\s*(\w+)\s*==$")


def progress_path(out: str | Path) -> Path:
    return artifacts_sub(out, "webui") / PROGRESS_FILENAME


def read_progress(out: str | Path) -> dict | None:
    path = progress_path(out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_progress(out: str | Path, snap: dict) -> None:
    path = progress_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _parse_hms(text: str) -> float | None:
    text = (text or "").strip()
    m = re.match(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$", text)
    if not m:
        return None
    h, mi, s = (g or "0" for g in m.groups())
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _load_story(out: str | Path) -> dict:
    path = resolve(out) / "storyboard.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _existing_stems(folder: Path, exts: tuple[str, ...]) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts}


def _parse_only_ids(only: str | None) -> set[str] | None:
    if not only:
        return None
    ids = {s.strip() for s in str(only).replace("，", ",").split(",") if s.strip()}
    return ids or None


def build_plan(
    out: str | Path,
    stages: list[str] | tuple[str, ...] | None = None,
    *,
    only: str | None = None,
    characters: str | None = None,
    image_parts: str | None = None,
) -> list[dict]:
    """按分镜生成完整步骤表；阶段权重均分到该阶段各原子步。

    ``only``：镜头级重跑时只列这些镜头。
    ``characters``：显式要跑的肖像 id；队列里有肖像时传入，避免被 only 滤掉。
    未传 ``characters`` 且带了 ``only`` 时不列肖像（单跑静帧复用已有头像）。
    ``image_parts``：``start`` / ``end`` / ``both``（默认）。只跑尾帧时不要列首帧进度。
    """
    parts = str(image_parts or "both").strip().lower()
    if parts not in {"start", "end", "both"}:
        parts = "both"
    want_start = parts in {"start", "both"}
    want_end = parts in {"end", "both"}

    # 队列里首帧/尾帧会拆成两次 images 命令，stage_keys 可能重复；步骤表只建一份
    selected: list[str] = []
    seen_stages: set[str] = set()
    for s in stages or STAGES:
        if s in STAGE_WEIGHTS and s not in seen_stages:
            selected.append(s)
            seen_stages.add(s)
    if not selected:
        selected = list(STAGES)
    only_ids = _parse_only_ids(only)
    char_filter = _parse_only_ids(characters)
    story = _load_story(out)
    shots = [s for s in (story.get("shots") or []) if str(s.get("id") or "").strip()]
    chars = [c for c in (story.get("characters") or []) if str(c.get("id") or "").strip()]
    shot_ids = [str(s["id"]).strip() for s in shots]
    if only_ids:
        shot_ids = [sid for sid in shot_ids if sid in only_ids]
    dialogue_ids = [
        str(s["id"]).strip()
        for s in shots
        if str(s.get("dialogue") or "").strip()
        and (not only_ids or str(s["id"]).strip() in only_ids)
    ]
    char_ids = [str(c["id"]).strip() for c in chars]
    char_names = {str(c["id"]).strip(): str(c.get("name") or c["id"]).strip() for c in chars}
    if char_filter is not None:
        portrait_ids = [cid for cid in char_ids if cid in char_filter]
    elif not only_ids:
        portrait_ids = list(char_ids)
    else:
        portrait_ids = []

    by_stage: dict[str, list[tuple[str, str]]] = {s: [] for s in selected}

    if "storyboard" in by_stage:
        by_stage["storyboard"].append(("storyboard:generate", "生成分镜"))

    if "screenplay" in by_stage:
        by_stage["screenplay"].append(("screenplay:validate", "校验分镜"))

    if "images" in by_stage:
        from pipeline.motion import wants_end_frame

        shot_by_id = {str(s.get("id") or "").strip(): s for s in shots}
        for cid in portrait_ids:
            name = char_names.get(cid, cid)
            by_stage["images"].append((f"images:portrait:{cid}", f"肖像 {name}"))
        for sid in shot_ids:
            if want_start:
                by_stage["images"].append((f"images:{sid}", f"静帧 {sid}"))
            # 需要首尾帧时拆成两步；单跑尾帧只列尾帧，避免空着的首帧进度条
            if want_end and wants_end_frame(shot_by_id.get(sid) or {}):
                by_stage["images"].append((f"images:{sid}:end", f"尾帧 {sid}"))
        if not by_stage["images"]:
            by_stage["images"].append(("images:_phase", "静帧生成"))

    if "audio" in by_stage:
        for sid in dialogue_ids:
            by_stage["audio"].append((f"audio:{sid}", f"对白 {sid}"))
        if not by_stage["audio"]:
            by_stage["audio"].append(("audio:_skip", "对白（无）"))

    if "videos" in by_stage:
        for sid in shot_ids:
            by_stage["videos"].append((f"videos:{sid}", f"{sid} 视频"))
        if not by_stage["videos"]:
            by_stage["videos"].append(("videos:_phase", "视频生成"))

    if "music" in by_stage:
        by_stage["music"].append(("music:bgm", "配乐"))

    if "assemble" in by_stage:
        for sid in shot_ids:
            by_stage["assemble"].append((f"assemble:{sid}", f"整理 {sid}"))
        by_stage["assemble"].append(("assemble:mux", "混音成片"))
        if len(by_stage["assemble"]) == 1 and not shot_ids:
            pass

    steps: list[dict] = []
    for stage in selected:
        units = by_stage.get(stage) or [(f"{stage}:_phase", STAGE_LABELS.get(stage, stage))]
        w = float(STAGE_WEIGHTS.get(stage, 0.1))
        each = w / max(1, len(units))
        for sid, label in units:
            steps.append(
                {
                    "id": sid,
                    "stage": stage,
                    "label": label,
                    "weight": round(each, 6),
                    "pct": 0.0,
                    "step": 0,
                    "total": 1,
                    "eta_s": None,
                    "status": "pending",
                }
            )
    # 浮点归一：保证权重和 ≈ 选中阶段权重和
    total_w = sum(float(STAGE_WEIGHTS.get(s, 0)) for s in selected)
    got = sum(s["weight"] for s in steps) or 1.0
    if abs(got - total_w) > 1e-6 and steps:
        scale = total_w / got
        for s in steps:
            s["weight"] = round(s["weight"] * scale, 6)
    return steps


class ProgressEngine:
    """线程安全的进度引擎：feed_line / observe_disk → snapshot(seq)。"""

    def __init__(
        self,
        out: str,
        *,
        stages: list[str] | None = None,
        task_id: str = "",
        only: str | None = None,
        characters: str | None = None,
        image_parts: str | None = None,
        on_update: Callable[[dict], None] | None = None,
    ) -> None:
        self.out = out
        self.task_id = task_id
        self.only = only
        self.characters = characters
        self.image_parts = str(image_parts or "both").strip().lower() or "both"
        self.stages = []
        seen_stages: set[str] = set()
        for s in stages or list(STAGES):
            if s in STAGE_WEIGHTS and s not in seen_stages:
                self.stages.append(s)
                seen_stages.add(s)
        if not self.stages:
            self.stages = list(STAGES)
        self.on_update = on_update
        self.lock = threading.Lock()
        self.seq = 0
        self.status = "running"
        self.current_step_id: str | None = None
        self.current_stage: str | None = self.stages[0] if self.stages else None
        self.stage_status: dict[str, str] = {s: "pending" for s in self.stages}
        if self.stages:
            self.stage_status[self.stages[0]] = "running"
        self.steps: dict[str, dict] = {}
        self.step_order: list[str] = []
        self._portrait_by_name: dict[str, str] = {}
        self._video_merge = None
        self._video_merge_sid: str | None = None
        self._last_emit = 0.0
        self._last_emit_pct = -1.0
        self._subscribers: list[queue.Queue] = []
        self._soft_started: dict[str, float] = {}
        self._rebuild_plan()

    # -- plan --------------------------------------------------------------
    def _rebuild_plan(self) -> None:
        planned = build_plan(
            self.out,
            self.stages,
            only=self.only,
            characters=self.characters,
            image_parts=self.image_parts,
        )
        old = dict(self.steps)
        self.steps = {}
        self.step_order = []
        self._portrait_by_name = {}
        for spec in planned:
            sid = spec["id"]
            prev = old.get(sid)
            row = dict(spec)
            if prev:
                row["pct"] = max(float(prev.get("pct") or 0), float(row.get("pct") or 0))
                row["step"] = max(int(prev.get("step") or 0), int(row.get("step") or 0))
                row["total"] = max(int(prev.get("total") or 1), int(row.get("total") or 1))
                row["status"] = prev.get("status") or row["status"]
                row["eta_s"] = prev.get("eta_s")
                if row["status"] == "done":
                    row["pct"] = 100.0
            self.steps[sid] = row
            self.step_order.append(sid)
            if sid.startswith("images:portrait:"):
                # label 「肖像 林晚」
                name = (row.get("label") or "").replace("肖像 ", "", 1).strip()
                if name:
                    self._portrait_by_name[name] = sid

    def ensure_plan(self) -> None:
        snap = None
        with self.lock:
            self._rebuild_plan()
            snap = self._emit_locked(force=True)
        if snap:
            self._notify(snap)

    # -- subscribers -------------------------------------------------------
    def subscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self._subscribers.append(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _notify(self, snap: dict) -> None:
        """在锁外调用，避免 on_update → Job.snapshot 死锁。"""
        with self.lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(snap)
            except Exception:
                pass
        if self.on_update:
            try:
                self.on_update(snap)
            except Exception:
                pass

    def _should_emit(self, *, force: bool, overall: float) -> bool:
        if force:
            return True
        now = time.time()
        if now - self._last_emit >= 0.1:
            return True
        if abs(overall - self._last_emit_pct) >= 0.5:
            return True
        return False

    def _emit_locked(self, *, force: bool = False) -> dict | None:
        """持锁：bump seq、写盘；返回待推送 snap（锁外再 _notify）。"""
        snap = self._snapshot_locked()
        if not self._should_emit(force=force, overall=float(snap["overall_pct"])):
            return None
        self.seq += 1
        snap["seq"] = self.seq
        self._last_emit = time.time()
        self._last_emit_pct = float(snap["overall_pct"])
        try:
            write_progress(self.out, snap)
        except Exception:
            pass
        return snap

    # -- snapshot ----------------------------------------------------------
    def _stage_rows_locked(self) -> list[dict]:
        rows = []
        for stage in self.stages:
            ids = [sid for sid in self.step_order if self.steps[sid]["stage"] == stage]
            if not ids:
                rows.append(
                    {
                        "key": stage,
                        "label": STAGE_LABELS.get(stage, stage),
                        "weight": float(STAGE_WEIGHTS.get(stage, 0)),
                        "pct": 100.0 if self.stage_status.get(stage) == "done" else 0.0,
                        "status": self.stage_status.get(stage, "pending"),
                    }
                )
                continue
            wsum = sum(float(self.steps[i]["weight"]) for i in ids) or 1.0
            pct = sum(float(self.steps[i]["weight"]) * float(self.steps[i]["pct"]) for i in ids) / wsum
            st = self.stage_status.get(stage, "pending")
            if st == "pending" and any(self.steps[i]["status"] == "running" for i in ids):
                st = "running"
            if st != "done" and all(self.steps[i]["status"] == "done" for i in ids):
                st = "done"
                self.stage_status[stage] = "done"
            if st == "done":
                pct = 100.0
            rows.append(
                {
                    "key": stage,
                    "label": STAGE_LABELS.get(stage, stage),
                    "weight": float(STAGE_WEIGHTS.get(stage, 0)),
                    "pct": round(min(100.0, pct), 1),
                    "status": st,
                }
            )
        return rows

    def _snapshot_locked(self) -> dict:
        steps = [dict(self.steps[sid]) for sid in self.step_order]
        # weight × pct(0–100)：全完成时 Σ = 100 × Σweight；部分阶段按选中权重归一
        overall = sum(float(s["weight"]) * float(s["pct"]) for s in steps)
        w_sel = sum(float(STAGE_WEIGHTS.get(s, 0)) for s in self.stages) or 1.0
        overall_pct = round(min(100.0, overall / w_sel), 1)
        if self.status == "success":
            overall_pct = 100.0
        stages = self._stage_rows_locked()
        # 兼容旧前端字段
        running = [dict(s, id=s["id"]) for s in steps if s.get("status") == "running"]
        stage_pct = {r["key"]: r["pct"] for r in stages}
        return {
            "seq": self.seq,
            "task_id": self.task_id,
            "status": self.status,
            "overall_pct": overall_pct,
            "current_step_id": self.current_step_id,
            "current_stage": self.current_stage,
            "stages": stages,
            "steps": steps,
            "items": running,
            "stage_pct": stage_pct,
            "current": next((s["label"] for s in steps if s["id"] == self.current_step_id), None),
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot_locked()

    # -- stage cursor ------------------------------------------------------
    def set_stage(self, stage: str, status: str = "running") -> None:
        snap = None
        with self.lock:
            if stage not in self.stage_status:
                return
            if status == "running":
                idx = self.stages.index(stage) if stage in self.stages else -1
                for s in self.stages[: max(0, idx)]:
                    self.stage_status[s] = "done"
                    for sid in self.step_order:
                        if self.steps[sid]["stage"] == s and self.steps[sid]["status"] != "failed":
                            self.steps[sid]["status"] = "done"
                            self.steps[sid]["pct"] = 100.0
                self.stage_status[stage] = "running"
                self.current_stage = stage
            else:
                self.stage_status[stage] = status
                if status == "done":
                    for sid in self.step_order:
                        if self.steps[sid]["stage"] == stage and self.steps[sid]["status"] != "failed":
                            self.steps[sid]["status"] = "done"
                            self.steps[sid]["pct"] = 100.0
            snap = self._emit_locked(force=True)
        if snap:
            self._notify(snap)

    # -- step updates ------------------------------------------------------
    def _resolve_portrait(self, name: str) -> str | None:
        if name in self._portrait_by_name:
            return self._portrait_by_name[name]
        for sid, row in self.steps.items():
            if sid.startswith("images:portrait:") and name in (row.get("label") or ""):
                return sid
        return None

    def _activate(self, step_id: str) -> None:
        if step_id not in self.steps:
            return
        # 收掉其它 running（同阶段）
        for sid, row in self.steps.items():
            if sid == step_id:
                continue
            if row.get("status") == "running":
                row["status"] = "done"
                row["pct"] = 100.0
                row["eta_s"] = None
        row = self.steps[step_id]
        if row.get("status") == "pending":
            row["status"] = "running"
            row["pct"] = max(float(row.get("pct") or 0), 0.0)
        elif row.get("status") == "done":
            # 重入：保持 done，除非显式 force_reset
            pass
        else:
            row["status"] = "running"
        if row.get("status") == "running":
            self.current_step_id = step_id
            self.current_stage = row["stage"]
            self.stage_status[row["stage"]] = "running"
            self._soft_started.setdefault(step_id, time.time())

    def bump_step(
        self,
        step_id: str,
        *,
        pct: float | None = None,
        step: int | None = None,
        total: int | None = None,
        eta_s: float | None = None,
        status: str | None = None,
        force_reset: bool = False,
    ) -> None:
        snap = None
        with self.lock:
            self._bump_locked(
                step_id, pct=pct, step=step, total=total, eta_s=eta_s, status=status, force_reset=force_reset
            )
            snap = self._emit_locked(force=status in {"done", "failed"} or force_reset)
        if snap:
            self._notify(snap)

    def _bump_locked(
        self,
        step_id: str,
        *,
        pct: float | None = None,
        step: int | None = None,
        total: int | None = None,
        eta_s: float | None = None,
        status: str | None = None,
        force_reset: bool = False,
    ) -> None:
        if step_id not in self.steps:
            return
        row = self.steps[step_id]
        if force_reset:
            row["pct"] = 0.0
            row["step"] = 0
            row["total"] = max(1, int(total or row.get("total") or 1))
            row["status"] = "running"
            row["eta_s"] = None
            self.current_step_id = step_id
            self._soft_started[step_id] = time.time()
        if status == "failed":
            row["status"] = "failed"
            row["eta_s"] = None
            return
        if row.get("status") == "done" and not force_reset:
            return
        if row.get("status") == "pending":
            self._activate(step_id)
            row = self.steps[step_id]
        if total is not None:
            row["total"] = max(int(row.get("total") or 1), int(total))
        if step is not None:
            row["step"] = max(int(row.get("step") or 0), int(step))
        new_pct = pct
        if new_pct is None and step is not None and total:
            new_pct = 100.0 * float(step) / max(1, int(total))
        if new_pct is not None:
            row["pct"] = max(float(row.get("pct") or 0), min(100.0, float(new_pct)))
        if eta_s is not None and row.get("status") == "running":
            row["eta_s"] = eta_s
        if status == "done" or (step is not None and total and step >= total):
            row["status"] = "done"
            row["pct"] = 100.0
            row["step"] = int(row.get("total") or 1)
            row["eta_s"] = None
        elif row.get("status") != "failed":
            row["status"] = "running"
            self.current_step_id = step_id

    # -- line feed ---------------------------------------------------------
    def feed_line(self, line: str, stage: str | None = None) -> None:
        text = (line or "").strip()
        if not text:
            return
        snap = None
        with self.lock:
            self._feed_locked(text, stage)
            snap = self._emit_locked()
        if snap:
            self._notify(snap)

    def _feed_locked(self, text: str, stage: str | None) -> None:
        m = _STAGE_SEP.match(text)
        if m:
            st = m.group(1)
            if st in self.stage_status:
                idx = self.stages.index(st) if st in self.stages else -1
                for s in self.stages[: max(0, idx)]:
                    self.stage_status[s] = "done"
                    for sid in self.step_order:
                        if self.steps[sid]["stage"] == s and self.steps[sid]["status"] != "failed":
                            self.steps[sid]["status"] = "done"
                            self.steps[sid]["pct"] = 100.0
                self.stage_status[st] = "running"
                self.current_stage = st
                # 分镜出现后补全步骤表
                if st == "images" or (resolve(self.out) / "storyboard.json").exists():
                    self._rebuild_plan()
            return

        tm = _TQDM.match(text)
        if tm:
            desc = tm.group("desc").strip()
            n, total = int(tm.group("n")), int(tm.group("total"))
            eta_m = _TQDM_ETA.search(text)
            eta_s = _parse_hms(eta_m.group("eta")) if eta_m else None
            step_id = self._step_id_for_desc(desc)
            if step_id == "videos:_denoise":
                step_id = self._merge_denoise(n, total, eta_s)
                if step_id:
                    return
                return
            if not step_id:
                return
            # 过期 leave：其它步在跑且本行已满格 → 忽略
            running = [sid for sid, r in self.steps.items() if r.get("status") == "running"]
            if running and step_id not in running and n >= total:
                return
            if self.steps.get(step_id, {}).get("status") == "done" and n >= total:
                return
            if step_id not in running:
                self._activate(step_id)
            self._bump_locked(step_id, step=n, total=total, pct=100.0 * n / max(1, total), eta_s=eta_s)
            return

        if _STORYBOARD_START.match(text) and "storyboard:generate" in self.steps:
            self._activate("storyboard:generate")
            if self.steps["storyboard:generate"]["status"] != "done":
                self.steps["storyboard:generate"]["status"] = "running"
                self.steps["storyboard:generate"]["pct"] = max(
                    float(self.steps["storyboard:generate"].get("pct") or 0), 5.0
                )
            return

        if _STORYBOARD_DONE.match(text) and "storyboard:generate" in self.steps:
            self._bump_locked("storyboard:generate", pct=100.0, status="done")
            if (resolve(self.out) / "storyboard.json").exists():
                self._rebuild_plan()
            return

        if _MUX_START.match(text) and "assemble:mux" in self.steps:
            self._activate("assemble:mux")
            if self.steps["assemble:mux"]["status"] != "done":
                self.steps["assemble:mux"]["status"] = "running"
                self.steps["assemble:mux"]["pct"] = max(
                    float(self.steps["assemble:mux"].get("pct") or 0), 1.0
                )
            return

        if _MUX_DONE.match(text) and "assemble:mux" in self.steps:
            self._bump_locked("assemble:mux", pct=100.0, status="done")
            return

        for pattern in _ITEM_DONE:
            mm = pattern.match(text)
            if mm:
                sid = mm.group(1)
                for cand in (f"videos:{sid}", f"images:{sid}", f"audio:{sid}"):
                    if cand in self.steps:
                        self._bump_locked(cand, pct=100.0, status="done")
                        break
                return

        for pattern in _ITEM_FAIL:
            mm = pattern.match(text)
            if mm:
                sid = mm.group(1)
                for cand in (f"videos:{sid}", f"images:{sid}", f"audio:{sid}", f"assemble:{sid}"):
                    if cand in self.steps:
                        self._bump_locked(cand, status="failed")
                        break
                return

        for pattern, kind in _ITEM_START:
            mm = pattern.match(text)
            if mm:
                name = mm.group(1)
                step_id = None
                if kind == "portrait":
                    step_id = self._resolve_portrait(name)
                elif kind == "images":
                    step_id = f"images:{name}"
                elif kind == "images_end":
                    step_id = f"images:{name}:end"
                elif kind == "audio":
                    step_id = f"audio:{name}"
                elif kind == "assemble":
                    step_id = f"assemble:{name}"
                if step_id and step_id in self.steps:
                    self._activate(step_id)
                    if self.steps[step_id]["status"] != "done":
                        self.steps[step_id]["status"] = "running"
                        self.steps[step_id]["pct"] = max(float(self.steps[step_id].get("pct") or 0), 1.0)
                return

    def _step_id_for_desc(self, desc: str) -> str | None:
        for pattern, template in _DESC_MAP:
            mm = pattern.match(desc)
            if not mm:
                continue
            if template == "portrait:{name}":
                return self._resolve_portrait(mm.group("name"))
            if template == "videos:_denoise":
                return "videos:_denoise"
            return template.format(**{k: mm.group(k) for k in mm.groupdict()})
        return None

    def _merge_denoise(self, n: int, total: int, eta_s: float | None) -> str | None:
        """原始 Denoising 行：挂到当前视频步，用 StageMerge(8,3)。"""
        from pipeline.progress import StageMerge

        sid = self.current_step_id
        if not sid or not sid.startswith("videos:"):
            # 找第一个未完成的视频步
            for cand in self.step_order:
                if cand.startswith("videos:") and self.steps[cand]["status"] not in {"done", "failed"}:
                    sid = cand
                    break
        if not sid or sid not in self.steps:
            return None
        if self._video_merge is None or self._video_merge_sid != sid:
            self._video_merge = StageMerge(DENOISE_PLAN)
            self._video_merge_sid = sid
        done, overall = self._video_merge.update(n, total)
        self._activate(sid)
        self._bump_locked(
            sid,
            step=done,
            total=overall or sum(DENOISE_PLAN),
            pct=100.0 * done / max(1, overall or sum(DENOISE_PLAN)),
            eta_s=eta_s,
        )
        return sid

    # -- disk observe ------------------------------------------------------
    def observe_disk(self) -> None:
        snap = None
        with self.lock:
            self._observe_disk_locked()
            snap = self._emit_locked()
        if snap:
            self._notify(snap)

    def _observe_disk_locked(self) -> None:
        base = resolve(self.out)
        # 分镜就绪后重建 plan（CLI 先校验再出图）
        if (base / "storyboard.json").exists() and not any(
            sid.startswith("images:") and sid != "images:_phase" for sid in self.step_order
        ):
            self._rebuild_plan()

        frames = _existing_stems(artifacts_sub(self.out, "frames"), (".png", ".jpg", ".jpeg", ".webp"))
        portraits = _existing_stems(artifacts_sub(self.out, "characters"), (".png", ".jpg", ".jpeg", ".webp"))
        voices = _existing_stems(artifacts_sub(self.out, "audio", "voice"), (".wav",))
        clips = _existing_stems(artifacts_sub(self.out, "clips"), (".mp4",))
        music_ok = artifacts_sub(self.out, "audio", "music.wav").exists() or artifacts_sub(
            self.out, "audio", "bgm.wav"
        ).exists()
        final_ok = (base / "final.mp4").exists()

        for sid, row in self.steps.items():
            if row.get("status") == "failed":
                continue
            stage = row["stage"]
            done = False
            if sid == "storyboard:generate":
                done = (base / "storyboard.json").exists()
            elif sid == "screenplay:validate":
                done = (base / "storyboard.json").exists()
            elif sid.startswith("images:portrait:"):
                cid = sid.split(":", 2)[-1]
                done = cid in portraits
            elif sid.startswith("images:") and sid.endswith(":end"):
                # images:s001:end → frames/s001_end.png
                shot_id = sid[len("images:") : -len(":end")]
                done = f"{shot_id}_end" in frames
            elif sid.startswith("images:") and sid != "images:_phase":
                done = sid.split(":", 1)[-1] in frames
            elif sid.startswith("audio:") and sid != "audio:_skip":
                done = sid.split(":", 1)[-1] in voices
            elif sid == "audio:_skip":
                done = self.stage_status.get("audio") == "done" or self.current_stage not in (None, "audio")
            elif sid.startswith("videos:"):
                done = sid.split(":", 1)[-1] in clips
            elif sid == "music:bgm":
                done = music_ok
            elif sid.startswith("assemble:") and sid != "assemble:mux":
                # 整理镜无独立产物；混音完成前保持，由阶段推进
                done = False
            elif sid == "assemble:mux":
                done = final_ok

            if done and row.get("status") != "done":
                row["status"] = "done"
                row["pct"] = 100.0
                row["eta_s"] = None

        # 无 tqdm 的 running 步：时间软爬到 30%
        now = time.time()
        for sid, row in self.steps.items():
            if row.get("status") != "running":
                continue
            if int(row.get("total") or 1) > 1 and int(row.get("step") or 0) > 0:
                continue  # 已有真实步数
            t0 = self._soft_started.get(sid) or now
            self._soft_started.setdefault(sid, t0)
            soft = min(30.0, (now - t0) * 2.0)  # ~15s 到 30%
            row["pct"] = max(float(row.get("pct") or 0), soft)

        # 从日志补视频/配乐/静帧 tqdm（Attached 路径）
        self._observe_logs_locked()

        # 阶段状态：按步骤汇总
        for stage in self.stages:
            ids = [i for i in self.step_order if self.steps[i]["stage"] == stage]
            if not ids:
                continue
            if all(self.steps[i]["status"] == "done" for i in ids):
                self.stage_status[stage] = "done"
            elif any(self.steps[i]["status"] == "running" for i in ids):
                self.stage_status[stage] = "running"
            elif any(self.steps[i]["status"] == "done" for i in ids) and self.stage_status.get(stage) == "pending":
                self.stage_status[stage] = "running"

    def _observe_logs_locked(self) -> None:
        logs = artifacts_sub(self.out, "logs")
        # 当前视频镜的 Denoising
        for sid in self.step_order:
            if not sid.startswith("videos:"):
                continue
            row = self.steps[sid]
            if row.get("status") == "done":
                continue
            shot = sid.split(":", 1)[-1]
            log = logs / f"{shot}.log"
            if not log.exists():
                continue
            info = self._latest_tqdm(log, desc_substr="Denoising", plan=DENOISE_PLAN)
            if not info:
                continue
            self._activate(sid)
            self._bump_locked(
                sid,
                step=info["n"],
                total=info["total"],
                pct=info["pct"],
                eta_s=info.get("eta_s"),
            )
            break

        # webui 日志里的静帧 / 配乐
        if logs.is_dir():
            cands = sorted(logs.glob("webui-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                text = self._read_tail(cands[0])
                for raw in reversed(re.split(r"[\r\n]+", text)):
                    line = raw.strip()
                    tm = _TQDM.match(line)
                    if not tm:
                        continue
                    desc = tm.group("desc").strip()
                    step_id = self._step_id_for_desc(desc)
                    if not step_id or step_id == "videos:_denoise":
                        continue
                    if step_id not in self.steps:
                        continue
                    if self.steps[step_id].get("status") == "done":
                        continue
                    n, total = int(tm.group("n")), int(tm.group("total"))
                    self._activate(step_id)
                    self._bump_locked(step_id, step=n, total=total, pct=100.0 * n / max(1, total))
                    break

        bgm_log = logs / "bgm.log"
        if bgm_log.exists() and "music:bgm" in self.steps and self.steps["music:bgm"]["status"] != "done":
            info = self._latest_tqdm(bgm_log, desc_substr="MusicGen") or self._latest_tqdm(
                bgm_log, desc_substr="配乐"
            )
            if info:
                self._activate("music:bgm")
                self._bump_locked(
                    "music:bgm",
                    step=info["n"],
                    total=info["total"],
                    pct=info["pct"],
                    eta_s=info.get("eta_s"),
                )

    def _read_tail(self, path: Path, nbytes: int = 24576) -> str:
        try:
            return path.read_bytes()[-nbytes:].decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _latest_tqdm(
        self, path: Path, *, desc_substr: str, plan: tuple[int, ...] | None = None
    ) -> dict | None:
        text = self._read_tail(path)
        if not text:
            return None
        merge = None
        if plan:
            from pipeline.progress import StageMerge

            merge = StageMerge(plan)
        eta_s = None
        n = total = 0
        saw = False
        for raw in re.split(r"[\r\n]+", text):
            line = raw.strip()
            m = _TQDM.match(line)
            if not m:
                continue
            if desc_substr not in m.group("desc"):
                continue
            saw = True
            n, total = int(m.group("n")), int(m.group("total"))
            eta_m = _TQDM_ETA.search(line)
            eta_s = _parse_hms(eta_m.group("eta")) if eta_m else None
            if merge is not None:
                merge.update(n, total)
        if not saw:
            return None
        if merge is not None:
            n, total = merge.n, merge.total or sum(plan or ())
        return {
            "n": n,
            "total": max(1, total),
            "pct": round(100.0 * n / max(1, total), 1),
            "eta_s": eta_s,
        }

    # -- lifecycle ---------------------------------------------------------
    def finalize(self, *, success: bool) -> None:
        snap = None
        with self.lock:
            self.status = "success" if success else "failed"
            for row in self.steps.values():
                st = row.get("status")
                if st == "failed":
                    continue
                if success:
                    # 成功收尾：pending（如未单独激活的混音成片）也要标完成，避免总进度 100% 却留空步
                    if st in {"pending", "running"}:
                        row["status"] = "done"
                        row["pct"] = 100.0
                        row["eta_s"] = None
                elif st == "running":
                    row["status"] = "failed"
                    row["eta_s"] = None
            if success:
                for s in self.stages:
                    self.stage_status[s] = "done"
            snap = self._emit_locked(force=True)
        if snap:
            self._notify(snap)

    def force_reset_steps(self, step_ids: list[str] | None = None) -> None:
        snap = None
        with self.lock:
            targets = step_ids or list(self.step_order)
            for sid in targets:
                if sid not in self.steps:
                    continue
                self.steps[sid]["pct"] = 0.0
                self.steps[sid]["step"] = 0
                self.steps[sid]["status"] = "pending"
                self.steps[sid]["eta_s"] = None
                self._soft_started.pop(sid, None)
            snap = self._emit_locked(force=True)
        if snap:
            self._notify(snap)
