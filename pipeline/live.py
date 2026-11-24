"""流水线 live 状态：CLI produce 与网页控制台共用。

``work/webui/live.json`` 由 produce/run 写入；网页服务启动或刷新时读它，
也能用进程扫描兜底发现「先开了 produce、后开网页」的任务，并按磁盘产物
推断当前阶段与逐镜进度。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from pipeline.projects import PROJECTS_DIRNAME, artifacts_dir, artifacts_sub, project_path_str
from pipeline.util import ROOT, resolve

LIVE_FILENAME = "live.json"
STAGES = ("storyboard", "screenplay", "images", "audio", "videos", "music", "assemble")

_TQDM_LINE = re.compile(
    r"^(?P<desc>.+?):\s+\d+%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<]*)<(?P<eta>[^\],]*)"
)
_PRODUCE_CMD = re.compile(
    r"(?:python\S*\s+-m\s+pipeline|pipeline)\s+(produce|run)\b(.*)$"
)


def live_path(out: str | Path) -> Path:
    return artifacts_dir(out) / "webui" / LIVE_FILENAME


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 进程在，只是没权限发信号
    except OSError:
        return False


def read_live(out: str | Path) -> dict | None:
    path = live_path(out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_live(out: str | Path, patch: dict) -> dict:
    """合并写入 live.json；保留未知字段。"""
    path = live_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = read_live(out) or {}
    cur.update(patch)
    cur["out"] = out_rel(out)
    cur["updated_at"] = time.time()
    if "started_at" not in cur:
        cur["started_at"] = cur["updated_at"]
    if "pid" not in cur:
        cur["pid"] = os.getpid()
    if "pgid" not in cur:
        try:
            cur["pgid"] = os.getpgid(0)
        except OSError:
            cur["pgid"] = cur["pid"]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return cur


def clear_live(out: str | Path, *, status: str = "success", returncode: int | None = 0) -> None:
    """任务结束：把终态写进 live，再挪成 finished.json，避免网页误判仍在跑。"""
    path = live_path(out)
    cur = read_live(out) or {}
    cur.update(
        {
            "status": status,
            "returncode": returncode,
            "finished_at": time.time(),
            "updated_at": time.time(),
            "current_stage": None,
        }
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        finished = path.with_name("finished.json")
        finished.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        if path.exists():
            path.unlink()
    except Exception:
        pass


def mark_stage(out: str | Path, stage: str, stages: list[str] | tuple[str, ...] | None = None) -> dict:
    """进入某个阶段：之前的标 done，当前标 running。"""
    selected = list(stages or STAGES)
    if stage not in selected:
        selected = list(STAGES)
    stage_status = {s: "pending" for s in selected}
    if stage in selected:
        idx = selected.index(stage)
        for s in selected[:idx]:
            stage_status[s] = "done"
        stage_status[stage] = "running"
    return write_live(
        out,
        {
            "status": "running",
            "stages": selected,
            "stage_status": stage_status,
            "current_stage": stage,
            "source": "cli",
        },
    )


def out_rel(out: str | Path) -> str:
    """规范化项目路径：优先绝对路径（给人看、给任务记录用）。"""
    path = resolve(out)
    return str(path).replace("\\", "/")


def project_name_from_out(out: str) -> str:
    text = (out or "").replace("\\", "/").strip()
    if text.startswith(f"{PROJECTS_DIRNAME}/"):
        return text[len(PROJECTS_DIRNAME) + 1 :].split("/")[0]
    return Path(text).name


def stable_cli_task_id(out: str) -> str:
    """同一项目的 CLI produce 用稳定 id，刷新网页不会裂成多条。"""
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", project_name_from_out(out) or "job").strip("-")
    return f"cli-{slug}"[:40] or "cli-job"


def discover_produce_processes() -> list[dict]:
    """扫描本机仍在跑的 ``pipeline produce|run``。"""
    import subprocess

    try:
        raw = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    found: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "pipeline" not in line:
            continue
        # 跳过网页服务自身
        if re.search(r"pipeline\s+web\b", line):
            continue
        m = re.match(r"^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$", line)
        if not m:
            continue
        pid, ppid, pgid, cmd = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        mm = _PRODUCE_CMD.search(cmd)
        if not mm:
            continue
        kind, rest = mm.group(1), mm.group(2)
        out = _parse_out_from_argv(rest)
        if not out and kind == "produce":
            pm = re.search(r"--project\s+(\S+)", rest)
            if pm:
                out = project_path_str(pm.group(1).strip().strip("'\""))
        if not out:
            continue
        found.append(
            {
                "pid": pid,
                "ppid": ppid,
                "pgid": pgid,
                "cmd": cmd,
                "out": out_rel(out),
                "kind": kind,
            }
        )
    # 同一 out 可能同时有 uv run 父进程与 python -m 子进程：留叶子（被别人当父的丢掉）
    pids = {f["pid"] for f in found}
    leaves = [f for f in found if f["pid"] not in {x["ppid"] for x in found} or f["pid"] not in pids]
    # 更稳：每个 out 只留 command 含 `python` 且 `-m pipeline` 的那个
    by_out: dict[str, dict] = {}
    for f in found:
        cur = by_out.get(f["out"])
        score = (1 if "-m pipeline" in f["cmd"] else 0) + (1 if "python" in f["cmd"] else 0)
        if cur is None or score >= cur.get("_score", -1):
            f = dict(f)
            f["_score"] = score
            by_out[f["out"]] = f
    return [{k: v for k, v in f.items() if k != "_score"} for f in by_out.values()]


def _parse_out_from_argv(rest: str) -> str | None:
    m = re.search(r"--out\s+(\S+)", rest)
    if m:
        return out_rel(m.group(1).strip().strip("'\""))
    m = re.search(r"--project\s+(\S+)", rest)
    if m:
        return project_path_str(m.group(1).strip().strip("'\""))
    return None


def shot_ids(out: str | Path) -> list[str]:
    path = resolve(out) / "storyboard.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(s.get("id") or "").strip() for s in (data.get("shots") or []) if str(s.get("id") or "").strip()]


def _existing_stems(folder: Path, exts: tuple[str, ...]) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts}


def infer_stage_status(out: str | Path, stages: list[str] | None = None) -> dict[str, str]:
    """按磁盘产物粗判各阶段：有成片→全 done；否则按帧/音/视频/配乐推进。"""
    selected = list(stages or STAGES)
    base = resolve(out)
    ids = shot_ids(out)
    frames = _existing_stems(artifacts_sub(out, "frames"), (".png", ".jpg", ".jpeg", ".webp"))
    clips = _existing_stems(artifacts_sub(out, "clips"), (".mp4",))
    voices = _existing_stems(artifacts_sub(out, "audio", "voice"), (".wav",))
    music = artifacts_sub(out, "audio", "music.wav")
    final = base / "final.mp4"
    dialogue_ids = []
    try:
        data = json.loads((base / "storyboard.json").read_text(encoding="utf-8"))
        dialogue_ids = [
            str(s.get("id") or "")
            for s in (data.get("shots") or [])
            if str(s.get("dialogue") or "").strip() and str(s.get("id") or "").strip()
        ]
    except Exception:
        dialogue_ids = []

    sb_ok = (base / "storyboard.json").exists()
    done_flags = {
        "storyboard": sb_ok,
        "screenplay": sb_ok,
        "images": bool(ids) and all(i in frames for i in ids),
        "audio": (not dialogue_ids) or all(i in voices for i in dialogue_ids),
        "videos": bool(ids) and all(i in clips for i in ids),
        "music": music.exists(),
        "assemble": final.exists(),
    }
    # 部分完成也算「做过」：有任一产物则 storyboard/screenplay/images 至少开始过
    started = {
        "storyboard": done_flags["storyboard"],
        "screenplay": done_flags["screenplay"],
        "images": bool(frames) or done_flags["images"],
        "audio": bool(voices) or done_flags["audio"],
        "videos": bool(clips) or done_flags["videos"],
        "music": done_flags["music"],
        "assemble": done_flags["assemble"],
    }
    status: dict[str, str] = {s: "pending" for s in selected}
    current = None
    for s in selected:
        if done_flags.get(s):
            status[s] = "done"
        elif started.get(s) or current is None:
            if current is None and not done_flags.get(s):
                # 找第一个未完成的作为 running
                current = s
                status[s] = "running"
                break
    if current is None and selected:
        # 全完成
        for s in selected:
            status[s] = "done"
    else:
        # current 之后保持 pending；之前未标 done 的补 done（跳过空阶段）
        if current in selected:
            idx = selected.index(current)
            for s in selected[:idx]:
                if status[s] != "done":
                    status[s] = "done"
    return status


def _parse_hms(text: str) -> float | None:
    text = (text or "").strip()
    m = re.match(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$", text)
    if not m:
        return None
    h, mi, s = (g or "0" for g in m.groups())
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _read_log_tail(path: Path, nbytes: int = 24576) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()[-nbytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _latest_tqdm_from_text(
    text: str,
    *,
    desc_substr: str | None = None,
    desc_regex: re.Pattern[str] | None = None,
    plan: tuple[int, ...] | None = None,
) -> dict | None:
    """从日志文本里取最后一条匹配的 tqdm；可选按 LTX plan 合并多段。"""
    if not text:
        return None
    merge = None
    if plan:
        from pipeline.progress import StageMerge

        merge = StageMerge(plan)
    eta_s = None
    desc = None
    n = total = 0
    saw = False
    for raw in re.split(r"[\r\n]+", text):
        line = raw.strip()
        m = _TQDM_LINE.match(line)
        if not m:
            continue
        d = m.group("desc").strip()
        if desc_substr and desc_substr not in d:
            continue
        if desc_regex and not desc_regex.search(d):
            continue
        saw = True
        desc = d
        n, total = int(m.group("n")), int(m.group("total"))
        eta_s = _parse_hms(m.group("eta"))
        if merge is not None:
            merge.update(n, total)
    if not saw:
        return None
    if merge is not None:
        n, total = merge.n, merge.total or sum(plan or ())
    return {
        "desc": desc or "",
        "n": n,
        "total": max(1, total),
        "pct": round(100.0 * n / max(1, total), 1),
        "eta_s": eta_s,
        "status": "done" if n >= total else "running",
    }


def _latest_tqdm_from_log(
    path: Path,
    *,
    desc_substr: str | None = None,
    desc_regex: re.Pattern[str] | None = None,
    plan: tuple[int, ...] | None = None,
) -> dict | None:
    return _latest_tqdm_from_text(
        _read_log_tail(path),
        desc_substr=desc_substr,
        desc_regex=desc_regex,
        plan=plan,
    )


def _latest_webui_log(out: str | Path) -> Path | None:
    logs = artifacts_sub(out, "logs")
    if not logs.is_dir():
        return None
    cands = sorted(logs.glob("webui-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


_IMAGE_TQDM_DESC = re.compile(r"^(静帧|空镜)\s+(?P<id>\S+)$|^(肖像)\s+(?P<name>.+)$")


def _story_meta(out: str | Path) -> dict:
    """分镜里的镜头 id、对白镜头、角色 id。"""
    base = resolve(out)
    ids = shot_ids(out)
    dialogue_ids: list[str] = []
    char_ids: list[str] = []
    try:
        data = json.loads((base / "storyboard.json").read_text(encoding="utf-8"))
        dialogue_ids = [
            str(s.get("id") or "")
            for s in (data.get("shots") or [])
            if str(s.get("dialogue") or "").strip() and str(s.get("id") or "").strip()
        ]
        char_ids = [str(c.get("id") or "").strip() for c in (data.get("characters") or []) if str(c.get("id") or "").strip()]
    except Exception:
        dialogue_ids = [i for i in ids]
    return {"ids": ids, "dialogue_ids": dialogue_ids, "char_ids": char_ids}


def _pct(done: float, total: int) -> float:
    if total <= 0:
        return 100.0
    if done >= total:
        return 100.0
    # 进行中封顶 99.9，留一点给「阶段尚未标 done」
    return round(min(99.9, 100.0 * done / total), 1)


def infer_progress(out: str | Path, stage: str | None, *, live_item: dict | None = None) -> dict:
    """根据产物 + 日志 tqdm 拼细粒度 progress（可把网页 live 条目的步数并进来）。"""
    ids = shot_ids(out)
    items: list[dict] = []
    current = None
    stage_pct: dict[str, float] = {}
    if not stage:
        return {"items": items, "stage_pct": stage_pct, "current": current}
    meta = _story_meta(out)

    if stage == "storyboard":
        sb = resolve(out) / "storyboard.json"
        stage_pct["storyboard"] = 100.0 if sb.exists() else 20.0
        current = "生成分镜"
        items.append(
            {
                "id": "misc:storyboard",
                "label": "生成分镜",
                "pct": stage_pct["storyboard"],
                "step": 1 if sb.exists() else 0,
                "total": 1,
                "eta_s": None,
                "status": "done" if sb.exists() else "running",
            }
        )

    elif stage == "screenplay":
        sb = resolve(out) / "storyboard.json"
        stage_pct["screenplay"] = 100.0 if sb.exists() else 30.0
        current = "校验分镜"
        items.append(
            {
                "id": "misc:screenplay",
                "label": "校验分镜",
                "pct": stage_pct["screenplay"],
                "step": 1 if sb.exists() else 0,
                "total": 1,
                "eta_s": None,
                "status": "done" if sb.exists() else "running",
            }
        )

    elif stage == "images":
        frames = _existing_stems(artifacts_sub(out, "frames"), (".png", ".jpg", ".jpeg", ".webp"))
        chars = _existing_stems(artifacts_sub(out, "characters"), (".png", ".jpg", ".jpeg", ".webp"))
        char_ids = meta["char_ids"]
        unit_total = len(char_ids) + len(ids)
        done_chars = sum(1 for c in char_ids if c in chars)
        done_frames = sum(1 for sid in ids if sid in frames)
        done_n = done_chars + done_frames
        # 当前条：优先网页 live；否则扫最新 webui 日志里的 静帧/空镜/肖像 tqdm
        tqdm_info = None
        label = None
        item_id = None
        if live_item and int(live_item.get("total") or 0) > 1:
            tqdm_info = {
                "n": int(live_item.get("step") or 0),
                "total": int(live_item.get("total") or 1),
                "pct": float(live_item.get("pct") or 0),
                "eta_s": live_item.get("eta_s"),
            }
            label = live_item.get("label") or "静帧"
            item_id = live_item.get("id") or f"shot:{label}"
        else:
            webui = _latest_webui_log(out)
            if webui:
                tqdm_info = _latest_tqdm_from_log(webui, desc_regex=_IMAGE_TQDM_DESC)
                if tqdm_info:
                    label = tqdm_info["desc"]
                    mm = _IMAGE_TQDM_DESC.match(label)
                    if mm and mm.groupdict().get("id"):
                        item_id = f"shot:{mm.group('id')}"
                    elif mm and mm.groupdict().get("name"):
                        item_id = f"misc:肖像 {mm.group('name')}"
                    else:
                        item_id = f"misc:{label}"
        frac = 0.0
        if tqdm_info and tqdm_info.get("status") != "done":
            frac = tqdm_info["n"] / max(1, tqdm_info["total"])
            items.append(
                {
                    "id": item_id or "shot:image",
                    "label": label or "静帧",
                    "pct": tqdm_info["pct"],
                    "step": tqdm_info["n"],
                    "total": tqdm_info["total"],
                    "eta_s": tqdm_info.get("eta_s"),
                    "status": "running",
                }
            )
            current = label
        elif done_n < unit_total:
            # 还没拿到步数：标下一个缺的产物
            missing_char = next((c for c in char_ids if c not in chars), None)
            missing_shot = next((s for s in ids if s not in frames), None)
            if missing_char:
                label = f"肖像 {missing_char}"
                item_id = f"misc:{label}"
            else:
                label = missing_shot or "静帧"
                item_id = f"shot:{label}"
            items.append(
                {
                    "id": item_id,
                    "label": label,
                    "pct": 0.0,
                    "step": 0,
                    "total": 1,
                    "eta_s": None,
                    "status": "running",
                }
            )
            current = label
        stage_pct["images"] = _pct(done_n + frac, max(1, unit_total)) if unit_total else 100.0

    elif stage == "audio":
        voices = _existing_stems(artifacts_sub(out, "audio", "voice"), (".wav",))
        want = meta["dialogue_ids"] or ids
        done_n = sum(1 for sid in want if sid in voices)
        running_sid = next((sid for sid in want if sid not in voices), None)
        if running_sid:
            items.append(
                {
                    "id": f"audio:{running_sid}",
                    "label": running_sid,
                    "pct": 0.0,
                    "step": 0,
                    "total": 1,
                    "eta_s": None,
                    "status": "running",
                }
            )
            current = f"对白 {running_sid}"
        # TTS 无步数：用「已完成条数 + 当前半格」让总条缓慢爬
        frac = 0.35 if running_sid else 0.0
        stage_pct["audio"] = _pct(done_n + frac, max(1, len(want))) if want else 100.0

    elif stage == "videos":
        clips = _existing_stems(artifacts_sub(out, "clips"), (".mp4",))
        done_n = sum(1 for sid in ids if sid in clips)
        running_sid = next((sid for sid in ids if sid not in clips), None)
        tqdm_info = None
        if live_item and int(live_item.get("total") or 0) > 1:
            tqdm_info = {
                "n": int(live_item.get("step") or 0),
                "total": int(live_item.get("total") or 1),
                "pct": float(live_item.get("pct") or 0),
                "eta_s": live_item.get("eta_s"),
            }
        elif running_sid:
            tqdm_info = _latest_tqdm_from_log(
                artifacts_sub(out, "logs", f"{running_sid}.log"),
                desc_substr="Denoising",
                plan=(8, 3),
            )
        if running_sid:
            if tqdm_info:
                items.append(
                    {
                        "id": f"shot:{running_sid}",
                        "label": f"{running_sid} a2v",
                        "pct": tqdm_info["pct"],
                        "step": tqdm_info["n"],
                        "total": tqdm_info["total"],
                        "eta_s": tqdm_info.get("eta_s"),
                        "status": "running",
                    }
                )
                current = f"{running_sid} a2v"
            else:
                items.append(
                    {
                        "id": f"shot:{running_sid}",
                        "label": running_sid,
                        "pct": 0.0,
                        "step": 0,
                        "total": 1,
                        "eta_s": None,
                        "status": "running",
                    }
                )
                current = running_sid
        frac = (tqdm_info["n"] / max(1, tqdm_info["total"])) if tqdm_info else (0.05 if running_sid else 0.0)
        stage_pct["videos"] = _pct(done_n + frac, max(1, len(ids))) if ids else 100.0

    elif stage == "music":
        # 配乐文件可能叫 music.wav 或 bgm.wav
        music = artifacts_sub(out, "audio", "music.wav")
        bgm = artifacts_sub(out, "audio", "bgm.wav")
        done = music.exists() or bgm.exists()
        tqdm_info = None
        if live_item and int(live_item.get("total") or 0) > 1:
            tqdm_info = {
                "n": int(live_item.get("step") or 0),
                "total": int(live_item.get("total") or 1),
                "pct": float(live_item.get("pct") or 0),
                "eta_s": live_item.get("eta_s"),
            }
        if not done:
            tqdm_info = tqdm_info or _latest_tqdm_from_log(
                artifacts_sub(out, "logs", "bgm.log"),
                desc_substr="MusicGen",
            )
            # ShotBar 合并后的「配乐」条也可能在 webui 日志里
            if not tqdm_info:
                webui = _latest_webui_log(out)
                if webui:
                    tqdm_info = _latest_tqdm_from_log(webui, desc_substr="配乐") or _latest_tqdm_from_log(
                        webui, desc_substr="MusicGen"
                    )
        if done:
            items.append(
                {
                    "id": "misc:配乐",
                    "label": "配乐",
                    "pct": 100.0,
                    "step": 1,
                    "total": 1,
                    "eta_s": None,
                    "status": "done",
                }
            )
            stage_pct["music"] = 100.0
            current = "配乐"
        elif tqdm_info:
            items.append(
                {
                    "id": "misc:配乐",
                    "label": "配乐",
                    "pct": tqdm_info["pct"],
                    "step": tqdm_info["n"],
                    "total": tqdm_info["total"],
                    "eta_s": tqdm_info.get("eta_s"),
                    "status": "running",
                }
            )
            stage_pct["music"] = min(99.0, float(tqdm_info["pct"]))
            current = "配乐"
        else:
            items.append(
                {
                    "id": "misc:配乐",
                    "label": "配乐",
                    "pct": 5.0,
                    "step": 0,
                    "total": 1,
                    "eta_s": None,
                    "status": "running",
                }
            )
            stage_pct["music"] = 5.0
            current = "配乐"

    elif stage == "assemble":
        final = resolve(out) / "final.mp4"
        norms = _existing_stems(artifacts_sub(out, "normalized"), (".mp4",))
        done_n = sum(1 for sid in ids if sid in norms)
        if final.exists():
            stage_pct["assemble"] = 100.0
            current = "合成混音"
            items.append(
                {
                    "id": "misc:合成",
                    "label": "合成混音",
                    "pct": 100.0,
                    "step": len(ids),
                    "total": max(1, len(ids)),
                    "eta_s": None,
                    "status": "done",
                }
            )
        else:
            running_sid = next((sid for sid in ids if sid not in norms), None)
            # 整理很快：用已整理镜头数；全部整理完后等 ffmpeg 混音再给到 95
            if running_sid:
                items.append(
                    {
                        "id": f"assemble:{running_sid}",
                        "label": f"整理 {running_sid}",
                        "pct": 0.0,
                        "step": done_n,
                        "total": max(1, len(ids)),
                        "eta_s": None,
                        "status": "running",
                    }
                )
                current = f"整理 {running_sid}"
                stage_pct["assemble"] = _pct(done_n + 0.5, max(1, len(ids)))
            else:
                items.append(
                    {
                        "id": "misc:合成",
                        "label": "合成混音",
                        "pct": 90.0,
                        "step": max(1, len(ids)),
                        "total": max(1, len(ids)),
                        "eta_s": None,
                        "status": "running",
                    }
                )
                current = "合成混音"
                stage_pct["assemble"] = 92.0

    return {"items": items, "stage_pct": stage_pct, "current": current}


def build_live_snapshot(out: str, *, proc: dict | None = None, live: dict | None = None) -> dict:
    """拼一份可供网页 Job.snapshot 对齐的任务快照。"""
    live = live or read_live(out) or {}
    stages = list(live.get("stages") or STAGES)
    stage_status = live.get("stage_status") if isinstance(live.get("stage_status"), dict) else None
    if not stage_status:
        stage_status = infer_stage_status(out, stages)
    # 进程还在但 live 过期时，用磁盘再推一次
    if proc and pid_alive(proc.get("pid")):
        stage_status = infer_stage_status(out, stages)
    current = next((s for s in stages if stage_status.get(s) == "running"), None)
    progress = infer_progress(out, current)
    pid = (proc or {}).get("pid") or live.get("pid")
    started = live.get("started_at") or (proc and time.time()) or time.time()
    return {
        "id": live.get("task_id") or stable_cli_task_id(out),
        "name": project_name_from_out(out) or out,
        "novel": live.get("novel") or f"{out}/novel.txt",
        "out": out,
        "only": live.get("only"),
        "detail": live.get("detail") or "CLI produce（网页自动挂载）",
        "status": "running" if pid_alive(pid) else (live.get("status") or "interrupted"),
        "stages": [{"key": s, "label": s, "status": stage_status.get(s, "pending")} for s in stages],
        "stage_status": stage_status,
        "current_stage": current,
        "progress": progress,
        "returncode": live.get("returncode"),
        "error": None if pid_alive(pid) else (live.get("error") or "进程已结束"),
        "created_at": started,
        "started_at": started,
        "finished_at": live.get("finished_at"),
        "log_path": str(artifacts_sub(out, "logs")),
        "media_base": out,
        "pid": pid,
        "pgid": (proc or {}).get("pgid") or live.get("pgid"),
        "source": live.get("source") or "cli",
    }
