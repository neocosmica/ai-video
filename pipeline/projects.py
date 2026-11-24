"""项目目录约定：每部短片一个独立文件夹，与它相关的内容都放在这个文件夹里。

    projects/
      demo/
        project.json      # 项目元数据（新建时生成）
        novel.txt         # 小说原文（或 demo_novel.txt 等任何 .txt）
        storyboard.json   # 分镜（produce 会把外部分镜拷到这里）
        final.mp4         # 成片
        frames/ clips/ audio/ ...   # 流水线产物

所有项目都放在根目录的 `projects/` 下；旧布局 `input/` + `output/<片名>/`
仍能读（见 ``legacy_project`` / ``list_projects``），但新项目一律用新布局。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pipeline.util import ROOT

PROJECTS_DIRNAME = "projects"
PROJECTS_DIR = ROOT / PROJECTS_DIRNAME
PROJECT_META = "project.json"
# 过程产物子文件夹：帧、片段、配音、日志、中间合成文件都收进 <项目>/work/
WORK_DIRNAME = "work"

# 项目名：中文/英文/数字/连字符/下划线/点，不允许路径分隔符
SAFE_NAME = re.compile(r"^[\w\u4e00-\u9fff][\w\-.\u4e00-\u9fff]*$")
_LEGACY_NOVEL_RE = re.compile(r".+_novel\.txt$", re.IGNORECASE)
_LEGACY_STORYBOARD_RE = re.compile(r".+_storyboard\.json$", re.IGNORECASE)


def check_name(name: str) -> str:
    name = (name or "").strip().strip("/")
    if not name or not SAFE_NAME.match(name):
        raise SystemExit(f"项目名不合法：{name!r}（只用中英文、数字、连字符、下划线）")
    if name == "webui":
        raise SystemExit("项目名不能叫 webui：那是控制台前端文件的目录")
    return name


def projects_dir() -> Path:
    """根目录下的项目文件夹 projects/，没有就创建。"""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def project_dir(name: str) -> Path:
    """某个项目的文件夹 projects/<name>/，没有就创建。"""
    return projects_dir() / check_name(name)


def novel_path(name: str) -> Path:
    return project_dir(name) / "novel.txt"


def storyboard_path(name: str) -> Path:
    return project_dir(name) / "storyboard.json"


def legacy_project(name: str) -> Path:
    """旧布局的项目目录 output/<name>/（不再新建，仅供兼容读取）。"""
    return ROOT / "output" / check_name(name)


def artifacts_dir(out: str | Path) -> Path:
    """流水线过程产物的根目录：<out>/work/。

    `out` 可以是绝对路径，也可以是相对仓库根的路径（同 pipeline.util.resolve）。
    项目顶层只保留 novel.txt / storyboard.json / final.mp4 / project.json，
    其余一律进 work/。
    """
    base = Path(out) if Path(out).is_absolute() else ROOT / out
    return base / WORK_DIRNAME


def artifacts_sub(out: str | Path, *parts: str) -> Path:
    """产物子目录：artifacts_sub(out, "audio", "voice") → <out>/work/audio/voice。"""
    return artifacts_dir(out).joinpath(*parts)


# --force 清产物时保留：任务历史
_KEEP_WORK_NAMES = frozenset({"webui"})
# 各阶段对应的 work/ 子路径（相对 work/）与项目顶层文件
_STAGE_WORK_DIRS = {
    "images": ("frames", "characters"),
    "audio": ("audio/voice", "audio/a2v"),
    "videos": ("clips",),
    "music": (),
    "assemble": ("normalized",),
}
_STAGE_WORK_FILES = {
    "audio": ("audio/manifest.json",),
    "music": ("audio/bgm.wav", "audio/music.wav", "logs/bgm.log"),
    "assemble": ("concat.txt", "joined.mp4", "subs.ass", "soundtrack.m4a", "final.mp4"),
}
_STAGE_ROOT_FILES = {
    "assemble": ("final.mp4",),
}


def _parse_only_ids(only: str | None) -> set[str] | None:
    if not only:
        return None
    ids = {s.strip() for s in only.replace("，", ",").split(",") if s.strip()}
    return ids or None


def _rm_path(path: Path, removed: list[str], root: Path) -> None:
    if not path.exists():
        return
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(rel)


def _rm_shot_files(folder: Path, ids: set[str], removed: list[str], root: Path) -> None:
    if not folder.is_dir():
        return
    for path in list(folder.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        # s001 / s001-retry / s001_p1
        base = stem.split("-")[0].split("_")[0]
        if stem in ids or base in ids:
            _rm_path(path, removed, root)


def clear_artifacts(
    out: str | Path,
    *,
    stages: list[str] | tuple[str, ...] | None = None,
    only: str | None = None,
) -> list[str]:
    """--force 启动前清掉过程产物，避免旧帧/旧镜头掺进新片。

    不删除：novel.txt、storyboard.json、project.json、work/webui/（任务历史）。
    ``only`` 有值时只清这些镜头 id 对应的帧/音/视频文件。
    """
    base = Path(out) if Path(out).is_absolute() else ROOT / out
    work = artifacts_dir(base)
    removed: list[str] = []
    only_ids = _parse_only_ids(only)
    selected = [s for s in (stages or ()) if s and s not in {"storyboard", "screenplay"}]
    if not selected:
        selected = ["images", "audio", "videos", "music", "assemble"]

    if only_ids:
        if "images" in selected:
            _rm_shot_files(work / "frames", only_ids, removed, base)
        if "audio" in selected:
            _rm_shot_files(work / "audio" / "voice", only_ids, removed, base)
            _rm_shot_files(work / "audio" / "a2v", only_ids, removed, base)
        if "videos" in selected:
            _rm_shot_files(work / "clips", only_ids, removed, base)
            _rm_shot_files(work / "logs", only_ids, removed, base)
            _rm_shot_files(work / "normalized", only_ids, removed, base)
        if "assemble" in selected:
            for name in _STAGE_ROOT_FILES.get("assemble", ()):
                _rm_path(base / name, removed, base)
            for name in ("joined.mp4", "concat.txt", "subs.ass", "soundtrack.m4a", "final.mp4"):
                _rm_path(work / name, removed, base)
        return removed

    # 全量：勾了几乎整条流水线时，直接清空 work/（保留 webui）+ 顶层 final.mp4
    heavy = {"images", "audio", "videos", "music", "assemble"}
    if heavy.issubset(set(selected)) and work.is_dir():
        for child in list(work.iterdir()):
            if child.name in _KEEP_WORK_NAMES or child.name.startswith("."):
                continue
            _rm_path(child, removed, base)
        for name in _STAGE_ROOT_FILES.get("assemble", ()):
            _rm_path(base / name, removed, base)
        return removed

    for stage in selected:
        for rel in _STAGE_WORK_DIRS.get(stage, ()):
            _rm_path(work.joinpath(*rel.split("/")), removed, base)
        for rel in _STAGE_WORK_FILES.get(stage, ()):
            _rm_path(work.joinpath(*rel.split("/")), removed, base)
        for name in _STAGE_ROOT_FILES.get(stage, ()):
            _rm_path(base / name, removed, base)
        if stage == "videos":
            # 镜头日志跟着视频走
            logs = work / "logs"
            if logs.is_dir():
                for path in list(logs.iterdir()):
                    if path.is_file() and path.suffix == ".log" and path.name.startswith("s"):
                        _rm_path(path, removed, base)
    return removed


def strip_storyboard_suffix(stem: str) -> str:
    return re.sub(r"_?storyboard$", "", stem, flags=re.IGNORECASE)


def novel_candidates(story_path: Path) -> list[Path]:
    """给定分镜文件，猜同一个项目文件夹里的小说（按优先级）。

    优先 novel.txt，其次 <片名>_novel.txt（兼容旧命名），再退到任意 .txt。
    """
    folder = story_path.parent
    if not folder.is_dir():
        return []
    out: list[Path] = []
    preferred = folder / "novel.txt"
    if preferred.exists():
        out.append(preferred)
    stem = strip_storyboard_suffix(story_path.stem)
    legacy = folder / f"{stem}_novel.txt"
    if legacy.exists() and legacy not in out:
        out.append(legacy)
    for path in sorted(folder.glob("*.txt")):
        if path not in out:
            out.append(path)
    return out


def find_novel(directory: Path) -> Path | None:
    """项目文件夹里的小说：novel.txt 优先，其次 *_novel.txt，再退到任意 .txt。"""
    if not directory.is_dir():
        return None
    candidates = novel_candidates(directory / "storyboard.json")
    return candidates[0] if candidates else None


def write_meta(directory: Path, **fields) -> None:
    """新建/更新 project.json（存在时合并，不丢已有字段）。"""
    path = directory / PROJECT_META
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(fields)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _describe(path: Path) -> dict:
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    novel = find_novel(path)
    try:
        novel_rel = str(novel.relative_to(ROOT)) if novel else ""
    except ValueError:
        novel_rel = str(novel) if novel else ""
    return {
        "name": path.name,
        "path": rel,
        "has_final": (path / "final.mp4").exists(),
        "has_storyboard": (path / "storyboard.json").exists(),
        "novel": novel_rel,
        "has_work": has_work_dir(path),
    }


def is_artifacts_only(directory: Path) -> bool:
    """只有流水线产物、没有小说/分镜的项目文件夹（迁移时用来识别误嵌套）。"""
    return not (directory / "novel.txt").exists() and not (directory / "storyboard.json").exists()


def has_work_dir(directory: Path) -> bool:
    """项目文件夹里是否有过程产物子文件夹（work/ 或旧布局的散落产物）。"""
    if (directory / WORK_DIRNAME).is_dir():
        return True
    return any(
        (directory / name).exists()
        for name in ("frames", "clips", "audio", "characters", "logs", "normalized")
    )


def list_projects() -> list[dict]:
    """列出全部项目（新布局 projects/<片名>/ 优先，旧布局 output/<片名>/ 兼容列出）。"""
    items: list[dict] = []
    seen: set[str] = set()
    if PROJECTS_DIR.is_dir():
        for path in sorted(PROJECTS_DIR.iterdir()):
            if path.name.startswith(".") or not path.is_dir():
                continue
            # 只有产物的同名文件夹：把它当成上层项目的产物目录合并，不单列为一个项目
            if is_artifacts_only(path):
                inner = [
                    p for p in sorted(path.iterdir())
                    if p.is_dir() and not p.name.startswith(".") and not is_artifacts_only(p)
                ]
                if len(inner) == 1:
                    item = _describe(inner[0])
                    item["path"] = f"{PROJECTS_DIRNAME}/{path.name}/{inner[0].name}"
                    item["final_only"] = True
                    items.append(item)
                    seen.add(inner[0].name)
                    continue
            seen.add(path.name)
            items.append(_describe(path))
    legacy_root = ROOT / "output"
    if legacy_root.is_dir():
        for path in sorted(legacy_root.iterdir()):
            if path.name.startswith(".") or not path.is_dir() or path.name in seen:
                continue
            item = _describe(path)
            item["legacy"] = True
            items.append(item)
    return items


def create_project(name: str, *, title: str = "") -> Path:
    """新建项目文件夹并写 project.json 骨架。"""
    directory = project_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / PROJECT_META
    if not meta_path.exists():
        write_meta(directory, name=directory.name, title=title or directory.name, created_via="cli")
    return directory


def apply_project_defaults(args: argparse.Namespace) -> None:
    """--project <片名> → 补齐 --out / --novel / --storyboard 的项目文件夹默认值。

    规则（显式传参永远优先）：
    - --out 缺省 → ``projects/<片名>``；传了 --project 而 --out 还是旧默认的
      ``output/<片名>`` 时，也改到 ``projects/<片名>``；
    - --novel 缺省 → ``projects/<片名>/novel.txt``（存在才用）；
    - --storyboard 缺省 → ``projects/<片名>/storyboard.json``（存在才用）。
    """
    name = getattr(args, "project", None)
    if not name:
        return
    name = check_name(name)
    directory = project_dir(name)
    out = getattr(args, "out", None)
    if not out or out == f"output/{name}":
        args.out = f"{PROJECTS_DIRNAME}/{name}"
    if getattr(args, "novel", None) is None and (directory / "novel.txt").exists():
        args.novel = f"{PROJECTS_DIRNAME}/{name}/novel.txt"
    if getattr(args, "storyboard", None) is None and (directory / "storyboard.json").exists():
        args.storyboard = f"{PROJECTS_DIRNAME}/{name}/storyboard.json"


def cmd_projects(args: argparse.Namespace) -> None:
    new = getattr(args, "new", None)
    if new:
        directory = create_project(new, title=getattr(args, "title", "") or "")
        print(f"已新建项目：{directory}")
        print(f"  小说放这里：{directory / 'novel.txt'}")
        print(f"  分镜放这里：{directory / 'storyboard.json'}")
        print(f"  成片跑完后在：{directory / 'final.mp4'}")
        return
    items = list_projects()
    if not items:
        print(f"还没有项目。新建一个：./scripts/run.sh projects --new <片名>")
        return
    print(f"项目根目录：{PROJECTS_DIR}")
    for item in items:
        flags = []
        if item.get("novel"):
            flags.append("小说")
        if item["has_storyboard"]:
            flags.append("分镜")
        if item["has_final"]:
            flags.append("成片")
        if item.get("has_work"):
            flags.append("产物")
        legacy = "（旧布局 output/，建议迁到 projects/）" if item.get("legacy") else ""
        misplaced = "（产物在嵌套子文件夹，建议拍平到 " + f"{PROJECTS_DIRNAME}/{item['name']}/" + "）" if item.get("final_only") else ""
        print(f"- {item['name']}：{item['path']}/  [{'、'.join(flags) or '空'}]{legacy}{misplaced}")
