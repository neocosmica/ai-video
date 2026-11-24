"""项目目录约定：每部短片一个独立文件夹，与它相关的内容都放在这个文件夹里。

    ~/AI短视频项目/
      demo/
        project.json      # 项目元数据（新建时生成）
        novel.txt         # 小说原文
        storyboard.json   # 分镜
        final.mp4         # 成片
        work/             # 过程产物：frames/ clips/ audio/ ...

所有项目默认放在主目录的「AI短视频项目」下（可用 ``NOVEL_VIDEO_PROJECTS`` 覆盖）。
旧布局 ``ROOT/projects/``、``output/<片名>/`` 仍能读，但新项目一律用主目录布局。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from pipeline.util import ROOT, resolve

# 对外逻辑前缀（仅兼容旧任务记录）；新写入一律用绝对路径
PROJECTS_DIRNAME = "projects"
PROJECTS_FOLDER_NAME = "AI短视频项目"
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


_MIGRATE_DONE = False


def _projects_root() -> Path:
    override = (os.environ.get("NOVEL_VIDEO_PROJECTS") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / PROJECTS_FOLDER_NAME).resolve()


def projects_dir() -> Path:
    """用户项目根目录（默认 ~/AI短视频项目），没有就创建。

    首次访问时把仓库里旧的 ``projects/`` 迁过来（目标已有同名则跳过）。
    """
    global _MIGRATE_DONE
    root = _projects_root()
    root.mkdir(parents=True, exist_ok=True)
    if not _MIGRATE_DONE:
        _MIGRATE_DONE = True
        try:
            migrate_repo_projects(source=legacy_repo_projects_dir())
        except Exception:
            pass
    return root


def display_path(path: str | Path) -> str:
    """给人看的路径：主目录缩成 ~。"""
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(str(p)))
    home = Path.home().resolve()
    try:
        rel = p.relative_to(home)
        return f"~/{rel.as_posix()}"
    except ValueError:
        return p.as_posix()


def project_path_str(name: str) -> str:
    """项目文件夹的绝对路径字符串（接口 / 任务记录用）。"""
    return str(project_dir(name).resolve())


def project_dir(name: str) -> Path:
    """某个项目的文件夹 <项目根>/<name>/。"""
    return projects_dir() / check_name(name)


def novel_path(name: str) -> Path:
    return project_dir(name) / "novel.txt"


def storyboard_path(name: str) -> Path:
    return project_dir(name) / "storyboard.json"


def legacy_repo_projects_dir() -> Path:
    """仓库里旧的 projects/（迁移源）。"""
    return ROOT / PROJECTS_DIRNAME


def legacy_project(name: str) -> Path:
    """旧布局的项目目录 output/<name>/（不再新建，仅供兼容读取）。"""
    return ROOT / "output" / check_name(name)


def artifacts_dir(out: str | Path) -> Path:
    """流水线过程产物的根目录：<out>/work/。

    `out` 可以是绝对路径，也可以是相对路径（同 pipeline.util.resolve）。
    项目顶层只保留 novel.txt / storyboard.json / final.mp4 / project.json，
    其余一律进 work/。
    """
    return resolve(out) / WORK_DIRNAME


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
    base = resolve(out)
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
    abs_path = str(path.resolve())
    novel = find_novel(path)
    return {
        "name": path.name,
        "path": abs_path,
        "display_path": display_path(path),
        "has_final": (path / "final.mp4").exists(),
        "has_storyboard": (path / "storyboard.json").exists(),
        "novel": str(novel.resolve()) if novel else "",
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
    """列出全部项目（主目录项目根优先，旧布局 output/<片名>/ 兼容列出）。"""
    items: list[dict] = []
    seen: set[str] = set()
    root = projects_dir()
    if root.is_dir():
        for path in sorted(root.iterdir()):
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
    - --out 缺省 → 绝对路径 ``<项目根>/<片名>``；传了 --project 而 --out 还是旧默认的
      ``output/<片名>`` / ``projects/<片名>`` 时，也改到绝对路径；
    - --novel 缺省 → ``<项目>/novel.txt``（存在才用）；
    - --storyboard 缺省 → ``<项目>/storyboard.json``（存在才用）。
    """
    name = getattr(args, "project", None)
    if not name:
        return
    name = check_name(name)
    directory = project_dir(name)
    abs_out = str(directory.resolve())
    out = getattr(args, "out", None)
    if not out or out in {f"output/{name}", f"{PROJECTS_DIRNAME}/{name}"}:
        args.out = abs_out
    if getattr(args, "novel", None) is None and (directory / "novel.txt").exists():
        args.novel = str((directory / "novel.txt").resolve())
    if getattr(args, "storyboard", None) is None and (directory / "storyboard.json").exists():
        args.storyboard = str((directory / "storyboard.json").resolve())


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
    root = projects_dir()
    if not items:
        print(f"还没有项目。新建一个：./scripts/run.sh projects --new <片名>")
        print(f"项目根目录：{display_path(root)}")
        return
    print(f"项目根目录：{display_path(root)}")
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
        legacy = "（旧布局 output/，建议迁到主目录项目根）" if item.get("legacy") else ""
        shown = item.get("display_path") or item["path"]
        print(f"- {item['name']}：{shown}/  [{'、'.join(flags) or '空'}]{legacy}")


# ---------------------------------------------------------------------------
# 仓库 projects/ → ~/AI短视频项目 迁移
# ---------------------------------------------------------------------------

_TEXT_REWRITE_SUFFIXES = {".json", ".txt", ".ass", ".log", ".md"}


def _rewrite_paths_in_text(text: str, mapping: list[tuple[str, str]]) -> str:
    """按从长到短替换旧绝对路径 / projects/ 前缀。"""
    out = text
    for old, new in mapping:
        if old and old in out:
            out = out.replace(old, new)
    return out


def _rewrite_file_paths(path: Path, mapping: list[tuple[str, str]]) -> bool:
    if path.suffix.lower() not in _TEXT_REWRITE_SUFFIXES:
        return False
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    new = _rewrite_paths_in_text(raw, mapping)
    if new == raw:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def _build_path_mapping(old_dir: Path, new_dir: Path) -> list[tuple[str, str]]:
    """生成 (旧, 新) 替换对，长的在前。"""
    pairs: list[tuple[str, str]] = []
    old_abs = str(old_dir.resolve())
    new_abs = str(new_dir.resolve())
    pairs.append((old_abs, new_abs))
    # 相对逻辑前缀 projects/<name>
    rel = f"{PROJECTS_DIRNAME}/{old_dir.name}"
    pairs.append((rel + "/", new_abs + "/"))
    pairs.append((rel, new_abs))
    # 按长度降序，避免短前缀先吃掉长路径
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def migrate_repo_projects(*, source: Path | None = None) -> list[str]:
    """把仓库 ``projects/`` 下的项目移到主目录项目根，并改写路径字符串。

    目标已有同名文件夹则跳过。返回已迁移的项目名列表。
    """
    global _MIGRATE_DONE
    src_root = source or legacy_repo_projects_dir()
    if not src_root.is_dir():
        _MIGRATE_DONE = True
        return []
    dest_root = _projects_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for child in sorted(src_root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        if child.name == "webui":
            continue
        dest = dest_root / child.name
        if dest.exists():
            continue
        mapping = _build_path_mapping(child, dest)
        shutil.move(str(child), str(dest))
        # 改写迁过去之后的文本文件
        for path in dest.rglob("*"):
            if path.is_file():
                _rewrite_file_paths(path, mapping)
        moved.append(child.name)
    _MIGRATE_DONE = True
    return moved
