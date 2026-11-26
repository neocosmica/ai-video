"""列出并删除本机 Hugging Face hub 里的模型缓存。

不管仓库在不在本应用的模型目录里，只要落在 hub 的 ``models--*`` 下就列出来。
当前配置和页面上还没保存的选择标成「使用中」（含 Gemma、T5 这类伴随权重）。
删哪些由调用方点名，这里不会自己把没选中的清掉。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pipeline.backends.catalog import STAGE_FIELDS, STAGE_LABELS, STAGES, download_specs, models_for, resolve_stage_model
from pipeline.config import Settings
from pipeline.hf_cache import (
    cache_nbytes,
    list_model_dirs,
    locate_repo_dir,
    remove_repo_cache,
    repo_from_cache_dir,
    resolve_snapshot,
    snapshot_ready,
)


def _active_repos(settings: Settings, keep: Mapping[str, str] | None = None) -> set[str]:
    """正在用的仓库：已保存的配置，再加上调用方传来的未保存选择。"""
    try:
        repos = {item["repo"] for item in download_specs(settings)}
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    if not keep:
        return repos
    if not isinstance(keep, Mapping):
        raise ValueError("keep 需要是阶段到模型 id 的对象")
    for stage, raw in keep.items():
        if stage not in STAGE_FIELDS:
            known = "、".join(STAGES)
            raise ValueError(f"未知阶段 {stage!r}（可选 {known}）")
        try:
            model = resolve_stage_model(str(stage), str(raw))
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc
        for item in model.downloads:
            repos.add(item.repo)
    return repos


def _catalog_by_repo() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        for model in models_for(stage):
            for weight in model.downloads:
                if weight.repo in found:
                    continue
                primary = weight.repo == model.repo
                found[weight.repo] = {
                    "label": model.label if primary else weight.label,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "id": model.id if primary else weight.id,
                    "required_files": tuple(weight.required_files),
                    "weight_globs": tuple(weight.weight_globs),
                }
    return found


def list_hub_caches(
    settings: Settings,
    *,
    hub: Path | None = None,
    keep: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """hub 里的全部模型。``in_use`` 表示当前选择还要用它。"""
    active = _active_repos(settings, keep)
    catalog = _catalog_by_repo()
    items: list[dict[str, Any]] = []
    for repo, path in list_model_dirs(hub):
        info = catalog.get(repo)
        nbytes = cache_nbytes(path)
        if info is None:
            label = repo
            stage = ""
            stage_label = ""
            item_id = repo
            ready = nbytes > 0
            known = False
        else:
            label = str(info["label"])
            stage = str(info["stage"])
            stage_label = str(info["stage_label"])
            item_id = str(info["id"])
            snap = resolve_snapshot(repo, hub=hub)
            ready, _detail = snapshot_ready(
                snap,
                required_files=tuple(info["required_files"]),
                weight_globs=tuple(info["weight_globs"]),
            )
            known = True
        items.append(
            {
                "repo": repo,
                "label": label,
                "stage": stage,
                "stage_label": stage_label,
                "id": item_id,
                "bytes": nbytes,
                "ready": ready,
                "known": known,
                "in_use": repo in active,
            }
        )
    items.sort(key=lambda item: (bool(item["in_use"]), -int(item["bytes"]), str(item["repo"])))
    unused = [item for item in items if not item["in_use"]]
    return {
        "items": items,
        "count": len(items),
        "total_bytes": sum(int(item["bytes"]) for item in items),
        "unused_count": len(unused),
        "unused_bytes": sum(int(item["bytes"]) for item in unused),
    }


def delete_selected_caches(
    repos: list[str],
    *,
    hub: Path | None = None,
) -> dict[str, Any]:
    """只删除点名的仓库。名字必须能对上 hub 里的模型目录。"""
    wanted: list[str] = []
    seen: set[str] = set()
    for raw in repos:
        repo = str(raw).strip().strip("/")
        if not repo or repo in seen:
            continue
        seen.add(repo)
        wanted.append(repo)
    if not wanted:
        raise ValueError("先选择要删除的模型")

    deleted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    freed = 0
    for repo in wanted:
        path = locate_repo_dir(repo, hub)
        if path is None or repo_from_cache_dir(path.name) != repo:
            errors.append({"repo": repo, "error": "缓存里没有这个模型"})
            continue
        try:
            nbytes = remove_repo_cache(repo, hub)
        except OSError as exc:
            errors.append({"repo": repo, "error": str(exc)})
            continue
        freed += nbytes
        deleted.append({"repo": repo, "bytes": nbytes})
    return {
        "ok": not errors,
        "deleted": deleted,
        "freed_bytes": freed,
        "errors": errors,
    }


def selftest_model_cache() -> None:
    """临时 hub：全部模型都列出来；只删点名的那些。"""
    import os
    import tempfile

    settings = Settings()
    nine = "mlx-community/Qwen3.5-9B-4bit"
    small = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    gemma = "mlx-community/gemma-3-12b-it-4bit"
    ltx_q8 = "dgrauet/ltx-2.3-mlx-q8"
    music_large = "facebook/musicgen-large"
    stranger_repo = "someone/else"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hub = root / "hub"
        hub.mkdir()
        outside = root / "outside"
        outside.mkdir()
        secret = outside / "secret.bin"
        secret.write_bytes(b"keep-me")

        def plant(repo: str, payload: bytes) -> Path:
            from pipeline.hf_cache import repo_dir_name

            folder = hub / repo_dir_name(repo)
            blobs = folder / "blobs"
            blobs.mkdir(parents=True)
            blob = blobs / "weight"
            blob.write_bytes(payload)
            os.link(blob, blobs / "weight-hardlink")
            snap = folder / "snapshots" / "rev"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").symlink_to(blob)
            refs = folder / "refs"
            refs.mkdir()
            (refs / "main").write_text("rev", encoding="utf-8")
            return folder

        plant(small, b"selected-llm")
        plant(nine, b"unused-llm-bytes")
        plant(gemma, b"shared-gemma")
        plant(ltx_q8, b"unused-ltx")
        plant(music_large, b"unused-music")
        stranger = hub / "models--someone--else"
        stranger.mkdir()
        (stranger / "w.bin").write_bytes(b"other-app")
        linked = hub / "models--mlx-community--Qwen3.5-35B-A3B-4bit"
        linked.symlink_to(outside, target_is_directory=True)
        lock = hub / ".locks" / "models--mlx-community--Qwen3.5-9B-4bit"
        lock.mkdir(parents=True)
        (lock / "lock").write_text("1", encoding="utf-8")

        nine_dir = locate_repo_dir(nine, hub)
        assert nine_dir is not None
        ref_bytes = len("rev")
        assert cache_nbytes(nine_dir) == len(b"unused-llm-bytes") + ref_bytes

        preview = list_hub_caches(settings, hub=hub)
        by_repo = {item["repo"]: item for item in preview["items"]}
        assert set(by_repo) == {nine, small, gemma, ltx_q8, music_large, stranger_repo}
        assert by_repo[small]["in_use"] is True
        assert by_repo[gemma]["in_use"] is True
        assert by_repo[nine]["in_use"] is False
        assert by_repo[nine]["known"] is True
        assert by_repo[nine]["label"] == "Qwen3.5-9B 4-bit"
        assert by_repo[nine]["ready"] is True
        assert by_repo[stranger_repo]["known"] is False
        assert by_repo[stranger_repo]["in_use"] is False
        assert by_repo[stranger_repo]["label"] == stranger_repo
        assert all(not item["in_use"] for item in preview["items"][: preview["unused_count"]])
        assert preview["unused_count"] == 4

        via_ollama = list_hub_caches(Settings(llm_model="ollama:qwen2.5:7b"), hub=hub)
        assert next(item for item in via_ollama["items"] if item["repo"] == small)["in_use"] is False
        browsing = list_hub_caches(settings, hub=hub, keep={"storyboard": "ollama:qwen2.5:7b"})
        assert next(item for item in browsing["items"] if item["repo"] == small)["in_use"] is True
        held = list_hub_caches(settings, hub=hub, keep={"video": "ltx-2.3-q8"})
        assert next(item for item in held["items"] if item["repo"] == ltx_q8)["in_use"] is True

        try:
            list_hub_caches(settings, hub=hub, keep={"nope": "x"})
            raise AssertionError("未知阶段应拒绝")
        except ValueError as exc:
            assert "未知阶段" in str(exc)

        try:
            delete_selected_caches([], hub=hub)
            raise AssertionError("空选择应拒绝")
        except ValueError as exc:
            assert "先选择" in str(exc)

        result = delete_selected_caches([nine, stranger_repo, "../etc/passwd"], hub=hub)
        gone = {item["repo"] for item in result["deleted"]}
        assert gone == {nine, stranger_repo}
        assert any(item["repo"] == "../etc/passwd" for item in result["errors"])
        assert locate_repo_dir(small, hub) is not None
        assert locate_repo_dir(gemma, hub) is not None
        assert locate_repo_dir(ltx_q8, hub) is not None
        assert locate_repo_dir(music_large, hub) is not None
        assert locate_repo_dir(nine, hub) is None
        assert not stranger.exists()
        assert not lock.exists()
        assert secret.read_bytes() == b"keep-me"
        assert linked.is_symlink()

        forced = delete_selected_caches([small], hub=hub)
        assert {item["repo"] for item in forced["deleted"]} == {small}
        assert locate_repo_dir(small, hub) is None
        assert locate_repo_dir(gemma, hub) is not None
