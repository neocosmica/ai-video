"""模型调用入口。

业务阶段（分镜校验、镜头拆分、时间轴、混音、组装）留在 ``pipeline/`` 顶层。
这里只做两件事：

1. 选定本机后端（现在只注册了 MLX；CUDA 按同样的 ``open_*`` 接进来）
2. 按用户为每个阶段选的模型，打开对应运行时

阶段模块因此不再 ``import mlx_lm`` / ``mflux`` / ``mlx_audio``。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pipeline.backends.catalog import (
    STAGE_FIELDS,
    STAGE_LABELS,
    STAGES,
    StageModel,
    choice_disk_size,
    models_for,
    resolve_stage_model,
)

BackendName = Literal["mlx", "cuda"]
KNOWN_BACKENDS = ("auto", "mlx", "cuda")

# 每个阶段目前注册了哪些 runtime。新平台加运行时模块后，把名字填到这里。
REGISTERED_RUNTIMES: dict[str, frozenset[str]] = {
    "storyboard": frozenset({"mlx_lm", "ollama"}),
    "images": frozenset({"mflux_flux2"}),
    "tts": frozenset({"mlx_tts"}),
    "music": frozenset({"musicgen"}),
    "video": frozenset({"ltx_distilled"}),
}


@dataclass(frozen=True)
class BackendInfo:
    name: BackendName
    reason: str

    @property
    def is_mlx(self) -> bool:
        return self.name == "mlx"

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def resolve_backend(config_value: str | None = None) -> BackendInfo:
    """``NOVEL_VIDEO_BACKEND`` 优先于 config.json 的 ``backend``。

    ``auto``：macOS 用 MLX。非 macOS 本应走 CUDA，但 CUDA 运行时还没注册，
    所以直接说明差什么，而不是悄悄去加载 MLX。
    """
    env = os.environ.get("NOVEL_VIDEO_BACKEND", "").strip().lower()
    want = env or (config_value or "auto").strip().lower()
    if want not in KNOWN_BACKENDS:
        raise SystemExit(f"未知 backend：{want!r}（可选 {'、'.join(KNOWN_BACKENDS)}）")
    if want == "cuda" or (want == "auto" and sys.platform != "darwin"):
        raise SystemExit(
            "当前安装只注册了 MLX 运行时（pipeline/backends 里的 mlx_*）。"
            "CUDA 要按同一套接口接入：open_storyboard、open_images、open_speech、"
            "render_music、open_video，并在 REGISTERED_RUNTIMES 里登记 runtime。"
            "在此之前请把 backend 留成 auto（macOS）或 mlx。"
        )
    if want == "mlx" and sys.platform != "darwin":
        return BackendInfo(name="mlx", reason="配置强制 mlx（MLX 仅在 macOS / Apple Silicon 可用）")
    if want == "mlx":
        return BackendInfo(name="mlx", reason="配置强制 mlx")
    return BackendInfo(name="mlx", reason="macOS：用 MLX")


def bind(settings, stage: str) -> StageModel:
    """解析这个阶段该用的模型，并确认它的运行时已经注册。"""
    if stage not in STAGE_FIELDS:
        raise SystemExit(f"未知阶段 {stage!r}")
    info = resolve_backend(getattr(settings, "backend", "auto"))
    model = resolve_stage_model(stage, getattr(settings, STAGE_FIELDS[stage], ""))
    # Ollama 只接管分镜聊天，静帧 / 对白 / 视频仍走当前后端。
    if model.backend != info.name and model.backend != "ollama":
        raise SystemExit(
            f"{STAGE_LABELS[stage]}模型 {model.id} 属于 {model.backend}，"
            f"当前后端是 {info.name}（{info.reason}）。"
        )
    if model.runtime not in REGISTERED_RUNTIMES.get(stage, frozenset()):
        registered = "、".join(sorted(REGISTERED_RUNTIMES.get(stage, ()))) or "无"
        raise SystemExit(
            f"{STAGE_LABELS[stage]}模型 {model.id} 的运行时 {model.runtime} 还没注册。"
            f"已注册：{registered}。"
        )
    return model


def open_storyboard(settings):
    """加载分镜聊天模型。用完调用 ``close()``。"""
    model = bind(settings, "storyboard")
    if model.runtime == "ollama":
        from pipeline.backends.ollama_storyboard import OllamaStoryboardLLM

        return OllamaStoryboardLLM(model)
    from pipeline.backends.mlx_storyboard import MlxStoryboardLLM

    return MlxStoryboardLLM(model)


def open_images(settings, *, kind: str, load_label: str = "文生图模型"):
    """加载静帧模型。``kind`` 为 ``t2i`` 或 ``edit``。用完调用 ``close()``。"""
    model = bind(settings, "images")
    from pipeline.backends.mlx_images import MlxFluxSession

    return MlxFluxSession(model, kind=kind, load_label=load_label)


def open_speech(settings):
    """加载对白模型。用完调用 ``close()``。"""
    model = bind(settings, "tts")
    from pipeline.backends.mlx_audio import MlxSpeech

    return MlxSpeech(model)


def render_music(settings, *, prompt: str, seconds: float, seed: int, dest: Path, log: Path) -> None:
    """按选中的配乐模型写一首 wav。"""
    model = bind(settings, "music")
    from pipeline.backends.mlx_audio import render_music as _render

    _render(settings, model, prompt=prompt, seconds=seconds, seed=seed, dest=dest, log=log)


def open_video(settings):
    """准备视频运行时（补丁、本地权重路径）。渲染用 ``render()``。"""
    model = bind(settings, "video")
    from pipeline.backends.mlx_video import MlxVideoRuntime

    return MlxVideoRuntime(settings, model)


def _weight_ready(item) -> bool:
    from pipeline.hf_cache import resolve_snapshot, snapshot_ready

    snap = resolve_snapshot(item.repo)
    ok, _detail = snapshot_ready(
        snap,
        required_files=tuple(item.required_files),
        weight_globs=tuple(item.weight_globs),
    )
    return ok


def _stage_models(stage: str, ollama_scan) -> tuple[StageModel, ...]:
    """静态目录。分镜再附上这一次 ``ollama list`` 查到的模型。"""
    catalog = models_for(stage)
    if stage != "storyboard":
        return catalog
    from pipeline.backends.ollama_storyboard import storyboard_choices

    scanned = storyboard_choices(ollama_scan)
    known = {model.id for model in catalog}
    extra = tuple(model for model in scanned if model.id not in known)
    return catalog + extra


def _prefer_local_ollama(selected: StageModel, choices: tuple[StageModel, ...]) -> StageModel:
    """配置里写了不带 tag 的名字时，对齐到本机扫到的那一条，好让下拉框选中。"""
    if selected.runtime != "ollama":
        return selected
    from pipeline.backends.ollama_storyboard import names_match

    for choice in choices:
        if choice.id == selected.id:
            return choice
    for choice in choices:
        if choice.runtime == "ollama" and names_match(selected.repo, choice.repo):
            return choice
    return selected


def _choice_ready(model: StageModel, ollama_scan) -> bool:
    if model.runtime == "ollama":
        from pipeline.backends.ollama_storyboard import model_is_local

        return model_is_local(model.repo, ollama_scan)
    if not model.downloads:
        return False
    return all(_weight_ready(item) for item in model.downloads)


def selection_view(settings) -> dict[str, Any]:
    """给网页和 doctor 的当前选择。配置写错时仍返回目录，方便改回去。"""
    try:
        info = resolve_backend(getattr(settings, "backend", "auto"))
        backend: dict[str, Any] = {"name": info.name, "reason": info.reason, "error": ""}
    except SystemExit as exc:
        backend = {"name": "", "reason": "", "error": str(exc)}

    from pipeline.backends.ollama_storyboard import scan_local_models

    ollama_scan = scan_local_models()
    stages = []
    for stage in STAGES:
        raw = str(getattr(settings, STAGE_FIELDS[stage], "") or "")
        choices_models = _stage_models(stage, ollama_scan)
        selected_runtime = ""
        try:
            selected = _prefer_local_ollama(resolve_stage_model(stage, raw), choices_models)
            selected_id = selected.id
            selected_label = selected.label
            selected_repo = selected.repo
            selected_runtime = selected.runtime
            selected_size = choice_disk_size(selected)
            error = ""
            ready = _choice_ready(selected, ollama_scan)
            if selected.runtime == "ollama" and not ready:
                error = ollama_scan.detail or f"Ollama 里没有模型 {selected.repo}。"
        except SystemExit as exc:
            selected_id = raw
            selected_label = raw
            selected_repo = raw
            selected_size = ""
            error = str(exc)
            ready = False
        choices = [
            {
                "id": model.id,
                "label": model.label,
                "repo": model.repo,
                "note": model.note,
                "runtime": model.runtime,
                "ready": _choice_ready(model, ollama_scan),
                "size": choice_disk_size(model),
            }
            for model in choices_models
        ]
        stages.append(
            {
                "stage": stage,
                "label": STAGE_LABELS[stage],
                "field": STAGE_FIELDS[stage],
                "selected": selected_id,
                "selected_label": selected_label,
                "selected_repo": selected_repo,
                "selected_size": selected_size,
                "ready": ready,
                "error": error,
                "runtime": selected_runtime,
                "choices": choices,
            }
        )
    return {"backend": backend, "stages": stages}


def save_selection(path: Path, selections: dict[str, str]) -> None:
    """把阶段选择写回 config.json。只改模型字段，其它键原样保留。值为短 id。"""
    if not selections:
        raise SystemExit("没有要保存的模型选择")
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"配置不是对象：{path}")
    unknown = [key for key in selections if key not in STAGE_FIELDS]
    if unknown:
        raise SystemExit(f"未知阶段：{'、'.join(unknown)}（可选 {'、'.join(STAGES)}）")
    for stage, raw in selections.items():
        model = resolve_stage_model(stage, raw)
        data[STAGE_FIELDS[stage]] = model.id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
