"""各阶段可选模型目录。

业务代码不写死仓库名。配置里的 ``llm_model`` / ``image_model`` /
``tts_model`` / ``music_model`` / ``video_model`` 填这里的短 id，
或填原来的 Hugging Face repo（别名，旧 config.json 不用改）。

换平台时新增一条 ``backend`` + ``runtime`` 不同的目录项，并在
``pipeline/backends`` 里注册对应的 ``open_*``。同一阶段可以并存多条，
用户选哪条就加载哪条；下载清单只含当前选中的权重。

分镜还可以写成 ``ollama:<模型名>``（例如 ``ollama:qwen2.5:7b``）。
这类条目不在下面的静态目录里，由本机已下载的 Ollama 模型现扫出来，
推理走 Ollama，不进 Hugging Face 下载清单。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any


STAGES: tuple[str, ...] = ("storyboard", "images", "tts", "music", "video")

STAGE_LABELS: dict[str, str] = {
    "storyboard": "分镜",
    "images": "静帧",
    "tts": "对白",
    "music": "配乐",
    "video": "视频",
}

# Settings 上的字段。改配置、写 config.json 都走这张表。
STAGE_FIELDS: dict[str, str] = {
    "storyboard": "llm_model",
    "images": "image_model",
    "tts": "tts_model",
    "music": "music_model",
    "video": "video_model",
}


@dataclass(frozen=True)
class WeightSpec:
    """一份要下载 / 校验的权重。"""

    id: str
    label: str
    repo: str
    allow_patterns: tuple[str, ...] | None = None
    required_files: tuple[str, ...] = ()
    weight_globs: tuple[str, ...] = ()
    # 下载前估磁盘。0 表示自定义仓库，体积未知。
    approx_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "repo": self.repo,
            "allow_patterns": list(self.allow_patterns) if self.allow_patterns else None,
            "required_files": self.required_files,
            "weight_globs": self.weight_globs,
            "approx_bytes": self.approx_bytes,
        }


@dataclass(frozen=True)
class StageModel:
    """一个阶段里的一个可选项。``runtime`` 决定调用哪套加载代码。"""

    stage: str
    id: str
    label: str
    backend: str
    runtime: str
    repo: str
    downloads: tuple[WeightSpec, ...]
    aliases: tuple[str, ...] = ()
    note: str = ""
    # 同 runtime 的自定义 repo（mlx-lm / CLI --model）。图像加载器绑死了架构，不能自定义。
    accepts_custom_repo: bool = False
    # mflux ModelConfig 的方法名。空则沿用 Klein 4B。
    variant: str = ""


_LTX_ALLOW = (
    "transformer-distilled-1.1.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "spatial_upscaler_x2_v1_1_config.json",
    "vae_decoder.safetensors",
    "vae_encoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "connector.safetensors",
    "config.json",
    "embedded_config.json",
    "quantize_config.json",
    "split_model.json",
    "LICENSE",
    "README.md",
    ".gitattributes",
)
_LTX_REQUIRED = ("audio_vae.safetensors", "vocoder.safetensors", "config.json")

# LTX-2.5 distilled：Gemma 4 在包内，文件名和 2.3 不同。
_LTX25_ALLOW = (
    "transformer-distilled.safetensors",
    "text_encoder.safetensors",
    "text_encoder_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "spatial_upscaler_x2_v1_0.safetensors",
    "spatial_upscaler_x2_v1_0_config.json",
    "temporal_upscaler_x2_v1_0.safetensors",
    "temporal_upscaler_x2_v1_0_config.json",
    "vae_decoder_conv.safetensors",
    "vae_encoder_conv.safetensors",
    "vae_decoder_av.safetensors",
    "vae_encoder_av.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "connector.safetensors",
    "duration_head.safetensors",
    "embedded_config.json",
    "config.json",
    "quantize_config.json",
    "split_model.json",
    "LICENSE",
    "README.md",
    ".gitattributes",
)
_LTX25_REQUIRED = (
    "transformer-distilled.safetensors",
    "text_encoder.safetensors",
    "text_encoder_config.json",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "connector.safetensors",
    "vae_decoder_conv.safetensors",
    "spatial_upscaler_x2_v1_0.safetensors",
)

def _gb(n: float) -> int:
    """按 GiB 计的磁盘预估，略大于实测，避免估小了才开始下载。"""
    return int(round(n * 1024**3))


_GEMMA = WeightSpec(
    id="gemma",
    label="视频提示词编码器 Gemma 3 12B",
    repo="mlx-community/gemma-3-12b-it-4bit",
    weight_globs=("model*.safetensors",),
    approx_bytes=_gb(8.0),
)

_MUSIC_ALLOW = (
    "state_dict.bin",
    "compression_state_dict.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "generation_config.json",
    "preprocessor_config.json",
    "README.md",
    ".gitattributes",
)
_MUSIC_REQUIRED = ("state_dict.bin", "compression_state_dict.bin")

_T5 = WeightSpec(
    id="t5",
    label="配乐条件编码器 T5-base",
    repo="t5-base",
    allow_patterns=(
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "spiece.model",
    ),
    required_files=("model.safetensors", "config.json"),
    approx_bytes=_gb(0.95),
)


def _music(model_id: str, label: str, repo: str, note: str, *, approx: int) -> StageModel:
    return StageModel(
        stage="music",
        id=model_id,
        label=label,
        backend="mlx",
        runtime="musicgen",
        repo=repo,
        aliases=(repo,),
        note=note,
        accepts_custom_repo=True,
        downloads=(
            WeightSpec(
                id="musicgen",
                label=label,
                repo=repo,
                allow_patterns=_MUSIC_ALLOW,
                required_files=_MUSIC_REQUIRED,
                approx_bytes=approx,
            ),
            _T5,
        ),
    )


def _ltx(
    model_id: str,
    label: str,
    repo: str,
    note: str,
    *,
    approx: int,
    allow: tuple[str, ...] = _LTX_ALLOW,
    required: tuple[str, ...] = _LTX_REQUIRED,
    gemma: bool = True,
) -> StageModel:
    downloads = [
        WeightSpec(
            id="ltx",
            label=label,
            repo=repo,
            allow_patterns=allow,
            required_files=required,
            approx_bytes=approx,
        )
    ]
    if gemma:
        downloads.append(_GEMMA)
    return StageModel(
        stage="video",
        id=model_id,
        label=label,
        backend="mlx",
        runtime="ltx_distilled",
        repo=repo,
        aliases=(repo,),
        note=note,
        accepts_custom_repo=True,
        downloads=tuple(downloads),
    )


_CATALOG: dict[str, tuple[StageModel, ...]] = {
    "storyboard": (
        StageModel(
            stage="storyboard",
            id="qwen3-4b-instruct-4bit",
            label="Qwen3-4B Instruct 4-bit",
            backend="mlx",
            runtime="mlx_lm",
            repo="mlx-community/Qwen3-4B-Instruct-2507-4bit",
            aliases=("mlx-community/Qwen3-4B-Instruct-2507-4bit",),
            note="默认。16 GB 统一内存可跑",
            accepts_custom_repo=True,
            downloads=(
                WeightSpec(
                    id="llm",
                    label="分镜语言模型 Qwen3-4B",
                    repo="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    weight_globs=("model*.safetensors",),
                    approx_bytes=_gb(2.4),
                ),
            ),
        ),
        StageModel(
            stage="storyboard",
            id="qwen3.5-35b-a3b-4bit",
            label="Qwen3.5-35B-A3B 4-bit",
            backend="mlx",
            runtime="mlx_lm",
            repo="mlx-community/Qwen3.5-35B-A3B-4bit",
            aliases=("mlx-community/Qwen3.5-35B-A3B-4bit",),
            note="32 GB 推荐。Qwen3.5 MoE",
            accepts_custom_repo=True,
            downloads=(
                WeightSpec(
                    id="llm",
                    label="分镜语言模型 Qwen3.5-35B-A3B",
                    repo="mlx-community/Qwen3.5-35B-A3B-4bit",
                    weight_globs=("model*.safetensors",),
                    approx_bytes=_gb(21),
                ),
            ),
        ),
        StageModel(
            stage="storyboard",
            id="qwen3.5-9b-4bit",
            label="Qwen3.5-9B 4-bit",
            backend="mlx",
            runtime="mlx_lm",
            repo="mlx-community/Qwen3.5-9B-4bit",
            aliases=("mlx-community/Qwen3.5-9B-4bit",),
            note="Qwen3.5 9B",
            accepts_custom_repo=True,
            downloads=(
                WeightSpec(
                    id="llm",
                    label="分镜语言模型 Qwen3.5-9B",
                    repo="mlx-community/Qwen3.5-9B-4bit",
                    weight_globs=("model*.safetensors",),
                    approx_bytes=_gb(6.4),
                ),
            ),
        ),
    ),
    "images": (
        StageModel(
            stage="images",
            id="flux2-klein-4b",
            label="FLUX.2 Klein 4B 4-bit",
            backend="mlx",
            runtime="mflux_flux2",
            repo="mflux-community/flux2-klein-4b-mflux-q4",
            aliases=("mflux-community/flux2-klein-4b-mflux-q4",),
            note="默认。mflux 预量化权重，肖像 / 静帧 / 尾帧共用",
            downloads=(
                WeightSpec(
                    id="flux",
                    label="文生图 FLUX.2 Klein 4B",
                    repo="mflux-community/flux2-klein-4b-mflux-q4",
                    approx_bytes=_gb(4.6),
                ),
            ),
        ),
        StageModel(
            stage="images",
            id="flux2-klein-9b-q8",
            label="FLUX.2 Klein 9B 8-bit",
            backend="mlx",
            runtime="mflux_flux2",
            repo="mflux-community/flux2-klein-9b-mflux-q8",
            aliases=("mflux-community/flux2-klein-9b-mflux-q8",),
            note="32 GB 推荐。Klein 9B 8-bit，肖像 / 静帧 / 尾帧共用",
            variant="flux2_klein_9b",
            downloads=(
                WeightSpec(
                    id="flux",
                    label="文生图 FLUX.2 Klein 9B",
                    repo="mflux-community/flux2-klein-9b-mflux-q8",
                    approx_bytes=_gb(19),
                ),
            ),
        ),
    ),
    "tts": (
        StageModel(
            stage="tts",
            id="qwen3-tts-1.7b-customvoice-8bit",
            label="Qwen3-TTS 1.7B CustomVoice 8-bit",
            backend="mlx",
            runtime="mlx_tts",
            repo="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            aliases=("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",),
            note="默认。音色 Serena / Vivian / Dylan / Uncle_Fu / Eric",
            accepts_custom_repo=True,
            downloads=(
                WeightSpec(
                    id="tts",
                    label="对白 TTS Qwen3-TTS",
                    repo="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    weight_globs=("model*.safetensors",),
                    approx_bytes=_gb(3.2),
                ),
            ),
        ),
        StageModel(
            stage="tts",
            id="qwen3-tts-1.7b-customvoice-bf16",
            label="Qwen3-TTS 1.7B CustomVoice bf16",
            backend="mlx",
            runtime="mlx_tts",
            repo="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            aliases=("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",),
            note="32 GB 推荐。同一套音色，bf16",
            accepts_custom_repo=True,
            downloads=(
                WeightSpec(
                    id="tts",
                    label="对白 TTS Qwen3-TTS bf16",
                    repo="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                    weight_globs=("model*.safetensors",),
                    approx_bytes=_gb(4.8),
                ),
            ),
        ),
    ),
    "music": (
        _music(
            "musicgen-small",
            "MusicGen small",
            "facebook/musicgen-small",
            "默认。16 GB 机器优先用这个",
            approx=_gb(1.3),
        ),
        _music(
            "musicgen-medium",
            "MusicGen medium",
            "facebook/musicgen-medium",
            "同一套 MusicGen 运行时，模型更大，内存占用更高",
            approx=_gb(4.2),
        ),
        _music(
            "musicgen-large",
            "MusicGen large",
            "facebook/musicgen-large",
            "32 GB 推荐。单声道 large",
            approx=_gb(7.2),
        ),
    ),
    "video": (
        _ltx(
            "ltx-2.3-q4",
            "LTX-2.3 4-bit distilled",
            "dgrauet/ltx-2.3-mlx-q4",
            "默认。16 GB 统一内存 + low-ram",
            approx=_gb(20),
        ),
        _ltx(
            "ltx-2.3-q8",
            "LTX-2.3 8-bit distilled",
            "dgrauet/ltx-2.3-mlx-q8",
            "32 GB 上的 2.3 高画质档",
            approx=_gb(27),
        ),
        _ltx(
            "ltx-2.5-q4",
            "LTX-2.5 4-bit distilled",
            "reverie-studio/ltx-2.5-mlx-q4-distilled",
            "32 GB 推荐。Gemma 4 在包内，国内镜像可直接下载",
            allow=_LTX25_ALLOW,
            required=_LTX25_REQUIRED,
            gemma=False,
            approx=_gb(32),
        ),
    ),
}


def choice_disk_size(model: StageModel) -> str:
    """这个选项的权重大约占多少磁盘。伴随的 Gemma、T5 单独写出来。"""
    from pipeline.disk_space import format_bytes

    parts = [item for item in model.downloads if item.approx_bytes > 0]
    if not parts:
        return ""
    if len(parts) == 1:
        return format_bytes(parts[0].approx_bytes)
    names = {"gemma": "Gemma", "t5": "T5"}
    main = format_bytes(parts[0].approx_bytes)
    rest = " + ".join(
        f"{names.get(item.id, item.label)} {format_bytes(item.approx_bytes)}" for item in parts[1:]
    )
    return f"{main} + {rest}"


def models_for(stage: str) -> tuple[StageModel, ...]:
    try:
        return _CATALOG[stage]
    except KeyError as exc:
        known = "、".join(STAGES)
        raise SystemExit(f"未知阶段 {stage!r}（可选 {known}）") from exc


def default_model(stage: str) -> StageModel:
    return models_for(stage)[0]


def weight_by_id(model: StageModel, weight_id: str) -> WeightSpec:
    for item in model.downloads:
        if item.id == weight_id:
            return item
    raise SystemExit(f"模型 {model.id} 没有权重项 {weight_id}")


_OLLAMA_PREFIX = "ollama:"
# 与 ollama list 里的名字对齐：qwen2.5:7b、library 外的 namespace/name:tag、hf.co/...
_OLLAMA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,200}$")


def ollama_model_name(raw: str) -> str:
    """``ollama:qwen2.5:7b`` → ``qwen2.5:7b``。不是这个前缀时不要调。"""
    name = (raw or "").strip()[len(_OLLAMA_PREFIX) :].strip()
    if not name or ".." in name or not _OLLAMA_NAME.fullmatch(name):
        raise SystemExit(
            f"无效的 Ollama 模型名 {raw!r}。应写成 ollama:模型名，例如 ollama:qwen2.5:7b。"
        )
    return name


def make_ollama_model(name: str, *, label: str = "", note: str = "") -> StageModel:
    """分镜走本机 Ollama。没有 Hugging Face 权重，下载清单为空。"""
    return StageModel(
        stage="storyboard",
        id=f"{_OLLAMA_PREFIX}{name}",
        label=label or f"Ollama · {name}",
        backend="ollama",
        runtime="ollama",
        repo=name,
        downloads=(),
        note=note or "本机 Ollama 已下载的模型。分镜推理走 Ollama，不下载 Hugging Face 权重。",
    )


def _looks_like_repo(text: str) -> bool:
    if not text or " " in text or "\n" in text:
        return False
    # org/name，或本机目录
    return "/" in text or text.startswith(".") or text.startswith("~")


def _custom_from(base: StageModel, repo: str) -> StageModel:
    """沿用这条的 runtime，只把主权重仓库换成用户写的 repo。伴随权重（Gemma、T5）不动。"""
    downloads: list[WeightSpec] = []
    replaced = False
    for item in base.downloads:
        if not replaced and item.repo == base.repo:
            downloads.append(
                replace(item, repo=repo, label=f"{item.label}（自定义）", approx_bytes=0)
            )
            replaced = True
        else:
            downloads.append(item)
    if not replaced:
        raise SystemExit(f"模型 {base.id} 的下载清单里没有主权重 {base.repo}")
    return replace(
        base,
        id=repo,
        label=repo,
        repo=repo,
        aliases=(),
        note="自定义仓库，运行时与该阶段默认模型相同",
        downloads=tuple(downloads),
    )


def resolve_stage_model(stage: str, raw: str | None) -> StageModel:
    """把配置值收成目录项。短 id、repo 别名、同 runtime 的自定义 repo 都可以。"""
    models = models_for(stage)
    text = (raw or "").strip()
    if not text:
        return models[0]
    if stage == "storyboard" and text.startswith(_OLLAMA_PREFIX):
        return make_ollama_model(ollama_model_name(text))
    for model in models:
        if text == model.id or text == model.repo or text in model.aliases:
            return model
    if _looks_like_repo(text) and any(model.accepts_custom_repo for model in models):
        return _custom_from(models[0], text)
    names = "、".join(model.id for model in models)
    label = STAGE_LABELS.get(stage, stage)
    extra = ""
    if any(model.accepts_custom_repo for model in models):
        extra = "；也可以填同一个运行时能加载的 Hugging Face repo。"
    raise SystemExit(f"未知{label}模型 {text!r}。可选：{names}{extra}")


def download_specs(settings) -> list[dict[str, Any]]:
    """当前配置实际要的权重。没选中的档位不会进安装清单。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stage, field in STAGE_FIELDS.items():
        model = resolve_stage_model(stage, getattr(settings, field, ""))
        for item in model.downloads:
            if item.repo in seen:
                continue
            seen.add(item.repo)
            out.append(item.as_dict())
    return out


def selftest_catalog() -> None:
    """不加载模型：目录解析、自定义 repo、下载清单跟配置走。"""
    from pipeline.config import Settings

    seen: dict[tuple[str, str], str] = {}
    for stage in STAGES:
        assert models_for(stage), stage
        for model in models_for(stage):
            assert model.stage == stage
            assert model.backend == "mlx"
            assert model.downloads and model.downloads[0].repo == model.repo
            for key in (model.id, model.repo, *model.aliases):
                prev = seen.get((stage, key))
                if prev is not None and prev != model.id:
                    raise AssertionError(f"{stage} 键 {key!r} 同时属于 {prev} 和 {model.id}")
                seen[(stage, key)] = model.id

    story = resolve_stage_model("storyboard", "qwen3-4b-instruct-4bit")
    assert story.repo == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert resolve_stage_model("storyboard", story.repo).id == story.id
    assert resolve_stage_model("images", "flux2-klein-4b").runtime == "mflux_flux2"
    assert resolve_stage_model("video", "dgrauet/ltx-2.3-mlx-q4").id == "ltx-2.3-q4"
    assert resolve_stage_model("video", "ltx-2.3-q8").repo == "dgrauet/ltx-2.3-mlx-q8"
    ltx25 = resolve_stage_model("video", "ltx-2.5-q4")
    assert ltx25.repo == "reverie-studio/ltx-2.5-mlx-q4-distilled"
    assert all(item.id != "gemma" for item in ltx25.downloads)
    assert "text_encoder.safetensors" in ltx25.downloads[0].required_files
    assert resolve_stage_model("music", "").id == "musicgen-small"
    assert resolve_stage_model("storyboard", "qwen3.5-35b-a3b-4bit").repo == (
        "mlx-community/Qwen3.5-35B-A3B-4bit"
    )
    assert resolve_stage_model("images", "flux2-klein-9b-q8").variant == "flux2_klein_9b"
    assert resolve_stage_model("tts", "qwen3-tts-1.7b-customvoice-bf16").runtime == "mlx_tts"
    assert resolve_stage_model("music", "musicgen-large").repo == "facebook/musicgen-large"

    custom = resolve_stage_model("storyboard", "mlx-community/Qwen3-1.7B-4bit")
    assert custom.runtime == "mlx_lm"
    assert custom.repo == "mlx-community/Qwen3-1.7B-4bit"
    assert custom.downloads[0].repo == custom.repo
    assert custom.downloads[0].approx_bytes == 0
    big = resolve_stage_model("storyboard", "qwen3.5-35b-a3b-4bit")
    small_llm = resolve_stage_model("storyboard", "qwen3-4b-instruct-4bit")
    assert big.downloads[0].approx_bytes > small_llm.downloads[0].approx_bytes > 0
    assert resolve_stage_model("video", "ltx-2.5-q4").downloads[0].approx_bytes > 0
    assert "GB" in choice_disk_size(small_llm)
    assert choice_disk_size(custom) == ""
    assert "Gemma" in choice_disk_size(resolve_stage_model("video", "ltx-2.3-q4"))
    assert "Gemma" not in choice_disk_size(resolve_stage_model("video", "ltx-2.5-q4"))
    assert "T5" in choice_disk_size(resolve_stage_model("music", "musicgen-small"))

    ollama = resolve_stage_model("storyboard", "ollama:qwen2.5:7b")
    assert ollama.runtime == "ollama" and ollama.backend == "ollama"
    assert ollama.id == "ollama:qwen2.5:7b" and ollama.repo == "qwen2.5:7b"
    assert ollama.downloads == ()
    assert resolve_stage_model("storyboard", "ollama:hf.co/user/model:Q4_K_M").repo == (
        "hf.co/user/model:Q4_K_M"
    )
    try:
        resolve_stage_model("storyboard", "ollama:")
        raise AssertionError("空的 ollama 模型名应失败")
    except SystemExit as exc:
        assert "Ollama" in str(exc)
    try:
        resolve_stage_model("storyboard", "ollama:../secret")
        raise AssertionError("越界的 ollama 模型名应失败")
    except SystemExit as exc:
        assert "Ollama" in str(exc)
    try:
        resolve_stage_model("images", "some-org/not-flux")
        raise AssertionError("静帧不应接受任意仓库")
    except SystemExit as exc:
        assert "flux2-klein-4b" in str(exc)

    defaults = Settings()
    specs = download_specs(defaults)
    repos = [item["repo"] for item in specs]
    assert "dgrauet/ltx-2.3-mlx-q4" in repos
    assert "dgrauet/ltx-2.3-mlx-q8" not in repos
    assert repos.count("mlx-community/gemma-3-12b-it-4bit") == 1
    assert "t5-base" in repos
    assert "mlx-community/Qwen3.5-35B-A3B-4bit" not in repos
    assert "mflux-community/flux2-klein-9b-mflux-q8" not in repos
    assert "facebook/musicgen-large" not in repos
    assert "reverie-studio/ltx-2.5-mlx-q4-distilled" not in repos

    wide = Settings(
        llm_model="qwen3.5-35b-a3b-4bit",
        image_model="flux2-klein-9b-q8",
        tts_model="qwen3-tts-1.7b-customvoice-bf16",
        music_model="musicgen-large",
    )
    wide_repos = [item["repo"] for item in download_specs(wide)]
    assert "mlx-community/Qwen3.5-35B-A3B-4bit" in wide_repos
    assert "mlx-community/Qwen3-4B-Instruct-2507-4bit" not in wide_repos
    assert "mflux-community/flux2-klein-9b-mflux-q8" in wide_repos
    assert "mflux-community/flux2-klein-4b-mflux-q4" not in wide_repos
    assert "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16" in wide_repos
    assert "facebook/musicgen-large" in wide_repos
    assert "facebook/musicgen-small" not in wide_repos

    via_ollama = Settings(llm_model="ollama:qwen2.5:7b")
    ollama_repos = [item["repo"] for item in download_specs(via_ollama)]
    assert "mlx-community/Qwen3-4B-Instruct-2507-4bit" not in ollama_repos
    assert all(not str(repo).startswith("ollama:") for repo in ollama_repos)

    q8 = Settings(video_model="ltx-2.3-q8", music_model="musicgen-medium")
    picked = [item["repo"] for item in download_specs(q8)]
    assert "dgrauet/ltx-2.3-mlx-q8" in picked
    assert "dgrauet/ltx-2.3-mlx-q4" not in picked
    assert "facebook/musicgen-medium" in picked
    assert "facebook/musicgen-small" not in picked

    ltx25 = Settings(video_model="ltx-2.5-q4")
    ltx25_repos = [item["repo"] for item in download_specs(ltx25)]
    assert "reverie-studio/ltx-2.5-mlx-q4-distilled" in ltx25_repos
    assert "dgrauet/ltx-2.3-mlx-q4" not in ltx25_repos
    assert "mlx-community/gemma-3-12b-it-4bit" not in ltx25_repos

    try:
        resolve_stage_model("video", "not-a-model")
        raise AssertionError("非法 id 应失败")
    except SystemExit as exc:
        assert "ltx-2.3-q4" in str(exc)

    import os

    from pipeline.backends import bind, resolve_backend, save_selection

    saved_backend = os.environ.pop("NOVEL_VIDEO_BACKEND", None)
    try:
        info = resolve_backend("auto")
        assert info.name == "mlx" and info.is_mlx
        picked_ollama = bind(Settings(llm_model="ollama:qwen2.5:7b"), "storyboard")
        assert picked_ollama.runtime == "ollama" and picked_ollama.backend == "ollama"
        try:
            resolve_backend("cuda")
            raise AssertionError("未注册的 cuda 后端应拒绝")
        except SystemExit as exc:
            assert "MLX" in str(exc)
        try:
            resolve_backend("metal")
            raise AssertionError("未知后端应拒绝")
        except SystemExit as exc:
            assert "metal" in str(exc)
    finally:
        if saved_backend is not None:
            os.environ["NOVEL_VIDEO_BACKEND"] = saved_backend

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({"image_width": 576, "video_model": "ltx-2.3-q4"}), encoding="utf-8")
        save_selection(path, {"video": "ltx-2.3-q8", "music": "musicgen-medium"})
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["image_width"] == 576
        assert saved["video_model"] == "ltx-2.3-q8"
        assert saved["music_model"] == "musicgen-medium"
        save_selection(path, {"storyboard": "ollama:qwen2.5:7b"})
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["llm_model"] == "ollama:qwen2.5:7b"
        assert saved["video_model"] == "ltx-2.3-q8"
        loaded = Settings.load(path)
        assert resolve_stage_model("video", loaded.video_model).repo == "dgrauet/ltx-2.3-mlx-q8"
        assert resolve_stage_model("storyboard", loaded.llm_model).runtime == "ollama"
