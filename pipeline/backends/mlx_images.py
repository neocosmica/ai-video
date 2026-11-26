"""静帧：MLX 上的 mflux FLUX.2 Klein。"""

from __future__ import annotations

import gc
import sys

from pipeline.backends.catalog import StageModel
from pipeline.hf_cache import resolve_snapshot

_BAR_PATCHED = False


class MlxFluxSession:
    def __init__(self, spec: StageModel, *, kind: str, load_label: str = "文生图模型"):
        if spec.runtime != "mflux_flux2":
            raise SystemExit(f"静帧运行时不是 mflux_flux2：{spec.runtime}")
        if kind not in {"t2i", "edit"}:
            raise SystemExit(f"未知静帧加载类型：{kind}")
        self.spec = spec
        self.kind = kind
        weights = resolve_snapshot(spec.repo)
        if weights is None:
            raise SystemExit(
                f"找不到{spec.label}的权重（{spec.repo}）。请先运行 scripts/setup.sh，"
                "或在网页里点「安装依赖」。"
            )
        print("加载 mflux / transformers…", flush=True)
        from mflux.models.common.config import ModelConfig

        if kind == "t2i":
            try:
                from mflux.models.flux2.variants import Flux2Klein
            except ImportError:
                from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
            cls = Flux2Klein
        else:
            try:
                from mflux.models.flux2.variants import Flux2KleinEdit
            except ImportError:
                from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
            cls = Flux2KleinEdit
        config_name = spec.variant or "flux2_klein_4b"
        factory = getattr(ModelConfig, config_name, None)
        if not callable(factory):
            raise SystemExit(f"mflux 没有配置 {config_name}（模型 {spec.id}）")
        config = factory()
        print(f"加载{load_label} {weights}")
        # 目录里的权重已经量化过，不要再 quantize
        self._model = cls(model_config=config, quantize=None, model_path=str(weights))

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        desc: str,
        images=None,
        image_strength: float | None = None,
    ):
        kwargs = {
            "seed": seed,
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": width,
            "height": height,
            "guidance": 1.0,
        }
        if images:
            kwargs["image_paths"] = images
        if image_strength is not None:
            kwargs["image_strength"] = image_strength
        _use_step_bar(desc)
        try:
            return self._model.generate_image(**kwargs)
        except TypeError:
            kwargs.pop("guidance", None)
            if image_strength is not None:
                kwargs.pop("image_strength", None)
            if images:
                kwargs.pop("image_paths", None)
                kwargs["image_paths"] = images
                if image_strength is not None:
                    kwargs["image_strength"] = image_strength
            try:
                return self._model.generate_image(**kwargs)
            except TypeError:
                kwargs.pop("image_strength", None)
                return self._model.generate_image(**kwargs)

    def close(self) -> None:
        self._model = None
        gc.collect()


def _use_step_bar(desc: str) -> None:
    """给 mflux 去噪循环的 tqdm 加上说明。每张图新建 Config，所以各有一条。"""
    global _BAR_PATCHED
    from tqdm import tqdm

    from mflux.models.common.config.config import Config

    Config._novel_step_desc = desc  # type: ignore[attr-defined]
    if _BAR_PATCHED:
        return

    def time_steps(self):
        if self._time_steps is None:
            label = getattr(Config, "_novel_step_desc", "") or None
            # initial/total 按完整去噪步数计：img2img 从半程起跑时，条也要从 2/4 走到 4/4，
            # 不要变成 1/2→2/2，否则网页上看起来总在 50% 就结束。
            self._time_steps = tqdm(
                range(self.init_time_step, self.num_inference_steps),
                desc=label,
                initial=self.init_time_step,
                total=self.num_inference_steps,
                miniters=1,
                mininterval=0.2,
                file=sys.stderr,
            )
        return self._time_steps

    Config.time_steps = property(time_steps)
    _BAR_PATCHED = True
