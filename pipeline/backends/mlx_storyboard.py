"""分镜聊天：MLX 上的 mlx-lm。"""

from __future__ import annotations

import gc

from pipeline.backends.catalog import StageModel


class MlxStoryboardLLM:
    def __init__(self, spec: StageModel):
        if spec.runtime != "mlx_lm":
            raise SystemExit(f"分镜运行时不是 mlx_lm：{spec.runtime}")
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise SystemExit(
                "缺少 mlx-lm。请重跑 ./scripts/setup.sh，或在主环境安装：uv sync"
            ) from exc
        self.spec = spec
        self._generate = generate
        self._make_sampler = make_sampler
        print(f"加载分镜 LLM：{spec.label}（{spec.repo}）", flush=True)
        from pipeline.hf_cache import ensure_snapshot

        primary = spec.downloads[0]
        path = ensure_snapshot(
            spec.repo,
            weight_globs=tuple(primary.weight_globs),
            required_files=tuple(primary.required_files),
            label="分镜语言模型",
        )
        try:
            self._model, self._tokenizer = load(str(path))
        except FileNotFoundError as exc:
            raise SystemExit(
                f"分镜模型权重打不开：{exc}\n"
                f"可删掉该目录后重开，让它重新下载：{path}"
            ) from exc

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampler = self._make_sampler(temp=temperature)
        return self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=True,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
