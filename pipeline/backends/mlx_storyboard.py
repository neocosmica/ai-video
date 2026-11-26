"""分镜聊天：MLX 上的 mlx-lm。"""

from __future__ import annotations

import gc

from pipeline.backends.catalog import StageModel


def force_close_think(text: str) -> str:
    """若思考块未闭合，补上 </think>，便于继续生成 JSON。"""
    if "</think>" in text:
        return text
    return text.rstrip() + "\n</think>\n\n"


class MlxStoryboardLLM:
    def __init__(
        self,
        spec: StageModel,
        *,
        think_max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ):
        if spec.runtime != "mlx_lm":
            raise SystemExit(f"分镜运行时不是 mlx_lm：{spec.runtime}")
        try:
            from mlx_lm import generate, load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise SystemExit(
                "缺少 mlx-lm。请重跑 ./scripts/setup.sh，或在主环境安装：uv sync"
            ) from exc
        self.spec = spec
        self.think_max_tokens = max(64, int(think_max_tokens))
        effort = str(reasoning_effort or "low").strip().lower()
        self.reasoning_effort = effort if effort in {"low", "medium", "high"} else "low"
        self._generate = generate
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        print(
            f"加载分镜 LLM：{spec.label}（{spec.repo}，"
            f"effort={self.reasoning_effort}, think≤{self.think_max_tokens}）",
            flush=True,
        )
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

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        think_max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # 开 thinking，但用 think_max_tokens 硬截断，防止把额度全耗在推理上。
        prompt = self._apply_chat_template(messages)
        sampler = self._make_sampler(temp=temperature)
        prev = self.think_max_tokens
        if think_max_tokens is not None:
            self.think_max_tokens = max(64, int(think_max_tokens))
        try:
            return self._generate_with_think_budget(
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
        finally:
            self.think_max_tokens = prev

    def _apply_chat_template(self, messages: list[dict]) -> str:
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_with_think_budget(self, *, prompt: str, max_tokens: int, sampler) -> str:
        # 给 JSON 答案留约 512；不再硬砍成 max_tokens/2（角色步额度小会被卡在 768）
        answer_reserve = min(512, max(64, max_tokens // 4))
        think_budget = min(self.think_max_tokens, max(64, max_tokens - answer_reserve))
        text = ""
        last = None
        think_closed = False
        forced = False

        print("=" * 10, flush=True)
        for response in self._stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            last = response
            if response.text:
                print(response.text, end="", flush=True)
                text += response.text
            if think_closed:
                continue
            if "</think>" in text:
                think_closed = True
                continue
            if response.generation_tokens >= think_budget:
                forced = True
                break

        if forced:
            text = force_close_think(text)
            print("\n</think>\n\n", end="", flush=True)
            used = last.generation_tokens if last is not None else think_budget
            remain = max(0, int(max_tokens) - int(used) - 8)
            print(
                f"\n提示：思考已达 {think_budget} tokens，截断后续推理并续写 JSON"
                f"（剩 {remain} tokens）",
                flush=True,
            )
            if remain > 32:
                cont = self._generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt + text,
                    max_tokens=remain,
                    sampler=sampler,
                    verbose=True,
                )
                text += cont or ""
            else:
                print("=" * 10, flush=True)
                if last is not None:
                    print(
                        f"Prompt: {last.prompt_tokens} tokens, "
                        f"{last.prompt_tps:.3f} tokens-per-sec",
                        flush=True,
                    )
                    print(
                        f"Generation: {last.generation_tokens}+ tokens (think capped), "
                        f"{last.generation_tps:.3f} tokens-per-sec",
                        flush=True,
                    )
                    print(f"Peak memory: {last.peak_memory:.3f} GB", flush=True)
            return text

        print(flush=True)
        print("=" * 10, flush=True)
        if last is None or not text:
            print("No text generated for this prompt", flush=True)
            return text
        print(
            f"Prompt: {last.prompt_tokens} tokens, "
            f"{last.prompt_tps:.3f} tokens-per-sec",
            flush=True,
        )
        print(
            f"Generation: {last.generation_tokens} tokens, "
            f"{last.generation_tps:.3f} tokens-per-sec",
            flush=True,
        )
        print(f"Peak memory: {last.peak_memory:.3f} GB", flush=True)
        return text

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
