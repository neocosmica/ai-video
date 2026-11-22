#!/usr/bin/env python3
"""给 vendor/ltx-2-mlx 打上 Gemma/connector 分阶段加载补丁。

可重复执行；已打过则跳过。setup.sh 在 clone/sync 后调用。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "NOVEL_VIDEO_STAGED_GEMMA_LOAD"

BLOCKS = (
    ROOT
    / "vendor"
    / "ltx-2-mlx"
    / "packages"
    / "ltx-pipelines-mlx"
    / "src"
    / "ltx_pipelines_mlx"
    / "utils"
    / "blocks.py"
)
ENCODER = (
    ROOT
    / "vendor"
    / "ltx-2-mlx"
    / "packages"
    / "ltx-core-mlx"
    / "src"
    / "ltx_core_mlx"
    / "text_encoders"
    / "gemma"
    / "encoders"
    / "base_encoder.py"
)

LOAD_ENCODE_REPLACEMENT = '''    def load(self) -> None:
        """Load Gemma text tower only.

        Connector (~6GB on LTX-2.3) is loaded just-in-time in :meth:`encode`
        after Gemma hidden states are materialized and the tower is freed.
        Co-resident Gemma+connector peaks near 14GB and is unstable on 16GB
        unified-memory Macs.
        # NOVEL_VIDEO_STAGED_GEMMA_LOAD
        """
        self._load_text_encoder_only()

    def _load_text_encoder_only(self) -> None:
        """Load Gemma / Gemma4 text tower if needed."""
        self._encoder_kind = select_text_encoder(self.model_dir)
        if self._text_encoder is not None:
            return
        if self._encoder_kind == "gemma4":
            from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

            self._text_encoder = Gemma4TextEncoder()
            self._text_encoder.load(self.model_dir)
        else:
            self._text_encoder = GemmaLanguageModel()
            self._text_encoder.load(self.gemma_model_id)
        aggressive_cleanup()

    def _load_feature_extractor_only(self) -> None:
        """Load embeddings connector after the text tower has been freed."""
        if self._feature_extractor is not None:
            return
        if self._encoder_kind is None:
            self._encoder_kind = select_text_encoder(self.model_dir)
        double_precision_rope = LTXModelConfig.from_checkpoint_dir(self.model_dir).double_precision_rope
        self._feature_extractor = GemmaFeaturesExtractorV2(double_precision_rope=double_precision_rope)
        connector_weights = load_split_safetensors(self.model_dir / "connector.safetensors", prefix="connector.")
        if self._encoder_kind == "gemma4":
            projection_weights = load_split_safetensors(
                self.model_dir / "text_encoder.safetensors",
                prefix="text_encoder.text_embedding_projection.",
            )
            connector_weights.update({f"text_embedding_projection.{k}": v for k, v in projection_weights.items()})
        self._feature_extractor.connector.load_weights(list(connector_weights.items()))
        aggressive_cleanup()

    def free(self) -> None:
        """Drop strong refs; rely on GC + aggressive_cleanup to reclaim memory."""
        self._text_encoder = None
        self._feature_extractor = None
        aggressive_cleanup()

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode a single prompt to ``(video_embeds, audio_embeds)``.

        Stages memory: Gemma → hidden states → free Gemma → connector → embeds.
        Caller is responsible for freeing via :meth:`free` when done with
        the encoder. For one-shot use, prefer :meth:`__call__`.
        """
        import os

        self._load_text_encoder_only()
        assert self._text_encoder is not None

        max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", "1024"))
        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(prompt, max_length=max_length)
        # Materialize hidden states so we can drop the 8GB tower before the
        # ~6GB connector lands on unified memory.
        mx.eval(attention_mask, *all_hidden_states)
        self._text_encoder = None
        aggressive_cleanup()

        self._load_feature_extractor_only()
        assert self._feature_extractor is not None
        video_embeds, audio_embeds = self._feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds
'''

ENCODER_LOAD_REPLACEMENT = '''    def load(self, model_path: str | None = None) -> None:
        """Load the Gemma model via mlx-lm.

        Args:
            model_path: Path or HuggingFace repo ID.
        """
        from mlx_lm import load as mlx_lm_load

        path = model_path or self._model_path
        if path is None:
            raise ValueError("model_path must be provided")

        # Prefer an on-disk snapshot so mlx_lm skips HuggingFace hub I/O
        # (hub calls can hang or re-fetch under mirror/quota pressure).
        # lazy=True: don't mx.eval the full 8GB tower at import time; the
        # per-layer encode path materializes weights as needed.
        resolved = _resolve_local_model_path(path)
        self._model_path = resolved
        self._model, self._tokenizer = mlx_lm_load(resolved, lazy=True)
'''

RESOLVE_HELPER = '''
def _resolve_local_model_path(path_or_repo: str) -> str:
    """Resolve a HF repo id to a local snapshot path when cached.

    # NOVEL_VIDEO_STAGED_GEMMA_LOAD
    """
    candidate = Path(path_or_repo)
    if candidate.exists():
        return str(candidate)

    # models--org--name/snapshots/<rev>
    repo = path_or_repo.strip().strip("/")
    cache_name = "models--" + repo.replace("/", "--")
    hub = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    snaps = hub / "snapshots"
    if snaps.is_dir():
        ref = hub / "refs" / "main"
        if ref.is_file():
            rev = ref.read_text(encoding="utf-8").strip()
            snap = snaps / rev
            if snap.is_dir():
                return str(snap)
        ranked = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ranked:
            return str(ranked[0])
    return path_or_repo


'''


def _patch_blocks(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"已打过补丁：{path.relative_to(ROOT)}")
        return False
    pattern = re.compile(
        r"    def load\(self\) -> None:.*?return video_embeds, audio_embeds\n",
        re.DOTALL,
    )
    new_text, n = pattern.subn(LOAD_ENCODE_REPLACEMENT + "\n", text, count=1)
    if n != 1:
        raise SystemExit(f"无法匹配 PromptEncoder.load/encode：{path}")
    path.write_text(new_text, encoding="utf-8")
    print(f"已补丁：{path.relative_to(ROOT)}")
    return True


def _patch_encoder(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text and "def _resolve_local_model_path" in text and "lazy=True" in text:
        print(f"已打过补丁：{path.relative_to(ROOT)}")
        return False
    # 匹配上游原版，或半截补丁（已改 load 但缺 helper/标记）
    pattern = re.compile(
        r"    def load\(self, model_path: str \| None = None\) -> None:.*?"
        r"self\._model, self\._tokenizer = mlx_lm_load\([^\n]+\)\n",
        re.DOTALL,
    )
    new_text, n = pattern.subn(ENCODER_LOAD_REPLACEMENT + "\n", text, count=1)
    if n != 1:
        raise SystemExit(f"无法匹配 GemmaLanguageModel.load：{path}")
    if "def _resolve_local_model_path" not in new_text:
        anchor = "@functools.lru_cache(maxsize=2)\ndef _load_system_prompt"
        if anchor not in new_text:
            raise SystemExit(f"找不到插入点 _load_system_prompt：{path}")
        new_text = new_text.replace(anchor, RESOLVE_HELPER + anchor, 1)
    if MARKER not in new_text or "def _resolve_local_model_path" not in new_text:
        raise SystemExit(f"补丁后仍缺少 helper/标记：{path}")
    path.write_text(new_text, encoding="utf-8")
    print(f"已补丁：{path.relative_to(ROOT)}")
    return True


def main() -> int:
    if not BLOCKS.exists() or not ENCODER.exists():
        print("还没有 vendor/ltx-2-mlx，跳过 Gemma 补丁", file=sys.stderr)
        return 0
    _patch_blocks(BLOCKS)
    _patch_encoder(ENCODER)
    for path in (BLOCKS, ENCODER):
        text = path.read_text(encoding="utf-8")
        if MARKER not in text:
            raise SystemExit(f"补丁后仍缺少标记：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
