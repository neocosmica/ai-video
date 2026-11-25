"""CUDA 文本编码器：Gemma3（LTX-2 用），4bit 量化 + 编码后立刻释放。

显存账（RTX 4070 12 GB 实测）
----------------------------
    bf16  Gemma3 12B → 约 22 GB，装不下
    4bit  nf4       → 峰值 7.77 GB，可装  ✅

但 7.77 GB 仍与 transformer（8.57 GB）冲突，**必须分阶段**：

    段1 载入 text_encoder        free 3.02 GB, peak 7.77 GB
        编码 prompt → hidden_states
        **立刻把 embeds 搬到 CPU**，丢掉 hidden_states
        释放模型                 free 10.79 GB   ← 完全回收
    段2 载入 transformer         峰值 7.69 GB
    段3 载入 VAE                 峰值 3.64 GB

峰值 = max(单段) 而非累加 —— 这是 12 GB 能跑 19B 的根据。

坑：``hidden_states`` 是元组，若继续持有引用，``del model`` 不会真正释放
显存（实测释放后 free 只回到 3.00 GB）。必须先 ``.cpu()`` 落到主机内存再删。
"""

from __future__ import annotations

from typing import Any

from pipeline.backends import require_cuda

DEFAULT_REPO = "Lightricks/LTX-2"
DEFAULT_SUBFOLDER = "text_encoder"


def load(*, repo: str = DEFAULT_REPO, subfolder: str = DEFAULT_SUBFOLDER,
         quant: str = "4bit", device: str = "cuda") -> tuple[Any, Any]:
    """加载 Gemma3 文本编码器与 tokenizer。返回 ``(model, tokenizer)``。"""
    require_cuda(what="文本编码阶段")
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
    kwargs: dict[str, Any] = {
        "device_map": {"": 0},
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quant != "bf16":
        raise SystemExit(f"未知文本编码器量化档：{quant!r}（可选 4bit / bf16）")

    model = Gemma3ForConditionalGeneration.from_pretrained(repo, subfolder=subfolder, **kwargs)
    model.eval()
    return model, tokenizer


def encode_and_free(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_length: int = 1024,
    scale_factor: int = 8,
) -> dict:
    """编码一段文本 → ``{"hidden": 全部层, "mask": 注意力掩码}``（都在 CPU）。

    **必须取全部层的 hidden_states**：LTX 的 ``LTX2TextConnectors`` 配置里
    ``text_proj_in_factor=49``，说明它期望 Gemma 全部 49 层堆叠后的张量
    ``(B, seq, caption_channels=3840, 49)``。只取最后一层会得到错误形状，
    进而让视频画面变成网格噪声（实测）。

    返回的 tensor 在主机内存，不占显存，可与后续阶段共存。
    调用后 ``model`` 已被释放，不要再用。
    """
    import torch

    ids = tokenizer(
        prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=int(max_length),
    )
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)

    # 全部层堆叠：每层 (B,seq,3840) → stack 到 (B,seq,3840,num_layers)
    hidden = torch.stack(list(out.hidden_states), dim=-1)
    mask = ids.get("attention_mask")
    hidden_cpu = hidden.detach().to(torch.bfloat16).cpu()
    mask_cpu = mask.detach().cpu() if mask is not None else None

    # 坚决丢干净，否则显存不回收（见模块 docstring 的「坑」）
    del out, hidden, ids, mask
    from pipeline.backends import cuda_memory

    cuda_memory.empty_cache()
    return {"hidden": hidden_cpu, "mask": mask_cpu,
            "scale_factor": int(scale_factor)}
