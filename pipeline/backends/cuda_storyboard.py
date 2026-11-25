"""CUDA 分镜后端：Qwen3.5-4B（4bit）从小说生成分镜 JSON。

实测（RTX 4070 12 GB）：
    bf16     显存 7.9 GB，17.8 tok/s（装 triton-windows 后）
    4bit     显存 ~2.4 GB

关键坑：必须 ``enable_thinking=False``
------------------------------------
Qwen3.5 默认输出 ``<think>`` 思考链，会把 JSON 挤掉导致解析失败（实测 ``shots=0``）。
关闭后 prompt 尾部生成 ``<think>\\n\\n</think>\\n\\n``，模型直接吐纯 JSON：

    apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)

Windows 上还有一层：**没有官方 triton** 时 ``flash-linear-attention`` 退化成
参考实现，Qwen3.5 只有 12.5 tok/s；装 ``triton-windows`` 后恢复 17.8 tok/s。
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.backends import cuda_memory, require_cuda

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_QUANT = "4bit"  # 4bit | bf16


def build_quant_config(quant: str):
    """构造量化配置；``bf16`` 返回 None（不量化）。"""
    if quant == "bf16":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if quant != "4bit":
        raise SystemExit(f"未知量化档：{quant!r}（可选 4bit / bf16）")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(*, model: str = DEFAULT_MODEL, quant: str = DEFAULT_QUANT,
               device: str = "cuda") -> tuple[Any, Any]:
    """加载 tokenizer + model。返回 ``(model, tokenizer)``。"""
    require_cuda(what="分镜阶段")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    qcfg = build_quant_config(quant)
    kwargs: dict[str, Any] = {
        "device_map": {"": 0},
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    estimator = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    estimator.eval()
    return estimator, tokenizer


def chat(
    model: Any,
    tokenizer: Any,
    *,
    system: str,
    user: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float | None = None,
) -> str:
    """一次对话生成；**关掉 thinking**，返回纯文本。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,      # ← 关键：不开会在 JSON 前面灌思考链
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        if top_p:
            gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(text: str) -> dict:
    """从模型输出里抓出第一个 JSON 对象；失败抛 ``ValueError``。"""
    import json

    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip()
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("输出里没有 JSON 对象")
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:idx + 1])
    raise ValueError("JSON 对象未闭合（可能被 max_tokens 截断）")


# ---------------------------------------------------------------------------
# 命令入口（与 mac 端 pipeline/storyboard_gen.py 对齐）
# ---------------------------------------------------------------------------


def cmd_storyboard_cuda(settings, args) -> None:
    """CUDA 分镜阶段：小说 → storyboard.json。

    提示词与 mac 端 ``storyboard_gen`` 共用同一套结构约束（字段名、音色、
    英文 motion/music prompt），保证两端产出的分镜可互相通用。
    """
    from pipeline.backends import cuda_memory
    from pipeline.util import dump_json, load_json, resolve

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    story_path = out / "storyboard.json"
    novel = resolve(args.novel) if getattr(args, "novel", None) else out / "novel.txt"
    if not novel.exists():
        raise SystemExit(f"找不到小说：{novel}")
    if story_path.exists() and not getattr(args, "force", False):
        print(f"已有分镜，跳过生成：{story_path}（要重写请加 --force）", flush=True)
        return

    novel_text = novel.read_text(encoding="utf-8")
    if not novel_text.strip():
        raise SystemExit(f"小说是空的：{novel}")

    model_name = getattr(settings, "cuda_llm_model", DEFAULT_MODEL)
    quant = getattr(settings, "cuda_llm_quant", DEFAULT_QUANT)
    max_shots = int(getattr(args, "max_shots", 0) or settings.max_shots)
    print(f"小说 → 分镜：{model_name}（{quant}）", flush=True)

    schema = (
        '{"title":"标题","style":"中文画面风格",'
        '"music_prompt":"English BGM, no vocals",'
        '"characters":[{"id":"pinyin","name":"中文名","gender":"female|male",'
        '"voice":"Serena|Vivian|Dylan|Uncle_Fu|Eric","appearance":"年龄性别发型衣服"}],'
        '"shots":[{"id":"s001","characters":["id"],"location":"地点",'
        '"image_prompt":"中文画面，勿写对白字","motion_prompt":"English camera only",'
        '"speaker":"说话人中文名或空","dialogue":"原文对白，没有就空",'
        '"emotion":"calm|warm|sad|tense|surprised"}]}'
    )
    system = "你是专业短剧分镜师，只输出 JSON，不输出任何其他文字。"
    user = (
        f"把下面小说改编成竖屏短剧分镜。只输出 JSON，不要任何分析或解释。\n\n"
        f"结构（严格遵守字段名）：\n{schema}\n\n"
        f"要求：{max_shots} 个镜头以内；motion_prompt 与 music_prompt 必须是英文；"
        f"dialogue 必须逐字抄原文引号内台词；不要输出思考过程。\n\n"
        f"小说：\n<<<\n{novel_text}\n>>>"
    )

    with cuda_memory.stage("storyboard", min_free_gib=3.0):
        model, tokenizer = load_model(model=model_name, quant=quant)
        last_err = None
        data = None
        for attempt in range(2):
            temp = settings.llm_temperature if attempt == 0 else 0.2
            raw = chat(
                model, tokenizer, system=system, user=user,
                max_new_tokens=int(settings.llm_max_tokens), temperature=temp,
            )
            print(f"  生成 {len(raw)} 字符（尝试 {attempt+1}）", flush=True)
            try:
                data = extract_json(raw)
                if not isinstance(data.get("shots"), list) or not data["shots"]:
                    raise ValueError("JSON 缺少 shots")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                data = None
                print(f"  解析失败：{exc}", flush=True)
        del model, tokenizer
    cuda_memory.empty_cache()

    if data is None:
        raise SystemExit(f"分镜生成失败：{last_err}")

    data.setdefault("title", novel.stem)
    data["novel"] = ""
    dump_json(story_path, data)
    n_shots = len(data.get("shots") or [])
    n_chars = len(data.get("characters") or [])
    print(f"分镜已写入：{story_path}（{n_shots} 镜 / {n_chars} 角色）", flush=True)
