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
    # 从原文抽出全部引号台词，明确列给模型——漏句会被 screenplay 硬校验拦下
    from pipeline.screenplay import extract_novel_dialogues

    quotes = extract_novel_dialogues(novel_text) or []
    quote_block = ""
    if quotes:
        listed = "\n".join(f"  {i+1}. 「{q}」" for i, q in enumerate(quotes))
        quote_block = (
            f"\n小说里有 {len(quotes)} 句引号台词，**必须逐字全部出现在某个镜头的 dialogue 里，"
            f"一句都不能漏、不能改写、不能合并**：\n{listed}\n"
            f"每句台词单独占一个镜头，speaker 写说话人的中文名。\n"
        )

    user = (
        f"把下面小说改编成竖屏短剧分镜。只输出 JSON，不要任何分析或解释。\n\n"
        f"结构（严格遵守字段名）：\n{schema}\n\n"
        f"要求：{max_shots} 个镜头以内；motion_prompt 与 music_prompt 必须是英文；"
        f"dialogue 必须逐字抄原文引号内台词；不要输出思考过程。\n"
        f"{quote_block}\n"
        f"小说：\n<<<\n{novel_text}\n>>>"
    )

    with cuda_memory.stage("storyboard", min_free_gib=3.0):
        model, tokenizer = load_model(model=model_name, quant=quant)
        last_err = None
        data = None
        ask = user
        for attempt in range(3):
            temp = settings.llm_temperature if attempt == 0 else 0.2
            raw = chat(
                model, tokenizer, system=system, user=ask,
                max_new_tokens=int(settings.llm_max_tokens), temperature=temp,
            )
            print(f"  生成 {len(raw)} 字符（尝试 {attempt+1}）", flush=True)
            try:
                data = extract_json(raw)
                if not isinstance(data.get("shots"), list) or not data["shots"]:
                    raise ValueError("JSON 缺少 shots")
                # 台词覆盖校验：漏句就**在已生成的分镜上补镜**，不整篇重写
                # （实测整篇重写会让模型越写越短，最终退化到没有 shots）
                from pipeline.audio import _character_map
                from pipeline.screenplay import (
                    _coverage_blob,
                    _spoken_key,
                    extract_novel_dialogues,
                )

                expected = extract_novel_dialogues(novel_text) or []
                while True:
                    blob = _coverage_blob(data.get("shots") or [])
                    missing = [q for q in expected if _spoken_key(q) not in blob]
                    if not missing or len(data.get("shots") or []) >= max_shots:
                        break
                    listed = "\n".join(f"  「{q}」" for q in missing)
                    add_user = (
                        "在上面分镜的基础上，**补齐这些还没进 dialogue 的台词**。\n"
                        f"把它们各自补成一个新镜头（保持已有镜头不变，只追加）：\n{listed}\n\n"
                        "只输出新增镜头的 JSON 数组，形如：\n"
                        '[{"id":"s009","characters":["id"],"location":"地点",'
                        '"image_prompt":"中文画面","motion_prompt":"English camera only",'
                        '"speaker":"说话人中文名","dialogue":"原文对白","emotion":"tense"}]\n\n'
                        f"小说原文：\n<<<\n{novel_text}\n>>>"
                    )
                    print(f"  漏 {len(missing)} 句，补镜", flush=True)
                    extra_raw = chat(
                        model, tokenizer, system=system, user=add_user,
                        max_new_tokens=1024, temperature=0.2,
                    )
                    added = _extract_shot_array(extra_raw)
                    if not added:
                        print("  补镜没拿到新镜头，改由下方兜底补", flush=True)
                        break
                    existing = {str(x.get("id")) for x in data.get("shots") or []}
                    for item in added:
                        sid = str(item.get("id") or "").strip()
                        if not sid or sid in existing:
                            sid = f"s{len(data['shots'])+1:03d}"
                        item["id"] = sid
                        existing.add(sid)
                        data["shots"].append(item)
                        print(f"    + {sid} {str(item.get('dialogue'))[:24]}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                data = None
                print(f"  解析失败：{exc}", flush=True)
        del model, tokenizer
    cuda_memory.empty_cache()

    if data is None:
        raise SystemExit(f"分镜生成失败：{last_err}")

    # 兜底：若仍有台词没进 dialogue，直接按原文补空镜（不改台词，只补镜头），
    # 保证 screenplay 的硬校验能过——台词完整性优先于「模型是否听话」
    _backfill_missing_dialogues(data, novel_text, max_shots)

    data.setdefault("title", novel.stem)
    data["novel"] = ""
    dump_json(story_path, data)
    n_shots = len(data.get("shots") or [])
    n_chars = len(data.get("characters") or [])
    print(f"分镜已写入：{story_path}（{n_shots} 镜 / {n_chars} 角色）", flush=True)



def _extract_shot_array(text: str) -> list[dict]:
    """从模型输出里抓第一个 JSON 数组（补镜用）；兼容被包在 {"shots": [...]} 里的情况。"""
    import json

    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip().replace("```", "")
    start = cleaned.find("[")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start:idx + 1])
                    except Exception:
                        break
                    return [x for x in value if isinstance(x, dict)]
    # 退化：模型可能给了 {"shots": [...]}
    try:
        obj = extract_json(cleaned)
        shots = obj.get("shots")
        return [x for x in shots if isinstance(x, dict)] if isinstance(shots, list) else []
    except Exception:
        return []



def _backfill_missing_dialogues(data: dict, novel_text: str, max_shots: int) -> None:
    """把原文里还没覆盖的引号台词补成独立镜头。

    只在模型漏句时触发。不改写台词，只新增镜头，speaker 按上下文推断，
    推断不出就留空（后续 audio 会按 gender 选音色）。
    """
    from pipeline.screenplay import _coverage_blob, _spoken_key, extract_novel_dialogues

    expected = extract_novel_dialogues(novel_text) or []
    if not expected:
        return
    shots = data.setdefault("shots", [])
    chars = data.get("characters") or []
    by_name = {str(c.get("name") or ""): c for c in chars if c.get("name")}

    # 从原文里找「台词前后紧邻的人名」，用来猜 speaker
    def guess_speaker(line: str) -> str:
        pos = novel_text.find(line)
        window = novel_text[max(0, pos - 40): pos + len(line) + 40] if pos >= 0 else ""
        for name in by_name:
            if name and name in window:
                return name
        return str(shots[0].get("speaker") or "") if shots else ""

    added = 0
    while True:
        blob = _coverage_blob(shots)
        missing = [q for q in expected if _spoken_key(q) not in blob]
        if not missing or len(shots) >= max_shots:
            break
        line = missing[0]
        sid = f"s{len(shots)+1:03d}"
        speaker = guess_speaker(line)
        cast = []
        if speaker and speaker in by_name:
            cid = str(by_name[speaker].get("id") or "")
            if cid:
                cast = [cid]
        prev = shots[-1] if shots else {}
        shots.append({
            "id": sid,
            "characters": cast or list(prev.get("characters") or []),
            "location": prev.get("location") or "便利店门口",
            "image_prompt": f"{speaker or '人物'}说这句话的近景，表情自然，与上一镜同一场景",
            "motion_prompt": "Locked-off camera, natural lip motion",
            "speaker": speaker,
            "dialogue": line,
            "emotion": "tense",
        })
        added += 1
    if added:
        print(f"  兜底补了 {added} 个镜头（原文台词必须全覆盖）", flush=True)
