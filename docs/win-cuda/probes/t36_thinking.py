"""验证 Qwen3.5 enable_thinking=False 是否生效。"""
import os, sys, io, torch, time, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
MODEL="Qwen/Qwen3.5-4B"
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg,
        device_map={"":0}, dtype=torch.bfloat16)
print("[1] 加载完成", flush=True)

msgs=[{"role":"user","content":"只输出 JSON：{\"title\":\"测试\",\"n\":3}"}]
# 尝试 enable_thinking=False
for kw in [{"enable_thinking": False}, {}]:
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception as e:
        print("  template 失败", kw, type(e).__name__, str(e)[:80], flush=True); continue
    print("\n[尝试] %s -> prompt 尾部: %r" % (kw, text[-180:]), flush=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t0=time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    n = out.shape[1]-inp.input_ids.shape[1]
    print("  %.1fs %d tok = %.1f tok/s" % (time.time()-t0, n, n/(time.time()-t0)), flush=True)
    print("  输出:", repr(gen[:200]), flush=True)
