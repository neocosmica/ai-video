"""探针1b：关闭 thinking、加大 max_tokens，验证 JSON 能否稳定产出。"""
import os, time, torch, json, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
torch.cuda.reset_peak_memory_stats()

MODEL = "Qwen/Qwen3.5-4B"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg,
                                             device_map={"": 0}, dtype=torch.bfloat16)
print("加载完成", flush=True)

NOVEL = """雨下得很大。林晚把风衣领子竖起来，推开街角便利店的玻璃门。灯是白的，关东煮的热气糊在柜台上。
她要了一杯热美式。收银的年轻人抬头的一瞬间，两个人都停住了。是周屿。三年前那个说好一起离开这座城的人，现在站在收银台后面，穿着便利店的绿围裙。"""

SCHEMA = '''{"title":"标题","style":"中文画面风格","music_prompt":"English BGM, no vocals","characters":[{"id":"pinyin","name":"中文名","gender":"female|male","voice":"Serena|Vivian|Dylan|Uncle_Fu|Eric","appearance":"年龄性别发型衣服"}],"shots":[{"id":"s001","characters":["id"],"location":"地点","image_prompt":"中文画面，勿写对白字","motion_prompt":"English camera only","speaker":"说话人中文名或空","dialogue":"原文对白，没有就空","emotion":"calm|warm|sad|tense|surprised"}]}'''

PROMPT = f"""把下面小说改编成竖屏短剧分镜。只输出 JSON，不要任何分析或解释。

结构（严格遵守字段名）：
{SCHEMA}

要求：3-5 个镜头；motion_prompt 与 music_prompt 必须是英文；dialogue 必须逐字抄原文；不要输出思考过程。

小说：
{NOVEL}"""

msgs=[{"role":"system","content":"你只输出 JSON，不输出任何其他文字。"},
      {"role":"user","content":PROMPT}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

for attempt, mt in enumerate([2048, 4096], 1):
    print(f"\n=== 尝试 {attempt}: max_new_tokens={mt} ===", flush=True)
    t0=time.time()
    inp = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=mt, do_sample=True, temperature=0.3,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    n = out.shape[1]-inp.input_ids.shape[1]
    dt=time.time()-t0
    print(f"{dt:.1f}s  {n} tokens  {n/dt:.1f} tok/s", flush=True)
    print(f"[VRAM peak] {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    # 尝试解析
    m = re.search(r"\{.*\}", gen, re.DOTALL)
    if m:
        try:
            d=json.loads(m.group(0))
            print(f"[JSON OK] title={d.get('title')} shots={len(d.get('shots',[]))}", flush=True)
            print(json.dumps(d, ensure_ascii=False)[:1200], flush=True)
            break
        except Exception as e:
            print(f"[JSON 解析失败] {str(e)[:120]}", flush=True)
    else:
        print("[无 JSON]", flush=True)
    print("原始输出尾部:", repr(gen[-300:]), flush=True)
