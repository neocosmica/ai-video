"""探针1：分镜 LLM Qwen3.5-4B 4bit，测加载+生成+显存。"""
import os, time, gc, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("torch", torch.__version__, flush=True)
torch.cuda.reset_peak_memory_stats()

MODEL = "Qwen/Qwen3.5-4B"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"\n[1] 加载 {MODEL} 4bit ...", flush=True)
t0 = time.time()
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg,
                                             device_map={"": 0}, dtype=torch.bfloat16, low_cpu_mem_usage=True)
dt = time.time()-t0
print(f"加载完成 {dt:.1f}s", flush=True)
free,total = torch.cuda.mem_get_info()
print(f"[VRAM] load: peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB free={free/2**30:.2f}GB", flush=True)

NOVEL = """雨下得很大。林晚把风衣领子竖起来，推开街角便利店的玻璃门。灯是白的，关东煮的热气糊在柜台上。
她要了一杯热美式。收银的年轻人抬头的一瞬间，两个人都停住了。是周屿。三年前那个说好一起离开这座城的人，现在站在收银台后面，穿着便利店的绿围裙。"""

PROMPT = f"""你是短剧分镜师。把小说改编成竖屏短剧分镜，输出严格 JSON。

只输出 JSON，不要解释。结构：
{{"title":"标题","style":"中文画面风格","music_prompt":"English BGM, no vocals",
"characters":[{{"id":"pinyin","name":"中文名","gender":"female|male","voice":"Serena|Vivian|Dylan|Uncle_Fu|Eric","appearance":"年龄性别发型衣服"}}],
"shots":[{{"id":"s001","characters":["id"],"location":"地点","image_prompt":"中文画面，勿写对白字","motion_prompt":"English camera only","speaker":"说话人中文名或空","dialogue":"原文对白，没有就空","emotion":"calm|warm|sad|tense|surprised"}}]}}

约束：竖屏短剧，3-5 镜；motion_prompt 与 music_prompt 必须是英文；dialogue 必须抄原文，不许编。

小说：
<<<
{NOVEL}
>>>"""

print("\n[2] 生成分镜 ...", flush=True)
t0 = time.time()
msgs=[{"role":"system","content":"你是专业短剧分镜师。"},{"role":"user","content":PROMPT}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**inp, max_new_tokens=2048, temperature=0.3, do_sample=True)
gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
dt2 = time.time()-t0
n = out.shape[1]-inp.input_ids.shape[1]
print(f"生成完成 {dt2:.1f}s  {n} tokens  {n/dt2:.1f} tok/s", flush=True)
free,total = torch.cuda.mem_get_info()
print(f"[VRAM] gen: peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB free={free/2**30:.2f}GB", flush=True)
print("\n--- 输出 ---", flush=True)
print(gen[:2500], flush=True)

# 校验 JSON
import json, re
try:
    m = re.search(r"\{.*\}", gen, re.DOTALL)
    data = json.loads(m.group(0))
    print(f"\n[JSON OK] title={data.get('title')} shots={len(data.get('shots',[]))} chars={len(data.get('characters',[]))}", flush=True)
except Exception as e:
    print(f"\n[JSON FAIL] {type(e).__name__} {str(e)[:200]}", flush=True)
