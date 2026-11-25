"""验证：LTX-2.3 的 GGUF 是否匹配 diffusers 0.43 架构（6行表=2.3，9行表=2.5）。"""
import os, json, urllib.request
os.environ["PYTHONUTF8"]="1"

# 1. 检查 diffusers 期望的行数（从源码已知：video_mod_param_num / audio_mod_param_num）
import inspect
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel, LTX2VideoTransformerBlock
src = inspect.getsource(LTX2VideoTransformerBlock.__init__)
print("=== 行数计算逻辑 ===", flush=True)
for line in src.splitlines():
    if "mod_param_num" in line:
        print("  ", line.strip(), flush=True)

# 2. 检查 2.3 GGUF 的表形状
GGUF23 = r"C:/Users/annas/.cache/huggingface/hub/models--unsloth--LTX-2.3-GGUF"
print("\n=== 2.3 GGUF 是否存在 ===", flush=True)
print("  path exists:", os.path.exists(GGUF23), flush=True)

# 3. 尝试用 Lightricks/LTX-2 (非gated) 的 config 加载 2.3 GGUF
print("\n=== 下载 Lightricks/LTX-2 transformer config（非gated）===", flush=True)
from huggingface_hub import hf_hub_download
try:
    c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json")
    print("  OK:", c, flush=True)
    cfg = json.load(open(c))
    for k in ["num_layers","audio_num_attention_heads","audio_attention_head_dim","caption_channels","use_prompt_adaln_single"]:
        print(f"    {k}: {cfg.get(k)}", flush=True)
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:150], flush=True)
