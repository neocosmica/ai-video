"""查 LTX-2 (非 gated) 的 model_index 与各组件体积，规划只下必要部分。"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
REPO="Lightricks/LTX-2"
mi = hf_hub_download(repo_id=REPO, filename="model_index.json")
idx = json.load(open(mi))
print("=== model_index ===", flush=True)
for k,v in idx.items():
    if isinstance(v, list): print("  %-20s %s" % (k,v), flush=True)
print("\n=== 关键文件体积 ===", flush=True)
api = HfApi()
files = api.list_repo_files(REPO)
want = [f for f in files if any(x in f for x in ["text_encoder/","connectors/","vae/","audio_vae/","vocoder/","scheduler/","tokenizer/"])]
for f in want[:30]: print("  ", f, flush=True)
print("\ntotal relevant:", len(want), flush=True)
