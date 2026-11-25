"""下载 LTX-2.5 GGUF 各组件（huggingface_hub，走 hf-mirror）。"""
import os, time, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download

REPO = "unsloth/LTX-2.5-GGUF"
FILES = [
    "distilled/ltx-2.5-22b-distilled-Q3_K_M.gguf",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]
for f in FILES:
    t0 = time.time()
    try:
        p = hf_hub_download(repo_id=REPO, filename=f, resume_download=True)
        sz = os.path.getsize(p) / 2**30
        print(f"OK   {f}  {sz:.2f}GB  {time.time()-t0:.0f}s", flush=True)
        print(f"     -> {p}", flush=True)
    except Exception as e:
        print(f"FAIL {f}: {type(e).__name__} {str(e)[:200]}", flush=True)
