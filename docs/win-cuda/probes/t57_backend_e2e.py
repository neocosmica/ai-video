"""用新后端模块真跑一次 CUDA 视频生成（验证 pipeline/backends 可用）。"""
import os, sys, io, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = pathlib.Path(r"C:/Users/annas/avwin/repo")
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["PYTHONUTF8"]="1"

import torch
print("[1] 用 pipeline.backends 跑视频", flush=True)
from pipeline.backends import cuda_memory, cuda_video

frames, height, width = 49, 384, 640
lat_t, lat_h, lat_w = cuda_video.latent_dims(height, width, frames)
tokens = lat_t*lat_h*lat_w
print("  latent %dx%dx%d = %d tokens" % (lat_t,lat_h,lat_w,tokens), flush=True)

latents = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
audio_latents = torch.randn(1, 41, 128, device="cuda", dtype=torch.bfloat16)
prompt_embeds = torch.randn(1, 32, 3840, device="cuda", dtype=torch.bfloat16)
audio_prompt_embeds = torch.randn(1, 32, 3840, device="cuda", dtype=torch.bfloat16)

print("\n[2] 分阶段：transformer -> 释放 -> VAE", flush=True)
out = pathlib.Path(r"C:/Users/annas/probes/backend_e2e.mp4")
path = cuda_video.generate_shot(
    latents=latents, audio_latents=audio_latents,
    prompt_embeds=prompt_embeds, audio_prompt_embeds=audio_prompt_embeds,
    timesteps=[1000.0, 750.0, 500.0],
    frames=frames, height=height, width=width, audio_frames=41,
    out_path=out, fps=24,
    offline=True,
)
print("\n[3] 产出:", path, os.path.getsize(path)/1024**2, "MB", flush=True)
print("    帧数=%d 尺寸=%dx%d" % (frames, width, height), flush=True)
