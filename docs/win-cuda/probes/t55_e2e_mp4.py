"""端到端：transformer 前向 -> 释放 -> VAE 解码 -> 写 mp4。"""
import os, sys, io, torch, time, gc, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from huggingface_hub import hf_hub_download

H,W,T,FPS = 12,20,7,24
N = H*W*T; N_a = 41

def vram(tag):
    free,_=torch.cuda.mem_get_info()
    print("  [%s] free=%.2fGB peak=%.2fGB" % (tag, free/2**30, torch.cuda.max_memory_allocated()/2**30), flush=True)

# --- 阶段1: transformer ---
print("[1] 载 transformer Q2_K", flush=True)
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q2_K.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c, quantization_config=qcfg,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
tr.eval(); torch.cuda.synchronize(); vram("transformer loaded")

print("[2] 跑 3 步去噪（模拟真实采样）", flush=True)
vc = tr.rope.prepare_video_coords(1,T,H,W,torch.device("cuda"))
ac = tr.audio_rope.prepare_audio_coords(1,N_a,torch.device("cuda"))
lat = torch.randn(1,N,128,device="cuda",dtype=torch.bfloat16)
aud = torch.randn(1,N_a,128,device="cuda",dtype=torch.bfloat16)
enc = torch.randn(1,32,3840,device="cuda",dtype=torch.bfloat16)
aenc = torch.randn(1,32,3840,device="cuda",dtype=torch.bfloat16)
vel=[]
t0=time.time()
for step in range(3):
    ts = torch.full((1,N), 1000.0*(1-step/3), device="cuda", dtype=torch.bfloat16)
    ats = torch.full((1,), 1000.0*(1-step/3), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        o = tr(hidden_states=lat, audio_hidden_states=aud,
               encoder_hidden_states=enc, audio_encoder_hidden_states=aenc,
               timestep=ts, audio_timestep=ats, video_coords=vc, audio_coords=ac,
               num_frames=T, height=H*32, width=W*32, audio_num_frames=N_a, return_dict=False)
    v = o[0] if isinstance(o, tuple) else o
    vel.append(v.float())
    print("  step%d %.2fs" % (step, time.time()-t0), flush=True)
vel = torch.stack(vel).mean(0).to(torch.bfloat16)
best = lat - vel   # 简化：当作 denoised latent
print("  去噪后 latent:", tuple(best.shape), flush=True)
del tr, o, vel; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
vram("after release")

# --- 阶段2: VAE ---
print("[3] 载 VAE 解码", flush=True)
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
vc_ = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/config.json", local_files_only=True)
vd  = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/diffusion_pytorch_model.safetensors", local_files_only=True)
vae = AutoencoderKLLTX2Video.from_single_file(vd, config=vc_, torch_dtype=torch.bfloat16).to("cuda")
vae.eval(); vram("vae loaded")
lat4d = best.reshape(1, 128, T, H, W)
t1=time.time()
with torch.no_grad():
    dec = vae.decode(lat4d, return_dict=False)[0]
torch.cuda.synchronize()
print("  解码 %.1fs shape=%s" % (time.time()-t1, tuple(dec.shape)), flush=True)
vram("after decode")

# --- 写 mp4 ---
print("[4] 写 mp4", flush=True)
frames = ((dec[0].float().clamp(-1,1) + 1) / 2)      # (3,F,H,W) -> [0,1]
frames = frames.permute(1,2,3,0).cpu().numpy()        # (F,H,W,3)
frames = (frames*255).astype("uint8")
print("  帧:", frames.shape, flush=True)
import imageio
out = r"C:/Users/annas/probes/e2e_test.mp4"
try:
    imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8)
    sz = os.path.getsize(out)/1024**2
    print("  [OK] %s  %.2fMB" % (out, sz), flush=True)
except Exception as e:
    print("  [imageio FAIL]", str(e)[:200], flush=True)
    # 退回 ffmpeg
    import numpy as np
    for i,f in enumerate(frames):
        imageio.imwrite(r"C:/Users/annas/probes/f_%03d.png"%i, f)
    print("  已写 PNG 序列", flush=True)
