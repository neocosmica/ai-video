"""LTX-2 19B VAE 解码端到端：latent -> 视频帧 -> mp4。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from huggingface_hub import hf_hub_download

print("[1] 载 VAE", flush=True)
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
vc = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/config.json", local_files_only=True)
vd = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/diffusion_pytorch_model.safetensors", local_files_only=True)
t0=time.time()
vae = AutoencoderKLLTX2Video.from_single_file(vd, config=vc, torch_dtype=torch.bfloat16).to("cuda")
vae.eval(); torch.cuda.synchronize()
print("  %.0fs VRAM=%.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/2**30), flush=True)
print("  config: scaling=%.4f, latents=%s" % (vae.config.scaling_factor, getattr(vae.config,"latent_channels","?")) if hasattr(vae,"config") else "", flush=True)
torch.cuda.reset_peak_memory_stats()

print("[2] 解码随机 latent (7x12x20, 384x640)", flush=True)
H,W,T = 12,20,7
lat = torch.randn(1, 128, T, H, W, device="cuda", dtype=torch.bfloat16) * 0.5
t1=time.time()
try:
    with torch.no_grad():
        out = vae.decode(lat, return_dict=False)
    torch.cuda.synchronize()
    dec = out[0] if isinstance(out, tuple) else out
    print("  [OK] %.1fs" % (time.time()-t1), flush=True)
    print("  输出 shape:", tuple(dec.shape), "dtype:", dec.dtype, flush=True)
    print("  值域: [%.2f, %.2f]" % (dec.min(), dec.max()), flush=True)
    print("  [VRAM] peak=%.2fGB free=%.2fGB" % (
        torch.cuda.max_memory_allocated()/2**30, torch.cuda.mem_get_info()[0]/2**30), flush=True)
except Exception as e:
    print("  [FAIL]", type(e).__name__, flush=True); print("  "+str(e)[:450], flush=True)
