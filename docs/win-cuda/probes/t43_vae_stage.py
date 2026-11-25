"""关键：验证 Q2_K transformer 跑完后释放，再载 VAE 解码的显存编排。"""
import os, sys, io, torch, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from huggingface_hub import hf_hub_download

print("[1] 载 transformer Q2_K", flush=True)
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q2_K.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c, quantization_config=qcfg,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
tr.eval(); torch.cuda.synchronize()
free,tot=torch.cuda.mem_get_info()
print("  transformer 上卡 free=%.2fGB" % (free/2**30), flush=True)

print("[2] 跑一次前向", flush=True)
H,W,T=12,20,7; N=H*W*T; N_a=41
vc = tr.rope.prepare_video_coords(1,T,H,W,torch.device("cuda"))
ac = tr.audio_rope.prepare_audio_coords(1,N_a,torch.device("cuda"))
with torch.no_grad():
    out = tr(hidden_states=torch.randn(1,N,128,device="cuda",dtype=torch.bfloat16),
             audio_hidden_states=torch.randn(1,N_a,128,device="cuda",dtype=torch.bfloat16),
             encoder_hidden_states=torch.randn(1,32,3840,device="cuda",dtype=torch.bfloat16),
             audio_encoder_hidden_states=torch.randn(1,32,3840,device="cuda",dtype=torch.bfloat16),
             timestep=torch.full((1,N),1000.0,device="cuda",dtype=torch.bfloat16),
             audio_timestep=torch.full((1,),1000.0,device="cuda",dtype=torch.bfloat16),
             video_coords=vc, audio_coords=ac, num_frames=T, height=H*32, width=W*32,
             audio_num_frames=N_a, return_dict=False)
torch.cuda.synchronize(); print("  前向 OK", flush=True)

print("[3] 释放 transformer", flush=True)
del tr, out; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
free,tot=torch.cuda.mem_get_info()
print("  释放后 free=%.2fGB" % (free/2**30), flush=True)

print("[4] 载 VAE 解码", flush=True)
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
vc_ = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/config.json", local_files_only=True)
vd = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/diffusion_pytorch_model.safetensors", local_files_only=True)
try:
    vae = AutoencoderKLLTX2Video.from_single_file(vd, config=vc_, torch_dtype=torch.bfloat16)
    vae = vae.to("cuda"); vae.eval()
    torch.cuda.synchronize()
    free,tot=torch.cuda.mem_get_info()
    print("  VAE 上卡 free=%.2fGB  peak=%.2fGB" % (free/2**30, torch.cuda.max_memory_allocated()/2**30), flush=True)
    print("  [OK] 分阶段编排可行", flush=True)
except Exception as e:
    print("  [FAIL]", type(e).__name__, str(e)[:250], flush=True)
