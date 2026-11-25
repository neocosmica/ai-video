"""补装 imageio + ffmpeg 后端，完成 mp4 输出。"""
import subprocess, sys, io, os, torch, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"
PY=r"C:/Users/annas/avwin/.venv/Scripts/python.exe"
T="https://pypi.tuna.tsinghua.edu.cn/simple/"

print("[1] 装 imageio + imageio-ffmpeg", flush=True)
p=subprocess.run([PY,"-m","pip","install","imageio","imageio-ffmpeg","-i",T],
                 capture_output=True,text=True,timeout=1200,encoding="utf-8",errors="replace")
print("  rc=%d %s" % (p.returncode, ((p.stdout or "")[-250:]).replace("\n"," ")), flush=True)

# 复用已有的 transformer/VAE 流程，直接写 mp4
print("\n[2] 完整链路 -> mp4", flush=True)
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"; os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from huggingface_hub import hf_hub_download
H,W,Tf,FPS = 12,20,7,24
N=H*W*Tf; N_a=41

from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q2_K.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c, quantization_config=qcfg,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
tr.eval()
vc = tr.rope.prepare_video_coords(1,Tf,H,W,torch.device("cuda"))
ac = tr.audio_rope.prepare_audio_coords(1,N_a,torch.device("cuda"))
lat = torch.randn(1,N,128,device="cuda",dtype=torch.bfloat16)
enc = torch.randn(1,32,3840,device="cuda",dtype=torch.bfloat16)
with torch.no_grad():
    o = tr(hidden_states=lat, audio_hidden_states=torch.randn(1,N_a,128,device="cuda",dtype=torch.bfloat16),
           encoder_hidden_states=enc, audio_encoder_hidden_states=enc,
           timestep=torch.full((1,N),500.0,device="cuda",dtype=torch.bfloat16),
           audio_timestep=torch.full((1,),500.0,device="cuda",dtype=torch.bfloat16),
           video_coords=vc, audio_coords=ac, num_frames=Tf, height=H*32, width=W*32,
           audio_num_frames=N_a, return_dict=False)
print("  transformer OK", flush=True)
del tr, o; gc.collect(); torch.cuda.empty_cache()

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
vc2 = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/config.json", local_files_only=True)
vd2 = hf_hub_download(repo_id="Lightricks/LTX-2", filename="vae/diffusion_pytorch_model.safetensors", local_files_only=True)
vae = AutoencoderKLLTX2Video.from_single_file(vd2, config=vc2, torch_dtype=torch.bfloat16).to("cuda")
vae.eval()
with torch.no_grad():
    dec = vae.decode(lat.reshape(1,128,Tf,H,W).to(torch.bfloat16), return_dict=False)[0]
print("  解码 OK", tuple(dec.shape), flush=True)

import imageio, numpy as np
frames = ((dec[0].float().clamp(-1,1)+1)/2).permute(1,2,3,0).cpu().numpy()
frames = (frames*255).astype("uint8")
out = r"C:/Users/annas/probes/e2e_test.mp4"
imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
print("  [OK] mp4: %s %.2fMB %d帧 %dx%d @%dps" % (
    out, os.path.getsize(out)/1024**2, len(frames), frames.shape[2], frames.shape[1], FPS), flush=True)
