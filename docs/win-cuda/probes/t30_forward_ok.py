"""决定性：官方坐标 API 到位，验证 LTX-2 19B Q3_K_S 前向 + 显存峰值。"""
import os, sys, io, torch, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from huggingface_hub import hf_hub_download
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

g = hf_hub_download(repo_id="unsloth/LTX-2-GGUF", filename="ltx-2-19b-dev-Q3_K_S.gguf", local_files_only=True)
c = hf_hub_download(repo_id="Lightricks/LTX-2", filename="transformer/config.json", local_files_only=True)
qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
tr = LTX2VideoTransformer3DModel.from_single_file(g, config=c, quantization_config=qcfg,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
tr = tr.to("cuda"); tr.eval()
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

H,W,T = 12, 20, 7
N = H*W*T; N_a = 41
vc = tr.rope.prepare_video_coords(1, T, H, W, torch.device("cuda"))
ac = tr.audio_rope.prepare_audio_coords(1, N_a, torch.device("cuda"))
print("[1] coords:", tuple(vc.shape), tuple(ac.shape), flush=True)

hidden = torch.randn(1, N, 128, device="cuda", dtype=torch.bfloat16)
enc = torch.randn(1, 32, 3840, device="cuda", dtype=torch.bfloat16)
aud_h = torch.randn(1, N_a, 128, device="cuda", dtype=torch.bfloat16)
aud_e = torch.randn(1, 32, 2048, device="cuda", dtype=torch.bfloat16)
ts = torch.full((1, N), 1000.0, device="cuda", dtype=torch.bfloat16)
ats = torch.full((1,), 1000.0, device="cuda", dtype=torch.bfloat16)

print("[2] 前向 x2 ...", flush=True)
times=[]
for run in range(2):
    torch.cuda.synchronize(); t0=time.time()
    with torch.no_grad():
        out = tr(hidden_states=hidden, audio_hidden_states=aud_h,
                 encoder_hidden_states=enc, audio_encoder_hidden_states=aud_e,
                 timestep=ts, audio_timestep=ats,
                 video_coords=vc, audio_coords=ac,
                 num_frames=T, height=H*32, width=W*32, audio_num_frames=N_a,
                 return_dict=False)
    torch.cuda.synchronize()
    dt=time.time()-t0; times.append(dt)
    print("  run%d %.2fs" % (run, dt), flush=True)
print("[OK] 前向通过 平均 %.2fs/step" % (sum(times)/len(times)), flush=True)
print("[VRAM] peak=%.2fGB reserved=%.2fGB" % (torch.cuda.max_memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30), flush=True)
free,tot=torch.cuda.mem_get_info(); print("[VRAM] free=%.2fGB" % (free/2**30), flush=True)
