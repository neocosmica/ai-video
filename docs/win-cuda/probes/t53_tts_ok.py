"""TTS 正确用法实测。"""
import os, sys, io, torch, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["HF_HUB_DISABLE_XET"]="1"

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print("[1] 加载模型", flush=True)
t0=time.time()
m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0", torch_dtype=torch.bfloat16)
print("  %.0fs VRAM=%.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/2**30), flush=True)
print("  speakers:", m.get_supported_speakers(), flush=True)
print("  languages:", m.get_supported_languages(), flush=True)
torch.cuda.reset_peak_memory_stats()

print("\n[2] 合成对白", flush=True)
text = "你没事吧？有没有被威胁？没有做傻事吧？没有受伤吧？"
t1=time.time()
wavs, sr = m.generate_custom_voice(text=text, speaker="Serena", language="Chinese")
dur = len(wavs[0])/sr
print("  [OK] %.1fs sr=%d 音频=%.2fs" % (time.time()-t1, sr, dur), flush=True)
import soundfile as sf, numpy as np
out = r"C:/Users/annas/probes/tts_test.wav"
sf.write(out, np.asarray(wavs[0]), sr)
print("  保存:", out, flush=True)
print("  [VRAM] peak=%.2fGB" % (torch.cuda.max_memory_allocated()/2**30), flush=True)
free,tot=torch.cuda.mem_get_info(); print("  free=%.2fGB" % (free/2**30), flush=True)
