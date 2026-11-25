"""阶段0 探针最终报告。"""
import os, sys, io, json, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"]="1"

r = {
 "phase": "0-feasibility",
 "machine": {"os":"Windows 11 10.0.26300","cpu":"i5-12400F","ram_gb":31.8,"free_disk_gb":288},
 "cuda": {"device":torch.cuda.get_device_properties(0).name,
          "vram_gb":round(torch.cuda.get_device_properties(0).total_memory/2**30,2),
          "cc":"8.9","torch":torch.__version__,
          "bf16_tflops":34.3},
 "installed": {"diffusers":"0.43.0","transformers":"5.24.1","accelerate":"1.16.0",
               "bitsandbytes":"0.50.2","triton-windows":"3.9.0","flash-linear-attention":"0.6.0"},
 "results": [
  {"stage":"环境","item":"torch cu128 + CUDA","status":"PASS","detail":"RTX 4070 11.99GB, 34.3 TFLOPS bf16"},
  {"stage":"环境","item":"triton-windows","status":"PASS","detail":"3.9.0 可装，解决 Windows 无 triton 的死结"},
  {"stage":"环境","item":"bitsandbytes 4bit nf4","status":"PASS","detail":"可用；小模型 4bit 仅 1.67GB"},
  {"stage":"分镜","item":"Qwen3.5-4B 4bit 生成","status":"PARTIAL",
   "detail":"能出文本；默认思考链导致 JSON 空壳；bf16 17.8 tok/s（triton 装后）"},
  {"stage":"视频","item":"LTX-2 19B Q3_K_S GGUF 加载","status":"PASS","detail":"9.7s，11.83B 参数"},
  {"stage":"视频","item":"LTX-2 19B 全量上卡","status":"FAIL",
   "detail":"11.04GB(Q4_K_S) 上卡后 free=0.00GB，OOM"},
  {"stage":"视频","item":"LTX-2 19B Q3_K_S 前向","status":"PASS",
   "detail":"2.78s/step，峰值 9.76GB，free 0.08GB（余量极小）"},
  {"stage":"视频","item":"LTX-2.5 + diffusers 0.43","status":"FAIL",
   "detail":"架构不兼容：2.5 需 9 行 mod 表（audio_cross_attn_mod），diffusers 建 6 行"},
 ],
 "blockers": [
   "LTX-2.5 GGUF 无法用 diffusers 0.43 加载（需 audio_cross_attn_mod=True，但 low_cpu_mem_usage 与量化冲突）",
   "LTX-2 19B Q3_K_S 峰值 9.76GB，只剩 0.08GB 余量，VAE 解码需额外显存 → 必须 offload 或用 Q2_K",
   "Lightricks/LTX-2.5-Diffusers 与 flux-3-image-klein 均 gated，无 HF token 无法下载",
   "Qwen3.5 默认输出思考链，需 disable thinking 才能稳定出 JSON",
 ],
 "next": [
   "验证 Q2_K (7.54GB) 显存余量是否够 VAE 解码",
   "验证完整 LTX2Pipeline（含 VAE/connector/text_encoder）的 offload 编排",
   "验证 FLUX.3 GGUF 在 diffusers 内加载路径",
   "分镜加 enable_thinking=False 或换 Qwen3.5-2B",
 ],
}
p = os.path.join(os.path.dirname(__file__), "phase0_report.json")
json.dump(r, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("报告已写入:", p, flush=True)
print(open(p, encoding="utf-8").read()[:3000], flush=True)
