# 推理后端分叉

Mac 走 **MLX**（原有实现，行为不变），Windows/Linux + NVIDIA 走 **CUDA**（PyTorch）。
两边不是「同一份代码换个 flag」，而是两套权重 + 两套运行时。

## 目录

| 文件 | 作用 |
|---|---|
| `__init__.py` | 后端判定（`auto` / `mlx` / `cuda`）+ CUDA 硬校验 |
| `cuda_memory.py` | **显存编排**：分阶段插载、段间释放（Mac 补丁思路的 CUDA 版） |
| `mlx_memory.py` | MLX 侧对称封装（`aggressive_cleanup` / `get_memory_stats`） |
| `cuda_storyboard.py` | 分镜：Qwen3.5-4B（**必须 `enable_thinking=False`**） |
| `cuda_images.py` | 静帧：FLUX.2 Klein 4B + `enable_model_cpu_offload` |
| `cuda_audio.py` | 对白 Qwen3-TTS（**独立 venv 子进程**）+ 配乐 MusicGen |
| `cuda_video.py` | 视频：LTX-2 19B GGUF + transformer→释放→VAE 分阶段 |

## 显存账（RTX 4070 12 GB 实测）

| 阶段 | 模型 | 峰值 |
|---|---|---|
| 分镜 | Qwen3.5-4B 4bit | 3.4 GB |
| 静帧 | FLUX.2 Klein 4B | 7.80 GB |
| 对白 | Qwen3-TTS-1.7B | 4.04 GB |
| 配乐 | MusicGen small | 1.19 GB |
| 视频 | LTX-2 19B Q2_K | 8.57 GB |

关键：**每阶段独立串行 + 段间释放**，全局峰值取最重阶段而非累加。
视频内部再分两段（transformer → VAE），实测释放后 free 从 3.1 → 10.8 GiB。

## 为什么要分阶段（对齐 Mac 端补丁）

`scripts/apply_ltx_gemma_patch.py` 在 Mac 上把 Gemma（~8 GB）与 connector（~6 GB）
拆开、各段用完立刻 `aggressive_cleanup()`，峰值从 ~14 GB 压到 max(单段)，
这才让 16 GB Mac 跑得动 22B。

CUDA 侧同理：

```
with cuda_memory.stage("videos/transformer"):
    model = load_transformer(...)   # 只驻留 transformer
    latents = denoise(model, ...)
    del model                        # 出块后引用归零
# 段间自动 empty_cache()
with cuda_memory.stage("videos/vae"):
    vae = load_vae(...)
    frames = decode_video(vae, latents, ...)
```

## 关键实测结论（详见 `docs/win-cuda/PHASE0-PROBE.md`）

1. **Windows 无官方 triton** → 装 `triton-windows`，否则 Qwen 只有 12.5 tok/s
2. **Q3_K_S 余量仅 0.08 GB**，接 VAE 必崩；**Q2_K 是 12 GB 唯一可跑档位**
3. **TTS 必须独立 venv**：`qwen-tts` 钉 `transformers==4.57.3`，与 LTX2 的 `>=5` 冲突
4. **LTX-2.5 与 diffusers 0.43 架构不兼容**（9 vs 6 行 mod 表）→ 用 LTX-2 19B

## 配置

`config.json`：

```json
{
  "backend": "auto",
  "cuda_video_gguf": "ltx-2-19b-dev-Q2_K.gguf",
  "cuda_video_gguf_repo": "unsloth/LTX-2-GGUF",
  "cuda_conf_repo": "Lightricks/LTX-2",
  "cuda_image_model": "black-forest-labs/FLUX.2-klein-4B",
  "cuda_llm_model": "Qwen/Qwen3.5-4B",
  "cuda_llm_quant": "4bit",
  "cuda_tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
}
```

环境变量 `NOVEL_VIDEO_BACKEND` 可覆盖 `backend`；`NOVEL_VIDEO_TTS_PYTHON`
指定 TTS 独立解释器。

## 想追平 Mac 画质

Mac 端 `ltx-2.3-mlx-q4` 是 3.92 bit/param；CUDA 侧 Q2_K 约 3.15。
要追平可用 `Q4_K_M`（13.22 GB）配逐层 offload：

```python
from pipeline.backends.cuda_memory import offload_layers_hook
model = load_transformer(gguf_file="ltx-2-19b-dev-Q4_K_M.gguf")
offload_layers_hook(model)      # 权重留 CPU，按层上卡
```

代价是 PCIe 来回搬，速度会慢数倍。
