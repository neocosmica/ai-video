# Windows + NVIDIA 部署与使用

本文档对应 `feat/windows-nvidia` 分支：在 Windows 上用 **PyTorch / diffusers 原生 CUDA 栈**
跑完整流程，Mac 侧 MLX 行为保持不变。

> 想先了解**整体方案**（分层设计、模型选型、技术障碍、验收证据、改动面、遗留项），
> 见 [`DESIGN.md`](DESIGN.md)。本文只讲怎么装、怎么用。
> 阶段 0 实测数据见 [`PHASE0-PROBE.md`](PHASE0-PROBE.md)。

## 为什么是两套栈

Mac 的模型全是 MLX 格式（`mlx-community/*-4bit`、`mflux-community/*`、
`dgrauet/ltx-*-mlx-*`），CUDA 读不了；反过来 PyTorch 权重在 Mac 没有 Metal 加速等价物。
所以不是「换个 flag」，而是**两套权重 + 两套运行时**：

```
pipeline/
  platform.py          # 跨平台差异集中处（venv 路径、内存、字体、进程、壳命令）
  backends/
    __init__.py        # auto / mlx / cuda 判定
    cuda_memory.py     # 显存编排：分阶段插载 + 段间释放
    cuda_storyboard.py # 分镜 Qwen3.5-4B
    cuda_images.py     # 静帧 FLUX.2 Klein 4B
    cuda_audio.py      # 对白 Qwen3-TTS（独立 venv）+ 配乐 MusicGen
    cuda_textenc.py    # 文本编码 Gemma3（编码后立刻释放）
    cuda_video.py      # 视频 LTX-2 19B GGUF
```

公共逻辑（项目、分镜校验、时间轴、混音、组装）留在 `pipeline/` 顶层，两端共用。

## 安装

```powershell
# 首次部署
.\scripts\setup-win.ps1

# 环境检查
.\scripts\run-win.ps1 doctor
```

`setup-win.ps1` 会：
1. 建主 venv，装 `torch cu128` + `diffusers` + `transformers>=5` + `bitsandbytes`
2. **装 `triton-windows`**（Windows 无官方 triton，缺了 Qwen3.5 只有 12.5 tok/s）
3. 建 **TTS 独立 venv**（`qwen-tts` 钉 `transformers==4.57.3`，与主环境冲突）
4. 下载 gyan.dev 静态 `ffmpeg` 到 `tools/`
5. 冒烟：CUDA 可用性 + `doctor`

## 模型与显存（RTX 4070 12 GB 实测）

| 阶段 | 模型 | 峰值显存 | 耗时 |
|---|---|---|---|
| 分镜 | Qwen3.5-4B 4bit | 3.2 GB | ~60 s |
| 静帧 | FLUX.2 Klein 4B | 7.8 GB | 15 s/张 |
| 对白 | Qwen3-TTS-1.7B | 4.0 GB | 18 s/句 |
| 配乐 | MusicGen small | 1.2 GB | 7-20 s |
| 文本编码 | Gemma3 12B 4bit | 7.8 GB | ~90 s 首次 |
| 视频 | LTX-2 19B Q2_K GGUF | 8.5 GB | 2.3 s/步 |

**关键**：每阶段独立串行 + 段间释放，全局峰值 = 最重阶段，不累加。
视频内部再分三段（文本编码 → transformer → VAE），实测释放后 free 回到 10.8 GiB。

## 使用

```powershell
# 网页控制台（与 Mac 相同）
.\scripts\run-win.ps1

# 命令行成片
.\scripts\run-win.ps1 produce --project demo

# 单阶段
.\scripts\run-win.ps1 storyboard --project demo --force
.\scripts\run-win.ps1 images   --project demo --force
.\scripts\run-win.ps1 audio    --project demo --force
.\scripts\run-win.ps1 videos   --project demo --force
.\scripts\run-win.ps1 music    --project demo --force
.\scripts\run-win.ps1 assemble --project demo --force
```

项目路径、产物结构、分镜字段约束与 Mac 端**完全一致**（见根目录 `AGENTS.md`）。

## 配置

`config.json` 相关字段：

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

环境变量：
- `NOVEL_VIDEO_BACKEND`：覆盖 `backend`（`auto` / `mlx` / `cuda`）
- `NOVEL_VIDEO_TTS_PYTHON`：TTS 独立解释器路径

## 视频 I2V 正确性

视频阶段使用静帧图作为 LTX-2 I2V conditioning：

1. VAE encode 单张静帧（不要先复制成 49 帧再编码；causal 时间压缩窗口会错位）
2. 按 VAE 的 `latents_mean/std` 归一化，再按 LTX token 顺序 pack
3. 首帧 latent 固定为 conditioning，mask 令其 timestep=0；每个去噪步骤都锁回首帧
4. 使用 token 数计算 scheduler shift；transformer 输出按 diffusers LTX2 约定处理为 velocity
5. 最终 unpack + denormalize，再 VAE decode

曾出现网格状伪影的错误实现有两处：忽略 normalize；以及把 LTX transformer 输出误当作 x0。
当前实测抽帧已恢复真实画面（人物、便利店环境、雨夜光照均保留）。

## 已知 Windows 专有坑（都已修）

| 坑 | 现象 | 解法 |
|---|---|---|
| `triton` 无官方 wheel | Qwen 只有 12.5 tok/s | 装 `triton-windows` |
| `os.sysconf` 不存在 | 读内存崩 | `platform.total_memory_bytes()` 用 `GlobalMemoryStatusEx` |
| `os.getpgid` 不存在 | produce 第一步就崩（AttributeError 不是 OSError） | `getattr` 探测 |
| `import pty` | 模块导入即崩 | 延迟导入 |
| ffmpeg 路径缺 `.exe` | 找不到 ffmpeg | `config.py` 自动补后缀 |
| **ASS 字幕 `fontsdir`** | 盘符冒号破坏滤镜解析，三种转义全失败 | Windows 不带 `fontsdir`，走系统字体 |
| `qwen-tts` 版本冲突 | `check_model_inputs()` TypeError | TTS 用独立 venv |
| MusicGen `generator` 参数 | transformers 5.x 不认 | 改 `torch.manual_seed` |
| MusicGen `pad_token_id` | eos=2048 超词表 → CUDA device-side assert | 不传该参数 |
| 分镜漏台词 | screenplay 硬校验拦截 | 提示词列全台词 + 增量补镜 + 兜底补空镜 |

## 与 Mac 的质量差异

| 阶段 | Mac (MLX) | Windows (CUDA) |
|---|---|---|
| 视频量化 | 3.92 bit/param（4bit group64，仅 transformer） | **3.15 bit/param（GGUF Q2_K）** |
| 其他阶段 | 4bit / 8bit | 4bit / bf16（相当或更好） |

视频档位偏低是因为 12 GB 显存装不下更高档。要追平可用 `Q4_K_M`（13.2 GB）配逐层
offload（权重留 CPU、按层上卡），代价是速度慢数倍：

```python
from pipeline.backends.cuda_memory import offload_layers_hook
model = load_transformer(gguf_file="ltx-2-19b-dev-Q4_K_M.gguf")
offload_layers_hook(model)
```

## 调试

```powershell
.\scripts\run-win.ps1 doctor          # 平台/显存/后端/字体/缓存
.\scripts\run-win.ps1 selftest        # 逻辑自检
```

失败先看 `doctor`；再看项目 `work/logs/` 与 `work/webui/finished.json`。
探针脚本与实测记录见 `docs/win-cuda/PHASE0-PROBE.md`、`docs/win-cuda/probes/`。
