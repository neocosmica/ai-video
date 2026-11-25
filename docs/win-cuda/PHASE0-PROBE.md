# Windows + NVIDIA（CUDA）支持：阶段 0 探针报告

探针日期：2026-11-25
开发机：`annas@192.168.0.147`（Windows 11，RTX 4070）
分支：`feat/windows-nvidia`

本报告记录「是否可行」的实测结论，不含推测。每项均为开发机上真实跑过的结果。

---

## 1. 硬件与系统

| 项 | 实测值 |
|---|---|
| 系统 | Windows 11 `10.0.26300` |
| CPU | Intel i5-12400F（6 核） |
| 内存 | 31.8 GB |
| GPU | **NVIDIA GeForce RTX 4070** |
| 显存 | **11.99 GB**（可用 11.99，实际 `mem_get_info` 报 10.7–10.9 可用） |
| 计算能力 | `8.9`（Ada） |
| 磁盘 | C: 剩余 181 GB |
| Python | 3.11.9 |

CUDA 基准（`torch.randn(4096,4096,bf16)` 单次矩阵乘）：**34.3 TFLOPS**。

## 2. 环境依赖（全部实测可装）

| 包 | 版本 | 备注 |
|---|---|---|
| torch | `2.11.0+cu128` | 走 `download.pytorch.org/whl/cu128`，cp311/windows 有官方 wheel |
| diffusers | `0.43.0` | 内置 `ltx2/`、`flux2/`、`flux3` transformer 模块 |
| transformers | `5.24.1` | 含 `Gemma3ForConditionalGeneration`、`Gemma4ForConditionalGeneration` |
| accelerate | `1.16.0` | offload 必需 |
| bitsandbytes | `0.50.2` | **Windows 有官方 wheel**，nf4 可用 |
| **triton-windows** | `3.9.0.post30` | **关键**：Windows 无官方 triton；此社区版装上后 fla 才生效 |
| flash-linear-attention | `0.6.0` | Qwen3.5 线性注意力加速，依赖 triton |

### 关键发现：triton 是 Windows 上的死结

Windows 上**没有官方 triton**（`pip install triton` 报 `No matching distribution`）。
后果：`flash-linear-attention` 全部退化成 PyTorch 参考实现，Qwen3.5 只有 **12.5 tok/s**。

装 `triton-windows==3.9.0.post30` 后：
- `fla.ops.gated_delta_rule` 可导入
- Qwen3.5 提速到 **17.8 tok/s**

**这是 Windows 侧必须写进安装脚本的依赖，不能漏。**

## 3. 各阶段实测结果

### 3.1 分镜 LLM：Qwen3.5-4B ✅（需关 thinking）

| 配置 | 速度 | 显存 | JSON |
|---|---|---|---|
| bf16 不量化 | 17.8 tok/s | 7.9 GB | ✅（关 thinking 后） |
| bnb 4bit nf4 | — | ~2.4 GB | ✅ |

**关键问题**：Qwen3.5 默认输出 `<think>` 思考链，导致 JSON 被思考内容挤占而截断（实测 `shots=0`）。

**修复**（实测有效）：
```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,      # ← 关键
)
```
关闭后 prompt 尾部生成 `<think>\n\n</think>\n\n`，模型直接输出纯 JSON：
```
输出: '{"title":"测试","n":3}\n'
```

### 3.2 图像：FLUX.2 Klein 4B ✅（已实测出图）

```python
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()   # seq: text_encoder->transformer->vae
```

| 指标 | 值 |
|---|---|
| 首次加载 | 244 s（含下载） |
| 出图 576×1024，4 步 | **15 s** |
| 显存峰值 | **7.80 GB** |
| 余量 | 10.78 GB free |

产出图确认：写实华人面孔、雨夜便利店暖光、电影感，符合竖屏短剧静帧需求。

> 注：`black-forest-labs/FLUX.2-klein-4B` 是 **非 gated**（`gated=False`），可直接下载。
> `FLUX.2-klein-9B` 与 `FLUX.3` 官方仓库均为 `gated: auto`，无 HF token 时 403。

### 3.3 配乐：MusicGen small ✅

```python
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", dtype=torch.bfloat16).to("cuda")
```

| 指标 | 值 |
|---|---|
| 加载 | 70 s（含下载） |
| 生成（256 tokens ≈ 5.06 s 音频） | **7 s** |
| 显存峰值 | **1.19 GB** |

### 3.3b 对白 TTS：Qwen3-TTS ✅（**必须独立 venv**）

**这是整份探针里最重要的架构发现。**

`qwen-tts==0.1.1` 钉死 `transformers==4.57.3`，而 diffusers 0.43 的 LTX2 管道需要
`transformers>=5.x`（`Gemma4ForConditionalGeneration` 等）。两者**无法共存于同一 venv**：

```
transformers 5.24.1 → qwen_tts 导入即崩：
  TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

**解法：TTS 用独立 venv**（与 Mac 端 `venvs/main` / `venvs/ltx` / `venvs/musicgen`
三环境并存的思路一致）。

```
C:/Users/annas/avwin/.venv        ← 主环境：torch 2.11+cu128, transformers 5.24, diffusers 0.43
C:/Users/annas/avwin/venv-tts     ← TTS：torch 2.11+cu128, transformers 4.57.3, qwen-tts
```

实测结果：

| 指标 | 值 |
|---|---|
| 加载 Qwen3-TTS-12Hz-1.7B-CustomVoice | 30 s，显存 3.89 GB |
| 合成 6.88 s 中文对白 | **18.3 s** |
| 显存峰值 | **4.04 GB** |
| 输出 | 24 kHz 单声道 PCM 16bit |

支持音色（与项目 `voices.py` 对齐）：`serena` `vivian` `dylan` `uncle_fu` `eric`
及 `aiden` `eric` `ono_anna` `ryan` `sohee`。
支持语言含 `chinese` / `english` / `auto`。

正确调用方式（`device` 传 `device_map`，方法名是 `generate_custom_voice`）：

```python
m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0", torch_dtype=torch.bfloat16)
wavs, sr = m.generate_custom_voice(text="...", speaker="Serena", language="Chinese")
```

### 3.4 视频：LTX-2 19B（GGUF）⚠️ 可行但档位受限

这是整个方案最关键的一段，结论如下。

**架构兼容性**：diffusers 0.43 的 `LTX2VideoTransformer3DModel` 对应 **LTX-2（19B）**，
与 `Lightricks/LTX-2`（非 gated）匹配。加载方式：

```python
from diffusers import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

qcfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
transformer = LTX2VideoTransformer3DModel.from_single_file(
    gguf_path,
    config="<Lightricks/LTX-2>/transformer/config.json",
    quantization_config=qcfg,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,          # 量化必须为 True
)
```

实测结果（384×640，49 帧，latent 7×12×20 = 1680 tokens）：

| 档位 | 权重体积 | 前向峰值 | 余量 | 每步耗时 |
|---|---|---|---|---|
| Q4_K_S | 11.04 GB | **OOM** | 0.00 GB | ✗ |
| Q3_K_S | 8.82 GB | 9.76 GB | **0.08 GB** | 2.78 s |
| **Q2_K** | **7.54 GB** | **8.57 GB** | **1.51 GB** | **2.33 s** |

**结论**：
- `Q3_K_S` 前向能过，但余量仅 0.08 GB，接 VAE 解码必崩
- **`Q2_K` 是目前 12 GB 卡上唯一能跑完整流程的档位**
- `Q2_K` 约 3.15 bit/param；Mac 端 MLX 是 3.92 bit/param（`quantize_config.json`: 4bit/group64/仅 transformer）

**分阶段编排已验证可行**（这是 Mac 端 Gemma 补丁思路在 CUDA 上的等价实现）：

```
[1] transformer Q2_K 上卡          free = 3.12 GB
[2] 前向（1680 tokens）            通过
[3] del transformer + empty_cache  free = 10.78 GB   ← 完全回收
[4] VAE 上卡                      free = 8.47 GB, peak 8.57 GB
```

说明「跑完 transformer 立刻释放 → 再载 VAE 解码」的串行编排在 12 GB 卡上完全成立，
峰值 = max(单段) 而非 sum(所有段)，与 Mac 16 GB 上跑 22B 的原理一致。

**LTX-2.5 不可用（实测）**：
`unsloth/LTX-2.5-GGUF` 的权重需要 **9 行** mod 表（`audio_cross_attn_mod=True`），
diffusers 0.43 默认建 **6 行**。自建 config 能让表对上，但
`quantization` 与 `low_cpu_mem_usage=False` 互斥，最终卡在 meta tensor。
且 `Lightricks/LTX-2.5-Diffusers` 是 **gated**，无法取官方 config。

### 3.5 文本编码器：Gemma3 ✅（4bit 后必须分阶段释放）

LTX-2 的 `model_index.json` 声明 `text_encoder: Gemma3ForConditionalGeneration`。

| 量化 | 显存峰值 | 结论 |
|---|---|---|
| bf16 | 约 22 GB | 装不下 |
| **4bit nf4** | **7.77 GB** | ✅ 可用（6.60B 参数） |

**关键坑**：`hidden_states` 是元组，若继续持有引用，`del model` 不会真正释放显存。
实测错误写法释放后 free 只回到 3.00 GB（因为 hidden_states 还在 GPU）。
正确做法是**先把 embeds `.cpu()` 落到主机内存，再删所有引用**：

```python
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
hidden = out.hidden_states[-1].detach().to(torch.bfloat16)
embeds = hidden.cpu()          # ← 关键：先落主机内存
del out, hidden, ids
cuda_memory.empty_cache()
```

**三阶段编排实测**（这是 12 GB 跑通完整链路的核心证据）：

```
段1 载入 text_encoder   free 3.02 GB  peak 7.77 GB
    编码 → embeds 落 CPU
    释放                free 10.79 GB   ← 完全回收
段2 载入 transformer    free 3.07 GB  peak 7.69 GB
    释放                free 10.79 GB
段3 载入 VAE            peak 3.64 GB
```

**峰值 = max(单段) = 7.77 GB，不累加。** 若三段同时驻留需 7.77+8.57+3.64 ≈ 20 GB，
12 GB 卡必爆。

## 4. 已被证伪的路线

| 路线 | 结果 |
|---|---|
| MLX-on-Windows（`mlx-cuda-13`） | 有 wheel，但 mlx extras 在 Windows 不生效，且生态全是 mlx 量化权重，CUDA 读不了 → 放弃 |
| LTX-2.5 + diffusers 0.43 | 架构不兼容（9 vs 6 行表）+ gated → 放弃 |
| FLUX.3 GGUF + diffusers | `Flux3Transformer2DModel` 存在，但 GGUF 是 ComfyUI 命名（`content_mode_blocks.image.*`），需自建映射；官方 config gated |
| `device_map="auto"` | `LTX2VideoTransformer3DModel` 缺 `_no_split_modules` → 不可用 |
| 逐层 `cpu_offload` hook | GGUF 参数可挂钩，但需配合完整 pipeline 才生效 |

## 5. 全部阶段汇总

| 阶段 | 模型 | 状态 | 显存峰值 | 耗时 |
|---|---|---|---|---|
| 分镜 | Qwen3.5-4B（4bit / bf16） | ✅ | 3.4 GB | 17.8 tok/s |
| 静帧 | FLUX.2 Klein 4B + offload | ✅ | 7.80 GB | 15 s/张 |
| 对白 | Qwen3-TTS-1.7B（独立 venv） | ✅ | 4.04 GB | 18.3 s/6.9s |
| 配乐 | MusicGen small | ✅ | 1.19 GB | 7 s |
| 视频 | LTX-2 19B Q2_K GGUF | ✅ | 8.57 GB | 2.33 s/step |
| 视频 VAE | LTX-2 19B VAE（分阶段） | ✅ | 8.57 GB | — |

**关键的显存结论**：各阶段独立串行、每段结束释放，则全局峰值由最重的阶段决定
（视频 ~8.6 GB），而非累加。12 GB 卡有余量，这是方案成立的基础。

## 6. 环境布局（Windows 双 venv）

```
avwin/
  .venv/      主环境：torch 2.11+cu128, diffusers 0.43, transformers 5.24,
              accelerate 1.16, bitsandbytes 0.50.2, triton-windows 3.9, fla 0.6
  venv-tts/   TTS 环境：torch 2.11+cu128, transformers 4.57.3, qwen-tts 0.1.1
```

与 Mac 端 `~/Library/Caches/novel-video/venvs/{main,ltx,musicgen}` 的多环境布局同构，
因此 `pipeline/backends/` 抽象层可直接复用「按阶段选解释器」的设计。

## 7. 完整 produce 实测（2026-11-25）

开发机上用全新项目通过 `pipeline produce` 一条命令跑完整流程：

```
storyboard → screenplay → images → audio → videos → music → assemble
```

实测产物：
- `final.mp4`：**7.88 秒，189 帧，384×640 @24fps，H.264 + AAC，5.61 MB**
- 5 个镜头视频、5 张静帧、2 张角色肖像、5 段对白配音、配乐全部生成
- 总耗时 **901.8 秒（15 分钟）**，`produce` 返回码 0

阶段峰值（视频阶段内部）：

```
text_encoder   7.77 GiB → 释放后 free 10.79 GiB
connectors     2.67 GiB → 释放
transformer    8.49 GiB → 释放后 free 10.78 GiB
VAE            3.00 GiB
```

**ffmpeg/字幕跨平台问题**：Windows 上 `fontsdir=C:\\WINDOWS\\Fonts` 会因盘符冒号破坏滤镜语法，
三种转义均失败。`platform.ass_filter()` 在 Windows 不加 `fontsdir`（ffmpeg 自动加载系统字体），
Mac/Linux 保留原 `fontsdir` 行为。

**重要质量问题（进行中）**：初版完整 produce 虽然生成 MP4，但抽帧为网格噪点。
已发现并修复：
- latent 必须按 VAE `latents_mean/std` normalize，解码时 denormalize
- Gemma 必须给 connector 输入全部 49 层 hidden_states，且输出经过 LTX2TextConnectors
- 官方 I2V 需要 conditioning mask 固定首帧，并按 token 数算 scheduler shift
- 官方 I2V 对模型输出做 velocity↔x0 转换，scheduler 仅更新非 conditioning 帧

已重写为官方 LTX2 I2V 步进逻辑；正在最后复测画面质量。**阶段验收以抽帧为真实画面为准，不能只以 rc=0 认定完成。**

## 8. 下一步

1. 确认最新 I2V 修复的抽帧无网格伪影、首帧与输入静帧一致
2. 重跑完整 `produce` 形成最终回归证据
3. 网页控制台端到端验收（目标 c）
4. 把安装/端到端结果补充入 Windows 部署文档

## 9. 复现方式

探针脚本位于 `docs/win-cuda/probes/`，通过 SSH 以 base64 上传执行（开发机无 bash）。
主环境 `C:/Users/annas/avwin/.venv`，TTS 环境 `C:/Users/annas/avwin/venv-tts`。
安装步骤见 `install_torch.py`、`install_rest.py`、`t17_install_fast.py`、`t48_tts_isolation.py`。
