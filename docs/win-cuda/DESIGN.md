# Windows + NVIDIA 适配方案（总览）

本文是这次适配的**完整方案说明**，回答「为什么这么改、分了几层、每层怎么落地、
怎么验证、边界在哪」。

配套文档：

| 文档 | 内容 |
|---|---|
| 本文 `DESIGN.md` | 方案与架构、改动面、决策理由、验收结论 |
| [`PHASE0-PROBE.md`](PHASE0-PROBE.md) | 阶段 0 全链路实测数据（显存 / 耗时 / 失败路线） |
| [`README.md`](README.md) | 安装与使用手册 |
| [`probes/`](probes/) | 全部实测脚本（49 个 `.py`） |
| [`../../pipeline/backends/README.md`](../../pipeline/backends/README.md) | 后端模块级说明 |

---

## 1 目标与约束

用户在需求里定了三条硬约束，方案围绕它们展开：

| 约束 | 原话 | 方案响应 |
|---|---|---|
| q1 | 「Windows + 英伟达显卡的最佳实践，不要将就苹果生态」 | 用 PyTorch/diffusers 原生 CUDA 栈，模型换 NVIDIA 生态同档最新，不沿用 MLX 权重或为其让路 |
| q2 | 「暂时在分支开发……尽量保持 mac 可用，把两边不同的部分尽量整体打包分叉」 | 新建 `pipeline/backends/` 整块分叉；公共逻辑一行不改；Mac 侧既有文件仅做平台适配 |
| q3 | 「要验收到 c」（c = 网页控制台跑通 produce 产出 `final.mp4`） | 网页控制台两种启动路径都在开发机实测通过，见第 6 节 |

补充约束：开发在 `feat/windows-nvidia` 分支；验收以**抽帧是真画面**为准，不是只看返回码。

---

## 2 核心事实：这是一次「换栈」，不是「换开关」

Mac 端模型全部是 MLX 格式（`mlx-community/*-4bit`、`mflux-community/*`、
`dgrauet/ltx-*-mlx-*`），CUDA 读不了；反之 PyTorch 权重在 Mac 没有 Metal 加速等价物。

因此不是一个 runtime flag 能解决，而是**两套权重 + 两套运行时**同时存在，
必须做物理上的一半一半分叉。这是整个方案的出发点。

第二个事实决定了架构的第二个特点：**12 GB 显存装不下整条链**。

各阶段峰值（RTX 4070 12 GB 实测，明细见 `PHASE0-PROBE.md` §5）：

| 阶段 | 模型 | 峰值显存 |
|---|---|---|
| 分镜 | Qwen3.5-4B 4bit | 3.4 GB |
| 静帧 | FLUX.2 Klein 4B | 7.80 GB |
| 对白 | Qwen3-TTS-1.7B | 4.04 GB |
| 配乐 | MusicGen small | 1.19 GB |
| 文本编码 | Gemma3 12B 4bit | 7.77 GB |
| 视频 transformer | LTX-2 19B Q2_K | 8.49 GB |
| 视频 connectors | LTX-2 connectors | 2.67 GB |
| 视频 VAE | LTX-2 VAE | 3.64 GB |

若全部常驻约需 20 GB。方案沿用 Mac 端「分阶段插载 + 用完即释放」的思路
（Mac 侧对应 `scripts/apply_ltx_gemma_patch.py`：把 Gemma 文本塔与 connector 拆开、
各段用完 `aggressive_cleanup()`，峰值从 ~14 GB 压到 max(单段)），
在 CUDA 上实现为 `cuda_memory.stage()` 上下文管理器。

实测释放效果（视频阶段内部，`PHASE0-PROBE.md` §7）：

```
text_encoder   峰值 7.77 GiB → 释放后 free 回到 10.79 GiB
connectors     峰值 2.67 GiB → 释放
transformer    峰值 8.49 GiB → 释放后 free 回到 10.78 GiB
VAE            峰值 3.00 GiB
```

**这是方案能成立的根本**：全局峰值取「最重的那一段」而非累加。

---

## 3 分层设计

改动按「差异性质」分三层，避免把平台判断散落到业务代码里。

```
┌─────────────────────────────────────────────────────┐
│ 业务逻辑层（pipeline/*.py 顶层）                     │
│   projects / screenplay / timeline / assemble ...   │
│   → 一行未改，两端共用                                │
├─────────────────────────────────────────────────────┤
│ 平台抽象层（pipeline/platform.py）                   │
│   内存 / 字体 / 进程树 / venv 路径 / 壳命令 / 缓存    │
│   → 把「OS 差异」收成一个文件                         │
├─────────────────────────────────────────────────────┤
│ 后端分叉层（pipeline/backends/）                     │
│   mlx（原有，行为不变） ｜ cuda（本次新增）           │
│   → 把「推理运行时差异」整块隔离                       │
└─────────────────────────────────────────────────────┘
```

### 3.1 平台抽象层 `pipeline/platform.py`（约 300 行）

把跨平台差异集中到一处，其余代码只调函数、不写 `if sys.platform`。

| 能力 | macOS / Linux | Windows |
|---|---|---|
| `total_memory_bytes()` | `os.sysconf(SC_PHYS_PAGES×SC_PAGE_SIZE)` | `GlobalMemoryStatusEx`（ctypes，不引 psutil） |
| `free_memory_percent()` | `memory_pressure` / `/proc/meminfo` | ctypes `MEMORYSTATUSEX` |
| `nvidia_vram_bytes()` | — | `nvidia-smi` 解析 |
| `venv_bin_dir()` / `python_in_venv()` | `bin/` | `Scripts/` |
| `console_script()` | 原名 | 自动补 `.exe` |
| `kill_process_tree()` | `os.killpg(SIGTERM)` | `taskkill /F /T /PID` |
| `font_dir()` / `subtitle_font_name()` | `/System/Library/Fonts/Supplemental` + `Songti SC` | `C:\Windows\Fonts` + `Microsoft YaHei` |
| **`ass_filter()`** | 保留 `fontsdir=` | **不带 `fontsdir`**（见 5.1） |
| `user_cache_dir()` / `app_support_dir()` | `~/Library/...` | `%LOCALAPPDATA%` / `%APPDATA%` |

### 3.2 后端分叉层 `pipeline/backends/`

| 文件 | 职责 | 关键点 |
|---|---|---|
| `__init__.py` | `resolve_backend()` 判定 `auto`/`mlx`/`cuda`；`require_cuda()` 硬校验 | `auto` = macOS→mlx，否则有 NVIDIA 卡→cuda |
| `cuda_memory.py` | 分阶段插载、段间 `empty_cache()`、`ensure_headroom()`、`offload_layers_hook()` | 方案基石 |
| `mlx_memory.py` | MLX 侧对称封装（`aggressive_cleanup` / `get_memory_stats`） | 保持两侧 API 同构 |
| `cuda_storyboard.py` | Qwen3.5-4B 4bit 生成分镜 | **必须 `enable_thinking=False`**，否则输出思维链、JSON 解析出 0 镜 |
| `cuda_images.py` | FLUX.2 Klein 4B，`enable_model_cpu_offload()` | 实测 7.80 GB / 15 s 每张 |
| `cuda_audio.py` | 对白 Qwen3-TTS（**独立 venv 子进程**）+ 配乐 MusicGen | 见 5.3 |
| `cuda_textenc.py` | Gemma3 文本编码，**输出全部 49 层**，编码后立刻释放 | 见 5.2 |
| `cuda_video.py` | LTX-2 19B GGUF + connectors + VAE 分段编排 | 见 5.2，全方案技术重心 |

### 3.3 分叉接线点（6 处，均为「先判后端，再转分支」）

```
pipeline/storyboard_gen.py:473  → cuda_storyboard.cmd_storyboard_cuda
pipeline/images.py:296          → cuda_images.cmd_images_cuda
pipeline/audio.py:667           → cuda_audio.cmd_audio_cuda
pipeline/audio.py:727           → cuda_audio.cmd_music_cuda
pipeline/video.py:246           → cuda_video.cmd_videos_cuda
```

统一形态，MLX 原路径原样保留在 `else` 分支：

```python
from pipeline.backends import resolve_backend
if resolve_backend(settings.backend).is_cuda:
    from pipeline.backends.cuda_video import cmd_videos_cuda
    cmd_videos_cuda(settings, args)
    return
# 以下为原 MLX 实现，一字未改
```

「整体打包分叉」的落地方式就是：**新增目录整块加入，既有文件只加分支入口**。

---

## 4 模型选型

选型原则（用户要求）：视频阶段与 Mac **同档位、同量化**但取最新版本；
其他阶段同档位或稍大、确保显卡能跑，也都取最新。

| 阶段 | 选定模型 | 量化 | 选型理由 |
|---|---|---|---|
| 视频 | `Lightricks/LTX-2` 19B，GGUF `Q2_K` | ~3.15 bit/param | diffusers 0.43 的 `LTX2VideoTransformer3DModel` 与 19B 架构匹配；该 repo 非 gated。LTX-2.5 新版架构不兼容（见 §7 证伪路线） |
| 图像 | `black-forest-labs/FLUX.2-klein-4B` | bf16 + cpu offload | 非 gated。FLUX.3 更新但官方 repo `gated: auto`（403），社区 GGUF 用 ComfyUI 张量名、与 `Flux3Transformer2DModel` 不兼容 |
| 分镜 | `Qwen/Qwen3.5-4B` | 4bit | 实测最新 4B 档（Qwen3.6/3.8 只有 27B+） |
| 对白 | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | bf16 | 无更新版本 |
| 配乐 | `facebook/musicgen-small` | bf16 | 无更新官方 small 档 |

**质量取舍的诚实交代**：视频档位低于 Mac（3.15 vs 3.92 bit/param），因为 12 GB
装不下 Q3_K_S 以上（Q3_K_S 峰值 9.76 GB、余量仅 0.08 GB，接 VAE 必崩）。
用户询问「视频质量会下降吗」时已明确回答：**会降一点**，要追平需 Q4_K_M + 逐层 offload，
速度会慢数倍，未实测。

---

## 5 关键技术障碍与解法

按影响面排序，前两条是「不改就不出正确画面」的级别。

### 5.1 ASS 字幕滤镜：Windows 盘符冒号破坏语法

`ffmpeg` 的 `-vf ass=...:fontsdir=C:\WINDOWS\Fonts` 在 Windows 上必失败。
实测四种写法（`probes/t73_fontsdir.py`）：

| 写法 | 结果 |
|---|---|
| `fontsdir=C:\WINDOWS\Fonts` | `Error parsing a filter description`，rc≠0 |
| `fontsdir=C:/WINDOWS/Fonts`（正斜杠） | 同样失败 |
| `fontsdir=C\:\WINDOWS\Fonts`（转义冒号） | 同样失败 |
| **不带 `fontsdir`** | **rc=0**（ffmpeg 自动加载系统字体） |

解法：`platform.ass_filter()` 按平台分支 —— Windows 省略 `fontsdir`，
macOS/Linux 保留原写法（无冒号，且行为需保持不变）。

### 5.2 LTX-2 I2V：从网格噪点修到真实画面

初版完整 produce 虽然返回码 0、也生成了 MP4，但抽帧是**彩色网格噪点**。
这是本方案投入最大的排障，逐项定位出四个错误：

| # | 错误 | 症状 | 正确做法 |
|---|---|---|---|
| 1 | latent 未归一化 | 画面整体色偏错乱 | VAE encode 后按 `vae.latents_mean/std` + `scaling_factor` normalize；解码前 denormalize |
| 2 | Gemma 只取最后一层 | connector 输入形状不对 | LTX-2 `connectors` 配置 `text_proj_in_factor=49`，必须喂**全部 49 层** `hidden_states` |
| 3 | 缺 `LTX2TextConnectors` | prompt embeds 语义不成立 | 加 connectors 阶段（权重 2.67 GB，用完释放） |
| 4 | 把 transformer 输出误当 x0 | 去噪方向错，退化成噪声 | 按 diffusers LTX2 语义：输出是 **velocity**；仅 guidance 时在 v↔x0 间转换 |
| 5 | 未锁首帧 | 首帧漂移、网格伪影 | conditioning mask 令首帧 token 的 timestep=0，**每个去噪步都锁回** conditioning latent |
| 6 | 单帧被复制成 49 帧再 VAE encode | causal 时间压缩窗口错位 | 先 encode **单帧**，再沿 latent 时间轴复制 |

正确链路（`cuda_video.generate_shot` 的实际顺序）：

```
① text_encoder（Gemma3 12B 4bit）      → 49 层 hidden 落 CPU → 释放
② connectors（LTX2TextConnectors）     → prompt_embeds → 释放
③ vae_encode（单帧）                   → normalize + pack → 初始 latent → 释放
④ transformer（LTX-2 19B Q2_K）        → 去噪（锁首帧） → 释放
⑤ vae（decode）                        → unpack + denormalize → 像素 → 释放
⑥ ffmpeg 写 mp4
```

每段之间 `empty_cache()`，实测 free 回到 10.78–10.79 GiB。
**验收标准据此定为「抽帧必须是真实画面」**，仅看 rc=0 会漏掉这类静默错误。

### 5.3 依赖冲突：`qwen-tts` 与主环境装不到一起

`qwen-tts` 钉 `transformers==4.57.3`，而 LTX-2 侧需要 `transformers>=5`。
硬装会炸：`check_model_inputs() missing 1 required positional argument: 'func'`。

解法：**TTS 单独 venv**（`venvs/tts`），`cuda_audio.synth_line()` 通过子进程调用。
与 Mac 端 `venvs/{main,ltx,musicgen}` 的多环境布局同构，抽象层可直接复用。

### 5.4 Windows 平台专有崩溃点

| 坑 | 现象 | 解法 |
|---|---|---|
| 无官方 `triton` wheel | Qwen3.5 仅 12.5 tok/s | 装 `triton-windows` → 17.8 tok/s |
| `os.sysconf` 不存在 | 读内存直接崩 | `platform.total_memory_bytes()` 走 ctypes |
| **`os.getpgid` 不存在** | produce 第一步崩；**抛 `AttributeError` 而非 `OSError`**，原 `except OSError` 抓不住 | `getattr(os, "getpgid", None)` 探测 |
| 顶层 `import pty` | 模块导入即崩 | 延迟导入（上方已有 `use_pty` 守卫） |
| ffmpeg 路径缺 `.exe` | 找不到 ffmpeg | `config.py` 按平台补后缀 |
| MusicGen `generator` 参数 | transformers 5.x 不认 | 改 `torch.manual_seed` |
| MusicGen `pad_token_id` | `eos=2048` 超词表 `0..2047` → CUDA **device-side assert** | 不传该参数 |
| **`/bin/bash -c` 硬编码** | 网页多阶段串跑必失败 | `web._sequence_cmd()` 按平台分支（见 5.5） |

### 5.5 网页控制台：多阶段串跑不能依赖 bash

`web.py` 原用 `/bin/bash -c 'echo == stage == && cmd'` 拼接多阶段命令，Windows 无 bash。

解法 `_sequence_cmd()`：
- **POSIX**：原样保留 `bash -c`（行为不变，已实测 `build_pipeline_cmd` /
  `build_artifact_groups_cmd` 仍返回 `/bin/bash`）
- **Windows**：用当前解释器 `-c` 顺序启动子进程，仍打印 `== stage ==` 分隔行
  （进度解析器依赖它），首阶段非零退出即中止后续

### 5.6 分镜台词覆盖

`screenplay` 有硬校验：原文所有引号台词必须出现在某镜 `dialogue` 里，否则 `SystemExit`。
CUDA 侧 Qwen3.5 首次生成会漏句。

解法三层（`cuda_storyboard.cmd_storyboard_cuda`）：
1. 提示词把原文全部引号台词**逐条列出**，要求一句不漏
2. 漏句时**在原 JSON 上增量补镜** —— 实测整篇重写会让模型越写越短
   （963 → 724 → 691 字符，最终退化到没有 `shots`）
3. 兜底：仍缺则按原文补空镜，保证硬校验能过

原则：**台词完整性 > 模型是否听话**，且不改写台词（符合仓库 `AGENTS.md` 禁令）。

---

## 6 验收证据

### 6.1 命令行 produce（`pipeline produce`）

全新项目一条命令跑七个阶段：

- `final.mp4`：**7.88 秒 / 189 帧 / 384×640@24fps / H.264 + AAC / 5.61 MB**
- 5 镜头视频、5 静帧、2 角色肖像、5 段配音、配乐齐备
- 总耗时 901.8 秒，返回码 0

### 6.2 网页控制台（目标 c）

脚本 [`probes/t95_web_two_round.py`](probes/t95_web_two_round.py)，走与前端相同的 API 序列：

```
round 1: plan(storyboard) → queue/add → queue/start  ⇒ storyboard.json
round 2: plan(final)=13 项 → queue/add → queue/start ⇒ final.mp4
```

- 两轮任务状态均 **success**
- 队列 13 项：2 肖像 + 3 静帧 + 3 配音 + 3 片段 + 配乐 + 成片
- `final.mp4`：**4.12 秒 / 99 帧 / 384×640@24fps / H.264 + AAC**
- 抽帧确认为真实画面（雨夜便利店 + 人物 + 中文字幕正确叠加）

### 6.3 网页连续多阶段分支

勾选连续多个阶段时网页走 `pipeline run --from`，该分支单独验（本节数据见
`PHASE0-PROBE.md` §7.2）：

- `run --from storyboard --until assemble`：`rc=0` / 640 秒
- 七个阶段分隔行齐备，产出 `final.mp4` 4.46 秒 / 106 帧
- 抽帧确认为真实画面

**两条启动路径都有实测证据**，这是「验收到 c」的完整闭环。

---

## 7 已证伪的路线（避免后人重走）

| 路线 | 结论 | 原因 |
|---|---|---|
| LTX-2.5 | 放弃 | 与 diffusers 0.43 的 `LTX2VideoTransformer3DModel` 架构不匹配（mod 表 9 vs 6 行）；改配置后仍报 `low_cpu_mem_usage cannot be False when using quantization` 与 meta-tensor 错误；`Lightricks/LTX-2.5-Diffusers` 为 gated（403） |
| LTX-2.3 | 不用 | 同上系版本线，不如直接对齐 19B |
| FLUX.3 | 放弃 | 官方 repo `gated: auto`（403）；社区 GGUF 用 ComfyUI 张量名（`content_mode_blocks.image.*`），与 `Flux3Transformer2DModel` 不兼容 |
| GGUF `Q3_K_S` / `Q4_K_S` | 不可行 | Q3_K_S 峰值 9.76 GB、余量 0.08 GB（接 VAE 必崩）；Q4_K_S 11.04 GB → OOM |
| `device_map="auto"`（transformer） | 不可用 | `LTX2VideoTransformer3DModel` 缺 `_no_split_modules`，直接报错 → 改分阶段加载 |
| 手搓 Euler 步进 | 错误 | LTX 的 sigma 表需按 token 数算 shift；手写 `1/N` 步长得到噪声画面 |

---

## 8 改动面

相对 `main`（基线 `816454c`）：**77 文件、+5197 / −60 行**（含本文档与探针脚本）。

新增（整块分叉部分）：

| 路径 | 行数 | 内容 |
|---|---|---|
| `pipeline/backends/cuda_video.py` | 610 | LTX-2 视频全链路 |
| `pipeline/backends/cuda_audio.py` | 396 | TTS + MusicGen |
| `pipeline/backends/cuda_storyboard.py` | 380 | 分镜 |
| `pipeline/platform.py` | 301 | 平台抽象 |
| `pipeline/backends/cuda_images.py` | 199 | 静帧 |
| `pipeline/backends/__init__.py` | 166 | 后端判定 |
| `pipeline/backends/cuda_memory.py` | 172 | 显存编排 |
| `pipeline/backends/cuda_textenc.py` | 102 | 文本编码 |
| `pipeline/backends/mlx_memory.py` | 84 | MLX 对称封装 |
| `scripts/setup-win.ps1` / `run-win.ps1` | 96 / 32 | 安装与启动 |
| `docs/win-cuda/*` | — | 文档与 49 个探针脚本 |

既有文件（仅平台适配，均保持 Mac 行为）：

| 文件 | 改动性质 |
|---|---|
| `pipeline/budget.py` | `total_memory_bytes` → `platform` |
| `pipeline/runtime.py` | `clear_uf_hidden` 非 macOS 空操作 |
| `pipeline/assemble.py` | 字体目录/名称 + `ass_filter` |
| `pipeline/progress.py` | `import pty` 延迟 |
| `pipeline/util.py` | `app_support_dir()` |
| `pipeline/live.py` | `os.getpgid` 探测 |
| `pipeline/config.py` | 新增 cuda 字段 + ffmpeg `.exe` |
| `pipeline/web.py` | `_sequence_cmd` 平台分支 |
| `pipeline/{storyboard_gen,images,audio,video}.py` | 加后端分叉入口（+7~14 行） |
| `pipeline/__main__.py` | `doctor` 打印平台/显存/后端 |

**未改动**（两端共用的公共逻辑，已验证 diff 为空）：
`artifact_plan.py`、`run_queue.py`、`screenplay.py`、`projects.py`、`voices.py`。

### Mac 侧回归核对

本地环境是 Linux（跑不了 MLX），因此**只做了逻辑层核对，未在真 Mac 跑成片**：

- `resolve_backend("auto")` 仍解析为 `mlx` ✅
- 非 Windows 平台 `ass_filter()` 仍保留 `fontsdir` ✅
- POSIX `_sequence_cmd` 仍返回 `/bin/bash` 命令 ✅
- 公共逻辑模块 diff 为空 ✅

**需 Mac 实机复跑一次成片才算完整回归** — 这是本方案唯一的未闭合项，见第 10 节。

---

## 9 使用与排障

```powershell
# 首次部署
.\scripts\setup-win.ps1

# 环境检查（打印平台 / 显存 / 后端 / 字体 / 缓存）
.\scripts\run-win.ps1 doctor

# 网页控制台
.\scripts\run-win.ps1

# 命令行成片
.\scripts\run-win.ps1 produce --project demo
```

配置（`config.json`）：

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

环境变量：`NOVEL_VIDEO_BACKEND` 覆盖 `backend`；`NOVEL_VIDEO_TTS_PYTHON` 指定 TTS 解释器。

网络（开发机实测可用）：hf-mirror ✅、tuna PyPI ✅、download.pytorch.org ✅、gyan.dev ✅；
github 直连 ❌（超时）。

---

## 10 边界与遗留

诚实列出，供后续决定：

1. **Mac 实机回归未做**（本环境是 Linux，无 Metal）。逻辑核对已通过，但需在真 Mac
   跑一次 `produce` 确认成片，才算完整闭环。
2. **视频画质低于 Mac 一档**（3.15 vs 3.92 bit/param）。追平路线（Q4_K_M + 逐层
   offload）与 `offload_layers_hook` 已写好接口，但**速度未实测**。
3. **网页验收是脚本驱动 API**，走的是与前端相同的接口序列，但不是人工点击界面。
   如需真实点击流程需人工复验。
4. **本机缺 numpy**，`pipeline selftest` 无法运行；用的是模块导入检查
   （9 个跨平台/依赖图模块全通过）。
5. **9 个镜头以上长片未压测**。当前验收是 3/5 镜项目，显存峰值与镜头数基本解耦
   （按镜串行），但长时间运行的稳定性未验证。
6. 开发机上的验收临时项目（`projects/webdemo`、`webacc`、`runfrom`）与 `probes/`
   源码包属实验残留，不在仓库内。
