# 本地小说到竖屏短剧

在 **Apple Silicon Mac**（推荐 16GB+ 统一内存）上，把小说做成竖屏短剧。分镜、图像、对白、视频、配乐都在本机算。

## 怎么用

1. 双击仓库根目录的 **`AI短视频.app`**（可拖到程序坞 / 桌面）。
2. 首次打开会在「选择项目」页自动检查并安装依赖、下载模型（进度条）；插电、保持网络畅通（默认国内镜像）。
3. 装好后选项目或新建 → 粘贴小说 → 生成/改分镜 → 勾选阶段 → 点「启动任务」。
4. 菜单栏图标可打开网页、复制地址、退出；成片在对应项目文件夹的 `final.mp4`。

须在本机已准备好的同一台 Mac 上使用。异常时可在页面点重试安装。

## 项目怎么放

每个项目一个文件夹，全部在 `projects/` 下；顶层只放源与成片，过程产物进 `work/`：

```
projects/<片名>/
  novel.txt
  storyboard.json
  final.mp4
  work/          # frames/ clips/ audio/ logs/ ...
```

网页里新建项目即可；小说、分镜、任务历史、镜头产物都跟该项目走（任务历史在 `work/webui/state.json`）。

## 页面流程

1. **选项目**（可新建）→ 之后界面上的小说、分镜、任务、镜头都属于它  
2. **填小说**或载入已有文本；可开分镜编辑器（内置超长对白硬校验，给出可照抄的拆镜方案）  
3. **勾选阶段**起任务：校验分镜 → 静帧 → 配音 → 视频 → 配乐 → 合成（可强制重做、只重做某些镜头）  
4. **看进度**：总进度条、逐条剩余时间、镜头卡小条；日志实时滚动，出错标红  
5. **重跑**：失败可从最早失败阶段续跑；镜头网格可预览静帧 / 视频 / 配音，并单镜重做  

数据依赖（不是死板阶段链表）：分镜 ← 小说；静帧与配音都只吃分镜；视频 ← 静帧 +（有对白则）配音；配乐按各镜时长计划定长；成片 ← 各镜视频（+ 可选配乐）。

## 分镜约定

- 台词：小说「」/“”/"" 引号原文进 `dialogue`，禁止漏句删字；空镜 `""`
- 人物：`gender`（female/male）；可选 `voice`（女 Serena/Vivian，男 Dylan/Uncle_Fu/Eric）
- 有对白镜：`emotion`（calm/warm/sad/tense/bitter/nostalgic/surprised 等）；**不要**在 emotion 里写语速
- `motion_prompt`、顶层 `music_prompt`：**英文**；外貌只取原文可见特征
- 超长对白按标点拆成相邻镜，每镜换景别/角度与独立画面、运镜提示；**不许删字压短**

单镜上限按本机总内存配置估算（16 GB 约 97 帧、32 GB 约 217 帧，硬顶 241 帧）。配音只合成一次、不调语速，镜长跟着走；仍超上限时在停顿处切子镜头（同轨连贯、复用静帧），无停顿可切则允许该镜变长。

## 成片怎么拼出来

- 有对白：配音 → 写镜长 → 静帧 + 音频生成视频（a2v）；空镜：静帧直接动起来（I2V）
- 成片混音用对白原轨 + 配乐（剥掉视频片内音，避免双轨）
- 配乐在视频之后按实际镜长生成（成片时长 + 一点余量），对白期间压低、**空镜恢复满电平**

| 配置项 | 作用 |
|---|---|
| `voice_volume` | 对白增益（默认 1.35） |
| `music_target_lufs` / `music_volume` | 配乐响度统一后再乘相对音量 |
| `music_duck_gain` / `music_duck_pad` | 对白期间压低倍数 / 窗口余量（秒） |
| `mix_limit` | 总线限幅（0.89 ≈ -1 dBFS） |
| `music_max_seconds` | 单次配乐秒数上限（默认 180） |

## 本机模型与磁盘占用

首次安装合计约 **40.4 GB**（约 93% 是模型权重）。权重在 `~/.cache/huggingface/hub/`，Python 环境在 `~/Library/Caches/novel-video/venvs/`。

| 步骤 | 模型 | 仓库 | 占用 |
|---|---|---|---:|
| 小说→分镜 | Qwen3-4B Instruct 4-bit | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2.12 GB |
| 文生图 | FLUX.2 Klein 4B 4-bit | `mflux-community/flux2-klein-4b-mflux-q4` | 4.30 GB |
| 对白 | Qwen3-TTS 1.7B CustomVoice 8-bit | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 2.87 GB |
| 视频 | LTX-2.3 int4 distilled（a2v / I2V） | `dgrauet/ltx-2.3-mlx-q4` | 19.07 GB |
| 视频提示词编码 | Gemma 3 12B 4-bit（仅编码，不写分镜） | `mlx-community/gemma-3-12b-it-4bit` | 7.51 GB |
| 配乐 | MusicGen small | `facebook/musicgen-small` | 1.01 GB |
| 配乐条件 | T5-base | `t5-base` | 852 MB |
| **权重小计** | | | **37.72 GB** |

| 类别 | 说明 | 磁盘 |
|---|---|---:|
| HF 模型权重 | 上表 7 仓 | **37.72 GB** |
| Python 环境 | `main` 1.15 GB / `musicgen` 985 MB / `ltx` 324 MB | **2.43 GB** |
| `tools/` | ffmpeg + ffprobe（仓内已有） | 156 MB |
| uv Python | 本机已有 3.11+ 可跳过 | 115 MB |
| `vendor/` | LTX / MusicGen 源码（已入库） | 11 MB |
| **合计** | | **约 40.4 GB** |

LTX 盘上约 19 GB ≠ 运行时占满内存：靠 int4、分块流式加载、Gemma / 视频模型 / 解码错峰，16 GB 统一内存也可跑。下载默认走国内镜像（Hugging Face → hf-mirror；PyPI → 清华等）。

## 预期耗时（M4 Air 16GB）

静帧几十秒到数分钟；对白很快；视频一镜约数分钟到十几分钟；配乐通常接近实时。插电，不要合盖。

---

## 附录：命令行

日常请用 App。需要排障、自动化或无图形界面时再用下面命令。

### 安装与自检

```bash
chmod +x scripts/setup.sh scripts/run.sh scripts/env.sh
./scripts/setup.sh
./scripts/run.sh doctor
./scripts/run.sh selftest
```

`setup.sh`：仓内 `vendor/` 与 `tools/ffmpeg`（缺失才镜像补）、三个 `.venv` 链到 Caches、确认 LTX Gemma 分阶段加载、预拉 HF 权重。

### 网页与成片

```bash
./scripts/run.sh                        # 同 App：网页控制台 http://127.0.0.1:8790
./scripts/run.sh web --port 8790
NOVEL_VIDEO_NO_MENUBAR=1 ./scripts/run.sh

./scripts/run.sh projects --new <片名>
./scripts/run.sh budget --project <片名>
./scripts/run.sh storyboard --project <片名>
./scripts/run.sh screenplay --strict --project <片名>
./scripts/run.sh produce --project <片名>
```

`produce` 串跑：`storyboard → screenplay → images → audio → videos → music → assemble`。已有分镜默认不覆盖，重写加 `--force`。

### 局部重跑

```bash
./scripts/run.sh audio --project demo --force
./scripts/run.sh videos --project demo --force
./scripts/run.sh music --project demo
./scripts/run.sh assemble --project demo

./scripts/run.sh run --from images|audio|videos|assemble --project demo
```

### 镜像与 Python

| 用途 | 源 |
|---|---|
| Hugging Face | `hf-mirror.com`（禁用 XET） |
| PyPI / uv | 清华 `pypi.tuna.tsinghua.edu.cn` |
| uv 托管 Python | npmmirror `python-build-standalone` |
| GitHub（仅 vendor 缺失） | `ghfast.top` 等 |
| Homebrew ffmpeg（仅 tools 缺失） | 清华 bottles |

环境变量见 `scripts/env.sh`。Python 优先本机 3.14 / 3.13 / 3.12 / 3.11；可用 `NOVEL_VIDEO_PYTHON=3.14` 强制指定。
