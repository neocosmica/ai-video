# 本地小说到竖屏短剧

在 **Apple Silicon Mac**（推荐 16GB+ 统一内存）上，把小说做成竖屏短剧。图像、视频、对白、配乐都在本机算；**只有分镜 JSON 由 Cursor Agent 写**，写好后用一条脚本跑到成片，中途不再依赖 Agent。

## 本机模型

| 步骤 | 模型 | 说明 |
|---|---|---|
| 文生图 | FLUX.2 Klein 4B 4-bit | 竖屏静帧 + 参考图锁脸 |
| 对白 | Qwen3-TTS 1.7B CustomVoice 8-bit | 先于视频；按性别选音色，按情绪写 instruct |
| 图+音频生视频 | LTX-2.3 int4 distilled **a2v** | 有对白：静帧 + TTS wav 条件；空镜仍 I2V；帧数跟对白时长 |
| 配乐 | MusicGen small | 读英文 `music_prompt`，无人声器乐；按时长轴估长 |

首次 `setup.sh` 主要拉 **模型权重**（数十 GB）并安装 Python 依赖；`vendor/`（LTX、MusicGen 源码）和 `tools/`（ffmpeg）已在本仓库内，默认不再从 GitHub 克隆。请插电，保持网络畅通（默认走国内镜像）。

## 工作流（两步）

1. **Agent 只做这一步**：读小说，写 `input/demo_storyboard.json`（或 `output/<片名>/storyboard.json`）。人物填 `gender`/`voice`，镜头填 `emotion`，顶层英文 `music_prompt`。
2. **一条脚本成片**（校验 → 图 → **对白+配乐** → **A2V/I2V 视频** → 混音字幕）：

```bash
./scripts/run.sh produce \
  --storyboard input/demo_storyboard.json \
  --novel input/novel.txt \
  --out output/demo
```

成片在 `output/demo/final.mp4`。`produce` 会把分镜拷进输出目录，再串跑全部阶段；**不需要再逐步喊 Agent**。

有对白的镜头：TTS → 写入 `video_frames` → `ltx-2-mlx a2v --distilled --audio … --image …`。无对白空镜：定长 I2V（`--no-audio`）。成片混音仍用 TTS 原轨 + BGM（A2V 片内音轨在 normalize 时剥掉，避免双轨）。

女声常用 `Serena`（温柔）/ `Vivian`（偏亮）；男声常用 `Dylan` / `Uncle_Fu` / `Eric`。情绪例如 `calm` `warm` `sad` `bitter` `surprised`。

## 国内镜像

`scripts/env.sh` / `setup.sh` **一律走国内源，不回退海外官方地址**：

- Hugging Face → `https://hf-mirror.com`（并禁用 XET，避免海外 CDN）
- PyPI / uv → 清华 `pypi.tuna.tsinghua.edu.cn`
- uv 托管 Python → npmmirror `cdn.npmmirror.com/binaries/python-build-standalone`
- GitHub 克隆 → `ghfast.top` / `mirror.ghproxy.com` / `ghproxy.net`（按序尝试；**仅当仓内 `vendor/` 缺失时**）
- ffmpeg 缺省用仓内 `tools/`；若缺失则经清华 Homebrew 镜像安装

Python：**优先本机已有 3.14 / 3.13 / 3.12 / 3.11**（`requires-python >=3.11`，3.14 可用）；没有才从 npmmirror 安装。可用 `NOVEL_VIDEO_PYTHON=3.14` 强制指定。

## 新机器部署

```bash
chmod +x scripts/setup.sh scripts/run.sh scripts/env.sh
./scripts/setup.sh
./scripts/run.sh doctor
./scripts/run.sh selftest
```

`setup.sh` 会：

- 使用仓内 `vendor/ltx-2-mlx`、`vendor/musicgen-mlx`、`tools/ffmpeg`（缺了才从国内镜像下载/克隆）
- 把主 / LTX / MusicGen 的 `.venv` 装到 `~/Library/Caches/novel-video/venvs/`，项目里只留符号链接
- 确认 LTX 已带 Gemma/connector **分阶段加载**改动（16GB 机器必需）
- 预拉模型权重到本机 Hugging Face 缓存（权重不入库，走 hf-mirror）

异常时先看 `./scripts/run.sh doctor`：确认三个 `.venv` 都指向 Caches、Metal 可用、LTX 补丁为「已打」。

## 换一部小说

改 `input/` 下文本，让 Agent 重写分镜 JSON，再：

```bash
./scripts/run.sh produce \
  --storyboard output/<片名>/storyboard.json \
  --novel input/<小说>.txt \
  --out output/<片名>
```

若分镜已经在 `--out/storyboard.json`，可省略 `--storyboard`。

只重做声音（会重写 `video_frames`；若要口型跟新对白，需再跑 videos）：

```bash
./scripts/run.sh audio --out output/demo --force
./scripts/run.sh videos --out output/demo --force
./scripts/run.sh assemble --out output/demo
```

从中间阶段续跑仍可用 `run --from images|audio|videos|assemble`。

## 预期耗时（M4 Air 16GB）

静帧几十秒到数分钟；对白很快；视频一镜约数分钟到十几分钟（a2v 与 distilled I2V 同量级）；配乐按秒数，MusicGen small 通常接近实时。插电，不要合盖。
