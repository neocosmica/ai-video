# 本地小说到竖屏短剧

在 **Apple Silicon Mac**（推荐 16GB+ 统一内存）上，把小说做成竖屏短剧。图像、视频、对白、配乐都在本机算；分镜由 Cursor Agent 写 JSON，不加载聊天 LLM。

## 本机模型

| 步骤 | 模型 | 说明 |
|---|---|---|
| 文生图 | FLUX.2 Klein 4B 4-bit | 竖屏静帧 + 参考图锁脸 |
| 图生视频 | LTX-2.3 int4 distilled | 第一帧锁定，约 2 秒/镜；提示词编码器是 Gemma 3 12B 4-bit |
| 对白 | Qwen3-TTS 1.7B CustomVoice 8-bit | 按性别选音色，按情绪写 instruct |
| 配乐 | MusicGen small | 读英文 `music_prompt`，无人声器乐 |

首次 `setup.sh` 主要拉 **模型权重**（数十 GB）并安装 Python 依赖；`vendor/`（LTX、MusicGen 源码）和 `tools/`（ffmpeg）已在本仓库内，默认不再从 GitHub 克隆。请插电，保持网络畅通（默认走国内镜像）。

## Agentic 链路

1. Agent 写 `storyboard.json`（人物 `gender`/`voice`，镜头 `emotion`，顶层英文 `music_prompt`）
2. `screenplay` 校验（含引号对白覆盖检查）
3. 参考图 / 静帧（Flux）
4. 镜头视频（LTX）
5. 对白 + 配乐（TTS / MusicGen）
6. 混音、字幕、成片 `final.mp4`

女声常用 `Serena`（温柔）/ `Vivian`（偏亮）；男声常用 `Dylan` / `Uncle_Fu` / `Eric`。情绪例如 `calm` `warm` `sad` `bitter` `surprised`。

## 国内镜像

`scripts/env.sh` 默认：

- Hugging Face → `https://hf-mirror.com`（并禁用 XET，避免海外 CDN）
- PyPI / uv → 清华 `pypi.tuna.tsinghua.edu.cn`
- GitHub 克隆 → `ghfast.top` 前缀，**仅当仓内 `vendor/` 缺失时回退使用**

## 新机器部署

```bash
chmod +x scripts/setup.sh scripts/run.sh scripts/env.sh
./scripts/setup.sh
./scripts/run.sh doctor
./scripts/run.sh selftest
```

`setup.sh` 会：

- 使用仓内 `vendor/ltx-2-mlx`、`vendor/musicgen-mlx`、`tools/ffmpeg`（缺了才回退下载/克隆）
- 把主 / LTX / MusicGen 的 `.venv` 装到 `~/Library/Caches/novel-video/venvs/`，项目里只留符号链接
- 确认 LTX 已带 Gemma/connector **分阶段加载**改动（16GB 机器必需）
- 预拉模型权重到本机 Hugging Face 缓存（权重不入库）

异常时先看 `./scripts/run.sh doctor`：确认三个 `.venv` 都指向 Caches、Metal 可用、LTX 补丁为「已打」。

## 跑通演示（小说已带 `input/novel.txt`）

```bash
mkdir -p output/demo
cp input/demo_storyboard.json output/demo/storyboard.json
./scripts/run.sh screenplay --out output/demo --novel input/novel.txt
./scripts/run.sh run --from images --out output/demo
```

成片在 `output/demo/final.mp4`。

换一部小说时：改 `input/` 下文本，让 Agent 重写对应 `output/<片名>/storyboard.json`，再 `screenplay` → `run --from images`。

只重做声音：

```bash
./scripts/run.sh audio --out output/demo --force
./scripts/run.sh assemble --out output/demo
```

## 预期耗时（M4 Air 16GB）

静帧几十秒到数分钟；视频一镜约数分钟到十几分钟；对白几秒到几十秒；配乐按秒数，MusicGen small 通常接近实时。插电，不要合盖。
