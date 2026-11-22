# 本地小说到竖屏短剧

这台机器是 MacBook Air M4，8 核 GPU，16GB 统一内存。图像、视频、对白、配乐都在本机算；分镜由 Cursor Agent 写 JSON，不加载聊天 LLM。

## 本机模型

| 步骤 | 模型 | 说明 |
|---|---|---|
| 文生图 | FLUX.2 Klein 4B 4-bit | 竖屏静帧 + 参考图锁脸 |
| 图生视频 | LTX-2.3 int4 distilled | 第一帧锁定，约 2 秒/镜；提示词编码器是 Gemma 3 12B 4-bit |
| 对白 | Qwen3-TTS 1.7B CustomVoice 8-bit | 按性别选音色，按情绪写 instruct |
| 配乐 | MusicGen small | 读英文 `music_prompt`，无人声器乐 |

## Agentic 链路

1. Agent 写 `storyboard.json`（人物 `gender`/`voice`，镜头 `emotion`，顶层英文 `music_prompt`）
2. `screenplay` 校验
3. 参考图 / 静帧（Flux）
4. 镜头视频（LTX）
5. 对白 + 配乐（TTS / MusicGen）
6. 混音、字幕、成片 `final.mp4`

女声常用 `Serena`（温柔）/ `Vivian`（偏亮）；男声常用 `Dylan` / `Uncle_Fu` / `Eric`。情绪例如 `calm` `warm` `sad` `bitter` `surprised`。

## 国内镜像

`scripts/env.sh` 默认：

- Hugging Face → `https://hf-mirror.com`（并禁用 XET，避免海外 CDN）
- PyPI / uv → 清华 `pypi.tuna.tsinghua.edu.cn`
- GitHub 克隆 → `ghfast.top` 前缀，失败再回退官方

## 准备

```bash
chmod +x scripts/setup.sh scripts/run.sh scripts/env.sh
./scripts/setup.sh
```

## 运行

```bash
./scripts/run.sh screenplay --out output/demo
./scripts/run.sh run --from images --out output/demo
```

只重做声音：

```bash
./scripts/run.sh audio --out output/demo --force
./scripts/run.sh assemble --out output/demo
```

## 预期

静帧几十秒；视频一镜约五分钟；对白几秒到几十秒；配乐按秒数，MusicGen small 通常接近实时。插电，不要合盖。
