# 本地小说到竖屏短剧

在 **Apple Silicon Mac**（推荐 16GB+ 统一内存）上，把小说做成竖屏短剧。分镜、图像、视频、对白、配乐都在本机算；`produce` 一条命令从小说跑到成片。

## 本机模型

| 步骤 | 模型 | 说明 |
|---|---|---|
| 小说→分镜 | Qwen3-4B Instruct 4-bit（MLX） | ≈3.3 GB，适合 16 GB Mac；写 `storyboard.json` |
| 文生图 | FLUX.2 Klein 4B 4-bit | 竖屏静帧 + 参考图锁脸 |
| 对白 | Qwen3-TTS 1.7B CustomVoice 8-bit | 先于视频；按性别选音色，按情绪写 instruct |
| 图+音频生视频 | LTX-2.3 int4 distilled **a2v** | 有对白：静帧 + TTS wav 条件；空镜仍 I2V；帧数跟对白时长 |
| 配乐 | MusicGen small | 读英文 `music_prompt`，无人声器乐；按 `video_frames` 定长 |

首次 `setup.sh` 主要拉 **模型权重**（数十 GB）并安装 Python 依赖；`vendor/`（LTX、MusicGen 源码）和 `tools/`（ffmpeg）已在本仓库内，默认不再从 GitHub 克隆。请插电，保持网络畅通（默认走国内镜像）。

## 项目路径约定

**每个项目一个独立文件夹，这部片的小说、分镜、产物全在里面**；全部项目放在根目录 `projects/` 下。
项目顶层只放「源」和「成片」，过程产物全部收进 `work/`：

```
projects/
  demo/            # 一个项目 = 一个文件夹
    novel.txt        # 小说
    storyboard.json  # 分镜
    final.mp4        # 成片
    work/            # 过程产物（自动生成）：frames/ clips/ audio/ logs/ ...
  scifi/
    ...
```

新建：`./scripts/run.sh projects --new <片名>`；列出：`./scripts/run.sh projects`。
CLI 全部子命令支持 `--project <片名>`，等价于 `--out projects/<片名>` 并自动取该文件夹里的 `novel.txt` / `storyboard.json`。

## 工作流

0. **先拿单镜上限**（只读本机总内存配置，按经验 token 预算表 + 当前分辨率现算；不读系统实时状态）：

```bash
./scripts/run.sh budget --project scifi
```

输出「N 帧 ≈ S 秒」和「台词约 C 字」。超额度对白必须拆成相邻多镜，**不许删字压短**。
经验值：16 GB→3120 tokens（384×640 下 97 帧）、32 GB→6720 tokens（217 帧）；更高内存顶到 241 帧。

1. **小说→分镜**（本机 Qwen3-4B；已有分镜且未 `--force` 会跳过）：

```bash
./scripts/run.sh storyboard --project scifi
```

也可手写 / 改 `storyboard.json`。人物填 `gender`/`voice`，镜头填 `emotion`，顶层英文 `music_prompt`。**台词按小说原样**；超额度按标点拆镜。

2. **硬校验**（秒级；`--strict` 超额度会失败并给出可照抄的拆法）：

```bash
./scripts/run.sh screenplay --strict --project scifi
```

3. **一条脚本成片**（生成分镜 → 校验 → 图 → 对白 → **A2V/I2V** → **配乐** → 混音字幕）：

```bash
./scripts/run.sh produce --project scifi
```

成片在 `projects/<片名>/final.mp4`。已有分镜时默认不覆盖（audio 可能已自动拆过镜），要重写分镜请加 `--force`。

有对白的镜头：TTS → 写入 `video_frames` → `ltx-2-mlx a2v --distilled --audio … --image …`。无对白空镜：定长 I2V（`--no-audio`）。成片混音仍用 TTS 原轨 + BGM（A2V 片内音轨在 normalize 时剥掉，避免双轨）。

**配乐在视频之后生成**（`music` 阶段），此时每一镜的 `video_frames` 都已确定，配乐长度就是成片实际长度 + 2 秒余量，
不再靠 `-stream_loop` 循环拼接、不会在中间留硬接缝。已有配乐短于成片时会自动重新生成。

**混音音量**：对白与配乐各自先做响度统一，再相加（`amix normalize=0`，不按路数除，人声不会随镜头数变轻）。
配乐在对白期间逐帧下压到 `music_duck_gain`（默认 ×0.3 ≈ -10 dB），**空镜段恢复满电平**——
不是「只要有对白就全程压低」，空镜是配乐唯一能露出的地方。总线末尾 `alimiter` 兜底防削波。

| 配置项 | 作用 |
|---|---|
| `voice_volume` | 对白人声增益（默认 1.35） |
| `music_target_lufs` / `music_volume` | 配乐先 loudnorm 到该响度，再乘相对音量 |
| `music_duck_gain` / `music_duck_pad` | 对白期间压低倍数 / 窗口前后余量（秒） |
| `mix_limit` | 混音总线限幅（0.89 ≈ -1 dBFS） |
| `music_max_seconds` | 单次生成配乐的秒数上限（默认 180） |

**配音语速不做任何特殊调整**：`generate_custom_voice` 不传 speed，只有 `emotion` → instruct（语气），
台词**只合成一次**。以前「台词读不完」是因为 `video_frames_max` 卡在 97 帧（4 秒），对白念 6–10 秒时
超出部分被 `atrim` 静默切掉。现在：

- 单镜上限由 `budget` 按本机总内存查表现算（`video_frames_auto`，硬顶 241 帧 ≈ 10 秒）；写分镜时用 `--strict` 把超长对白先拆好；
- 实测配音仍超过上限时，在句中停顿处**切成相邻子镜头**（同一次配音，语调连贯；复用同一静帧、字幕跟着切点）；
- 一口气念完、没有停顿可切时，**保留原样让这一镜变长**，绝不重念或改语速；
- 最后 `assert_no_overflow` 校验「画面 ≥ 配音」，**不可能再静默截台词**。

女声常用 `Serena`（温柔）/ `Vivian`（偏亮）；男声常用 `Dylan` / `Uncle_Fu` / `Eric`。情绪例如 `calm` `warm` `sad` `bitter` `surprised`。

## 网页控制台（新增）

不想敲命令行时，用自带的标准库网页界面填小说、起任务、看进度、重跑步骤（零新增依赖）。**无参数默认就是起网页**（不加载模型；管线在页面里按需点「启动任务」再拉起）：

**访达双击**：仓库根目录的 `打开控制台.app`（带图标）。未启动时会先在 Terminal 拉起后端，再打开浏览器；已在跑则直接开网页。可拖到程序坞 / 桌面。

```bash
./scripts/run.sh                        # 默认 → 网页控制台 http://127.0.0.1:8790
./scripts/run.sh web --port 8790        # 显式写 web / 改端口也可以
```

- **① 选择项目**：打开网页第一件事是选项目（每个项目一张卡片，显示小说/分镜/成片与任务统计），也可当场新建；之后界面上的小说、分镜、任务、镜头产物全部属于这个项目。
- **② 填写小说片段**：粘贴片段（自动存成 `projects/<片名>/novel.txt`）或载入已有 txt；可顺手打开分镜编辑器改 `storyboard.json`，内置 `--strict` 硬校验（给出可照抄的拆镜方案）。
- **③ 起任务**：勾选要跑的阶段（校验分镜 → 静帧 → 配音 → 视频 → 配乐 → 合成），等价于 `run --from …`；支持 `--force` 与「只重做某些镜头」。
- **④ 看进度**：三档粒度——总进度条（各阶段按权重合成，显示「正在：视频生成」等）；进行中条目逐条显示 `n/total` 和剩余时间（解析子进程 tqdm 输出，视频每镜的两段 Denoising 8+3 步会合并成一条）；镜头网格里每张卡自己也有小进度条。日志实时滚动，出错行标红。
- **⑤ 重跑步骤**：任务失败后可一键「从最早失败阶段续跑」；镜头网格里逐镜预览**静帧、视频、配音试听**（`<audio>` 播放 `audio/voice/*.wav`），单独重跑某一镜的图或视频；产物每 5 秒自动刷新（可关）。
- 任务历史跟项目走：存在 `projects/<片名>/work/webui/state.json`，重启服务不丢（中断的任务会标记为「中断」）。
- 局限：本界面不加载模型也不写分镜内容——分镜仍由你/Agent 按第 3 节写好；`web` 进程必须跑在装好依赖的 Mac 上（和 `produce` 同机）。

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

改项目文件夹里的 `novel.txt`，让 Agent 重写分镜 JSON，再：

```bash
./scripts/run.sh produce --project <片名>
```

若分镜已经在项目文件夹的 `storyboard.json`，可省略 `--storyboard`。

只重做声音（`audio` 重写 `video_frames`；若要口型跟新对白，需再跑 videos）：

```bash
./scripts/run.sh audio --project demo --force
./scripts/run.sh videos --project demo --force
./scripts/run.sh music --project demo       # 镜头定长后重算配乐长度
./scripts/run.sh assemble --project demo
```

只调混音/配乐音量，不必重做画面：

```bash
./scripts/run.sh music --project demo --force   # 只改音量时可省略 --force
./scripts/run.sh assemble --project demo --force
```

从中间阶段续跑仍可用 `run --from images|audio|videos|assemble`。

## 预期耗时（M4 Air 16GB）

静帧几十秒到数分钟；对白很快；视频一镜约数分钟到十几分钟（a2v 与 distilled I2V 同量级）；配乐按秒数，MusicGen small 通常接近实时。插电，不要合盖。
