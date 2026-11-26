# 小说 → 竖屏短剧（Agent 指南）

本仓库在 Apple Silicon Mac 上把小说做成竖屏短剧。小说→分镜也由本机 LLM（Qwen3-4B Instruct 4bit）跑；Agent 可选改分镜或直接 `produce`。

本机模型：Qwen3-4B（分镜）→ FLUX.2 Klein 4B（静帧）→ Qwen3-TTS（对白，并写 `video_frames`）→ LTX（服从帧数）∥ MusicGen（按 `video_frames` 定长）→ assemble。

## 工作流

```
novel.txt → budget → storyboard（本机 LLM）→ screenplay --strict → 网页/CLI 成片 → final.mp4
```

日常入口：`./scripts/run.sh`（无参数）默认起网页控制台；管线在页面里点「启动任务」再拉起。命令行仍可用 `produce`。

`produce` 串跑：`storyboard → screenplay → images → audio → videos → music → assemble`（执行顺序≠数据依赖）。
镜长在 **audio** 就写进分镜 `video_frames`；配乐按该计划定长，**不依赖 clips 已渲完**。

### 1. 输入输出约定

**每个项目一个独立文件夹，所有和这部片相关的内容都在里面**；默认全部在主目录的 `~/AI短视频项目/` 下（`NOVEL_VIDEO_PROJECTS` 可改）。
项目顶层只放「源」和「成片」，过程产物全部收进 `work/`：

```
~/AI短视频项目/
  demo/            # 一个项目 = 一个文件夹
    novel.txt        # 小说
    storyboard.json  # 分镜（produce 也会把外部分镜拷到这里）
    final.mp4        # 成片
    work/            # 过程产物（自动生成）：frames/ clips/ audio/ logs/ ...
  scifi/
    ...
```

| 用途 | 路径 |
|---|---|
| 项目文件夹 | `~/AI短视频项目/<片名>/` |
| 小说 | `~/AI短视频项目/<片名>/novel.txt` |
| 分镜 | `~/AI短视频项目/<片名>/storyboard.json` |
| 成片 | `~/AI短视频项目/<片名>/final.mp4` |
| 过程产物 | `~/AI短视频项目/<片名>/work/`（帧、片段、配音、日志；脚本自动管理） |

新建项目：`./scripts/run.sh projects --new <片名>`。CLI 全部子命令都支持 `--project <片名>`，等价于 `--out` 指向该绝对路径并自动补 novel/storyboard。首次启动会把仓库旧 `projects/` 迁到主目录。更新 App：`./scripts/macos/pack-app.sh`。

首次部署：`./scripts/setup.sh` → `./scripts/run.sh doctor` → `./scripts/run.sh selftest`。

### 2. 先拿单镜上限

```bash
./scripts/run.sh budget --project <片名>
```

记下：

- **单镜上限**：N 帧 ≈ S 秒
- **单镜台词额度**：约 C 字

依据：本机**总内存配置**查经验 token 预算表（16 GB→3120 tokens、32 GB→6720 tokens），再按 `config.json` 的分辨率现算帧数。不读系统实时状态。超额度对白必须拆镜，**不许删字**。

### 3. 生成 / 改分镜

默认本机生成（已有 `storyboard.json` 且未 `--force` 会跳过）：

```bash
./scripts/run.sh storyboard --project <片名>
# 或直接 produce（含生成）
./scripts/run.sh produce --project <片名>
```

也可手写 / Agent 改 `storyboard.json`。字段约束：

- `gender`: `female` / `male`
- `voice`: 女 Serena/Vivian；男 Dylan/Uncle_Fu/Eric
- `emotion`: calm/warm/sad/tense/bitter/nostalgic/surprised/whisper 等
- `motion_prompt` 与 `music_prompt` **必须英文**
- `dialogue`：小说引号台词原样；空镜 `""`
- 外貌只取原文可见特征

### 4. 拆镜（不许缩写）

- 所有「」/“”/"" 引号台词必须出现在某个镜头的 `dialogue`，允许跨相邻镜头覆盖，禁止漏句删字。
- 字数 > C：按标点拆成相邻镜头（`s007` / `s007b` / …）；管线生成后也会自动拆超长对白。
- 每镜独立 `image_prompt`（换景别/角度）和英文 `motion_prompt`。
- 不要为时长改写台词，不要在 emotion 里写语速。

### 5. 硬校验

```bash
./scripts/run.sh screenplay --strict --project <片名>
```

失败则按提示拆镜，直到通过。

### 6. 成片

```bash
./scripts/run.sh produce --project <片名>
```

项目文件夹里已有分镜时默认不覆盖；要覆盖加 `--force`。启动后不要逐步干预；失败查 `./scripts/run.sh doctor`。

**启动后立刻把这条终端调到前台**：取终端记录目录里最新的、命令是 `produce` 的那份文件，读文件头 `pid:`，再打开 `vscode-terminal:/<pid>`，用户要能自己看进度。用户说「看进展 / 打开终端」时同样这么做，不要重起进程。

### 7. 网页单跑（按产物依赖，不是按阶段链表）

`produce` 仍可按 `storyboard → screenplay → images → audio → videos → music → assemble` 串跑；**单跑只认数据依赖**。

依赖（传递闭包）：

- **分镜 ← 小说**（最前；缺 `storyboard.json` 则生成，已有则跳过）
- `images ∥ audio`：都只吃分镜，互不依赖
- 有角静帧 ← 肖像；视频 ← 静帧 +（有对白则）配音
- **`video_frames`（镜长计划）**：对白镜由 TTS 实测写入，空镜用默认帧；视频服从它
- **配乐 ← 对白 wav（有对白的镜）**：按各镜 `video_frames` 加总定长，**不依赖 clips**
- 成片 ← 各镜视频（+ 可选配乐）

可单跑产物：分镜 / 肖像 / 静帧 / 配音 / 视频 / 配乐 / 成片。入口在小说下、角色卡、镜头卡、配乐旁、成片旁。

流程：点 ↻ → `POST /api/artifact/plan` → 对话框确认 → `POST /api/queue/add` 写入**左边待跑队列**（去重、补缺失前置）→ 点「启动任务」→ `POST /api/queue/start` 拉起一条**历史任务**。右侧「历史任务」= 每次启动一条记录。停止只杀进程，队列保留并复位为待跑。

- **前置（传递）**：缺失则入队；已有则跳过
- **目标 / 勾选后置**：入队强制重做；start 时清强制项及其全部后向受影响产物（未勾选后置也会删；重跑分镜会清空下游过程产物）
- 队列项被其它项依赖时不可删；可一键清空待跑

实现：`pipeline/run_queue.py` + `pipeline/artifact_plan.py`。

## 禁止

- 改写 / 删字 / 压短引号台词
- emotion 里写「快说／放慢」
- 编造原文没有的配饰
- 逐步跑 images/audio/videos（用 `produce`）；单跑产物请走网页确认框或 `artifact` API
- 另起聊天 LLM 进程抢内存；分镜只用管线里的 `storyboard` 阶段（Qwen3-4B 4bit）

## 自救

- audio 自动拆成 `s007_p1/_p2`：说明分镜写宽了，按 `--strict` 提示改。
- 卡在加载模型、CPU≈0%：先 `doctor`。
