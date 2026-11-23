# 小说 → 竖屏短剧（Agent 指南）

本仓库在 Apple Silicon Mac 上把小说做成竖屏短剧。Agent **只负责读小说、写分镜 JSON**；其余用 CLI 一条命令跑完。

本机模型：FLUX.2 Klein 4B（静帧）→ Qwen3-TTS（对白）→ LTX-2.3 int4 distilled a2v/i2v（视频）→ MusicGen small（配乐，按成片长度）。

## 工作流

```
读小说 → budget → 写 storyboard.json → screenplay --strict → produce → final.mp4
```

`produce` 内部阶段顺序：`screenplay → images → audio（只做对白）→ videos → music（配乐）→ assemble`。
配乐排在视频之后：镜头帧数全部确定后才知道成片多长，配乐按这个长度生成，不会短于画面。

### 1. 输入输出约定

**每个项目一个独立文件夹，所有和这部片相关的内容都在里面**；全部项目放在根目录的 `projects/` 下。
项目顶层只放「源」和「成片」，过程产物全部收进 `work/`：

```
projects/
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
| 项目文件夹 | `projects/<片名>/` |
| 小说 | `projects/<片名>/novel.txt` |
| 分镜 | `projects/<片名>/storyboard.json` |
| 成片 | `projects/<片名>/final.mp4` |
| 过程产物 | `projects/<片名>/work/`（帧、片段、配音、日志；脚本自动管理） |

新建项目：`./scripts/run.sh projects --new <片名>`。CLI 全部子命令都支持 `--project <片名>`，等价于 `--out projects/<片名>` 并自动补 novel/storyboard 默认路径。

首次部署：`./scripts/setup.sh` → `./scripts/run.sh doctor` → `./scripts/run.sh selftest`。

### 2. 先拿单镜上限

```bash
./scripts/run.sh budget --project <片名>
```

记下：

- **单镜上限**：N 帧 ≈ S 秒
- **单镜台词额度**：约 C 字

依据：本机**总内存配置**查经验 token 预算表（16 GB→3120 tokens、32 GB→6720 tokens），再按 `config.json` 的分辨率现算帧数。不读系统实时状态。超额度对白必须拆镜，**不许删字**。

### 3. 写分镜 JSON

```json
{
  "title": "短剧标题",
  "style": "写实华语短剧，自然肤色，电影感光线，浅景深",
  "music_prompt": "English instrumental BGM, no vocals",
  "characters": [
    {
      "id": "linwan",
      "name": "林晚",
      "gender": "female",
      "voice": "Serena",
      "appearance": "二十多岁女性，黑色长发，米色风衣"
    }
  ],
  "shots": [
    {
      "id": "s001",
      "characters": ["linwan"],
      "location": "雨夜天台",
      "image_prompt": "中景，雨夜天台，林晚侧身望向城市灯光",
      "motion_prompt": "Slow cinematic push in, subtle rain",
      "speaker": "林晚",
      "dialogue": "他是我父亲。",
      "emotion": "sad"
    }
  ]
}
```

约束：

- `gender`: `female` / `male`
- `voice`: 女 Serena/Vivian；男 Dylan/Uncle_Fu/Eric
- `emotion`: calm/warm/sad/tense/bitter/nostalgic/surprised/whisper 等
- `motion_prompt` 与 `music_prompt` **必须英文**
- `dialogue`：小说引号台词原样；空镜 `""`
- 外貌只取原文可见特征

### 4. 拆镜（不许缩写）

- 所有「」/“”/"" 引号台词必须出现在某个镜头的 `dialogue`，允许跨相邻镜头覆盖，禁止漏句删字。
- 字数 > C：按标点拆成相邻镜头（`s007` / `s007b` / …）。
- 每镜独立 `image_prompt`（换景别/角度）和英文 `motion_prompt`。
- 不要为时长改写台词，不要在 emotion 里写语速。

示例（额度 21 字，原文 33 字）：

原文：`他是我父亲。他死在地球那天，我在火星。我只想再看一次他唱歌的样子。`

- `s007` → `他是我父亲。他死在地球那天，我在火星。`（近景，手按芯片）
- `s007b` → `我只想再看一次他唱歌的样子。`（特写抬眼望窗外，运镜不同）

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

## 禁止

- 改写 / 删字 / 压短引号台词
- emotion 里写「快说／放慢」
- 编造原文没有的配饰
- 逐步跑 images/audio/videos（用 `produce`）
- 重新拉取聊天用 LLM

## 自救

- audio 自动拆成 `s007_p1/_p2`：说明分镜写宽了，按 `--strict` 提示改。
- 卡在加载模型、CPU≈0%：先 `doctor`。
