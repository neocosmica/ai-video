---
name: novel-video
description: >-
  把小说做成竖屏短剧：先按本机总内存查单镜上限，再写分镜 JSON（长对白拆镜不删字），
  最后一条命令成片，并把跑 produce 的终端调到前台让用户能看进度。
  Use when the user asks to make a short drama / 短剧 / storyboard
  from a novel, or mentions novel-video / produce / 分镜 / 看进展 / 打开终端.
disable-model-invocation: true
---

# 小说 → 竖屏短剧

**你只负责：读小说、写分镜 JSON。** 写完后用一条脚本成片；不要逐步盯 images / audio / videos / assemble，也不要下载或调用本机聊天 LLM。

## 流程

```
读小说 → budget（拿单镜上限）→ 写 storyboard.json → screenplay --strict → produce
```

### 1. 准备输入

- 小说：`input/<片名>_novel.txt`（或用户指定路径）
- 分镜目标：`input/<片名>_storyboard.json`
- 输出：`output/<片名>/`

### 2. 先拿单镜上限（必做）

```bash
./scripts/run.sh budget --out output/<片名>
```

记下输出里的：

- **单镜上限**：N 帧 ≈ S 秒
- **单镜台词额度**：约 C 字

这是按本机**总内存配置**查经验 token 预算表、再按 `config.json` 分辨率现算出来的，不读系统实时状态。超额度的对白必须拆成相邻多镜，**不许删字压短**。

### 3. 写分镜 JSON

最小结构：

```json
{
  "title": "短剧标题",
  "style": "写实华语短剧，自然肤色，电影感光线，浅景深",
  "music_prompt": "English instrumental BGM, no vocals, melancholic strings",
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
      "image_prompt": "中景，雨夜天台，林晚侧身望向城市灯光，浅景深",
      "motion_prompt": "Slow cinematic push in, subtle rain on coat",
      "speaker": "林晚",
      "dialogue": "他是我父亲。",
      "emotion": "sad"
    }
  ]
}
```

字段约束：

| 字段 | 规则 |
|---|---|
| `gender` | `female` / `male` |
| `voice` | 女 `Serena`/`Vivian`；男 `Dylan`/`Uncle_Fu`/`Eric` |
| `emotion` | `calm`/`warm`/`sad`/`tense`/`bitter`/`nostalgic`/`surprised`/`whisper` 等 |
| `motion_prompt` / `music_prompt` | **必须英文** |
| `dialogue` | 小说引号台词**原样**；空镜用 `""` |
| `image_prompt` | 中文：景别、地点、动作、光线；不要写对白字 |
| 外貌 | 只取原文可见特征，勿编造配饰 |

### 静帧要为视频阶段服务（两条硬规则，踩过坑）

`image_prompt` 决定静帧，静帧直接决定 A2V 成片质量。以下两条必须遵守：

1. **有台词的镜头（`dialogue` 非空且画面有人）：嘴不要紧闭。**
   静帧里嘴紧闭，A2V 生成视频时口型对不上，嘴部僵硬甚至全程不动。
   写法：在 `image_prompt` 里明确加「嘴唇微张」「似要开口」「正要说话的瞬间」等描述。
2. **画面有人脸的镜头：静帧里必须有清晰人脸。**
   人脸太小（远景）、背对、被遮挡或虚化时，视频阶段容易生成异常人脸（扭曲、变形、串成别人的脸）。
   写法：人物出镜用近景／中近景，写明「正脸或四分之三侧脸」「面部清晰」「光线落在脸上」；
   确实要用大远景交代环境时，让人物背对镜头或虚化在画面边缘，不给视频模型「补脸」的机会。

### 4. 拆镜规则（核心）

- 小说里所有「」/“”/"" 引号台词必须出现在某个镜头的 `dialogue`，允许跨相邻镜头连续覆盖，**禁止漏句、删字**。
- 一句对白字数 > 单镜额度 C：按标点语义拆成相邻镜头（`s007` / `s007b` / `s007c`）。
- 拆开的每镜写**独立**的 `image_prompt`（换景别或角度）和英文 `motion_prompt`，否则几段视频像卡带。
- **不要**为时长改写台词，**不要**在 emotion 里写「快说／放慢」。

**示例**：额度 21 字，原文 33 字：

> 他是我父亲。他死在地球那天，我在火星。我只想再看一次他唱歌的样子。

拆成：

```json
[
  {
    "id": "s007",
    "image_prompt": "近景，林晚低头，手按胸口芯片，眼眶微红",
    "motion_prompt": "Locked-off camera, faint hand tremor",
    "speaker": "林晚",
    "dialogue": "他是我父亲。他死在地球那天，我在火星。",
    "emotion": "sad"
  },
  {
    "id": "s007b",
    "image_prompt": "特写，林晚抬眼望向窗外火星夜空",
    "motion_prompt": "Slow tilt up to the window lights",
    "speaker": "林晚",
    "dialogue": "我只想再看一次他唱歌的样子。",
    "emotion": "nostalgic"
  }
]
```

### 5. 硬校验

```bash
./scripts/run.sh screenplay --strict \
  --storyboard input/<片名>_storyboard.json \
  --novel input/<片名>_novel.txt \
  --out output/<片名>
```

失败时按提示拆镜重写，再跑直到通过。

### 6. 一键成片

```bash
./scripts/run.sh produce \
  --storyboard input/<片名>_storyboard.json \
  --novel input/<片名>_novel.txt \
  --out output/<片名>
```

成片在 `output/<片名>/final.mp4`。输出目录已有分镜时默认不覆盖（audio 可能已自动拆过镜），要覆盖加 `--force`。启动后不要再插手各阶段；失败再查 `./scripts/run.sh doctor`。

### 7. 一定要把运行的终端调到前台（必做）

`produce` 一启动，**立刻**把这条终端在前台打开，用户要能自己看进度条；不要等用户来问。

1. 在终端记录目录里找这条终端（按修改时间取最新、标题对得上 produce 的那份）：

```bash
cd "/Users/wangxinyu/.cursor/projects/Users-wangxinyu-ai/terminals" && ls -t *.txt | head -3
```

2. 打开文件头，确认 `command:` 是刚才那条 `produce`、`status: running`；文件头的 `pid:` 就是终端 id（文件名里的数字只是记录编号，不一定可用）。
3. 用 `open_resource` 打开 `vscode-terminal:/<pid>`：

```json
{"uri": "vscode-terminal:/<pid>"}
```

注意几点：

- 终端文件要**重新读一遍**再打开；跑起来几秒后才会生成这份记录。
- 找到文件后，把 `pid:` 那行的值当成终端 id，不要拿文件名当 id 硬试。
- 打开失败就退回：让用户在 Cursor 的终端列表里点标题为 produce 的那条，并说明是哪条。
- 用户说「看进展」「打开终端」时同样走这一步，不用重新起进程。

### 8. 汇报进展的口径

用户问进展时，**不要靠猜**，读两处再回答：

```bash
tail -30 "/Users/wangxinyu/.cursor/projects/Users-wangxinyu-ai/terminals/<pid>.txt"
ls -lt output/<片名>/frames | head
```

- 阶段顺序固定是 `screenplay → images → audio → videos → music → assemble`，当前在哪个 `== stage ==` 就看最后出现的那一个。
- 静帧进度可以数字数 `frames/*.png`；视频进度看 `logs/<shot>.log`。
- 顺带把最新出好的静帧贴给用户看，比只报百分比直观。

## 禁止事项

- 不改写、不删字、不压短小说引号台词
- 不为了时长在 emotion 里写语速要求
- 不编造原文没有的配饰 / 外貌
- 不逐步跑 images/audio/videos（用 `produce`）
- 不让 `produce` 在后台默默跑而不把终端调到前台
- 不重新拉取聊天用 LLM

## 失败自救

- `audio` 若仍发现某镜念不完，会在句中停顿处自动拆成 `s007_p1/_p2` 并复用静帧。看到这类日志说明分镜写宽了，回头按 `--strict` 提示改。
- 报 `[metal::load_device] No Metal device available`：进程跑在沙箱里，拿不到 GPU。整条 `produce` 要在沙箱外跑（申请 `all` 权限），别去查模型或 venv。
- 卡在「加载文生图」或 Gemma 之前、CPU≈0%：先 `./scripts/run.sh doctor`。
