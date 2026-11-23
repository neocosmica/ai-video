/* 小说 → 竖屏短剧 控制台（无构建、无依赖）
   打开网页第一件事：选项目；之后所有东西都属于这个项目。 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ALL_STAGES = ["screenplay", "images", "audio", "videos", "music", "assemble"];
const STAGE_LABELS = {
  screenplay: "校验分镜",
  images: "静帧生成",
  audio: "对白配音",
  videos: "视频生成",
  music: "配乐生成",
  assemble: "合成混音",
};

const state = {
  project: "", // 当前打开的项目名（projects/<片名> 的 <片名>）
  tasks: [], // 只装当前项目的任务
  projects: [],
  novels: [],
  storyboards: [],
  selectedTaskId: null,
  logOffset: 0,
  followLog: true,
  selectedStages: new Set(ALL_STAGES),
  currentOut: "",
  currentStoryboard: "",
  sbData: null,
  sbPath: "",
  polling: null,
};

// ---------------------------------------------------------------- utilities
function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function toast(msg, bad = false) {
  const el = $("#start-error");
  el.hidden = false;
  el.textContent = msg;
  el.style.color = bad ? "var(--bad)" : "var(--muted)";
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, bad ? 8000 : 4000);
}

// ---------------------------------------------------------------- projects
async function refreshState() {
  try {
    const q = state.project ? `?project=${encodeURIComponent(state.project)}` : "";
    const data = await api(`/api/state${q}`);
    state.projects = data.projects || [];
    state.novels = data.novels || [];
    state.storyboards = data.storyboards || [];
    if (state.project) state.tasks = data.tasks || [];
    setConn(true);
    renderProjects();
    renderDatalists();
    renderTasks();
    renderShotsIfVisible();
  } catch (err) {
    setConn(false);
  }
}

function setConn(on) {
  const el = $("#conn");
  el.className = "conn " + (on ? "on" : "off");
  el.textContent = on ? "● 已连接" : "● 未连接（服务未启动？）";
}

let projectsSig = "";

function renderProjects() {
  const wrap = $("#project-grid");
  const sig = JSON.stringify([
    state.project,
    state.projects.map((p) => [p.name, p.path, p.has_final, p.has_storyboard, !!p.novel, p.tasks, p.last_activity]),
  ]);
  if (sig === projectsSig) return;
  projectsSig = sig;
  if (!state.projects.length) {
    wrap.innerHTML = `<div class="empty">还没有项目。下面新建一个。</div>`;
    return;
  }
  wrap.innerHTML = state.projects.map((p) => {
    const t = p.tasks || { done: 0, running: 0, failed: 0, total: 0 };
    const badges = [
      p.novel ? "📖 小说" : "",
      p.has_storyboard ? "📋 分镜" : "",
      p.has_final ? "🎬 成片" : "",
    ].filter(Boolean).join("　");
    const stat = [
      t.running ? `<span class="dot run">${t.running} 运行中…</span>` : "",
      t.done ? `<span class="dot done">${t.done} 完成</span>` : "",
      t.failed ? `<span class="dot fail">${t.failed} 失败</span>` : "",
      !t.total ? `<span class="dot">还没跑过任务</span>` : "",
    ].join("");
    const active = p.name === state.project ? " active" : "";
    return `<div class="project-card${active}" data-name="${esc(p.name)}" data-path="${esc(p.path)}">
      <div class="p-name">${esc(p.name)}${p.legacy ? ' <span class="hint">（旧布局）</span>' : ""}</div>
      <div class="p-meta hint mono">${esc(p.path)}/</div>
      <div class="p-badges">${badges || '<span class="hint">空项目</span>'}</div>
      <div class="p-stat">${stat}</div>
    </div>`;
  }).join("");
  $$(".project-card", wrap).forEach((el) => {
    el.addEventListener("click", () => openProject(el.dataset.name, el.dataset.path));
  });
}

function renderDatalists() {
  $("#novel-list").innerHTML = (state.novels || [])
    .map((n) => `<option value="${esc(n.path)}">${esc(n.name)}</option>`).join("");
  $("#sb-list").innerHTML = (state.storyboards || [])
    .map((n) => `<option value="${esc(n.path)}">${esc(n.name)}</option>`).join("");
}

async function createProject() {
  const name = $("#new-project-name").value.trim();
  if (!name) { pickerMsg("先填片名", true); return; }
  try {
    const p = await api("/api/projects/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, title: name }),
    });
    $("#new-project-name").value = "";
    await refreshState();
    openProject(p.name, p.path);
  } catch (err) {
    pickerMsg(`新建失败：${err.message}`, true);
  }
}

function pickerMsg(msg, bad = false) {
  const el = $("#picker-msg");
  el.hidden = false;
  el.textContent = msg;
  el.style.color = bad ? "var(--bad)" : "var(--muted)";
}

// 选项目 → 工作区所有字段归位到这个项目
async function openProject(name, path) {
  state.project = name;
  state.currentOut = path || `projects/${name}`;
  state.currentStoryboard = `${state.currentOut}/storyboard.json`;
  state.selectedTaskId = null;
  state.logOffset = 0;
  stageTrackSig = "";
  liveItemsSig = "";
  tasksSig = "";
  $("#log").textContent = "";
  $("#project-picker").hidden = true;
  $("#workspace").hidden = false;
  $("#col-tasks").hidden = false;
  $("#shots-card").hidden = false;
  $("#task-detail-card").hidden = true;
  $("#current-project-name").textContent = `${state.currentOut}/`;
  $("#title").value = name;
  $("#novel-path").value = state.novels.some((n) => n.path === `${state.currentOut}/novel.txt`)
    ? `${state.currentOut}/novel.txt` : "";
  $("#novel-text").value = "";
  $("#storyboard-path").value = `${state.currentOut}/storyboard.json`;
  clearShotCards();
  shotsLastOut = "";
  try {
    await api("/api/project/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch {}
  await refreshState();
  renderShots();
  scheduleShotsAutoRefresh();
}

function backToProjects() {
  state.project = "";
  state.selectedTaskId = null;
  if (state.polling) { clearInterval(state.polling); state.polling = null; }
  if (shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  $("#workspace").hidden = true;
  $("#col-tasks").hidden = true;
  $("#shots-card").hidden = true;
  $("#sb-editor-card").hidden = true;
  $("#task-detail-card").hidden = true;
  $("#project-picker").hidden = false;
  projectsSig = "";
  refreshState();
}

// ---------------------------------------------------------------- start form
function renderStagePicks() {
  const wrap = $("#stage-picks");
  wrap.innerHTML = "";
  for (const s of ALL_STAGES) {
    const el = document.createElement("label");
    el.className = "stage-pick" + (state.selectedStages.has(s) ? " on" : "");
    el.innerHTML = `<input type="checkbox" ${state.selectedStages.has(s) ? "checked" : ""}>${STAGE_LABELS[s]}`;
    el.querySelector("input").addEventListener("change", (ev) => {
      if (ev.target.checked) state.selectedStages.add(s);
      else state.selectedStages.delete(s);
      el.classList.toggle("on", ev.target.checked);
    });
    wrap.appendChild(el);
  }
}

async function loadNovel() {
  const path = $("#novel-path").value.trim();
  if (!path) { toast("先填小说文件路径", true); return; }
  try {
    const data = await api("/api/file/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    $("#novel-text").value = data.content;
    toast(`已载入 ${path}（${data.content.length} 字）`);
  } catch (err) {
    toast(`载入失败：${err.message}`, true);
  }
}

async function startTask() {
  const payload = {
    title: $("#title").value.trim(),
    name: state.project,
    novel: $("#novel-path").value.trim(),
    novel_text: $("#novel-text").value.trim(),
    storyboard: $("#storyboard-path").value.trim(),
    out: state.currentOut,
    stages: [...state.selectedStages],
    only: $("#only-shots").value.trim(),
    force: $("#force").checked,
  };
  if (!payload.novel && !payload.novel_text) {
    toast("请先载入小说或直接粘贴片段", true);
    return;
  }
  if (!payload.stages.length) {
    toast("至少选择一个阶段", true);
    return;
  }
  try {
    const job = await api("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast(`任务已启动：${job.id}`);
    await refreshState();
    selectTask(job.id);
  } catch (err) {
    toast(`启动失败：${err.message}`, true);
  }
}

// ---------------------------------------------------------------- task list
let tasksSig = "";

function renderTasks() {
  const wrap = $("#task-list");
  if (!state.tasks.length) {
    if (tasksSig !== "empty") {
      tasksSig = "empty";
      wrap.innerHTML = `<div class="empty">这个项目还没有任务。左侧启动一个。</div>`;
    }
    return;
  }
  // 内容没变就不重建 DOM：任务列表轮询时重建会让列表闪烁
  const sig = JSON.stringify([
    state.selectedTaskId,
    state.tasks.map((t) => [t.id, t.name, t.status, t.created_at, (t.stages || []).map((s) => [s.key, s.status])]),
  ]);
  if (sig === tasksSig) return;
  tasksSig = sig;
  wrap.innerHTML = state.tasks.map((t) => {
    const dots = (t.stages || []).map((s) => {
      const cls = s.status === "done" ? "done" : s.status === "running" ? "run" : s.status === "failed" ? "fail" : "";
      return `<span class="dot ${cls}">${STAGE_LABELS[s.key] || s.key}${s.status === "running" ? "…" : s.status === "failed" ? "✕" : s.status === "done" ? "✓" : ""}</span>`;
    }).join("");
    const active = t.id === state.selectedTaskId ? " active" : "";
    return `<div class="task-item${active}" data-id="${t.id}">
      <div class="t-top">
        <span class="t-name">${esc(t.name || t.out)}</span>
        <span class="badge ${t.status}">${statusLabel(t.status)}</span>
        <span class="t-time">${fmtTime(t.created_at)}</span>
      </div>
      <div class="t-stages">${dots}</div>
    </div>`;
  }).join("");
  $$(".task-item", wrap).forEach((el) => {
    el.addEventListener("click", () => selectTask(el.dataset.id));
  });
}

function statusLabel(status) {
  return { running: "运行中", success: "完成", failed: "失败", interrupted: "中断" }[status] || status;
}

// ---------------------------------------------------------------- task detail
async function selectTask(id) {
  state.selectedTaskId = id;
  state.logOffset = 0;
  state.followLog = true;
  stageTrackSig = "";
  liveItemsSig = "";
  // 新任务新日志：清掉上一个任务的日志，别让旧失败信息挂在下面
  $("#log").textContent = "";
  const job = state.tasks.find((t) => t.id === id);
  if (!job) return;
  $("#task-detail-card").hidden = false;
  $("#task-detail-title").textContent = `任务：${job.name || job.out}`;
  renderStages(job);
  $("#task-meta").textContent = `输出：${job.out} · 创建：${fmtTime(job.created_at)}`;
  $("#task-stop").disabled = job.status !== "running";
  $("#shots-out-label").textContent = job.out;
  state.currentOut = job.out;
  state.currentStoryboard = `${job.out}/storyboard.json`;
  startLogPolling(id);
  renderTasks();
  renderShotsIfVisible();
}

let stageTrackSig = "";

function renderStages(job) {
  const wrap = $("#stage-track");
  const sig = JSON.stringify([(job.stages || []).map((s) => [s.key, s.status])]);
  if (sig !== stageTrackSig) {
    stageTrackSig = sig;
    wrap.innerHTML = (job.stages || []).map((s) => {
      const cls = s.status === "done" ? "done" : s.status === "running" ? "running" : s.status === "failed" ? "failed" : "";
      const icon = s.status === "done" ? "✓" : s.status === "running" ? "⏳" : s.status === "failed" ? "✕" : "·";
      return `<div class="stage-step ${cls}">${icon} ${STAGE_LABELS[s.key] || s.key}</div>`;
    }).join("");
  }
  renderOverall(job);
  renderLiveItems(job);
}

// STAGE_WEIGHTS 与后端 pipeline/web.py 保持一致
const STAGE_WEIGHTS = {
  screenplay: 0.02, images: 0.18, audio: 0.10, videos: 0.60, music: 0.05, assemble: 0.05,
};

function renderOverall(job) {
  const fill = $("#overall-fill");
  const pctEl = $("#overall-pct");
  const stageEl = $("#overall-stage");
  let pct = 0;
  let currentLabel = "";
  for (const s of job.stages || []) {
    const w = STAGE_WEIGHTS[s.key] || 0.1;
    if (s.status === "done") pct += w * 100;
    else if (s.status === "running") {
      const sp = job.progress?.stage_pct?.[s.key];
      const inner = typeof sp === "number" && sp > 0 ? Math.min(99, sp) : (job.progress?.current ? 8 : 2);
      pct += w * inner;
      currentLabel = `正在：${STAGE_LABELS[s.key] || s.key}`;
    }
  }
  if (job.status === "success") { pct = 100; currentLabel = "完成"; }
  if (job.status === "failed") currentLabel = "失败（可点「重跑失败阶段」）";
  if (job.status === "interrupted") currentLabel = "已中断";
  const pctText = `${Math.min(100, pct).toFixed(1)}%`;
  const width = `${Math.min(100, pct).toFixed(1)}%`;
  const fillCls = "overall-fill " + (job.status === "success" ? "done" : job.status === "failed" ? "failed" : "");
  // 值没变不写 DOM：进度条每 1.2s 刷新一次，重写会打断 CSS 过渡动画
  if (fill.style.width !== width) fill.style.width = width;
  if (fill.className !== fillCls) fill.className = fillCls;
  if (pctEl.textContent !== pctText) pctEl.textContent = pctText;
  if (stageEl.textContent !== currentLabel) stageEl.textContent = currentLabel;
}

function fmtEta(sec) {
  if (sec == null || isNaN(sec)) return "";
  sec = Math.round(sec);
  if (sec < 60) return `剩 ${sec}s`;
  const m = Math.floor(sec / 60);
  return `剩 ${m}m${String(sec % 60).padStart(2, "0")}s`;
}

let liveItemsSig = "";

function renderLiveItems(job) {
  const wrap = $("#live-items");
  const items = job.progress?.items || [];
  // 任务已结束就不再显示逐条进度：避免「整理 s001…」这类条目结束后还挂着
  if (job.status !== "running") {
    if (liveItemsSig !== "ended") { liveItemsSig = "ended"; wrap.innerHTML = ""; }
    return;
  }
  const running = items.filter((i) => i.status === "running");
  const recent = items.filter((i) => i.status !== "running").slice(-4);
  const shown = [...running, ...recent].slice(0, 8);
  const sig = JSON.stringify(shown.map((i) => [i.id, i.status, i.step, i.total, i.pct, i.eta_s]));
  if (sig === liveItemsSig) return;
  liveItemsSig = sig;
  if (!shown.length) { wrap.innerHTML = ""; return; }
  wrap.innerHTML = shown.map((i) => {
    const cls = i.status === "done" ? "done" : i.status === "failed" ? "failed" : "";
    const stepText = i.total > 1 ? `${i.step}/${i.total}` : "";
    const pct = i.status === "done" ? 100 : i.pct || 0;
    const eta = i.status === "running" ? fmtEta(i.eta_s) : i.status === "done" ? "✓" : "✕";
    return `<div class="live-item ${cls}">
      <span class="li-label" title="${esc(i.label || "")}">${esc(i.label || i.id)}</span>
      <span class="li-step mono">${stepText}</span>
      <div class="li-bar"><div class="li-fill" style="width:${pct}%"></div></div>
      <span class="li-eta">${eta}</span>
    </div>`;
  }).join("");
}

function startLogPolling(id) {
  if (state.polling) {
    clearInterval(state.polling);
    state.polling = null;
  }
  const pull = async () => {
    // 任务切换后停止旧轮询
    if (state.selectedTaskId !== id) { clearInterval(state.polling); state.polling = null; return; }
    try {
      const data = await api(`/api/log/${id}?after=${state.logOffset}`);
      if (state.selectedTaskId !== id) return;
      if (data.lines && data.lines.length) {
        appendLogLines(data.lines);
        state.logOffset = data.total;
      }
      const job = state.tasks.find((t) => t.id === id);
      if (job) {
        renderStages(job);
        $("#task-stop").disabled = job.status !== "running";
        if (job.status === "failed" || job.status === "success" || job.status === "interrupted") {
          renderShotsIfVisible();
        }
      }
    } catch (err) {
      // 忽略瞬时错误
    }
  };
  pull();
  state.polling = setInterval(pull, 1200);
}

function appendLogLines(lines) {
  const log = $("#log");
  const nearBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
  const frag = document.createElement("div");
  frag.innerHTML = lines.map((line) => {
    const bad = /(Traceback|Error|SystemExit|失败|找不到|缺少)/.test(line);
    return `<div class="${bad ? "err-line" : ""}">${esc(line)}</div>`;
  }).join("");
  log.appendChild(frag);
  while (log.childNodes.length > 4000) log.removeChild(log.firstChild);
  if (nearBottom || state.followLog) log.scrollTop = log.scrollHeight;
}

async function stopTask() {
  const id = state.selectedTaskId;
  if (!id) return;
  try {
    await api("/api/task/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    toast("已请求停止");
  } catch (err) {
    toast(`停止失败：${err.message}`, true);
  }
}

async function rerunFailed() {
  const job = state.tasks.find((t) => t.id === state.selectedTaskId);
  if (!job) return;
  const failed = (job.stages || []).filter((s) => s.status === "failed").map((s) => s.key);
  if (!failed.length) {
    toast("没有失败阶段可重跑", true);
    return;
  }
  // 从最早失败的那个阶段起重跑其后全部
  const keys = ALL_STAGES.filter((s) => job.stages.some((x) => x.key === s));
  const firstIdx = Math.min(...failed.map((s) => keys.indexOf(s)));
  const stages = keys.slice(firstIdx);
  try {
    const newJob = await api("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: `${job.name || job.out} · 续跑`,
        novel: job.novel,
        out: job.out,
        stages,
        only: job.only,
      }),
    });
    toast(`已启动续跑：${stages.join(" → ")}`);
    await refreshState();
    selectTask(newJob.id);
  } catch (err) {
    toast(`续跑失败：${err.message}`, true);
  }
}

async function clearTasks() {
  try {
    const res = await api("/api/tasks/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    toast(res.kept ? `已清空本项目的记录（运行中的 ${res.kept} 个任务保留）` : "已清空本项目的记录");
    if (!state.tasks.some((t) => t.id === state.selectedTaskId && t.status === "running")) {
      // 当前选中的任务被清掉时，收起详情面板
      state.selectedTaskId = null;
      $("#task-detail-card").hidden = true;
      $("#log").textContent = "";
    }
    await refreshState();
  } catch (err) {
    toast(`清空失败：${err.message}`, true);
  }
}

// ---------------------------------------------------------------- shots
let shotsTimer = null;
let mediaCache = null;
let renderShotsSeq = 0; // 防止重叠的异步渲染互相覆盖
let shotsLastOut = "";
const shotCards = new Map(); // shot id -> 已渲染的卡片 DOM（增量更新，不整格重建）

function clearShotCards() {
  for (const [, card] of shotCards) card.remove();
  shotCards.clear();
}

function ensureShotCard(id) {
  let card = shotCards.get(id);
  if (!card) {
    const node = $("#shot-card-tpl").content.cloneNode(true);
    card = $(".shot-card", node);
    $(".shot-id", card).textContent = id;
    shotCards.set(id, card);
  }
  return card;
}

function mediaSig(entry) {
  return entry ? `${entry.mtime}:${entry.size}` : "";
}

function renderShotsIfVisible() {
  if (!state.project) return;
  $("#shots-card").hidden = false;
  renderShots();
  scheduleShotsAutoRefresh();
}

function scheduleShotsAutoRefresh() {
  const on = $("#auto-refresh")?.checked;
  if (shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  if (on !== false) shotsTimer = setInterval(renderShots, 5000);
}

async function renderShots() {
  const out = state.currentOut;
  if (!out || !state.project) return;
  const seq = ++renderShotsSeq;
  $("#shots-out-label").textContent = out;
  // 换了输出目录：旧卡片全部清掉，避免闪现上一个片的产物
  if (out !== shotsLastOut) {
    clearShotCards();
    shotsLastOut = out;
    mediaCache = null;
  }
  // 1) 拉产物清单
  let media = null;
  try {
    // 只编码非路径分隔符，后端 safe_path 接受 projects/xxx 形式
    media = await api(`/api/media?out=${out.split("/").map(encodeURIComponent).join("/")}`);
    mediaCache = media;
  } catch (err) {
    media = mediaCache;
  }
  if (seq !== renderShotsSeq) return;
  if (!media && shotCards.size) return; // 产物清单暂时拿不到：保持现状，别清空预览
  if (media?.final) $("#final-badge").hidden = false;
  else $("#final-badge").hidden = true;
  // 2) 拉分镜（决定镜头顺序和台词）
  let shots = [];
  let reuseMap = {};
  try {
    const data = await api("/api/storyboard/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: `${out}/storyboard.json` }),
    });
    shots = data.storyboard.shots || [];
    for (const s of shots) if (s.reuse_frame) reuseMap[s.id] = s.reuse_frame;
  } catch (err) {
    // 没分镜就只按产物文件展示
    shots = [];
  }
  if (seq !== renderShotsSeq) return;
  // 3) 进度（视频 n/total 等来自日志解析）
  const job = state.tasks.find((t) => t.id === state.selectedTaskId);
  const progItems = {};
  if (job?.progress?.items) {
    // 只认 shot: 前缀的镜头条目；misc:配乐、assemble:整理 等不对应镜头卡
    for (const it of job.progress.items) {
      if ((it.id || "").startsWith("shot:")) progItems[it.label] = it;
    }
  }
  // 4) 如果连分镜都没有，先展示已有媒体
  if (!shots.length) {
    for (const f of media?.frames || []) shots.push({ id: f.id });
    for (const c of media?.clips || []) if (!shots.some((s) => s.id === c.id)) shots.push({ id: c.id });
  }
  const grid = $("#shots-grid");
  if (!shots.length) {
    clearShotCards();
    if (!grid.querySelector(":scope > .empty")) {
      grid.innerHTML = `<div class="empty">还没有产物。启动任务跑到「静帧生成」后，这里会出现每个镜头的图 / 视频 / 配音。</div>`;
    }
    return;
  }
  // 5) 增量渲染：卡片按 shot id 复用，只有媒体文件变化才重建预览
  const emptyEl = grid.querySelector(":scope > .empty");
  if (emptyEl) emptyEl.remove();
  const desired = [];
  for (const shot of shots) {
    desired.push(shot.id);
    renderShotCard(shot, media, reuseMap, progItems);
  }
  // 删掉不再存在的镜头卡
  for (const [id, card] of [...shotCards]) {
    if (!desired.includes(id)) {
      card.remove();
      shotCards.delete(id);
    }
  }
  // 保持分镜顺序（insertBefore 移动已有节点，不会重新加载媒体）
  let prev = null;
  for (const id of desired) {
    const card = shotCards.get(id);
    const expected = prev ? prev.nextSibling : grid.firstChild;
    if (card !== expected) grid.insertBefore(card, prev ? prev.nextSibling : grid.firstChild);
    prev = card;
  }
}

function renderShotCard(shot, media, reuseMap, progItems) {
  const card = ensureShotCard(shot.id);
  const clipEntry = (media?.clips || []).find((c) => c.id === shot.id);
  const frameEntry = (media?.frames || []).find((f) => f.id === (reuseMap[shot.id] || shot.id));
  const voiceEntry = (media?.voices || []).find((v) => v.id === shot.id);
  const prog = progItems[shot.id];
  const failed = shot.video_error || (prog && prog.status === "failed");
  const sigs = {
    clip: mediaSig(clipEntry),
    frame: mediaSig(frameEntry),
    voice: mediaSig(voiceEntry),
  };
  // 媒体文件没变就不动 DOM：这是「视频闪 / 加载圈不停」的修复点
  if (JSON.stringify(sigs) !== JSON.stringify(card._sig || {})) {
    card._sig = sigs;
    card._hasFrame = !!frameEntry;
    const preview = $(".shot-preview", card);
    const audioEl = $(".shot-audio", card);
    if (clipEntry) {
      const poster = frameEntry ? ` poster="${esc(frameEntry.url)}"` : "";
      const preload = frameEntry ? "none" : "metadata"; // 有静帧当海报就不预取，点了播放才加载
      preview.innerHTML = `<video src="${esc(clipEntry.url)}"${poster} muted ${preload} controls></video>`;
    } else if (frameEntry) {
      preview.innerHTML = `<img src="${esc(frameEntry.url)}" loading="lazy" alt="">`;
    } else {
      preview.innerHTML = "";
    }
    audioEl.innerHTML = voiceEntry ? `<audio src="${esc(voiceEntry.url)}" controls preload="none"></audio>` : "";
  }
  // 台词 / 状态 / 进度条：就地文本更新，值没变不写 DOM
  const dlg = $(".shot-dialogue", card);
  if (dlg.textContent !== (shot.dialogue || "")) dlg.textContent = shot.dialogue || "";
  const statusEl = $(".shot-status", card);
  let statusText, statusCls;
  if (clipEntry) {
    statusText = failed ? `视频失败：${shot.video_error || ""}` : "视频已生成";
    statusCls = "shot-status " + (failed ? "bad" : "ok");
  } else if (frameEntry) {
    statusText = failed ? `视频失败：${shot.video_error || "见日志"}` : "静帧已生成" + (shot.dialogue ? "，视频待跑" : "（空镜）");
    statusCls = "shot-status " + (failed ? "bad" : "");
  } else {
    statusText = failed ? `失败：${shot.video_error || "见日志"}` : "静帧未生成";
    statusCls = "shot-status " + (failed ? "bad" : "");
  }
  if (statusEl.textContent !== statusText) statusEl.textContent = statusText;
  if (statusEl.className !== statusCls) statusEl.className = statusCls;
  const progEl = $(".shot-progress", card);
  const progFill = $(".shot-progress-fill", card);
  const progText = $(".shot-progress-text", card);
  let progShow = false, progWidth = "", progLabel = "", progFillCls = "shot-progress-fill";
  if (prog && prog.status === "running") {
    progShow = true;
    progWidth = `${prog.pct || 0}%`;
    progLabel = prog.total > 1 ? `${prog.step}/${prog.total} · ${fmtEta(prog.eta_s)}` : "生成中…";
  } else if (prog && prog.status === "failed") {
    progShow = true;
    progWidth = "100%";
    progFillCls += " failed";
    progLabel = "失败";
  } else if (clipEntry || (frameEntry && !shot.dialogue)) {
    progShow = true;
    progWidth = "100%";
    progFillCls += " done";
  }
  progEl.classList.toggle("show", progShow);
  progText.classList.toggle("show", progShow && (prog?.status === "running" || prog?.status === "failed"));
  if (progFill.style.width !== progWidth) progFill.style.width = progWidth;
  if (progFill.className !== progFillCls) progFill.className = progFillCls;
  if (progText.textContent !== progLabel) progText.textContent = progLabel;
  // 重跑按钮只在卡片首次创建时挂一次
  if (!card._btns) {
    card._btns = true;
    const actions = $(".shot-actions", card);
    const btnFrame = document.createElement("button");
    btnFrame.textContent = "↻ 静帧";
    btnFrame.onclick = () => rerunShots([shot.id], ["images"]);
    const btnVideo = document.createElement("button");
    btnVideo.textContent = "↻ 视频";
    btnVideo.onclick = () => rerunShots([shot.id], card._hasFrame ? ["videos"] : ["images", "videos"]);
    actions.appendChild(btnFrame);
    actions.appendChild(btnVideo);
  }
}

async function rerunShots(shots, stages) {
  const out = state.currentOut;
  if (!out) return;
  try {
    const job = await api("/api/shots/rerun", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        out,
        storyboard: state.currentStoryboard,
        shots,
        stages,
        novel: $("#novel-path").value.trim(),
      }),
    });
    toast(`已启动镜头重跑：${shots.join(", ")}`);
    await refreshState();
    selectTask(job.id);
  } catch (err) {
    toast(`重跑失败：${err.message}`, true);
  }
}

// ---------------------------------------------------------------- storyboard editor
async function openStoryboardEditor() {
  let path = $("#storyboard-path").value.trim();
  const out = state.currentOut;
  if (!path && out) path = `${out}/storyboard.json`;
  if (!path) { toast("先选项目", true); return; }
  const name = state.project || (out ? out.split("/").pop() : "");
  let sb;
  try {
    const data = await api("/api/storyboard/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    sb = data.storyboard;
  } catch (err) {
    // 新建一个空骨架
    sb = {
      title: $("#title").value.trim() || name || "新片",
      style: "写实华语短剧，自然肤色，电影感光线，浅景深",
      music_prompt: "English instrumental BGM, no vocals",
      characters: [],
      shots: [],
    };
  }
  state.sbData = sb;
  state.sbPath = path;
  $("#sb-editor-card").hidden = false;
  $("#sb-path-label").textContent = path;
  renderStoryboardEditor();
}

function renderStoryboardEditor() {
  const sb = state.sbData;
  if (!sb) return;
  const summary = $("#sb-summary");
  summary.textContent = `${sb.characters?.length || 0} 个角色 · ${sb.shots?.length || 0} 镜${sb.title ? ` · 《${sb.title}》` : ""}`;
  const wrap = $("#sb-shots");
  wrap.innerHTML = "";

  // 角色简表
  const charBlock = document.createElement("div");
  charBlock.className = "sb-shot";
  charBlock.style.gridColumn = "1 / -1";
  charBlock.innerHTML = `<div class="shot-meta"><span class="mono">characters</span></div>
    <div class="fields">
      <label>JSON</label>
      <textarea rows="3" id="sb-chars">${esc(JSON.stringify(sb.characters || [], null, 1))}</textarea>
    </div>`;
  wrap.appendChild(charBlock);

  (sb.shots || []).forEach((shot, idx) => {
    const row = document.createElement("div");
    row.className = "sb-shot";
    row.innerHTML = `<div class="shot-meta">
        <span class="mono">#${idx + 1} ${esc(shot.id || "")}</span>
        <button class="ghost sb-del">删除</button>
      </div>
      <div class="fields">
        <label>id</label><input data-f="id" value="${esc(shot.id || "")}">
        <label>location</label><input data-f="location" value="${esc(shot.location || "")}">
        <label>characters</label><input data-f="characters" value="${esc((shot.characters || []).join(","))}">
        <label>image_prompt</label><textarea data-f="image_prompt" rows="2">${esc(shot.image_prompt || "")}</textarea>
        <label>motion_prompt</label><textarea data-f="motion_prompt" rows="2">${esc(shot.motion_prompt || "")}</textarea>
        <label>speaker</label><input data-f="speaker" value="${esc(shot.speaker || "")}">
        <label>dialogue</label><textarea data-f="dialogue" rows="2">${esc(shot.dialogue || "")}</textarea>
        <label>emotion</label><input data-f="emotion" value="${esc(shot.emotion || "")}">
      </div>`;
    $(".sb-del", row).addEventListener("click", () => {
      state.sbData.shots.splice(idx, 1);
      renderStoryboardEditor();
    });
    $$(".fields [data-f]", row).forEach((input) => {
      input.addEventListener("change", () => {
        const f = input.dataset.f;
        if (f === "characters") {
          shot.characters = input.value.split(/[,，\s]+/).filter(Boolean);
        } else {
          shot[f] = input.value.trim();
        }
      });
    });
    wrap.appendChild(row);
  });

  const addBtn = document.createElement("button");
  addBtn.textContent = "＋ 添加镜头";
  addBtn.className = "ghost";
  addBtn.addEventListener("click", () => {
    state.sbData.shots.push({
      id: `s${String((state.sbData.shots?.length || 0) + 1).padStart(3, "0")}`,
      characters: [], location: "", image_prompt: "", motion_prompt: "",
      speaker: "", dialogue: "", emotion: "calm",
    });
    renderStoryboardEditor();
  });
  wrap.appendChild(addBtn);
}

async function saveStoryboard() {
  if (!state.sbData) return;
  // 角色文本框
  const charsEl = $("#sb-chars");
  if (charsEl) {
    try {
      state.sbData.characters = JSON.parse(charsEl.value);
    } catch (err) {
      toast(`characters 不是合法 JSON：${err.message}`, true);
      return;
    }
  }
  try {
    await api("/api/storyboard/write", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.sbPath, storyboard: state.sbData }),
    });
    toast(`已保存 ${state.sbPath}`);
  } catch (err) {
    toast(`保存失败：${err.message}`, true);
  }
}

async function validateStoryboard() {
  await saveStoryboard();
  try {
    const data = await api("/api/storyboard/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.sbPath, strict: true }),
    });
    $("#sb-msg").textContent = `${data.shots} 镜 · 单镜台词额度 ${data.limit_chars} 字 ≈ ${data.max_seconds}s`;
    $("#sb-summary").textContent = data.report || "";
    toast(`校验通过：${data.shots} 镜，${data.dialogue_lines} 条对白`);
  } catch (err) {
    $("#sb-summary").textContent = "";
    toast(`校验失败：${err.message}`, true);
  }
}

// ---------------------------------------------------------------- SSE + init
function connectEvents() {
  const es = new EventSource("/api/events");
  es.onopen = () => setConn(true);
  es.onerror = () => setConn(false);
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.event === "change") refreshState();
    } catch {}
  };
}

async function loadBudget() {
  try {
    const b = await api("/api/budget");
    $("#budget-info").textContent =
      `单镜上限 ${b.frames} 帧 ≈ ${b.seconds}s · 台词额度约 ${b.dialogue_chars} 字/镜（${b.total_gib} GB 内存，${b.resolution}@${b.fps}fps）`;
  } catch {
    $("#budget-info").textContent = "";
  }
}

function bind() {
  $("#novel-load").addEventListener("click", loadNovel);
  $("#start-btn").addEventListener("click", startTask);
  $("#storyboard-edit").addEventListener("click", openStoryboardEditor);
  $("#sb-save").addEventListener("click", saveStoryboard);
  $("#sb-validate").addEventListener("click", validateStoryboard);
  $("#sb-close").addEventListener("click", () => { $("#sb-editor-card").hidden = true; });
  $("#task-stop").addEventListener("click", stopTask);
  $("#task-rerun-failed").addEventListener("click", rerunFailed);
  $("#tasks-clear").addEventListener("click", clearTasks);
  $("#auto-refresh")?.addEventListener("change", scheduleShotsAutoRefresh);
  $("#new-project-btn").addEventListener("click", createProject);
  $("#new-project-name").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") createProject();
  });
  $("#back-to-projects").addEventListener("click", backToProjects);
}

bind();
renderStagePicks();
loadBudget();
refreshState();
connectEvents();
setInterval(refreshState, 4000);
