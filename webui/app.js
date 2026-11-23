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
let refreshInFlight = false;
let refreshQueued = false;

async function refreshState() {
  // 合并并发刷新：SSE 狂推 change 时不要叠一堆 /api/state，避免偶发失败把连接灯打闪
  if (refreshInFlight) {
    refreshQueued = true;
    return;
  }
  refreshInFlight = true;
  try {
    const q = state.project ? `?project=${encodeURIComponent(state.project)}` : "";
    const data = await api(`/api/state${q}`);
    state.projects = data.projects || [];
    state.novels = data.novels || [];
    state.storyboards = data.storyboards || [];
    if (state.project) {
      // 合并任务：保留本地已通过进度 SSE 拿到的更新 seq，避免 4s 全量刷新把条打回去
      const prevById = Object.create(null);
      for (const t of state.tasks || []) prevById[t.id] = t;
      state.tasks = (data.tasks || []).map((t) => {
        const old = prevById[t.id];
        if (!old?.progress) return t;
        const oldSeq = Number(old.progress.seq) || 0;
        const newSeq = Number(t.progress?.seq) || 0;
        if (oldSeq > newSeq) {
          return { ...t, progress: old.progress, stages: old.stages?.length ? old.stages : t.stages };
        }
        return t;
      });
    }
    setConn(true);
    renderProjects();
    renderDatalists();
    renderTasks();
    // 列表刷新不覆盖进度 SSE；只同步按钮等元数据
    if (state.selectedTaskId) {
      const job = state.tasks.find((t) => t.id === state.selectedTaskId);
      if (job) {
        $("#task-stop").disabled = job.status !== "running";
        // 若尚无进度 SSE 数据，用 state 里的 progress 渲染一次
        if (job.progress && (Number(job.progress.seq) || 0) >= appliedProgressSeq) {
          appliedProgressSeq = Number(job.progress.seq) || appliedProgressSeq;
          renderStages(job);
        }
      }
    }
    renderShotsIfVisible();
  } catch (err) {
    setConn(false);
  } finally {
    refreshInFlight = false;
    if (refreshQueued) {
      refreshQueued = false;
      setTimeout(refreshState, 400);
    }
  }
}

// 连接灯：已连接立刻亮；未连接延迟确认，避免 SSE 重连/瞬时请求失败高频闪
let connShown = null; // null | true | false
let connOffTimer = null;

function setConn(on) {
  if (on) {
    if (connOffTimer) {
      clearTimeout(connOffTimer);
      connOffTimer = null;
    }
    if (connShown === true) return;
    connShown = true;
    const el = $("#conn");
    el.className = "conn on";
    el.textContent = "● 已连接";
    return;
  }
  if (connShown === false || connOffTimer) return;
  connOffTimer = setTimeout(() => {
    connOffTimer = null;
    if (connShown === false) return;
    connShown = false;
    const el = $("#conn");
    el.className = "conn off";
    el.textContent = "● 未连接（服务未启动？）";
  }, 2500);
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
async function openProject(name, path, { autoSelectRunning = true } = {}) {
  state.project = name;
  state.currentOut = path || `projects/${name}`;
  state.currentStoryboard = `${state.currentOut}/storyboard.json`;
  state.selectedTaskId = null;
  state.logOffset = 0;
  stageTrackSig = "";
  liveItemsSig = "";
  tasksSig = "";
  try { localStorage.setItem("nv_last_project", name); } catch {}
  $("#log").textContent = "";
  $("#project-picker").hidden = true;
  $("#workspace").hidden = false;
  $("#col-tasks").hidden = false;
  $("#shots-card").hidden = false;
  $("#task-detail-card").hidden = true;
  $("#current-project-name").textContent = name || state.currentOut;
  $("#title").value = name;
  // 默认路径固定到项目 novel.txt；有没有文件由自动载入决定
  const defaultNovel = `${state.currentOut}/novel.txt`;
  const known = (state.novels || []).find((n) => n.path === defaultNovel || n.path.startsWith(`${state.currentOut}/`));
  $("#novel-path").value = known?.path || defaultNovel;
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
  // refresh 后 novels 清单可能更全，再对齐一次默认路径并自动载入正文
  if (!(state.novels || []).some((n) => n.path === $("#novel-path").value.trim())) {
    const again = (state.novels || []).find((n) => n.path === defaultNovel) || (state.novels || []).find((n) => n.path.startsWith(`${state.currentOut}/`));
    $("#novel-path").value = again?.path || defaultNovel;
  }
  await loadNovel({ quiet: true });
  renderShots();
  scheduleShotsAutoRefresh();
  // 有运行中的任务就自动打开详情，避免重开网页后还要再点一次才看得到进度
  if (autoSelectRunning) {
    const running = state.tasks.find((t) => t.status === "running");
    if (running) selectTask(running.id);
  }
}

function backToProjects() {
  state.project = "";
  state.selectedTaskId = null;
  try { localStorage.removeItem("nv_last_project"); } catch {}
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

async function loadNovel({ quiet = false } = {}) {
  const path = $("#novel-path").value.trim();
  if (!path) {
    $("#novel-text").value = "";
    if (!quiet) toast("没有小说路径", true);
    return;
  }
  try {
    const data = await api("/api/file/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    $("#novel-text").value = data.content;
    if (!quiet) toast(`已载入 ${path}（${data.content.length} 字）`);
  } catch (err) {
    $("#novel-text").value = "";
    // 打开项目时文件还不存在：安静失败，留给用户粘贴
    if (!quiet) toast(`载入失败：${err.message}`, true);
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
let progressEs = null;
let appliedProgressSeq = -1;

async function selectTask(id) {
  state.selectedTaskId = id;
  state.logOffset = 0;
  state.followLog = true;
  stageTrackSig = "";
  liveItemsSig = "";
  appliedProgressSeq = -1;
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
  connectProgress(id);
  startLogPolling(id);
  renderTasks();
  renderShotsIfVisible();
}

function connectProgress(taskId) {
  if (progressEs) {
    try { progressEs.close(); } catch {}
    progressEs = null;
  }
  if (!taskId) return;
  const es = new EventSource(`/api/progress/${encodeURIComponent(taskId)}`);
  progressEs = es;
  const apply = (payload) => {
    if (state.selectedTaskId !== taskId) return;
    const prog = payload?.progress || payload;
    if (!prog || typeof prog !== "object") return;
    const seq = Number(prog.seq) || 0;
    if (seq < appliedProgressSeq) return;
    appliedProgressSeq = seq;
    const job = state.tasks.find((t) => t.id === taskId);
    if (!job) return;
    job.progress = prog;
    if (Array.isArray(prog.stages) && prog.stages.length) {
      job.stages = prog.stages.map((s) => ({
        key: s.key,
        label: s.label || STAGE_LABELS[s.key] || s.key,
        status: s.status || "pending",
        pct: s.pct,
        weight: s.weight,
      }));
    }
    if (prog.current_stage !== undefined) job.current_stage = prog.current_stage;
    if (prog.status && prog.status !== "running") {
      // 引擎终态；任务级 status 仍以 /api/state 为准，这里只刷新 UI
    }
    renderStages(job);
    $("#task-stop").disabled = job.status !== "running";
  };
  es.addEventListener("progress", (ev) => {
    try { apply(JSON.parse(ev.data)); } catch {}
  });
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.event === "progress" || data.progress) apply(data);
    } catch {}
  };
  es.addEventListener("end", () => {
    if (progressEs === es) {
      try { es.close(); } catch {}
      progressEs = null;
    }
  });
  es.onerror = () => {
    // 浏览器会自动重连；连接灯由全局 /api/events 管，这里不抢
  };
}

let stageTrackSig = "";

function renderStages(job) {
  const wrap = $("#stage-track");
  const stages = job.stages || [];
  const sig = JSON.stringify(stages.map((s) => [s.key, s.status, s.pct]));
  if (sig !== stageTrackSig) {
    stageTrackSig = sig;
    wrap.innerHTML = stages.map((s) => {
      const cls = s.status === "done" ? "done" : s.status === "running" ? "running" : s.status === "failed" ? "failed" : "";
      const icon = s.status === "done" ? "✓" : s.status === "running" ? "⏳" : s.status === "failed" ? "✕" : "·";
      const pct = typeof s.pct === "number" ? ` ${Math.min(100, s.pct).toFixed(0)}%` : "";
      return `<div class="stage-step ${cls}">${icon} ${STAGE_LABELS[s.key] || s.label || s.key}<span class="stage-pct mono">${pct}</span></div>`;
    }).join("");
  }
  renderOverall(job);
  renderLiveItems(job);
}

function renderOverall(job) {
  const fill = $("#overall-fill");
  const pctEl = $("#overall-pct");
  const stageEl = $("#overall-stage");
  const prog = job.progress || {};
  let pct = typeof prog.overall_pct === "number" ? prog.overall_pct : 0;
  if (job.status === "success") pct = 100;
  let currentLabel = "";
  if (job.status === "success") currentLabel = "完成";
  else if (job.status === "failed") currentLabel = "失败（可点「重跑失败阶段」）";
  else if (job.status === "interrupted") currentLabel = "已中断";
  else if (prog.current) currentLabel = `正在：${prog.current}`;
  else if (job.current_stage) currentLabel = `正在：${STAGE_LABELS[job.current_stage] || job.current_stage}`;
  const pctText = `${Math.min(100, pct).toFixed(1)}%`;
  const width = `${Math.min(100, pct).toFixed(1)}%`;
  const fillCls = "overall-fill " + (job.status === "success" ? "done" : job.status === "failed" ? "failed" : "");
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
  const prog = job.progress || {};
  const steps = Array.isArray(prog.steps) ? prog.steps : [];
  const currentId = prog.current_step_id;
  // 完整步骤表；无 steps 时退回旧 items
  const list = steps.length
    ? steps
    : (prog.items || []).map((i) => ({ ...i, status: i.status || "running" }));
  const sig = JSON.stringify([
    job.status,
    currentId,
    list.map((i) => [i.id, i.status, i.step, i.total, i.pct, i.eta_s]),
  ]);
  if (sig === liveItemsSig) return;
  liveItemsSig = sig;
  if (!list.length) {
    wrap.innerHTML = job.status === "running"
      ? `<div class="empty" style="padding:8px">等待步骤表…</div>`
      : "";
    return;
  }
  let lastStage = null;
  const parts = [];
  for (const i of list) {
    if (i.stage && i.stage !== lastStage) {
      lastStage = i.stage;
      parts.push(`<div class="step-group">${esc(STAGE_LABELS[i.stage] || i.stage)}</div>`);
    }
    const step = Number(i.step) || 0;
    const total = Math.max(1, Number(i.total) || 1);
    const pct = typeof i.pct === "number" ? Math.min(100, i.pct) : (total > 1 ? (100 * step) / total : 0);
    const stepText = total > 1 ? `${step}/${total}` : (pct > 0 && pct < 100 ? `${pct.toFixed(0)}%` : "");
    const eta = i.status === "running" ? fmtEta(i.eta_s) : "";
    const cls = [
      "live-item",
      i.status === "done" ? "done" : "",
      i.status === "failed" ? "failed" : "",
      i.status === "running" || i.id === currentId ? "current" : "",
      i.status === "pending" ? "pending" : "",
    ].filter(Boolean).join(" ");
    parts.push(`<div class="${cls}">
      <span class="li-label" title="${esc(i.label || "")}">${esc(i.label || i.id)}</span>
      <span class="li-step mono">${stepText}</span>
      <div class="li-bar"><div class="li-fill" style="width:${pct}%"></div></div>
      <span class="li-eta">${eta}</span>
    </div>`);
  }
  wrap.innerHTML = parts.join("");
  const cur = wrap.querySelector(".live-item.current");
  if (cur) cur.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function startLogPolling(id) {
  if (state.polling) {
    clearInterval(state.polling);
    state.polling = null;
  }
  const pull = async () => {
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
        // 日志只负责文本；进度以 /api/progress SSE 为准。
        // 断线兜底：仅当本地还没有更新的 seq 时才吃 log 里的 progress
        const incoming = data.progress;
        const incomingSeq = Number(incoming?.seq) || 0;
        if (incoming && incomingSeq > appliedProgressSeq) {
          appliedProgressSeq = incomingSeq;
          job.progress = incoming;
          if (Array.isArray(data.stages) && data.stages.length) job.stages = data.stages;
          renderStages(job);
        }
        if (data.status) job.status = data.status;
        if (data.current_stage !== undefined) job.current_stage = data.current_stage;
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

function fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function renderFinalVideo(final) {
  const wrap = $("#final-wrap");
  const badge = $("#final-badge");
  const player = $("#final-player");
  const link = $("#final-download");
  const meta = $("#final-meta");
  if (!wrap || !player) return;
  const has = !!(final && final.url);
  wrap.hidden = !has;
  if (badge) badge.hidden = !has;
  if (!has) {
    if (player.src) {
      player.removeAttribute("src");
      player.load();
    }
    return;
  }
  const sig = mediaSig(final);
  if (player.dataset.sig !== sig) {
    const paused = player.paused;
    const t = player.currentTime || 0;
    player.src = final.url;
    player.dataset.sig = sig;
    player.load();
    // 文件更新时尽量停在原位置；首次载入不自动播
    if (!paused && t > 0) {
      player.addEventListener("loadedmetadata", () => {
        try { player.currentTime = Math.min(t, player.duration || t); } catch {}
      }, { once: true });
    }
  }
  if (link) {
    link.href = final.url;
    link.setAttribute("download", "final.mp4");
  }
  if (meta) {
    meta.textContent = `${fmtBytes(final.size)} · ${final.path || "final.mp4"}`;
  }
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
  renderFinalVideo(media?.final || null);
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
    // 只认 shot: 前缀；label 可能是「s012 a2v」，按 id 抽镜头号对齐卡片
    for (const it of job.progress.items) {
      const raw = it.id || "";
      if (!raw.startsWith("shot:")) continue;
      const sid = raw.slice(5).split(/\s+/)[0];
      if (sid) progItems[sid] = it;
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
  // 镜头卡只在「正在生成 / 失败」时显示进度条；已完成的用状态文案即可，避免一排绿条
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

const VOICE_OPTIONS = {
  female: ["Serena", "Vivian"],
  male: ["Dylan", "Uncle_Fu", "Eric"],
};

function bindSbFields(root, target, { onCharacters } = {}) {
  $$(".fields [data-f]", root).forEach((input) => {
    const apply = () => {
      const f = input.dataset.f;
      if (f === "characters" && onCharacters) {
        onCharacters(input.value);
        return;
      }
      if (f === "seed") {
        const n = parseInt(input.value, 10);
        target.seed = Number.isFinite(n) ? n : 0;
        return;
      }
      target[f] = input.value.trim();
    };
    input.addEventListener("change", apply);
    input.addEventListener("input", apply);
  });
}

function renderStoryboardEditor() {
  const sb = state.sbData;
  if (!sb) return;
  if (!Array.isArray(sb.characters)) sb.characters = [];
  if (!Array.isArray(sb.shots)) sb.shots = [];
  const summary = $("#sb-summary");
  summary.textContent = `${sb.characters.length} 个角色 · ${sb.shots.length} 镜${sb.title ? ` · 《${sb.title}》` : ""}`;
  const wrap = $("#sb-shots");
  wrap.innerHTML = "";

  // 角色：解析后按字段编辑，不再塞一整段 JSON
  const charHead = document.createElement("div");
  charHead.className = "sb-section-head";
  charHead.innerHTML = `<span class="mono">characters</span><span class="hint">id / 姓名 / 性别 / 音色 / 外貌</span>`;
  wrap.appendChild(charHead);

  sb.characters.forEach((ch, idx) => {
    const gender = ch.gender === "male" ? "male" : "female";
    const voices = VOICE_OPTIONS[gender] || VOICE_OPTIONS.female;
    const voice = ch.voice && voices.includes(ch.voice) ? ch.voice : (voices[0] || "");
    ch.gender = gender;
    if (!ch.voice) ch.voice = voice;
    const row = document.createElement("div");
    row.className = "sb-shot sb-char";
    row.innerHTML = `<div class="shot-meta">
        <span class="mono">${esc(ch.name || ch.id || `角色${idx + 1}`)}</span>
        <button type="button" class="ghost sb-del">删除</button>
      </div>
      <div class="fields">
        <label>id</label><input data-f="id" value="${esc(ch.id || "")}" placeholder="chenyuan">
        <label>name</label><input data-f="name" value="${esc(ch.name || "")}" placeholder="陈媛">
        <label>gender</label>
        <select data-f="gender">
          <option value="female"${gender === "female" ? " selected" : ""}>female</option>
          <option value="male"${gender === "male" ? " selected" : ""}>male</option>
        </select>
        <label>voice</label>
        <select data-f="voice">
          ${voices.map((v) => `<option value="${esc(v)}"${(ch.voice || voice) === v ? " selected" : ""}>${esc(v)}</option>`).join("")}
        </select>
        <label>appearance</label><textarea data-f="appearance" rows="2" placeholder="只写原文可见外貌">${esc(ch.appearance || "")}</textarea>
        <label>seed</label><input data-f="seed" type="number" value="${esc(ch.seed ?? "")}">
      </div>`;
    $(".sb-del", row).addEventListener("click", () => {
      sb.characters.splice(idx, 1);
      renderStoryboardEditor();
    });
    bindSbFields(row, ch);
    $$("select[data-f]", row).forEach((sel) => {
      sel.addEventListener("change", () => {
        ch[sel.dataset.f] = sel.value;
        if (sel.dataset.f === "gender") renderStoryboardEditor(); // 换性别 → 刷新音色选项
      });
    });
    wrap.appendChild(row);
  });

  const addChar = document.createElement("button");
  addChar.type = "button";
  addChar.textContent = "＋ 添加角色";
  addChar.className = "ghost";
  addChar.addEventListener("click", () => {
    sb.characters.push({
      id: `char${sb.characters.length + 1}`,
      name: "",
      gender: "female",
      voice: "Serena",
      appearance: "",
      seed: Math.floor(Math.random() * 0xffffffff),
    });
    renderStoryboardEditor();
  });
  wrap.appendChild(addChar);

  const shotHead = document.createElement("div");
  shotHead.className = "sb-section-head";
  shotHead.innerHTML = `<span class="mono">shots</span><span class="hint">镜头列表</span>`;
  wrap.appendChild(shotHead);

  sb.shots.forEach((shot, idx) => {
    const row = document.createElement("div");
    row.className = "sb-shot";
    row.innerHTML = `<div class="shot-meta">
        <span class="mono">#${idx + 1} ${esc(shot.id || "")}</span>
        <button type="button" class="ghost sb-del">删除</button>
      </div>
      <div class="fields">
        <label>id</label><input data-f="id" value="${esc(shot.id || "")}">
        <label>location</label><input data-f="location" value="${esc(shot.location || "")}">
        <label>characters</label><input data-f="characters" value="${esc((shot.characters || []).join(","))}" placeholder="角色 id，逗号分隔">
        <label>image_prompt</label><textarea data-f="image_prompt" rows="2">${esc(shot.image_prompt || "")}</textarea>
        <label>motion_prompt</label><textarea data-f="motion_prompt" rows="2">${esc(shot.motion_prompt || "")}</textarea>
        <label>speaker</label><input data-f="speaker" value="${esc(shot.speaker || "")}">
        <label>dialogue</label><textarea data-f="dialogue" rows="2">${esc(shot.dialogue || "")}</textarea>
        <label>emotion</label><input data-f="emotion" value="${esc(shot.emotion || "")}">
      </div>`;
    $(".sb-del", row).addEventListener("click", () => {
      sb.shots.splice(idx, 1);
      renderStoryboardEditor();
    });
    bindSbFields(row, shot, {
      onCharacters: (value) => {
        shot.characters = value.split(/[,，\s]+/).filter(Boolean);
      },
    });
    wrap.appendChild(row);
  });

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.textContent = "＋ 添加镜头";
  addBtn.className = "ghost";
  addBtn.addEventListener("click", () => {
    sb.shots.push({
      id: `s${String((sb.shots.length || 0) + 1).padStart(3, "0")}`,
      characters: [], location: "", image_prompt: "", motion_prompt: "",
      speaker: "", dialogue: "", emotion: "calm",
    });
    renderStoryboardEditor();
  });
  wrap.appendChild(addBtn);
}

async function saveStoryboard() {
  if (!state.sbData) return;
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
let sseRefreshTimer = null;

function connectEvents() {
  // 全局事件只刷新任务列表元数据；细进度走 /api/progress/<id>
  const es = new EventSource("/api/events");
  es.onopen = () => setConn(true);
  es.onerror = () => setConn(false);
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.event !== "change") return;
      if (sseRefreshTimer) return;
      sseRefreshTimer = setTimeout(() => {
        sseRefreshTimer = null;
        refreshState();
      }, 1500);
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
  // 改路径后失焦/回车即自动载入，不再需要「载入」按钮
  $("#novel-path").addEventListener("change", () => loadNovel({ quiet: false }));
  $("#novel-path").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); loadNovel({ quiet: false }); }
  });
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
connectEvents();
setInterval(refreshState, 4000);

(async function boot() {
  await refreshState();
  // 有运行中的项目优先打开；否则恢复上次打开的项目
  const runningProj = (state.projects || []).find((p) => (p.tasks || {}).running > 0);
  let last = "";
  try { last = localStorage.getItem("nv_last_project") || ""; } catch {}
  const target = runningProj || (state.projects || []).find((p) => p.name === last);
  if (target) {
    await openProject(target.name, target.path);
  }
})();
