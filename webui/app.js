/* 小说 → 竖屏短剧 控制台（无构建、无依赖）
   打开网页第一件事：选项目；之后所有东西都属于这个项目。 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ALL_STAGES = ["storyboard", "screenplay", "images", "audio", "videos", "music", "assemble"];
const STAGE_LABELS = {
  storyboard: "生成分镜",
  screenplay: "校验分镜",
  images: "静帧生成",
  audio: "对白配音",
  videos: "视频生成",
  music: "配乐生成",
  assemble: "合成混音",
};

const state = {
  project: "", // 当前打开的项目名（~/AI短视频项目/<片名> 的文件夹名）
  tasks: [], // 历史任务（点一次启动算一条）
  queue: { items: [], updated_at: 0 }, // 左边待跑队列
  projects: [],
  novels: [],
  storyboards: [],
  selectedTaskId: null,
  logOffset: 0,
  followLog: true,
  selectedStages: new Set(ALL_STAGES),
  currentOut: "",
  currentStoryboard: "",
  shotSb: null, // 镜头网格里编辑的分镜（与磁盘同步；脏时不被自动刷新覆盖）
  shotSbPath: "",
  shotSbDirty: false,
  shotSbExists: false, // 磁盘上是否有 storyboard.json
  shotSbMtime: null,
  portraits: {}, // id → {url, mtime}，镜头卡迷你头像用
  polling: null,
  fps: 24,
  setup: { ready: false, status: "unknown", pct: 0, message: "", items: [] },
  setupTimer: null,
};

// ---------------------------------------------------------------- utilities
function projectNovelPath(out = state.currentOut) {
  return out ? `${out.replace(/\/+$/, "")}/novel.txt` : "";
}

function projectStoryboardPath(out = state.currentOut) {
  return out ? `${out.replace(/\/+$/, "")}/storyboard.json` : "";
}

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
  if (!res.ok) {
    const err = new Error(data.error || data.strict_error || `HTTP ${res.status}`);
    err.data = data;
    err.status = res.status;
    throw err;
  }
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

/** 任务完成：只走前端系统通知声（不调用后台播音） */
const jobStatusSeen = new Map();

function playTaskDoneChime() {
  try {
    if (!("Notification" in window)) return;
    const title = "任务完成";
    const body = "短剧流水线已跑完";
    const show = () => {
      const n = new Notification(title, { body, silent: false });
      setTimeout(() => n.close(), 4000);
    };
    if (Notification.permission === "granted") show();
    else if (Notification.permission !== "denied") {
      Notification.requestPermission().then((p) => { if (p === "granted") show(); });
    }
  } catch (_) {}
}

function jobHasStage(job, stage) {
  for (const s of job?.stages || []) {
    const key = typeof s === "string" ? s : s?.key;
    if (key === stage) return true;
  }
  return false;
}

async function reloadStoryboardAfterGen() {
  const out = state.currentOut;
  if (!out) return;
  // 管线刚写出 storyboard.json：必须以磁盘为准刷新。
  // 空项目粘贴小说会 markShotSbDirty，若因此跳过，界面会一直停在「还没有分镜」。
  try {
    const novelKeep = novelValue();
    state.shotSb = null;
    await loadShotSb(out, { force: true });
    const sb = state.shotSb || {};
    if (!isTitleEditing()) setTitleDisplay(String(sb.title || "").trim());
    if (!isStyleEditing()) setStyleDisplay(String(sb.style || "").trim());
    if (!isMusicEditing()) setMusicDisplay(String(sb.music_prompt || "").trim());
    setMusicSeedDisplay(sb.music_seed);
    clearShotSbDirty();
    // 正文可能已在「从小说生成」时落盘；保留当前窗格文案，避免闪成空
    if (novelKeep && !isNovelEditing()) setNovelDisplay(novelKeep);
    await renderShots();
    toast("分镜已更新");
  } catch (err) {
    toast(`刷新分镜失败：${err.message || err}`, true);
  }
}

function noteJobStatuses(tasks) {
  for (const job of tasks || []) {
    if (!job?.id) continue;
    const next = job.status || "";
    const prev = jobStatusSeen.get(job.id);
    jobStatusSeen.set(job.id, next);
    if (prev === "running" && next === "success") {
      playTaskDoneChime();
      if (jobHasStage(job, "storyboard")) reloadStoryboardAfterGen();
    }
  }
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
      // 合并历史任务：保留本地已通过进度 SSE 拿到的更新 seq
      const prevById = Object.create(null);
      for (const t of state.tasks || []) prevById[t.id] = t;
      state.tasks = (data.tasks || []).map((t) => {
        const old = prevById[t.id];
        let merged = t;
        if (old?.progress) {
          const oldSeq = Number(old.progress.seq) || 0;
          const newSeq = Number(t.progress?.seq) || 0;
          if (oldSeq > newSeq && t.status === "running") {
            merged = { ...t, progress: old.progress, stages: old.stages?.length ? old.stages : t.stages };
          }
        }
        return normalizeJobStages(merged);
      });
      noteJobStatuses(state.tasks);
      if (data.queue) state.queue = data.queue;
    }
    setConn(true);
    renderProjects();
    renderDatalists();
    renderRunQueue();
    renderTasks();
    if (state.selectedTaskId) {
      const job = state.tasks.find((t) => t.id === state.selectedTaskId);
      if (job) {
        if (job.progress && (Number(job.progress.seq) || 0) >= appliedProgressSeq) {
          appliedProgressSeq = Number(job.progress.seq) || appliedProgressSeq;
          renderStages(job);
        }
      }
    }
    syncRunControls();
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

/** 刷新后若有生成任务则锁定进对应项目（勿嵌进 refreshState，以免 openProject 二次刷新被跳过） */
async function refreshStateAndLock() {
  await refreshState();
  await ensureLockedOnGeneratingProject();
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
    const t = p.tasks || { done: 0, running: 0, failed: 0, queued: 0, total: 0 };
    const badges = [
      p.novel ? "📖 小说" : "",
      p.has_storyboard ? "📋 分镜" : "",
      p.has_final ? "🎬 成片" : "",
    ].filter(Boolean).join("　");
    const stat = [
      t.running ? `<span class="dot run">${t.running} 运行中…</span>` : "",
      t.queued ? `<span class="dot">${t.queued} 待跑</span>` : "",
      t.done ? `<span class="dot done">${t.done} 完成</span>` : "",
      t.failed ? `<span class="dot fail">${t.failed} 失败</span>` : "",
      !t.total ? `<span class="dot">还没跑过任务</span>` : "",
    ].join("");
    const active = p.name === state.project ? " active" : "";
    return `<div class="project-card${active}" data-name="${esc(p.name)}" data-path="${esc(p.path)}">
      <div class="p-name">${esc(p.name)}${p.legacy ? ' <span class="hint">（旧布局）</span>' : ""}</div>
      <div class="p-meta hint mono">${esc(p.display_path || p.path)}/</div>
      <div class="p-badges">${badges || '<span class="hint">空项目</span>'}</div>
      <div class="p-stat">${stat}</div>
    </div>`;
  }).join("");
  $$(".project-card", wrap).forEach((el) => {
    el.addEventListener("click", () => openProject(el.dataset.name, el.dataset.path));
  });
}

function renderDatalists() {
  // 路径已固定为项目默认，不再展示文件选择器
}

function setupBusy() {
  const st = state.setup?.status;
  return st === "installing" || st === "checking" || (st === "need_install" && !state.setup?.ready);
}

function inProjectWorkspace() {
  const picker = $("#project-picker");
  return !!state.project || (picker && picker.hidden);
}

/** 依赖未齐时强制回到项目选择页；保留上次项目名，装完可自动再进 */
function leaveProjectForSetup(reason) {
  const picker = $("#project-picker");
  if (!inProjectWorkspace()) {
    if (picker) picker.hidden = false;
    return false;
  }
  state.project = "";
  state.selectedTaskId = null;
  state.shotSb = null;
  state.shotSbPath = "";
  state.shotSbExists = false;
  state.shotSbMtime = null;
  clearShotSbDirty();
  if (state.polling) { clearInterval(state.polling); state.polling = null; }
  if (typeof shotsTimer !== "undefined" && shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  $("#shots-card").hidden = true;
  $("#produce-card").hidden = true;
  if (picker) picker.hidden = false;
  projectsSig = "";
  if (reason) pickerMsg(reason, true);
  refreshState();
  return true;
}

function renderSetupBanner(st) {
  state.setup = st || state.setup || {};
  const banner = $("#setup-banner");
  const picker = $("#project-picker");
  if (!banner || !picker) return;
  const ready = !!state.setup.ready;
  const failed = state.setup.status === "failed";
  const busy = setupBusy();
  if (!ready && inProjectWorkspace()) {
    leaveProjectForSetup(
      failed ? "依赖安装失败，已回到项目列表" : "依赖还在安装，已退出项目页"
    );
  }
  banner.hidden = ready && !failed;
  if (ready && !failed) {
    // 刚就绪时短暂显示，再藏
    if (banner.dataset.justReady === "1") {
      banner.hidden = false;
      banner.classList.add("ready");
      banner.classList.remove("failed");
      setTimeout(() => {
        if (state.setup.ready) banner.hidden = true;
        banner.dataset.justReady = "";
      }, 1800);
    }
  } else {
    banner.dataset.justReady = "";
    banner.classList.toggle("failed", failed);
    banner.classList.toggle("ready", false);
  }
  picker.classList.toggle("setup-busy", busy || failed);
  const pct = Math.max(0, Math.min(100, Number(state.setup.pct) || 0));
  const title = $("#setup-title");
  const pctEl = $("#setup-pct");
  const fill = $("#setup-fill");
  const msg = $("#setup-msg");
  const actions = $("#setup-actions");
  if (title) {
    title.textContent = failed
      ? "依赖安装失败"
      : ready
        ? "依赖与模型已就绪"
        : "正在安装依赖与模型…";
  }
  if (pctEl) pctEl.textContent = `${pct.toFixed(0)}%`;
  if (fill) fill.style.width = `${pct}%`;
  if (msg) {
    const err = state.setup.error ? `：${state.setup.error}` : "";
    msg.textContent = failed
      ? `安装失败${err}。可点重试，或在终端跑 ./scripts/setup.sh`
      : (state.setup.message || "首次打开会自动安装 Python 依赖并下载模型（含分镜语言模型），请保持网络畅通。");
  }
  const list = $("#setup-items");
  const items = Array.isArray(state.setup.items) ? state.setup.items : [];
  if (list) {
    if (!items.length) {
      list.hidden = true;
      list.innerHTML = "";
    } else {
      list.hidden = false;
      list.innerHTML = items.map((it) => {
        const mark = it.ok ? "✓" : "·";
        return `<li class="${it.ok ? "ok" : "bad"}">${mark} ${esc(it.label)}</li>`;
      }).join("");
    }
  }
  if (actions) actions.hidden = !failed;
  syncSetupEntry();
}

/** 项目列表上的安装入口：检查显示已就绪时也留着，判断失误可以手动再装。 */
function syncSetupEntry() {
  const btn = $("#setup-install");
  const hint = $("#setup-entry-hint");
  if (!btn) return;
  const failed = state.setup?.status === "failed";
  const busy = setupBusy();
  const ready = !!state.setup?.ready && !failed && !busy;
  btn.disabled = busy;
  btn.textContent = failed ? "重试安装" : busy ? "正在安装…" : "安装依赖";
  if (!hint) return;
  if (failed) hint.textContent = "安装失败，可再试一次。";
  else if (busy) hint.textContent = "正在安装，请稍候。";
  else if (ready) hint.textContent = "当前检查为已就绪。若仍缺模型或运行报错，点这里重新安装。";
  else hint.textContent = "即使检查显示已就绪，也可以再装一次，用来补上下载不完整的模型。";
}

function projectFromQuery() {
  try {
    return (new URLSearchParams(location.search).get("project") || "").trim();
  } catch {
    return "";
  }
}

function findRunningProject() {
  return (state.projects || []).find((p) => (p.tasks || {}).running > 0) || null;
}

/** 生成中锁定到该项目；否则可按 ?project= / 上次项目进入 */
let lockProjectInFlight = false;

async function ensureLockedOnGeneratingProject() {
  if (lockProjectInFlight) return !!state.project && !!findRunningProject();
  if (setupBusy() || state.setup?.status === "failed" || !state.setup?.ready) return false;
  const running = findRunningProject();
  if (!running) return false;
  if (state.project === running.name) {
    const runTask = (state.tasks || []).find((t) => t.status === "running");
    if (runTask && state.selectedTaskId !== runTask.id) selectTask(runTask.id);
    return true;
  }
  lockProjectInFlight = true;
  try {
    await openProject(running.name, running.path);
    return state.project === running.name;
  } finally {
    lockProjectInFlight = false;
  }
}

async function maybeOpenLastProjectAfterSetup() {
  if (!state.setup?.ready || state.project) return;
  if (!$("#project-picker") || $("#project-picker").hidden) return;
  await refreshState();
  const locked = await ensureLockedOnGeneratingProject();
  if (locked || state.project) return;
  const fromQuery = projectFromQuery();
  let last = "";
  try { last = localStorage.getItem("nv_last_project") || ""; } catch {}
  const target =
    (fromQuery && (state.projects || []).find((p) => p.name === fromQuery)) ||
    (state.projects || []).find((p) => p.name === last);
  if (target) await openProject(target.name, target.path);
}

async function refreshSetup() {
  try {
    const res = await fetch("/api/setup/status");
    const st = await res.json().catch(() => ({}));
    // 旧服务进程还没有新路由时会 404；本机依赖其实已装，不能当成「安装失败」
    if (res.status === 404 || (st && st.error === "not found")) {
      const bypass = {
        ready: true,
        status: "ready",
        pct: 100,
        message: "依赖检查接口未就绪（请重启控制台后再验）",
        error: "",
        items: [],
      };
      renderSetupBanner(bypass);
      stopSetupPolling();
      return bypass;
    }
    if (!res.ok) {
      throw Object.assign(new Error(st.error || `HTTP ${res.status}`), { data: st, status: res.status });
    }
    const wasReady = !!state.setup?.ready;
    renderSetupBanner(st);
    if (st.ready && !wasReady) {
      const banner = $("#setup-banner");
      if (banner) banner.dataset.justReady = "1";
      renderSetupBanner(st);
      maybeOpenLastProjectAfterSetup();
    }
    if (st.ready || st.status === "failed") {
      stopSetupPolling();
    } else {
      startSetupPolling();
    }
    return st;
  } catch (err) {
    // 网络闪断等：不锁死项目页；下次轮询再试
    console.warn("setup status", err);
    startSetupPolling();
    return state.setup;
  }
}

function startSetupPolling() {
  if (state.setupTimer) return;
  state.setupTimer = setInterval(() => { refreshSetup(); }, 1500);
}

function stopSetupPolling() {
  if (state.setupTimer) {
    clearInterval(state.setupTimer);
    state.setupTimer = null;
  }
}

async function retrySetup() {
  try {
    const st = await api("/api/setup/ensure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    renderSetupBanner(st);
    startSetupPolling();
  } catch (err) {
    pickerMsg(`重试失败：${err.message}`, true);
  }
}

async function createProject() {
  if (setupBusy() || state.setup?.status === "failed") {
    pickerMsg(state.setup?.status === "failed" ? "请先重试安装依赖" : "依赖还在安装，请稍候", true);
    return;
  }
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
  if (setupBusy() || state.setup?.status === "failed") {
    leaveProjectForSetup(
      state.setup?.status === "failed" ? "请先重试安装依赖" : "依赖还在安装，请稍候再进项目"
    );
    return;
  }
  state.project = name;
  state.currentOut = path || "";
  if (!state.currentOut) {
    toast("缺少项目路径");
    return;
  }
  state.currentStoryboard = projectStoryboardPath(state.currentOut);
  state.selectedTaskId = null;
  state.logOffset = 0;
  stageTrackSig = "";
  liveItemsSig = "";
  tasksSig = "";
  try { localStorage.setItem("nv_last_project", name); } catch {}
  $("#log").textContent = "";
  $("#project-picker").hidden = true;
  $("#shots-card").hidden = false;
  $("#produce-card").hidden = false;
  $("#current-project-name").textContent = name || state.currentOut;
  if (!isTitleEditing()) setTitleDisplay("");
  if (!isNovelEditing()) setNovelDisplay("");
  if (!isStyleEditing()) setStyleDisplay("");
  if (!isMusicEditing()) setMusicDisplay("");
  setMusicSeedDisplay("");
  clearShotCards();
  shotsLastOut = "";
  state.shotSb = null;
  state.shotSbPath = "";
  state.shotSbExists = false;
  state.shotSbMtime = null;
  clearShotSbDirty();
  const castEl = $("#final-cast");
  if (castEl) {
    delete castEl.dataset.sig;
    castEl.innerHTML = "";
  }
  clearShotsMsg();
  try {
    await api("/api/project/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch {}
  await refreshState();
  await loadNovel({ quiet: true });
  await loadProjectTitle({ fallback: name });
  await renderShots();
  scheduleAutoValidate();
  scheduleShotsAutoRefresh();
  // 只自动挂「在跑」；历史/中断不占详情，避免重启后像还有任务在跑、启动钮却是灰的
  const running = autoSelectRunning ? state.tasks.find((t) => t.status === "running") : null;
  if (running) selectTask(running.id);
  else showTaskDetailIdle();
}

function backToProjects() {
  const running = findRunningProject();
  if (running) {
    toast(`「${running.name}」正在生成，已锁定在项目页`);
    if (state.project !== running.name) ensureLockedOnGeneratingProject();
    return;
  }
  state.project = "";
  state.selectedTaskId = null;
  state.shotSb = null;
  state.shotSbPath = "";
  state.shotSbExists = false;
  state.shotSbMtime = null;
  clearShotSbDirty();
  try { localStorage.removeItem("nv_last_project"); } catch {}
  if (state.polling) { clearInterval(state.polling); state.polling = null; }
  if (shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  $("#shots-card").hidden = true;
  $("#produce-card").hidden = true;
  $("#project-picker").hidden = false;
  projectsSig = "";
  refreshState();
}

// ---------------------------------------------------------------- start form
function renderStagePicks() {
  const wrap = $("#stage-picks");
  if (!wrap) return;
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

function titleValue() {
  return ($("#title")?.value || "").trim();
}

function novelValue() {
  return ($("#novel-text")?.value || "").trim();
}

function setTitleDisplay(text) {
  const v = String(text || "").trim();
  const input = $("#title");
  const display = $("#title-display");
  if (input) input.value = v;
  if (display) display.textContent = v || "未命名短剧";
}

function setNovelDisplay(text) {
  const v = String(text || "");
  const input = $("#novel-text");
  const display = $("#novel-display");
  if (input) input.value = v;
  if (display) {
    const shown = v.trim();
    display.textContent = shown || "暂无正文 · 点击编辑";
    display.classList.toggle("empty", !shown);
  }
}

function isTitleEditing() {
  return !!$("#title") && !$("#title").hidden;
}

function isNovelEditing() {
  return !!$("#novel-text") && !$("#novel-text").hidden;
}

function beginEditTitle() {
  const input = $("#title");
  const display = $("#title-display");
  if (!input || !display || isTitleEditing()) return;
  input.hidden = false;
  display.hidden = true;
  input.focus();
  input.select();
}

async function commitTitle({ cancel = false } = {}) {
  const input = $("#title");
  const display = $("#title-display");
  if (!input || !display || !isTitleEditing()) return;
  if (cancel) {
    input.value = display.textContent === "未命名短剧" ? "" : display.textContent;
  }
  const next = input.value.trim();
  const prev = display.textContent === "未命名短剧" ? "" : display.textContent.trim();
  input.hidden = true;
  display.hidden = false;
  setTitleDisplay(next);
  if (cancel || next === prev) return;
  if (state.shotSb) state.shotSb.title = next;
  markShotSbDirty();
}

function beginEditNovel() {
  const input = $("#novel-text");
  const display = $("#novel-display");
  if (!input || !display || isNovelEditing()) return;
  input.hidden = false;
  display.hidden = true;
  input.focus();
  // 光标放到末尾，方便续写
  const len = input.value.length;
  try { input.setSelectionRange(len, len); } catch {}
}

async function commitNovel({ cancel = false } = {}) {
  const input = $("#novel-text");
  const display = $("#novel-display");
  if (!input || !display || !isNovelEditing()) return;
  if (cancel) {
    input.value = display.classList.contains("empty") ? "" : display.textContent;
  }
  const next = input.value;
  const prev = display.classList.contains("empty") ? "" : display.textContent;
  input.hidden = true;
  display.hidden = false;
  setNovelDisplay(next);
  if (cancel || next === prev) return;
  markShotSbDirty();
}

async function loadNovel({ quiet = false } = {}) {
  const path = projectNovelPath();
  if (!path) {
    setNovelDisplay("");
    if (!quiet) toast("先选项目", true);
    return;
  }
  // 窗格内有未保存编辑时，不要用磁盘正文盖掉
  if (quiet && (state.shotSbDirty || isNovelEditing())) return;
  try {
    const data = await api("/api/file/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    setNovelDisplay(data.content || "");
    if (!quiet) toast(`已载入 ${path}（${data.content.length} 字）`);
  } catch (err) {
    setNovelDisplay("");
    // 打开项目时文件还不存在：安静失败，留给用户粘贴
    if (!quiet) toast(`载入失败：${err.message}`, true);
  }
}

async function loadProjectTitle({ fallback = "" } = {}) {
  const path = projectStoryboardPath();
  let title = "";
  if (path) {
    try {
      const data = await api("/api/storyboard/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      title = String(data.storyboard?.title || "").trim();
    } catch {}
  }
  setTitleDisplay(title || fallback || state.project || "");
}

function syncTitleToShotSb() {
  const title = titleValue();
  if (!state.shotSb) return;
  if ((state.shotSb.title || "") === title) return;
  state.shotSb.title = title;
  markShotSbDirty();
}

function styleValue() {
  return ($("#style-text")?.value || "").trim();
}

function musicValue() {
  return ($("#music-text")?.value || "").trim();
}

function setStyleDisplay(text) {
  const v = String(text || "").trim();
  const input = $("#style-text");
  const display = $("#style-display");
  if (input) input.value = v;
  if (display) {
    display.textContent = v || "点击填写画面风格";
    display.classList.toggle("empty", !v);
  }
}

function setMusicDisplay(text) {
  const v = String(text || "").trim();
  const input = $("#music-text");
  const display = $("#music-display");
  if (input) input.value = v;
  if (display) {
    display.textContent = v || "点击填写 music_prompt";
    display.classList.toggle("empty", !v);
  }
}

function setMusicSeedDisplay(seed) {
  const el = $("#music-seed-display");
  const input = $("#music-seed-input");
  if (!el) return;
  const has = seed != null && seed !== "";
  const text = has ? String(seed) : "—";
  el.textContent = text;
  el.classList.toggle("empty", !has);
  if (input && input.hidden) input.value = has ? String(seed) : "";
}

function isMusicSeedEditing() {
  const input = $("#music-seed-input");
  return !!(input && !input.hidden);
}

function beginEditMusicSeed() {
  const display = $("#music-seed-display");
  const input = $("#music-seed-input");
  if (!display || !input || isMusicSeedEditing()) return;
  input.value = display.classList.contains("empty") ? "" : display.textContent.trim();
  display.hidden = true;
  input.hidden = false;
  input.focus();
  try { input.select(); } catch {}
}

function commitMusicSeed({ cancel = false } = {}) {
  const display = $("#music-seed-display");
  const input = $("#music-seed-input");
  if (!display || !input || !isMusicSeedEditing()) return;
  if (cancel) {
    input.value = state.shotSb?.music_seed != null ? String(state.shotSb.music_seed) : "";
  }
  const raw = String(input.value || "").trim();
  const n = raw === "" ? null : parseInt(raw, 10);
  input.hidden = true;
  display.hidden = false;
  if (!state.shotSb) {
    setMusicSeedDisplay(n);
    return;
  }
  const prev = state.shotSb.music_seed == null ? null : Number(state.shotSb.music_seed);
  const next = Number.isFinite(n) ? (n >>> 0) : null;
  if (next == null) {
    // 空值：按 prompt/title 重算稳定 seed，保证推理始终有数
    stableSeedAsync("bgm", state.shotSb.music_prompt || "", state.shotSb.title || "").then((s) => {
      state.shotSb.music_seed = s;
      setMusicSeedDisplay(s);
      markShotSbDirty();
    });
    return;
  }
  setMusicSeedDisplay(next);
  if (prev !== next) {
    state.shotSb.music_seed = next;
    markShotSbDirty();
  }
}

function rerollMusicSeed() {
  if (!state.shotSb) {
    toast("请先打开项目", true);
    return;
  }
  if (isMusicSeedEditing()) commitMusicSeed({ cancel: true });
  state.shotSb.music_seed = randomSeed();
  setMusicSeedDisplay(state.shotSb.music_seed);
  markShotSbDirty();
}

/** 一键重掷：music_seed + 全部角色 seed + 全部镜头 seed */
async function rerollAllSeeds() {
  const out = state.currentOut;
  if (!out) {
    toast("请先打开项目", true);
    return;
  }
  const sb = await loadShotSb(out);
  if (isMusicSeedEditing()) commitMusicSeed({ cancel: true });

  sb.music_seed = randomSeed();
  setMusicSeedDisplay(sb.music_seed);

  let charN = 0;
  for (const ch of sb.characters || []) {
    ch.seed = randomSeed();
    charN += 1;
  }
  let shotN = 0;
  for (const shot of sb.shots || []) {
    shot.seed = randomSeed();
    shotN += 1;
  }

  // 刷新角色卡与镜头卡上的 seed 展示
  const castEl = $("#final-cast");
  if (castEl) delete castEl.dataset.sig;
  renderFinalMeta({
    characters: sb.characters,
    style: sb.style,
    music_prompt: sb.music_prompt,
    music_seed: sb.music_seed,
    title: sb.title,
    out,
    shots: sb.shots,
  });
  for (const [, card] of shotCards) {
    if (!card._shot) continue;
    const wrap = $(".shot-iedit[data-f=seed]", card);
    if (wrap) fillShotInlineField(wrap, card._shot);
    card.classList.add("dirty");
  }

  markShotSbDirty();
  toast(`已重掷全部 seed（配乐 + ${charN} 角色 + ${shotN} 镜）`);
}

function isStyleEditing() {
  return !!$("#style-text") && !$("#style-text").hidden;
}

function isMusicEditing() {
  return !!$("#music-text") && !$("#music-text").hidden;
}

function beginEditStyle() {
  const input = $("#style-text");
  const display = $("#style-display");
  if (!input || !display || isStyleEditing()) return;
  input.hidden = false;
  display.hidden = true;
  input.focus();
  const len = input.value.length;
  try { input.setSelectionRange(len, len); } catch {}
}

function beginEditMusic() {
  const input = $("#music-text");
  const display = $("#music-display");
  if (!input || !display || isMusicEditing()) return;
  input.hidden = false;
  display.hidden = true;
  input.focus();
  const len = input.value.length;
  try { input.setSelectionRange(len, len); } catch {}
}

async function commitStyle({ cancel = false } = {}) {
  const input = $("#style-text");
  const display = $("#style-display");
  if (!input || !display || !isStyleEditing()) return;
  if (cancel) input.value = display.classList.contains("empty") ? "" : display.textContent;
  const next = input.value.trim();
  const prev = display.classList.contains("empty") ? "" : display.textContent.trim();
  input.hidden = true;
  display.hidden = false;
  setStyleDisplay(next);
  if (cancel || next === prev) return;
  if (state.shotSb) state.shotSb.style = next;
  markShotSbDirty();
}

async function commitMusic({ cancel = false } = {}) {
  const input = $("#music-text");
  const display = $("#music-display");
  if (!input || !display || !isMusicEditing()) return;
  if (cancel) input.value = display.classList.contains("empty") ? "" : display.textContent;
  const next = input.value.trim();
  const prev = display.classList.contains("empty") ? "" : display.textContent.trim();
  input.hidden = true;
  display.hidden = false;
  setMusicDisplay(next);
  if (cancel || next === prev) return;
  if (state.shotSb) state.shotSb.music_prompt = next;
  markShotSbDirty();
}

function syncMetaToShotSb() {
  if (!state.shotSb) return;
  syncTitleToShotSb();
  const style = styleValue();
  const music = musicValue();
  if ((state.shotSb.style || "") !== style) {
    state.shotSb.style = style;
    markShotSbDirty();
  }
  if ((state.shotSb.music_prompt || "") !== music) {
    state.shotSb.music_prompt = music;
    markShotSbDirty();
  }
}

function hasUnsavedShotsEdits() {
  return !!(
    state.shotSbDirty
    || isTitleEditing()
    || isNovelEditing()
    || isStyleEditing()
    || isMusicEditing()
    || isMusicSeedEditing()
  );
}

/** 网页暂存快照：给启动/单跑用；不写项目源，脏标记保持到用户点保存 */
async function webDraftForPipeline() {
  await flushOpenEditsInShotsCard();
  if (!state.currentOut) {
    toast("请先打开项目", true);
    return null;
  }
  if (!state.shotSb) await loadShotSb(state.currentOut);
  const draft = { novel_text: novelValue() };
  // 无分镜 / 空壳不传 storyboard_data，避免 materialize 写出假 storyboard.json
  // 把「成片」依赖算成无前置
  const missing = !!(state.shotSb?._missing || state.shotSbExists === false);
  if (!missing && state.shotSb) {
    const snap = snapshotShotSbFromUi();
    const shots = Array.isArray(snap.shots) ? snap.shots : [];
    if (shots.some((s) => s && s.id) || state.shotSbDirty) {
      draft.storyboard_data = snap;
    }
  }
  return draft;
}

function queueItems() {
  return state.queue?.items || [];
}

function queuedCount() {
  return queueItems().length;
}

function runningTask() {
  return (state.tasks || []).find((t) => t.status === "running") || null;
}

function syncRunControls() {
  const n = queuedCount();
  const running = runningTask();
  const countEl = $("#queue-count");
  if (countEl) countEl.textContent = n ? `（${n}）` : "";
  const btn = $("#start-btn");
  if (btn) {
    btn.disabled = !!running || n === 0;
    btn.title = running
      ? "有任务在跑，先停止"
      : n === 0
        ? "先点上方 ↻ 加入待跑队列"
        : `启动队列里 ${n} 项`;
  }
  const stopBtn = $("#task-stop");
  if (stopBtn) {
    stopBtn.disabled = !running;
    stopBtn.title = running ? "停止后台进程；待跑队列保留" : "当前没有在跑任务";
  }
  const clearBtn = $("#queue-clear");
  if (clearBtn) clearBtn.disabled = n === 0 || !!running;
  const hint = $("#run-hint") || $(".run-hint");
  if (hint) {
    if (running) {
      hint.textContent = `正在跑：${running.name || "管线"} · 可随时停止；停止后队列保留，进度归零`;
    } else if (n > 0) {
      hint.textContent = `待跑 ${n} 项 · 点启动开跑；上方 ↻ 可继续添加`;
    } else {
      hint.textContent = "在上方点 ↻ 打开对话框，确认后加入下方待跑队列；队列非空即可启动。";
    }
  }
}

let runQueueSig = "";

function renderRunQueue() {
  const wrap = $("#run-queue");
  if (!wrap) return;
  syncRunControls();
  const items = queueItems();
  const sig = JSON.stringify(items.map((it) => [it.key, it.force, it.status, it.removable]));
  if (sig === runQueueSig) return;
  runQueueSig = sig;
  if (!items.length) {
    wrap.innerHTML = `<div class="empty">还没有待跑项。点上方 ↻ 添加。</div>`;
    return;
  }
  wrap.innerHTML = items.map((it) => {
    const force = it.force ? "重做" : "补缺";
    const st = it.status === "running" ? "跑着" : it.status === "done" ? "完成" : "待跑";
    const dis = it.removable === false ? " disabled" : "";
    const title = it.block_reason || (it.removable === false ? "被其它项依赖，不能删" : "从队列移除");
    return `<div class="queue-item" data-key="${esc(it.key)}">
      <span class="qi-label">${esc(it.label || it.key)}</span>
      <span class="qi-meta">${force} · ${st}</span>
      <button type="button" class="ghost qi-del" data-key="${esc(it.key)}" title="${esc(title)}"${dis}>删除</button>
    </div>`;
  }).join("");
  $$(".qi-del", wrap).forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (btn.disabled) return;
      removeQueueItem(btn.dataset.key);
    });
  });
}

async function removeQueueItem(key) {
  if (!state.currentOut || !key) return;
  try {
    const res = await api("/api/queue/remove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out: state.currentOut, key }),
    });
    state.queue = res.queue || state.queue;
    runQueueSig = "";
    renderRunQueue();
    toast("已从待跑队列移除");
  } catch (err) {
    toast(`无法删除：${err.message}`, true);
  }
}

async function clearRunQueue() {
  if (!state.currentOut) return;
  if (!queuedCount()) return;
  try {
    const res = await api("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out: state.currentOut }),
    });
    state.queue = res.queue || { items: [] };
    runQueueSig = "";
    renderRunQueue();
    toast("已清空待跑队列");
  } catch (err) {
    toast(`清空失败：${err.message}`, true);
  }
}

async function generateStoryboardFromNovel() {
  if (!state.currentOut) {
    toast("请先打开项目", true);
    return;
  }
  await flushOpenEditsInShotsCard();
  const novel = novelValue();
  if (!novel.trim()) {
    toast("请先填写小说正文", true);
    return;
  }
  // 正文先落盘，规划接口才能认定「小说已就绪」
  const novelPath = projectNovelPath();
  try {
    if (novelPath) {
      await api("/api/file/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: novelPath, content: novel }),
      });
    }
  } catch (err) {
    toast(`保存正文失败：${err.message || err}`, true);
    return;
  }
  // 正文已落盘。空项目只有小说脏时清掉，否则生成成功后会被脏标记挡住分镜刷新。
  if (!state.shotSbExists) clearShotSbDirty();
  await openRerunDialog({ kind: "storyboard", id: "" });
}

async function startTask() {
  if (!state.currentOut) {
    toast("请先打开项目", true);
    return;
  }
  if (runningTask()) {
    toast("已有任务在跑，请先停止", true);
    return;
  }
  if (!queuedCount()) {
    toast("待跑队列为空；请先点上方 ↻ 添加", true);
    return;
  }
  const draft = await webDraftForPipeline();
  if (!draft) return;
  try {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().catch(() => {});
    }
    const res = await api("/api/queue/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out: state.currentOut, ...draft }),
    });
    toast(res.message || "已启动");
    const tid = res.id || res.task_id;
    if (tid) jobStatusSeen.set(tid, "running");
    if (res.queue) state.queue = res.queue;
    runQueueSig = "";
    await refreshState();
    if (tid) selectTask(tid);
    syncRunControls();
  } catch (err) {
    toast(`启动失败：${err.message}`, true);
  }
}

// ---------------------------------------------------------------- task list
let tasksSig = "";

function renderTasks() {
  const wrap = $("#task-list");
  // 历史：排除旧版误写入的 queued 占位
  const hist = (state.tasks || []).filter((t) => t.status !== "queued");
  if (!hist.length) {
    if (tasksSig !== "empty") {
      tasksSig = "empty";
      wrap.innerHTML = `<div class="empty">还没有历史。点一次「启动任务」会产生一条。</div>`;
    }
    return;
  }
  const sig = JSON.stringify([
    state.selectedTaskId,
    hist.map((t) => [t.id, t.name, t.status, t.created_at, (t.stages || []).map((s) => [s.key, s.status])]),
  ]);
  if (sig === tasksSig) return;
  tasksSig = sig;
  const ordered = [...hist].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  wrap.innerHTML = ordered.map((t) => {
    const dots = (t.stages || []).map((s) => {
      const cls = s.status === "done" ? "done" : s.status === "running" ? "run" : s.status === "failed" ? "fail" : "";
      const mark = s.status === "running" ? "…" : s.status === "failed" ? "✕" : s.status === "done" ? "✓" : "";
      return `<span class="dot ${cls}">${STAGE_LABELS[s.key] || s.key}${mark}</span>`;
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
  return {
    running: "运行中",
    success: "完成",
    failed: "失败",
    interrupted: "中断",
  }[status] || status;
}

// ---------------------------------------------------------------- task detail
let progressEs = null;
let appliedProgressSeq = -1;
let detailEpoch = 0; // 停止/切空态时递增，作废进行中的进度回调

function showTaskDetailIdle() {
  detailEpoch += 1;
  state.selectedTaskId = null;
  stageTrackSig = "";
  liveItemsSig = "";
  appliedProgressSeq = -1;
  clearEtaAnchors();
  if (progressEs) {
    try { progressEs.close(); } catch {}
    progressEs = null;
  }
  if (state.polling) { clearInterval(state.polling); state.polling = null; }
  $("#task-detail-title").textContent = "任务";
  const fill = $("#overall-fill");
  if (fill) {
    fill.style.width = "0%";
    fill.className = "overall-fill";
  }
  const pctEl = $("#overall-pct");
  if (pctEl) pctEl.textContent = "0%";
  const stageEl = $("#overall-stage");
  if (stageEl) stageEl.textContent = "未在跑 · 队列非空时可点启动";
  const track = $("#stage-track");
  if (track) track.innerHTML = "";
  const live = $("#live-items");
  if (live) live.innerHTML = "";
  const meta = $("#task-meta");
  if (meta) meta.textContent = "";
  $("#log").textContent = "";
  setLogCollapsed(true);
  renderTasks();
  syncRunControls(); // 顺带按是否还有在跑设置停止钮
}

/** 任务已结束时，阶段里不允许再留 running（否则会一直 pulse） */
function normalizeJobStages(job) {
  if (!job || !Array.isArray(job.stages)) return job;
  if (job.status === "running" || job.status === "queued") return job;
  const fall = job.status === "success" ? "done" : "failed";
  job.stages = job.stages.map((s) => (
    s.status === "running" ? { ...s, status: fall } : s
  ));
  if (job.progress && typeof job.progress === "object") {
    job.progress = {
      ...job.progress,
      status: job.status,
      current_stage: null,
      current: job.status === "interrupted" ? "已中断"
        : job.status === "failed" ? "失败"
          : job.status === "success" ? "完成" : job.progress.current,
    };
  }
  return job;
}

async function selectTask(id) {
  state.selectedTaskId = id;
  state.logOffset = 0;
  state.followLog = true;
  stageTrackSig = "";
  liveItemsSig = "";
  appliedProgressSeq = -1;
  $("#log").textContent = "";
  setLogCollapsed(true);
  const job = state.tasks.find((t) => t.id === id);
  if (!job) {
    showTaskDetailIdle();
    return;
  }
  $("#task-detail-title").textContent = `本次运行：${job.name || job.out}`;
  renderStages(job);
  $("#task-meta").textContent = `输出：${job.out} · 创建：${fmtTime(job.created_at)}`;
  syncRunControls();
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
  const epoch = detailEpoch;
  const es = new EventSource(`/api/progress/${encodeURIComponent(taskId)}`);
  progressEs = es;
  const apply = (payload) => {
    // 停止/切空态后作废；避免异步进度把 running 阶段画回去闪烁
    if (epoch !== detailEpoch || state.selectedTaskId !== taskId) return;
    const prog = payload?.progress || payload;
    if (!prog || typeof prog !== "object") return;
    const seq = Number(prog.seq) || 0;
    if (seq < appliedProgressSeq) return;
    appliedProgressSeq = seq;
    const job = state.tasks.find((t) => t.id === taskId);
    if (!job) return;
    // 已中断/失败/完成：忽略引擎还在推的 running 快照
    if (job.status && job.status !== "running") {
      normalizeJobStages(job);
      if (epoch !== detailEpoch || state.selectedTaskId !== taskId) return;
      renderStages(job);
      syncRunControls();
      return;
    }
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
    if (epoch !== detailEpoch || state.selectedTaskId !== taskId) return;
    renderStages(job);
    syncRunControls();
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
  if (!job || !state.selectedTaskId || job.id !== state.selectedTaskId) return;
  normalizeJobStages(job);
  const wrap = $("#stage-track");
  const stages = job.stages || [];
  // 任务非 running 时，阶段绝不再用 running 样式（停了还闪就是这里）
  const paintStatus = (s) => {
    if (job.status !== "running" && s.status === "running") {
      return job.status === "success" ? "done" : "failed";
    }
    return s.status;
  };
  // 结构签名不含 pct：百分比只改文字，避免整段重绘把 pulse 动画掐断重开（闪得不正常）
  const structSig = JSON.stringify([job.status, stages.map((s) => [s.key, paintStatus(s)])]);
  if (structSig !== stageTrackSig) {
    stageTrackSig = structSig;
    wrap.innerHTML = stages.map((s) => {
      const st = paintStatus(s);
      const cls = st === "done" ? "done" : st === "running" ? "running" : st === "failed" ? "failed" : "";
      const icon = st === "done" ? "✓" : st === "running" ? "⏳" : st === "failed" ? "✕" : "·";
      const pct = typeof s.pct === "number" ? ` ${Math.min(100, s.pct).toFixed(0)}%` : "";
      return `<div class="stage-step ${cls}" data-stage="${esc(s.key)}">${icon} ${STAGE_LABELS[s.key] || s.label || s.key}<span class="stage-pct mono">${pct}</span></div>`;
    }).join("");
  } else {
    for (const s of stages) {
      const el = wrap.querySelector(`.stage-step[data-stage="${CSS.escape(s.key)}"] .stage-pct`);
      if (!el) continue;
      el.textContent = typeof s.pct === "number" ? ` ${Math.min(100, s.pct).toFixed(0)}%` : "";
    }
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
  else if (job.status === "failed") currentLabel = "失败";
  else if (job.status === "interrupted") currentLabel = "已中断";
  else if (job.status === "queued") currentLabel = "待跑 · 点「启动任务」开始";
  else if (prog.current) currentLabel = `正在：${prog.current}`;
  else if (job.current_stage) currentLabel = `正在：${STAGE_LABELS[job.current_stage] || job.current_stage}`;
  const pctText = `${Math.min(100, pct).toFixed(1)}%`;
  const width = `${Math.min(100, pct).toFixed(1)}%`;
  const fillCls = "overall-fill " + (
    job.status === "success" ? "done"
      : (job.status === "failed" || job.status === "interrupted") ? "failed"
        : ""
  );
  if (fill.style.width !== width) fill.style.width = width;
  if (fill.className !== fillCls) fill.className = fillCls;
  if (pctEl.textContent !== pctText) pctEl.textContent = pctText;
  if (stageEl.textContent !== currentLabel) stageEl.textContent = currentLabel;
}

function fmtEta(sec) {
  if (sec == null || isNaN(sec)) return "";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `剩 ${sec}s`;
  const m = Math.floor(sec / 60);
  return `剩 ${m}m${String(sec % 60).padStart(2, "0")}s`;
}

let liveItemsSig = "";
/** stepId → { eta_s, at }：服务端 ETA 锚点；本地按墙钟递减，新进度到来则整锚重置 */
const etaAnchors = new Map();
let etaTickTimer = null;

function stopEtaTick() {
  if (etaTickTimer) {
    clearInterval(etaTickTimer);
    etaTickTimer = null;
  }
}

function clearEtaAnchors() {
  etaAnchors.clear();
  stopEtaTick();
}

function syncEtaAnchors(list) {
  const now = Date.now();
  const seen = new Set();
  for (const i of list || []) {
    if (i.status !== "running" || i.eta_s == null || i.eta_s === "") continue;
    const id = String(i.id || "");
    if (!id) continue;
    const eta = Number(i.eta_s);
    if (!Number.isFinite(eta)) continue;
    seen.add(id);
    const prev = etaAnchors.get(id);
    // 服务端 ETA 常原样重复推送；只在数值真变（或首次）时重锚，避免倒计时反复跳回
    if (!prev || prev.server_eta !== eta) {
      etaAnchors.set(id, { eta_s: eta, server_eta: eta, at: now });
    }
  }
  for (const id of [...etaAnchors.keys()]) {
    if (!seen.has(id)) etaAnchors.delete(id);
  }
  if (etaAnchors.size) ensureEtaTick();
  else stopEtaTick();
}

function etaRemaining(id) {
  const a = etaAnchors.get(String(id || ""));
  if (!a) return null;
  return Math.max(0, a.eta_s - (Date.now() - a.at) / 1000);
}

function paintLiveEtas() {
  for (const el of $$("#live-items .live-item[data-id]")) {
    const etaEl = $(".li-eta", el);
    if (!etaEl) continue;
    const rem = el.classList.contains("current") ? etaRemaining(el.dataset.id) : null;
    const text = rem == null ? "" : fmtEta(rem);
    if (etaEl.textContent !== text) etaEl.textContent = text;
  }
}

function ensureEtaTick() {
  if (etaTickTimer) return;
  etaTickTimer = setInterval(() => {
    if (!etaAnchors.size) {
      stopEtaTick();
      return;
    }
    paintLiveEtas();
  }, 1000);
}

function liveItemPct(i) {
  const step = Number(i.step) || 0;
  const total = Math.max(1, Number(i.total) || 1);
  if (typeof i.pct === "number") return Math.min(100, i.pct);
  return total > 1 ? (100 * step) / total : 0;
}

function liveItemStepText(i, pct) {
  const step = Number(i.step) || 0;
  const total = Math.max(1, Number(i.total) || 1);
  if (total > 1) return `${step}/${total}`;
  if (pct > 0 && pct < 100) return `${pct.toFixed(0)}%`;
  return "";
}

function liveItemClass(job, i) {
  return [
    "live-item",
    i.status === "done" ? "done" : "",
    i.status === "failed" ? "failed" : "",
    // 只给真正 running 的项标 current；已完成的不要因 current_step_id 残留高亮/抢焦
    job.status === "running" && i.status === "running" ? "current" : "",
    i.status === "pending" ? "pending" : "",
  ].filter(Boolean).join(" ");
}

function patchLiveItem(el, job, i) {
  const pct = liveItemPct(i);
  const width = `${pct}%`;
  const cls = liveItemClass(job, i);
  const stepText = liveItemStepText(i, pct);
  const eta = job.status === "running" && i.status === "running"
    ? fmtEta(etaRemaining(i.id) ?? i.eta_s) : "";
  if (el.className !== cls) el.className = cls;
  const stepEl = $(".li-step", el);
  if (stepEl && stepEl.textContent !== stepText) stepEl.textContent = stepText;
  const fill = $(".li-fill", el);
  // 就地改 width，CSS transition 才能跟总进度条一样平滑
  if (fill && fill.style.width !== width) fill.style.width = width;
  const etaEl = $(".li-eta", el);
  if (etaEl && etaEl.textContent !== eta) etaEl.textContent = eta;
}

function renderLiveItems(job) {
  const wrap = $("#live-items");
  const prog = job.progress || {};
  const steps = Array.isArray(prog.steps) ? prog.steps : [];
  const fall = job.status === "success" ? "done" : "failed";
  const paintItem = (i) => {
    if (job.status === "running") return i;
    if (i.status === "running" || i.status === "pending") return { ...i, status: fall };
    return i;
  };
  // 完整步骤表；无 steps 时退回旧 items
  const raw = steps.length
    ? steps
    : (prog.items || []).map((i) => ({ ...i, status: i.status || "running" }));
  const list = raw.map(paintItem);
  // 进度一到就重置 ETA 锚点；列表 DOM 未变时只重绘秒数
  syncEtaAnchors(job.status === "running" ? list : []);
  // 结构签名：只决定要不要重建 DOM；进度变化走就地 patch，保留条宽 transition
  const structureSig = JSON.stringify([
    job.status === "running" ? "run" : job.status,
    list.map((i) => [i.id, i.stage || "", i.label || ""]),
  ]);
  const progressSig = JSON.stringify([
    job.status,
    list.map((i) => [i.id, i.status, i.step, i.total, i.pct, i.eta_s]),
  ]);
  if (progressSig === liveItemsSig) {
    if (job.status === "running") paintLiveEtas();
    return;
  }
  const structureSame = structureSig === (wrap.dataset.structureSig || "");
  liveItemsSig = progressSig;
  wrap.dataset.structureSig = structureSig;

  if (!list.length) {
    clearEtaAnchors();
    wrap.innerHTML = job.status === "running"
      ? `<div class="empty" style="padding:8px">等待步骤表…</div>`
      : "";
    return;
  }

  const existing = $$(".live-item[data-id]", wrap);
  const canPatch = structureSame
    && existing.length === list.length
    && existing.every((el, idx) => el.dataset.id === String(list[idx].id || ""));

  if (canPatch) {
    existing.forEach((el, idx) => patchLiveItem(el, job, list[idx]));
    return;
  }

  let lastGroup = null;
  const parts = [];
  for (const i of list) {
    // 肖像与静帧同属 images 阶段，列表里单独成组，避免「肖像进度条」被静帧标题盖住
    const groupKey = String(i.id || "").startsWith("images:portrait:")
      ? "portraits"
      : (i.stage || "");
    const groupLabel = groupKey === "portraits"
      ? "肖像生成"
      : (STAGE_LABELS[i.stage] || i.stage);
    if (groupKey && groupKey !== lastGroup) {
      lastGroup = groupKey;
      parts.push(`<div class="step-group">${esc(groupLabel)}</div>`);
    }
    const pct = liveItemPct(i);
    const stepText = liveItemStepText(i, pct);
    const eta = job.status === "running" && i.status === "running"
      ? fmtEta(etaRemaining(i.id) ?? i.eta_s) : "";
    parts.push(`<div class="${liveItemClass(job, i)}" data-id="${esc(i.id || "")}">
      <span class="li-label" title="${esc(i.label || "")}">${esc(i.label || i.id)}</span>
      <span class="li-step mono">${stepText}</span>
      <div class="li-bar"><div class="li-fill" style="width:${pct}%"></div></div>
      <span class="li-eta">${eta}</span>
    </div>`);
  }
  wrap.innerHTML = parts.join("");
  // 只改 #live-items 的 scrollTop，不调用 scrollIntoView（会带动整页）
  const cur = wrap.querySelector(".live-item.current");
  if (cur) {
    const delta = cur.getBoundingClientRect().top - wrap.getBoundingClientRect().top + wrap.scrollTop;
    const top = delta - (wrap.clientHeight - cur.offsetHeight) / 2;
    wrap.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }
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
        noteJobStatuses([job]);
        syncRunControls();
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
  if (!log.hidden && (nearBottom || state.followLog)) log.scrollTop = log.scrollHeight;
}

function setLogCollapsed(collapsed) {
  const section = $("#log-section");
  const log = $("#log");
  const toggle = $("#log-toggle");
  if (!section || !log || !toggle) return;
  section.classList.toggle("collapsed", collapsed);
  log.hidden = collapsed;
  toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  const hint = toggle.querySelector(".hint");
  if (hint) hint.textContent = collapsed ? "默认折叠 · 点击展开" : "点击折叠";
  if (!collapsed && state.followLog) log.scrollTop = log.scrollHeight;
}

function toggleLogSection() {
  const collapsed = !$("#log-section")?.classList.contains("collapsed");
  setLogCollapsed(collapsed);
}

async function stopTask() {
  if (!state.currentOut) {
    toast("请先打开项目", true);
    return;
  }
  // 本地立刻：历史标中断；队列保持，状态改回待跑；进度清零
  for (const job of state.tasks || []) {
    if (job.status === "running") {
      job.status = "interrupted";
      job.error = "用户停止";
      (job.stages || []).forEach((s) => {
        if (s.status === "running") s.status = "failed";
      });
    }
  }
  if (state.queue?.items) {
    state.queue.items = state.queue.items.map((it) => ({ ...it, status: "pending" }));
  }
  runQueueSig = "";
  showTaskDetailIdle();
  mediaCache = null;
  tasksSig = "";
  renderRunQueue();
  renderTasks();
  syncRunControls();
  try {
    const res = await api("/api/task/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out: state.currentOut }),
    });
    toast(res.message || "已停止");
    if (res.queue) state.queue = res.queue;
    refreshInFlight = false;
    refreshQueued = false;
    await refreshState();
    showTaskDetailIdle();
    mediaCache = null;
    await renderShots();
    runQueueSig = "";
    renderRunQueue();
    syncRunControls();
  } catch (err) {
    toast(`停止失败：${err.message}`, true);
    await refreshState();
    showTaskDetailIdle();
    syncRunControls();
  }
}

async function clearTasks() {
  try {
    const res = await api("/api/tasks/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    toast(res.kept ? `已清空本项目的记录（运行中的 ${res.kept} 个任务保留）` : "已清空本项目的记录");
    if (!state.tasks.some((t) => t.id === state.selectedTaskId && t.status === "running")) {
      // 当前选中的任务被清掉时，详情面板保留为空态
      showTaskDetailIdle();
    }
    await refreshState();
    if (!state.selectedTaskId) {
      const pick = state.tasks.find((t) => t.status === "running");
      if (pick) selectTask(pick.id);
      else showTaskDetailIdle();
    }
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
let shotAddCard = null; // 网格末尾「＋」空卡

function clearShotCards() {
  for (const [, card] of shotCards) card.remove();
  shotCards.clear();
  if (shotAddCard) {
    shotAddCard.remove();
    shotAddCard = null;
  }
}

function ensureShotAddCard() {
  if (shotAddCard && document.contains(shotAddCard)) return shotAddCard;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "shot-card shot-add-card";
  btn.title = "添加镜头";
  btn.setAttribute("aria-label", "添加镜头");
  btn.innerHTML = `<span class="shot-add-plus" aria-hidden="true">＋</span>`;
  btn.addEventListener("click", () => { addShotOnGrid(); });
  shotAddCard = btn;
  return btn;
}

function placeShotAddCard(grid) {
  if (!grid) return;
  const add = ensureShotAddCard();
  if (grid.lastElementChild !== add) grid.appendChild(add);
}

function clearShotsMsg() {
  const msg = $("#shots-msg");
  if (!msg) return;
  msg.hidden = true;
  msg.classList.remove("ok", "bad");
  msg.innerHTML = "";
}

function setShotsMsg(html, { ok = false, bad = false } = {}) {
  const msg = $("#shots-msg");
  if (!msg) return;
  const text = String(html || "").trim();
  if (!text) {
    clearShotsMsg();
    return;
  }
  msg.hidden = false;
  msg.classList.toggle("ok", !!ok);
  msg.classList.toggle("bad", !!bad);
  msg.innerHTML = text;
}

function formatBudgetTips(data) {
  const gib = data.total_gib != null ? `${data.total_gib} GB` : "本机";
  const tier = data.tier_gib != null ? `${data.tier_gib} GB 档` : "";
  const res = data.resolution || "";
  const fps = data.fps ?? "?";
  const frames = data.max_frames ?? "?";
  const sec = data.max_seconds ?? "?";
  const cps = data.chars_per_second ?? 2.4;
  const limit = data.limit_chars ?? "?";
  const hw = [gib, tier, res ? `${res}@${fps}fps` : ""].filter(Boolean).join(" · ");
  return `<p class="shots-msg-tip">硬件：${esc(hw || "未读取到内存档")}</p>
    <p class="shots-msg-tip">单镜上限：${esc(String(frames))} 帧 ≈ ${esc(String(sec))}s（总内存 → token 预算表 → 按 ${esc(String(res) || "分辨率")}@${esc(String(fps))}fps 折算）</p>
    <p class="shots-msg-tip">台词额度：${esc(String(sec))}s × ${esc(String(cps))} 字/秒（Qwen3-TTS 中文实测）≈ <strong>${esc(String(limit))} 字/镜</strong>；超了按标点拆相邻镜，不许删字。</p>`;
}

function formatValidateFailure(data, fallback = "") {
  const violations = Array.isArray(data?.violations) ? data.violations : [];
  const budget = formatBudgetTips(data || {});
  if (!violations.length) {
    const detail = esc(String(data?.strict_error || data?.error || fallback || "校验未通过"));
    return `<div class="shots-msg-title">硬校验未通过</div>${budget}<p class="shots-msg-tip">${detail}</p>`;
  }
  const items = violations.map((v) => {
    const rows = (v.splits || []).map((p) =>
      `<div class="shots-msg-split"><dt>${esc(p.id)}</dt><dd>${esc(p.dialogue)}</dd></div>`
    ).join("");
    return `<div class="shots-msg-item">
      <div class="shots-msg-item-head">${esc(v.id)} · ${v.chars} 字 ≈ ${v.seconds}s</div>
      ${rows}
    </div>`;
  }).join("");
  return `<div class="shots-msg-title">对白超单镜额度 · ${violations.length} 镜</div>
    ${budget}
    <p class="shots-msg-tip">不要删字压短，按下表拆成相邻镜头，每镜换景别/角度：</p>
    ${items}`;
}

let validateTimer = null;
let validateSeq = 0;

function scheduleAutoValidate() {
  if (validateTimer) clearTimeout(validateTimer);
  validateTimer = setTimeout(() => {
    validateTimer = null;
    validateShotSbFromGrid({ auto: true });
  }, 450);
}

function markShotSbDirty() {
  state.shotSbDirty = true;
  if (state.shotSb) delete state.shotSb._missing;
  const el = $("#shots-dirty");
  if (el) el.hidden = false;
  scheduleAutoValidate();
}

function clearShotSbDirty() {
  state.shotSbDirty = false;
  const el = $("#shots-dirty");
  if (el) el.hidden = true;
  for (const [, card] of shotCards) card.classList.remove("dirty");
}

function nextShotId(shots) {
  let max = 0;
  for (const s of shots || []) {
    const m = String(s.id || "").match(/^s(\d+)/i);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `s${String(max + 1).padStart(3, "0")}`;
}

/** 32-bit 无符号随机 seed（与管线 stable_seed / 角色新建一致） */
function randomSeed() {
  return Math.floor(Math.random() * 0xffffffff);
}

/** 与 pipeline.util.stable_seed 一致：sha256(parts.join("|")) 前 8 位十六进制 */
async function stableSeedAsync(...parts) {
  const data = new TextEncoder().encode(parts.map((p) => String(p ?? "")).join("|"));
  const buf = await crypto.subtle.digest("SHA-256", data);
  const hex = [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return parseInt(hex.slice(0, 8), 16);
}

/** 给缺 seed 的字段补上稳定值，返回是否改过 */
async function ensureStorySeeds(sb) {
  if (!sb) return false;
  let changed = false;
  const title = String(sb.title || "").trim();
  const style = String(sb.style || "").trim();
  const musicPrompt = String(sb.music_prompt || "").trim();
  if (sb.music_seed == null || sb.music_seed === "") {
    sb.music_seed = await stableSeedAsync("bgm", musicPrompt, title);
    changed = true;
  } else {
    sb.music_seed = Number(sb.music_seed) >>> 0;
  }
  if (!Array.isArray(sb.characters)) sb.characters = [];
  for (const ch of sb.characters) {
    if (ch.seed == null || ch.seed === "") {
      ch.seed = await stableSeedAsync("portrait", ch.name || "", ch.appearance || "");
      changed = true;
    } else {
      ch.seed = Number(ch.seed) >>> 0;
    }
  }
  if (!Array.isArray(sb.shots)) sb.shots = [];
  for (const shot of sb.shots) {
    if (shot.seed == null || shot.seed === "") {
      shot.seed = await stableSeedAsync(
        String(shot.id || ""),
        String(shot.image_prompt || ""),
        String(shot.motion_prompt || ""),
      );
      changed = true;
    } else {
      shot.seed = Number(shot.seed) >>> 0;
    }
  }
  return changed;
}

const EMOTION_OPTIONS = [
  "calm", "warm", "sad", "tense", "bitter", "nostalgic", "surprised", "whisper",
];

const VOICE_OPTIONS = {
  female: ["Serena", "Vivian"],
  male: ["Dylan", "Uncle_Fu", "Eric"],
};

const GENDER_OPTIONS = [
  { value: "female", label: "女" },
  { value: "male", label: "男" },
];

function storyCharacters() {
  return Array.isArray(state.shotSb?.characters) ? state.shotSb.characters : [];
}

function characterLabel(ch) {
  if (!ch) return "";
  return ch.name || ch.id || "";
}

function characterNameById(id) {
  const ch = storyCharacters().find((c) => c.id === id);
  return ch ? characterLabel(ch) : id;
}

function shotCharacters(shot) {
  return Array.isArray(shot?.characters) ? shot.characters.filter(Boolean) : [];
}

function shotFieldRaw(shot, key) {
  if (!shot) return "";
  if (key === "characters") return shotCharacters(shot).join(",");
  if (key === "seed") return shot.seed == null || shot.seed === "" ? "" : String(shot.seed);
  return shot[key] == null ? "" : String(shot[key]);
}

function shotFieldShown(shot, key, placeholder) {
  if (key === "characters") {
    const ids = shotCharacters(shot);
    if (!ids.length) return { text: placeholder || "出场角色", empty: true };
    return {
      text: ids.map((id) => characterNameById(id) || id).join("、"),
      empty: false,
    };
  }
  const raw = shotFieldRaw(shot, key).trim();
  if (!raw) return { text: placeholder || "点击选择", empty: true };
  return { text: raw, empty: false };
}

function fillSelectOptions(select, key, shot) {
  if (!select || select.tagName !== "SELECT") return;
  const cur = shotFieldRaw(shot, key);
  let html = `<option value="">（空）</option>`;
  if (key === "emotion") {
    html += EMOTION_OPTIONS.map((e) =>
      `<option value="${esc(e)}"${e === cur ? " selected" : ""}>${esc(e)}</option>`
    ).join("");
  } else if (key === "speaker") {
    const names = [...new Set(storyCharacters().map(characterLabel).filter(Boolean))];
    html += names.map((n) =>
      `<option value="${esc(n)}"${n === cur ? " selected" : ""}>${esc(n)}</option>`
    ).join("");
    if (cur && !names.includes(cur)) {
      html += `<option value="${esc(cur)}" selected>${esc(cur)}</option>`;
    }
  }
  select.innerHTML = html;
  select.value = cur;
}

function characterBySpeakerName(name) {
  const n = String(name || "").trim();
  if (!n) return null;
  return storyCharacters().find((c) => characterLabel(c) === n) || null;
}

function miniAvatarHtml(ch) {
  const person = ch && typeof ch === "object" ? ch : { name: String(ch || "?") };
  const url = castAvatarUrl(person, { portraits: state.portraits || {} });
  const initial = esc(castInitial(person));
  if (url) {
    return `<span class="shot-mini-avatar" aria-hidden="true"><img class="shot-mini-avatar-img" src="${esc(url)}" alt="" loading="lazy" onerror="const p=this.parentElement;const fb=p&&p.querySelector('.shot-mini-avatar-fb');this.remove();if(fb)fb.hidden=false"><span class="shot-mini-avatar-fb" hidden>${initial}</span></span>`;
  }
  return `<span class="shot-mini-avatar" aria-hidden="true"><span class="shot-mini-avatar-fb">${initial}</span></span>`;
}

function fillCharChecks(box, shot) {
  if (!box) return;
  const selected = new Set(shotCharacters(shot));
  const chars = storyCharacters();
  if (!chars.length) {
    box.innerHTML = `<div class="hint">暂无角色，先在演职人员里添加</div>`;
    return;
  }
  const known = new Set();
  let html = chars.map((ch) => {
    const id = ch.id || "";
    if (!id) return "";
    known.add(id);
    const label = characterLabel(ch) || id;
    return `<label class="shot-char-check"><input type="checkbox" value="${esc(id)}"${selected.has(id) ? " checked" : ""}>${miniAvatarHtml(ch)}<span>${esc(label)}</span></label>`;
  }).join("");
  for (const id of selected) {
    if (known.has(id)) continue;
    html += `<label class="shot-char-check"><input type="checkbox" value="${esc(id)}" checked>${miniAvatarHtml({ id, name: id })}<span>${esc(id)}</span></label>`;
  }
  box.innerHTML = html;
}

function readCharChecks(box) {
  if (!box) return [];
  return $$("input[type=checkbox]:checked", box).map((el) => el.value).filter(Boolean);
}

function isShotFieldEditing(wrap) {
  const input = $(".shot-iedit-input", wrap);
  return !!input && !input.hidden;
}

function fillShotInlineField(wrap, shot) {
  if (!wrap || isShotFieldEditing(wrap)) return;
  const key = wrap.dataset.f;
  const placeholder = wrap.dataset.placeholder || "点击选择";
  const display = $(".shot-iedit-display", wrap);
  const input = $(".shot-iedit-input", wrap);
  if (!display || !input) return;
  if (wrap.dataset.kind === "select") fillSelectOptions(input, key, shot);
  else if (wrap.dataset.kind === "checks") fillCharChecks(input, shot);
  else input.value = shotFieldRaw(shot, key);
  const shown = shotFieldShown(shot, key, placeholder);
  display.classList.toggle("empty", shown.empty);
  if (!shown.empty && key === "speaker") {
    const ch = characterBySpeakerName(shown.text) || { name: shown.text };
    display.innerHTML = `${miniAvatarHtml(ch)}<span class="shot-mini-name">${esc(shown.text)}</span>`;
    display.classList.add("with-avatars");
  } else if (!shown.empty && key === "characters") {
    display.innerHTML = shotCharacters(shot).map((id) => {
      const ch = storyCharacters().find((c) => c.id === id) || { id, name: id };
      const label = characterNameById(id) || id;
      return `<span class="shot-char-chip">${miniAvatarHtml(ch)}<span class="shot-mini-name">${esc(label)}</span></span>`;
    }).join("");
    display.classList.add("with-avatars");
  } else {
    display.textContent = shown.text;
    display.classList.remove("with-avatars");
  }
}

function closeShotFieldOutside(wrap) {
  if (wrap._outsideClose) {
    document.removeEventListener("pointerdown", wrap._outsideClose, true);
    wrap._outsideClose = null;
  }
}

function beginShotFieldEdit(wrap, card) {
  const display = $(".shot-iedit-display", wrap);
  const input = $(".shot-iedit-input", wrap);
  if (!display || !input || isShotFieldEditing(wrap)) return;
  if (wrap.dataset.kind === "select") {
    fillSelectOptions(input, wrap.dataset.f, card?._shot);
  } else if (wrap.dataset.kind === "checks") {
    fillCharChecks(input, card?._shot);
  }
  display.hidden = true;
  input.hidden = false;
  if (wrap.dataset.kind === "checks") {
    // 点到卡片外再收起（比 blur 更稳，空白处也能触发）
    closeShotFieldOutside(wrap);
    wrap._outsideClose = (ev) => {
      if (wrap.contains(ev.target)) return;
      commitShotField(wrap, card);
    };
    setTimeout(() => document.addEventListener("pointerdown", wrap._outsideClose, true), 0);
    const first = $("input[type=checkbox]", input);
    if (first) first.focus();
    return;
  }
  input.focus();
  if (input.tagName === "SELECT") {
    try { input.showPicker?.(); } catch {}
  } else if (input.tagName === "INPUT") {
    try { input.select(); } catch {}
  } else if (input.tagName === "TEXTAREA") {
    const len = input.value.length;
    try { input.setSelectionRange(len, len); } catch {}
  }
}

async function commitShotField(wrap, card, { cancel = false } = {}) {
  const display = $(".shot-iedit-display", wrap);
  const input = $(".shot-iedit-input", wrap);
  const shot = card?._shot;
  if (!display || !input || !shot || !isShotFieldEditing(wrap)) return;
  closeShotFieldOutside(wrap);
  const key = wrap.dataset.f;
  const placeholder = wrap.dataset.placeholder || "点击选择";
  const prevRaw = shotFieldRaw(shot, key);
  if (cancel) {
    if (wrap.dataset.kind === "select") fillSelectOptions(input, key, shot);
    else if (wrap.dataset.kind === "checks") fillCharChecks(input, shot);
    else input.value = prevRaw;
  }
  input.hidden = true;
  display.hidden = false;

  let changed = false;
  if (key === "characters") {
    const next = wrap.dataset.kind === "checks"
      ? readCharChecks(input)
      : (input.value ? [input.value] : []);
    const prev = shotCharacters(shot);
    if (next.join(",") !== prev.join(",")) {
      shot.characters = next;
      changed = true;
    }
  } else if (key === "seed") {
    const raw = String(input.value || "").trim();
    const n = raw === "" ? null : parseInt(raw, 10);
    const next = Number.isFinite(n) ? n : null;
    const prev = shot.seed == null ? null : Number(shot.seed);
    if (next !== prev) {
      if (next == null) delete shot.seed;
      else shot.seed = next;
      changed = true;
    }
  } else {
    const nextRaw = input.value;
    const normalized = key === "dialogue" || key.endsWith("_prompt") ? nextRaw : nextRaw.trim();
    if ((shot[key] || "") !== normalized) {
      shot[key] = normalized;
      changed = true;
    }
  }

  const shown = shotFieldShown(shot, key, placeholder);
  display.textContent = shown.text;
  display.classList.toggle("empty", shown.empty);
  if (!changed) return;

  card.classList.add("dirty");
  markShotSbDirty();
  if (key === "dialogue") syncShotDialogueUi(card, shot);
}

function shotHasDialogue(shot) {
  return Boolean(String(shot?.dialogue || "").trim());
}

/** 空镜隐去说话人 / 情绪 / 配音播放器 / ↻ 配音 */
function syncShotDialogueUi(card, shot) {
  if (!card) return;
  const has = shotHasDialogue(shot);
  const row2 = $(".shot-row2", card);
  if (row2) row2.hidden = !has;
  const audioEl = $(".shot-audio", card);
  if (audioEl) audioEl.hidden = !has;
  if (card._btnAudio) card._btnAudio.hidden = !has;
}

function rerollShotSeed(card) {
  const shot = card?._shot;
  if (!shot) return;
  const wrap = $(".shot-iedit[data-f=seed]", card);
  if (wrap && isShotFieldEditing(wrap)) {
    // 先退出编辑态，避免旧输入盖住新 seed
    const input = $(".shot-iedit-input", wrap);
    const display = $(".shot-iedit-display", wrap);
    if (input) input.hidden = true;
    if (display) display.hidden = false;
  }
  shot.seed = randomSeed();
  if (wrap) fillShotInlineField(wrap, shot);
  card.classList.add("dirty");
  markShotSbDirty();
}

function bindShotCardFields(card) {
  $$(".shot-iedit", card).forEach((wrap) => {
    if (wrap._bound) return;
    wrap._bound = true;
    const display = $(".shot-iedit-display", wrap);
    const input = $(".shot-iedit-input", wrap);
    const multiline = wrap.hasAttribute("data-multiline");
    const isSelect = wrap.dataset.kind === "select";
    const isChecks = wrap.dataset.kind === "checks";
    display?.addEventListener("click", () => beginShotFieldEdit(wrap, card));
    input?.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        commitShotField(wrap, card, { cancel: true });
      } else if (!isSelect && !isChecks && ev.key === "Enter" && (!multiline || ev.metaKey || ev.ctrlKey)) {
        ev.preventDefault();
        commitShotField(wrap, card);
      }
    });
    if (isSelect) {
      input?.addEventListener("change", () => commitShotField(wrap, card));
    }
    if (isChecks) {
      // 勾选即时写回内存；点到外面再收起成文案
      input?.addEventListener("change", (ev) => {
        if (!ev.target?.matches?.("input[type=checkbox]")) return;
        const shot = card._shot;
        if (!shot) return;
        shot.characters = readCharChecks(input);
        card.classList.add("dirty");
        markShotSbDirty();
      });
    } else {
      input?.addEventListener("blur", () => commitShotField(wrap, card));
    }
  });
  const reroll = $(".seed-reroll[data-seed-for=shot]", card);
  if (reroll && !reroll._bound) {
    reroll._bound = true;
    reroll.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      rerollShotSeed(card);
    });
  }
}

function fillShotCardFields(card, shot) {
  $$(".shot-iedit", card).forEach((wrap) => fillShotInlineField(wrap, shot));
}

const SHOT_CARD_UI = 9;

function ensureShotCard(id) {
  let card = shotCards.get(id);
  if (card && card._ui !== SHOT_CARD_UI) {
    card.remove();
    shotCards.delete(id);
    card = null;
  }
  if (!card) {
    const node = $("#shot-card-tpl").content.cloneNode(true);
    card = $(".shot-card", node);
    card._ui = SHOT_CARD_UI;
    $(".shot-id", card).textContent = id;
    $(".shot-more-toggle", card)?.addEventListener("click", () => {
      const more = $(".shot-more", card);
      if (!more) return;
      const open = more.hidden;
      more.hidden = !open;
      $(".shot-more-toggle", card).textContent = open ? "提示词 ▾" : "提示词 ▸";
    });
    shotCards.set(id, card);
  }
  return card;
}

function emptyStoryboardShell() {
  return {
    title: "",
    style: "",
    music_prompt: "",
    characters: [],
    shots: [],
    _missing: true,
  };
}

async function loadShotSb(out, { force = false } = {}) {
  const path = `${out}/storyboard.json`;
  // 未脏时默认跟文件走；force / 路径变了才跳过「同路径缓存」
  if (!force && state.shotSb && state.shotSbPath === path && state.shotSbMtime != null) {
    // 仍去读盘比 mtime，避免删文件后界面残留
  }
  let data;
  try {
    data = await api("/api/storyboard/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  } catch {
    data = { exists: false, storyboard: null, mtime: null };
  }
  const exists = !!data.exists && !!data.storyboard;
  const mtime = data.mtime == null ? null : Number(data.mtime);
  // 磁盘上从无到有：即使本地因粘贴小说标了脏，也要吃进新分镜（否则控件不刷新）
  const appeared = exists && state.shotSbExists === false;
  // 有未保存编辑：不拿磁盘盖掉（「从无到有」除外）
  if (!force && !appeared && state.shotSbDirty && state.shotSb && state.shotSbPath === path) {
    return state.shotSb;
  }
  // 未脏且文件没变：复用内存（保留本地补过的 seed 显示）
  if (
    !force
    && state.shotSb
    && state.shotSbPath === path
    && state.shotSbExists === exists
    && state.shotSbMtime === mtime
    && exists
  ) {
    return state.shotSb;
  }

  state.shotSbPath = path;
  state.currentStoryboard = path;
  state.shotSbExists = exists;
  state.shotSbMtime = mtime;

  if (!exists) {
    state.shotSb = emptyStoryboardShell();
    clearShotSbDirty();
    clearShotsMsg();
    if (!isTitleEditing()) setTitleDisplay(state.project || "");
    if (!isStyleEditing()) setStyleDisplay("");
    if (!isMusicEditing()) setMusicDisplay("");
    setMusicSeedDisplay(null);
    return state.shotSb;
  }

  state.shotSb = data.storyboard;
  if (!Array.isArray(state.shotSb.shots)) state.shotSb.shots = [];
  if (!Array.isArray(state.shotSb.characters)) state.shotSb.characters = [];
  delete state.shotSb._missing;
  // 缺 seed 时按管线同款算法补齐；若与磁盘不一致则标脏，提示保存写回
  const filled = await ensureStorySeeds(state.shotSb);
  clearShotSbDirty();
  if (filled) markShotSbDirty();
  setMusicSeedDisplay(state.shotSb.music_seed);
  if (!isStyleEditing()) setStyleDisplay(String(state.shotSb.style || "").trim());
  if (!isMusicEditing()) setMusicDisplay(String(state.shotSb.music_prompt || "").trim());
  // 片名取分镜 title，不要用项目文件夹名顶上
  const sbTitle = String(state.shotSb.title || "").trim();
  if (sbTitle && !isTitleEditing()) {
    const cur = titleValue();
    if (!cur || cur === state.project) setTitleDisplay(sbTitle);
  }
  return state.shotSb;
}

async function flushOpenEditsInShotsCard() {
  if (isTitleEditing()) await commitTitle();
  if (isNovelEditing()) await commitNovel();
  if (isStyleEditing()) await commitStyle();
  if (isMusicEditing()) await commitMusic();
  if (isMusicSeedEditing()) commitMusicSeed();
  for (const wrap of $$("#shots-grid .shot-iedit, #final-cast .cast-iedit")) {
    const input = $(".shot-iedit-input, .cast-iedit-input", wrap);
    if (!input || input.hidden) continue;
    const card = wrap.closest(".shot-card, .cast-card");
    if (wrap.classList.contains("cast-iedit")) await commitCastField(wrap, card);
    else await commitShotField(wrap, card);
  }
}

async function saveShotSbFromGrid({ quiet = false } = {}) {
  await flushOpenEditsInShotsCard();
  if (!state.shotSb || !state.shotSbPath) {
    const out = state.currentOut;
    if (out) await loadShotSb(out);
  }
  if (!state.shotSb || !state.shotSbPath) {
    toast("没有可保存的分镜", true);
    return false;
  }
  syncMetaToShotSb();
  try {
    await api("/api/storyboard/write", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.shotSbPath, storyboard: state.shotSb }),
    });
    const novelPath = projectNovelPath();
    if (novelPath) {
      await api("/api/file/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: novelPath, content: novelValue() }),
      });
    }
    clearShotSbDirty();
    for (const [, card] of shotCards) card.classList.remove("dirty");
    if (!quiet) toast(`已保存分镜与正文`);
    scheduleAutoValidate();
    return true;
  } catch (err) {
    toast(`保存失败：${err.message}`, true);
    return false;
  }
}

async function revertShotSbFromGrid() {
  const out = state.currentOut;
  if (!out) {
    toast("请先打开项目", true);
    return;
  }
  const dirty = state.shotSbDirty || isTitleEditing() || isNovelEditing()
    || !!document.querySelector("#shots-grid .shot-iedit-input:not([hidden]), #final-cast .cast-iedit-input:not([hidden])");
  if (dirty && !confirm("还原后，未保存的修改会全部丢掉，确定？")) return;

  // 关掉编辑态，避免还原后还挂着旧输入框
  if (isTitleEditing()) {
    const input = $("#title");
    const display = $("#title-display");
    if (input && display) {
      input.hidden = true;
      display.hidden = false;
    }
  }
  if (isNovelEditing()) {
    const input = $("#novel-text");
    const display = $("#novel-display");
    if (input && display) {
      input.hidden = true;
      display.hidden = false;
    }
  }
  if (isStyleEditing()) {
    const input = $("#style-text");
    const display = $("#style-display");
    if (input && display) {
      input.hidden = true;
      display.hidden = false;
    }
  }
  if (isMusicEditing()) {
    const input = $("#music-text");
    const display = $("#music-display");
    if (input && display) {
      input.hidden = true;
      display.hidden = false;
    }
  }
  if (isMusicSeedEditing()) {
    const input = $("#music-seed-input");
    const display = $("#music-seed-display");
    if (input && display) {
      input.hidden = true;
      display.hidden = false;
    }
  }

  state.shotSb = null;
  state.shotSbPath = "";
  clearShotCards();
  const castEl = $("#final-cast");
  if (castEl) {
    delete castEl.dataset.sig;
    castEl.innerHTML = "";
  }

  try {
    await loadShotSb(out, { force: true });
    await loadNovel({ quiet: true });
    // 片名 / 风格 / 配乐跟磁盘分镜走
    const sb = state.shotSb || {};
    setTitleDisplay(String(sb.title || "").trim());
    setStyleDisplay(String(sb.style || "").trim());
    setMusicDisplay(String(sb.music_prompt || "").trim());
    setMusicSeedDisplay(sb.music_seed);
    await renderShots();
    // 不在此处强制清脏：loadShotSb 若补了磁盘没有的 seed，界面≠文件，应继续标脏
    clearShotsMsg();
    toast("已还原为文件里的原始状态");
    scheduleAutoValidate();
  } catch (err) {
    toast(`还原失败：${err.message}`, true);
  }
}

/** 从网页当前编辑态拼一份分镜快照（镜头以卡片 _shot 为准） */
function snapshotShotSbFromUi() {
  const base = state.shotSb || {
    title: titleValue() || state.project || "",
    style: styleValue(),
    music_prompt: musicValue(),
    characters: [],
    shots: [],
  };
  syncMetaToShotSb();
  const byId = new Map();
  for (const [, card] of shotCards) {
    if (card?._shot?.id) byId.set(card._shot.id, card._shot);
  }
  const shots = [];
  const seen = new Set();
  for (const s of base.shots || []) {
    const live = byId.get(s.id) || s;
    shots.push(live);
    seen.add(live.id);
  }
  for (const [id, shot] of byId) {
    if (!seen.has(id)) shots.push(shot);
  }
  return {
    ...base,
    title: titleValue() || base.title || "",
    style: styleValue() || base.style || "",
    music_prompt: musicValue() || base.music_prompt || "",
    music_seed: base.music_seed,
    characters: Array.isArray(base.characters) ? base.characters : [],
    shots,
  };
}

async function validateShotSbFromGrid({ auto = false } = {}) {
  // 硬校验只读网页当前状态，不写盘；通过不显示，失败才展示
  const seq = ++validateSeq;
  if (!auto) await flushOpenEditsInShotsCard();
  // 自动校验：若还在输入框里，等失焦后再跑
  if (auto) {
    const editing = document.activeElement?.closest?.(
      "#shots-grid .shot-iedit, #final-cast .cast-iedit, #title-wrap, #novel-wrap, #final-style-section, #final-music-section",
    );
    if (editing) {
      scheduleAutoValidate(); // 还在输入：等失焦后再验
      return;
    }
  }
  const out = state.currentOut;
  if (!out) {
    if (!auto) toast("请先打开项目", true);
    return;
  }
  if (!state.shotSb) {
    try { await loadShotSb(out); } catch { return; }
  }
  if (!state.shotSb) return;
  // 无分镜文件 / 无镜头：不算失败，清掉「硬校验未通过」
  if (state.shotSb._missing || state.shotSbExists === false) {
    clearShotsMsg();
    return;
  }
  const sb = snapshotShotSbFromUi();
  state.shotSb = sb;
  const shots = Array.isArray(sb.shots) ? sb.shots : [];
  if (!shots.length) {
    clearShotsMsg();
    return;
  }
  const novel = novelValue();
  try {
    const data = await api("/api/storyboard/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: state.shotSbPath || projectStoryboardPath(),
        strict: true,
        storyboard: sb,
        novel_text: novel,
      }),
    });
    if (seq !== validateSeq) return;
    if (data.skipped) {
      clearShotsMsg();
      return;
    }
    if (data.ok === false || data.strict_error || (data.violations || []).length) {
      setShotsMsg(formatValidateFailure(data), { bad: true });
      return;
    }
    clearShotsMsg();
  } catch (err) {
    if (seq !== validateSeq) return;
    const payload = err.data || { strict_error: err.message, violations: [] };
    setShotsMsg(formatValidateFailure(payload, err.message), { bad: true });
  }
}

async function addShotOnGrid() {
  const out = state.currentOut;
  if (!out) return;
  const sb = await loadShotSb(out);
  sb.shots.push({
    id: nextShotId(sb.shots),
    characters: [],
    location: "",
    image_prompt: "",
    motion_prompt: "",
    speaker: "",
    dialogue: "",
    emotion: "calm",
    seed: randomSeed(),
  });
  markShotSbDirty();
  await renderShots();
}

async function deleteShotOnGrid(shotId) {
  if (!state.shotSb) return;
  if (!confirm(`删除镜头 ${shotId}？未保存的修改会一起丢掉该镜。`)) return;
  state.shotSb.shots = state.shotSb.shots.filter((s) => s.id !== shotId);
  const card = shotCards.get(shotId);
  if (card) {
    card.remove();
    shotCards.delete(shotId);
  }
  markShotSbDirty();
  await renderShots();
}

function mediaSig(entry) {
  return entry ? `${entry.mtime}:${entry.size}` : "";
}

/** 产物 URL 带 mtime，文件一变就换地址，避免 video/img 吃到旧缓存 */
function mediaUrl(entry) {
  if (!entry?.url) return "";
  const t = entry.mtime;
  if (t == null || t === "") return entry.url;
  const url = String(entry.url);
  return `${url}${url.includes("?") ? "&" : "?"}t=${t}`;
}

function shotDurationSec(shot) {
  const fps = Math.max(1, Number(state.fps) || 24);
  if (shot?.video_frames != null && shot.video_frames !== "") {
    return Math.max(0, Number(shot.video_frames) / fps);
  }
  if (shot?.audio_duration != null && shot.audio_duration !== "") {
    return Math.max(0, Number(shot.audio_duration));
  }
  return null;
}

function fillShotDuration(card, shot) {
  const el = $(".shot-dur", card);
  if (!el) return;
  const sec = shotDurationSec(shot);
  if (sec == null || !Number.isFinite(sec)) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = `${sec.toFixed(2)}秒`;
  el.title = shot?.video_frames != null
    ? `${shot.video_frames} 帧 @ ${state.fps || 24}fps`
    : "按配音时长";
}

function bindShotAudio(root) {
  const audio = $("audio", root);
  const playBtn = $(".shot-audio-play", root);
  const seek = $(".shot-audio-seek", root);
  if (!audio || !playBtn || !seek || root._audioBound) return;
  root._audioBound = true;
  let seeking = false;
  const syncSeek = () => {
    const dur = Number.isFinite(audio.duration) ? audio.duration : 0;
    if (!seeking) {
      seek.max = String(dur || 0);
      seek.value = String(audio.currentTime || 0);
    }
  };
  playBtn.addEventListener("click", () => {
    if (audio.paused) {
      // 同时只播一个镜头配音；也停掉配乐
      pauseOtherShotAudio(audio);
      const bgm = $("#music-player");
      if (bgm && !bgm.paused) bgm.pause();
      const bgmBtn = $("#music-play");
      if (bgmBtn) bgmBtn.textContent = "▶";
      audio.play().catch(() => {});
      playBtn.textContent = "❚❚";
    } else {
      audio.pause();
      playBtn.textContent = "▶";
    }
  });
  audio.addEventListener("loadedmetadata", syncSeek);
  audio.addEventListener("timeupdate", syncSeek);
  audio.addEventListener("ended", () => {
    playBtn.textContent = "▶";
    syncSeek();
  });
  audio.addEventListener("pause", () => {
    if (!seeking) playBtn.textContent = "▶";
  });
  seek.addEventListener("pointerdown", () => { seeking = true; });
  seek.addEventListener("pointerup", () => { seeking = false; });
  seek.addEventListener("input", () => {
    audio.currentTime = Number(seek.value) || 0;
  });
}

function fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function genderLabel(g) {
  return { female: "女", male: "男" }[String(g || "").toLowerCase()] || (g || "");
}

function castInitial(ch) {
  const raw = String(ch.name || ch.id || "?").trim();
  return raw ? raw.slice(0, 1) : "?";
}

function castAvatarUrl(ch, info = {}) {
  // 只认 /api/media 扫到的真实文件；删了就不返回 URL，避免浏览器缓存假图
  const id = String(ch.id || "").trim();
  if (!id) return "";
  // info 没带 portraits 时回退 state（加角色局部刷新不会丢已有头像）
  const portraits = info.portraits || state.portraits || {};
  const entry = portraits[id];
  if (!entry) return "";
  const url = typeof entry === "string" ? entry : String(entry.url || "");
  if (!url) return "";
  const mtime = typeof entry === "object" ? entry.mtime : null;
  if (mtime == null || mtime === "") return url;
  return `${url}${url.includes("?") ? "&" : "?"}t=${mtime}`;
}

function ensureCastAvatarFallback(box, ch) {
  let fb = $(".cast-avatar-fallback", box);
  if (!fb) {
    fb = document.createElement("span");
    fb.className = "cast-avatar-fallback";
    box.appendChild(fb);
  }
  fb.textContent = castInitial(ch);
  return fb;
}

function syncCastAvatar(card, info = {}) {
  const ch = card?._ch;
  const box = $(".cast-avatar", card);
  if (!ch || !box) return;
  const url = castAvatarUrl(ch, info);
  const fb = ensureCastAvatarFallback(box, ch);
  let img = $(".cast-avatar-img", box);
  if (!url) {
    if (img) img.remove();
    fb.hidden = false;
    return;
  }
  if (!img) {
    img = document.createElement("img");
    img.className = "cast-avatar-img";
    img.loading = "lazy";
    img.addEventListener("error", () => {
      img.remove();
      fb.hidden = false;
    });
    box.insertBefore(img, fb);
  }
  img.alt = ch.name || ch.id || "角色";
  if (img.dataset.src !== url) {
    img.dataset.src = url;
    img.src = url;
    img.style.display = "";
    fb.hidden = true;
  }
}

function isCastEditing() {
  const ae = document.activeElement;
  return !!(ae && ae.closest && ae.closest("#final-cast .cast-iedit"));
}

function castListSig(chars) {
  return (chars || []).map((c) => `${c.id || ""}:${c.name || ""}`).join("|");
}

function liveCharacters(info = {}) {
  if (state.shotSb && Array.isArray(state.shotSb.characters)) return state.shotSb.characters;
  return Array.isArray(info.characters) ? info.characters : [];
}

function nextCharId(chars) {
  let max = 0;
  for (const c of chars || []) {
    const m = String(c.id || "").match(/^c(\d+)$/i);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  // 也兼容已有非 cNNN id：另找空位
  const used = new Set((chars || []).map((c) => c.id));
  let n = max + 1;
  while (used.has(`c${String(n).padStart(3, "0")}`)) n += 1;
  return `c${String(n).padStart(3, "0")}`;
}

function voiceOptionsFor(gender) {
  const g = String(gender || "").toLowerCase();
  return VOICE_OPTIONS[g] || [...VOICE_OPTIONS.female, ...VOICE_OPTIONS.male];
}

function fillCastSelect(input, field, ch) {
  if (!input || input.tagName !== "SELECT") return;
  const cur = String(ch?.[field] || "");
  let opts = [];
  if (field === "gender") {
    opts = GENDER_OPTIONS.map((o) => ({ value: o.value, label: o.label }));
  } else if (field === "voice") {
    opts = voiceOptionsFor(ch?.gender).map((v) => ({ value: v, label: v }));
  }
  input.innerHTML = opts.map((o) =>
    `<option value="${esc(o.value)}"${o.value === cur ? " selected" : ""}>${esc(o.label)}</option>`
  ).join("");
  if (cur && !opts.some((o) => o.value === cur)) {
    input.insertAdjacentHTML("afterbegin", `<option value="${esc(cur)}" selected>${esc(cur)}</option>`);
  }
}

function castFieldShown(ch, field) {
  if (field === "id") {
    const t = String(ch.id || "").trim();
    return { text: t || "id", empty: !t };
  }
  if (field === "name") {
    const t = String(ch.name || "").trim();
    return { text: t || "姓名", empty: !t };
  }
  if (field === "gender") {
    const t = genderLabel(ch.gender) || "性别";
    return { text: t, empty: !ch.gender };
  }
  if (field === "voice") {
    const t = ch.voice ? `配音 ${ch.voice}` : "配音";
    return { text: t, empty: !ch.voice };
  }
  if (field === "appearance") {
    const t = String(ch.appearance || "").trim();
    return { text: t || "外貌未写", empty: !t };
  }
  if (field === "seed") {
    const t = ch.seed == null || ch.seed === "" ? "" : String(ch.seed);
    return { text: t || "可选", empty: !t };
  }
  return { text: "", empty: true };
}

function syncCastDisplays(card) {
  const ch = card?._ch;
  if (!ch) return;
  $$(".cast-iedit", card).forEach((wrap) => {
    if (isCastFieldEditing(wrap)) return;
    const display = $(".cast-iedit-display", wrap);
    if (!display) return;
    const shown = castFieldShown(ch, wrap.dataset.f);
    display.textContent = shown.text;
    display.classList.toggle("empty", shown.empty);
  });
  const initial = $(".cast-avatar-fallback", card);
  if (initial) initial.textContent = castInitial(ch);
}

function syncCastCardMedia(card, info = {}) {
  syncCastDisplays(card);
  syncCastAvatar(card, info);
}

function isCastFieldEditing(wrap) {
  const input = $(".cast-iedit-input", wrap);
  return !!(input && !input.hidden);
}

function beginEditCastField(wrap, card) {
  const display = $(".cast-iedit-display", wrap);
  const input = $(".cast-iedit-input", wrap);
  const ch = card?._ch;
  if (!display || !input || !ch || isCastFieldEditing(wrap)) return;
  const key = wrap.dataset.f;
  if (wrap.dataset.kind === "select") fillCastSelect(input, key, ch);
  else if (key === "seed") input.value = ch.seed == null ? "" : String(ch.seed);
  else input.value = String(ch[key] || "");
  input.hidden = false;
  display.hidden = true;
  input.focus();
  if (input.tagName === "SELECT") {
    try { input.showPicker?.(); } catch {}
  } else if (input.tagName === "INPUT") {
    try { input.select(); } catch {}
  } else {
    const len = input.value.length;
    try { input.setSelectionRange(len, len); } catch {}
  }
}

async function persistCastChange(_key) {
  // 只标脏；真正写盘交给「保存分镜」
  if (!state.shotSb || !state.shotSbPath) {
    const out = state.currentOut;
    if (out) await loadShotSb(out);
  }
  markShotSbDirty();
  refreshShotCharacterSelects();
}

async function commitCastField(wrap, card, { cancel = false } = {}) {
  const display = $(".cast-iedit-display", wrap);
  const input = $(".cast-iedit-input", wrap);
  const ch = card?._ch;
  if (!display || !input || !ch || !isCastFieldEditing(wrap)) return;
  const key = wrap.dataset.f;
  const prev = key === "seed"
    ? (ch.seed == null ? "" : String(ch.seed))
    : String(ch[key] || "");
  if (cancel) {
    if (wrap.dataset.kind === "select") fillCastSelect(input, key, ch);
    else input.value = prev;
  }
  let next = input.value;
  if (key !== "appearance") next = next.trim();
  input.hidden = true;
  display.hidden = false;

  let changed = false;
  if (key === "seed") {
    const n = next === "" ? null : parseInt(next, 10);
    const normalized = Number.isFinite(n) ? n : null;
    const prevN = ch.seed == null ? null : Number(ch.seed);
    if (normalized !== prevN) {
      if (normalized == null) delete ch.seed;
      else ch.seed = normalized;
      changed = true;
    }
  } else if (prev !== next) {
    if (key === "id") {
      const used = storyCharacters().some((c) => c !== ch && c.id === next);
      if (!next) {
        toast("角色 id 不能为空", true);
        syncCastDisplays(card);
        return;
      }
      if (used) {
        toast(`角色 id「${next}」已存在`, true);
        syncCastDisplays(card);
        return;
      }
      const oldId = ch.id;
      ch.id = next;
      card.dataset.id = next;
      // 同步镜头里引用的旧 id
      if (oldId && oldId !== next && state.shotSb?.shots) {
        for (const shot of state.shotSb.shots) {
          if (!Array.isArray(shot.characters)) continue;
          shot.characters = shot.characters.map((id) => (id === oldId ? next : id));
        }
        refreshShotCharacterSelects();
      }
      changed = true;
    } else {
      ch[key] = next;
      changed = true;
    }
  }
  // 换性别时配音若不在可选列表里，清到默认
  if (key === "gender" && changed) {
    const voices = voiceOptionsFor(ch.gender);
    if (ch.voice && !voices.includes(ch.voice)) {
      ch.voice = voices[0] || "";
    } else if (!ch.voice) {
      ch.voice = voices[0] || "";
    }
  }
  syncCastDisplays(card);
  if (!changed) return;
  await persistCastChange(key);
}

function rerollCastSeed(card) {
  const ch = card?._ch;
  if (!ch) return;
  const wrap = $(".cast-iedit[data-f=seed]", card);
  if (wrap && isCastFieldEditing(wrap)) {
    const input = $(".cast-iedit-input", wrap);
    const display = $(".cast-iedit-display", wrap);
    if (input) input.hidden = true;
    if (display) display.hidden = false;
  }
  ch.seed = randomSeed();
  syncCastDisplays(card);
  markShotSbDirty();
}

function bindCastCard(card, info = {}) {
  if (card._bound) return;
  card._bound = true;
  $$(".cast-iedit", card).forEach((wrap) => {
    const display = $(".cast-iedit-display", wrap);
    const input = $(".cast-iedit-input", wrap);
    if (!display || !input) return;
    display.addEventListener("click", () => beginEditCastField(wrap, card));
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        commitCastField(wrap, card, { cancel: true });
      } else if (ev.key === "Enter" && input.tagName !== "TEXTAREA") {
        ev.preventDefault();
        commitCastField(wrap, card);
      } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && input.tagName === "TEXTAREA") {
        ev.preventDefault();
        commitCastField(wrap, card);
      }
    });
    input.addEventListener("change", () => {
      if (wrap.dataset.kind === "select") commitCastField(wrap, card);
    });
    input.addEventListener("blur", () => commitCastField(wrap, card));
  });
  const reroll = $(".seed-reroll[data-seed-for=cast]", card);
  if (reroll) {
    reroll.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      rerollCastSeed(card);
    });
  }
  $(".cast-portrait-rerun", card)?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    const ch = card._ch;
    if (!ch?.id) return;
    openRerunDialog({ kind: "portrait", id: ch.id });
  });
  $(".cast-del", card)?.addEventListener("click", async () => {
    const ch = card._ch;
    if (!ch || !state.shotSb) return;
    const label = ch.name || ch.id || "角色";
    if (!confirm(`删除角色「${label}」？镜头里引用的 id 不会自动改。`)) return;
    state.shotSb.characters = (state.shotSb.characters || []).filter((c) => c !== ch && c.id !== ch.id);
    const castEl = $("#final-cast");
    if (castEl) delete castEl.dataset.sig;
    renderFinalMeta({ ...info, characters: state.shotSb.characters });
    await persistCastChange("delete");
  });
}

function castCardHtml(ch, info = {}) {
  const nameShown = castFieldShown(ch, "name");
  const genderShown = castFieldShown(ch, "gender");
  const voiceShown = castFieldShown(ch, "voice");
  const appShown = castFieldShown(ch, "appearance");
  const avatar = castAvatarUrl(ch, info);
  const initial = esc(castInitial(ch));
  const name = esc(ch.name || ch.id || "角色");
  // 头像只在产物清单里有文件时才挂 img；加载失败则退回首字
  const avatarHtml = avatar
    ? `<img class="cast-avatar-img" src="${esc(avatar)}" data-src="${esc(avatar)}" alt="${name}" loading="lazy" onerror="const box=this.parentElement;const fb=box&&box.querySelector('.cast-avatar-fallback');this.remove();if(fb)fb.hidden=false">
       <span class="cast-avatar-fallback" hidden>${initial}</span>`
    : `<span class="cast-avatar-fallback">${initial}</span>`;
  const idShown = castFieldShown(ch, "id");
  const seedShown = castFieldShown(ch, "seed");
  return `<div class="cast-card" data-id="${esc(ch.id || "")}">
    <button type="button" class="ghost cast-del" title="删除角色">×</button>
    <div class="cast-avatar-col">
      <div class="cast-avatar">${avatarHtml}</div>
      <button type="button" class="ghost cast-portrait-rerun" title="单跑肖像">↻ 肖像</button>
    </div>
    <div class="cast-body">
      <div class="cast-iedit" data-f="id" data-placeholder="id">
        <div class="cast-iedit-display cast-id-line${idShown.empty ? " empty" : ""}">${esc(idShown.text)}</div>
        <input class="cast-iedit-input" type="text" hidden placeholder="拼音 id，如 chenyuan">
      </div>
      <div class="cast-iedit" data-f="name" data-placeholder="姓名">
        <div class="cast-iedit-display cast-name${nameShown.empty ? " empty" : ""}">${esc(nameShown.text)}</div>
        <input class="cast-iedit-input" type="text" hidden>
      </div>
      <div class="cast-meta-row">
        <div class="cast-iedit" data-f="gender" data-kind="select" data-placeholder="性别">
          <div class="cast-iedit-display cast-role-line${genderShown.empty ? " empty" : ""}">${esc(genderShown.text)}</div>
          <select class="cast-iedit-input" hidden></select>
        </div>
        <div class="cast-iedit" data-f="voice" data-kind="select" data-placeholder="配音">
          <div class="cast-iedit-display cast-role-line${voiceShown.empty ? " empty" : ""}">${esc(voiceShown.text)}</div>
          <select class="cast-iedit-input" hidden></select>
        </div>
      </div>
      <div class="cast-iedit" data-f="appearance" data-multiline data-placeholder="外貌">
        <div class="cast-iedit-display cast-appearance${appShown.empty ? " empty" : ""}">${esc(appShown.text)}</div>
        <textarea class="cast-iedit-input" hidden rows="3"></textarea>
      </div>
      <div class="cast-iedit" data-f="seed" data-placeholder="可选">
        <div class="seed-row">
          <div class="shot-field-label">seed</div>
          <button type="button" class="ghost seed-reroll" title="生成新随机数" data-seed-for="cast">🎲</button>
        </div>
        <div class="cast-iedit-display cast-id-line${seedShown.empty ? " empty" : ""}">${esc(seedShown.text)}</div>
        <input class="cast-iedit-input" type="number" hidden>
      </div>
    </div>
  </div>`;
}

function placeCastAddCard(castEl) {
  if (!castEl) return;
  let add = $(".cast-add-card", castEl);
  if (!add) {
    add = document.createElement("button");
    add.type = "button";
    add.className = "cast-card cast-add-card";
    add.title = "添加角色";
    add.setAttribute("aria-label", "添加角色");
    add.innerHTML = `<span class="shot-add-plus" aria-hidden="true">＋</span>`;
    add.addEventListener("click", () => { addCastCharacter(); });
  }
  if (castEl.lastElementChild !== add) castEl.appendChild(add);
}

function rebuildCastCards(chars, info = {}) {
  const castEl = $("#final-cast");
  if (!castEl) return;
  castEl.innerHTML = chars.map((ch) => castCardHtml(ch, info)).join("");
  $$(".cast-card:not(.cast-add-card)", castEl).forEach((card, i) => {
    card._ch = chars[i];
    bindCastCard(card, info);
  });
  placeCastAddCard(castEl);
  castEl.dataset.sig = castListSig(chars);
}

async function addCastCharacter() {
  const out = state.currentOut;
  if (!out) {
    toast("请先打开项目", true);
    return;
  }
  const sb = await loadShotSb(out);
  const id = nextCharId(sb.characters);
  const ch = {
    id,
    name: "新角色",
    gender: "female",
    voice: VOICE_OPTIONS.female[0],
    appearance: "",
    seed: randomSeed(),
  };
  sb.characters.push(ch);
  const castEl = $("#final-cast");
  const info = { portraits: state.portraits || {}, characters: sb.characters, out, shots: sb.shots };
  // 只插新卡，不整表 innerHTML，避免已有头像闪没
  if (castEl && !isCastEditing()) {
    const hint = castEl.querySelector(":scope > .hint");
    if (hint) hint.remove();
    const wrap = document.createElement("div");
    wrap.innerHTML = castCardHtml(ch, info);
    const card = wrap.firstElementChild;
    if (card) {
      card._ch = ch;
      bindCastCard(card, info);
      const addBtn = $(".cast-add-card", castEl);
      if (addBtn) castEl.insertBefore(card, addBtn);
      else castEl.appendChild(card);
    }
    placeCastAddCard(castEl);
    castEl.dataset.sig = castListSig(sb.characters);
  } else {
    if (castEl) delete castEl.dataset.sig;
    renderFinalMeta(info);
  }
  await persistCastChange("add");
  // 自动进入姓名编辑
  const card = castEl?.querySelector(`.cast-card[data-id="${CSS.escape(id)}"]`);
  const nameWrap = card && $(".cast-iedit[data-f=name]", card);
  if (nameWrap) beginEditCastField(nameWrap, card);
}

function refreshShotCharacterSelects() {
  for (const [, card] of shotCards) {
    const wrap = $(".shot-iedit[data-f=characters]", card);
    if (!wrap || isShotFieldEditing(wrap)) continue;
    fillShotInlineField(wrap, card._shot);
  }
}

function renderFinalMeta(info = {}) {
  const castEl = $("#final-cast");
  // 片名 / 正文默认展示态；编辑中不覆盖。无分镜时片名退回项目名，风格/配乐清空。
  if (!isTitleEditing()) {
    if (info.missing) {
      setTitleDisplay(state.project || "");
    } else {
      const sbTitle = String(info.title || state.shotSb?.title || "").trim();
      if (sbTitle) {
        const cur = titleValue();
        if (!cur || cur === state.project) setTitleDisplay(sbTitle);
      }
    }
  }
  const novel = String(info.novelText || "").trim();
  if (novel && !isNovelEditing() && !novelValue()) setNovelDisplay(novel);

  const chars = liveCharacters(info);
  if (castEl) {
    if (isCastEditing()) {
      // 编辑中不重建，避免打断输入
    } else if (castEl.dataset.sig === castListSig(chars) && castEl.querySelector(".cast-add-card")) {
      // 名单未变：同步文案；头像按产物清单增删/换 mtime 刷新
      $$(".cast-card:not(.cast-add-card)", castEl).forEach((card, i) => {
        if (chars[i]) {
          card._ch = chars[i];
          syncCastCardMedia(card, info);
        }
      });
      placeCastAddCard(castEl);
    } else {
      rebuildCastCards(chars, info);
    }
  }

  if (!isStyleEditing()) {
    setStyleDisplay(String(info.style || state.shotSb?.style || "").trim());
  }
  if (!isMusicEditing()) {
    setMusicDisplay(String(info.music_prompt || state.shotSb?.music_prompt || "").trim());
  }
  const musicSeed = info.missing ? null : (info.music_seed ?? state.shotSb?.music_seed);
  setMusicSeedDisplay(musicSeed);
}

function pauseOtherShotAudio(except) {
  $$("#shots-grid audio").forEach((a) => {
    if (a !== except && !a.paused) a.pause();
  });
  $$(".shot-audio-play").forEach((b) => { b.textContent = "▶"; });
}

function syncMusicPlayer(musicEntry) {
  const btn = $("#music-play");
  const audio = $("#music-player");
  if (!btn || !audio) return;
  if (!musicEntry?.url) {
    btn.hidden = true;
    if (!audio.paused) audio.pause();
    audio.removeAttribute("src");
    delete audio.dataset.sig;
    btn.textContent = "▶";
    return;
  }
  btn.hidden = false;
  const url = mediaUrl(musicEntry);
  const sig = mediaSig(musicEntry);
  if (audio.dataset.sig !== sig || audio.getAttribute("src") !== url) {
    const wasPlaying = !audio.paused;
    audio.src = url;
    audio.dataset.sig = sig;
    audio.load();
    if (wasPlaying) audio.play().catch(() => {});
  }
}

function bindMusicPlayer() {
  const btn = $("#music-play");
  const audio = $("#music-player");
  if (!btn || !audio || btn._bound) return;
  btn._bound = true;
  btn.addEventListener("click", () => {
    if (audio.paused) {
      pauseOtherShotAudio(null);
      const final = $("#final-player");
      if (final && !final.paused) final.pause();
      audio.play().catch(() => {});
      btn.textContent = "❚❚";
    } else {
      audio.pause();
      btn.textContent = "▶";
    }
  });
  audio.addEventListener("ended", () => { btn.textContent = "▶"; });
  audio.addEventListener("pause", () => { btn.textContent = "▶"; });
  audio.addEventListener("play", () => { btn.textContent = "❚❚"; });
}

function syncFinalPlayerLayout(player) {
  /** 左列宽 = 右栏高度 × 视频宽高比；右栏 flex 吃剩，互不重叠。 */
  if (!player) return;
  const poster = player.closest?.(".movie-poster");
  const card = poster?.closest?.(".movie-card");
  const info = card?.querySelector?.(".movie-info");
  if (!poster) return;

  const layout = () => {
    poster.style.removeProperty("aspect-ratio");
    const has = !!(player.getAttribute("src") && !player.hidden);
    const vw = player.videoWidth || 0;
    const vh = player.videoHeight || 0;
    if (!has || !vw || !vh) {
      poster.classList.remove("has-video");
      poster.style.removeProperty("width");
      return;
    }
    poster.classList.add("has-video");
    if (window.matchMedia("(max-width: 860px)").matches) {
      poster.style.removeProperty("width");
      return;
    }
    const h = Math.max(info?.offsetHeight || 0, card?.clientHeight || 0);
    if (h < 40) return;
    const maxW = Math.min(Math.floor(window.innerWidth * 0.45), 560);
    const w = Math.max(120, Math.min(maxW, Math.round((h * vw) / vh)));
    poster.style.width = `${w}px`;
  };

  layout();
  if (!poster._finalLayoutBound) {
    poster._finalLayoutBound = true;
    if (info && typeof ResizeObserver !== "undefined") {
      poster._finalLayoutObs = new ResizeObserver(() => layout());
      poster._finalLayoutObs.observe(info);
    }
    window.addEventListener("resize", layout);
    player.addEventListener("loadedmetadata", layout);
  }
}

function renderFinalVideo(final, info = {}) {
  const wrap = $("#final-wrap");
  const badge = $("#final-badge");
  const player = $("#final-player");
  const link = $("#final-download");
  const meta = $("#final-meta");
  const empty = $("#final-empty");
  if (!wrap || !player) return;
  wrap.hidden = false;
  const has = !!(final && final.url);
  if (badge) badge.hidden = !has;
  if (empty) empty.hidden = has;
  player.hidden = !has;
  renderFinalMeta(info);
  if (!has) {
    if (player.src) {
      player.removeAttribute("src");
      player.load();
    }
    syncFinalPlayerLayout(player);
    if (link) link.hidden = true;
    if (meta) meta.textContent = "";
    return;
  }
  const url = mediaUrl(final);
  const sig = mediaSig(final);
  // sig 变了、或上次加载失败（黑屏/error）：强制换源重载
  const broken = !!(player.error) || (player.readyState === 0 && player.networkState === HTMLMediaElement.NETWORK_NO_SOURCE);
  if (player.dataset.sig !== sig || player.getAttribute("src") !== url || broken) {
    const paused = player.paused;
    const t = player.currentTime || 0;
    player.src = url;
    player.dataset.sig = sig;
    player.onerror = () => {
      // 下一轮自动刷新再试（assemble 刚写完时偶发读到半截）
      delete player.dataset.sig;
    };
    player.load();
    // 文件更新时尽量停在原位置；首次载入不自动播
    if (!paused && t > 0) {
      player.addEventListener("loadedmetadata", () => {
        try { player.currentTime = Math.min(t, player.duration || t); } catch {}
      }, { once: true });
    }
  }
  syncFinalPlayerLayout(player);
  // 右栏内容可能稍后撑开高度，再跟一帧
  requestAnimationFrame(() => syncFinalPlayerLayout(player));
  if (link) {
    const title = (titleValue() || info.title || state.project || "final").trim() || "final";
    link.hidden = false;
    link.href = url;
    link.setAttribute("download", `${title}.mp4`);
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
  if (shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  shotsTimer = setInterval(renderShots, 5000);
}

async function renderShots() {
  const out = state.currentOut;
  if (!out || !state.project) return;
  const seq = ++renderShotsSeq;
  const shown = (state.projects.find((p) => p.name === state.project) || {}).display_path || out;
  $("#shots-out-label").textContent = shown;
  // 换了输出目录：旧卡片全部清掉，避免闪现上一个片的产物
  if (out !== shotsLastOut) {
    clearShotCards();
    shotsLastOut = out;
    mediaCache = null;
  }
  // 1) 拉产物清单
  let media = null;
  try {
    // 只编码非路径分隔符，保留 /；绝对路径也可
    media = await api(`/api/media?out=${out.split("/").map(encodeURIComponent).join("/")}`);
    mediaCache = media;
  } catch (err) {
    media = mediaCache;
  }
  if (seq !== renderShotsSeq) return;
  if (!media && shotCards.size) return; // 产物清单暂时拿不到：保持现状，别清空预览
  // 2) 分镜：未脏时跟磁盘走（含「文件被删 → 列表清空」）
  const sb = await loadShotSb(out, { force: !state.shotSbDirty });
  if (seq !== renderShotsSeq) return;
  // 正文跟 novel.txt（编辑中不覆盖）
  if (!isNovelEditing() && !state.shotSbDirty) {
    await loadNovel({ quiet: true });
  }
  if (seq !== renderShotsSeq) return;
  const shots = Array.isArray(sb.shots) ? sb.shots.filter((s) => s && s.id) : [];
  const reuseMap = {};
  for (const s of shots) if (s.reuse_frame) reuseMap[s.id] = s.reuse_frame;
  const missingSb = !!sb._missing || state.shotSbExists === false;
  const sbMeta = {
    title: missingSb
      ? (state.project || "").trim()
      : (sb.title || titleValue() || state.project || "").trim(),
    characters: Array.isArray(sb.characters) ? sb.characters : [],
    style: sb.style || "",
    music_prompt: sb.music_prompt || "",
    music_seed: sb.music_seed,
    novelText: novelValue(),
    missing: missingSb,
  };
  const portraits = {};
  for (const c of media?.characters || []) {
    if (c?.id && c?.url) portraits[c.id] = { url: c.url, mtime: c.mtime };
  }
  state.portraits = portraits;
  const framesById = {};
  for (const f of media?.frames || []) {
    if (f?.id && f?.url) framesById[f.id] = f.url;
  }
  renderFinalVideo(media?.final || null, {
    ...sbMeta,
    out,
    shots,
    portraits,
    framesById,
  });
  syncMusicPlayer(media?.music || null);
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
  // 4) 镜头列表只认分镜文件；没有分镜就不靠 frames/clips 凑卡
  const grid = $("#shots-grid");
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
  if (!desired.length) {
    const tip = document.createElement("div");
    tip.className = "empty";
    tip.textContent = sbMeta.missing
      ? "还没有分镜文件。可用上方「从小说生成分镜」，或点 ＋ 手写。"
      : "分镜里还没有镜头。点 ＋ 添加。";
    grid.insertBefore(tip, grid.firstChild);
  }
  placeShotAddCard(grid);
}

function syncShotPreviewAspect(preview, video) {
  if (!preview || !video) return;
  const apply = () => {
    const w = video.videoWidth || 0;
    const h = video.videoHeight || 0;
    if (w > 0 && h > 0) preview.style.aspectRatio = `${w} / ${h}`;
  };
  if (video.videoWidth) apply();
  else video.addEventListener("loadedmetadata", apply, { once: true });
}

function bindShotPreviewHover(preview) {
  if (!preview || preview._hoverBound) return;
  preview._hoverBound = true;
  preview.addEventListener("mouseenter", () => {
    const video = $(".shot-clip", preview);
    if (!video) return;
    preview.classList.add("is-hover");
    video.muted = true; // 悬停自动播需静音，否则浏览器会拦
    const play = video.play();
    if (play && typeof play.catch === "function") play.catch(() => {});
  });
  preview.addEventListener("mouseleave", () => {
    preview.classList.remove("is-hover");
    const video = $(".shot-clip", preview);
    if (!video) return;
    video.pause();
    try { video.currentTime = 0; } catch {}
  });
}

function renderShotCard(shot, media, reuseMap, progItems) {
  const card = ensureShotCard(shot.id);
  card._shot = shot;
  $(".shot-id", card).textContent = shot.id || "";
  fillShotDuration(card, shot);
  bindShotCardFields(card);
  fillShotCardFields(card, shot);
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
      const frameUrl = frameEntry ? mediaUrl(frameEntry) : "";
      const poster = frameUrl ? ` poster="${esc(frameUrl)}"` : "";
      const still = frameUrl
        ? `<img class="shot-still" src="${esc(frameUrl)}" loading="lazy" alt="">`
        : "";
      // 框以视频比例为准；默认静帧 cover 铺满，悬停播视频
      preview.classList.add("has-clip");
      preview.classList.toggle("no-still", !frameUrl);
      preview.classList.remove("is-hover");
      preview.style.removeProperty("aspect-ratio");
      preview.innerHTML = `${still}<video class="shot-clip" src="${esc(mediaUrl(clipEntry))}"${poster} muted preload="metadata" controls playsinline></video>${
        frameUrl ? `<span class="play-badge" aria-hidden="true">▶</span>` : ""
      }`;
      const clipEl = $(".shot-clip", preview);
      syncShotPreviewAspect(preview, clipEl);
      bindShotPreviewHover(preview);
    } else if (frameEntry) {
      preview.classList.remove("has-clip", "no-still", "is-hover");
      preview.style.removeProperty("aspect-ratio");
      preview.innerHTML = `<img class="shot-still" src="${esc(mediaUrl(frameEntry))}" loading="lazy" alt="">`;
    } else {
      preview.classList.remove("has-clip", "no-still", "is-hover");
      preview.style.removeProperty("aspect-ratio");
      preview.innerHTML = "";
    }
    if (voiceEntry) {
      audioEl.innerHTML = `<div class="shot-audio-bar">
        <button type="button" class="ghost shot-audio-play" title="播放/暂停">▶</button>
        <input type="range" class="shot-audio-seek" min="0" max="0" step="0.01" value="0">
        <audio src="${esc(mediaUrl(voiceEntry))}" preload="metadata"></audio>
      </div>`;
      audioEl._audioBound = false;
      bindShotAudio(audioEl);
    } else {
      audioEl.innerHTML = "";
      audioEl._audioBound = false;
    }
  }
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
  card._hasVoice = !!voiceEntry;
  // 重跑 / 删除只挂一次；经确认框展开前向/后向传递闭包
  if (!card._btns) {
    card._btns = true;
    const actions = $(".shot-actions", card);
    const btnFrame = document.createElement("button");
    btnFrame.textContent = "↻ 静帧";
    btnFrame.title = "单跑本镜静帧（先确认前置/后置）";
    btnFrame.onclick = () => openRerunDialog({ kind: "frame", id: card._shot.id });
    const btnVideo = document.createElement("button");
    btnVideo.textContent = "↻ 视频";
    btnVideo.title = "单跑本镜视频（先确认前置/后置）";
    btnVideo.onclick = () => openRerunDialog({ kind: "clip", id: card._shot.id });
    const btnAudio = document.createElement("button");
    btnAudio.textContent = "↻ 配音";
    btnAudio.title = "单跑本镜对白（先确认前置/后置）";
    btnAudio.onclick = () => openRerunDialog({ kind: "voice", id: card._shot.id });
    card._btnAudio = btnAudio;
    actions.appendChild(btnFrame);
    actions.appendChild(btnVideo);
    actions.appendChild(btnAudio);
    $(".shot-del", card)?.addEventListener("click", () => deleteShotOnGrid(card._shot.id));
  }
  syncShotDialogueUi(card, shot);
}

let rerunPlan = null;

function closeRerunDialog() {
  const dlg = $("#rerun-dialog");
  if (dlg?.open) dlg.close();
  rerunPlan = null;
}

function rerunPostChecks() {
  return $$("#rerun-posts input[type=checkbox]");
}

function syncRerunItemMeta(input) {
  const meta = input.closest(".rerun-item")?.querySelector(".ri-meta");
  if (!meta) return;
  const on = input.checked;
  meta.textContent = on ? "勾选 · 将重做" : "未勾选 · 将过时";
  meta.classList.toggle("stale", !on);
}

/** 后置总复选框：全选为勾上（文案「全不选」），部分勾为半选，一个都没勾则「全选」。 */
function syncRerunPostsAll() {
  const box = $("#rerun-posts-all");
  const wrap = $("#rerun-posts-all-wrap");
  const text = $("#rerun-posts-all-text");
  if (!box || !wrap) return;
  const inputs = rerunPostChecks();
  if (!inputs.length) {
    wrap.hidden = true;
    box.checked = false;
    box.indeterminate = false;
    if (text) text.textContent = "全选";
    return;
  }
  wrap.hidden = false;
  const n = inputs.filter((el) => el.checked).length;
  const all = n === inputs.length;
  box.checked = all;
  box.indeterminate = n > 0 && !all;
  if (text) text.textContent = all ? "全不选" : "全选";
}

function setAllRerunPosts(on) {
  rerunPostChecks().forEach((input) => {
    input.checked = on;
    syncRerunItemMeta(input);
  });
  syncRerunPostsAll();
}

function renderRerunItem(node, { required = false, target = false, checked = false } = {}) {
  const id = `rerun-${node.kind}-${node.id || "x"}`;
  const meta = target
    ? `<div class="ri-meta">将重做</div>`
    : required
      ? `<div class="ri-meta ${node.status === "missing" ? "missing" : "ready"}">${
          node.status === "missing" ? "缺失 · 将生成" : "已有 · 将跳过"
        }</div>`
      : `<div class="ri-meta stale">${checked ? "勾选 · 将重做" : "未勾选 · 将过时"}</div>`;
  const disabled = required || target ? " disabled" : "";
  const isChecked = required || target || checked ? " checked" : "";
  return `<label class="rerun-item${required ? " required" : ""}${target ? " target" : ""}" for="${esc(id)}">
    <input type="checkbox" id="${esc(id)}" data-kind="${esc(node.kind)}" data-id="${esc(node.id || "")}"${isChecked}${disabled}>
    <span><div>${esc(node.label || node.kind)}</div>${meta}</span>
  </label>`;
}

async function openRerunDialog(target) {
  const out = state.currentOut;
  if (!out) {
    toast("请先打开项目", true);
    return;
  }
  const draft = await webDraftForPipeline();
  if (!draft) return;
  const dlg = $("#rerun-dialog");
  const errEl = $("#rerun-error");
  const okBtn = $("#rerun-ok");
  if (!dlg) return;
  errEl.hidden = true;
  errEl.textContent = "";
  okBtn.disabled = true;
  $("#rerun-title").textContent = "分析依赖…";
  $("#rerun-summary").textContent = "";
  $("#rerun-prereqs").innerHTML = "";
  $("#rerun-target").innerHTML = "";
  $("#rerun-posts").innerHTML = "";
  syncRerunPostsAll();
  dlg.showModal();
  try {
    const plan = await api("/api/artifact/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ out, kind: target.kind, id: target.id || "", ...draft }),
    });
    rerunPlan = plan;
    if (!plan.ok) {
      $("#rerun-title").textContent = "无法添加";
      errEl.hidden = false;
      errEl.textContent = plan.error || "规划失败";
      okBtn.disabled = true;
      return;
    }
    $("#rerun-title").textContent = `添加：${plan.target.label}`;
    $("#rerun-summary").textContent = plan.summary || "";
    const preWrap = $("#rerun-prereqs");
    if (!(plan.prereqs || []).length) {
      preWrap.innerHTML = `<div class="rerun-empty">无产物前置</div>`;
    } else {
      preWrap.innerHTML = plan.prereqs.map((p) => renderRerunItem(p, { required: true })).join("");
    }
    $("#rerun-target").innerHTML = renderRerunItem(plan.target, { target: true });
    const postWrap = $("#rerun-posts");
    if (!(plan.posts || []).length) {
      postWrap.innerHTML = `<div class="rerun-empty">无后置影响</div>`;
    } else {
      postWrap.innerHTML = plan.posts
        .map((p) => renderRerunItem(p, { checked: p.suggested !== false }))
        .join("");
      postWrap.querySelectorAll("input[type=checkbox]").forEach((input) => {
        input.addEventListener("change", () => {
          syncRerunItemMeta(input);
          syncRerunPostsAll();
        });
      });
      syncRerunPostsAll();
    }
    okBtn.disabled = false;
  } catch (err) {
    $("#rerun-title").textContent = "无法添加";
    errEl.hidden = false;
    errEl.textContent = err.message || String(err);
    okBtn.disabled = true;
  }
}

async function confirmRerunFromDialog() {
  if (!rerunPlan?.ok || !rerunPlan.target) return;
  const draft = await webDraftForPipeline();
  if (!draft) return;
  const out = state.currentOut;
  const posts = [];
  $$("#rerun-posts input[type=checkbox]").forEach((input) => {
    if (!input.checked || input.disabled) return;
    posts.push({ kind: input.dataset.kind, id: input.dataset.id || "" });
  });
  try {
    const res = await api("/api/queue/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        out,
        kind: rerunPlan.target.kind,
        id: rerunPlan.target.id || "",
        posts,
        ...draft,
      }),
    });
    closeRerunDialog();
    if (res.queue) state.queue = res.queue;
    runQueueSig = "";
    renderRunQueue();
    syncRunControls();
    const nAdd = (res.added || []).length;
    const nUp = (res.upgraded || []).length;
    toast(
      nAdd || nUp
        ? `已入队${nAdd ? ` ${nAdd} 项` : ""}${nUp ? `，升级 ${nUp} 项` : ""}`
        : "已在队列中（未重复添加）"
    );
    await refreshState();
    runQueueSig = "";
    renderRunQueue();
    syncRunControls();
  } catch (err) {
    toast(`添加失败：${err.message}`, true);
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
        refreshStateAndLock();
      }, 1500);
    } catch {}
  };
}

async function loadBudget() {
  try {
    const b = await api("/api/budget");
    if (b.fps) state.fps = Number(b.fps) || 24;
    $("#budget-info").textContent =
      `单镜上限 ${b.frames} 帧 ≈ ${b.seconds}s · 台词额度约 ${b.dialogue_chars} 字/镜（${b.total_gib} GB 内存，${b.resolution}@${b.fps}fps）`;
  } catch {
    $("#budget-info").textContent = "";
  }
}

function bind() {
  $("#start-btn").addEventListener("click", startTask);
  $("#title-display")?.addEventListener("click", beginEditTitle);
  $("#title")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      commitTitle();
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      commitTitle({ cancel: true });
    }
  });
  $("#title")?.addEventListener("blur", () => commitTitle());
  $("#novel-display")?.addEventListener("click", beginEditNovel);
  $("#novel-text")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      commitNovel({ cancel: true });
    } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      commitNovel();
    }
  });
  $("#novel-text")?.addEventListener("blur", () => commitNovel());
  $("#style-display")?.addEventListener("click", beginEditStyle);
  $("#style-text")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      commitStyle({ cancel: true });
    } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      commitStyle();
    }
  });
  $("#style-text")?.addEventListener("blur", () => commitStyle());
  $("#music-display")?.addEventListener("click", beginEditMusic);
  $("#music-text")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      commitMusic({ cancel: true });
    } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      commitMusic();
    }
  });
  $("#music-text")?.addEventListener("blur", () => commitMusic());
  $("#music-seed-display")?.addEventListener("click", beginEditMusicSeed);
  $("#music-seed-input")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      commitMusicSeed({ cancel: true });
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      commitMusicSeed();
    }
  });
  $("#music-seed-input")?.addEventListener("blur", () => commitMusicSeed());
  $("#music-seed-reroll")?.addEventListener("click", rerollMusicSeed);
  bindMusicPlayer();
  $("#music-rerun")?.addEventListener("click", () => openRerunDialog({ kind: "music", id: "" }));
  $("#final-rerun")?.addEventListener("click", () => openRerunDialog({ kind: "final", id: "" }));
  $("#rerun-posts-all")?.addEventListener("click", () => {
    const inputs = rerunPostChecks();
    const next = !(inputs.length > 0 && inputs.every((el) => el.checked));
    // 原生切换在 click 的默认动作里发生，可能盖掉同步写入的 checked。
    // 等这次点击结束后再按「是否已经全选」写入最终状态。
    setTimeout(() => setAllRerunPosts(next), 0);
  });
  $("#rerun-form")?.addEventListener("submit", (ev) => {
    const submitter = ev.submitter;
    const value = submitter?.value || "cancel";
    if (value === "ok") {
      ev.preventDefault();
      confirmRerunFromDialog();
    } else {
      rerunPlan = null;
    }
  });
  $("#shots-save")?.addEventListener("click", saveShotSbFromGrid);
  $("#novel-gen-sb")?.addEventListener("click", generateStoryboardFromNovel);
  $("#shots-revert")?.addEventListener("click", revertShotSbFromGrid);
  $("#shots-reroll-seeds")?.addEventListener("click", rerollAllSeeds);
  $("#task-stop")?.addEventListener("click", stopTask);
  $("#queue-clear")?.addEventListener("click", clearRunQueue);
  $("#tasks-clear").addEventListener("click", clearTasks);
  $("#log-toggle")?.addEventListener("click", toggleLogSection);
  $("#new-project-btn").addEventListener("click", createProject);
  $("#new-project-name").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") createProject();
  });
  $("#setup-retry")?.addEventListener("click", retrySetup);
  $("#setup-install")?.addEventListener("click", retrySetup);
  $("#back-to-projects").addEventListener("click", backToProjects);
}

bind();
renderStagePicks();
loadBudget();
connectEvents();
setInterval(refreshStateAndLock, 4000);

(async function boot() {
  const setup = await refreshSetup();
  await refreshState();
  // 依赖未齐时停在项目选择页，展示安装进度
  if (!setup?.ready) return;
  // 生成中 → 锁定进该项目；否则 ?project= / 上次项目
  if (await ensureLockedOnGeneratingProject()) return;
  const fromQuery = projectFromQuery();
  let last = "";
  try { last = localStorage.getItem("nv_last_project") || ""; } catch {}
  const target =
    (fromQuery && (state.projects || []).find((p) => p.name === fromQuery)) ||
    (state.projects || []).find((p) => p.name === last);
  if (target) await openProject(target.name, target.path);
})();
