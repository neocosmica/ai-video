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
  shotSb: null, // 镜头网格里编辑的分镜（与磁盘同步；脏时不被自动刷新覆盖）
  shotSbPath: "",
  shotSbDirty: false,
  polling: null,
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
  // 路径已固定为项目默认，不再展示文件选择器
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
  $("#task-detail-card").hidden = true;
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
  clearShotSbDirty();
  const castEl = $("#final-cast");
  if (castEl) delete castEl.dataset.sig;
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
  // 有运行中的任务就自动打开详情，避免重开网页后还要再点一次才看得到进度
  if (autoSelectRunning) {
    const running = state.tasks.find((t) => t.status === "running");
    if (running) selectTask(running.id);
  }
}

function backToProjects() {
  state.project = "";
  state.selectedTaskId = null;
  state.shotSb = null;
  state.shotSbPath = "";
  clearShotSbDirty();
  try { localStorage.removeItem("nv_last_project"); } catch {}
  if (state.polling) { clearInterval(state.polling); state.polling = null; }
  if (shotsTimer) { clearInterval(shotsTimer); shotsTimer = null; }
  $("#shots-card").hidden = true;
  $("#produce-card").hidden = true;
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

async function startTask() {
  // 只允许「保存分镜」写盘；有未保存修改时拒绝启动，避免管线读旧文件或旁路回写
  await flushOpenEditsInShotsCard();
  if (hasUnsavedShotsEdits()) {
    toast("有未保存修改，请先点「保存分镜」再生成", true);
    return;
  }
  const payload = {
    title: titleValue(),
    name: state.project,
    novel: projectNovelPath(),
    storyboard: projectStoryboardPath(),
    out: state.currentOut,
    stages: [...state.selectedStages],
    only: $("#only-shots").value.trim(),
    force: $("#force").checked,
  };
  if (!payload.novel) {
    toast("请先保存小说正文（点「保存分镜」）", true);
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
  setLogCollapsed(true);
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
    return `<label class="shot-char-check"><input type="checkbox" value="${esc(id)}"${selected.has(id) ? " checked" : ""}>${esc(label)}</label>`;
  }).join("");
  for (const id of selected) {
    if (known.has(id)) continue;
    html += `<label class="shot-char-check"><input type="checkbox" value="${esc(id)}" checked>${esc(id)}</label>`;
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
  display.textContent = shown.text;
  display.classList.toggle("empty", shown.empty);
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

const SHOT_CARD_UI = 6;

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

async function loadShotSb(out, { force = false } = {}) {
  const path = `${out}/storyboard.json`;
  // 同路径且已有内存副本：直接复用（脏编辑不被磁盘覆盖）
  if (!force && state.shotSb && state.shotSbPath === path) return state.shotSb;
  if (!force && state.shotSbDirty && state.shotSb && state.shotSbPath === path) return state.shotSb;
  try {
    const data = await api("/api/storyboard/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    state.shotSb = data.storyboard || { title: state.project || "", characters: [], shots: [] };
  } catch {
    state.shotSb = {
      title: titleValue() || state.project || "新片",
      style: "写实华语短剧，自然肤色，电影感光线，浅景深",
      music_prompt: "English instrumental BGM, no vocals",
      characters: [],
      shots: [],
    };
  }
  if (!Array.isArray(state.shotSb.shots)) state.shotSb.shots = [];
  if (!Array.isArray(state.shotSb.characters)) state.shotSb.characters = [];
  state.shotSbPath = path;
  state.currentStoryboard = path;
  // 缺 seed 时按管线同款算法补齐；若与磁盘不一致则标脏，提示保存写回
  const filled = await ensureStorySeeds(state.shotSb);
  clearShotSbDirty();
  if (filled) markShotSbDirty();
  setMusicSeedDisplay(state.shotSb.music_seed);
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
    // 与浮层编辑器共用同一份数据
    if (state.sbPath === state.shotSbPath) state.sbData = state.shotSb;
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
  const sb = snapshotShotSbFromUi();
  state.shotSb = sb;
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

function fmtAudioTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

function bindShotAudio(root) {
  const audio = $("audio", root);
  const playBtn = $(".shot-audio-play", root);
  const seek = $(".shot-audio-seek", root);
  const timeEl = $(".shot-audio-time", root);
  if (!audio || !playBtn || !seek || !timeEl || root._audioBound) return;
  root._audioBound = true;
  let seeking = false;
  const syncTime = () => {
    const dur = Number.isFinite(audio.duration) ? audio.duration : 0;
    if (!seeking) {
      seek.max = String(dur || 0);
      seek.value = String(audio.currentTime || 0);
    }
    timeEl.textContent = dur
      ? `${fmtAudioTime(audio.currentTime)} / ${fmtAudioTime(dur)}`
      : fmtAudioTime(audio.currentTime);
  };
  playBtn.addEventListener("click", () => {
    if (audio.paused) {
      // 同时只播一个镜头配音
      $$("audio").forEach((a) => {
        if (a !== audio && !a.paused) a.pause();
      });
      $$(".shot-audio-play").forEach((b) => {
        if (b !== playBtn) b.textContent = "▶";
      });
      audio.play().catch(() => {});
      playBtn.textContent = "❚❚";
    } else {
      audio.pause();
      playBtn.textContent = "▶";
    }
  });
  audio.addEventListener("loadedmetadata", syncTime);
  audio.addEventListener("timeupdate", syncTime);
  audio.addEventListener("ended", () => {
    playBtn.textContent = "▶";
    syncTime();
  });
  audio.addEventListener("pause", () => {
    if (!seeking) playBtn.textContent = "▶";
  });
  seek.addEventListener("pointerdown", () => { seeking = true; });
  seek.addEventListener("pointerup", () => { seeking = false; });
  seek.addEventListener("input", () => {
    const t = Number(seek.value) || 0;
    audio.currentTime = t;
    timeEl.textContent = Number.isFinite(audio.duration)
      ? `${fmtAudioTime(t)} / ${fmtAudioTime(audio.duration)}`
      : fmtAudioTime(t);
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
  const id = String(ch.id || "").trim();
  if (!id) return "";
  const portraits = info.portraits || {};
  if (portraits[id]) return portraits[id];
  // 产物目录约定：work/characters/<id>.png（服务端 media 也会扫这一层）
  const out = String(info.out || state.currentOut || "").replace(/\/+$/, "");
  if (out) return `/media/${out}/work/characters/${encodeURIComponent(id)}.png`;
  // 再回退：该角色出场的第一镜静帧
  const shots = Array.isArray(info.shots) ? info.shots : [];
  const frames = info.framesById || {};
  for (const shot of shots) {
    const list = shot.characters || [];
    if (list.includes(id) && frames[shot.id]) return frames[shot.id];
  }
  return "";
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
  const avatarHtml = avatar
    ? `<img class="cast-avatar-img" src="${esc(avatar)}" alt="${name}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.hidden=false">
       <span class="cast-avatar-fallback" hidden>${initial}</span>`
    : `<span class="cast-avatar-fallback">${initial}</span>`;
  const idShown = castFieldShown(ch, "id");
  const seedShown = castFieldShown(ch, "seed");
  return `<div class="cast-card" data-id="${esc(ch.id || "")}">
    <button type="button" class="ghost cast-del" title="删除角色">×</button>
    <div class="cast-avatar">${avatarHtml}</div>
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

function rebuildCastCards(chars, info = {}) {
  const castEl = $("#final-cast");
  if (!castEl) return;
  if (!chars.length) {
    castEl.innerHTML = `<div class="hint">暂无角色信息，点右上角「＋ 角色」添加</div>`;
    castEl.dataset.sig = "";
    return;
  }
  castEl.innerHTML = chars.map((ch) => castCardHtml(ch, info)).join("");
  const cards = $$(".cast-card", castEl);
  cards.forEach((card, i) => {
    card._ch = chars[i];
    bindCastCard(card, info);
  });
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
  if (castEl) delete castEl.dataset.sig;
  renderFinalMeta({ characters: sb.characters, out, shots: sb.shots });
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
  // 片名 / 正文默认展示态；编辑中不覆盖
  const sbTitle = String(info.title || state.shotSb?.title || "").trim();
  if (sbTitle && !isTitleEditing()) {
    const cur = titleValue();
    if (!cur || cur === state.project) setTitleDisplay(sbTitle);
  }
  const novel = String(info.novelText || "").trim();
  if (novel && !isNovelEditing() && !novelValue()) setNovelDisplay(novel);

  const chars = liveCharacters(info);
  if (castEl) {
    if (isCastEditing()) {
      // 编辑中不重建，避免打断输入
    } else if (castEl.dataset.sig === castListSig(chars) && castEl.querySelector(".cast-card")) {
      // 名单未变：只同步展示文案 / 头像
      $$(".cast-card", castEl).forEach((card, i) => {
        if (chars[i]) {
          card._ch = chars[i];
          syncCastDisplays(card);
        }
      });
    } else {
      rebuildCastCards(chars, info);
    }
  }

  if (!isStyleEditing()) {
    const style = String(info.style || state.shotSb?.style || "").trim();
    if (style || !styleValue()) setStyleDisplay(style);
  }
  if (!isMusicEditing()) {
    const music = String(info.music_prompt || state.shotSb?.music_prompt || "").trim();
    if (music || !musicValue()) setMusicDisplay(music);
  }
  const musicSeed = info.music_seed ?? state.shotSb?.music_seed;
  setMusicSeedDisplay(musicSeed);
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
    if (link) link.hidden = true;
    if (meta) meta.textContent = "";
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
    const title = (titleValue() || info.title || state.project || "final").trim() || "final";
    link.hidden = false;
    link.href = final.url;
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
  // 2) 分镜：网格内编辑用内存副本；脏时不从磁盘覆盖
  const sb = await loadShotSb(out, { force: false });
  if (seq !== renderShotsSeq) return;
  let shots = Array.isArray(sb.shots) ? sb.shots : [];
  const reuseMap = {};
  for (const s of shots) if (s.reuse_frame) reuseMap[s.id] = s.reuse_frame;
  let sbMeta = {
    title: (sb.title || titleValue() || state.project || "").trim(),
    characters: Array.isArray(sb.characters) ? sb.characters : [],
    style: sb.style || styleValue() || "",
    music_prompt: sb.music_prompt || musicValue() || "",
    music_seed: sb.music_seed,
    novelText: novelValue(),
  };
  // 正文优先用成片卡片里的编辑器；空则尝试读项目 novel.txt
  if (!sbMeta.novelText) {
    try {
      const novel = await api("/api/file/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: projectNovelPath(out) }),
      });
      if (novel?.content) sbMeta.novelText = String(novel.content);
    } catch {}
  }
  if (seq !== renderShotsSeq) return;
  const portraits = {};
  for (const c of media?.characters || []) {
    if (c?.id && c?.url) portraits[c.id] = c.url;
  }
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
      grid.innerHTML = `<div class="empty">还没有镜头。可点「＋ 镜头」手写分镜，或启动任务生成产物。</div>`;
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
  card._shot = shot;
  $(".shot-id", card).textContent = shot.id || "";
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
      const poster = frameEntry ? ` poster="${esc(frameEntry.url)}"` : "";
      const preload = frameEntry ? "none" : "metadata"; // 有静帧当海报就不预取，点了播放才加载
      preview.innerHTML = `<video src="${esc(clipEntry.url)}"${poster} muted ${preload} controls></video>`;
    } else if (frameEntry) {
      preview.innerHTML = `<img src="${esc(frameEntry.url)}" loading="lazy" alt="">`;
    } else {
      preview.innerHTML = "";
    }
    if (voiceEntry) {
      audioEl.innerHTML = `<div class="shot-audio-bar">
        <button type="button" class="ghost shot-audio-play" title="播放/暂停">▶</button>
        <input type="range" class="shot-audio-seek" min="0" max="0" step="0.01" value="0">
        <span class="shot-audio-time mono">0:00</span>
        <audio src="${esc(voiceEntry.url)}" preload="metadata"></audio>
      </div>`;
      bindShotAudio(audioEl);
    } else {
      audioEl.innerHTML = "";
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
  // 重跑 / 删除只挂一次；回调读 card._shot，避免闭包过期
  if (!card._btns) {
    card._btns = true;
    const actions = $(".shot-actions", card);
    const btnFrame = document.createElement("button");
    btnFrame.textContent = "↻ 静帧";
    btnFrame.onclick = () => rerunShots([card._shot.id], ["images"]);
    const btnVideo = document.createElement("button");
    btnVideo.textContent = "↻ 视频";
    btnVideo.onclick = () => rerunShots([card._shot.id], card._hasFrame ? ["videos"] : ["images", "videos"]);
    const btnAudio = document.createElement("button");
    btnAudio.textContent = "↻ 配音";
    btnAudio.onclick = () => rerunShots([card._shot.id], ["audio"]);
    actions.appendChild(btnFrame);
    actions.appendChild(btnVideo);
    actions.appendChild(btnAudio);
    $(".shot-del", card)?.addEventListener("click", () => deleteShotOnGrid(card._shot.id));
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
        novel: projectNovelPath(),
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
  const path = projectStoryboardPath();
  const out = state.currentOut;
  if (!path) { toast("先选项目", true); return; }
  const name = state.project || (out ? out.split("/").pop() : "");
  let sb;
  // 网格里已有未保存编辑时，角色编辑器直接接同一份对象
  if (state.shotSb && state.shotSbPath === path) {
    sb = state.shotSb;
  } else {
    try {
      const data = await api("/api/storyboard/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      sb = data.storyboard;
    } catch (err) {
      sb = {
        title: titleValue() || name || "新片",
        style: "写实华语短剧，自然肤色，电影感光线，浅景深",
        music_prompt: "English instrumental BGM, no vocals",
        characters: [],
        shots: [],
      };
    }
  }
  state.sbData = sb;
  state.sbPath = path;
  state.shotSb = sb;
  state.shotSbPath = path;
  $("#sb-editor-card").hidden = false;
  $("#sb-path-label").textContent = path;
  renderStoryboardEditor();
}

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
        <label>seed</label>
        <div class="seed-row">
          <input data-f="seed" type="number" value="${esc(ch.seed ?? "")}">
          <button type="button" class="ghost seed-reroll" title="生成新随机数">🎲</button>
        </div>
      </div>`;
    $(".sb-del", row).addEventListener("click", () => {
      sb.characters.splice(idx, 1);
      renderStoryboardEditor();
    });
    $(".seed-reroll", row)?.addEventListener("click", () => {
      ch.seed = randomSeed();
      const inp = $("input[data-f=seed]", row);
      if (inp) inp.value = String(ch.seed);
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
      seed: randomSeed(),
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
        <label>seed</label>
        <div class="seed-row">
          <input data-f="seed" type="number" value="${esc(shot.seed ?? "")}">
          <button type="button" class="ghost seed-reroll" title="生成新随机数">🎲</button>
        </div>
      </div>`;
    $(".sb-del", row).addEventListener("click", () => {
      sb.shots.splice(idx, 1);
      renderStoryboardEditor();
    });
    $(".seed-reroll", row)?.addEventListener("click", () => {
      shot.seed = randomSeed();
      const inp = $("input[data-f=seed]", row);
      if (inp) inp.value = String(shot.seed);
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
      seed: randomSeed(),
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
    // 角色编辑器保存后同步回镜头网格
    if (state.sbPath === state.shotSbPath || state.sbPath === `${state.currentOut}/storyboard.json`) {
      state.shotSb = state.sbData;
      state.shotSbPath = state.sbPath;
      clearShotSbDirty();
    }
    toast(`已保存 ${state.sbPath}`);
    await renderShots();
  } catch (err) {
    toast(`保存失败：${err.message}`, true);
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
  $("#sb-save").addEventListener("click", saveStoryboard);
  $("#sb-close").addEventListener("click", () => { $("#sb-editor-card").hidden = true; });
  $("#shots-save")?.addEventListener("click", saveShotSbFromGrid);
  $("#shots-revert")?.addEventListener("click", revertShotSbFromGrid);
  $("#shots-reroll-seeds")?.addEventListener("click", rerollAllSeeds);
  $("#shots-add")?.addEventListener("click", addShotOnGrid);
  $("#shots-chars")?.addEventListener("click", openStoryboardEditor);
  $("#cast-add")?.addEventListener("click", addCastCharacter);
  $("#task-stop").addEventListener("click", stopTask);
  $("#task-rerun-failed").addEventListener("click", rerunFailed);
  $("#tasks-clear").addEventListener("click", clearTasks);
  $("#log-toggle")?.addEventListener("click", toggleLogSection);
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
