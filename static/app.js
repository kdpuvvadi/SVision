// SVision inspection software
// Copyright (C) 2026 KD Puvvadi
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

const state = {
  projects: [],
  project: null,
  catalog: [],
  health: null,
  lastResults: [],
  pendingFiles: [],
  sourceUrl: "",
  selectedId: null,
  settingTab: "name",
  layout: "inspect",
  editorOpen: false,
  flowOpen: false,
  outputId: null,
  resultIndex: 0,
  zoom: { scale: 1, x: 0, y: 0, drag: null },
  drawing: null,
  plcMeasureSeq: null,
  plcBusyUi: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function tools() {
  return state.project?.flow?.tools || [];
}

function selectedTool() {
  return tools().find((t) => t.id === state.selectedId) || null;
}

function specFor(type) {
  return state.catalog.find((item) => item.type === type);
}

function toolTypeName(tool) {
  return specFor(tool?.type)?.title || tool?.type || "Tool";
}

function toolDisplayName(tool) {
  const title = toolTypeName(tool);
  const name = (tool?.name || "").trim();
  if (!name || (name === "Camera" && tool?.type !== "camera")) return title;
  return name;
}

function cameraTool() {
  return tools().find((tool) => tool.type === "camera") || null;
}

function imageInputReady() {
  return !!cameraTool();
}

function cameraIsUpload() {
  const camera = cameraTool();
  return !!camera && (camera.config?.source || "upload") === "upload";
}

function flowPayload() {
  return {
    logic: "all",
    tools: tools().map((tool) => ({
      id: tool.id,
      type: tool.type,
      name: tool.name,
      config: tool.config || {},
    })),
  };
}

async function boot() {
  state.health = await api("/api/health");
  state.catalog = await api("/api/tools");
  const urls = state.health.urls.filter((u) => !u.includes("127.0.0.1"));
  const where = urls.length ? urls[0] : state.health.urls[0];
  const version = (state.health.version || "").trim();
  $("hostInfo").textContent = version
    ? `${where}  ·  v${version}  ·  Copyright (C) 2026 KD Puvvadi`
    : `${where}  ·  Copyright (C) 2026 KD Puvvadi`;
  renderCatalog();
  try {
    state.settings = await api("/api/settings");
  } catch {
    state.settings = { layout: "inspect", flow_open: false };
  }
  await loadProjects();
  bind();
  startPlcLiveFeed();
}

function renderCatalog() {
  const html = state.catalog.map((tool) => {
    const taken = tool.singleton && tools().some((item) => item.type === tool.type);
    return `<button type="button" data-add="${tool.type}" ${taken ? "disabled" : ""}>${escapeHtml(tool.title)}</button>`;
  }).join("");
  if ($("toolButtons")) $("toolButtons").innerHTML = html;
  if ($("sceneTools")) $("sceneTools").innerHTML = html;
}

async function loadProjects(selectId) {
  state.projects = await api("/api/projects");
  if (!state.projects.length) {
    const created = await api("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Scene 0" }),
    });
    state.projects = [created];
    selectId = created.id;
  }
  const sel = $("projectSelect");
  sel.innerHTML = state.projects.map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`).join("");
  const id = selectId || state.project?.id || state.projects[0].id;
  sel.value = id;
  await openProject(id);
}

async function openProject(id) {
  state.project = await api(`/api/projects/${id}`);
  if (!state.project.flow) state.project.flow = { logic: "all", tools: [] };
  if (!tools().some((t) => t.id === state.selectedId)) state.selectedId = tools()[0]?.id || null;
  renderFlow();
  renderCatalog();
  renderEditor();
  updateImageInput();
  await restoreCurrentImage();
}

function renderFlow() {
  const list = tools();
  const editing = state.layout === "edit" && state.flowOpen;
  $("editFlowBtn").hidden = state.layout !== "edit" || editing;
  $("flowActions").hidden = !editing;
  $("flowEditBox").hidden = !editing;
  document.body.classList.toggle("flow-open", editing);
  $("flowList").innerHTML = list.length
    ? list.map((tool, i) => `
        <li class="${tool.id === state.selectedId || tool.id === state.outputId ? "selected" : ""}" data-id="${tool.id}">
          <div>
            <b>${i + 1}. ${escapeHtml(toolDisplayName(tool))}</b>
            <span>${escapeHtml(specFor(tool.type)?.title || tool.type)}</span>
          </div>
          ${editing ? `<div class="flow-actions">
            <button type="button" data-up="${tool.id}" ${i === 0 ? "disabled" : ""}>Up</button>
            <button type="button" data-down="${tool.id}" ${i === list.length - 1 ? "disabled" : ""}>Down</button>
            <button type="button" data-remove="${tool.id}">Remove</button>
          </div>` : ""}
        </li>`).join("")
    : `<li class="off"><b>No tools</b></li>`;
  renderSceneSettings();
}

function renderEditor() {
  const list = tools();
  $("editFlowList").innerHTML = list.length
    ? list.map((tool, i) => `
        <li class="${tool.id === state.selectedId ? "selected" : ""}" data-id="${tool.id}">
          <b>${i + 1}. ${escapeHtml(toolDisplayName(tool))}</b>
          <span>${escapeHtml(specFor(tool.type)?.title || tool.type)}</span>
          <div class="flow-actions">
            <button type="button" data-up="${tool.id}" ${i === 0 ? "disabled" : ""}>Up</button>
            <button type="button" data-down="${tool.id}" ${i === list.length - 1 ? "disabled" : ""}>Down</button>
            <button type="button" data-remove="${tool.id}">Remove</button>
          </div>
        </li>`).join("")
    : `<p class="hint">Nothing in the flow. Each added tool is its own setup.</p>`;
  renderOptions();
}

function settingTabs(spec) {
  const tabs = [{ key: "name", label: "Name" }];
  for (const field of spec.fields || []) tabs.push({ key: field.key, label: field.label, field });
  for (const name of spec.panels || []) {
    const labels = { samples: "Samples", shape_model: "Model" };
    tabs.push({ key: `panel:${name}`, label: labels[name] || name, panel: name });
  }
  tabs.push({ key: "help", label: "Help" });
  return tabs;
}

function renderSceneSettings() {
  const panel = $("sceneSettings");
  if (!panel) return;
  const tool = state.flowOpen ? selectedTool() : null;
  if (!tool) {
    panel.hidden = true;
    panel.innerHTML = "";
    document.body.classList.remove("tool-settings");
    return;
  }
  panel.hidden = false;
  document.body.classList.add("tool-settings");
  renderOptionsInto(panel);
}

function renderOptions() {
  if (state.layout === "edit") renderOptionsInto($("sceneSettings"));
  else renderOptionsInto($("toolPanel"));
}

function renderOptionsInto(panel) {
  if (!panel) return;
  const tool = selectedTool();
  if (!tool) {
    delete panel.dataset.toolId;
    panel.innerHTML = `<div class="empty">Select a tool, or add one from the list.</div>`;
    return;
  }
  const spec = specFor(tool.type);
  if (!spec) {
    panel.innerHTML = `<div class="empty">Unknown tool.</div>`;
    return;
  }
  tool.config = { ...(spec.defaults || {}), ...(tool.config || {}) };
  const tabs = settingTabs(spec);
  if (!tabs.some((tab) => tab.key === state.settingTab)) state.settingTab = tabs[0].key;
  const active = tabs.find((tab) => tab.key === state.settingTab);
  const body = active.key === "help"
    ? helpHtml(spec)
    : active.field
      ? fieldHtml(tool, active.field)
      : active.panel
        ? panelHtml(tool, active.panel)
        : `<label>Name<input data-name type="text" value="${escapeHtml(toolDisplayName(tool))}" placeholder="${escapeHtml(spec.title)}" /></label>`;
  panel.dataset.toolId = tool.id;
  if (!(tool.name || "").trim() || (tool.name === "Camera" && tool.type !== "camera")) tool.name = spec.title;
  panel.innerHTML = `
    <div class="opt">
      <div class="opt-head"><h2>${escapeHtml(spec.title)}</h2><span id="saveState"></span></div>
      ${toolOutputHtml(tool)}
      <div class="setting-tabs">
        ${tabs.map((tab) => `<button type="button" data-stab="${tab.key}" class="${tab.key === state.settingTab ? "active" : ""}">${escapeHtml(tab.label)}</button>`).join("")}
      </div>
      <div class="setting-body">${body}</div>
    </div>`;
  if (active.field?.kind === "roi") drawRoiCanvas();
  if (active.panel === "shape_model") drawModelCanvas();
  drawToolOutputRegion(tool);
}

function toolRegion(tool) {
  const cfg = tool?.config || {};
  const roi = cfg.roi || cfg.search_roi;
  if (!roi || Number(roi.w) <= 0.01 || Number(roi.h) <= 0.01) return null;
  return roi;
}

function toolOutputHtml(tool) {
  const result = measuredTool(tool.id);
  if (!result) return `<p class="hint">Measure to see this tool's output.</p>`;
  const image = toolImage(result);
  const img = image
    ? `<div class="tool-output-frame"><img class="tool-output-img" src="${image}" alt="Output of ${escapeHtml(toolDisplayName(tool))}" /><canvas class="region-overlay"></canvas></div>`
    : "";
  return `<div class="tool-output">${img}<p>${escapeHtml(toolDetail(result))}</p></div>`;
}

function measuredTool(id) {
  const item = state.lastResults[state.resultIndex] || null;
  return item?.tools?.find((tool) => tool.id === id) || null;
}

function helpHtml(spec) {
  const parts = [`<p>${escapeHtml(spec.summary || "No help for this tool.")}</p>`];
  if (spec.status) parts.push(`<p class="hint">${escapeHtml(spec.status)}</p>`);
  return `<div class="tool-help">${parts.join("")}</div>`;
}

function fieldHtml(tool, field) {
  const value = tool.config?.[field.key];
  if (field.kind === "text") {
    return `<label>${escapeHtml(field.label)}<input data-key="${field.key}" type="text" value="${escapeHtml(value || "")}" placeholder="${escapeHtml(field.placeholder || "")}" /></label>`;
  }
  if (field.kind === "select") {
    const options = (field.options || []).map((opt) =>
      `<option value="${escapeHtml(opt.value)}" ${value === opt.value ? "selected" : ""}>${escapeHtml(opt.label)}</option>`
    ).join("");
    return `<label>${escapeHtml(field.label)}<select data-key="${field.key}">${options}</select></label>`;
  }
  if (field.kind === "range") {
    const num = Number(value ?? field.min ?? 0);
    return `<label>${escapeHtml(field.label)}
      <input id="minConfidence" data-key="${field.key}" type="range" min="${field.min}" max="${field.max}" step="${field.step}" value="${num}" />
      <span id="confValue">${num.toFixed(2)}</span>
    </label>`;
  }
  if (field.kind === "roi") {
    return `<div class="roi-row">
      <span>${escapeHtml(field.label)}</span>
      <button id="clearRoi" type="button" data-roi-key="${field.key}">Clear</button>
      <span id="roiLabel">${roiText(value)}</span>
    </div>
    <div class="roi-stage"><canvas id="roiCanvas" data-roi-key="${field.key}" width="640" height="280"></canvas></div>`;
  }
  return "";
}

function panelHtml(tool, name) {
  if (name === "shape_model") return shapeModelPanel(tool);
  if (name !== "samples") return "";
  const samples = tool.state?.samples || [];
  const ok = samples.filter((s) => s.label === "OK");
  const ng = samples.filter((s) => s.label === "NG");
  const model = tool.state?.model || {};
  let status = "No model trained. Add OK and NG samples for this tool, then train.";
  if (model.trained_at) {
    status = model.stale
      ? `This tool's model is out of date (${model.trained_at}). Retrain.`
      : `Trained ${model.trained_at}. ${model.ok_count} OK / ${model.ng_count} NG.`;
  }
  return `
    <div id="modelStatus" class="model-status ${model.stale ? "warn" : ""}">${escapeHtml(status)}</div>
    <div class="sample-actions">
      <label class="file-btn ok">Add OK<input id="okFiles" type="file" accept="image/*" multiple hidden /></label>
      <label class="file-btn ng">Add NG<input id="ngFiles" type="file" accept="image/*" multiple hidden /></label>
      <button id="trainBtn" type="button" class="primary">Train model</button>
    </div>
    <div class="sample-grid">
      <div><h3>OK <span>${ok.length}</span></h3><div class="thumbs">${ok.map((s) => thumb(tool, s)).join("") || "<p class='hint'>No OK samples.</p>"}</div></div>
      <div><h3>NG <span>${ng.length}</span></h3><div class="thumbs">${ng.map((s) => thumb(tool, s)).join("") || "<p class='hint'>No NG samples.</p>"}</div></div>
    </div>`;
}

function shapeModelPanel(tool) {
  const model = tool.state?.shape_model || {};
  const saved = !!model.saved;
  const src = saved ? `/api/projects/${state.project.id}/tools/${tool.id}/shape-model?t=${encodeURIComponent(model.registered_at || "")}` : "";
  return `
    <p class="hint">Choose an image, draw a rectangle around the pattern, then Register. The pattern can be any shape.</p>
    <div class="sample-actions">
      <label class="file-btn">Choose image<input id="modelFile" type="file" accept="image/*" hidden /></label>
      <button id="registerModel" type="button" class="primary">Register</button>
      <button id="deleteModel" type="button">Delete</button>
    </div>
    <div class="roi-stage"><canvas id="modelCanvas" width="640" height="280"></canvas></div>
    <div class="model-status ${saved ? "" : "warn"}">${saved ? `Registered model ${model.width}×${model.height}` : "Model image is not saved."}</div>
    ${saved ? `<img class="model-preview" src="${src}" alt="Registered model" />` : ""}`;
}

function thumb(tool, sample) {
  const src = `/api/projects/${state.project.id}/tools/${tool.id}/samples/${sample.id}?thumb=1`;
  return `<div class="thumb"><img src="${src}" alt="" /><div><span>${sample.width}×${sample.height}</span><button type="button" data-del="${sample.id}">Delete</button></div></div>`;
}

function roiText(roi) {
  if (!roi) return "Full image";
  return `x ${Number(roi.x).toFixed(2)}  y ${Number(roi.y).toFixed(2)}  w ${Number(roi.w).toFixed(2)}  h ${Number(roi.h).toFixed(2)}`;
}

function readOptionsIntoTool() {
  const tool = selectedTool();
  if (!tool) return;
  const root = [...document.querySelectorAll("[data-tool-id]")].find((el) => el.dataset.toolId === tool.id);
  if (!root) return;
  const name = root.querySelector("[data-name]");
  if (name) tool.name = name.value.trim() || toolTypeName(tool);
  tool.config = tool.config || {};
  root.querySelectorAll("[data-key]").forEach((el) => {
    const key = el.dataset.key;
    tool.config[key] = el.type === "range" ? Number(el.value) : el.value;
  });
}

function updateImageInput() {
  const camera = cameraTool();
  const ready = imageInputReady();
  const upload = cameraIsUpload();
  const zone = $("dropZone");
  const fileBtn = $("inspectFiles")?.closest(".file-btn");
  const run = $("runInspect");
  zone.classList.toggle("locked", !ready);
  if (fileBtn) fileBtn.classList.toggle("disabled", !ready || !upload);
  if (run) run.disabled = !ready;
  $("inspectFiles").disabled = !ready || !upload;
  if ($("preview").src && !$("preview").hidden) return;
  if (!camera) {
    $("emptyTitle").textContent = "Add a Camera tool";
    $("emptyDetail").textContent = "Image input is available only after Camera is in the flow.";
  } else if ((camera.config?.source || "upload") === "usb") {
    $("emptyTitle").textContent = "USB camera";
    $("emptyDetail").textContent = `Live grab from USB index ${camera.config?.usb_index || 0}. Measure runs on the worker thread.`;
  } else if ((camera.config?.source || "upload") === "gige") {
    $("emptyTitle").textContent = "GigE camera";
    $("emptyDetail").textContent = camera.config?.gige_ip
      ? `Live grab from ${camera.config.gige_ip}. Measure runs on the worker thread.`
      : "Set the GigE IP on the Camera tool.";
  } else {
    $("emptyTitle").textContent = "Drop images here";
    $("emptyDetail").textContent = "Or choose files. This Camera tool uses uploaded images.";
  }
  $("emptyState").hidden = false;
  $("preview").hidden = true;
}

async function saveFlow() {
  readOptionsIntoTool();
  state.project = await api(`/api/projects/${state.project.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ flow: flowPayload() }),
  });
  renderFlow();
  renderCatalog();
  renderEditor();
  updateImageInput();
}

function showScene() {
  state.editorOpen = false;
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("inspect").classList.add("active");
}

function showEditor() {
  state.editorOpen = true;
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("flow").classList.add("active");
}

function showLayout(name, options = {}) {
  state.layout = name === "edit" ? "edit" : "inspect";
  state.editorOpen = false;
  if (!options.keepFlow) state.flowOpen = false;
  if (state.layout !== "edit") state.flowOpen = false;
  if (!options.keepSelection) state.selectedId = null;
  document.body.classList.toggle("mode-inspect", state.layout === "inspect");
  document.body.classList.toggle("mode-edit", state.layout === "edit");
  const layoutLabel = state.layout === "edit" ? "Edit" : "Inspection";
  $("layoutName").textContent = layoutLabel;
  $("brandLayout").textContent = layoutLabel;
  $("sceneLayout").textContent = `Layout: ${layoutLabel}`;
  showScene();
  renderFlow();
  renderCatalog();
}

async function persistLayout() {
  try {
    state.settings = await api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        layout: state.layout,
        flow_open: state.layout === "edit" && state.flowOpen,
      }),
    });
  } catch {
    // Layout still applies in this session if settings cannot be written.
  }
}

function plcDefaults() {
  return {
    production_project_id: state.project?.id || "",
    done_pulse_ms: 80,
    modbus: { enabled: false, host: "0.0.0.0", port: 1502, unit_id: 1 },
    opcua: { enabled: false, endpoint: "opc.tcp://0.0.0.0:4840/svision/" },
  };
}

async function openPlcSettings() {
  const settings = await api("/api/settings");
  state.settings = settings;
  const plc = { ...plcDefaults(), ...(settings.plc || {}) };
  plc.modbus = { ...plcDefaults().modbus, ...(plc.modbus || {}) };
  plc.opcua = { ...plcDefaults().opcua, ...(plc.opcua || {}) };
  const projects = state.projects.length ? state.projects : await api("/api/projects");
  state.projects = projects;
  $("plcProject").innerHTML = projects.map((p) =>
    `<option value="${escapeHtml(p.id)}" ${p.id === plc.production_project_id ? "selected" : ""}>${escapeHtml(p.name)}</option>`
  ).join("");
  if (!plc.production_project_id && projects[0]) $("plcProject").value = projects[0].id;
  $("plcDonePulse").value = plc.done_pulse_ms;
  $("plcModbusEnabled").checked = !!plc.modbus.enabled;
  $("plcModbusHost").value = plc.modbus.host || "0.0.0.0";
  $("plcModbusPort").value = plc.modbus.port || 1502;
  $("plcModbusUnit").value = plc.modbus.unit_id || 1;
  $("plcOpcEnabled").checked = !!plc.opcua.enabled;
  $("plcOpcEndpoint").value = plc.opcua.endpoint || "opc.tcp://0.0.0.0:4840/svision/";
  await refreshPlcStatus();
  $("plcModal").hidden = false;
}

async function refreshPlcStatus() {
  try {
    const status = await api("/api/plc/status");
    const hs = status.handshake || {};
    const mb = status.modbus || {};
    const opc = status.opcua || {};
    $("plcStatus").textContent = [
      `Handshake  ready=${hs.ready} busy=${hs.busy} ok=${hs.ok} ng=${hs.ng} error=${hs.error}`,
      `Modbus  running=${mb.running} ${mb.host || ""}:${mb.port || ""} ${mb.error || ""}`,
      `OPC UA  running=${opc.running} ${opc.endpoint || ""} ${opc.error || ""}`,
      hs.last_error ? `Last error: ${hs.last_error}` : "",
    ].filter(Boolean).join("\n");
  } catch (err) {
    $("plcStatus").textContent = err.message;
  }
}

function startPlcLiveFeed() {
  if (state.plcFeedStarted) return;
  state.plcFeedStarted = true;
  const tick = async () => {
    let busy = false;
    try {
      busy = await pollPlcLive();
    } catch {
      /* ignore poll errors */
    }
    state.plcFeedTimer = setTimeout(tick, busy ? 350 : 1200);
  };
  tick();
}

async function pollPlcLive() {
  const status = await api("/api/plc/status");
  const hs = status.handshake || {};
  const prod = hs.production_project_id || "";
  const viewingProd = !!state.project && prod && state.project.id === prod;
  const seq = Number(hs.measure_seq || 0);

  if (state.plcMeasureSeq === null) {
    state.plcMeasureSeq = seq;
  }

  if (hs.busy && viewingProd) {
    if (!state.plcBusyUi) {
      state.plcBusyUi = true;
      $("verdict").className = "verdict idle";
      $("verdict").textContent = "…";
      $("elapsed").textContent = "…";
      $("verdictMsg").textContent = "PLC / Modbus / OPC measuring…";
    }
    return true;
  }

  if (state.plcBusyUi && !hs.busy) {
    state.plcBusyUi = false;
  }

  if (!seq || seq === state.plcMeasureSeq) return !!hs.busy;
  state.plcMeasureSeq = seq;

  if (!viewingProd) return !!hs.busy;

  if (hs.error || hs.judgment === "ERROR") {
    $("verdict").className = "verdict ng";
    $("verdict").textContent = "ERROR";
    $("elapsed").textContent = `${Number(hs.elapsed_ms || 0).toFixed(0)} ms`;
    $("verdictMsg").textContent = hs.last_error || "Measure failed.";
    return false;
  }

  await restoreCurrentImage();
  return false;
}

async function savePlcSettings() {
  const plc = {
    production_project_id: $("plcProject").value,
    done_pulse_ms: Number($("plcDonePulse").value) || 80,
    modbus: {
      enabled: $("plcModbusEnabled").checked,
      host: $("plcModbusHost").value.trim() || "0.0.0.0",
      port: Number($("plcModbusPort").value) || 1502,
      unit_id: Number($("plcModbusUnit").value) || 1,
    },
    opcua: {
      enabled: $("plcOpcEnabled").checked,
      endpoint: $("plcOpcEndpoint").value.trim() || "opc.tcp://0.0.0.0:4840/svision/",
    },
  };
  state.settings = await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plc }),
  });
  await refreshPlcStatus();
  $("plcStatus").textContent += "\nSaved.";
}

function bind() {
  const saved = state.settings?.layout === "edit" ? "edit" : "inspect";
  state.flowOpen = saved === "edit" && !!state.settings?.flow_open;
  showLayout(saved, { keepFlow: state.flowOpen });
  $("editFlowBtn").addEventListener("click", () => {
    state.flowOpen = true;
    renderFlow();
    persistLayout();
  });
  $("exitFlowBtn").addEventListener("click", () => {
    state.flowOpen = false;
    state.selectedId = null;
    if ($("sceneSaveState")) $("sceneSaveState").textContent = "";
    renderFlow();
    persistLayout();
  });
  $("switchLayout").addEventListener("click", async () => {
    if (state.layout === "edit") {
      try {
        await saveFlow();
      } catch (err) {
        $("flowSaveState").textContent = err.message;
        return;
      }
      showLayout("inspect");
      persistLayout();
      return;
    }
    showLayout("edit");
    persistLayout();
  });
  $("plcSettingsBtn").addEventListener("click", () => openPlcSettings().catch((err) => alert(err.message)));
  $("docsBtn").addEventListener("click", () => { window.open("/plc-docs", "_blank", "noopener"); });
  $("plcClose").addEventListener("click", () => { $("plcModal").hidden = true; });
  $("plcSave").addEventListener("click", () => savePlcSettings().catch((err) => { $("plcStatus").textContent = err.message; }));
  $("plcModal").addEventListener("click", (e) => {
    if (e.target.id === "plcModal") $("plcModal").hidden = true;
  });
  $("saveFlowBtn").addEventListener("click", async () => {
    try {
      await saveFlow();
      $("flowSaveState").textContent = "Saved";
    } catch (err) {
      $("flowSaveState").textContent = err.message;
    }
  });
  $("exitFlow").addEventListener("click", async () => {
    try {
      await saveFlow();
      showLayout("inspect");
    } catch (err) {
      $("flowSaveState").textContent = err.message;
    }
  });
  $("projectSelect").addEventListener("change", (e) => openProject(e.target.value));
  $("newProject").addEventListener("click", async () => {
    const name = prompt("Scene name", "Scene 1");
    if (!name) return;
    const created = await api("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadProjects(created.id);
  });
  $("renameProject").addEventListener("click", async () => {
    const name = prompt("Scene name", state.project.name);
    if (!name) return;
    await api(`/api/projects/${state.project.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadProjects(state.project.id);
  });
  $("deleteProject").addEventListener("click", async () => {
    if (!confirm(`Delete ${state.project.name}?`)) return;
    await api(`/api/projects/${state.project.id}`, { method: "DELETE" });
    state.project = null;
    await loadProjects();
  });
  $("exportProject").addEventListener("click", () => {
    if (!state.project?.id) return;
    window.location.href = `/api/projects/${state.project.id}/export`;
  });
  $("importProject").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const body = new FormData();
      body.append("file", file, file.name);
      const res = await fetch("/api/projects/import", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      await loadProjects(data.id);
    } catch (err) {
      alert(err.message || String(err));
    }
  });

  $("flowList").addEventListener("click", async (e) => {
    const up = e.target.dataset.up;
    const down = e.target.dataset.down;
    const remove = e.target.dataset.remove;
    if (up || down || remove) {
      const id = up || down || remove;
      const list = tools();
      const index = list.findIndex((t) => t.id === id);
      if (up && index > 0) [list[index - 1], list[index]] = [list[index], list[index - 1]];
      if (down && index < list.length - 1) [list[index + 1], list[index]] = [list[index], list[index + 1]];
      if (remove) {
        list.splice(index, 1);
        if (state.selectedId === id) state.selectedId = list[0]?.id || null;
      } else {
        state.selectedId = id;
      }
      await saveFlow();
      return;
    }
    const li = e.target.closest("[data-id]");
    if (!li) return;
    if (state.layout === "edit") readOptionsIntoTool();
    state.selectedId = li.dataset.id;
    showToolOutput(li.dataset.id);
  });

  $("editFlowList").addEventListener("click", async (e) => {
    const up = e.target.dataset.up;
    const down = e.target.dataset.down;
    const remove = e.target.dataset.remove;
    if (up || down || remove) {
      const id = up || down || remove;
      const list = tools();
      const index = list.findIndex((t) => t.id === id);
      if (up && index > 0) [list[index - 1], list[index]] = [list[index], list[index - 1]];
      if (down && index < list.length - 1) [list[index + 1], list[index]] = [list[index], list[index + 1]];
      if (remove) {
        list.splice(index, 1);
        if (state.selectedId === id) state.selectedId = list[0]?.id || null;
      } else {
        state.selectedId = id;
      }
      await saveFlow();
      return;
    }
    const li = e.target.closest("[data-id]");
    if (!li) return;
    readOptionsIntoTool();
    state.selectedId = li.dataset.id;
    state.settingTab = "name";
    showToolOutput(li.dataset.id);
  });

  $("sceneTools").addEventListener("click", async (e) => {
    const type = e.target.dataset.add;
    if (!type || e.target.disabled) return;
    try {
      const created = await api(`/api/projects/${state.project.id}/tools`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type }),
      });
      state.selectedId = created.id;
      state.settingTab = "name";
      state.flowOpen = true;
      await openProject(state.project.id);
      showLayout("edit", { keepFlow: true, keepSelection: true });
    } catch (err) {
      alert(err.message);
    }
  });
  $("sceneSave").addEventListener("click", async () => {
    try {
      await saveFlow();
      if ($("sceneSaveState")) $("sceneSaveState").textContent = "Saved";
      persistLayout();
    } catch (err) {
      if ($("sceneSaveState")) $("sceneSaveState").textContent = err.message;
    }
  });
  $("sceneSettings").addEventListener("click", async (e) => {
    if (e.target.id === "clearRoi") {
      const tool = selectedTool();
      const key = e.target.dataset.roiKey || "roi";
      if (tool) tool.config[key] = null;
      await saveFlow();
      return;
    }
    if (e.target.id === "registerModel") {
      await registerShapeModel();
      return;
    }
    if (e.target.id === "deleteModel") {
      const tool = selectedTool();
      await api(`/api/projects/${state.project.id}/tools/${tool.id}/shape-model`, { method: "DELETE" });
      await refreshSelectedTool();
      return;
    }
    if (e.target.id === "trainBtn") {
      await train();
      return;
    }
    const id = e.target.dataset.del;
    if (id) {
      const tool = selectedTool();
      await api(`/api/projects/${state.project.id}/tools/${tool.id}/samples/${id}`, { method: "DELETE" });
      await refreshSelectedTool();
      return;
    }
    const tab = e.target.dataset.stab;
    if (!tab) return;
    readOptionsIntoTool();
    state.settingTab = tab;
    renderSceneSettings();
  });
  $("sceneSettings").addEventListener("input", (e) => {
    if (e.target.id === "minConfidence" && $("confValue")) {
      $("confValue").textContent = Number(e.target.value).toFixed(2);
    }
    readOptionsIntoTool();
  });
  $("sceneSettings").addEventListener("change", async (e) => {
    if (e.target.id === "okFiles") {
      await uploadSamples("OK", e.target.files);
      e.target.value = "";
      return;
    }
    if (e.target.id === "ngFiles") {
      await uploadSamples("NG", e.target.files);
      e.target.value = "";
      return;
    }
    if (e.target.id === "modelFile") {
      const file = e.target.files?.[0];
      if (!file) return;
      state.modelFile = file;
      state.modelRoi = null;
      if (state.modelUrl) URL.revokeObjectURL(state.modelUrl);
      state.modelUrl = URL.createObjectURL(file);
      drawModelCanvas();
      return;
    }
    readOptionsIntoTool();
  });

  $("catalog").addEventListener("click", async (e) => {
    const type = e.target.dataset.add;
    if (!type || e.target.disabled) return;
    try {
      const created = await api(`/api/projects/${state.project.id}/tools`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type }),
      });
      state.selectedId = created.id;
      state.settingTab = "name";
      await openProject(state.project.id);
    } catch (err) {
      alert(err.message);
    }
  });

  $("toolPanel").addEventListener("click", (e) => {
    const tab = e.target.dataset.stab;
    if (!tab) return;
    readOptionsIntoTool();
    state.settingTab = tab;
    renderOptions();
  });

  $("toolPanel").addEventListener("input", (e) => {
    if (e.target.id === "minConfidence" && $("confValue")) {
      $("confValue").textContent = Number(e.target.value).toFixed(2);
    }
    readOptionsIntoTool();
  });
  $("toolPanel").addEventListener("change", (e) => {
    if (e.target.id === "okFiles" || e.target.id === "ngFiles" || e.target.id === "modelFile") return;
    saveFlow().catch((err) => alert(err.message));
  });
  $("toolPanel").addEventListener("click", async (e) => {
    if (e.target.id === "clearRoi") {
      const tool = selectedTool();
      const key = e.target.dataset.roiKey || "roi";
      if (tool) tool.config[key] = null;
      await saveFlow();
      return;
    }
    if (e.target.id === "registerModel") {
      await registerShapeModel();
      return;
    }
    if (e.target.id === "deleteModel") {
      const tool = selectedTool();
      await api(`/api/projects/${state.project.id}/tools/${tool.id}/shape-model`, { method: "DELETE" });
      await refreshSelectedTool();
      return;
    }
    if (e.target.id === "trainBtn") {
      await train();
      return;
    }
    const id = e.target.dataset.del;
    if (!id) return;
    const tool = selectedTool();
    await api(`/api/projects/${state.project.id}/tools/${tool.id}/samples/${id}`, { method: "DELETE" });
    await refreshSelectedTool();
  });
  $("toolPanel").addEventListener("change", async (e) => {
    if (e.target.id === "okFiles") {
      await uploadSamples("OK", e.target.files);
      e.target.value = "";
      return;
    }
    if (e.target.id === "ngFiles") {
      await uploadSamples("NG", e.target.files);
      e.target.value = "";
      return;
    }
    if (e.target.id === "modelFile") {
      const file = e.target.files?.[0];
      if (!file) return;
      state.modelFile = file;
      state.modelRoi = null;
      if (state.modelUrl) URL.revokeObjectURL(state.modelUrl);
      state.modelUrl = URL.createObjectURL(file);
      drawModelCanvas();
    }
  });

  document.addEventListener("mousedown", (e) => {
    if (e.target.id === "modelCanvas") {
      const p = canvasPoint(e, "modelCanvas");
      state.modelDrawing = { x: p.x, y: p.y, x2: p.x, y2: p.y };
      return;
    }
    if (e.target.id !== "roiCanvas") return;
    const p = canvasPoint(e);
    state.drawing = { x: p.x, y: p.y, x2: p.x, y2: p.y };
  });
  document.addEventListener("mousemove", (e) => {
    if (state.modelDrawing && $("modelCanvas")) {
      const p = canvasPoint(e, "modelCanvas");
      state.modelDrawing.x2 = p.x;
      state.modelDrawing.y2 = p.y;
      drawModelCanvas();
      return;
    }
    if (!state.drawing || !$("roiCanvas")) return;
    const p = canvasPoint(e);
    state.drawing.x2 = p.x;
    state.drawing.y2 = p.y;
    drawRoiCanvas();
  });
  window.addEventListener("mouseup", async () => {
    if (state.modelDrawing) {
      const d = state.modelDrawing;
      state.modelDrawing = null;
      const w = Math.abs(d.x2 - d.x);
      const h = Math.abs(d.y2 - d.y);
      if (w > 0.02 && h > 0.02) {
        state.modelRoi = { x: Math.min(d.x, d.x2), y: Math.min(d.y, d.y2), w, h };
      }
      drawModelCanvas();
      return;
    }
    if (!state.drawing) return;
    const d = state.drawing;
    state.drawing = null;
    const tool = selectedTool();
    const canvas = $("roiCanvas");
    const key = canvas?.dataset.roiKey || "roi";
    const w = Math.abs(d.x2 - d.x);
    const h = Math.abs(d.y2 - d.y);
    if (tool && w > 0.02 && h > 0.02) {
      tool.config[key] = {
        x: Math.min(d.x, d.x2),
        y: Math.min(d.y, d.y2),
        w,
        h,
      };
      await saveFlow();
    } else {
      drawRoiCanvas();
    }
  });

  $("inspectFiles").addEventListener("change", (e) => {
    if (!imageInputReady() || !cameraIsUpload()) return;
    const files = [...e.target.files];
    e.target.value = "";
    if (files.length) selectImages(files);
  });
  $("runInspect").addEventListener("click", runInspect);

  const zone = $("dropZone");
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault();
    zone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
  }));
  zone.addEventListener("drop", (e) => {
    if (!imageInputReady() || !cameraIsUpload()) return;
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    selectImages(files, { measure: true });
  });

  $("batchList").addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (!li) return;
    showResult(Number(li.dataset.i));
  });
  $("toolResults").addEventListener("click", (e) => {
    const card = e.target.closest("[data-tool]");
    if (!card) return;
    showToolOutput(card.dataset.tool);
  });
  bindZoom(zone);
}

async function uploadSamples(label, fileList) {
  const tool = selectedTool();
  const files = [...fileList];
  if (!tool || !files.length) return;
  const body = new FormData();
  body.append("label", label);
  files.forEach((f) => body.append("files", f));
  await api(`/api/projects/${state.project.id}/tools/${tool.id}/samples`, { method: "POST", body });
  await refreshSelectedTool();
}

async function train() {
  const tool = selectedTool();
  if (!tool) return;
  const btn = $("trainBtn");
  if (btn) btn.disabled = true;
  if ($("modelStatus")) $("modelStatus").textContent = "Training…";
  try {
    await saveFlow();
    await api(`/api/projects/${state.project.id}/tools/${tool.id}/train`, { method: "POST" });
    await refreshSelectedTool();
  } catch (err) {
    if ($("modelStatus")) $("modelStatus").textContent = err.message;
    else alert(err.message);
  } finally {
    if ($("trainBtn")) $("trainBtn").disabled = false;
  }
}

async function refreshSelectedTool() {
  const tab = state.settingTab;
  const selected = state.selectedId;
  const flowOpen = state.flowOpen;
  await openProject(state.project.id);
  state.settingTab = tab;
  state.selectedId = selected;
  state.flowOpen = flowOpen;
  if (state.layout === "edit") {
    showLayout("edit", { keepFlow: true, keepSelection: true });
  } else {
    renderOptions();
  }
}

function previewLocal(file) {
  state.outputId = null;
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(file);
  $("imageName").textContent = file.name;
  $("emptyState").hidden = true;
  $("preview").hidden = false;
  $("preview").src = state.sourceUrl;
  resetZoom();
  renderFlow();
}

function showStoredImage(item) {
  if (!item?.url) return;
  state.outputId = null;
  state.sourceUrl = "";
  $("imageName").textContent = item.filename || "Selected image";
  $("emptyState").hidden = true;
  $("preview").hidden = false;
  $("preview").src = `${item.url}?t=${Date.now()}`;
  resetZoom();
}

async function selectImages(files, options = {}) {
  state.pendingFiles = files;
  state.lastResults = [];
  $("toolResults").innerHTML = "";
  $("batchBox").hidden = true;
  previewLocal(files[0]);
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  try {
    const saved = await api(`/api/projects/${state.project.id}/current-image`, { method: "POST", body });
    state.pendingFiles = [];
    state.storedImage = true;
    if (!options.measure && saved.files?.[0]) showStoredImage(saved.files[0]);
  } catch (err) {
    $("verdictMsg").textContent = err.message;
  }
  if (options.measure) await runInspect();
}

async function restoreCurrentImage() {
  if (!state.project) return;
  try {
    const saved = await api(`/api/projects/${state.project.id}/current-image`);
    state.storedImage = !!saved.files?.length;
    if (saved.result?.results?.length) {
      state.lastResults = saved.result.results;
      showResult(0);
      renderBatch(saved.result);
      return;
    }
    state.lastResults = [];
    $("toolResults").innerHTML = "";
    $("batchBox").hidden = true;
    if (saved.files?.[0]) {
      showStoredImage(saved.files[0]);
      return;
    }
    $("preview").hidden = true;
    $("preview").removeAttribute("src");
    $("imageName").textContent = "No image";
    $("emptyState").hidden = false;
  } catch {
    state.storedImage = false;
  }
}

function toolImage(result) {
  return result?.preview || "";
}

function contentFit(boxW, boxH, imgW, imgH) {
  if (!boxW || !boxH || !imgW || !imgH) return null;
  const scale = Math.min(boxW / imgW, boxH / imgH);
  const w = imgW * scale;
  const h = imgH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

function strokeRegion(canvas, img, roi) {
  if (!canvas) return;
  const box = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!roi || !img) return;
  const fit = contentFit(canvas.width, canvas.height, img.naturalWidth || img.width, img.naturalHeight || img.height);
  if (!fit) return;
  ctx.strokeStyle = "#1ad4c0";
  ctx.lineWidth = Math.max(2, 2 * dpr);
  ctx.strokeRect(fit.x + Number(roi.x) * fit.w, fit.y + Number(roi.y) * fit.h, Number(roi.w) * fit.w, Number(roi.h) * fit.h);
}

function whenImageReady(img, fn) {
  if (!img) return;
  if (img.complete && img.naturalWidth) fn();
  else img.addEventListener("load", fn, { once: true });
}

function drawPreviewRegion(tool) {
  const canvas = $("regionOverlay");
  const img = $("preview");
  if (!canvas || !img) return;
  const roi = toolRegion(tool);
  if (!roi || img.hidden) {
    canvas.hidden = true;
    return;
  }
  canvas.hidden = false;
  whenImageReady(img, () => strokeRegion(canvas, img, roi));
}

function drawToolOutputRegion(tool) {
  const frame = document.querySelector("#sceneSettings .tool-output-frame");
  if (!frame) return;
  const img = frame.querySelector("img");
  const canvas = frame.querySelector("canvas");
  const roi = toolRegion(tool);
  if (!roi) return;
  whenImageReady(img, () => strokeRegion(canvas, img, roi));
}

function applyZoom() {
  const img = $("preview");
  const overlay = $("regionOverlay");
  const { scale, x, y } = state.zoom;
  const transform = `translate(${x}px, ${y}px) scale(${scale})`;
  img.style.transform = transform;
  if (overlay) overlay.style.transform = transform;
  $("dropZone").classList.toggle("zoomed", scale > 1);
}

function resetZoom() {
  state.zoom.scale = 1;
  state.zoom.x = 0;
  state.zoom.y = 0;
  state.zoom.drag = null;
  applyZoom();
}

function bindZoom(zone) {
  zone.addEventListener("wheel", (e) => {
    if ($("preview").hidden) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const prev = state.zoom.scale;
    const next = Math.min(12, Math.max(1, prev * factor));
    const rect = zone.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    state.zoom.x = cx - (cx - state.zoom.x) * (next / prev);
    state.zoom.y = cy - (cy - state.zoom.y) * (next / prev);
    state.zoom.scale = next;
    if (next <= 1.01) {
      state.zoom.scale = 1;
      state.zoom.x = 0;
      state.zoom.y = 0;
    }
    applyZoom();
  }, { passive: false });
  zone.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || state.zoom.scale <= 1 || e.target.closest("button")) return;
    state.zoom.drag = { x: e.clientX, y: e.clientY, ox: state.zoom.x, oy: state.zoom.y };
    zone.setPointerCapture(e.pointerId);
  });
  zone.addEventListener("pointermove", (e) => {
    const drag = state.zoom.drag;
    if (!drag) return;
    state.zoom.x = drag.ox + (e.clientX - drag.x);
    state.zoom.y = drag.oy + (e.clientY - drag.y);
    applyZoom();
  });
  zone.addEventListener("pointerup", () => {
    state.zoom.drag = null;
  });
  zone.addEventListener("dblclick", () => resetZoom());
  window.addEventListener("resize", () => {
    const tool = tools().find((item) => item.id === (state.outputId || state.selectedId));
    drawPreviewRegion(tool);
    drawToolOutputRegion(tool);
  });
}

function showToolOutput(id, options = {}) {
  state.outputId = id;
  state.selectedId = id;
  renderFlow();
  const flowTool = tools().find((tool) => tool.id === id);
  const result = measuredTool(id);
  if (state.editorOpen && !options.keepPanel) renderEditor();
  if (!state.editorOpen) {
    const image = toolImage(result);
    if (image) {
      $("emptyState").hidden = true;
      $("preview").hidden = false;
      $("preview").src = image;
      resetZoom();
      $("preview").onload = () => drawPreviewRegion(flowTool);
    }
    if (result) {
      const cls = (result.judgment || "idle").toLowerCase();
      $("verdict").className = `verdict ${cls}`;
      $("verdict").textContent = result.judgment || "--";
      $("elapsed").textContent = `${Number(result.elapsed_ms || 0).toFixed(1)} ms`;
      $("verdictMsg").textContent = toolDetail(result);
      $("imageName").textContent = `${toolDisplayName(flowTool || result)} output`;
    } else if (flowTool) {
      $("imageName").textContent = flowTool.name;
      $("verdictMsg").textContent = "Measure to see this tool's output.";
    }
    drawPreviewRegion(flowTool);
  }
  document.querySelectorAll("#toolResults .card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.tool === id);
  });
}

async function runInspect() {
  if (!imageInputReady()) {
    $("verdictMsg").textContent = "Add a Camera tool first.";
    return;
  }
  if (cameraIsUpload() && !state.pendingFiles.length && !state.storedImage) {
    $("verdictMsg").textContent = "Choose or drop images first.";
    return;
  }
  $("runInspect").disabled = true;
  $("verdict").className = "verdict idle";
  $("verdict").textContent = "…";
  $("verdictMsg").textContent = "Measuring on worker thread…";
  try {
    const body = new FormData();
    state.pendingFiles.forEach((f) => body.append("files", f));
    const data = await api(`/api/projects/${state.project.id}/inspect`, { method: "POST", body });
    state.pendingFiles = [];
    state.storedImage = cameraIsUpload() || state.storedImage;
    state.lastResults = data.results;
    showResult(0);
    renderBatch(data);
  } catch (err) {
    $("verdict").className = "verdict ng";
    $("verdict").textContent = "NG";
    $("verdictMsg").textContent = err.message;
    $("toolResults").innerHTML = "";
  } finally {
    $("runInspect").disabled = false;
  }
}

function showResult(index) {
  const item = state.lastResults[index];
  if (!item) return;
  state.resultIndex = index;
  state.outputId = null;
  const cls = item.judgment === "OK" ? "ok" : "ng";
  $("verdict").className = `verdict ${cls}`;
  $("verdict").textContent = item.judgment;
  $("elapsed").textContent = `${item.elapsed_ms.toFixed(1)} ms`;
  $("verdictMsg").textContent = item.message;
  $("imageName").textContent = item.filename;
  $("imageSize").textContent = `${item.width} × ${item.height}`;
  const measured = [...(item.tools || [])].reverse().find((tool) => tool.type !== "camera" && tool.preview)
    || (item.tools || [])[item.tools.length - 1];
  $("emptyState").hidden = true;
  $("preview").hidden = false;
  $("preview").src = measured?.preview || item.preview;
  if (measured?.id) state.outputId = measured.id;
  $("imageName").textContent = measured?.preview ? `${measured.name} output` : item.filename;
  resetZoom();
  const shown = tools().find((tool) => tool.id === measured?.id) || measured;
  drawPreviewRegion(shown);
  $("toolResults").innerHTML = item.tools.map(toolCard).join("");
  renderFlow();
}

function toolDetail(tool) {
  let detail = tool.message || "";
  if (tool.type === "camera") {
    detail = tool.message || tool.source || "";
  } else if (tool.type === "shape_search") {
    detail = tool.message || "No shape";
    if (tool.x != null) detail += `  x ${tool.x}  y ${tool.y}  angle ${tool.angle}°`;
  } else if (tool.type === "position_compensation") {
    detail = tool.message || "";
  } else if (tool.type === "classify") {
    detail = `${tool.label}  confidence ${(tool.confidence * 100).toFixed(1)}%  (need ${(tool.min_confidence * 100).toFixed(0)}%)`;
    if (tool.residual != null) detail += `  difference ${(tool.residual * 100).toFixed(1)}%`;
  } else if (tool.type === "ocr") {
    detail = tool.text ? `Read: ${tool.text}` : (tool.message || "No text");
    if (tool.expected) detail += `  expected: ${tool.expected}`;
  } else if (tool.type === "code") {
    detail = tool.codes?.length
      ? tool.codes.map((c) => `${c.type}: ${c.value}`).join(", ")
      : (tool.message || "No code");
    if (tool.expected) detail += `  expected: ${tool.expected}`;
  }
  return `${detail} · ${Number(tool.elapsed_ms || 0).toFixed(1)} ms`;
}

function toolCard(tool) {
  const cls = (tool.judgment || "").toLowerCase();
  const selected = tool.id === state.outputId ? " selected" : "";
  return `<div class="card ${cls}${selected}" data-tool="${tool.id}"><div class="row"><b>${escapeHtml(toolDisplayName(tool))}</b><span class="tag ${cls}">${tool.judgment || ""}</span></div><small>${escapeHtml(toolDetail(tool))}</small></div>`;
}

function renderBatch(data) {
  const box = $("batchBox");
  if (data.count < 2) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  $("batchSummary").textContent = `${data.ok} OK / ${data.ng} NG · ${data.elapsed_ms.toFixed(0)} ms total`;
  $("batchList").innerHTML = data.results.map((r, i) =>
    `<li data-i="${i}"><span>${escapeHtml(r.filename)}</span><span><b class="tag ${r.judgment.toLowerCase()}">${r.judgment}</b> ${r.elapsed_ms.toFixed(1)} ms</span></li>`
  ).join("");
}

async function registerShapeModel() {
  const tool = selectedTool();
  if (!tool || !state.modelFile) {
    alert("Choose an image, then draw the pattern.");
    return;
  }
  if (!state.modelRoi) {
    alert("Draw a rectangle around the pattern on the image.");
    return;
  }
  const body = new FormData();
  body.append("file", state.modelFile);
  body.append("x", state.modelRoi.x);
  body.append("y", state.modelRoi.y);
  body.append("w", state.modelRoi.w);
  body.append("h", state.modelRoi.h);
  await api(`/api/projects/${state.project.id}/tools/${tool.id}/shape-model`, { method: "POST", body });
  state.settingTab = "panel:shape_model";
  await refreshSelectedTool();
}

function drawModelCanvas() {
  const canvas = $("modelCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  const paint = () => {
    ctx.fillStyle = "#0a0c0e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (img.width) {
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (canvas.width - w) / 2;
      const y = (canvas.height - h) / 2;
      canvas._fit = { x, y, w, h };
      ctx.drawImage(img, x, y, w, h);
    } else {
      canvas._fit = { x: 0, y: 0, w: canvas.width, h: canvas.height };
      ctx.fillStyle = "#9aa6b2";
      ctx.fillText("Choose an image, then draw the pattern.", 16, 28);
    }
    const roi = state.modelDrawing
      ? {
          x: Math.min(state.modelDrawing.x, state.modelDrawing.x2),
          y: Math.min(state.modelDrawing.y, state.modelDrawing.y2),
          w: Math.abs(state.modelDrawing.x2 - state.modelDrawing.x),
          h: Math.abs(state.modelDrawing.y2 - state.modelDrawing.y),
        }
      : state.modelRoi;
    if (roi) {
      const fit = canvas._fit;
      ctx.strokeStyle = "#1ad4c0";
      ctx.lineWidth = 2;
      ctx.strokeRect(fit.x + roi.x * fit.w, fit.y + roi.y * fit.h, roi.w * fit.w, roi.h * fit.h);
    }
  };
  if (state.modelUrl) {
    img.onload = paint;
    img.src = state.modelUrl;
  } else {
    paint();
  }
}

function drawRoiCanvas() {
  const canvas = $("roiCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  const source = $("preview").src;
  const tool = selectedTool();
  const paint = () => {
    ctx.fillStyle = "#0a0c0e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (img.width) {
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (canvas.width - w) / 2;
      const y = (canvas.height - h) / 2;
      canvas._fit = { x, y, w, h };
      ctx.drawImage(img, x, y, w, h);
    } else {
      canvas._fit = { x: 0, y: 0, w: canvas.width, h: canvas.height };
    }
    const roi = state.drawing
      ? {
          x: Math.min(state.drawing.x, state.drawing.x2),
          y: Math.min(state.drawing.y, state.drawing.y2),
          w: Math.abs(state.drawing.x2 - state.drawing.x),
          h: Math.abs(state.drawing.y2 - state.drawing.y),
        }
      : tool?.config?.[canvas.dataset.roiKey || "roi"];
    if (roi) {
      const fit = canvas._fit;
      ctx.strokeStyle = "#1ad4c0";
      ctx.lineWidth = 2;
      ctx.strokeRect(fit.x + roi.x * fit.w, fit.y + roi.y * fit.h, roi.w * fit.w, roi.h * fit.h);
    }
  };
  if (source) {
    img.onload = paint;
    img.src = source;
  } else {
    paint();
  }
}

function canvasPoint(e, id = "roiCanvas") {
  const canvas = $(id);
  const rect = canvas.getBoundingClientRect();
  const fit = canvas._fit || { x: 0, y: 0, w: canvas.width, h: canvas.height };
  const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
  return {
    x: clamp((px - fit.x) / fit.w, 0, 1),
    y: clamp((py - fit.y) / fit.h, 0, 1),
  };
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

boot().catch((err) => {
  $("hostInfo").textContent = err.message;
});
