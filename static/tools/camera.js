// SVision Camera tool UI — Connect panel + GenTL field kinds.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  const DEFAULT_CTI =
    "C:\\Program Files\\Common Files\\OMRON_SENTECH\\GenTL\\v1_5\\StGenTL_MD_VC141_v1_5.cti";
  /** Survives saveFlow → renderSceneSettings remount (DOM status text alone does not). */
  const lastTestMsg = Object.create(null);

  T.registerFieldKind("cti", (tool, field) => {
    const value = tool.config?.[field.key];
    return `<label>${escapeHtml(field.label)}
      <div class="path-row">
        <input data-key="${field.key}" type="text" value="${escapeHtml(value || "")}" placeholder="${escapeHtml(field.placeholder || "")}" />
        <button type="button" data-browse-cti>Find CTI…</button>
      </div>
      <span class="hint">GenTL producer (.cti) on this IPC.</span>
      <div id="ctiPicker" class="cti-picker" hidden></div>
    </label>`;
  });

  T.registerFieldKind("gentl_device", (tool, field) => {
    const value = tool.config?.[field.key] || "";
    return `<label>${escapeHtml(field.label)}
      <div class="path-row">
        <input data-key="${field.key}" type="text" value="${escapeHtml(value)}" placeholder="${escapeHtml(field.placeholder || "")}" />
        <button type="button" data-scan-devices>Scan</button>
      </div>
      <span class="hint">Use Connect → Scan, or paste a device id / MAC.</span>
      <div id="devicePicker" class="cti-picker" hidden></div>
    </label>`;
  });

  T.registerPanel("gentl_setup", {
    label: "Connect",
    render(tool) {
      const cfg = tool.config || {};
      const source = cfg.source || "upload";
      const cti = cfg.cti_path || "";
      const device = cfg.device_id || "";
      const ip = cfg.gige_ip || "";
      if (source !== "gentl" && source !== "gige") {
        return `
          <p class="hint">Set <b>Source</b> to <b>GigE / GenTL</b>, then return here to connect the camera.</p>
          <p class="hint">Upload and USB do not use this Connect page.</p>`;
      }
      return `
        <p class="hint">GigE needs a vendor CTI file, then a scanned camera. IP alone does not open the camera.</p>
        <ol class="hint" style="margin:0 0 12px 1.2em; padding:0; display:flex; flex-direction:column; gap:6px;">
          <li>Find / confirm the Omron Sentech <b>CTI</b> producer.</li>
          <li>Optional: enter <b>GigE IP</b> (e.g. 192.168.250.4) to auto-pick.</li>
          <li><b>Scan</b>, click your camera in the list, then <b>Test grab</b>.</li>
          <li><b>Save</b> the flow, then Measure.</li>
        </ol>
        <label>CTI producer
          <div class="path-row">
            <input id="connectCti" data-key="cti_path" type="text" value="${escapeHtml(cti)}" placeholder="${escapeHtml(DEFAULT_CTI)}" />
            <button type="button" data-browse-cti>Find CTI…</button>
          </div>
        </label>
        <div id="ctiPicker" class="cti-picker" hidden></div>
        <label>GigE IP
          <input id="connectIp" data-key="gige_ip" type="text" value="${escapeHtml(ip)}" placeholder="192.168.250.4" />
        </label>
        <label>Camera id
          <div class="path-row">
            <input id="connectDevice" data-key="device_id" type="text" value="${escapeHtml(device)}" placeholder="Filled after Scan" />
            <button type="button" data-scan-devices>Scan</button>
          </div>
        </label>
        <div id="devicePicker" class="cti-picker" hidden></div>
        <div class="sample-actions" style="margin-top:10px;">
          <button type="button" id="cameraTestBtn" class="primary">Test grab</button>
        </div>
        <p id="cameraTestStatus" class="hint">${escapeHtml(lastTestMsg[tool.id] || "Not tested yet.")}</p>`;
    },
    mount(tool) {
      const cfg = tool.config || {};
      if ((cfg.source === "gentl" || cfg.source === "gige") && !(cfg.cti_path || "").trim()) {
        const input = document.getElementById("connectCti");
        if (input && !input.value) input.value = DEFAULT_CTI;
        tool.config.cti_path = DEFAULT_CTI;
      }
      const btn = document.getElementById("cameraTestBtn");
      if (!btn || btn.dataset.bound) return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", async () => {
        if (typeof readOptionsIntoTool === "function") readOptionsIntoTool();
        const current = typeof selectedTool === "function" ? selectedTool() : tool;
        const conf = current?.config || {};
        const tid = current?.id || tool.id;
        const status = document.getElementById("cameraTestStatus");
        lastTestMsg[tid] = "Opening camera…";
        if (status) status.textContent = lastTestMsg[tid];
        try {
          const out = await api("/api/cameras/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: "gentl",
              cti_path: conf.cti_path || "",
              device_id: conf.device_id || "",
              gige_ip: conf.gige_ip || "",
            }),
          });
          if (out.device_id && current) current.config.device_id = out.device_id;
          lastTestMsg[tid] = out.message || "OK";
          if (status) status.textContent = lastTestMsg[tid];
          if (typeof saveFlow === "function") await saveFlow();
          if (typeof updateImageInput === "function") updateImageInput();
        } catch (err) {
          lastTestMsg[tid] = err.message || String(err);
          if (status) status.textContent = lastTestMsg[tid];
        }
      });
    },
  });

  T.registerDetail("camera", (tool) => tool.message || tool.source || "");
})();
