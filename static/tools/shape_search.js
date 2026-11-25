// SVision Shape Search tool UI — isolated from other tools.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  T.registerPanel("shape_model", {
    label: "Model",
    render(tool) {
      const model = tool.state?.shape_model || {};
      const saved = !!model.saved;
      const cam = (typeof cameraTool === "function" ? cameraTool() : null) || null;
      const source = (cam?.config?.source || "upload").toLowerCase();
      const live = source === "gentl" || source === "gige" || source === "usb";
      const src = saved
        ? `/api/projects/${state.project.id}/tools/${tool.id}/shape-model?t=${encodeURIComponent(model.registered_at || "")}`
        : "";
      const hint = live
        ? "Grab from the Camera tool (or choose a file), draw a rectangle around the pattern, then Register."
        : "Choose an image, draw a rectangle around the pattern, then Register. The pattern can be any shape.";
      return `
        <p class="hint">${hint}</p>
        <div class="sample-actions">
          ${live ? `<button id="grabModelCamera" type="button" class="primary">Grab from camera</button>` : ""}
          <label class="file-btn">Choose image<input id="modelFile" type="file" accept="image/*" hidden /></label>
          <button id="registerModel" type="button" ${live ? "" : 'class="primary"'}>Register</button>
          <button id="deleteModel" type="button">Delete</button>
        </div>
        <p id="grabModelStatus" class="hint">${live ? "Camera is live — Grab captures a still for the model." : ""}</p>
        <div class="roi-stage"><canvas id="modelCanvas" width="640" height="280"></canvas></div>
        <div class="model-status ${saved ? "" : "warn"}">${saved ? `Registered model ${model.width}×${model.height}` : "Model image is not saved."}</div>
        ${saved ? `<img class="model-preview" src="${src}" alt="Registered model" />` : ""}`;
    },
    mount() {
      if (typeof drawModelCanvas === "function") drawModelCanvas();
    },
  });

  T.registerDetail("shape_search", (tool) => {
    let detail = tool.message || "No shape";
    if (tool.x != null) detail += `  x ${tool.x}  y ${tool.y}  angle ${tool.angle}°`;
    return detail;
  });
})();
