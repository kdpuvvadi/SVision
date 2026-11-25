// SVision ML Judgment tool UI — isolated from other tools.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  T.registerPanel("samples", {
    label: "Samples",
    render(tool) {
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
    },
  });

  T.registerDetail("classify", (tool) => {
    let detail = `${tool.label}  confidence ${(tool.confidence * 100).toFixed(1)}%  (need ${(tool.min_confidence * 100).toFixed(0)}%)`;
    if (tool.residual != null) detail += `  difference ${(tool.residual * 100).toFixed(1)}%`;
    return detail;
  });
})();
