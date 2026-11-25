// SVision Camera tool UI — GenTL field kinds + result detail.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  T.registerFieldKind("cti", (tool, field) => {
    const value = tool.config?.[field.key];
    return `<label>${escapeHtml(field.label)}
      <div class="path-row">
        <input data-key="${field.key}" type="text" value="${escapeHtml(value || "")}" placeholder="${escapeHtml(field.placeholder || "")}" />
        <button type="button" data-browse-cti>Find CTI…</button>
      </div>
      <span class="hint">GenTL producer (.cti) on this IPC. Required for GigE / GenTL cameras.</span>
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
      <span class="hint">Pick a device id after scanning with the CTI producer above.</span>
      <div id="devicePicker" class="cti-picker" hidden></div>
    </label>`;
  });

  T.registerDetail("camera", (tool) => tool.message || tool.source || "");
})();
