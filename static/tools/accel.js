// SVision hardware-acceleration field (shared by OCR, Shape Search, …).
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  let cache = null;
  let loading = null;

  async function loadDevices() {
    if (cache) return cache;
    if (loading) return loading;
    loading = fetch("/api/accel/devices")
      .then((r) => r.json())
      .then((data) => {
        cache = data.devices || [];
        return cache;
      })
      .catch(() => {
        cache = [{ id: "cpu", label: "CPU", ready: true, detail: "Always available" }];
        return cache;
      })
      .finally(() => {
        loading = null;
      });
    return loading;
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  T.registerFieldKind("accel", (tool, field) => {
    const value = tool.config?.[field.key] || "cpu";
    const id = `accel-${tool.id}-${field.key}`;
    // Options filled asynchronously after mount; placeholder keeps layout stable.
    return `<label class="accel-field">${escapeHtml(field.label || "Hardware acceleration")}
      <select id="${id}" data-key="${escapeHtml(field.key)}" data-accel-select="1">
        <option value="${escapeHtml(value)}" selected>${escapeHtml(value)}</option>
      </select>
      <span class="hint accel-detail" data-accel-detail="${id}"></span>
    </label>`;
  });

  async function fillSelects(root) {
    const devices = await loadDevices();
    const selects = [...(root || document).querySelectorAll("[data-accel-select]")];
    if (!selects.length) return;
    obs.disconnect();
    try {
      selects.forEach((sel) => {
        const current = sel.value || "cpu";
        const options = devices.map((d) => {
          const label = d.ready ? d.label : `${d.label} (unavailable)`;
          return `<option value="${escapeHtml(d.id)}" ${d.id === current ? "selected" : ""} ${d.ready ? "" : "disabled"}>${escapeHtml(label)}</option>`;
        });
        if (current && !devices.some((d) => d.id === current)) {
          options.unshift(`<option value="${escapeHtml(current)}" selected>${escapeHtml(current)}</option>`);
        }
        sel.innerHTML = options.join("");
        if (!devices.some((d) => d.id === sel.value && d.ready)) {
          const firstReady = devices.find((d) => d.ready);
          if (firstReady) sel.value = firstReady.id;
        }
        const detail = document.querySelector(`[data-accel-detail="${sel.id}"]`);
        const picked = devices.find((d) => d.id === sel.value) || devices[0];
        if (detail && picked) detail.textContent = picked.detail || "";
        sel.onchange = () => {
          const next = devices.find((d) => d.id === sel.value);
          if (detail) detail.textContent = next?.detail || "";
        };
      });
    } finally {
      if (document.body) obs.observe(document.body, { childList: true, subtree: true });
    }
  }

  let fillTimer = null;
  function scheduleFill() {
    if (fillTimer) return;
    fillTimer = setTimeout(() => {
      fillTimer = null;
      if (document.querySelector("[data-accel-select]")) fillSelects();
    }, 50);
  }

  // Refresh after scene settings render.
  const obs = new MutationObserver(scheduleFill);
  if (document.body) {
    obs.observe(document.body, { childList: true, subtree: true });
    scheduleFill();
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      obs.observe(document.body, { childList: true, subtree: true });
      scheduleFill();
    });
  }

  T.accelDevices = loadDevices;
})();
