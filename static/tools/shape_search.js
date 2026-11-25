// SVision Shape Search tool UI — model register with zoom + edge/corner adjust.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  const HANDLE = 8;
  const EDGE = 6;
  let editorBound = false;
  let view = { zoom: 1, panX: 0, panY: 0 };
  let drag = null; // { mode, handle, startX, startY, roi0, pan0 }
  let imgEl = null;

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function canvasEl() {
    return document.getElementById("modelCanvas");
  }

  function normPoint(e) {
    const canvas = canvasEl();
    if (!canvas || !canvas._fit) return null;
    const rect = canvas.getBoundingClientRect();
    const fit = canvas._fit;
    const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
    return {
      x: clamp((px - fit.x) / fit.w, 0, 1),
      y: clamp((py - fit.y) / fit.h, 0, 1),
      cx: px,
      cy: py,
    };
  }

  function hitHandle(p, roi) {
    if (!roi || !p) return null;
    const canvas = canvasEl();
    const fit = canvas._fit;
    const rx = fit.x + roi.x * fit.w;
    const ry = fit.y + roi.y * fit.h;
    const rw = roi.w * fit.w;
    const rh = roi.h * fit.h;
    const pts = {
      nw: [rx, ry],
      n: [rx + rw / 2, ry],
      ne: [rx + rw, ry],
      e: [rx + rw, ry + rh / 2],
      se: [rx + rw, ry + rh],
      s: [rx + rw / 2, ry + rh],
      sw: [rx, ry + rh],
      w: [rx, ry + rh / 2],
    };
    for (const [name, [hx, hy]] of Object.entries(pts)) {
      if (Math.abs(p.cx - hx) <= HANDLE && Math.abs(p.cy - hy) <= HANDLE) return name;
    }
    // Edge strips (between corners)
    if (Math.abs(p.cy - ry) <= EDGE && p.cx >= rx && p.cx <= rx + rw) return "n";
    if (Math.abs(p.cy - (ry + rh)) <= EDGE && p.cx >= rx && p.cx <= rx + rw) return "s";
    if (Math.abs(p.cx - rx) <= EDGE && p.cy >= ry && p.cy <= ry + rh) return "w";
    if (Math.abs(p.cx - (rx + rw)) <= EDGE && p.cy >= ry && p.cy <= ry + rh) return "e";
    if (p.cx >= rx && p.cx <= rx + rw && p.cy >= ry && p.cy <= ry + rh) return "move";
    return null;
  }

  function cursorFor(handle) {
    const map = {
      n: "ns-resize",
      s: "ns-resize",
      e: "ew-resize",
      w: "ew-resize",
      nw: "nwse-resize",
      se: "nwse-resize",
      ne: "nesw-resize",
      sw: "nesw-resize",
      move: "move",
    };
    return map[handle] || "crosshair";
  }

  function applyHandle(handle, roi0, p0, p) {
    const dx = p.x - p0.x;
    const dy = p.y - p0.y;
    let { x, y, w, h } = roi0;
    const x2 = x + w;
    const y2 = y + h;
    let nx = x;
    let ny = y;
    let nx2 = x2;
    let ny2 = y2;
    if (handle === "move") {
      nx = clamp(x + dx, 0, 1 - w);
      ny = clamp(y + dy, 0, 1 - h);
      return { x: nx, y: ny, w, h };
    }
    if (handle.includes("n")) ny = y + dy;
    if (handle.includes("s")) ny2 = y2 + dy;
    if (handle.includes("w")) nx = x + dx;
    if (handle.includes("e")) nx2 = x2 + dx;
    const out = {
      x: Math.min(nx, nx2),
      y: Math.min(ny, ny2),
      w: Math.abs(nx2 - nx),
      h: Math.abs(ny2 - ny),
    };
    out.x = clamp(out.x, 0, 1);
    out.y = clamp(out.y, 0, 1);
    out.w = clamp(out.w, 0.005, 1 - out.x);
    out.h = clamp(out.h, 0.005, 1 - out.y);
    return out;
  }

  function updateReadout() {
    const el = document.getElementById("modelRoiReadout");
    if (!el) return;
    const roi = state.modelRoi;
    if (!roi) {
      el.textContent = "No selection — drag to draw, then drag edges/corners to fine-tune.";
      return;
    }
    el.textContent = `x ${(roi.x * 100).toFixed(1)}%  y ${(roi.y * 100).toFixed(1)}%  w ${(roi.w * 100).toFixed(1)}%  h ${(roi.h * 100).toFixed(1)}%  · zoom ${view.zoom.toFixed(2)}×`;
  }

  function baseFit(canvas, iw, ih) {
    const scale = Math.min(canvas.width / iw, canvas.height / ih);
    const w = iw * scale * view.zoom;
    const h = ih * scale * view.zoom;
    const x = (canvas.width - w) / 2 + view.panX;
    const y = (canvas.height - h) / 2 + view.panY;
    return { x, y, w, h };
  }

  function paint() {
    const canvas = canvasEl();
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0a0c0e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const drawRoi = (roi) => {
      if (!roi) return;
      const fit = canvas._fit;
      const rx = fit.x + roi.x * fit.w;
      const ry = fit.y + roi.y * fit.h;
      const rw = roi.w * fit.w;
      const rh = roi.h * fit.h;
      ctx.save();
      ctx.fillStyle = "rgba(26, 212, 192, 0.12)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = "#1ad4c0";
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);
      const pts = [
        [rx, ry],
        [rx + rw / 2, ry],
        [rx + rw, ry],
        [rx + rw, ry + rh / 2],
        [rx + rw, ry + rh],
        [rx + rw / 2, ry + rh],
        [rx, ry + rh],
        [rx, ry + rh / 2],
      ];
      ctx.fillStyle = "#1ad4c0";
      pts.forEach(([hx, hy]) => {
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
      });
      ctx.restore();
    };

    const finish = () => {
      const roi = drag && drag.mode === "draw"
        ? {
            x: Math.min(drag.start.x, drag.cur.x),
            y: Math.min(drag.start.y, drag.cur.y),
            w: Math.abs(drag.cur.x - drag.start.x),
            h: Math.abs(drag.cur.y - drag.start.y),
          }
        : state.modelRoi;
      drawRoi(roi);
      updateReadout();
    };

    if (state.modelUrl) {
      if (!imgEl || imgEl.src !== state.modelUrl) {
        imgEl = new Image();
        imgEl.onload = () => paint();
        imgEl.src = state.modelUrl;
        ctx.fillStyle = "#9aa6b2";
        ctx.fillText("Loading…", 16, 28);
        return;
      }
      if (!imgEl.complete || !imgEl.width) {
        ctx.fillStyle = "#9aa6b2";
        ctx.fillText("Loading…", 16, 28);
        return;
      }
      canvas._fit = baseFit(canvas, imgEl.width, imgEl.height);
      ctx.drawImage(imgEl, canvas._fit.x, canvas._fit.y, canvas._fit.w, canvas._fit.h);
      finish();
      return;
    }

    canvas._fit = { x: 0, y: 0, w: canvas.width, h: canvas.height };
    ctx.fillStyle = "#9aa6b2";
    ctx.fillText("Grab or choose an image, then draw the pattern.", 16, 28);
    finish();
  }

  function setZoom(next, around) {
    const canvas = canvasEl();
    const z = clamp(next, 0.5, 8);
    if (canvas && around && imgEl?.width) {
      const before = baseFit(canvas, imgEl.width, imgEl.height);
      const u = before.w ? (around.cx - before.x) / before.w : 0.5;
      const v = before.h ? (around.cy - before.y) / before.h : 0.5;
      view.zoom = z;
      const scale = Math.min(canvas.width / imgEl.width, canvas.height / imgEl.height);
      const w = imgEl.width * scale * view.zoom;
      const h = imgEl.height * scale * view.zoom;
      view.panX = around.cx - u * w - (canvas.width - w) / 2;
      view.panY = around.cy - v * h - (canvas.height - h) / 2;
    } else {
      view.zoom = z;
    }
    paint();
  }

  function onWheel(e) {
    if (e.target.id !== "modelCanvas") return;
    e.preventDefault();
    const p = normPoint(e);
    if (!p) return;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setZoom(view.zoom * factor, p);
  }

  function onDown(e) {
    if (e.target.id !== "modelCanvas") return;
    e.preventDefault();
    e.stopPropagation();
    const p = normPoint(e);
    if (!p) return;
    if (e.button === 1 || e.shiftKey) {
      drag = { mode: "pan", startCx: p.cx, startCy: p.cy, pan0: { x: view.panX, y: view.panY } };
      return;
    }
    const handle = state.modelRoi ? hitHandle(p, state.modelRoi) : null;
    if (handle) {
      drag = {
        mode: "resize",
        handle,
        start: p,
        roi0: { ...state.modelRoi },
      };
      return;
    }
    drag = { mode: "draw", start: p, cur: p };
  }

  function onMove(e) {
    const canvas = canvasEl();
    if (!canvas) return;
    const p = normPoint(e);
    if (!drag) {
      if (e.target.id === "modelCanvas" && state.modelRoi && p) {
        canvas.style.cursor = cursorFor(hitHandle(p, state.modelRoi) || "");
      }
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    if (!p) return;
    if (drag.mode === "pan") {
      view.panX = drag.pan0.x + (p.cx - drag.startCx);
      view.panY = drag.pan0.y + (p.cy - drag.startCy);
      paint();
      return;
    }
    if (drag.mode === "draw") {
      drag.cur = p;
      paint();
      return;
    }
    if (drag.mode === "resize") {
      state.modelRoi = applyHandle(drag.handle, drag.roi0, drag.start, p);
      paint();
    }
  }

  function onUp(e) {
    if (!drag) return;
    e.stopPropagation();
    if (drag.mode === "draw") {
      const a = drag.start;
      const b = drag.cur;
      const w = Math.abs(b.x - a.x);
      const h = Math.abs(b.y - a.y);
      if (w > 0.01 && h > 0.01) {
        state.modelRoi = {
          x: Math.min(a.x, b.x),
          y: Math.min(a.y, b.y),
          w,
          h,
        };
      }
    }
    drag = null;
    paint();
  }

  function bindEditor() {
    if (editorBound) return;
    editorBound = true;
    document.addEventListener("wheel", onWheel, { passive: false });
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  }

  function resetView() {
    view = { zoom: 1, panX: 0, panY: 0 };
    paint();
  }

  // Own the global draw hook so app.js does not fight this editor.
  window.SVisionShapeModel = {
    ownsCanvas: true,
    draw: paint,
    resetView,
  };
  window.drawModelCanvas = paint;

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
        ? "Grab a still, zoom with the wheel, draw a box, then drag edges or corners to fine-tune before Register."
        : "Choose an image, zoom with the wheel (Shift-drag to pan), draw a box, then drag edges/corners to fine-tune.";
      return `
        <p class="hint">${hint}</p>
        <div class="sample-actions model-toolbar">
          ${live ? `<button id="grabModelCamera" type="button" class="primary">Grab from camera</button>` : ""}
          <label class="file-btn">Choose image<input id="modelFile" type="file" accept="image/*" hidden /></label>
          <button id="registerModel" type="button" ${live ? "" : 'class="primary"'}>Register</button>
          <button id="deleteModel" type="button">Delete</button>
          <button id="modelZoomIn" type="button" title="Zoom in">+</button>
          <button id="modelZoomOut" type="button" title="Zoom out">−</button>
          <button id="modelZoomReset" type="button" title="Reset view">1:1</button>
          <button id="modelClearRoi" type="button">Clear box</button>
        </div>
        <p id="grabModelStatus" class="hint">${live ? "Camera is live — Grab captures a still for the model." : ""}</p>
        <p id="modelRoiReadout" class="hint model-roi-readout">No selection</p>
        <div class="roi-stage model-stage"><canvas id="modelCanvas" width="720" height="420"></canvas></div>
        <div class="model-status ${saved ? "" : "warn"}">${saved ? `Registered model ${model.width}×${model.height}` : "Model image is not saved."}</div>
        ${saved ? `<img class="model-preview" src="${src}" alt="Registered model" />` : ""}`;
    },
    mount() {
      bindEditor();
      const zin = document.getElementById("modelZoomIn");
      const zout = document.getElementById("modelZoomOut");
      const zreset = document.getElementById("modelZoomReset");
      const clear = document.getElementById("modelClearRoi");
      if (zin) zin.onclick = () => setZoom(view.zoom * 1.25);
      if (zout) zout.onclick = () => setZoom(view.zoom / 1.25);
      if (zreset) zreset.onclick = () => resetView();
      if (clear) {
        clear.onclick = () => {
          state.modelRoi = null;
          paint();
        };
      }
      paint();
    },
  });

  T.registerDetail("shape_search", (tool) => {
    let detail = tool.message || "No shape";
    if (tool.x != null) detail += `  x ${tool.x}  y ${tool.y}  angle ${tool.angle}°`;
    if (tool.accel) detail += `  · ${tool.accel}`;
    return detail;
  });
})();
