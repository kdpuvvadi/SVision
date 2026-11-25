// SVision Shape Search tool UI — model register with zoom, rotate, edge/corner adjust.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  const HANDLE = 9;
  const EDGE = 7;
  const ROT_OFFSET = 28;
  let editorBound = false;
  let view = { zoom: 1, panX: 0, panY: 0 };
  let drag = null;
  let imgEl = null;

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function canvasEl() {
    return document.getElementById("modelCanvas");
  }

  function angleOf(roi) {
    return Number(roi?.angle || 0);
  }

  function setAngleInput(deg) {
    const el = document.getElementById("modelBoxAngle");
    if (el && document.activeElement !== el) el.value = String(Math.round(deg * 10) / 10);
  }

  function readAngleInput() {
    const el = document.getElementById("modelBoxAngle");
    return el ? Number(el.value) || 0 : 0;
  }

  function centerOf(roi) {
    return { cx: roi.x + roi.w / 2, cy: roi.y + roi.h / 2 };
  }

  function normPoint(e) {
    const canvas = canvasEl();
    if (!canvas || !canvas._fit) return null;
    const rect = canvas.getBoundingClientRect();
    const fit = canvas._fit;
    const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
    return {
      x: clamp((px - fit.x) / fit.w, -0.5, 1.5),
      y: clamp((py - fit.y) / fit.h, -0.5, 1.5),
      cx: px,
      cy: py,
    };
  }

  /** Screen-space geometry for a (possibly rotated) ROI. */
  function roiGeom(roi) {
    const canvas = canvasEl();
    const fit = canvas._fit;
    const cx = fit.x + (roi.x + roi.w / 2) * fit.w;
    const cy = fit.y + (roi.y + roi.h / 2) * fit.h;
    const bw = roi.w * fit.w;
    const bh = roi.h * fit.h;
    const ang = (angleOf(roi) * Math.PI) / 180;
    return { cx, cy, bw, bh, ang, cos: Math.cos(ang), sin: Math.sin(ang) };
  }

  function toLocal(p, geom) {
    const dx = p.cx - geom.cx;
    const dy = p.cy - geom.cy;
    return {
      lx: dx * geom.cos + dy * geom.sin,
      ly: -dx * geom.sin + dy * geom.cos,
    };
  }

  function handlePoints(geom) {
    const { cx, cy, bw, bh, cos, sin } = geom;
    const local = {
      nw: [-bw / 2, -bh / 2],
      n: [0, -bh / 2],
      ne: [bw / 2, -bh / 2],
      e: [bw / 2, 0],
      se: [bw / 2, bh / 2],
      s: [0, bh / 2],
      sw: [-bw / 2, bh / 2],
      w: [-bw / 2, 0],
      rot: [0, -bh / 2 - ROT_OFFSET],
    };
    const out = {};
    for (const [name, [lx, ly]] of Object.entries(local)) {
      out[name] = [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos];
    }
    return out;
  }

  function hitHandle(p, roi) {
    if (!roi || !p) return null;
    const geom = roiGeom(roi);
    const pts = handlePoints(geom);
    for (const [name, [hx, hy]] of Object.entries(pts)) {
      if (Math.abs(p.cx - hx) <= HANDLE && Math.abs(p.cy - hy) <= HANDLE) return name;
    }
    const { lx, ly } = toLocal(p, geom);
    const hw = geom.bw / 2;
    const hh = geom.bh / 2;
    if (Math.abs(ly + hh) <= EDGE && lx >= -hw && lx <= hw) return "n";
    if (Math.abs(ly - hh) <= EDGE && lx >= -hw && lx <= hw) return "s";
    if (Math.abs(lx + hw) <= EDGE && ly >= -hh && ly <= hh) return "w";
    if (Math.abs(lx - hw) <= EDGE && ly >= -hh && ly <= hh) return "e";
    if (lx >= -hw && lx <= hw && ly >= -hh && ly <= hh) return "move";
    return null;
  }

  function cursorFor(handle) {
    if (handle === "rot") return "grab";
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
    if (handle === "rot") {
      const fit = canvasEl()._fit;
      const c = centerOf(roi0);
      const sx = fit.x + c.cx * fit.w;
      const sy = fit.y + c.cy * fit.h;
      const a0 = Math.atan2(p0.cy - sy, p0.cx - sx);
      const a1 = Math.atan2(p.cy - sy, p.cx - sx);
      let deg = angleOf(roi0) + ((a1 - a0) * 180) / Math.PI;
      while (deg > 180) deg -= 360;
      while (deg < -180) deg += 360;
      return { ...roi0, angle: deg };
    }

    const geom0 = (() => {
      // Build geom from roi0 using current fit (same as paint).
      const fit = canvasEl()._fit;
      const cx = fit.x + (roi0.x + roi0.w / 2) * fit.w;
      const cy = fit.y + (roi0.y + roi0.h / 2) * fit.h;
      const bw = roi0.w * fit.w;
      const bh = roi0.h * fit.h;
      const ang = (angleOf(roi0) * Math.PI) / 180;
      return { cx, cy, bw, bh, ang, cos: Math.cos(ang), sin: Math.sin(ang), fit };
    })();

    const loc0 = toLocal(p0, geom0);
    const loc1 = toLocal(p, geom0);
    const dlx = loc1.lx - loc0.lx;
    const dly = loc1.ly - loc0.ly;

    let left = -geom0.bw / 2;
    let right = geom0.bw / 2;
    let top = -geom0.bh / 2;
    let bottom = geom0.bh / 2;

    if (handle === "move") {
      const dx = p.x - p0.x;
      const dy = p.y - p0.y;
      return {
        ...roi0,
        x: clamp(roi0.x + dx, 0, 1 - roi0.w),
        y: clamp(roi0.y + dy, 0, 1 - roi0.h),
      };
    }
    if (handle.includes("n")) top += dly;
    if (handle.includes("s")) bottom += dly;
    if (handle.includes("w")) left += dlx;
    if (handle.includes("e")) right += dlx;

    const minSpan = 8;
    if (right - left < minSpan) {
      if (handle.includes("w")) left = right - minSpan;
      else right = left + minSpan;
    }
    if (bottom - top < minSpan) {
      if (handle.includes("n")) top = bottom - minSpan;
      else bottom = top + minSpan;
    }

    const ncx = geom0.cx + ((left + right) / 2) * geom0.cos - ((top + bottom) / 2) * geom0.sin;
    const ncy = geom0.cy + ((left + right) / 2) * geom0.sin + ((top + bottom) / 2) * geom0.cos;
    const nbw = right - left;
    const nbh = bottom - top;
    const fit = geom0.fit;
    const w = nbw / fit.w;
    const h = nbh / fit.h;
    const cx = (ncx - fit.x) / fit.w;
    const cy = (ncy - fit.y) / fit.h;
    return {
      x: clamp(cx - w / 2, 0, 1),
      y: clamp(cy - h / 2, 0, 1),
      w: clamp(w, 0.005, 1),
      h: clamp(h, 0.005, 1),
      angle: angleOf(roi0),
    };
  }

  function updateReadout() {
    const el = document.getElementById("modelRoiReadout");
    if (!el) return;
    const roi = state.modelRoi;
    if (!roi) {
      el.textContent = "No selection — drag to draw, then drag edges/corners or the rotate handle.";
      return;
    }
    el.textContent =
      `x ${(roi.x * 100).toFixed(1)}%  y ${(roi.y * 100).toFixed(1)}%  ` +
      `w ${(roi.w * 100).toFixed(1)}%  h ${(roi.h * 100).toFixed(1)}%  ` +
      `∠ ${angleOf(roi).toFixed(1)}°  · zoom ${view.zoom.toFixed(2)}×`;
    setAngleInput(angleOf(roi));
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
      const geom = roiGeom(roi);
      const { cx, cy, bw, bh, ang } = geom;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(ang);
      ctx.fillStyle = "rgba(26, 212, 192, 0.12)";
      ctx.fillRect(-bw / 2, -bh / 2, bw, bh);
      ctx.strokeStyle = "#1ad4c0";
      ctx.lineWidth = 2;
      ctx.strokeRect(-bw / 2, -bh / 2, bw, bh);
      // Orientation tick (top)
      ctx.beginPath();
      ctx.moveTo(0, -bh / 2);
      ctx.lineTo(0, -bh / 2 - ROT_OFFSET);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, -bh / 2 - ROT_OFFSET, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffb020";
      ctx.fill();
      ctx.strokeStyle = "#1ad4c0";
      ctx.stroke();
      // Local-space handles
      const locals = [
        [-bw / 2, -bh / 2],
        [0, -bh / 2],
        [bw / 2, -bh / 2],
        [bw / 2, 0],
        [bw / 2, bh / 2],
        [0, bh / 2],
        [-bw / 2, bh / 2],
        [-bw / 2, 0],
      ];
      ctx.fillStyle = "#1ad4c0";
      locals.forEach(([hx, hy]) => ctx.fillRect(hx - 4, hy - 4, 8, 8));
      ctx.restore();
    };

    const finish = () => {
      const roi =
        drag && drag.mode === "draw"
          ? {
              x: Math.min(drag.start.x, drag.cur.x),
              y: Math.min(drag.start.y, drag.cur.y),
              w: Math.abs(drag.cur.x - drag.start.x),
              h: Math.abs(drag.cur.y - drag.start.y),
              angle: 0,
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
        roi0: { ...state.modelRoi, angle: angleOf(state.modelRoi) },
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
          angle: readAngleInput() || 0,
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
      const ang = state.modelRoi ? angleOf(state.modelRoi) : 0;
      const hint = live
        ? "Grab a still and draw a tight box around the part only (not the whole frame). Rotate with the orange handle if needed, then Register."
        : "Choose an image and draw a tight box around the part only. Avoid selecting most of the frame — the overlay follows the registered model.";
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
        <div class="roi-row model-angle-row">
          <span>Angle</span>
          <input id="modelBoxAngle" type="number" min="-180" max="180" step="0.5" value="${ang}" />
          <span class="hint">degrees · drag the orange handle to rotate</span>
        </div>
        <p id="grabModelStatus" class="hint">${live ? "Camera is live — Grab captures a still for the model." : ""}</p>
        <p id="modelRoiReadout" class="hint model-roi-readout">No selection</p>
        <div class="roi-stage model-stage"><canvas id="modelCanvas" width="720" height="420"></canvas></div>
        <div class="model-status ${saved ? (model.warning ? "warn" : "") : "warn"}">${saved ? `Registered model ${model.width}×${model.height}${model.warning ? ` — ${model.warning}` : ""}` : "Model image is not saved."}</div>
        ${saved ? `<img class="model-preview" src="${src}" alt="Registered model" />` : ""}`;
    },
    mount() {
      bindEditor();
      const zin = document.getElementById("modelZoomIn");
      const zout = document.getElementById("modelZoomOut");
      const zreset = document.getElementById("modelZoomReset");
      const clear = document.getElementById("modelClearRoi");
      const ang = document.getElementById("modelBoxAngle");
      if (zin) zin.onclick = () => setZoom(view.zoom * 1.25);
      if (zout) zout.onclick = () => setZoom(view.zoom / 1.25);
      if (zreset) zreset.onclick = () => resetView();
      if (clear) {
        clear.onclick = () => {
          state.modelRoi = null;
          paint();
        };
      }
      if (ang) {
        ang.oninput = () => {
          if (!state.modelRoi) return;
          state.modelRoi = { ...state.modelRoi, angle: Number(ang.value) || 0 };
          paint();
        };
      }
      paint();
    },
  });

  T.registerDetail("shape_search", (tool) => {
    let detail = tool.message || "No shape";
    if (tool.x != null) detail += `  x ${tool.x}  y ${tool.y}  angle ${tool.angle}°`;
    if (tool.method) detail += `  · ${tool.method}`;
    if (tool.accel) detail += `  · ${tool.accel}`;
    return detail;
  });
})();
