// SVision Shape Search — Omron-style model editor (region → figures → save).
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  const TOOLS = [
    { id: "rect", label: "▭", title: "Rectangle" },
    { id: "circle", label: "○", title: "Circle" },
    { id: "donut", label: "◎", title: "Double circle" },
    { id: "polygon", label: "⬠", title: "Polygon" },
  ];

  let modalBound = false;
  let editor = {
    open: false,
    step: "region", // region | figures
    tool: "rect",
    region: null,
    figures: [],
    selected: -1,
    polyDraft: null,
    drag: null,
    view: { zoom: 1, panX: 0, panY: 0 },
    img: null,
  };

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function canvasEl() {
    return document.getElementById("modelEditCanvas");
  }

  function statusEl() {
    return document.getElementById("modelEditStatus");
  }

  function setStatus(msg) {
    const el = statusEl();
    if (el) el.textContent = msg || "";
  }

  function modalEl() {
    return document.getElementById("modelModal");
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

  function figureLabel(fig) {
    const map = { rect: "Rectangle", circle: "Circle", donut: "Double circle", polygon: "Polygon" };
    return map[fig.type] || fig.type;
  }

  function renderFigureList() {
    const list = document.getElementById("modelFigureList");
    if (!list) return;
    if (!editor.figures.length) {
      list.innerHTML = `<li class="hint">No figures yet</li>`;
      return;
    }
    list.innerHTML = editor.figures
      .map(
        (fig, i) =>
          `<li class="${i === editor.selected ? "selected" : ""}" data-fig="${i}">${i + 1}. ${figureLabel(fig)}</li>`
      )
      .join("");
  }

  function syncStepUi() {
    const stepRegion = document.getElementById("modelStepRegion");
    const stepFigures = document.getElementById("modelStepFigures");
    const nextBtn = document.getElementById("modelEditNext");
    const backBtn = document.getElementById("modelEditBack");
    const saveBtn = document.getElementById("modelEditSave");
    const tools = document.getElementById("modelFigureTools");
    if (stepRegion) stepRegion.classList.toggle("active", editor.step === "region");
    if (stepFigures) stepFigures.classList.toggle("active", editor.step === "figures");
    if (nextBtn) nextBtn.hidden = editor.step !== "region";
    if (backBtn) backBtn.hidden = editor.step !== "figures";
    if (saveBtn) saveBtn.hidden = editor.step !== "figures";
    if (tools) tools.hidden = editor.step !== "figures";
    document.querySelectorAll("[data-fig-tool]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.figTool === editor.tool);
    });
    renderFigureList();
    paint();
  }

  function paint() {
    const canvas = canvasEl();
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0a0c0e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = editor.img;
    if (!img || !img.naturalWidth) {
      ctx.fillStyle = "#9aa6b2";
      ctx.fillText("Update image first", 24, 40);
      canvas._fit = null;
      return;
    }

    let src = img;
    let sx = 0;
    let sy = 0;
    let sw = img.naturalWidth;
    let sh = img.naturalHeight;
    // In figures step, zoom into the selected region.
    if (editor.step === "figures" && editor.region) {
      sx = editor.region.x * img.naturalWidth;
      sy = editor.region.y * img.naturalHeight;
      sw = editor.region.w * img.naturalWidth;
      sh = editor.region.h * img.naturalHeight;
    }
    const scale = Math.min(canvas.width / sw, canvas.height / sh) * editor.view.zoom;
    const fw = sw * scale;
    const fh = sh * scale;
    const fx = (canvas.width - fw) / 2 + editor.view.panX;
    const fy = (canvas.height - fh) / 2 + editor.view.panY;
    canvas._fit = { x: fx, y: fy, w: fw, h: fh, sx, sy, sw, sh };
    ctx.drawImage(img, sx, sy, sw, sh, fx, fy, fw, fh);

    const strokeFig = (fig, selected) => {
      ctx.save();
      ctx.strokeStyle = selected ? "#f0c14a" : "#1ad4c0";
      ctx.lineWidth = selected ? 2.5 : 2;
      ctx.fillStyle = selected ? "rgba(240,193,74,0.12)" : "rgba(26,212,192,0.10)";
      if (fig.type === "rect") {
        const x = fx + fig.x * fw;
        const y = fy + fig.y * fh;
        const w = fig.w * fw;
        const h = fig.h * fh;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      } else if (fig.type === "circle") {
        const cx = fx + fig.cx * fw;
        const cy = fy + fig.cy * fh;
        const r = fig.r * Math.min(fw, fh);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else if (fig.type === "donut") {
        const cx = fx + fig.cx * fw;
        const cy = fy + fig.cy * fh;
        const ro = fig.r_outer * Math.min(fw, fh);
        const ri = fig.r_inner * Math.min(fw, fh);
        ctx.beginPath();
        ctx.arc(cx, cy, ro, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, ri, 0, Math.PI * 2);
        ctx.stroke();
      } else if (fig.type === "polygon" && fig.points?.length) {
        ctx.beginPath();
        fig.points.forEach((p, i) => {
          const x = fx + p[0] * fw;
          const y = fy + p[1] * fh;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        if (fig.points.length >= 3) ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    };

    if (editor.step === "region") {
      const roi = editor.region || (editor.drag?.mode === "draw" ? dragRect(editor.drag) : null);
      if (roi) {
        ctx.fillStyle = "rgba(26,212,192,0.12)";
        ctx.strokeStyle = "#1ad4c0";
        ctx.lineWidth = 2;
        ctx.fillRect(fx + roi.x * fw, fy + roi.y * fh, roi.w * fw, roi.h * fh);
        ctx.strokeRect(fx + roi.x * fw, fy + roi.y * fh, roi.w * fw, roi.h * fh);
      }
    } else {
      editor.figures.forEach((fig, i) => strokeFig(fig, i === editor.selected));
      if (editor.polyDraft?.length) {
        strokeFig({ type: "polygon", points: editor.polyDraft }, true);
      }
      if (editor.drag?.mode === "draw-fig" && editor.drag.cur) {
        const draft = draftFigure(editor.drag);
        if (draft) strokeFig(draft, true);
      }
    }

    // Navigator thumbnail hint when in figures step
    const nav = document.getElementById("modelNavCanvas");
    if (nav && img.naturalWidth) {
      const nctx = nav.getContext("2d");
      nctx.fillStyle = "#0a0c0e";
      nctx.fillRect(0, 0, nav.width, nav.height);
      const ns = Math.min(nav.width / img.naturalWidth, nav.height / img.naturalHeight);
      const nw = img.naturalWidth * ns;
      const nh = img.naturalHeight * ns;
      const nx = (nav.width - nw) / 2;
      const ny = (nav.height - nh) / 2;
      nctx.drawImage(img, nx, ny, nw, nh);
      if (editor.region) {
        nctx.strokeStyle = "#ff4d5a";
        nctx.lineWidth = 2;
        nctx.strokeRect(
          nx + editor.region.x * nw,
          ny + editor.region.y * nh,
          editor.region.w * nw,
          editor.region.h * nh
        );
      }
    }
  }

  function dragRect(drag) {
    const a = drag.start;
    const b = drag.cur;
    return {
      x: Math.min(a.x, b.x),
      y: Math.min(a.y, b.y),
      w: Math.abs(b.x - a.x),
      h: Math.abs(b.y - a.y),
    };
  }

  function draftFigure(drag) {
    const a = drag.start;
    const b = drag.cur;
    if (editor.tool === "rect") {
      return {
        type: "rect",
        x: Math.min(a.x, b.x),
        y: Math.min(a.y, b.y),
        w: Math.abs(b.x - a.x),
        h: Math.abs(b.y - a.y),
      };
    }
    if (editor.tool === "circle" || editor.tool === "donut") {
      const cx = (a.x + b.x) / 2;
      const cy = (a.y + b.y) / 2;
      const r = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)) / 2;
      if (editor.tool === "donut") {
        return { type: "donut", cx, cy, r_outer: r, r_inner: r * 0.55 };
      }
      return { type: "circle", cx, cy, r };
    }
    return null;
  }

  function onDown(e) {
    if (!editor.open) return;
    const canvas = canvasEl();
    if (!canvas || e.target !== canvas) return;
    const p = normPoint(e);
    if (!p) return;
    e.preventDefault();
    if (e.button === 1 || e.shiftKey) {
      editor.drag = { mode: "pan", startCx: p.cx, startCy: p.cy, pan0: { ...editor.view } };
      return;
    }
    if (editor.step === "region") {
      editor.drag = { mode: "draw", start: p, cur: p };
      return;
    }
    if (editor.tool === "polygon") {
      if (!editor.polyDraft) editor.polyDraft = [];
      editor.polyDraft.push([p.x, p.y]);
      if (e.detail === 2 && editor.polyDraft.length >= 3) {
        editor.figures.push({ type: "polygon", points: editor.polyDraft.slice() });
        editor.selected = editor.figures.length - 1;
        editor.polyDraft = null;
        renderFigureList();
      }
      paint();
      return;
    }
    editor.drag = { mode: "draw-fig", start: p, cur: p };
  }

  function onMove(e) {
    if (!editor.open || !editor.drag) return;
    const p = normPoint(e);
    if (!p) return;
    if (editor.drag.mode === "pan") {
      editor.view.panX = editor.drag.pan0.panX + (p.cx - editor.drag.startCx);
      editor.view.panY = editor.drag.pan0.panY + (p.cy - editor.drag.startCy);
      paint();
      return;
    }
    editor.drag.cur = p;
    paint();
  }

  function onUp() {
    if (!editor.open || !editor.drag) return;
    if (editor.drag.mode === "draw") {
      const roi = dragRect(editor.drag);
      if (roi.w > 0.01 && roi.h > 0.01) editor.region = roi;
    } else if (editor.drag.mode === "draw-fig") {
      const fig = draftFigure(editor.drag);
      if (fig) {
        const ok =
          (fig.type === "rect" && fig.w > 0.01 && fig.h > 0.01) ||
          (fig.type === "circle" && fig.r > 0.01) ||
          (fig.type === "donut" && fig.r_outer > 0.01);
        if (ok) {
          editor.figures.push(fig);
          editor.selected = editor.figures.length - 1;
          renderFigureList();
        }
      }
    }
    editor.drag = null;
    paint();
  }

  function onWheel(e) {
    if (!editor.open) return;
    const canvas = canvasEl();
    if (!canvas || (e.target !== canvas && !canvas.contains(e.target))) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    editor.view.zoom = clamp(editor.view.zoom * factor, 0.5, 8);
    paint();
  }

  function bindModal() {
    if (modalBound) return;
    modalBound = true;
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
    document.addEventListener("wheel", onWheel, { passive: false });
  }

  async function loadPipelineImage() {
    const tool = typeof selectedTool === "function" ? selectedTool() : null;
    if (!tool || !state.project?.id) throw new Error("Select Shape Search in the flow first.");
    setStatus("Updating from previous tools…");
    const res = await fetch(
      `/api/projects/${state.project.id}/tools/${tool.id}/pipeline-image?fresh=1`,
      { method: "POST" }
    );
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || res.statusText || "Update failed.");
    }
    const blob = await res.blob();
    const file = new File([blob], "pipeline.jpg", { type: "image/jpeg" });
    state.modelFile = file;
    if (state.modelUrl) URL.revokeObjectURL(state.modelUrl);
    state.modelUrl = URL.createObjectURL(file);
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        editor.img = img;
        resolve();
      };
      img.onerror = () => reject(new Error("Could not load pipeline image."));
      img.src = state.modelUrl;
    });
    setStatus(`Updated ${new Date().toLocaleTimeString()} · ${editor.img.naturalWidth}×${editor.img.naturalHeight}`);
    paint();
  }

  async function openEditor() {
    bindModal();
    editor.open = true;
    editor.step = "region";
    editor.tool = "rect";
    editor.region = state.modelRoi;
    editor.figures = [];
    editor.selected = -1;
    editor.polyDraft = null;
    editor.view = { zoom: 1, panX: 0, panY: 0 };
    const modal = modalEl();
    if (modal) modal.hidden = false;
    syncStepUi();
    try {
      if (state.modelUrl) {
        await new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => {
            editor.img = img;
            resolve();
          };
          img.onerror = reject;
          img.src = state.modelUrl;
        });
        paint();
      } else {
        await loadPipelineImage();
      }
    } catch (err) {
      setStatus(err.message || String(err));
    }
  }

  function closeEditor() {
    editor.open = false;
    const modal = modalEl();
    if (modal) modal.hidden = true;
  }

  async function saveModel() {
    const tool = typeof selectedTool === "function" ? selectedTool() : null;
    if (!tool || !state.modelFile) {
      setStatus("Update the image first.");
      return;
    }
    if (!editor.region || editor.region.w < 0.01 || editor.region.h < 0.01) {
      setStatus("Draw a region first.");
      return;
    }
    const figures = editor.figures.length
      ? editor.figures
      : [{ type: "rect", x: 0, y: 0, w: 1, h: 1 }];
    setStatus("Saving solid model…");
    const body = new FormData();
    body.append("file", state.modelFile);
    body.append("x", editor.region.x);
    body.append("y", editor.region.y);
    body.append("w", editor.region.w);
    body.append("h", editor.region.h);
    body.append("angle", 0);
    body.append("figures", JSON.stringify(figures));
    await api(`/api/projects/${state.project.id}/tools/${tool.id}/shape-model`, { method: "POST", body });
    state.modelRoi = editor.region;
    state.settingTab = "panel:shape_model";
    await refreshSelectedTool();
    closeEditor();
  }

  function deleteSelectedFigure() {
    if (editor.selected < 0) return;
    editor.figures.splice(editor.selected, 1);
    editor.selected = Math.min(editor.selected, editor.figures.length - 1);
    renderFigureList();
    paint();
  }

  T.registerPanel("shape_model", {
    label: "Model",
    render(tool) {
      const model = tool.state?.shape_model || {};
      const saved = !!model.saved;
      const solidBits = saved
        ? `${model.figure_count || 0} figures · ${model.edge_points || 0} edge pts · ${model.width}×${model.height}`
        : "";
      const src = saved
        ? `/api/projects/${state.project.id}/tools/${tool.id}/shape-model?t=${encodeURIComponent(model.registered_at || "")}`
        : "";
      return `
        <div class="model-editor">
          <section class="model-block">
            <h3>Solid model</h3>
            <p class="hint">Open the model editor to select a region, draw figures (rectangle, circle, double circle, polygon), then save. <b>Update</b> pulls the image through every tool before this one.</p>
            <div class="sample-actions model-toolbar">
              <button id="openModelEditor" type="button" class="primary">Open model editor</button>
              <button id="deleteModel" type="button">Delete</button>
            </div>
            <div class="model-status ${saved ? (model.warning ? "warn" : "ok") : "warn"}">
              ${saved
                ? `Solid edge model saved · ${solidBits}${model.warning ? ` — ${escapeHtml(model.warning)}` : ""}`
                : "No solid model yet."}
            </div>
            ${saved ? `<img class="model-preview model-preview-lg" src="${src}" alt="Registered solid model" />` : ""}
          </section>
        </div>`;
    },
    mount() {},
  });

  T.registerDetail("shape_search", (tool) => {
    if (tool.judgment == null && tool.confidence == null) return tool.message || "No measurement";
    const corr =
      tool.correlation != null
        ? Number(tool.correlation).toFixed(1)
        : (Number(tool.confidence || 0) * 100).toFixed(1);
    const bits = [`Judge ${tool.judgment || "—"}`, `Correlation ${corr}`];
    if (tool.x != null) bits.push(`X ${tool.x}`);
    if (tool.y != null) bits.push(`Y ${tool.y}`);
    if (tool.angle != null) bits.push(`Angle ${tool.angle}`);
    return bits.join("  ·  ");
  });

  // Modal chrome wiring (elements live in index.html).
  function wireModalChrome() {
    const updateBtn = document.getElementById("modelEditUpdate");
    const nextBtn = document.getElementById("modelEditNext");
    const backBtn = document.getElementById("modelEditBack");
    const saveBtn = document.getElementById("modelEditSave");
    const cancelBtn = document.getElementById("modelEditCancel");
    const delFig = document.getElementById("modelFigDelete");
    const modal = modalEl();
    if (updateBtn) {
      updateBtn.onclick = () => loadPipelineImage().catch((err) => setStatus(err.message || err));
    }
    if (nextBtn) {
      nextBtn.onclick = () => {
        if (!editor.region || editor.region.w < 0.01) {
          setStatus("Draw a region on the image first.");
          return;
        }
        editor.step = "figures";
        editor.view = { zoom: 1, panX: 0, panY: 0 };
        if (!editor.figures.length) {
          editor.figures.push({ type: "rect", x: 0.05, y: 0.05, w: 0.9, h: 0.9 });
          editor.selected = 0;
        }
        syncStepUi();
        setStatus("Draw figures over the part, then Save model.");
      };
    }
    if (backBtn) {
      backBtn.onclick = () => {
        editor.step = "region";
        editor.view = { zoom: 1, panX: 0, panY: 0 };
        syncStepUi();
      };
    }
    if (saveBtn) saveBtn.onclick = () => saveModel().catch((err) => setStatus(err.message || err));
    if (cancelBtn) cancelBtn.onclick = () => closeEditor();
    if (delFig) delFig.onclick = () => deleteSelectedFigure();
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target.id === "modelModal") closeEditor();
      });
    }
    document.getElementById("modelFigureTools")?.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-fig-tool]");
      if (!btn) return;
      editor.tool = btn.dataset.figTool;
      editor.polyDraft = null;
      syncStepUi();
    });
    document.getElementById("modelFigureList")?.addEventListener("click", (e) => {
      const li = e.target.closest("[data-fig]");
      if (!li) return;
      editor.selected = Number(li.dataset.fig);
      renderFigureList();
      paint();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wireModalChrome);
  } else {
    wireModalChrome();
  }

  window.SVisionShapeModel = {
    ownsCanvas: true,
    open: openEditor,
    close: closeEditor,
    update: loadPipelineImage,
    tools: TOOLS,
  };
})();
