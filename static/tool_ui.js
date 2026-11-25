// SVision inspection software
// Copyright (C) 2026 KD Puvvadi
//
// Pluggable UI hooks for inspection tools. Each tool registers its own
// panels and result detail renderers so editing one tool does not touch others.

(function (global) {
  const panels = Object.create(null);
  const details = Object.create(null);
  const fieldKinds = Object.create(null);
  const loaded = new Set();

  function registerPanel(name, spec) {
    if (!name || !spec) return;
    panels[name] = spec;
  }

  function registerDetail(type, render) {
    if (!type || typeof render !== "function") return;
    details[type] = render;
  }

  function registerFieldKind(kind, render) {
    if (!kind || typeof render !== "function") return;
    fieldKinds[kind] = render;
  }

  function panel(name) {
    return panels[name] || null;
  }

  function detail(type) {
    return details[type] || null;
  }

  function fieldKind(kind) {
    return fieldKinds[kind] || null;
  }

  function panelLabel(name, fallback) {
    const spec = panels[name];
    return (spec && spec.label) || fallback || name;
  }

  async function ensureModules(modules) {
    const list = [...new Set((modules || []).filter(Boolean))];
    await Promise.all(
      list.map((src) => {
        if (loaded.has(src)) return Promise.resolve();
        loaded.add(src);
        return new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = src.includes("?") ? src : `${src}?v=7`;
          script.async = false;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error(`Failed to load tool UI ${src}`));
          document.head.appendChild(script);
        });
      })
    );
  }

  global.SVisionTools = {
    registerPanel,
    registerDetail,
    registerFieldKind,
    panel,
    detail,
    fieldKind,
    panelLabel,
    ensureModules,
  };
})(window);
