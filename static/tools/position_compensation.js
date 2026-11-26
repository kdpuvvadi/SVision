(function () {
  const T = window.SVisionTools;
  if (!T) return;
  T.registerDetail("position_compensation", (tool) => {
    if (tool.judgment == null) return tool.message || "No measurement";
    const bits = [`Judge ${tool.judgment || "—"}`];
    if (tool.width != null && tool.height != null) bits.push(`${tool.width}×${tool.height}`);
    if (tool.offset_x != null || tool.offset_y != null) {
      bits.push(`Δ ${Number(tool.offset_x || 0).toFixed(1)}, ${Number(tool.offset_y || 0).toFixed(1)}`);
    }
    if (tool.angle != null) bits.push(`∠ ${tool.angle}`);
    return bits.join("  ·  ");
  });
})();
