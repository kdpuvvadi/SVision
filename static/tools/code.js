(function () {
  const T = window.SVisionTools;
  if (!T) return;
  T.registerDetail("code", (tool) => {
    let detail = tool.codes?.length
      ? tool.codes.map((c) => `${c.type}: ${c.value}`).join(", ")
      : tool.message || "No code";
    if (tool.expected) detail += `  expected: ${tool.expected}`;
    return detail;
  });
})();
