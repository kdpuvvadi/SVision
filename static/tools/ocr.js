(function () {
  const T = window.SVisionTools;
  if (!T) return;
  T.registerDetail("ocr", (tool) => {
    let detail = tool.text ? `Read: ${tool.text}` : tool.message || "No text";
    if (tool.expected) detail += `  expected: ${tool.expected}`;
    return detail;
  });
})();
