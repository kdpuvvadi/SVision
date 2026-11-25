// SVision Robotic tool UI — isolated panel + detail registration.
(function () {
  const T = window.SVisionTools;
  if (!T) return;

  T.registerPanel("robot_register", {
    label: "Register",
    render(tool) {
      return typeof robotRegisterPanel === "function" ? robotRegisterPanel(tool) : "";
    },
    mount(tool) {
      if (typeof initRobotRegisterPanel === "function") initRobotRegisterPanel(tool);
    },
  });

  T.registerPanel("robot_calib", {
    label: "Calibrate",
    render(tool) {
      return typeof robotCalibPanel === "function" ? robotCalibPanel(tool) : "";
    },
    mount(tool) {
      if (typeof initRobotCalibPanel === "function") initRobotCalibPanel(tool);
    },
  });

  T.registerDetail("robotic", (tool) => {
    const objs = tool.objects || [];
    let detail = objs.length
      ? objs
          .slice(0, 4)
          .map(
            (o) =>
              `#${o.id} ${o.class_name || ""} (${Number(o.x).toFixed(0)},${Number(o.y).toFixed(0)},a=${Number(o.a).toFixed(1)})`
          )
          .join(" · ")
      : tool.message || "No objects";
    if (objs.length > 4) detail += ` · +${objs.length - 4}`;
    return detail;
  });
})();
