# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.config import THREAD_COUNT
from app.core.classifier import encode_jpeg
from app.storage.store import store
from app.tools.registry import get_tool

# Separate pools so a batch job never waits on the same workers it needs for tools.
_tool_pool = ThreadPoolExecutor(max_workers=max(3, THREAD_COUNT), thread_name_prefix="svision-tool")
_batch_pool = ThreadPoolExecutor(max_workers=THREAD_COUNT, thread_name_prefix="svision-batch")


def _fit_preview(image: np.ndarray, limit: int = 720) -> np.ndarray:
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= limit:
        return image
    scale = limit / long_edge
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _mark_pose(view: np.ndarray, result: dict, src_w: int, src_h: int) -> np.ndarray:
    if result.get("x") is None or not result.get("width") or not result.get("height") or src_w <= 0 or src_h <= 0:
        return view
    marked = view.copy()
    sx = marked.shape[1] / src_w
    sy = marked.shape[0] / src_h
    x = float(result["x"])
    y = float(result["y"])
    rw = float(result["width"])
    rh = float(result["height"])
    color = (40, 220, 160) if result.get("judgment") == "OK" else (60, 60, 230)
    x0, y0 = int((x - rw / 2.0) * sx), int((y - rh / 2.0) * sy)
    x1, y1 = int((x + rw / 2.0) * sx), int((y + rh / 2.0) * sy)
    cv2.rectangle(marked, (x0, y0), (x1, y1), color, 2)
    cv2.drawMarker(marked, (int(x * sx), int(y * sy)), color, cv2.MARKER_CROSS, 14, 2)
    return marked


def _tool_roi(spec: dict) -> dict | None:
    cfg = spec.get("config") or {}
    roi = cfg.get("roi") or cfg.get("search_roi")
    if not isinstance(roi, dict):
        return None
    try:
        width = float(roi.get("w", 0))
        height = float(roi.get("h", 0))
    except (TypeError, ValueError):
        return None
    if width <= 0.01 or height <= 0.01:
        return None
    return roi


def _draw_region(view: np.ndarray, roi: dict | None, src_w: int, src_h: int) -> np.ndarray:
    if not roi or src_w <= 0 or src_h <= 0:
        return view
    marked = view.copy()
    sx = marked.shape[1] / src_w
    sy = marked.shape[0] / src_h
    x = float(roi.get("x", 0))
    y = float(roi.get("y", 0))
    rw = float(roi.get("w", 1))
    rh = float(roi.get("h", 1))
    if max(x, y, rw, rh) <= 1.5:
        x0, y0 = int(x * src_w * sx), int(y * src_h * sy)
        x1, y1 = int((x + rw) * src_w * sx), int((y + rh) * src_h * sy)
    else:
        x0, y0 = int(x * sx), int(y * sy)
        x1, y1 = int((x + rw) * sx), int((y + rh) * sy)
    color = (40, 220, 220)
    thick = max(2, int(round(min(marked.shape[:2]) / 280)))
    cv2.rectangle(marked, (x0, y0), (x1, y1), color, thick)
    return marked


def _tool_preview(image: np.ndarray | None, result: dict, mark: bool, roi: dict | None = None) -> bytes:
    if image is None or image.size == 0:
        return b""
    src_h, src_w = image.shape[:2]
    view = _fit_preview(image)
    if mark:
        view = _mark_pose(view, result, src_w, src_h)
    view = _draw_region(view, roi, src_w, src_h)
    return encode_jpeg(view, 70)


def _step_image(spec: dict, source: np.ndarray, seen: np.ndarray, heat: np.ndarray | None) -> np.ndarray | None:
    if spec.get("type") == "camera":
        return source
    if heat is not None:
        return heat
    return seen


def _attach_preview(result: dict, image: np.ndarray | None, mark: bool, spec: dict | None = None) -> None:
    try:
        result["preview_jpeg"] = _tool_preview(image, result, mark, _tool_roi(spec or {}))
    except Exception:
        result["preview_jpeg"] = b""


def _annotate(image: np.ndarray, judgment: str, elapsed_ms: float, roi: dict | None) -> np.ndarray:
    view = image.copy()
    if roi:
        h, w = view.shape[:2]
        x = float(roi.get("x", 0))
        y = float(roi.get("y", 0))
        rw = float(roi.get("w", 1))
        rh = float(roi.get("h", 1))
        if max(x, y, rw, rh) <= 1.5:
            x0, y0 = int(x * w), int(y * h)
            x1, y1 = int((x + rw) * w), int((y + rh) * h)
        else:
            x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 220, 220), 2)

    color = (40, 220, 160) if judgment == "OK" else (60, 60, 230)
    label = f"{judgment}  {elapsed_ms:.1f} ms"
    cv2.rectangle(view, (0, 0), (min(view.shape[1], 280), 42), color, -1)
    cv2.putText(view, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def _run_one(project_id: str, image: np.ndarray, tool: dict, ctx: dict) -> tuple[dict, np.ndarray | None]:
    module = get_tool(tool["type"])
    return module.run(image, tool, project_id, ctx)


def _camera_tool(project: dict) -> dict | None:
    for tool in (project.get("flow") or {}).get("tools", []):
        if tool.get("type") == "camera":
            return tool
    return None


def inspect_array(project_id: str, image: np.ndarray, filename: str = "image") -> dict:
    started = time.perf_counter()
    project = store.get_project(project_id)
    camera = _camera_tool(project)
    if camera is None:
        raise ValueError("Add a Camera tool to the flow before inspecting.")
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("No image for inspection.")
    active = []
    for tool in (project.get("flow") or {}).get("tools", []):
        try:
            get_tool(tool.get("type", ""))
        except KeyError:
            continue
        active.append(tool)

    working = image
    ctx: dict = {"pose": None, "image": image}
    tools: list[dict] = []
    overlay = None
    index = 0
    while index < len(active):
        group = []
        while index < len(active) and not get_tool(active[index]["type"]).transforms_image:
            group.append(active[index])
            index += 1
        seen = working
        if group:
            futures = [_tool_pool.submit(_run_one, project_id, seen, tool, ctx) for tool in group]
            for spec, future in zip(group, futures):
                result, heat = future.result()
                if result.get("pose"):
                    ctx["pose"] = result["pose"]
                _attach_preview(result, _step_image(spec, image, seen, heat), heat is None and spec.get("type") != "camera", spec)
                tools.append(result)
                if heat is not None:
                    overlay = heat
        if index < len(active):
            spec = active[index]
            index += 1
            result, heat = _run_one(project_id, seen, spec, ctx)
            _attach_preview(result, _step_image(spec, image, seen, heat), heat is None and spec.get("type") != "camera", spec)
            tools.append(result)
            if ctx.get("image") is not None:
                working = ctx["image"]
            if heat is not None:
                overlay = heat

    if not tools:
        overall = "NG"
        message = "Flow is empty. Add tools in Edit flow."
    else:
        overall = "OK" if all(item.get("judgment") == "OK" for item in tools) else "NG"
        message = "All tools passed." if overall == "OK" else "One or more tools failed."

    elapsed = (time.perf_counter() - started) * 1000.0
    preview_src = overlay if overlay is not None else working
    preview = _annotate(preview_src, overall, elapsed, None)

    return {
        "filename": filename,
        "judgment": overall,
        "message": message,
        "elapsed_ms": elapsed,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "tools": tools,
        "preview_jpeg": encode_jpeg(preview, 82),
    }


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File is not a readable image.")
    return image


def inspect_many(project_id: str, files: list[tuple[str, bytes]]) -> dict:
    started = time.perf_counter()
    if not files:
        raise ValueError("No images uploaded.")

    def _one(item: tuple[str, bytes]) -> dict:
        name, raw = item
        return inspect_array(project_id, decode_image(raw), name)

    if len(files) == 1:
        results = [_one(files[0])]
    else:
        results = list(_batch_pool.map(_one, files))

    ok_count = sum(1 for item in results if item["judgment"] == "OK")
    return {
        "count": len(results),
        "ok": ok_count,
        "ng": len(results) - ok_count,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "results": results,
    }
