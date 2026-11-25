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

import cv2
import numpy as np

from app.core.features import _roi_box
from app.tools.base import Tool

_MAX_EDGE = 480


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 40, 120)


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += (nw - w) / 2.0
    matrix[1, 2] += (nh - h) / 2.0
    return cv2.warpAffine(gray, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def _small(image: np.ndarray, max_edge: int = _MAX_EDGE) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image, 1.0
    scale = max_edge / long_edge
    out = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return out, scale


def _nms(hits: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: float(h.get("confidence") or 0), reverse=True)
    kept: list[dict] = []
    for hit in ordered:
        box = hit["box"]
        if any(_iou(box, k["box"]) > iou_thresh for k in kept):
            continue
        kept.append(hit)
    return kept


def _iou(a: list[list[float]], b: list[list[float]]) -> float:
    ax = [p[0] for p in a]
    ay = [p[1] for p in a]
    bx = [p[0] for p in b]
    by = [p[1] for p in b]
    ax0, ay0, ax1, ay1 = min(ax), min(ay), max(ax), max(ay)
    bx0, by0, bx1, by1 = min(bx), min(by), max(bx), max(by)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def _rect_box(cx: float, cy: float, w: float, h: float, angle: float) -> list[list[float]]:
    rect = ((cx, cy), (w, h), angle)
    pts = cv2.boxPoints(rect)
    return [[float(p[0]), float(p[1])] for p in pts]


def apply_homography(x: float, y: float, hmat: np.ndarray | None) -> tuple[float | None, float | None]:
    if hmat is None:
        return None, None
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, hmat)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


class RoboticTool(Tool):
    type = "robotic"
    title = "Robotic"
    summary = "Detect objects for robot pick: template or contour, centers, angles, world calib, WebSocket stream."

    def fields(self) -> list[dict]:
        return [
            {
                "key": "mode",
                "label": "Detect mode",
                "kind": "select",
                "options": [
                    {"value": "template", "label": "Template"},
                    {"value": "contour", "label": "Contour"},
                ],
            },
            {
                "key": "class_mode",
                "label": "Class mode",
                "kind": "select",
                "options": [
                    {"value": "single", "label": "Single"},
                    {"value": "multi", "label": "Multi"},
                ],
            },
            {"key": "model_name", "label": "Model / class name", "kind": "text", "placeholder": "Part"},
            {
                "key": "angle_enable",
                "label": "Angle enable",
                "kind": "select",
                "options": [
                    {"value": "yes", "label": "On"},
                    {"value": "no", "label": "Off"},
                ],
            },
            {"key": "search_roi", "label": "Search region", "kind": "roi"},
            {
                "key": "min_confidence",
                "label": "Minimum confidence",
                "kind": "range",
                "min": 0.4,
                "max": 0.95,
                "step": 0.01,
            },
            {
                "key": "min_area",
                "label": "Min contour area (px)",
                "kind": "range",
                "min": 50,
                "max": 20000,
                "step": 50,
            },
            {
                "key": "min_count",
                "label": "Minimum objects",
                "kind": "range",
                "min": 1,
                "max": 20,
                "step": 1,
            },
            {
                "key": "stream_mode",
                "label": "WebSocket stream",
                "kind": "select",
                "options": [
                    {"value": "on_trigger", "label": "On trigger"},
                    {"value": "continuous", "label": "Continuous"},
                ],
            },
            {
                "key": "stream_hz",
                "label": "Continuous Hz",
                "kind": "range",
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "angle_offset",
                "label": "Robot angle offset",
                "kind": "range",
                "min": -180,
                "max": 180,
                "step": 1,
            },
        ]

    def panels(self) -> list[str]:
        return ["robot_register", "robot_calib"]

    def default_config(self) -> dict:
        return {
            "mode": "template",
            "class_mode": "single",
            "model_name": "Part",
            "angle_enable": "yes",
            "search_roi": None,
            "min_confidence": 0.6,
            "min_area": 400,
            "min_count": 1,
            "stream_mode": "on_trigger",
            "stream_hz": 5,
            "angle_offset": 0,
        }

    def status(self) -> str:
        return "Register templates (single or multi class), or use contour mode. Calibrate 4 points for robot XY. Stream over WebSocket."

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        from app.storage.store import store

        started = time.perf_counter()
        cfg = tool.get("config") or {}
        mode = str(cfg.get("mode") or "template")
        angle_enable = str(cfg.get("angle_enable") or "yes") == "yes"
        min_count = max(1, int(float(cfg.get("min_count") or 1)))
        angle_offset = float(cfg.get("angle_offset") or 0)
        model_name = (cfg.get("model_name") or "Part").strip() or "Part"
        hmat = store.robot_homography(project_id, tool["id"])

        if mode == "contour":
            objects = self._detect_contour(image, cfg, model_name, angle_enable)
        else:
            objects = self._detect_template(image, project_id, tool["id"], cfg, model_name, angle_enable)

        for obj in objects:
            rx, ry = apply_homography(obj["x"], obj["y"], hmat)
            obj["rx"] = rx
            obj["ry"] = ry
            obj["ra"] = (float(obj["a"]) + angle_offset) if angle_enable else 0.0

        objects.sort(key=lambda o: (o["y"], o["x"]))
        for i, obj in enumerate(objects, start=1):
            obj["id"] = i

        elapsed = (time.perf_counter() - started) * 1000.0
        judgment = "OK" if len(objects) >= min_count else "NG"
        overlay = self._draw(image, objects, model_name, angle_enable, hmat is not None)
        result = {
            "id": tool.get("id"),
            "type": self.type,
            "name": tool.get("name") or self.title,
            "judgment": judgment,
            "count": len(objects),
            "angle_enable": angle_enable,
            "model_name": model_name,
            "objects": objects,
            "calib": {"ready": hmat is not None},
            "elapsed_ms": elapsed,
            "message": f"{len(objects)} found" if objects else "No objects",
        }
        if objects:
            first = objects[0]
            result["pose"] = {
                "x": first["x"],
                "y": first["y"],
                "angle": first["a"],
                "width": first["width"],
                "height": first["height"],
                "confidence": first.get("confidence", 1.0),
                "shape": first.get("class_name") or model_name,
            }
            result["x"] = first["x"]
            result["y"] = first["y"]
            result["angle"] = first["a"]
        return result, overlay

    def _detect_template(
        self,
        image: np.ndarray,
        project_id: str,
        tool_id: str,
        cfg: dict,
        model_name: str,
        angle_enable: bool,
    ) -> list[dict]:
        from app.storage.store import store

        templates = store.robot_list_templates(project_id, tool_id)
        if not templates:
            return []
        min_conf = float(cfg.get("min_confidence") or 0.6)
        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        search = image[y0:y1, x0:x1]
        gray = _gray(search)
        search_s, scale = _small(gray)
        search_edges = _edges(search_s)
        angles = [0.0]
        if angle_enable:
            angles = list(np.linspace(-20, 20, 9))

        hits: list[dict] = []
        for templ in templates:
            model = templ["image"]
            mh, mw = model.shape[:2]
            model_s, _ = _small(model, max(32, int(_MAX_EDGE * max(mh, mw) / max(gray.shape))))
            best_for_peaks: list[dict] = []
            for ang in angles:
                rotated = _rotate(model_s, float(ang))
                if rotated.shape[0] >= search_edges.shape[0] or rotated.shape[1] >= search_edges.shape[1]:
                    continue
                edge = _edges(rotated)
                result = cv2.matchTemplate(search_edges, edge, cv2.TM_CCOEFF_NORMED)
                # Collect local peaks
                flat = result.reshape(-1)
                if flat.size == 0:
                    continue
                idxs = np.argpartition(flat, -min(8, flat.size))[-min(8, flat.size) :]
                for idx in idxs:
                    score = float(flat[idx])
                    if score < min_conf:
                        continue
                    py, px = np.unravel_index(int(idx), result.shape)
                    tw, th = edge.shape[1], edge.shape[0]
                    cx = (px + tw / 2.0) / scale + x0
                    cy = (py + th / 2.0) / scale + y0
                    w = tw / scale
                    h = th / scale
                    best_for_peaks.append(
                        {
                            "class_id": templ["class_id"],
                            "class_name": templ["class_name"],
                            "confidence": score,
                            "x": float(cx),
                            "y": float(cy),
                            "a": float(ang) if angle_enable else 0.0,
                            "width": float(w),
                            "height": float(h),
                            "box": _rect_box(cx, cy, w, h, float(ang) if angle_enable else 0.0),
                        }
                    )
            hits.extend(best_for_peaks)
        return _nms(hits)

    def _detect_contour(self, image: np.ndarray, cfg: dict, model_name: str, angle_enable: bool) -> list[dict]:
        min_area = float(cfg.get("min_area") or 400)
        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        region = image[y0:y1, x0:x1]
        gray = _gray(region)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects: list[dict] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), ang = rect
            if w < 1 or h < 1:
                continue
            if not angle_enable:
                ang = 0.0
            # OpenCV minAreaRect angle conventions vary; normalize to degrees
            objects.append(
                {
                    "class_id": "",
                    "class_name": model_name,
                    "confidence": 1.0,
                    "x": float(cx + x0),
                    "y": float(cy + y0),
                    "a": float(ang),
                    "width": float(w),
                    "height": float(h),
                    "box": [[float(p[0] + x0), float(p[1] + y0)] for p in cv2.boxPoints(rect)],
                }
            )
        return objects

    def _draw(
        self,
        image: np.ndarray,
        objects: list[dict],
        model_name: str,
        angle_enable: bool,
        calib_ready: bool,
    ) -> np.ndarray:
        view = image.copy()
        for obj in objects:
            pts = np.array(obj["box"], dtype=np.int32)
            cv2.polylines(view, [pts], True, (26, 212, 192), 2)
            cx, cy = int(round(obj["x"])), int(round(obj["y"]))
            cv2.drawMarker(view, (cx, cy), (0, 80, 255), cv2.MARKER_CROSS, 16, 2)
            name = obj.get("class_name") or model_name
            label = f"{name} #{obj['id']}  x={obj['x']:.0f} y={obj['y']:.0f}"
            if angle_enable:
                label += f" a={obj['a']:.1f}"
            if calib_ready and obj.get("rx") is not None:
                label += f"  rx={obj['rx']:.1f} ry={obj['ry']:.1f}"
            cv2.putText(view, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        return view
