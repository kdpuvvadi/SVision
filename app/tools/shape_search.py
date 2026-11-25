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

from app.core.accel import resolve_accel, use_opencl
from app.core.features import _roi_box
from app.tools.base import Tool
from app.tools.registry import register


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_MAX_EDGE = 320
_model_cache: dict[str, tuple[float, np.ndarray]] = {}


def _edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 40, 120)


def _rotate(gray: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate template; return (image, 2x3 matrix mapping src → dst pixels)."""
    h, w = gray.shape[:2]
    if abs(angle) < 0.05:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return gray, matrix
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += (nw - w) / 2.0
    matrix[1, 2] += (nh - h) / 2.0
    out = cv2.warpAffine(gray, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    return out, matrix


def _warp_rect_corners(mw: int, mh: int, matrix: np.ndarray) -> np.ndarray:
    """Map the model rectangle corners through the same matrix used to build the template."""
    corners = np.array([[0.0, 0.0], [mw, 0.0], [mw, mh], [0.0, mh]], dtype=np.float64)
    ones = np.ones((4, 1), dtype=np.float64)
    return (matrix @ np.hstack([corners, ones]).T).T


def _trim_model(gray: np.ndarray, pad: int = 2) -> np.ndarray:
    """Drop empty border so the overlay hugs the pattern, not padded canvas."""
    if gray.size == 0:
        return gray
    # Content = pixels that differ from the dominant border color.
    border = np.concatenate(
        [gray[0, :].ravel(), gray[-1, :].ravel(), gray[:, 0].ravel(), gray[:, -1].ravel()]
    )
    # Mode via median is stable for flat borders.
    bg = float(np.median(border))
    mask = np.abs(gray.astype(np.float32) - bg) > 12.0
    if not np.any(mask):
        return gray
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return gray
    return gray[y0:y1, x0:x1]


def _small(image: np.ndarray, max_edge: int = _MAX_EDGE) -> tuple[np.ndarray, float, float]:
    """Downscale uniformly; return image and actual per-axis scales (out/src)."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image, 1.0, 1.0
    scale = max_edge / long_edge
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, nw / float(w), nh / float(h)


def _angle_list(span: int, step: float) -> list[float]:
    span = max(0, int(span))
    if span <= 0:
        return [0.0]
    step = max(0.5, float(step or 2.0))
    values = list(np.arange(-span, span + 1e-6, step))
    ordered = [0.0] + [float(a) for a in values if abs(float(a)) > 1e-6]
    seen: set[float] = set()
    out: list[float] = []
    for a in ordered:
        key = round(a, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _match_template(search: np.ndarray, templ: np.ndarray, opencl: bool) -> tuple[float, int, int]:
    if templ.shape[0] >= search.shape[0] or templ.shape[1] >= search.shape[1]:
        return -1.0, 0, 0
    if opencl:
        try:
            cv2.ocl.setUseOpenCL(True)
            result = cv2.matchTemplate(cv2.UMat(search), cv2.UMat(templ), cv2.TM_CCOEFF_NORMED)
            result = result.get() if hasattr(result, "get") else result
        except Exception:
            result = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
    else:
        result = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
    _, value, _, loc = cv2.minMaxLoc(result)
    return float(value), int(loc[0]), int(loc[1])


def _map_points_to_full(pts_small: np.ndarray, x0: int, y0: int, sx: float, sy: float) -> np.ndarray:
    """Map search-space points to full-image pixels using actual resize scales."""
    out = np.empty_like(pts_small)
    out[:, 0] = x0 + pts_small[:, 0] / sx
    out[:, 1] = y0 + pts_small[:, 1] / sy
    return out


class ShapeSearchTool(Tool):
    type = "shape_search"
    title = "Shape Search"
    summary = "Register any pattern by drawing it on the image, then search for that model."
    ui_modules = ["/static/tools/accel.js", "/static/tools/shape_search.js"]

    def fields(self) -> list[dict]:
        return [
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
                "key": "angle_range",
                "label": "Angle search",
                "kind": "select",
                "options": [
                    {"value": "0", "label": "No rotation"},
                    {"value": "10", "label": "±10 degrees"},
                    {"value": "20", "label": "±20 degrees"},
                    {"value": "45", "label": "±45 degrees"},
                    {"value": "90", "label": "±90 degrees"},
                ],
            },
            {
                "key": "angle_step",
                "label": "Angle step",
                "kind": "select",
                "options": [
                    {"value": "5", "label": "5° (faster)"},
                    {"value": "2", "label": "2° (balanced)"},
                    {"value": "1", "label": "1° (finest)"},
                ],
            },
            {
                "key": "accel_device",
                "label": "Hardware acceleration",
                "kind": "accel",
            },
        ]

    def panels(self) -> list[str]:
        return ["shape_model"]

    def panel_labels(self) -> dict[str, str]:
        return {"shape_model": "Model"}

    def default_config(self) -> dict:
        return {
            "search_roi": None,
            "min_confidence": 0.6,
            "angle_range": "20",
            "angle_step": "2",
            "accel_device": "cpu",
        }

    def status(self) -> str:
        return (
            "Open Model. If Camera is USB/GenTL, Grab from camera; otherwise choose an image. "
            "Draw around the pattern (drag edges/corners to fine-tune), then Register."
        )

    def mount(self, app) -> None:
        from app.tools import shape_search_routes

        shape_search_routes.mount(app)

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        name = tool.get("name") or self.title
        min_confidence = float(cfg.get("min_confidence", 0.6))
        accel = resolve_accel(cfg.get("accel_device") or "cpu")
        opencl = use_opencl(cfg.get("accel_device"))
        model = self._load_model(project_id, tool.get("id", ""))
        if model is None:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "confidence": 0.0,
                "min_confidence": min_confidence,
                "accel": accel.get("id", "cpu"),
                "message": "Model image is not saved. Register a pattern on the Model tab.",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, image

        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        roi = image[y0:y1, x0:x1]
        small, sx, sy = _small(roi)
        gray_search = _gray(small)
        # Trim empty border, then resize with the SAME sx/sy as the search image
        # so overlay corners map back to full-resolution without scale drift.
        gray_model = _trim_model(_gray(model) if model.ndim == 3 else model)
        mw0, mh0 = int(gray_model.shape[1]), int(gray_model.shape[0])
        mw = max(1, int(round(mw0 * sx)))
        mh = max(1, int(round(mh0 * sy)))
        gray_model = cv2.resize(gray_model, (mw, mh), interpolation=cv2.INTER_AREA if (mw < mw0 or mh < mh0) else cv2.INTER_LINEAR)
        sh, sw = gray_search.shape[:2]
        if mh >= sh or mw >= sw:
            gray_model = cv2.resize(gray_model, (max(1, sw // 2), max(1, sh // 2)), interpolation=cv2.INTER_LINEAR)
            mh, mw = gray_model.shape[:2]

        search_edges = _edges(gray_search)
        span = int(cfg.get("angle_range") or 0)
        step = float(cfg.get("angle_step") or 2)
        angles = _angle_list(span, step)

        best = None
        for angle in angles:
            templ_gray, matrix = _rotate(gray_model, angle)
            if templ_gray.shape[0] >= gray_search.shape[0] or templ_gray.shape[1] >= gray_search.shape[1]:
                continue
            # Localize with gray (stable). Edges only refine confidence when they agree.
            score_g, px, py = _match_template(gray_search, templ_gray, opencl)
            score_e, px_e, py_e = _match_template(search_edges, _edges(templ_gray), opencl)
            score = max(0.0, float(score_g))
            if score_e > score_g and abs(px_e - px) <= 3 and abs(py_e - py) <= 3:
                score = max(score, float(score_e))
            if best is not None and score <= best["confidence"]:
                continue
            # Corners = model rectangle warped by the SAME matrix as the matched template,
            # then shifted to the match location — locks the outline to the part.
            pts_s = _warp_rect_corners(mw, mh, matrix)
            pts_s[:, 0] += px
            pts_s[:, 1] += py
            pts = _map_points_to_full(pts_s, x0, y0, sx, sy)
            center_s = np.array([[mw / 2.0, mh / 2.0]], dtype=np.float64)
            ones = np.ones((1, 1), dtype=np.float64)
            center_t = (matrix @ np.hstack([center_s, ones]).T).T[0]
            tip_local = np.array([[mw / 2.0, mh / 2.0 - mh * 0.55]], dtype=np.float64)
            tip_t = (matrix @ np.hstack([tip_local, ones]).T).T[0]
            tip_s = np.vstack([center_t, tip_t])
            tip_s[:, 0] += px
            tip_s[:, 1] += py
            tip = _map_points_to_full(tip_s, x0, y0, sx, sy)
            best = {
                "confidence": score,
                "x": float(pts[:, 0].mean()),
                "y": float(pts[:, 1].mean()),
                "width": float(mw) / sx,
                "height": float(mh) / sy,
                "angle": float(angle),
                "box": pts.tolist(),
                "tip": tip.tolist(),
            }

        elapsed = (time.perf_counter() - started) * 1000.0
        found = best is not None and best["confidence"] > 0

        if not found:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "confidence": 0.0,
                "min_confidence": min_confidence,
                "accel": accel.get("id", "cpu"),
                "message": "Registered pattern was not found.",
                "elapsed_ms": elapsed,
            }, None

        passed = best["confidence"] >= min_confidence
        pose = {
            "x": best["x"],
            "y": best["y"],
            "angle": best["angle"],
            "width": best["width"],
            "height": best["height"],
            "confidence": best["confidence"],
            "shape": "model",
            "box": best["box"],
        }
        overlay = self._draw_hit(image, best, passed)
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": name,
            "judgment": "OK" if passed else "NG",
            "confidence": best["confidence"],
            "min_confidence": min_confidence,
            "accel": accel.get("id", "cpu"),
            "x": round(best["x"], 1),
            "y": round(best["y"], 1),
            "angle": round(best["angle"], 1),
            "width": round(best["width"], 1),
            "height": round(best["height"], 1),
            "box": [[round(p[0], 1), round(p[1], 1)] for p in best["box"]],
            "pose": pose,
            "message": f"Pattern  confidence {best['confidence'] * 100:.1f}%  ∠{best['angle']:.1f}°",
            "elapsed_ms": elapsed,
        }, overlay

    @staticmethod
    def _draw_hit(image: np.ndarray, found: dict, passed: bool) -> np.ndarray:
        view = image.copy()
        color = (40, 220, 160) if passed else (60, 60, 230)
        thick = max(2, int(round(min(view.shape[:2]) / 320)))
        pts = np.asarray(found["box"], dtype=np.float64)
        cv2.polylines(view, [np.int32(np.round(pts))], True, color, thick, cv2.LINE_AA)
        tip = np.asarray(found.get("tip") or [], dtype=np.float64)
        if tip.shape == (2, 2):
            p0 = (int(round(tip[0, 0])), int(round(tip[0, 1])))
            p1 = (int(round(tip[1, 0])), int(round(tip[1, 1])))
            cv2.line(view, p0, p1, color, thick, cv2.LINE_AA)
            cv2.drawMarker(view, p0, color, cv2.MARKER_CROSS, 16, thick)
        else:
            cx, cy = int(round(found["x"])), int(round(found["y"]))
            cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_CROSS, 16, thick)
        label = f"{found['confidence'] * 100:.0f}%  {found['angle']:.1f} deg"
        lx = int(round(found["x"])) + 10
        ly = int(round(found["y"])) - 12
        cv2.putText(view, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(view, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return view

    @staticmethod
    def _load_model(project_id: str, tool_id: str) -> np.ndarray | None:
        from app.config import DATA_DIR

        path = DATA_DIR / "projects" / project_id / "tools" / tool_id / "shape_model.png"
        if not path.exists():
            return None
        key = f"{project_id}:{tool_id}"
        stamp = path.stat().st_mtime
        cached = _model_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            _model_cache[key] = (stamp, image)
        return image


register(ShapeSearchTool())
