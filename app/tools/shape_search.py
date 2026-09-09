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


def _preview(image: np.ndarray, limit: int = 960) -> np.ndarray:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= limit:
        return image.copy()
    scale = limit / long_edge
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_MAX_EDGE = 320
_model_cache: dict[str, tuple[float, np.ndarray]] = {}


def _edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 40, 120)


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
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


def _match(search_edges: np.ndarray, model: np.ndarray) -> tuple[float, int, int]:
    templ = _edges(model)
    if templ.shape[0] >= search_edges.shape[0] or templ.shape[1] >= search_edges.shape[1]:
        return -1.0, 0, 0
    result = cv2.matchTemplate(search_edges, templ, cv2.TM_CCOEFF_NORMED)
    _, value, _, loc = cv2.minMaxLoc(result)
    return float(value), int(loc[0]), int(loc[1])


class ShapeSearchTool(Tool):
    type = "shape_search"
    title = "Shape Search"
    summary = "Register any pattern by drawing it on the image, then search for that model."

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
                ],
            },
        ]

    def panels(self) -> list[str]:
        return ["shape_model"]

    def default_config(self) -> dict:
        return {"search_roi": None, "min_confidence": 0.6, "angle_range": "10"}

    def status(self) -> str:
        return "Open Model, choose an image, draw around the pattern, then Register. The pattern can be any shape."

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        name = tool.get("name") or self.title
        min_confidence = float(cfg.get("min_confidence", 0.6))
        model = self._load_model(project_id, tool.get("id", ""))
        view = image
        if model is None:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "confidence": 0.0,
                "min_confidence": min_confidence,
                "message": "Model image is not saved. Register a pattern on the Model tab.",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, view

        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        small, scale = _small(image[y0:y1, x0:x1])
        gray_search = _gray(small)
        model_small, model_scale = _small(model, max(8, int(max(model.shape[:2]) * scale)))
        gray_model = _gray(model_small)
        sh, sw = gray_search.shape[:2]
        mh, mw = gray_model.shape[:2]
        if mh >= sh or mw >= sw:
            gray_model = cv2.resize(gray_model, (max(1, sw // 2), max(1, sh // 2)), interpolation=cv2.INTER_LINEAR)
        search_gray = gray_search
        span = int(cfg.get("angle_range") or 0)
        angles = [0]
        if span > 0:
            angles = [0, -span, span]
        inv = 1.0 / scale

        best = None
        for angle in angles:
            templ = _rotate(gray_model, angle)
            if templ.shape[0] >= search_gray.shape[0] or templ.shape[1] >= search_gray.shape[1]:
                continue
            result = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            score = max(0.0, float(score))
            if best is None or score > best["confidence"]:
                px, py = int(loc[0]), int(loc[1])
                best = {
                    "confidence": score,
                    "x": x0 + (px + templ.shape[1] / 2.0) * inv,
                    "y": y0 + (py + templ.shape[0] / 2.0) * inv,
                    "width": float(templ.shape[1]) * inv,
                    "height": float(templ.shape[0]) * inv,
                    "angle": float(angle),
                    "tl": (x0 + px * inv, y0 + py * inv),
                }
            if angle == 0 and best["confidence"] >= min_confidence:
                break

        elapsed = (time.perf_counter() - started) * 1000.0
        view = None
        found = best is not None and best["confidence"] > 0

        if not found:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "confidence": 0.0,
                "min_confidence": min_confidence,
                "message": "Registered pattern was not found.",
                "elapsed_ms": elapsed,
            }, view

        passed = best["confidence"] >= min_confidence
        pose = {
            "x": best["x"],
            "y": best["y"],
            "angle": best["angle"],
            "width": best["width"],
            "height": best["height"],
            "confidence": best["confidence"],
            "shape": "model",
        }
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": name,
            "judgment": "OK" if passed else "NG",
            "confidence": best["confidence"],
            "min_confidence": min_confidence,
            "x": round(best["x"], 1),
            "y": round(best["y"], 1),
            "angle": round(best["angle"], 1),
            "width": round(best["width"], 1),
            "height": round(best["height"], 1),
            "pose": pose,
            "message": f"Pattern  confidence {best['confidence'] * 100:.1f}%",
            "elapsed_ms": elapsed,
        }, view

    @staticmethod
    def _mark(view: np.ndarray, found: dict, factor: float) -> None:
        color = (40, 220, 160) if found["confidence"] >= 0.55 else (60, 60, 230)
        tlx, tly = found["tl"]
        x0 = int(tlx / factor)
        y0 = int(tly / factor)
        x1 = int((tlx + found["width"]) / factor)
        y1 = int((tly + found["height"]) / factor)
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
        center = (int(round(found["x"] / factor)), int(round(found["y"] / factor)))
        cv2.drawMarker(view, center, color, cv2.MARKER_CROSS, 16, 2)

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
