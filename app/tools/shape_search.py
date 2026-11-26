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

_model_cache: dict[str, tuple[float, np.ndarray]] = {}
_SIFT_MAX_EDGE = 960
_TEMPLATE_MAX_EDGE = 480


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _content_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground mask: pixels that differ from the dominant border color."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    border = np.concatenate(
        [gray[0, :].ravel(), gray[-1, :].ravel(), gray[:, 0].ravel(), gray[:, -1].ravel()]
    )
    bg = float(np.median(border))
    mask = (np.abs(gray.astype(np.float32) - bg) > 12.0).astype(np.uint8)
    if int(mask.sum()) < 16:
        # Flat / busy border — fall back to Otsu so faint parts still count.
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Prefer the smaller class as content when the model is dark-on-light or light-on-dark.
        if int(mask.sum()) > mask.size // 2:
            mask = 1 - mask
    # Light open only — closing bridges nearby parts and fattens the overlay.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _trim_model(gray: np.ndarray, pad: int = 2) -> np.ndarray:
    """Drop flat border so matching focuses on the pattern."""
    if gray.size == 0:
        return gray
    mask = _content_mask(gray)
    if not np.any(mask):
        return gray
    ys, xs = np.where(mask > 0)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return gray
    return gray[y0:y1, x0:x1]


def _model_outlines(gray: np.ndarray) -> list[np.ndarray]:
    """External content contours in model-pixel coordinates (each Nx2)."""
    h, w = gray.shape[:2]
    fallback = [np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)]
    mask = (_content_mask(gray) * 255).astype(np.uint8)
    if not np.any(mask):
        return fallback
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = max(20.0, 0.002 * float(h * w))
    outlines: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Keep enough points that the warped silhouette still hugs the part.
        peri = float(cv2.arcLength(contour, True))
        eps = max(0.6, 0.0035 * peri)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 8:
            # Fallback: uniform resample of the raw contour.
            raw = contour.reshape(-1, 2).astype(np.float64)
            step = max(1, len(raw) // 48)
            pts = raw[::step]
        if pts.shape[0] >= 3:
            outlines.append(pts)
    return outlines or fallback


def _model_outline(gray: np.ndarray) -> np.ndarray:
    """Single outline used for tip/center: largest content contour."""
    outlines = _model_outlines(gray)
    return max(outlines, key=lambda pts: float(cv2.contourArea(pts.astype(np.float32))))


def _transform_points(aff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.transform(src, aff).reshape(-1, 2)


def _oriented_box_points(pts: np.ndarray) -> np.ndarray:
    """Tight oriented rectangle corners around points (drawing only; ignore OpenCV angle)."""
    if pts is None or len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    rect = cv2.minAreaRect(np.asarray(pts, dtype=np.float32))
    return cv2.boxPoints(rect).astype(np.float64)


def _downscale(gray: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return gray, 1.0
    scale = max_edge / float(long_edge)
    out = cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def _rotate_template(gray: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
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


def _edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 40, 120)


def _match_template_once(search: np.ndarray, templ: np.ndarray, opencl: bool) -> tuple[float, int, int]:
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


def _affine_angle_deg(aff: np.ndarray) -> float:
    # estimateAffinePartial2D / getRotationMatrix2D use [[c, s], [-s, c]] (scale absorbed in c,s).
    return float(np.degrees(np.arctan2(float(aff[0, 1]), float(aff[0, 0]))))


def _affine_scale(aff: np.ndarray) -> float:
    return float(np.hypot(aff[0, 0], aff[1, 0]))


def _transform_corners(aff: np.ndarray, width: float, height: float) -> np.ndarray:
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    return cv2.transform(corners, aff).reshape(-1, 2)


def _match_features(search: np.ndarray, model: np.ndarray) -> dict | None:
    """SIFT + similarity affine. Handles large models, rotation, and modest scale change."""
    search_s, s_scale = _downscale(search, _SIFT_MAX_EDGE)
    model_s, m_scale = _downscale(model, _SIFT_MAX_EDGE)
    detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.04, edgeThreshold=12)
    kp_m, des_m = detector.detectAndCompute(model_s, None)
    kp_s, des_s = detector.detectAndCompute(search_s, None)
    if des_m is None or des_s is None or len(kp_m) < 8 or len(kp_s) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        pairs = matcher.knnMatch(des_m, des_s, k=2)
    except cv2.error:
        return None
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None
    src = np.float32([kp_m[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    aff_s, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=3000,
        confidence=0.995,
    )
    if aff_s is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.ravel().sum())
    if inliers < 8:
        return None
    # Map affine from downscaled model/search back to full-resolution pixels.
    # x_s = s_scale * x_full_search, x_m = m_scale * x_full_model
    # x_s = A_s * x_m + t_s  =>  s*x = A_s * (m*X) + t_s  =>  x = (A_s*m/s) X + t_s/s
    scale_ratio = m_scale / s_scale
    aff = aff_s.astype(np.float64).copy()
    aff[:, :2] *= scale_ratio
    aff[:, 2] /= s_scale

    mh, mw = model.shape[:2]
    outline_srcs = _model_outlines(model)
    outlines = [_transform_points(aff, src) for src in outline_srcs]
    all_pts = np.vstack(outlines) if outlines else np.zeros((0, 2), dtype=np.float64)
    box = _oriented_box_points(all_pts)
    if box.shape[0] < 4:
        box = _transform_corners(aff, float(mw), float(mh))
    conf = float(inliers) / float(len(good))
    # Blend with match density so sparse luck doesn't look like a solid hit.
    conf = min(1.0, conf * min(1.0, inliers / 40.0) * 0.35 + conf * 0.65)
    # Orientation tick from the top-most content point, outward.
    main_src = max(outline_srcs, key=lambda pts: float(cv2.contourArea(pts.astype(np.float32))))
    top_local = main_src[int(np.argmin(main_src[:, 1]))]
    tip_anchor = _transform_points(aff, top_local.reshape(1, 2))[0]
    center = all_pts.mean(axis=0) if all_pts.shape[0] else box.mean(axis=0)
    outward = tip_anchor - center
    norm = float(np.hypot(outward[0], outward[1])) or 1.0
    tip_end = tip_anchor + outward * (12.0 / norm)
    scale = _affine_scale(aff)
    return {
        "confidence": conf,
        "x": float(center[0]),
        "y": float(center[1]),
        "width": float(mw) * scale,
        "height": float(mh) * scale,
        "angle": _affine_angle_deg(aff),
        "box": box.tolist(),
        "outline": all_pts.tolist(),
        "outlines": [pts.tolist() for pts in outlines],
        "tip": [tip_anchor.tolist(), tip_end.tolist()],
        "method": "sift",
        "inliers": inliers,
        "matches": len(good),
    }


def _match_template_pyramid(
    search: np.ndarray,
    model: np.ndarray,
    angles: list[float],
    opencl: bool,
) -> dict | None:
    """Fallback for small / low-texture models where SIFT has too few keypoints."""
    search_s, s_scale = _downscale(search, _TEMPLATE_MAX_EDGE)
    target_w = max(8, int(round(model.shape[1] * s_scale)))
    target_h = max(8, int(round(model.shape[0] * s_scale)))
    model_s = cv2.resize(model, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if model_s.shape[0] >= search_s.shape[0] or model_s.shape[1] >= search_s.shape[1]:
        fit = 0.55 * min(search_s.shape[0] / model_s.shape[0], search_s.shape[1] / model_s.shape[1])
        model_s = cv2.resize(
            model_s,
            (max(8, int(model_s.shape[1] * fit)), max(8, int(model_s.shape[0] * fit))),
            interpolation=cv2.INTER_AREA,
        )

    search_e = _edges(search_s)
    scales = [0.90, 0.95, 1.0, 1.05, 1.10]
    best = None
    mh0, mw0 = model_s.shape[:2]
    for scale in scales:
        mw = max(8, int(round(mw0 * scale)))
        mh = max(8, int(round(mh0 * scale)))
        scaled = cv2.resize(model_s, (mw, mh), interpolation=cv2.INTER_LINEAR)
        for angle in angles:
            templ, matrix = _rotate_template(scaled, angle)
            if templ.shape[0] >= search_s.shape[0] or templ.shape[1] >= search_s.shape[1]:
                continue
            score_g, px, py = _match_template_once(search_s, templ, opencl)
            score_e, px_e, py_e = _match_template_once(search_e, _edges(templ), opencl)
            score = max(0.0, float(score_g))
            if score_e > score and abs(px_e - px) <= 2 and abs(py_e - py) <= 2:
                score = float(score_e)
                px, py = px_e, py_e
            if best is not None and score <= best["confidence"]:
                continue
            # Contours in the pre-rotation scaled model, then through the same warp used for matching.
            outline_locals = _model_outlines(scaled)
            outlines = []
            for outline_local in outline_locals:
                ones = np.ones((len(outline_local), 1), dtype=np.float64)
                outline = (matrix @ np.hstack([outline_local, ones]).T).T
                outline[:, 0] = (outline[:, 0] + px) / s_scale
                outline[:, 1] = (outline[:, 1] + py) / s_scale
                outlines.append(outline)
            all_pts = np.vstack(outlines) if outlines else np.zeros((0, 2), dtype=np.float64)
            box = _oriented_box_points(all_pts)
            if box.shape[0] < 4:
                corners = np.array([[0.0, 0.0], [mw, 0.0], [mw, mh], [0.0, mh]], dtype=np.float64)
                ones4 = np.ones((4, 1), dtype=np.float64)
                box = (matrix @ np.hstack([corners, ones4]).T).T
                box[:, 0] = (box[:, 0] + px) / s_scale
                box[:, 1] = (box[:, 1] + py) / s_scale
            main_local = max(
                outline_locals,
                key=lambda pts: float(cv2.contourArea(pts.astype(np.float32))),
            )
            top_local = main_local[int(np.argmin(main_local[:, 1]))].reshape(1, 2)
            tip_anchor = (matrix @ np.hstack([top_local, np.ones((1, 1))]).T).T[0]
            tip_anchor[0] = (tip_anchor[0] + px) / s_scale
            tip_anchor[1] = (tip_anchor[1] + py) / s_scale
            center = all_pts.mean(axis=0)
            outward = tip_anchor - center
            norm = float(np.hypot(outward[0], outward[1])) or 1.0
            tip_end = tip_anchor + outward * (12.0 / norm)
            best = {
                "confidence": score,
                "x": float(center[0]),
                "y": float(center[1]),
                "width": float(mw) / s_scale,
                "height": float(mh) / s_scale,
                "angle": float(angle),
                "box": box.tolist(),
                "outline": all_pts.tolist(),
                "outlines": [pts.tolist() for pts in outlines],
                "tip": [tip_anchor.tolist(), tip_end.tolist()],
                "method": "template",
            }
    return best


class ShapeSearchTool(Tool):
    type = "shape_search"
    title = "Shape Search"
    summary = "Register a pattern, then find its pose (position, angle, scale) in the image."
    ui_modules = ["/static/tools/accel.js", "/static/tools/shape_search.js"]

    def fields(self) -> list[dict]:
        return [
            {"key": "search_roi", "label": "Search region", "kind": "roi"},
            {
                "key": "min_confidence",
                "label": "Minimum confidence",
                "kind": "range",
                "min": 0.35,
                "max": 0.95,
                "step": 0.01,
            },
            {
                "key": "angle_range",
                "label": "Template angle fallback",
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
                "label": "Template angle step",
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
            "min_confidence": 0.55,
            "angle_range": "20",
            "angle_step": "2",
            "accel_device": "cpu",
        }

    def status(self) -> str:
        return (
            "Open Model, grab/choose an image, draw tightly around the part, Register. "
            "Search uses feature matching (rotation-aware); template match is the fallback."
        )

    def mount(self, app) -> None:
        from app.tools import shape_search_routes

        shape_search_routes.mount(app)

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        name = tool.get("name") or self.title
        min_confidence = float(cfg.get("min_confidence", 0.55))
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
        search = _gray(image[y0:y1, x0:x1])
        model_gray = _trim_model(_gray(model))
        if min(model_gray.shape[:2]) < 8:
            model_gray = _gray(model)

        span = int(cfg.get("angle_range") or 0)
        step = float(cfg.get("angle_step") or 2)
        angles = _angle_list(span, step)

        best = _match_features(search, model_gray)
        # Prefer template only when features are weak / unavailable.
        if best is None or best["confidence"] < 0.45:
            templ = _match_template_pyramid(search, model_gray, angles, opencl)
            if templ is not None and (best is None or templ["confidence"] > best["confidence"]):
                best = templ

        elapsed = (time.perf_counter() - started) * 1000.0
        if best is None or best["confidence"] <= 0:
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

        # Offset ROI origin into full-image coordinates.
        box = np.asarray(best["box"], dtype=np.float64)
        box[:, 0] += x0
        box[:, 1] += y0
        outlines = []
        for ring in best.get("outlines") or [best.get("outline") or best["box"]]:
            pts = np.asarray(ring, dtype=np.float64)
            pts = pts.copy()
            pts[:, 0] += x0
            pts[:, 1] += y0
            outlines.append(pts)
        outline = np.vstack(outlines) if outlines else box
        tip = np.asarray(best["tip"], dtype=np.float64)
        tip[:, 0] += x0
        tip[:, 1] += y0
        center = outline.mean(axis=0)
        best = {
            **best,
            "x": float(center[0]),
            "y": float(center[1]),
            "box": box.tolist(),
            "outline": outline.tolist(),
            "outlines": [pts.tolist() for pts in outlines],
            "tip": tip.tolist(),
        }

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
            "outline": best["outline"],
            "outlines": best["outlines"],
            "method": best.get("method"),
        }
        overlay = self._draw_hit(image, best, passed)
        method = best.get("method") or "?"
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": name,
            "judgment": "OK" if passed else "NG",
            "confidence": best["confidence"],
            "min_confidence": min_confidence,
            "accel": accel.get("id", "cpu"),
            "method": method,
            "x": round(best["x"], 1),
            "y": round(best["y"], 1),
            "angle": round(best["angle"], 1),
            "width": round(best["width"], 1),
            "height": round(best["height"], 1),
            "box": [[round(p[0], 1), round(p[1], 1)] for p in best["box"]],
            "outline": [[round(p[0], 1), round(p[1], 1)] for p in best["outline"]],
            "outlines": [[[round(p[0], 1), round(p[1], 1)] for p in ring] for ring in best["outlines"]],
            "pose": pose,
            "message": (
                f"Pattern  confidence {best['confidence'] * 100:.1f}%  "
                f"∠{best['angle']:.1f}°  ({method})"
            ),
            "elapsed_ms": elapsed,
        }, overlay

    @staticmethod
    def _draw_hit(image: np.ndarray, found: dict, passed: bool) -> np.ndarray:
        view = image.copy()
        color = (40, 220, 160) if passed else (60, 60, 230)
        thick = max(2, int(round(min(view.shape[:2]) / 320)))
        rings = found.get("outlines") or [found.get("outline") or found.get("box") or []]
        all_pts = []
        for ring in rings:
            pts = np.asarray(ring, dtype=np.float64)
            if pts.shape[0] >= 3:
                cv2.polylines(view, [np.int32(np.round(pts))], True, color, thick, cv2.LINE_AA)
                all_pts.append(pts)
        tip = np.asarray(found.get("tip") or [], dtype=np.float64)
        if tip.shape == (2, 2):
            p0 = (int(round(tip[0, 0])), int(round(tip[0, 1])))
            p1 = (int(round(tip[1, 0])), int(round(tip[1, 1])))
            cv2.line(view, p0, p1, color, thick, cv2.LINE_AA)
            cv2.drawMarker(view, p0, color, cv2.MARKER_CROSS, 12, thick)
        label = f"{found['confidence'] * 100:.0f}%  {found['angle']:.1f} deg"
        stacked = np.vstack(all_pts) if all_pts else np.array([[found["x"], found["y"]]], dtype=np.float64)
        lx = int(round(float(found["x"]))) + 14
        ly = int(round(float(np.min(stacked[:, 1])))) - 10
        ly = max(18, ly)
        cv2.putText(view, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(view, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return view

    @staticmethod
    def _load_model(project_id: str, tool_id: str) -> np.ndarray | None:
        from pathlib import Path

        from app.config import DATA_DIR

        candidates = [
            DATA_DIR / "projects" / project_id / "tools" / tool_id / "shape_model.png",
            Path(__file__).resolve().parents[2] / "data" / "projects" / project_id / "tools" / tool_id / "shape_model.png",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None
        key = f"{project_id}:{tool_id}:{path}"
        stamp = path.stat().st_mtime
        cached = _model_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            _model_cache[key] = (stamp, image)
        return image


register(ShapeSearchTool())
