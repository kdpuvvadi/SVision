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

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.accel import resolve_accel
from app.core.features import _roi_box
from app.tools.base import Tool
from app.tools.registry import register

_model_cache: dict[str, tuple[float, np.ndarray]] = {}
_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}
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


def _figure_mask(h: int, w: int, figures: list[dict[str, Any]] | None) -> np.ndarray | None:
    """Build a binary mask from registered figures (normalized 0–1 inside the crop)."""
    if not figures:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    drew = False
    for fig in figures:
        kind = (fig.get("type") or "rect").lower()
        if kind in {"rect", "rectangle"}:
            x0 = int(round(float(fig.get("x", 0)) * w))
            y0 = int(round(float(fig.get("y", 0)) * h))
            x1 = int(round((float(fig.get("x", 0)) + float(fig.get("w", 1))) * w))
            y1 = int(round((float(fig.get("y", 0)) + float(fig.get("h", 1))) * h))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            drew = True
        elif kind == "circle":
            cx = int(round(float(fig.get("cx", 0.5)) * w))
            cy = int(round(float(fig.get("cy", 0.5)) * h))
            r = int(round(float(fig.get("r", 0.25)) * min(w, h)))
            cv2.circle(mask, (cx, cy), max(1, r), 255, -1)
            drew = True
        elif kind in {"donut", "double_circle", "ring"}:
            cx = int(round(float(fig.get("cx", 0.5)) * w))
            cy = int(round(float(fig.get("cy", 0.5)) * h))
            r_out = int(round(float(fig.get("r_outer", fig.get("r", 0.35))) * min(w, h)))
            r_in = int(round(float(fig.get("r_inner", r_out * 0.5 / max(min(w, h), 1))) * min(w, h)))
            cv2.circle(mask, (cx, cy), max(1, r_out), 255, -1)
            cv2.circle(mask, (cx, cy), max(0, min(r_in, r_out - 1)), 0, -1)
            drew = True
        elif kind == "polygon":
            pts = np.asarray(fig.get("points") or [], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                px = np.column_stack([pts[:, 0] * w, pts[:, 1] * h]).astype(np.int32)
                cv2.fillPoly(mask, [px], 255)
                drew = True
    return mask if drew else None


def build_solid_model(crop: np.ndarray, figures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build an Omron-style shape model from the selected region.

    Optional ``figures`` (rect / circle / donut / polygon) limit which pixels
    contribute. Edges inside those figures become the solid registered model.
    """
    gray = _gray(crop)
    gray = _trim_model(gray, pad=1)
    if min(gray.shape[:2]) < 8:
        raise ValueError("Draw a larger region around the pattern.")
    h, w = gray.shape[:2]
    mask = _figure_mask(h, w, figures)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    # Strengthen thin edges so the registered figure is visible and matchable.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_len = max(12, int(0.04 * (w + h)))
    edge_figures: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < min_len:
            continue
        peri = float(cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, max(0.8, 0.004 * peri), True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 3:
            continue
        edge_figures.append([[float(p[0]), float(p[1])] for p in pts])
    if not edge_figures:
        # Fall back to content outline so registration never stores an empty model.
        for ring in _model_outlines(gray):
            edge_figures.append([[float(p[0]), float(p[1])] for p in ring])
            edges = np.zeros_like(gray)
            cv2.polylines(edges, [np.int32(np.round(ring))], True, 255, 1)
    # Display image: gray part with cyan edge figure (Omron-style "model image").
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for fig in edge_figures:
        cv2.polylines(display, [np.int32(np.round(np.asarray(fig)))], True, (40, 220, 160), 1, cv2.LINE_AA)
    edge_points = int(np.count_nonzero(edges))
    return {
        "gray": gray,
        "edges": edges,
        "display": display,
        "figures": edge_figures,
        "user_figures": figures or [],
        "width": int(w),
        "height": int(h),
        "edge_points": edge_points,
        "figure_count": len(edge_figures),
    }


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
    # Confidence: inlier ratio is the quality signal. Small models rarely produce
    # 40 keypoints, so require a modest consensus (~12 inliers) instead of punishing them.
    inlier_ratio = float(inliers) / float(len(good))
    support = min(1.0, float(inliers) / 12.0)
    conf = float(np.clip(inlier_ratio * (0.55 + 0.45 * support), 0.0, 1.0))
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
    model_edges: np.ndarray | None = None,
) -> dict | None:
    """Edge-first correlation (Omron-style), with gray assist."""
    search_s, s_scale = _downscale(search, _TEMPLATE_MAX_EDGE)
    target_w = max(8, int(round(model.shape[1] * s_scale)))
    target_h = max(8, int(round(model.shape[0] * s_scale)))
    model_s = cv2.resize(model, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if model_edges is not None and model_edges.size:
        edges_s = cv2.resize(model_edges, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        edges_s = _edges(model_s)
    if model_s.shape[0] >= search_s.shape[0] or model_s.shape[1] >= search_s.shape[1]:
        fit = 0.55 * min(search_s.shape[0] / model_s.shape[0], search_s.shape[1] / model_s.shape[1])
        new_size = (max(8, int(model_s.shape[1] * fit)), max(8, int(model_s.shape[0] * fit)))
        model_s = cv2.resize(model_s, new_size, interpolation=cv2.INTER_AREA)
        edges_s = cv2.resize(edges_s, new_size, interpolation=cv2.INTER_AREA)

    search_e = _edges(search_s)
    scales = [0.90, 0.95, 1.0, 1.05, 1.10]
    best = None
    mh0, mw0 = model_s.shape[:2]
    for scale in scales:
        mw = max(8, int(round(mw0 * scale)))
        mh = max(8, int(round(mh0 * scale)))
        scaled = cv2.resize(model_s, (mw, mh), interpolation=cv2.INTER_LINEAR)
        scaled_e = cv2.resize(edges_s, (mw, mh), interpolation=cv2.INTER_LINEAR)
        for angle in angles:
            templ, matrix = _rotate_template(scaled, angle)
            templ_e, _ = _rotate_template(scaled_e, angle)
            if templ.shape[0] >= search_s.shape[0] or templ.shape[1] >= search_s.shape[1]:
                continue
            score_g, px_g, py_g = _match_template_once(search_s, templ, opencl)
            score_e, px, py = _match_template_once(search_e, templ_e, opencl)
            # Solid-model correlation is edge-led; gray breaks ties.
            if score_e >= 0.15:
                score = float(0.75 * max(0.0, score_e) + 0.25 * max(0.0, score_g))
                if abs(px_g - px) > 3 or abs(py_g - py) > 3:
                    # Prefer the gray peak if edges drifted.
                    if score_g > score_e:
                        score = float(score_g)
                        px, py = px_g, py_g
            else:
                score = max(0.0, float(score_g))
                px, py = px_g, py_g
            if best is not None:
                # OpenCL matchTemplate often spikes a near-perfect score at ±step on
                # an untilted part. Demand a real gain before accepting more rotation.
                gain = score - float(best["confidence"])
                more_rotated = abs(angle) > abs(float(best["angle"])) + 1e-6
                if more_rotated and gain < 0.02:
                    continue
                if gain <= 0:
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


def _poses_agree(a: dict, b: dict, model_shape: tuple[int, int]) -> bool:
    """True when two hits likely refer to the same part (centers within ~half model)."""
    mh, mw = float(model_shape[0]), float(model_shape[1])
    tol = 0.55 * max(mw, mh, 16.0)
    return float(np.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))) <= tol


def _pick_best_match(
    sift: dict | None,
    templ: dict | None,
    model_shape: tuple[int, int],
) -> dict | None:
    """Choose pose + confidence from SIFT and template.

    SIFT owns rotated/scaled pose. Template often scores the registered piece far
    higher on low-texture models — use that score when both agree on location.
    """
    if sift is None:
        return templ
    if templ is None:
        return sift
    if _poses_agree(sift, templ, model_shape):
        # Same hit: take the stronger method's full pose (don't graft a high
        # template score onto a drifted SIFT angle — that produced fake ±10°).
        if float(templ["confidence"]) >= float(sift["confidence"]):
            out = dict(templ)
            out["confidence"] = float(templ["confidence"])
            out["method"] = "sift+template" if float(sift["confidence"]) > 0.4 else "template"
            return out
        return sift
    # Different places — take the stronger hit.
    return sift if sift["confidence"] >= templ["confidence"] else templ


class ShapeSearchTool(Tool):
    type = "shape_search"
    title = "Shape Search"
    summary = "Register a solid edge model from a selected region, then find its pose in the image."
    ui_modules = ["/static/tools/accel.js", "/static/tools/shape_search.js"]

    def fields(self) -> list[dict]:
        meas = {"tab": "measurement", "tab_label": "Measurement"}
        return [
            {"key": "search_roi", "label": "Region setting", "kind": "roi"},
            {
                "key": "min_confidence",
                "label": "Correlation (min)",
                "kind": "range",
                "min": 0.35,
                "max": 0.95,
                "step": 0.01,
                **meas,
            },
            {
                "key": "angle_range",
                "label": "Angle range",
                "kind": "select",
                "options": [
                    {"value": "0", "label": "No rotation"},
                    {"value": "10", "label": "±10 degrees"},
                    {"value": "20", "label": "±20 degrees"},
                    {"value": "45", "label": "±45 degrees"},
                    {"value": "90", "label": "±90 degrees"},
                ],
                **meas,
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
                **meas,
            },
            {
                "key": "accel_device",
                "label": "Hardware acceleration",
                "kind": "accel",
                **meas,
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
            "Open Model editor: Update pulls the image through prior tools, select a region, "
            "draw figures (rectangle / circle / double circle / polygon), Save solid model."
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
        tool_id = tool.get("id", "")
        model = self._load_model(project_id, tool_id)
        if model is None:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "confidence": 0.0,
                "min_confidence": min_confidence,
                "accel": accel.get("id", "cpu"),
                "message": "Solid model is not saved. Open Model, draw a figure, Register.",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, image

        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        search = _gray(image[y0:y1, x0:x1])
        model_gray = _trim_model(_gray(model))
        if min(model_gray.shape[:2]) < 8:
            model_gray = _gray(model)
        model_edges = self._load_model_edges(project_id, tool_id)
        if model_edges is not None and model_edges.shape[:2] != model_gray.shape[:2]:
            model_edges = cv2.resize(
                model_edges,
                (model_gray.shape[1], model_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        span = int(cfg.get("angle_range") or 0)
        step = float(cfg.get("angle_step") or 2)
        angles = _angle_list(span, step)

        best = _match_features(search, model_gray)
        # OpenCL matchTemplate invents near-perfect scores at the wrong scale/angle
        # on this stack (e.g. 1.0 @ 10° while CPU has 0.98 @ 0°). Keep template on CPU.
        templ = _match_template_pyramid(search, model_gray, angles, False, model_edges)
        best = _pick_best_match(best, templ, model_gray.shape)

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
            "correlation": round(best["confidence"] * 100.0, 4),
            "message": (
                f"Correlation {best['confidence'] * 100:.1f}  "
                f"X {best['x']:.1f}  Y {best['y']:.1f}  "
                f"Angle {best['angle']:.1f}°  ({method})"
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
    def _model_dir(project_id: str, tool_id: str) -> Path | None:
        from app.config import DATA_DIR

        candidates = [
            DATA_DIR / "projects" / project_id / "tools" / tool_id,
            Path(__file__).resolve().parents[2] / "data" / "projects" / project_id / "tools" / tool_id,
        ]
        return next((p for p in candidates if (p / "shape_model.png").exists()), None)

    @classmethod
    def _load_model(cls, project_id: str, tool_id: str) -> np.ndarray | None:
        folder = cls._model_dir(project_id, tool_id)
        if folder is None:
            return None
        path = folder / "shape_model.png"
        key = f"gray:{project_id}:{tool_id}:{path}"
        stamp = path.stat().st_mtime
        cached = _model_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            _model_cache[key] = (stamp, image)
        return image

    @classmethod
    def _load_model_edges(cls, project_id: str, tool_id: str) -> np.ndarray | None:
        folder = cls._model_dir(project_id, tool_id)
        if folder is None:
            return None
        path = folder / "shape_model_edges.png"
        if not path.exists():
            return None
        key = f"edges:{project_id}:{tool_id}:{path}"
        stamp = path.stat().st_mtime
        cached = _model_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            _model_cache[key] = (stamp, image)
        return image


register(ShapeSearchTool())
