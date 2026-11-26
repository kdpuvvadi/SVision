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


def _trim_bbox(gray: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    """Content bounding box, or None if the image should stay full-frame."""
    if gray.size == 0:
        return None
    mask = _content_mask(gray)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None
    return y0, y1, x0, x1


def _trim_model(gray: np.ndarray, pad: int = 2) -> np.ndarray:
    """Drop flat border so matching focuses on the pattern."""
    box = _trim_bbox(gray, pad=pad)
    if box is None:
        return gray
    y0, y1, x0, x1 = box
    return gray[y0:y1, x0:x1]


def _trim_model_pair(
    gray: np.ndarray,
    edges: np.ndarray | None,
    pad: int = 2,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Trim gray + edges with the same crop so rotation overlays stay aligned."""
    box = _trim_bbox(gray, pad=pad)
    if box is None:
        return gray, edges
    y0, y1, x0, x1 = box
    g = gray[y0:y1, x0:x1]
    if edges is None or edges.size == 0:
        return g, edges
    if edges.shape[:2] != gray.shape[:2]:
        edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return g, edges[y0:y1, x0:x1]


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


def _rotate_template(
    gray: np.ndarray,
    angle: float,
    *,
    nearest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.warpAffine(gray, matrix, (nw, nh), flags=flags, borderValue=0)
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


def _fit_model_into_search(
    model: np.ndarray,
    edges: np.ndarray,
    search: np.ndarray,
    angles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink the model so every rotated template still fits inside search.

    A model that is only slightly smaller than the ROI grows under rotation
    (w|cos|+h|sin|) and then every non-zero angle is skipped — that produced
    NG + a broken overlay on rotated pieces.
    """
    sh, sw = search.shape[:2]
    mh, mw = model.shape[:2]
    max_ang = max((abs(float(a)) for a in angles), default=0.0)
    # Headroom for the top scale step (1.08) in the pyramid.
    scale_headroom = 1.10
    rad = np.radians(max_ang)
    c, s = abs(float(np.cos(rad))), abs(float(np.sin(rad)))
    rot_w = (mw * c + mh * s) * scale_headroom
    rot_h = (mh * c + mw * s) * scale_headroom
    margin = 0.92
    fit = min(1.0, (sw * margin) / max(rot_w, 1.0), (sh * margin) / max(rot_h, 1.0))
    # If the model barely fits the ROI, shrink so multi-scale can lock on the live part.
    if mw >= sw * 0.85 or mh >= sh * 0.85:
        area_cap = min(1.0, float(np.sqrt((0.50 * sh * sw) / max(mh * mw, 1.0))))
        fit = min(fit, max(0.50, area_cap))
    if fit >= 0.999:
        return model, edges
    new_size = (max(8, int(round(mw * fit))), max(8, int(round(mh * fit))))
    model = cv2.resize(model, new_size, interpolation=cv2.INTER_AREA)
    edges = cv2.resize(edges, new_size, interpolation=cv2.INTER_NEAREST)
    return model, edges


def _match_template_pyramid(
    search: np.ndarray,
    model: np.ndarray,
    angles: list[float],
    opencl: bool,
    model_edges: np.ndarray | None = None,
) -> dict | None:
    """Gray-led correlation with edge assist (Omron-style solid model)."""
    search_s, s_scale = _downscale(search, _TEMPLATE_MAX_EDGE)
    target_w = max(8, int(round(model.shape[1] * s_scale)))
    target_h = max(8, int(round(model.shape[0] * s_scale)))
    model_s = cv2.resize(model, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if model_edges is not None and model_edges.size:
        edges_s = cv2.resize(model_edges, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        edges_s = _edges(model_s)
    model_s, edges_s = _fit_model_into_search(model_s, edges_s, search_s, angles)

    search_e = _edges(search_s)
    # Broad scale sweep: registered models are often larger than the live part,
    # and rotated hits need smaller templates to stay inside the search ROI.
    scales = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.08]
    best = None
    mh0, mw0 = model_s.shape[:2]
    for scale in scales:
        mw = max(8, int(round(mw0 * scale)))
        mh = max(8, int(round(mh0 * scale)))
        scaled = cv2.resize(model_s, (mw, mh), interpolation=cv2.INTER_LINEAR)
        scaled_e = cv2.resize(edges_s, (mw, mh), interpolation=cv2.INTER_NEAREST)
        for angle in angles:
            templ, matrix = _rotate_template(scaled, angle)
            templ_e, _ = _rotate_template(scaled_e, angle, nearest=True)
            if templ.shape[0] >= search_s.shape[0] or templ.shape[1] >= search_s.shape[1]:
                continue
            score_g, px_g, py_g = _match_template_once(search_s, templ, opencl)
            score_e, px_e, py_e = _match_template_once(search_e, templ_e, opencl)
            # Gray owns pose under rotation — edge maps smear and invent peaks.
            # Only blend edges when their peak sits on the same hit.
            px, py = px_g, py_g
            score = max(0.0, float(score_g))
            if (
                score_e >= 0.22
                and abs(px_e - px_g) <= 4
                and abs(py_e - py_g) <= 4
            ):
                score = float(0.60 * score_g + 0.40 * max(0.0, score_e))
            if best is not None:
                gain = score - float(best["confidence"])
                # Prefer higher score. For tiny gains at larger |angle|, keep the
                # previous hit — suppresses ±step flicker without blocking real rotation.
                if gain < -1e-9:
                    continue
                if gain < 0.01 and abs(angle) > abs(float(best["angle"])) + 1e-6:
                    continue
                if gain <= 1e-9:
                    continue
            # Prefer the largest silhouette for overlay — tiny debris rings look broken when rotated.
            outline_locals = _model_outlines(scaled)
            main_local = max(
                outline_locals,
                key=lambda pts: float(cv2.contourArea(pts.astype(np.float32))),
            )
            outline_locals = [main_local]
            outlines = []
            for outline_local in outline_locals:
                ones = np.ones((len(outline_local), 1), dtype=np.float64)
                outline = (matrix @ np.hstack([outline_local, ones]).T).T
                outline[:, 0] = (outline[:, 0] + px) / s_scale
                outline[:, 1] = (outline[:, 1] + py) / s_scale
                outlines.append(outline)
            all_pts = np.vstack(outlines) if outlines else np.zeros((0, 2), dtype=np.float64)
            # Oriented model rectangle (stable under rotation) — always available for overlay.
            corners = np.array([[0.0, 0.0], [mw, 0.0], [mw, mh], [0.0, mh]], dtype=np.float64)
            ones4 = np.ones((4, 1), dtype=np.float64)
            box = (matrix @ np.hstack([corners, ones4]).T).T
            box[:, 0] = (box[:, 0] + px) / s_scale
            box[:, 1] = (box[:, 1] + py) / s_scale
            top_local = main_local[int(np.argmin(main_local[:, 1]))].reshape(1, 2)
            tip_anchor = (matrix @ np.hstack([top_local, np.ones((1, 1))]).T).T[0]
            tip_anchor[0] = (tip_anchor[0] + px) / s_scale
            tip_anchor[1] = (tip_anchor[1] + py) / s_scale
            center = all_pts.mean(axis=0) if all_pts.shape[0] else box.mean(axis=0)
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


def _build_hit_from_match(
    scaled: np.ndarray,
    matrix: np.ndarray,
    px: float,
    py: float,
    angle: float,
    score: float,
    s_scale: float = 1.0,
) -> dict:
    mh, mw = scaled.shape[:2]
    outline_locals = _model_outlines(scaled)
    main_local = max(
        outline_locals,
        key=lambda pts: float(cv2.contourArea(pts.astype(np.float32))),
    )
    ones = np.ones((len(main_local), 1), dtype=np.float64)
    outline = (matrix @ np.hstack([main_local, ones]).T).T
    outline[:, 0] = (outline[:, 0] + px) / s_scale
    outline[:, 1] = (outline[:, 1] + py) / s_scale
    corners = np.array([[0.0, 0.0], [mw, 0.0], [mw, mh], [0.0, mh]], dtype=np.float64)
    ones4 = np.ones((4, 1), dtype=np.float64)
    box = (matrix @ np.hstack([corners, ones4]).T).T
    box[:, 0] = (box[:, 0] + px) / s_scale
    box[:, 1] = (box[:, 1] + py) / s_scale
    top_local = main_local[int(np.argmin(main_local[:, 1]))].reshape(1, 2)
    tip_anchor = (matrix @ np.hstack([top_local, np.ones((1, 1))]).T).T[0]
    tip_anchor[0] = (tip_anchor[0] + px) / s_scale
    tip_anchor[1] = (tip_anchor[1] + py) / s_scale
    center = outline.mean(axis=0)
    outward = tip_anchor - center
    norm = float(np.hypot(outward[0], outward[1])) or 1.0
    tip_end = tip_anchor + outward * (12.0 / norm)
    return {
        "confidence": float(score),
        "x": float(center[0]),
        "y": float(center[1]),
        "width": float(mw) / s_scale,
        "height": float(mh) / s_scale,
        "angle": float(angle),
        "box": box.tolist(),
        "outline": outline.tolist(),
        "outlines": [outline.tolist()],
        "tip": [tip_anchor.tolist(), tip_end.tolist()],
        "method": "template",
    }


def _refine_template_hit(
    search: np.ndarray,
    model: np.ndarray,
    model_edges: np.ndarray | None,
    hit: dict,
    opencl: bool = False,
) -> dict:
    """Full-resolution polish around a pyramid hit — lifts borderline rotated scores."""
    if hit is None:
        return hit
    angle0 = float(hit.get("angle") or 0)
    edges = model_edges if model_edges is not None and model_edges.size else _edges(model)
    model_f, edges_f = _fit_model_into_search(model, edges, search, [angle0 - 4, angle0, angle0 + 4])
    search_e = _edges(search)
    best = hit
    for scale in (0.85, 0.92, 1.0, 1.06):
        mw = max(8, int(round(model_f.shape[1] * scale)))
        mh = max(8, int(round(model_f.shape[0] * scale)))
        scaled = cv2.resize(model_f, (mw, mh), interpolation=cv2.INTER_LINEAR)
        scaled_e = cv2.resize(edges_f, (mw, mh), interpolation=cv2.INTER_NEAREST)
        for angle in (angle0 - 2, angle0 - 1, angle0, angle0 + 1, angle0 + 2):
            templ, matrix = _rotate_template(scaled, angle)
            templ_e, _ = _rotate_template(scaled_e, angle, nearest=True)
            if templ.shape[0] >= search.shape[0] or templ.shape[1] >= search.shape[1]:
                continue
            score_g, px_g, py_g = _match_template_once(search, templ, opencl)
            score_e, px_e, py_e = _match_template_once(search_e, templ_e, opencl)
            score = max(0.0, float(score_g))
            if score_e >= 0.22 and abs(px_e - px_g) <= 4 and abs(py_e - py_g) <= 4:
                score = float(0.60 * score_g + 0.40 * max(0.0, score_e))
            if score <= float(best.get("confidence") or 0) + 1e-6:
                continue
            cand = _build_hit_from_match(scaled, matrix, px_g, py_g, angle, score, 1.0)
            cand["method"] = "template"
            best = cand
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

    SIFT is the rotation authority when angles diverge. Template usually scores
    the registered piece higher on low-texture models when angles agree.
    """
    if sift is None:
        return templ
    if templ is None:
        return sift
    if _poses_agree(sift, templ, model_shape):
        angle_delta = abs(float(sift.get("angle") or 0) - float(templ.get("angle") or 0))
        # Divergent angles on the same part: keep SIFT pose (handles real rotation).
        if angle_delta > 6.0 and float(sift["confidence"]) >= 0.35:
            out = dict(sift)
            out["method"] = "sift"
            return out
        if float(templ["confidence"]) >= float(sift["confidence"]):
            out = dict(templ)
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
            "angle_range": "45",
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

        span = int(cfg.get("angle_range") or 0)
        step = float(cfg.get("angle_step") or 2)
        angles = _angle_list(span, step)
        x0, y0, x1, y1 = _roi_box(image, cfg.get("search_roi"))
        search = _gray(image[y0:y1, x0:x1])
        model_edges = self._load_model_edges(project_id, tool_id)
        model_gray, model_edges = _trim_model_pair(_gray(model), model_edges)
        if min(model_gray.shape[:2]) < 8:
            model_gray = _gray(model)
            if model_edges is not None and model_edges.shape[:2] != model_gray.shape[:2]:
                model_edges = cv2.resize(
                    model_edges,
                    (model_gray.shape[1], model_gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        best = _match_features(search, model_gray)
        # If SIFT found a rotated pose outside the configured sweep, add a local window.
        if best is not None:
            sift_ang = float(best.get("angle") or 0)
            if abs(sift_ang) > 0.5 and not any(abs(sift_ang - a) <= step * 0.75 for a in angles):
                refine = [sift_ang + d for d in (-2 * step, -step, 0.0, step, 2 * step)]
                angles = sorted({round(float(a), 3) for a in (angles + refine)})
        # OpenCL matchTemplate invents near-perfect scores at the wrong scale/angle.
        # Keep template on CPU.
        templ = _match_template_pyramid(search, model_gray, angles, False, model_edges)
        if templ is not None:
            templ = _refine_template_hit(search, model_gray, model_edges, templ, False)
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
    def _frame_poly(pts: np.ndarray, pad_frac: float = 0.12, pad_px: float = 10.0) -> np.ndarray:
        """Grow an oriented frame so it sits around the part, not inside it."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            return pts
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (cx, cy), (w, h), ang = rect
        w = float(w) * (1.0 + pad_frac) + 2.0 * pad_px
        h = float(h) * (1.0 + pad_frac) + 2.0 * pad_px
        return cv2.boxPoints(((cx, cy), (max(1.0, w), max(1.0, h)), ang)).astype(np.float64)

    @staticmethod
    def _expand_poly(pts: np.ndarray, frac: float = 0.12, min_px: float = 6.0) -> np.ndarray:
        """Push polygon vertices outward from the centroid."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            return pts
        center = pts.mean(axis=0)
        out = np.empty_like(pts)
        for i, p in enumerate(pts):
            v = p - center
            n = float(np.hypot(v[0], v[1])) or 1.0
            grow = max(min_px, n * frac)
            out[i] = p + v * (grow / n)
        return out

    @staticmethod
    def _live_silhouette(image: np.ndarray, found: dict) -> np.ndarray | None:
        """Recover the actual part outline under the hit so the overlay frames the live item."""
        box = np.asarray(found.get("box") or [], dtype=np.float64)
        if box.shape[0] < 4:
            return None
        cx = float(found.get("x") or box[:, 0].mean())
        cy = float(found.get("y") or box[:, 1].mean())
        bw = float(found.get("width") or (np.linalg.norm(box[1] - box[0]) + 1.0))
        bh = float(found.get("height") or (np.linalg.norm(box[2] - box[1]) + 1.0))
        pad = int(max(24.0, 0.45 * max(bw, bh)))
        x0 = max(0, int(cx - bw * 0.5 - pad))
        y0 = max(0, int(cy - bh * 0.5 - pad))
        x1 = min(image.shape[1], int(cx + bw * 0.5 + pad))
        y1 = min(image.shape[0], int(cy + bh * 0.5 + pad))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        roi = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        # Soft threshold against local background — works for dark parts on light decks.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = float(np.percentile(blur, 70))
        mask = (blur < max(8.0, thr - 8.0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        # Dilate so recovered silhouette sits outside the visible edge, not on it.
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        lx, ly = cx - x0, cy - y0
        best = None
        best_score = -1.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 0.15 * bw * bh:
                continue
            m = cv2.moments(cnt)
            if m["m00"] <= 1e-6:
                continue
            mx, my = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dist = float(np.hypot(mx - lx, my - ly))
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best = cnt
        if best is None:
            return None
        pts = best.reshape(-1, 2).astype(np.float64)
        pts[:, 0] += x0
        pts[:, 1] += y0
        return pts

    @staticmethod
    def _draw_label(view: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
        """TrueType plate rendered at 2× then downscaled — stays sharp in the UI preview."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            Image = ImageDraw = ImageFont = None  # type: ignore
        base = max(22, int(round(min(view.shape[:2]) / 22.0)))
        pad = max(10, base // 3)
        border = max(2, base // 9)
        if Image is not None:
            font_path = next(
                (
                    p
                    for p in (
                        r"C:\Windows\Fonts\segoeui.ttf",
                        r"C:\Windows\Fonts\arial.ttf",
                        r"C:\Windows\Fonts\calibri.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    )
                    if __import__("os").path.exists(p)
                ),
                None,
            )
            up = 2
            try:
                font = ImageFont.truetype(font_path, base * up) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            probe = Image.new("RGB", (8, 8), (0, 0, 0))
            draw = ImageDraw.Draw(probe)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            plate_w_up = tw + pad * 2 * up
            plate_h_up = th + pad * 2 * up
            plate = Image.new("RGB", (plate_w_up, plate_h_up), (16, 18, 20))
            draw = ImageDraw.Draw(plate)
            draw.rectangle(
                (0, 0, plate_w_up - 1, plate_h_up - 1),
                outline=(int(color[2]), int(color[1]), int(color[0])),
                width=border * up,
            )
            draw.text(
                (pad * up - bbox[0], pad * up - bbox[1]),
                text,
                font=font,
                fill=(250, 252, 255),
            )
            plate_w = max(1, plate_w_up // up)
            plate_h = max(1, plate_h_up // up)
            rgb = cv2.cvtColor(np.asarray(plate), cv2.COLOR_RGB2BGR)
            rgb = cv2.resize(rgb, (plate_w, plate_h), interpolation=cv2.INTER_AREA)
            x, y = org
            x = int(np.clip(x, 2, max(2, view.shape[1] - plate_w - 2)))
            y = int(np.clip(y - plate_h, 2, max(2, view.shape[0] - plate_h - 2)))
            if view[y : y + plate_h, x : x + plate_w].shape[:2] == (plate_h, plate_w):
                view[y : y + plate_h, x : x + plate_w] = rgb
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(1.15, min(view.shape[:2]) / 560.0)
        thick = max(2, int(round(scale * 2.6)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        plate_w, plate_h = tw + pad * 2, th + baseline + pad * 2
        up = 3
        plate = np.full((plate_h * up, plate_w * up, 3), 16, dtype=np.uint8)
        cv2.rectangle(plate, (0, 0), (plate.shape[1] - 1, plate.shape[0] - 1), color, border * up, cv2.LINE_AA)
        cv2.putText(plate, text, (pad * up, (pad + th) * up), font, scale * up, (250, 252, 255), thick * up, cv2.LINE_AA)
        plate = cv2.resize(plate, (plate_w, plate_h), interpolation=cv2.INTER_AREA)
        x, y = org
        x = int(np.clip(x, 2, max(2, view.shape[1] - plate_w - 2)))
        y = int(np.clip(y - plate_h, 2, max(2, view.shape[0] - plate_h - 2)))
        if view[y : y + plate_h, x : x + plate_w].shape[:2] == (plate_h, plate_w):
            view[y : y + plate_h, x : x + plate_w] = plate

    @staticmethod
    def _draw_hit(image: np.ndarray, found: dict, passed: bool) -> np.ndarray:
        view = image.copy()
        color = (40, 220, 160) if passed else (60, 60, 230)
        thick = max(2, int(round(min(view.shape[:2]) / 240)))
        box = np.asarray(found.get("box") or [], dtype=np.float64)
        rings = found.get("outlines") or [found.get("outline") or []]
        live = ShapeSearchTool._live_silhouette(image, found)
        frame_src = live if live is not None and live.shape[0] >= 3 else None
        if frame_src is None:
            frame_src = box
            for ring in rings:
                pts = np.asarray(ring, dtype=np.float64)
                if pts.shape[0] >= 3:
                    frame_src = pts
                    break
        all_pts: list[np.ndarray] = []
        if frame_src is not None and frame_src.shape[0] >= 3:
            # One oriented frame around the live part — no inset inner contour.
            pad_frac = 0.14 if live is not None else 0.32
            pad_px = max(16.0, float(thick) * 6.0)
            frame = ShapeSearchTool._frame_poly(frame_src, pad_frac=pad_frac, pad_px=pad_px)
            cv2.polylines(view, [np.int32(np.round(frame))], True, color, thick, cv2.LINE_AA)
            all_pts.append(frame)
        tip = np.asarray(found.get("tip") or [], dtype=np.float64)
        if tip.shape == (2, 2):
            p0 = tip[0]
            p1 = tip[1]
            v = p1 - p0
            n = float(np.hypot(v[0], v[1])) or 1.0
            p1 = p0 + v * ((n + max(8.0, thick * 4)) / n)
            a = (int(round(p0[0])), int(round(p0[1])))
            b = (int(round(p1[0])), int(round(p1[1])))
            cv2.line(view, a, b, color, thick, cv2.LINE_AA)
            cv2.drawMarker(view, a, color, cv2.MARKER_CROSS, max(16, thick * 7), thick)
        label = f"{found['confidence'] * 100:.0f}%   {found['angle']:.1f} deg"
        stacked = np.vstack(all_pts) if all_pts else np.array([[found["x"], found["y"]]], dtype=np.float64)
        lx = int(round(float(np.max(stacked[:, 0])))) + 10
        ly = int(round(float(np.min(stacked[:, 1])))) - 8
        ShapeSearchTool._draw_label(view, label, (lx, ly), color)
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
