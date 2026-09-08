from __future__ import annotations

import threading

import cv2
import numpy as np

from app.config import FEATURE_SIZE

_local = threading.local()


def _hog() -> cv2.HOGDescriptor:
    hog = getattr(_local, "hog", None)
    if hog is None:
        hog = cv2.HOGDescriptor(
            (FEATURE_SIZE, FEATURE_SIZE),
            (16, 16),
            (16, 16),
            (8, 8),
            9,
        )
        _local.hog = hog
    return hog


def apply_roi(image: np.ndarray, roi: dict | None) -> np.ndarray:
    if not roi:
        return image
    h, w = image.shape[:2]
    x = float(roi.get("x", 0))
    y = float(roi.get("y", 0))
    rw = float(roi.get("w", 1))
    rh = float(roi.get("h", 1))
    if max(x, y, rw, rh) > 1.5:
        x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
    else:
        x0 = int(round(x * w))
        y0 = int(round(y * h))
        x1 = int(round((x + rw) * w))
        y1 = int(round((y + rh) * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return image[y0:y1, x0:x1]


def letterbox_gray(gray: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    h, w = gray.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((size, size), np.uint8)
    scale = size / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def prepare_gray(image: np.ndarray, roi: dict | None = None) -> np.ndarray:
    cropped = apply_roi(image, roi)
    if cropped.ndim == 2:
        gray = cropped
    else:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = letterbox_gray(gray, FEATURE_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def extract_features(image: np.ndarray, roi: dict | None = None) -> np.ndarray:
    gray = np.ascontiguousarray(prepare_gray(image, roi))
    hog = _hog().compute(gray)
    hog_vec = np.asarray(hog, dtype=np.float32).reshape(-1)

    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    small = small / 255.0

    grid = 4
    cell = FEATURE_SIZE // grid
    means = np.empty(grid * grid, np.float32)
    stds = np.empty(grid * grid, np.float32)
    for gy in range(grid):
        for gx in range(grid):
            patch = gray[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            idx = gy * grid + gx
            means[idx] = float(patch.mean()) / 255.0
            stds[idx] = float(patch.std()) / 255.0

    proj_x = gray.mean(axis=0).astype(np.float32)
    proj_y = gray.mean(axis=1).astype(np.float32)
    proj_x = cv2.resize(proj_x.reshape(1, -1), (32, 1), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0
    proj_y = cv2.resize(proj_y.reshape(-1, 1), (1, 32), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0

    vec = np.concatenate([hog_vec, small, means, stds, proj_x, proj_y]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def _roi_box(image: np.ndarray, roi: dict | None) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    if not roi:
        return 0, 0, w, h
    x = float(roi.get("x", 0))
    y = float(roi.get("y", 0))
    rw = float(roi.get("w", 1))
    rh = float(roi.get("h", 1))
    if max(x, y, rw, rh) > 1.5:
        x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
    else:
        x0, y0 = int(round(x * w)), int(round(y * h))
        x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def difference_map(image: np.ndarray, reference_gray: np.ndarray, roi: dict | None = None) -> np.ndarray:
    gray = prepare_gray(image, roi)
    ref = reference_gray
    if ref.shape != gray.shape:
        ref = cv2.resize(ref, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(gray, ref)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    view = image.copy()
    x0, y0, x1, y1 = _roi_box(view, roi)
    heat = cv2.resize(heat, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    region = view[y0:y1, x0:x1]
    view[y0:y1, x0:x1] = cv2.addWeighted(region, 0.45, heat, 0.55, 0)
    return view
