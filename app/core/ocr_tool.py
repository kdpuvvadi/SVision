from __future__ import annotations

import time

import cv2
import numpy as np

from app.core.features import apply_roi

_engine = None
_engine_error = ""
_REC_HEIGHT = 48


def ocr_available() -> bool:
    try:
        _reader()
        return True
    except Exception:
        return False


def _reader():
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error:
        raise RuntimeError(_engine_error)
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        _engine_error = "OCR library is missing. Run: pip install rapidocr-onnxruntime"
        raise RuntimeError(_engine_error) from exc
    _engine = RapidOCR()
    return _engine


def _match(text: str, expected: str, mode: str) -> bool:
    found = " ".join(text.split())
    want = " ".join((expected or "").split())
    if not want:
        return bool(found)
    if mode == "exact":
        return found.lower() == want.lower()
    if mode == "regex":
        import re

        try:
            return re.search(want, found, re.IGNORECASE) is not None
        except re.error:
            return False
    return want.lower() in found.lower()


def _valid_roi(roi: dict | None) -> bool:
    if not isinstance(roi, dict):
        return False
    try:
        width = float(roi.get("w", 0))
        height = float(roi.get("h", 0))
    except (TypeError, ValueError):
        return False
    return width > 0.01 and height > 0.01


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    dark = float(gray.mean()) < 127.0
    work = 255 - gray if dark else gray
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _trim(gray: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return gray
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    return gray[y0:y1, x0:x1]


def _line_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    row = mask.mean(axis=1)
    active = row > 6.0
    spans: list[tuple[int, int]] = []
    start = None
    for index, on in enumerate(active.tolist()):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= 4:
                spans.append((start, index))
            start = None
    if start is not None and mask.shape[0] - start >= 4:
        spans.append((start, mask.shape[0]))
    return spans or [(0, mask.shape[0])]


def _to_rec(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    if height < 1 or width < 1:
        return np.full((48, 16, 3), 255, np.uint8)
    scale = _REC_HEIGHT / float(height)
    view = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), _REC_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    view = cv2.copyMakeBorder(view, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)


def _thicken(gray: np.ndarray) -> np.ndarray:
    mask = _ink_mask(gray)
    if mask.size == 0 or int(mask.max()) == 0:
        return gray
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(mask, kernel, iterations=1)
    out = np.full_like(gray, 255)
    out[thick > 0] = 0
    return out


def _first_text(output) -> tuple[str, float]:
    if output is None:
        return "", 0.0
    if isinstance(output, tuple):
        output = output[0]
    if not output:
        return "", 0.0
    item = output[0]
    if isinstance(item, (list, tuple)):
        if len(item) >= 2 and isinstance(item[1], str):
            score = float(item[2]) if len(item) > 2 and not isinstance(item[2], (list, tuple)) else 1.0
            return item[1].strip(), score
        if item and isinstance(item[0], str):
            score = float(item[1]) if len(item) > 1 else 1.0
            return str(item[0]).strip(), score
    if isinstance(item, str):
        return item.strip(), 1.0
    return "", 0.0


def _recognizer():
    engine = _reader()
    rec = (
        getattr(engine, "text_recognizer", None)
        or getattr(engine, "text_rec", None)
        or getattr(engine, "recognizer", None)
    )
    if rec is None:
        raise RuntimeError("OCR recognizer is unavailable")
    return rec


def _recognize(view: np.ndarray) -> tuple[str, float]:
    output = _recognizer()([np.ascontiguousarray(view)])
    return _first_text(output)


def _prefer(candidates: list[tuple[str, float]], expected: str) -> str:
    wanted = " ".join((expected or "").split()).lower()
    ranked = [item for item in candidates if item[0]]
    if not ranked:
        return ""
    if wanted:
        hits = [item for item in ranked if wanted in item[0].lower() or item[0].lower() in wanted]
        if hits:
            hits.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
            return hits[0][0]
    ranked.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return ranked[0][0]


def _read_line(gray: np.ndarray, expected: str = "") -> str:
    mask = _ink_mask(gray)
    trimmed = _trim(gray, mask)
    if trimmed.size == 0:
        return ""
    wanted = " ".join((expected or "").split()).lower()
    found = []
    for view in (_to_rec(trimmed), _to_rec(_thicken(trimmed))):
        text, score = _recognize(view)
        found.append((text, score))
        if wanted and wanted in text.lower():
            return text
    return _prefer(found, expected)


def _read_crop(gray: np.ndarray, expected: str = "") -> str:
    mask = _ink_mask(gray)
    gray = _trim(gray, mask, pad=2)
    mask = _ink_mask(gray)
    lines = []
    for y0, y1 in _line_spans(mask):
        text = _read_line(gray[y0:y1], expected)
        if text:
            lines.append(text)
    return " ".join(lines)


def read_text(image: np.ndarray, roi: dict | None = None, expected: str = "", match_mode: str = "contains") -> dict:
    started = time.perf_counter()
    try:
        if not _valid_roi(roi):
            raise ValueError("Draw a region. OCR reads only that region.")
        cropped = apply_roi(image, roi)
        if cropped.ndim == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped
        text = _read_crop(gray, expected)
    except Exception as exc:
        return {
            "judgment": "NG",
            "text": "",
            "expected": expected,
            "available": False,
            "message": str(exc),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    passed = _match(text, expected, match_mode)
    return {
        "judgment": "OK" if passed else "NG",
        "text": text,
        "expected": expected,
        "match_mode": match_mode,
        "available": True,
        "message": text or "No text",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
