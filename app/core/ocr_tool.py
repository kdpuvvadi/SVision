from __future__ import annotations

import shutil
import time

import cv2
import numpy as np

from app.core.features import apply_roi

_TESSERACT = shutil.which("tesseract")


def tesseract_available() -> bool:
    if _TESSERACT:
        return True
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


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


def read_text(image: np.ndarray, roi: dict | None = None, expected: str = "", match_mode: str = "contains") -> dict:
    started = time.perf_counter()
    if not tesseract_available():
        return {
            "judgment": "NG",
            "text": "",
            "expected": expected,
            "available": False,
            "message": "Tesseract OCR is not installed on this IPC.",
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    import pytesseract  # type: ignore

    cropped = apply_roi(image, roi)
    gray = cropped if cropped.ndim == 2 else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1600:
        scale = 1600.0 / long_edge
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    elif long_edge < 400:
        scale = 400.0 / max(long_edge, 1)
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    variants = [gray, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)]

    best = ""
    for variant in variants:
        try:
            text = pytesseract.image_to_string(variant, config="--oem 1 --psm 6")
        except Exception as exc:
            return {
                "judgment": "NG",
                "text": "",
                "expected": expected,
                "available": False,
                "message": f"OCR failed: {exc}",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        cleaned = " ".join(text.split())
        if len(cleaned) > len(best):
            best = cleaned

    passed = _match(best, expected, match_mode)
    return {
        "judgment": "OK" if passed else "NG",
        "text": best,
        "expected": expected,
        "match_mode": match_mode,
        "available": True,
        "message": "",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
