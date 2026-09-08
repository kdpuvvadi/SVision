from __future__ import annotations

import time

import cv2
import numpy as np

from app.core.features import apply_roi

_qr = cv2.QRCodeDetector()
_qr_lock = __import__("threading").Lock()

try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore

    _HAS_ZBAR = True
except Exception:
    zbar_decode = None
    _HAS_ZBAR = False


def zbar_available() -> bool:
    return _HAS_ZBAR


def _decode_qr(gray: np.ndarray) -> list[dict]:
    found: list[dict] = []
    with _qr_lock:
        try:
            ok, infos, points, _ = _qr.detectAndDecodeMulti(gray)
        except Exception:
            ok, infos, points = False, [], None
        if ok and infos is not None:
            for text in infos:
                value = (text or "").strip()
                if value:
                    found.append({"type": "QR", "value": value})
        if not found:
            value, _points, _ = _qr.detectAndDecode(gray)
            value = (value or "").strip()
            if value:
                found.append({"type": "QR", "value": value})
    return found


def _decode_zbar(gray: np.ndarray) -> list[dict]:
    if not _HAS_ZBAR or zbar_decode is None:
        return []
    found: list[dict] = []
    try:
        for item in zbar_decode(gray):
            value = item.data.decode("utf-8", errors="replace").strip()
            if not value:
                continue
            found.append({"type": str(item.type), "value": value})
    except Exception:
        return []
    return found


def _unique(codes: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for code in codes:
        key = (code.get("type", ""), code.get("value", ""))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(code)
    return out


def read_codes(
    image: np.ndarray,
    roi: dict | None = None,
    expected: str = "",
    symbology: str = "auto",
) -> dict:
    started = time.perf_counter()
    cropped = apply_roi(image, roi)
    gray = cropped if cropped.ndim == 2 else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    codes = _decode_qr(gray)
    want = symbology.upper()
    if want in {"AUTO", "BARCODE", "1D", "DATAMATRIX"} or not codes:
        codes.extend(_decode_zbar(gray))
    codes = _unique(codes)

    if want not in {"", "AUTO"}:
        codes = [c for c in codes if c["type"].upper() == want or (want == "BARCODE" and c["type"].upper() != "QR")]

    values = [c["value"] for c in codes]
    expected_text = (expected or "").strip()
    if expected_text:
        passed = any(expected_text.lower() == v.lower() or expected_text.lower() in v.lower() for v in values)
    else:
        passed = len(values) > 0

    message = ""
    if not codes and not _HAS_ZBAR:
        message = "No QR found. Install pyzbar for 1D barcode and DataMatrix."

    return {
        "judgment": "OK" if passed else "NG",
        "codes": codes,
        "expected": expected,
        "symbology": symbology,
        "zbar": _HAS_ZBAR,
        "message": message,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
