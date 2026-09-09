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

from app.core.features import apply_roi

_zxing = None
_zxing_error = ""

# UI value -> zxing BarcodeFormat names. LinearCodes covers all 1D barcodes.
_FORMAT_NAMES = {
    "auto": ("Any",),
    "qr": ("QRCode", "MicroQRCode"),
    "datamatrix": ("DataMatrix",),
    "aztec": ("Aztec",),
    "pdf417": ("PDF417",),
    "barcode": ("LinearCodes",),
    "code128": ("Code128",),
    "code39": ("Code39",),
    "code93": ("Code93",),
    "ean13": ("EAN13",),
    "ean8": ("EAN8",),
    "upca": ("UPCA",),
    "upce": ("UPCE",),
    "itf": ("ITF",),
    "codabar": ("Codabar",),
}

_LABELS = {
    "auto": "Any",
    "qr": "QR",
    "datamatrix": "Data Matrix",
    "aztec": "Aztec",
    "pdf417": "PDF417",
    "barcode": "Barcode",
    "code128": "Code 128",
    "code39": "Code 39",
    "code93": "Code 93",
    "ean13": "EAN-13",
    "ean8": "EAN-8",
    "upca": "UPC-A",
    "upce": "UPC-E",
    "itf": "ITF",
    "codabar": "Codabar",
}


def code_types() -> list[dict]:
    return [{"value": key, "label": _LABELS[key]} for key in _FORMAT_NAMES]


def reader_available() -> bool:
    try:
        _library()
        return True
    except Exception:
        return False


def _library():
    global _zxing, _zxing_error
    if _zxing is not None:
        return _zxing
    if _zxing_error:
        raise RuntimeError(_zxing_error)
    try:
        import zxingcpp
    except ImportError as exc:
        _zxing_error = "Code reader is missing. Run: pip install zxing-cpp"
        raise RuntimeError(_zxing_error) from exc
    _zxing = zxingcpp
    return _zxing


def _key(symbology: str) -> str:
    key = (symbology or "auto").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {"datamatix": "datamatrix", "qrcode": "qr", "1d": "barcode", "linear": "barcode"}
    key = aliases.get(key, key)
    return key if key in _FORMAT_NAMES else "auto"


def _formats(symbology: str):
    zxingcpp = _library()
    names = _FORMAT_NAMES[_key(symbology)]
    flags = None
    for name in names:
        item = getattr(zxingcpp.BarcodeFormat, name, None)
        if item is None:
            continue
        flags = item if flags is None else flags | item
    if flags is None and _key(symbology) == "barcode":
        for name in ("Code128", "Code39", "Code93", "EAN13", "EAN8", "UPCA", "UPCE", "ITF", "Codabar"):
            item = getattr(zxingcpp.BarcodeFormat, name, None)
            if item is None:
                continue
            flags = item if flags is None else flags | item
    return flags


def _type_name(fmt) -> str:
    text = str(fmt)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return {
        "QRCode": "QR",
        "MicroQRCode": "QR",
        "DataMatrix": "DataMatrix",
        "PDF417": "PDF417",
        "Code128": "Code128",
        "Code39": "Code39",
        "Code93": "Code93",
        "EAN13": "EAN13",
        "EAN8": "EAN8",
        "UPCA": "UPCA",
        "UPCE": "UPCE",
    }.get(text, text)


def _decode(gray: np.ndarray, symbology: str) -> list[dict]:
    zxingcpp = _library()
    flags = _formats(symbology)
    kwargs = {"try_rotate": False, "try_downscale": True}
    if flags is not None and _key(symbology) != "auto":
        kwargs["formats"] = flags
    try:
        results = zxingcpp.read_barcodes(gray, **kwargs)
    except TypeError:
        results = zxingcpp.read_barcodes(gray)
    found = []
    for item in results or []:
        value = str(getattr(item, "text", "") or "").strip()
        if not value:
            continue
        found.append({"type": _type_name(getattr(item, "format", "")), "value": value})
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
    kind = _key(symbology)
    label = _LABELS[kind]
    try:
        cropped = apply_roi(image, roi)
        gray = cropped if cropped.ndim == 2 else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        codes = _unique(_decode(gray, kind))
        if not codes:
            codes = _unique(_decode(255 - gray, kind))
    except Exception as exc:
        return {
            "judgment": "NG",
            "codes": [],
            "expected": expected,
            "symbology": kind,
            "available": False,
            "message": str(exc),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    values = [c["value"] for c in codes]
    expected_text = (expected or "").strip()
    if expected_text:
        passed = any(expected_text.lower() == v.lower() or expected_text.lower() in v.lower() for v in values)
    else:
        passed = len(values) > 0

    message = ", ".join(f"{c['type']}: {c['value']}" for c in codes)
    if not codes:
        message = f"No {label} code"

    return {
        "judgment": "OK" if passed else "NG",
        "codes": codes,
        "expected": expected,
        "symbology": kind,
        "available": True,
        "message": message,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
