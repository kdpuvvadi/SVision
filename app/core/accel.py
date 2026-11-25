# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# Hardware acceleration discovery for inspection tools (OCR, Shape Search, …).

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Any


def _nvidia_gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    names: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.lower().startswith("gpu "):
            continue
        # GPU 0: NVIDIA RTX 500 Ada Generation Laptop GPU (UUID: ...)
        try:
            names.append(line.split(":", 1)[1].split("(", 1)[0].strip())
        except Exception:
            names.append(line)
    return names


def _ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _opencl_ok() -> bool:
    try:
        import cv2

        return bool(cv2.ocl.haveOpenCL())
    except Exception:
        return False


@lru_cache(maxsize=1)
def list_accel_devices() -> list[dict[str, Any]]:
    """Return selectable accel backends for tool settings."""
    providers = _ort_providers()
    nvidia = _nvidia_gpus()
    opencl = _opencl_ok()
    cuda_ort = any("CUDA" in p for p in providers)
    dml_ort = any("Dml" in p or "DirectML" in p for p in providers)

    devices: list[dict[str, Any]] = [
        {
            "id": "cpu",
            "label": "CPU",
            "kind": "cpu",
            "ready": True,
            "detail": "Always available",
        }
    ]

    if opencl:
        label = "OpenCL GPU"
        if nvidia:
            label = f"OpenCL ({nvidia[0]})"
        devices.append(
            {
                "id": "opencl",
                "label": label,
                "kind": "opencl",
                "ready": True,
                "detail": "OpenCV OpenCL (shape match / image ops)",
            }
        )

    if nvidia:
        detail_parts = []
        if opencl:
            detail_parts.append("OpenCV via OpenCL")
        if cuda_ort:
            detail_parts.append("OCR via CUDA")
        else:
            detail_parts.append("OCR needs onnxruntime-gpu")
        devices.append(
            {
                "id": "cuda",
                "label": f"NVIDIA ({nvidia[0]})",
                "kind": "cuda",
                "ready": bool(cuda_ort or opencl),
                "ocr_cuda": bool(cuda_ort),
                "detail": "; ".join(detail_parts),
            }
        )

    if dml_ort:
        devices.append(
            {
                "id": "directml",
                "label": "DirectML (Windows GPU)",
                "kind": "directml",
                "ready": True,
                "detail": "ONNX Runtime DirectML",
            }
        )

    return devices


def resolve_accel(device_id: str | None) -> dict[str, Any]:
    wanted = (device_id or "cpu").strip().lower() or "cpu"
    devices = {d["id"]: d for d in list_accel_devices()}
    if wanted in devices and devices[wanted].get("ready"):
        return devices[wanted]
    if wanted in devices and not devices[wanted].get("ready"):
        # Fall back but keep requested id for messaging.
        return {**devices["cpu"], "requested": wanted, "fallback": True}
    return devices["cpu"]


def use_opencl(device_id: str | None) -> bool:
    resolved = resolve_accel(device_id)
    return resolved.get("kind") in {"opencl", "cuda"} and _opencl_ok()


def ocr_use_cuda(device_id: str | None) -> bool:
    resolved = resolve_accel(device_id)
    if resolved.get("kind") != "cuda":
        return False
    providers = _ort_providers()
    return any("CUDA" in p for p in providers)
