from __future__ import annotations

import time

import numpy as np

from app.tools.base import Tool


class CameraTool(Tool):
    type = "camera"
    title = "Camera"
    summary = "Image source for this flow. Upload a file, or use a GigE camera later."
    singleton = True

    def fields(self) -> list[dict]:
        return [
            {
                "key": "source",
                "label": "Source",
                "kind": "select",
                "options": [
                    {"value": "upload", "label": "Upload image"},
                    {"value": "gige", "label": "GigE camera"},
                ],
            },
            {
                "key": "gige_ip",
                "label": "GigE IP",
                "kind": "text",
                "placeholder": "192.168.0.10",
            },
        ]

    def default_config(self) -> dict:
        return {"source": "upload", "gige_ip": ""}

    def status(self) -> str:
        return "Required. Drop or choose images only after this tool is in the flow."

    def catalog(self) -> dict:
        data = super().catalog()
        data["singleton"] = True
        data["role"] = "input"
        return data

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        source = cfg.get("source") or "upload"
        if source == "gige":
            ip = (cfg.get("gige_ip") or "").strip() or "no IP"
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": tool.get("name") or self.title,
                "judgment": "NG",
                "message": f"GigE camera {ip} is not connected.",
                "source": source,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, None
        if image is None or image.size == 0:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": tool.get("name") or self.title,
                "judgment": "NG",
                "message": "No image. Drop a file or choose images.",
                "source": source,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, None
        h, w = image.shape[:2]
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": tool.get("name") or self.title,
            "judgment": "OK",
            "message": f"Upload  {w}×{h}",
            "source": source,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }, None
