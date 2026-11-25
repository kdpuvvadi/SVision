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

import numpy as np

from app.tools.base import Tool
from app.tools.registry import register


class CameraTool(Tool):
    type = "camera"
    title = "Camera"
    summary = "Image source for this flow. Upload, USB, or GigE GenTL (.cti) cameras."
    singleton = True
    role = "input"
    placement = "prepend"
    ui_modules = ["/static/tools/camera.js"]

    def fields(self) -> list[dict]:
        return [
            {
                "key": "source",
                "label": "Source",
                "kind": "select",
                "options": [
                    {"value": "upload", "label": "Upload image"},
                    {"value": "usb", "label": "USB camera"},
                    {"value": "gentl", "label": "GigE / GenTL"},
                ],
            },
            {
                "key": "usb_index",
                "label": "USB index",
                "kind": "text",
                "placeholder": "0",
            },
            {
                "key": "cti_path",
                "label": "CTI producer",
                "kind": "cti",
                "placeholder": r"C:\Program Files\Common Files\OMRON_SENTECH\GenTL\v1_5\StGenTL_MD_VC141_v1_5.cti",
            },
            {
                "key": "device_id",
                "label": "Camera",
                "kind": "gentl_device",
                "placeholder": "Scan after choosing a CTI file",
            },
            {
                "key": "gige_ip",
                "label": "IP note",
                "kind": "text",
                "placeholder": "192.168.250.4 (optional note)",
            },
        ]

    def default_config(self) -> dict:
        return {
            "source": "upload",
            "usb_index": "0",
            "cti_path": "",
            "device_id": "",
            "gige_ip": "",
        }

    def status(self) -> str:
        return "Upload for offline checks. USB or GigE GenTL for live PLC measure."

    def mount(self, app) -> None:
        from app.tools import camera_routes

        camera_routes.mount(app)

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        source = cfg.get("source") or "upload"
        if source in {"gige", "gentl"} and (image is None or image.size == 0):
            label = (cfg.get("device_id") or cfg.get("gige_ip") or "no device").strip()
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": tool.get("name") or self.title,
                "judgment": "NG",
                "message": f"GenTL camera {label} produced no frame.",
                "source": source,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, None
        if source == "usb" and (image is None or image.size == 0):
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": tool.get("name") or self.title,
                "judgment": "NG",
                "message": f"USB camera index {cfg.get('usb_index') or 0} produced no frame.",
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
        if source in {"gige", "gentl"}:
            label = "GenTL"
        elif source == "usb":
            label = "USB"
        else:
            label = "Upload"
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": tool.get("name") or self.title,
            "judgment": "OK",
            "message": f"{label}  {w}×{h}",
            "source": source,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }, None


register(CameraTool())
