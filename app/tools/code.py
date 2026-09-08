from __future__ import annotations

import numpy as np

from app.core.code_tool import read_codes, zbar_available
from app.tools.base import Tool


class CodeTool(Tool):
    type = "code"
    title = "Code Read"
    summary = "Read a QR or barcode in this tool's own region."

    def fields(self) -> list[dict]:
        return [
            {"key": "expected", "label": "Expected value", "kind": "text", "placeholder": "Leave empty to require any code"},
            {
                "key": "symbology",
                "label": "Symbology",
                "kind": "select",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "QR", "label": "QR"},
                    {"value": "barcode", "label": "Barcode"},
                ],
            },
            {"key": "roi", "label": "Region", "kind": "roi"},
        ]

    def default_config(self) -> dict:
        return {"expected": "", "symbology": "auto", "roi": None}

    def status(self) -> str:
        if zbar_available():
            return "QR and barcode readers are available."
        return "QR works now. pyzbar adds 1D barcodes."

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        cfg = tool.get("config") or {}
        result = read_codes(image, cfg.get("roi"), cfg.get("expected") or "", cfg.get("symbology") or "auto")
        result.update({"id": tool.get("id"), "type": self.type, "name": tool.get("name") or self.title})
        return result, None
