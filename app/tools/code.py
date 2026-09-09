from __future__ import annotations

import numpy as np

from app.core.code_tool import code_types, read_codes, reader_available
from app.tools.base import Tool


class CodeTool(Tool):
    type = "code"
    title = "Code Read"
    summary = "Read the selected code type in this tool's region."

    def fields(self) -> list[dict]:
        return [
            {"key": "expected", "label": "Expected value", "kind": "text", "placeholder": "Leave empty to require any code"},
            {
                "key": "symbology",
                "label": "Code type",
                "kind": "select",
                "options": code_types(),
            },
            {"key": "roi", "label": "Region", "kind": "roi"},
        ]

    def default_config(self) -> dict:
        return {"expected": "", "symbology": "auto", "roi": None}

    def status(self) -> str:
        if reader_available():
            return "Choose QR, Data Matrix, barcode, or another type. Only that type is read."
        return "Code reader is missing. Run: pip install zxing-cpp"

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        cfg = tool.get("config") or {}
        result = read_codes(image, cfg.get("roi"), cfg.get("expected") or "", cfg.get("symbology") or "auto")
        result.update({"id": tool.get("id"), "type": self.type, "name": tool.get("name") or self.title})
        return result, None
