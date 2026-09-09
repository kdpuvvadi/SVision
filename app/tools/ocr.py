from __future__ import annotations

import numpy as np

from app.core.ocr_tool import ocr_available, read_text
from app.tools.base import Tool


class OcrTool(Tool):
    type = "ocr"
    title = "OCR"
    summary = "Read text only inside the drawn region."

    def fields(self) -> list[dict]:
        return [
            {"key": "expected", "label": "Expected text", "kind": "text", "placeholder": "Leave empty to require any text"},
            {
                "key": "match_mode",
                "label": "Match",
                "kind": "select",
                "options": [
                    {"value": "contains", "label": "Contains"},
                    {"value": "exact", "label": "Exact"},
                    {"value": "regex", "label": "Regex"},
                ],
            },
            {"key": "roi", "label": "Region", "kind": "roi"},
        ]

    def default_config(self) -> dict:
        return {"expected": "", "match_mode": "contains", "roi": None}

    def status(self) -> str:
        if ocr_available():
            return "Draw a region. Only that area is read."
        return "OCR library is missing. Run: pip install rapidocr-onnxruntime"

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        cfg = tool.get("config") or {}
        result = read_text(image, cfg.get("roi"), cfg.get("expected") or "", cfg.get("match_mode") or "contains")
        result.update({"id": tool.get("id"), "type": self.type, "name": tool.get("name") or self.title})
        return result, None
