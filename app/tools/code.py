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
