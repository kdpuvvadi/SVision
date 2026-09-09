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

from app.core.features import difference_map
from app.tools.base import Tool


class ClassifyTool(Tool):
    type = "classify"
    title = "ML Judgment"
    summary = "OK or NG from this tool's own samples."

    def fields(self) -> list[dict]:
        return [
            {
                "key": "min_confidence",
                "label": "Minimum confidence",
                "kind": "range",
                "min": 0.5,
                "max": 0.95,
                "step": 0.01,
            },
            {"key": "roi", "label": "Region", "kind": "roi"},
        ]

    def panels(self) -> list[str]:
        return ["samples"]

    def default_config(self) -> dict:
        return {"min_confidence": 0.55, "roi": None}

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        from app.storage.store import store

        cfg = tool.get("config") or {}
        model = store.get_model(project_id, tool["id"])
        pred = model.predict(image, roi=cfg.get("roi"), min_confidence=float(cfg.get("min_confidence", 0.55)))
        overlay = None
        if model.ok_reference is not None and pred["judgment"] == "NG":
            overlay = difference_map(image, model.ok_reference, cfg.get("roi"))
        result = {
            "id": tool.get("id"),
            "type": self.type,
            "name": tool.get("name") or self.title,
            "judgment": pred["judgment"],
            "label": pred["label"],
            "confidence": pred["confidence"],
            "distance_ok": pred["distance_ok"],
            "distance_ng": pred["distance_ng"],
            "residual": pred.get("residual", 0.0),
            "min_confidence": float(cfg.get("min_confidence", 0.55)),
            "elapsed_ms": pred["elapsed_ms"],
            "message": "",
        }
        return result, overlay
