from __future__ import annotations

from typing import Any

import numpy as np


class Tool:
    """One inspection capability. Each flow item is an independent instance."""

    type = ""
    title = ""
    summary = ""
    transforms_image = False

    def fields(self) -> list[dict[str, Any]]:
        return []

    def panels(self) -> list[str]:
        return []

    def default_config(self) -> dict[str, Any]:
        return {}

    def status(self) -> str:
        return ""

    def catalog(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "summary": self.summary,
            "status": self.status(),
            "fields": self.fields(),
            "panels": self.panels(),
            "defaults": self.default_config(),
        }

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        raise NotImplementedError
