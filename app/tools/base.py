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
