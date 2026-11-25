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

import threading
from typing import Any

import numpy as np

from app.core.cameras import CameraBackend, CameraStatus, create_backend, resolve_source


class ImageSource:
    """Facade over pluggable camera backends (upload, USB, GenTL/GigE)."""

    def __init__(self, source: str = "upload", config: dict[str, Any] | None = None, **legacy: Any) -> None:
        cfg = dict(config or {})
        if "gige_ip" in legacy and "gige_ip" not in cfg:
            cfg["gige_ip"] = legacy.get("gige_ip") or ""
        if "usb_index" in legacy and "usb_index" not in cfg:
            cfg["usb_index"] = legacy.get("usb_index") or 0
        self.source = resolve_source(source)
        self.config = cfg
        self._backend: CameraBackend = create_backend(self.source, cfg)

    @property
    def gige_ip(self) -> str:
        return str(self.config.get("gige_ip") or "")

    @property
    def usb_index(self) -> int:
        try:
            return int(self.config.get("usb_index") or 0)
        except (TypeError, ValueError):
            return 0

    def status(self) -> CameraStatus:
        return self._backend.status()

    def grab(self) -> np.ndarray:
        return self._backend.grab()

    def close(self) -> None:
        self._backend.close()


def camera_config_from_project(project: dict) -> tuple[str, dict[str, Any]]:
    camera_tool = None
    for tool in (project.get("flow") or {}).get("tools", []):
        if tool.get("type") == "camera":
            camera_tool = tool
            break
    cfg = dict((camera_tool or {}).get("config") or {})
    source = cfg.get("source") or "upload"
    return resolve_source(source), cfg


def camera_from_project(project: dict) -> ImageSource:
    source, cfg = camera_config_from_project(project)
    return camera_session.get(source, cfg)


class _CameraSession:
    """Reuse one live backend while the Camera tool config stays the same."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key: tuple | None = None
        self._source: ImageSource | None = None

    @staticmethod
    def _config_key(source: str, config: dict[str, Any]) -> tuple:
        items = tuple(sorted((str(k), str(v)) for k, v in (config or {}).items()))
        return (resolve_source(source), items)

    def get(self, source: str, config: dict[str, Any] | None = None) -> ImageSource:
        cfg = dict(config or {})
        key = self._config_key(source, cfg)
        with self._lock:
            if self._source is not None and self._key == key:
                return self._source
            if self._source is not None:
                self._source.close()
            self._source = ImageSource(source=source, config=cfg)
            self._key = key
            return self._source

    def close(self) -> None:
        with self._lock:
            if self._source is not None:
                self._source.close()
            self._source = None
            self._key = None


camera_session = _CameraSession()
