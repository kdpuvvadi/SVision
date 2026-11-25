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

    def grab_fresh(self) -> np.ndarray:
        backend = self._backend
        if hasattr(backend, "grab_fresh"):
            return backend.grab_fresh()
        return backend.grab()

    @property
    def last_grab_ms(self) -> float:
        return float(getattr(self._backend, "_last_grab_ms", 0.0) or 0.0)

    @property
    def last_open_ms(self) -> float:
        return float(getattr(self._backend, "_last_open_ms", 0.0) or 0.0)

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
    """Reuse live backends across measures.

    PLC ready-polling can touch a different production scene every few hundred ms.
    A single global slot would close the GenTL session each time and force an
    ~1–8 s reopen on the next Measure. Keep one backend per config key; only
    evict other exclusive USB/GenTL sessions when opening a conflicting live cam.
    """

    _EXCLUSIVE = frozenset({"usb", "gige", "gentl"})

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[tuple, ImageSource] = {}

    @staticmethod
    def _config_key(source: str, config: dict[str, Any]) -> tuple:
        items = tuple(sorted((str(k), str(v)) for k, v in (config or {}).items()))
        return (resolve_source(source), items)

    def get(self, source: str, config: dict[str, Any] | None = None) -> ImageSource:
        cfg = dict(config or {})
        key = self._config_key(source, cfg)
        with self._lock:
            hit = self._sessions.get(key)
            if hit is not None:
                return hit
            resolved = resolve_source(source)
            if resolved in self._EXCLUSIVE:
                for other_key, session in list(self._sessions.items()):
                    if other_key[0] in self._EXCLUSIVE:
                        session.close()
                        del self._sessions[other_key]
            created = ImageSource(source=source, config=cfg)
            self._sessions[key] = created
            return created

    def close(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()


camera_session = _CameraSession()
