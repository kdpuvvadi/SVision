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

import cv2
import numpy as np

from app.core.cameras.base import CameraBackend, CameraStatus
from app.core.cameras.registry import register_backend


@register_backend
class UsbBackend(CameraBackend):
    source = "usb"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            self.index = int(self.config.get("usb_index") or 0)
        except (TypeError, ValueError):
            self.index = 0
        self._cap: cv2.VideoCapture | None = None

    def status(self) -> CameraStatus:
        ok = self._open()
        return CameraStatus(
            mode=self.source,
            connected=ok,
            message=f"USB camera index {self.index} ready." if ok else f"USB camera index {self.index} is not available.",
        )

    def _open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        self.close()
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    def grab(self) -> np.ndarray:
        if not self._open():
            raise RuntimeError(f"USB camera index {self.index} is not available.")
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Camera grab failed. No frame.")
        return frame

    def grab_fresh(self, discard: int = 2) -> np.ndarray:
        if not self._open():
            raise RuntimeError(f"USB camera index {self.index} is not available.")
        assert self._cap is not None
        for _ in range(max(0, int(discard))):
            self._cap.read()
        return self.grab()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
