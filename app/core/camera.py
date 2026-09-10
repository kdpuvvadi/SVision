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

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraStatus:
    mode: str
    connected: bool
    message: str


class ImageSource:
    """Upload, USB OpenCV device, or GigE via OpenCV when a driver is available."""

    def __init__(self, source: str = "upload", gige_ip: str = "", usb_index: int = 0) -> None:
        self.source = (source or "upload").strip().lower()
        self.gige_ip = (gige_ip or "").strip()
        self.usb_index = int(usb_index or 0)
        self._cap: cv2.VideoCapture | None = None

    def status(self) -> CameraStatus:
        if self.source == "upload":
            return CameraStatus(
                mode="upload",
                connected=False,
                message="Upload images to inspect, or switch Camera to USB / GigE for live grab.",
            )
        if self.source == "usb":
            ok = self._open_usb()
            return CameraStatus(
                mode="usb",
                connected=ok,
                message=f"USB camera index {self.usb_index} ready." if ok else f"USB camera index {self.usb_index} is not available.",
            )
        if self.source == "gige":
            ok = self._open_gige()
            ip = self.gige_ip or "no IP"
            return CameraStatus(
                mode="gige",
                connected=ok,
                message=f"GigE camera {ip} ready." if ok else f"GigE camera {ip} is not connected.",
            )
        return CameraStatus(mode=self.source, connected=False, message=f"Unknown camera source {self.source}.")

    def _open_usb(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        self.close()
        cap = cv2.VideoCapture(self.usb_index)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    def _open_gige(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        self.close()
        if not self.gige_ip:
            return False
        candidates = [
            self.gige_ip,
            f"http://{self.gige_ip}/",
            f"rtsp://{self.gige_ip}/",
        ]
        for target in candidates:
            cap = cv2.VideoCapture(target)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    return True
            cap.release()
        return False

    def grab(self) -> np.ndarray:
        if self.source == "upload":
            raise RuntimeError("Camera is set to Upload. Choose or drop an image, or switch to USB / GigE.")
        if self.source == "usb":
            if not self._open_usb():
                raise RuntimeError(f"USB camera index {self.usb_index} is not available.")
        elif self.source == "gige":
            if not self._open_gige():
                raise RuntimeError(f"GigE camera {self.gige_ip or 'no IP'} is not connected.")
        else:
            raise RuntimeError(f"Unknown camera source {self.source}.")
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Camera grab failed. No frame.")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def camera_from_project(project: dict) -> ImageSource:
    camera_tool = None
    for tool in (project.get("flow") or {}).get("tools", []):
        if tool.get("type") == "camera":
            camera_tool = tool
            break
    cfg = (camera_tool or {}).get("config") or {}
    source = cfg.get("source") or "upload"
    try:
        usb_index = int(cfg.get("usb_index") or 0)
    except (TypeError, ValueError):
        usb_index = 0
    return ImageSource(source=source, gige_ip=cfg.get("gige_ip") or "", usb_index=usb_index)
