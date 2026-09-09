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


@dataclass
class CameraStatus:
    mode: str
    connected: bool
    message: str


class ImageSource:
    """Current source is uploaded files. GigE/GenICam is reserved for later."""

    def __init__(self, gige_ip: str = "") -> None:
        self.gige_ip = gige_ip

    def status(self) -> CameraStatus:
        if self.gige_ip:
            return CameraStatus(
                mode="upload",
                connected=False,
                message=f"GigE camera {self.gige_ip} is saved but not connected. Inspection uses uploaded images.",
            )
        return CameraStatus(
            mode="upload",
            connected=False,
            message="No camera connected. Upload images to inspect. GenICam/GigE can be added later.",
        )

    def grab(self):
        raise RuntimeError("Camera grab is not available. Upload an image to inspect.")
