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

from app.core.cameras.base import CameraBackend, CameraStatus
from app.core.cameras.registry import register_backend


@register_backend
class UploadBackend(CameraBackend):
    source = "upload"

    def status(self) -> CameraStatus:
        return CameraStatus(
            mode=self.source,
            connected=False,
            message="Upload images to inspect, or switch Camera to USB / GigE GenTL for live grab.",
        )

    def grab(self) -> np.ndarray:
        raise RuntimeError("Camera is set to Upload. Choose or drop an image, or switch to USB / GigE GenTL.")
