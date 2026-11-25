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

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.core.cameras.gentl import discover_cti_files, list_gentl_devices, test_gentl_camera


class CameraTestBody(BaseModel):
    cti_path: str = ""
    device_id: str = ""
    gige_ip: str = ""
    source: str = "gentl"
    usb_index: str = "0"


def mount(app: Any) -> None:
    @app.get("/api/cameras/cti")
    def list_camera_cti() -> dict:
        files = discover_cti_files()
        return {"files": files, "count": len(files)}

    @app.get("/api/cameras/devices")
    def list_camera_devices(cti_path: str = "", probe_ip: int = 1) -> dict:
        try:
            devices = list_gentl_devices(cti_path, probe_ip=bool(probe_ip))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"devices": devices, "count": len(devices), "cti_path": cti_path}

    @app.post("/api/cameras/test")
    def test_camera(body: CameraTestBody) -> dict:
        cfg = body.model_dump()
        result = test_gentl_camera(cfg)
        if not result.get("ok"):
            raise HTTPException(400, result.get("message") or "Camera test failed.")
        return result
