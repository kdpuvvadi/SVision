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

from app.core.cameras.gentl import discover_cti_files, list_gentl_devices


def mount(app: Any) -> None:
    @app.get("/api/cameras/cti")
    def list_camera_cti() -> dict:
        files = discover_cti_files()
        return {"files": files, "count": len(files)}

    @app.get("/api/cameras/devices")
    def list_camera_devices(cti_path: str = "") -> dict:
        try:
            devices = list_gentl_devices(cti_path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"devices": devices, "count": len(devices), "cti_path": cti_path}
