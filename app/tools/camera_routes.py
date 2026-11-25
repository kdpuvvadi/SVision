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
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.camera import camera_config_from_project, camera_from_project
from app.core.cameras.gentl import discover_cti_files, list_gentl_devices, test_gentl_camera
from app.core.classifier import encode_jpeg
from app.storage.store import current_images, store


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

    @app.post("/api/projects/{project_id}/camera/grab")
    def grab_project_camera(project_id: str, fresh: int = 1) -> Response:
        """Trigger a live grab for tools that need a still (Shape Search model, ROI, …)."""
        try:
            project = store.get_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        source, _cfg = camera_config_from_project(project)
        if source == "upload":
            raise HTTPException(
                400,
                "Camera source is Upload. Choose an image file, or set Camera → Source to USB / GenTL.",
            )
        try:
            cam = camera_from_project(project)
            backend = getattr(cam, "_backend", None)
            if fresh and backend is not None and hasattr(backend, "grab_fresh"):
                frame = backend.grab_fresh()
            else:
                frame = cam.grab()
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        jpeg = encode_jpeg(frame, 90)
        try:
            current_images.save(project_id, [("live-camera.jpg", jpeg)])
        except Exception:
            pass
        height, width = frame.shape[:2]
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "X-Image-Width": str(width),
                "X-Image-Height": str(height),
                "X-Camera-Source": source,
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )
