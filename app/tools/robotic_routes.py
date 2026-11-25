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

from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.robot_stream import robot_stream
from app.storage.store import store


class RobotClassBody(BaseModel):
    name: str = "Part"
    color: str = "#1ad4c0"
    class_id: str | None = None


class RobotCalibBody(BaseModel):
    points: list[dict[str, float]] = Field(default_factory=list)


class RobotConvertBody(BaseModel):
    px: float
    py: float


def mount(app: Any) -> None:
    @app.get("/api/projects/{project_id}/tools/{tool_id}/robot")
    def robot_status(project_id: str, tool_id: str) -> dict:
        try:
            return store.robot_state(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/projects/{project_id}/tools/{tool_id}/robot/classes")
    def robot_add_class(project_id: str, tool_id: str, body: RobotClassBody) -> dict:
        try:
            return store.robot_ensure_class(project_id, tool_id, body.name, body.color, body.class_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/robot/classes/{class_id}")
    def robot_delete_class(project_id: str, tool_id: str, class_id: str) -> dict:
        try:
            store.robot_delete_class(project_id, tool_id, class_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/projects/{project_id}/tools/{tool_id}/robot/register-images")
    async def robot_add_register_image(project_id: str, tool_id: str, file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Empty image.")
        try:
            return store.robot_add_register_image(project_id, tool_id, file.filename or "image.png", raw)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/projects/{project_id}/tools/{tool_id}/robot/register-images")
    def robot_list_register_images(project_id: str, tool_id: str) -> dict:
        try:
            items = store.robot_list_register_images(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "images": [
                {
                    **item,
                    "url": f"/api/projects/{project_id}/tools/{tool_id}/robot/register-images/{item['id']}",
                    "thumb": f"/api/projects/{project_id}/tools/{tool_id}/robot/register-images/{item['id']}?thumb=1",
                }
                for item in items
            ]
        }

    @app.get("/api/projects/{project_id}/tools/{tool_id}/robot/register-images/{image_id}")
    def robot_get_register_image(project_id: str, tool_id: str, image_id: str, thumb: int = 0) -> FileResponse:
        try:
            path = store.robot_register_image_path(project_id, tool_id, image_id, thumb=bool(thumb))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path)

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/robot/register-images/{image_id}")
    def robot_delete_register_image(project_id: str, tool_id: str, image_id: str) -> dict:
        try:
            store.robot_delete_register_image(project_id, tool_id, image_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/projects/{project_id}/tools/{tool_id}/robot/mark")
    async def robot_mark(
        project_id: str,
        tool_id: str,
        class_id: str = Form(...),
        image_id: str | None = Form(default=None),
        cx: float = Form(0.5),
        cy: float = Form(0.5),
        w: float = Form(0.2),
        h: float = Form(0.2),
        angle: float = Form(0),
        file: UploadFile | None = File(default=None),
    ) -> dict:
        raw = await file.read() if file is not None else None
        try:
            return store.robot_mark_instance(
                project_id,
                tool_id,
                class_id=class_id,
                image_id=image_id,
                image_bytes=raw or None,
                box={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/robot/classes/{class_id}/instances/{instance_id}")
    def robot_delete_instance(project_id: str, tool_id: str, class_id: str, instance_id: str) -> dict:
        try:
            store.robot_delete_instance(project_id, tool_id, class_id, instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.get("/api/projects/{project_id}/tools/{tool_id}/robot/calib")
    def robot_get_calib(project_id: str, tool_id: str) -> dict:
        try:
            return store.robot_calib(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.put("/api/projects/{project_id}/tools/{tool_id}/robot/calib")
    def robot_put_calib(project_id: str, tool_id: str, body: RobotCalibBody) -> dict:
        try:
            return store.robot_save_calib(project_id, tool_id, body.points)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/robot/calib")
    def robot_delete_calib(project_id: str, tool_id: str) -> dict:
        try:
            store.robot_clear_calib(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/projects/{project_id}/tools/{tool_id}/robot/convert")
    def robot_convert(project_id: str, tool_id: str, body: RobotConvertBody) -> dict:
        try:
            return store.robot_convert_point(project_id, tool_id, body.px, body.py)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.websocket("/ws/robot/{project_id}/{tool_id}")
    async def robot_ws(websocket: WebSocket, project_id: str, tool_id: str):
        await robot_stream.connect(project_id, tool_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            robot_stream.disconnect(project_id, tool_id, websocket)
        except Exception:
            robot_stream.disconnect(project_id, tool_id, websocket)
