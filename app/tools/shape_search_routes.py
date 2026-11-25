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

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.storage.store import store


def mount(app: Any) -> None:
    @app.post("/api/projects/{project_id}/tools/{tool_id}/shape-model")
    async def register_shape_model(
        project_id: str,
        tool_id: str,
        file: UploadFile = File(...),
        x: float = Form(0),
        y: float = Form(0),
        w: float = Form(1),
        h: float = Form(1),
    ) -> dict:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Choose an image, then draw the pattern.")
        try:
            return store.register_shape_model(project_id, tool_id, raw, {"x": x, "y": y, "w": w, "h": h})
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/projects/{project_id}/tools/{tool_id}/shape-model")
    def get_shape_model(project_id: str, tool_id: str) -> FileResponse:
        try:
            path = store.shape_model_path(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/shape-model")
    def delete_shape_model(project_id: str, tool_id: str) -> dict:
        try:
            store.delete_shape_model(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}
