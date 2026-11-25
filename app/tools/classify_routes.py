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
from fastapi.responses import Response

from app.storage.store import store


def mount(app: Any) -> None:
    @app.post("/api/projects/{project_id}/tools/{tool_id}/samples")
    async def upload_samples(
        project_id: str,
        tool_id: str,
        label: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> dict:
        payload: list[tuple[str, bytes]] = []
        for item in files:
            raw = await item.read()
            if not raw:
                continue
            payload.append((item.filename or "image.png", raw))
        if not payload:
            raise HTTPException(400, "No image files uploaded.")
        try:
            added = store.add_samples(project_id, tool_id, label, payload)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"added": added}

    @app.delete("/api/projects/{project_id}/tools/{tool_id}/samples/{sample_id}")
    def delete_sample(project_id: str, tool_id: str, sample_id: str) -> dict:
        try:
            store.delete_sample(project_id, tool_id, sample_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.get("/api/projects/{project_id}/tools/{tool_id}/samples/{sample_id}")
    def get_sample(project_id: str, tool_id: str, sample_id: str, thumb: int = 0) -> Response:
        try:
            path = store.sample_path(project_id, tool_id, sample_id, thumb=bool(thumb))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
        if path.suffix.lower() == ".png":
            media = "image/png"
        elif path.suffix.lower() == ".bmp":
            media = "image/bmp"
        from fastapi.responses import FileResponse

        return FileResponse(path, media_type=media)

    @app.post("/api/projects/{project_id}/tools/{tool_id}/train")
    def train_project(project_id: str, tool_id: str) -> dict:
        try:
            return store.train(project_id, tool_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
