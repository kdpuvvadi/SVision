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

import base64
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import DATA_DIR, HOST, PORT, ROOT, THREAD_COUNT, app_version, ensure_dirs, lan_addresses
from app.core.camera import ImageSource, camera_from_project
from app.core.handshake import handshake
from app.core.plc_runtime import plc_runtime
from app.storage.store import current_images, default_plc_settings, load_settings, save_settings, store
from app.tools.registry import catalog

ensure_dirs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    plc_runtime.start()
    try:
        yield
    finally:
        plc_runtime.stop()


app = FastAPI(title="SVision", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = ROOT / "static"


class ProjectCreate(BaseModel):
    name: str = "Scene 0"


class ProjectUpdate(BaseModel):
    name: str | None = None
    flow: dict[str, Any] | None = None
    camera: dict[str, Any] | None = None


class ToolCreate(BaseModel):
    type: str


class PlcModbusSettings(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 1502
    unit_id: int = 1


class PlcOpcuaSettings(BaseModel):
    enabled: bool = False
    endpoint: str = "opc.tcp://0.0.0.0:4840/svision/"


class PlcSettings(BaseModel):
    production_project_id: str = ""
    done_pulse_ms: int = 80
    modbus: PlcModbusSettings = Field(default_factory=PlcModbusSettings)
    opcua: PlcOpcuaSettings = Field(default_factory=PlcOpcuaSettings)


class SettingsUpdate(BaseModel):
    layout: str | None = None
    flow_open: bool | None = None
    plc: PlcSettings | None = None


def _jpeg_url(raw: bytes) -> str:
    if not raw:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _public_result(item: dict) -> dict:
    item["preview"] = _jpeg_url(item.pop("preview_jpeg", b""))
    for tool in item.get("tools") or []:
        tool["preview"] = _jpeg_url(tool.pop("preview_jpeg", b""))
    return item


def _public_batch(result: dict) -> dict:
    if "results" in result:
        result = dict(result)
        result["results"] = [_public_result(dict(item)) for item in result.get("results") or []]
        return result
    return _public_result(result)


@app.get("/api/health")
def health() -> dict:
    ips = lan_addresses()
    cam = ImageSource().status().__dict__
    try:
        settings = load_settings()
        pid = (settings.get("plc") or {}).get("production_project_id") or ""
        if pid:
            cam = camera_from_project(store.get_project(pid)).status().__dict__
    except Exception:
        pass
    return {
        "status": "ok",
        "version": app_version(),
        "host": HOST,
        "port": PORT,
        "urls": [f"http://127.0.0.1:{PORT}"] + [f"http://{ip}:{PORT}" for ip in ips],
        "threads": THREAD_COUNT,
        "data_dir": str(DATA_DIR),
        "camera": cam,
        "plc": plc_runtime.status(),
    }


@app.get("/api/settings")
def get_settings() -> dict:
    return load_settings()


@app.put("/api/settings")
def put_settings(body: SettingsUpdate) -> dict:
    plc = body.plc.model_dump() if body.plc is not None else None
    settings = save_settings(body.layout, body.flow_open, plc)
    if plc is not None:
        plc_runtime.apply_settings(settings.get("plc") or default_plc_settings())
    return settings


@app.get("/api/plc/status")
def plc_status() -> dict:
    return plc_runtime.status()


@app.post("/api/plc/trigger")
def plc_trigger(project_id: str | None = None) -> dict:
    pid = (project_id or handshake.production_project_id() or "").strip()
    if not pid:
        raise HTTPException(400, "No production scene selected.")
    job = handshake.request_measure(pid, source="auto")
    if not job.done_event.wait(timeout=120):
        raise HTTPException(504, "Measure timed out.")
    if job.error:
        raise HTTPException(400, job.error)
    result = job.result or {}
    public = _public_batch(result)
    if "results" in public:
        current_images.save_result(pid, public)
    return public


@app.get("/api/tools")
def list_tools() -> list[dict]:
    return catalog()


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return store.list_projects()


@app.post("/api/projects")
def create_project(body: ProjectCreate) -> dict:
    return store.create_project(body.name)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, body: ProjectUpdate) -> dict:
    try:
        return store.update_project(project_id, body.name, body.flow, body.camera)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    try:
        store.delete_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@app.post("/api/projects/{project_id}/tools")
def add_flow_tool(project_id: str, body: ToolCreate) -> dict:
    try:
        return store.add_flow_tool(project_id, body.type)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
    return FileResponse(path, media_type=media)


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


@app.post("/api/projects/{project_id}/tools/{tool_id}/train")
def train_project(project_id: str, tool_id: str) -> dict:
    try:
        return store.train(project_id, tool_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/projects/{project_id}/current-image")
def get_current_image(project_id: str) -> dict:
    try:
        store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    files = current_images.files(project_id)
    return {
        "files": [
            {
                "filename": item["filename"],
                "url": f"/api/projects/{project_id}/current-image/{item['index']}",
            }
            for item in files
        ],
        "result": current_images.result(project_id),
    }


@app.get("/api/projects/{project_id}/current-image/{index}")
def get_current_image_file(project_id: str, index: int) -> FileResponse:
    try:
        path = current_images.path(project_id, index)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)


@app.post("/api/projects/{project_id}/current-image")
async def put_current_image(project_id: str, files: list[UploadFile] = File(...)) -> dict:
    try:
        store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    payload: list[tuple[str, bytes]] = []
    for item in files:
        raw = await item.read()
        if raw:
            payload.append((item.filename or "image.png", raw))
    if not payload:
        raise HTTPException(400, "No image selected.")
    try:
        current_images.save(project_id, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await _current_payload(project_id)


async def _current_payload(project_id: str) -> dict:
    files = current_images.files(project_id)
    return {
        "files": [
            {
                "filename": item["filename"],
                "url": f"/api/projects/{project_id}/current-image/{item['index']}",
            }
            for item in files
        ],
        "result": None,
    }


@app.post("/api/projects/{project_id}/inspect")
async def inspect(project_id: str, files: list[UploadFile] | None = File(default=None)) -> dict:
    payload: list[tuple[str, bytes]] = []
    for item in files or []:
        raw = await item.read()
        if raw:
            payload.append((item.filename or "image.png", raw))
    try:
        store.get_project(project_id)
        if payload:
            current_images.save(project_id, payload)
        job = handshake.request_measure(project_id, source="auto")
        if not job.done_event.wait(timeout=120):
            raise ValueError("Measure timed out.")
        if job.error:
            raise ValueError(job.error)
        result = job.result or {}
        if "results" not in result:
            raise ValueError("Measure returned no result.")
        result = _public_batch(result)
        current_images.save_result(project_id, result)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/plc-docs")
def docs_page() -> FileResponse:
    return FileResponse(STATIC / "docs.html")


def main() -> None:
    import logging
    import sys
    import threading
    import webbrowser

    import uvicorn

    class _QuietPlcStatus(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return "/api/plc/status" not in msg

    logging.getLogger("uvicorn.access").addFilter(_QuietPlcStatus())

    ips = lan_addresses()
    version = app_version()
    label = f"SVision {version}".strip() if version else "SVision"
    print(f"{label}, Copyright (C) 2026 KD Puvvadi")
    print("SVision comes with ABSOLUTELY NO WARRANTY.")
    print("This is free software; see the file LICENSE for details.")
    print(f"  Local:   http://127.0.0.1:{PORT}")
    for ip in ips:
        print(f"  Network: http://{ip}:{PORT}   (open this on another PC)")
    print(f"  Threads: {THREAD_COUNT}")
    print(f"  Data:    {DATA_DIR}")
    print("Close this window to stop SVision.")
    if getattr(sys, "frozen", False):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
        uvicorn.run(app, host=HOST, port=PORT, reload=False, workers=1)
        return
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
