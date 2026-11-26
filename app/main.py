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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import DATA_DIR, HOST, MAX_UPLOAD_MB, PORT, ROOT, THREAD_COUNT, app_version, ensure_dirs, lan_addresses
from app.core.camera import ImageSource, camera_from_project, camera_session
from app.core.handshake import handshake
from app.core.plc_runtime import plc_runtime
from app.core.robot_stream import robot_stream
from app.storage.store import current_images, default_plc_settings, load_settings, save_settings, store
from app.tools.registry import catalog, mount_all, notify_project_deleted, notify_project_updated

ensure_dirs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio

    robot_stream.set_main_loop(asyncio.get_running_loop())
    plc_runtime.start()
    try:
        yield
    finally:
        robot_stream.stop_all()
        plc_runtime.stop()
        camera_session.close()


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


class PlcSocketSettings(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 9100


class PlcMqttSettings(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 1883
    topic: str = "svision/results"


class PlcSettings(BaseModel):
    production_project_id: str = ""
    done_pulse_ms: int = 80
    modbus: PlcModbusSettings = Field(default_factory=PlcModbusSettings)
    opcua: PlcOpcuaSettings = Field(default_factory=PlcOpcuaSettings)
    socket: PlcSocketSettings = Field(default_factory=PlcSocketSettings)
    mqtt: PlcMqttSettings = Field(default_factory=PlcMqttSettings)


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


@app.get("/api/accel/devices")
def accel_devices() -> dict:
    from app.core.accel import list_accel_devices

    return {"devices": list_accel_devices()}


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
        "robot_ws": f"ws://127.0.0.1:{PORT}/ws/robot/{{scene_id}}/{{tool_id}}",
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
        updated = store.update_project(project_id, body.name, body.flow, body.camera)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    notify_project_updated(updated)
    return updated


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    try:
        project = store.get_project(project_id)
        tools = list((project.get("flow") or {}).get("tools") or [])
        notify_project_deleted(project_id, tools)
        store.delete_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: str) -> Response:
    try:
        payload, filename = store.export_project_archive(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type="application/octet-stream", headers=headers)


@app.post("/api/projects/import")
async def import_project(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"Archive larger than {MAX_UPLOAD_MB} MB.")
    filename = (file.filename or "").lower()
    if filename and not (filename.endswith(".svision") or filename.endswith(".zip")):
        raise HTTPException(400, "Choose a .svision scene archive.")
    try:
        return store.import_project_archive(raw, name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc


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


@app.post("/api/projects/{project_id}/tools/{tool_id}/pipeline-image")
async def pipeline_image(project_id: str, tool_id: str, fresh: int = 1) -> Response:
    """Grab/load a frame, run every tool before ``tool_id``, return that working image."""
    try:
        project = store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    tool_ids = {t.get("id") for t in (project.get("flow") or {}).get("tools", [])}
    if tool_id not in tool_ids:
        raise HTTPException(404, "Tool not in this scene.")

    from app.core.camera import camera_config_from_project
    from app.core.classifier import encode_jpeg
    from app.core.engine import decode_image, pipeline_image_until

    source, _cfg = camera_config_from_project(project)
    frame = None
    if source in {"usb", "gige", "gentl"}:
        try:
            cam = camera_from_project(project)
            backend = getattr(cam, "_backend", None)
            if fresh and backend is not None and hasattr(backend, "grab_fresh"):
                frame = backend.grab_fresh()
            else:
                frame = cam.grab()
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            current_images.save(project_id, [("live-camera.jpg", encode_jpeg(frame, 90))])
        except Exception:
            pass
    else:
        stored = current_images.read(project_id)
        if not stored:
            raise HTTPException(400, "No image yet. Choose a file or set Camera to live.")
        name, raw = stored[0]
        try:
            frame = decode_image(raw)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        working = pipeline_image_until(project_id, frame, tool_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    jpeg = encode_jpeg(working, 90)
    height, width = working.shape[:2]
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Image-Width": str(width),
            "X-Image-Height": str(height),
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


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


mount_all(app)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/plc-docs")
def docs_page() -> FileResponse:
    return FileResponse(STATIC / "docs.html")


@app.get("/settings-docs")
def settings_docs_page() -> FileResponse:
    return FileResponse(STATIC / "docs.html")


@app.get("/socket-test")
def socket_test_page() -> FileResponse:
    return FileResponse(STATIC / "socket_test.html")


@app.websocket("/ws/socket-test")
async def socket_test_ws(websocket: WebSocket, host: str | None = None, port: int | None = None) -> None:
    from app.core.socket_test_bridge import socket_test_session

    await socket_test_session(websocket, host=host, port=port)


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
    print(f"  Socket:  http://127.0.0.1:{PORT}/socket-test")
    print(f"  Threads: {THREAD_COUNT}")
    print(f"  Data:    {DATA_DIR}")
    print("Close this window to stop SVision.")
    if getattr(sys, "frozen", False):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
        threading.Timer(1.4, lambda: webbrowser.open_new(f"http://127.0.0.1:{PORT}/socket-test")).start()
        uvicorn.run(app, host=HOST, port=PORT, reload=False, workers=1)
        return
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
