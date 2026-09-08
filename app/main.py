from __future__ import annotations

import base64
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import DATA_DIR, HOST, PORT, ROOT, THREAD_COUNT, ensure_dirs, lan_addresses
from app.core.camera import ImageSource
from app.core.engine import inspect_many
from app.storage.store import store
from app.tools.registry import catalog

ensure_dirs()

app = FastAPI(title="SVision", version="1.0.0")
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


def _jpeg_url(raw: bytes) -> str:
    if not raw:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _public_result(item: dict) -> dict:
    item["preview"] = _jpeg_url(item.pop("preview_jpeg", b""))
    for tool in item.get("tools") or []:
        tool["preview"] = _jpeg_url(tool.pop("preview_jpeg", b""))
    return item


@app.get("/api/health")
def health() -> dict:
    ips = lan_addresses()
    return {
        "status": "ok",
        "host": HOST,
        "port": PORT,
        "urls": [f"http://127.0.0.1:{PORT}"] + [f"http://{ip}:{PORT}" for ip in ips],
        "threads": THREAD_COUNT,
        "data_dir": str(DATA_DIR),
        "camera": ImageSource().status().__dict__,
    }


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


@app.post("/api/projects/{project_id}/inspect")
async def inspect(project_id: str, files: list[UploadFile] = File(...)) -> dict:
    payload: list[tuple[str, bytes]] = []
    for item in files:
        raw = await item.read()
        if raw:
            payload.append((item.filename or "image.png", raw))
    try:
        result = inspect_many(project_id, payload)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["results"] = [_public_result(item) for item in result["results"]]
    return result


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def main() -> None:
    import uvicorn

    ips = lan_addresses()
    print("SVision inspection server")
    print(f"  Local:   http://127.0.0.1:{PORT}")
    for ip in ips:
        print(f"  Network: http://{ip}:{PORT}   (open this on another PC)")
    print(f"  Threads: {THREAD_COUNT}")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
