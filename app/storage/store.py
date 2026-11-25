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

import io
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.core.classifier import SampleModel
from app.tools.registry import get_tool, new_tool, normalize_config

_lock = threading.RLock()
SVISION_FORMAT = "svision-scene"
SVISION_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def default_flow() -> dict:
    return {"logic": "all", "tools": []}


def _is_stock_flow(flow: dict) -> bool:
    tools = flow.get("tools") or []
    types = [tool.get("type") for tool in tools]
    return types == ["image_input", "classify", "ocr", "code"]


def _project_dir(project_id: str) -> Path:
    return DATA_DIR / "projects" / project_id


def _tool_dir(project_id: str, tool_id: str) -> Path:
    return _project_dir(project_id) / "tools" / tool_id


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _settings_path() -> Path:
    return DATA_DIR / "settings.json"


def default_plc_settings() -> dict:
    return {
        "production_project_id": "",
        "done_pulse_ms": 80,
        "modbus": {
            "enabled": False,
            "host": "0.0.0.0",
            "port": 1502,
            "unit_id": 1,
        },
        "opcua": {
            "enabled": False,
            "endpoint": "opc.tcp://0.0.0.0:4840/svision/",
        },
    }


def default_settings() -> dict:
    return {"layout": "inspect", "flow_open": False, "plc": default_plc_settings()}


def load_settings() -> dict:
    path = _settings_path()
    settings = default_settings()
    if path.exists():
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError):
            raw = {}
        layout = raw.get("layout")
        if layout in ("inspect", "edit"):
            settings["layout"] = layout
        settings["flow_open"] = bool(raw.get("flow_open")) and settings["layout"] == "edit"
        plc = raw.get("plc") if isinstance(raw.get("plc"), dict) else {}
        merged = default_plc_settings()
        merged["production_project_id"] = str(plc.get("production_project_id") or "")
        try:
            merged["done_pulse_ms"] = max(10, int(plc.get("done_pulse_ms") or 80))
        except (TypeError, ValueError):
            merged["done_pulse_ms"] = 80
        modbus = plc.get("modbus") if isinstance(plc.get("modbus"), dict) else {}
        opcua = plc.get("opcua") if isinstance(plc.get("opcua"), dict) else {}
        merged["modbus"].update({k: modbus[k] for k in merged["modbus"] if k in modbus})
        merged["opcua"].update({k: opcua[k] for k in merged["opcua"] if k in opcua})
        merged["modbus"]["enabled"] = bool(merged["modbus"].get("enabled"))
        merged["opcua"]["enabled"] = bool(merged["opcua"].get("enabled"))
        try:
            merged["modbus"]["port"] = int(merged["modbus"].get("port") or 1502)
        except (TypeError, ValueError):
            merged["modbus"]["port"] = 1502
        try:
            merged["modbus"]["unit_id"] = int(merged["modbus"].get("unit_id") or 1)
        except (TypeError, ValueError):
            merged["modbus"]["unit_id"] = 1
        settings["plc"] = merged
    return settings


def save_settings(
    layout: str | None = None,
    flow_open: bool | None = None,
    plc: dict | None = None,
) -> dict:
    current = load_settings()
    if layout is not None:
        current["layout"] = "edit" if layout == "edit" else "inspect"
    if flow_open is not None:
        current["flow_open"] = bool(flow_open) and current["layout"] == "edit"
    if plc is not None:
        merged = default_plc_settings()
        merged["production_project_id"] = str(plc.get("production_project_id") or "")
        try:
            merged["done_pulse_ms"] = max(10, int(plc.get("done_pulse_ms") or 80))
        except (TypeError, ValueError):
            merged["done_pulse_ms"] = 80
        modbus = plc.get("modbus") if isinstance(plc.get("modbus"), dict) else {}
        opcua = plc.get("opcua") if isinstance(plc.get("opcua"), dict) else {}
        merged["modbus"].update({k: modbus[k] for k in ("enabled", "host", "port", "unit_id") if k in modbus})
        merged["opcua"].update({k: opcua[k] for k in ("enabled", "endpoint") if k in opcua})
        merged["modbus"]["enabled"] = bool(merged["modbus"].get("enabled"))
        merged["opcua"]["enabled"] = bool(merged["opcua"].get("enabled"))
        try:
            merged["modbus"]["port"] = int(merged["modbus"].get("port") or 1502)
        except (TypeError, ValueError):
            merged["modbus"]["port"] = 1502
        try:
            merged["modbus"]["unit_id"] = int(merged["modbus"].get("unit_id") or 1)
        except (TypeError, ValueError):
            merged["modbus"]["unit_id"] = 1
        merged["modbus"]["host"] = str(merged["modbus"].get("host") or "0.0.0.0")
        merged["opcua"]["endpoint"] = str(merged["opcua"].get("endpoint") or "opc.tcp://0.0.0.0:4840/svision/")
        current["plc"] = merged
    _write_json(_settings_path(), current)
    return current


def _empty_state() -> dict:
    return {
        "samples": [],
        "model": {
            "trained_at": "",
            "ok_count": 0,
            "ng_count": 0,
            "stale": True,
            "train_accuracy": None,
        },
    }


class ProjectStore:
    def __init__(self) -> None:
        self._models: dict[str, SampleModel] = {}

    def list_projects(self) -> list[dict]:
        root = DATA_DIR / "projects"
        items: list[dict] = []
        if not root.exists():
            return items
        for folder in sorted(root.iterdir()):
            meta = folder / "project.json"
            if meta.exists():
                data = _read_json(meta)
                items.append(self._summary(data))
        items.sort(key=lambda p: p.get("updated", ""), reverse=True)
        return items

    def create_project(self, name: str) -> dict:
        project_id = uuid.uuid4().hex[:12]
        now = _now()
        data = {
            "id": project_id,
            "name": (name or "Scene 0").strip() or "Scene 0",
            "created": now,
            "updated": now,
            "flow": default_flow(),
            "camera": {"type": "upload", "gige_ip": ""},
        }
        folder = _project_dir(project_id)
        (folder / "tools").mkdir(parents=True, exist_ok=True)
        _write_json(folder / "project.json", data)
        return self._public(data)

    def get_project(self, project_id: str) -> dict:
        path = _project_dir(project_id) / "project.json"
        if not path.exists():
            raise FileNotFoundError("Project not found.")
        data = _read_json(path)
        changed = False
        if _is_stock_flow(data.get("flow") or {}):
            data["flow"] = default_flow()
            changed = True
        data["flow"] = data.get("flow") or default_flow()
        tools = []
        for tool in data["flow"].get("tools", []):
            try:
                tools.append(normalize_config(tool))
            except KeyError:
                continue
        if tools != data["flow"].get("tools"):
            data["flow"]["tools"] = tools
            changed = True
        if changed:
            data["updated"] = _now()
            _write_json(path, data)
        return self._public(data)

    def update_project(self, project_id: str, name: str | None = None, flow: dict | None = None, camera: dict | None = None) -> dict:
        with _lock:
            data = self._raw(project_id)
            if name is not None:
                data["name"] = name.strip() or data["name"]
            if flow is not None:
                data["flow"] = self._apply_flow(project_id, data, flow)
            if camera is not None:
                data["camera"] = camera
            data["updated"] = _now()
            _write_json(_project_dir(project_id) / "project.json", data)
            return self._public(data)

    def add_flow_tool(self, project_id: str, tool_type: str) -> dict:
        with _lock:
            data = self._raw(project_id)
            spec = get_tool(tool_type)
            tools = data.setdefault("flow", default_flow()).setdefault("tools", [])
            if getattr(spec, "singleton", False) and any(item.get("type") == tool_type for item in tools):
                raise ValueError(f"{spec.title} is already in the flow.")
            tool = new_tool(tool_type)
            if tool_type == "camera":
                tools.insert(0, tool)
            else:
                tools.append(tool)
            self._ensure_tool(project_id, tool["id"])
            data["updated"] = _now()
            _write_json(_project_dir(project_id) / "project.json", data)
            public = self._public(data)
            return next(item for item in public["flow"]["tools"] if item["id"] == tool["id"])

    def delete_project(self, project_id: str) -> None:
        with _lock:
            folder = _project_dir(project_id)
            if not folder.exists():
                raise FileNotFoundError("Project not found.")
            self._drop_project_models(project_id)
            shutil.rmtree(folder)

    def export_project_archive(self, project_id: str) -> tuple[bytes, str]:
        """Pack the scene folder into a .svision zip. Returns (bytes, download_name)."""
        folder = _project_dir(project_id)
        meta_path = folder / "project.json"
        if not meta_path.exists():
            raise FileNotFoundError("Project not found.")
        data = _read_json(meta_path)
        name = str(data.get("name") or "scene").strip() or "scene"
        safe = _SAFE_NAME.sub("_", name).strip("._") or "scene"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "svision.json",
                json.dumps(
                    {
                        "format": SVISION_FORMAT,
                        "version": SVISION_VERSION,
                        "name": name,
                        "exported_id": project_id,
                        "exported_at": _now(),
                    },
                    indent=2,
                ),
            )
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(folder).as_posix()
                if rel in {"svision.json"}:
                    continue
                zf.write(path, rel)
        return buf.getvalue(), f"{safe}.svision"

    def import_project_archive(self, raw: bytes, name: str | None = None) -> dict:
        """Import a .svision (zip) archive as a new scene with a fresh id."""
        if not raw:
            raise ValueError("Empty archive.")
        with tempfile.TemporaryDirectory(prefix="svision-import-") as tmp:
            root = Path(tmp)
            try:
                with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                    self._safe_extract(zf, root)
            except zipfile.BadZipFile as exc:
                raise ValueError("Not a valid .svision archive.") from exc
            scene_root = self._find_scene_root(root)
            meta_path = scene_root / "project.json"
            if not meta_path.exists():
                raise ValueError("Archive is missing project.json.")
            data = _read_json(meta_path)
            if not isinstance(data, dict):
                raise ValueError("Invalid project.json.")
            marker = root / "svision.json"
            if not marker.exists() and (scene_root / "svision.json").exists():
                marker = scene_root / "svision.json"
            if marker.exists():
                try:
                    info = _read_json(marker)
                except Exception as exc:
                    raise ValueError("Invalid svision.json.") from exc
                fmt = info.get("format")
                if fmt and fmt != SVISION_FORMAT:
                    raise ValueError("Unsupported .svision format.")
                try:
                    ver = int(info.get("version", 1))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid archive version.") from exc
                if ver > SVISION_VERSION:
                    raise ValueError("This .svision file needs a newer SVision.")

            new_id = uuid.uuid4().hex[:12]
            now = _now()
            data["id"] = new_id
            if name and name.strip():
                data["name"] = name.strip()
            else:
                data["name"] = str(data.get("name") or "Imported scene").strip() or "Imported scene"
            data["created"] = now
            data["updated"] = now
            data["flow"] = data.get("flow") or default_flow()
            tools = []
            for tool in data["flow"].get("tools") or []:
                try:
                    tools.append(normalize_config(tool))
                except KeyError:
                    continue
            data["flow"]["tools"] = tools
            data.setdefault("camera", {"type": "upload", "gige_ip": ""})

            dest = _project_dir(new_id)
            if dest.exists():
                raise RuntimeError("Project id collision; try again.")
            with _lock:
                shutil.copytree(scene_root, dest, ignore=shutil.ignore_patterns("svision.json"))
                _write_json(dest / "project.json", data)
                (dest / "tools").mkdir(parents=True, exist_ok=True)
                for tool in tools:
                    self._ensure_tool(new_id, tool["id"])
                return self._public(data)

    @staticmethod
    def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
        dest = dest.resolve()
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Archive contains unsafe paths.")
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise ValueError("Archive contains unsafe paths.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)

    @staticmethod
    def _find_scene_root(extracted: Path) -> Path:
        direct = extracted / "project.json"
        if direct.exists():
            return extracted
        candidates = [p for p in extracted.iterdir() if p.is_dir() and (p / "project.json").exists()]
        if len(candidates) == 1:
            return candidates[0]
        nested = list(extracted.rglob("project.json"))
        if len(nested) == 1:
            return nested[0].parent
        raise ValueError("Could not find a scene in the archive.")

    def add_samples(self, project_id: str, tool_id: str, label: str, files: list[tuple[str, bytes]]) -> list[dict]:
        label = label.strip().upper()
        if label not in {"OK", "NG"}:
            raise ValueError("Label must be OK or NG.")
        self._require_tool(project_id, tool_id)
        added: list[dict] = []
        with _lock:
            state = self._load_state(project_id, tool_id)
            folder = _tool_dir(project_id, tool_id) / "samples" / label
            folder.mkdir(parents=True, exist_ok=True)
            for filename, raw in files:
                image = self._decode(raw)
                sample_id = uuid.uuid4().hex[:10]
                ext = Path(filename).suffix.lower()
                if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                    ext = ".png"
                (folder / f"{sample_id}{ext}").write_bytes(raw)
                cv2.imwrite(str(folder / f"{sample_id}_thumb.jpg"), self._thumb(image), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                record = {
                    "id": sample_id,
                    "label": label,
                    "filename": Path(filename).name,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "added": _now(),
                }
                state["samples"].append(record)
                added.append(record)
            state["model"]["stale"] = True
            self._save_state(project_id, tool_id, state)
            self._models.pop(self._model_key(project_id, tool_id), None)
        return added

    def delete_sample(self, project_id: str, tool_id: str, sample_id: str) -> None:
        with _lock:
            state = self._load_state(project_id, tool_id)
            found = next((s for s in state["samples"] if s["id"] == sample_id), None)
            if not found:
                raise FileNotFoundError("Sample not found.")
            folder = _tool_dir(project_id, tool_id) / "samples" / found["label"]
            for path in folder.glob(f"{sample_id}*"):
                path.unlink(missing_ok=True)
            state["samples"] = [s for s in state["samples"] if s["id"] != sample_id]
            state["model"]["stale"] = True
            self._save_state(project_id, tool_id, state)
            self._models.pop(self._model_key(project_id, tool_id), None)

    def sample_path(self, project_id: str, tool_id: str, sample_id: str, thumb: bool = False) -> Path:
        state = self._load_state(project_id, tool_id)
        sample = next((s for s in state["samples"] if s["id"] == sample_id), None)
        if not sample:
            raise FileNotFoundError("Sample not found.")
        folder = _tool_dir(project_id, tool_id) / "samples" / sample["label"]
        if thumb:
            path = folder / f"{sample_id}_thumb.jpg"
            if path.exists():
                return path
        matches = [p for p in folder.glob(f"{sample_id}.*") if "_thumb" not in p.name]
        if not matches:
            raise FileNotFoundError("Sample file missing.")
        return matches[0]

    def load_sample_image(self, project_id: str, tool_id: str, sample_id: str) -> np.ndarray:
        return self._decode(self.sample_path(project_id, tool_id, sample_id).read_bytes())

    def register_shape_model(self, project_id: str, tool_id: str, raw: bytes, roi: dict | None) -> dict:
        self._require_tool(project_id, tool_id)
        image = self._decode(raw)
        x0, y0, x1, y1 = self._roi_pixels(image, roi)
        crop = image[y0:y1, x0:x1]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            raise ValueError("Draw a larger region around the pattern.")
        folder = _tool_dir(project_id, tool_id)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "shape_model.png"), crop)
        state = self._load_state(project_id, tool_id)
        state["shape_model"] = {
            "saved": True,
            "width": int(crop.shape[1]),
            "height": int(crop.shape[0]),
            "registered_at": _now(),
        }
        self._save_state(project_id, tool_id, state)
        return state["shape_model"]

    def delete_shape_model(self, project_id: str, tool_id: str) -> None:
        self._require_tool(project_id, tool_id)
        path = _tool_dir(project_id, tool_id) / "shape_model.png"
        path.unlink(missing_ok=True)
        state = self._load_state(project_id, tool_id)
        state["shape_model"] = {"saved": False}
        self._save_state(project_id, tool_id, state)

    def shape_model_path(self, project_id: str, tool_id: str) -> Path:
        path = _tool_dir(project_id, tool_id) / "shape_model.png"
        if not path.exists():
            raise FileNotFoundError("Model image is not saved.")
        return path

    # --- Robotic tool: classes, instances, calib, register images ---

    def _robot_root(self, project_id: str, tool_id: str) -> Path:
        return _tool_dir(project_id, tool_id)

    def _robot_classes_dir(self, project_id: str, tool_id: str) -> Path:
        return self._robot_root(project_id, tool_id) / "classes"

    def robot_state(self, project_id: str, tool_id: str) -> dict:
        self._require_tool(project_id, tool_id)
        state = self._load_state(project_id, tool_id)
        state.setdefault("robot", {"classes": [], "pending_marks": []})
        classes = []
        root = self._robot_classes_dir(project_id, tool_id)
        if root.exists():
            for folder in sorted(root.iterdir()):
                if not folder.is_dir():
                    continue
                meta_path = folder / "meta.json"
                meta = _read_json(meta_path) if meta_path.exists() else {"id": folder.name, "name": folder.name, "color": "#1ad4c0"}
                inst_dir = folder / "instances"
                instances = []
                if inst_dir.exists():
                    for jp in sorted(inst_dir.glob("*.json")):
                        try:
                            instances.append(_read_json(jp))
                        except Exception:
                            continue
                classes.append(
                    {
                        "id": meta.get("id") or folder.name,
                        "name": meta.get("name") or folder.name,
                        "color": meta.get("color") or "#1ad4c0",
                        "instance_count": len(instances),
                        "instances": instances,
                    }
                )
        state["robot"]["classes"] = classes
        calib = self.robot_calib(project_id, tool_id)
        state["robot"]["calib"] = calib
        return state

    def robot_ensure_class(self, project_id: str, tool_id: str, name: str, color: str = "#1ad4c0", class_id: str | None = None) -> dict:
        self._require_tool(project_id, tool_id)
        cid = (class_id or uuid.uuid4().hex[:10]).strip()
        folder = self._robot_classes_dir(project_id, tool_id) / cid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "instances").mkdir(parents=True, exist_ok=True)
        meta = {"id": cid, "name": (name or "Part").strip() or "Part", "color": color or "#1ad4c0"}
        _write_json(folder / "meta.json", meta)
        return meta

    def robot_delete_class(self, project_id: str, tool_id: str, class_id: str) -> None:
        self._require_tool(project_id, tool_id)
        folder = self._robot_classes_dir(project_id, tool_id) / class_id
        if folder.exists():
            shutil.rmtree(folder)

    def robot_add_register_image(self, project_id: str, tool_id: str, filename: str, raw: bytes) -> dict:
        self._require_tool(project_id, tool_id)
        image = self._decode(raw)
        folder = self._robot_root(project_id, tool_id) / "register_images"
        folder.mkdir(parents=True, exist_ok=True)
        img_id = uuid.uuid4().hex[:10]
        ext = Path(filename).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            ext = ".png"
        path = folder / f"{img_id}{ext}"
        path.write_bytes(raw)
        thumb = folder / f"{img_id}_thumb.jpg"
        cv2.imwrite(str(thumb), self._thumb(image), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        state = self._load_state(project_id, tool_id)
        robot = state.setdefault("robot", {})
        images = robot.setdefault("register_images", [])
        record = {
            "id": img_id,
            "filename": Path(filename).name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "added": _now(),
        }
        images.append(record)
        self._save_state(project_id, tool_id, state)
        return record

    def robot_list_register_images(self, project_id: str, tool_id: str) -> list[dict]:
        state = self.robot_state(project_id, tool_id)
        return list((state.get("robot") or {}).get("register_images") or [])

    def robot_register_image_path(self, project_id: str, tool_id: str, image_id: str, thumb: bool = False) -> Path:
        folder = self._robot_root(project_id, tool_id) / "register_images"
        if thumb:
            path = folder / f"{image_id}_thumb.jpg"
            if path.exists():
                return path
        matches = [p for p in folder.glob(f"{image_id}.*") if "_thumb" not in p.name]
        if not matches:
            raise FileNotFoundError("Register image not found.")
        return matches[0]

    def robot_delete_register_image(self, project_id: str, tool_id: str, image_id: str) -> None:
        folder = self._robot_root(project_id, tool_id) / "register_images"
        for path in folder.glob(f"{image_id}*"):
            path.unlink(missing_ok=True)
        state = self._load_state(project_id, tool_id)
        images = (state.get("robot") or {}).get("register_images") or []
        state.setdefault("robot", {})["register_images"] = [i for i in images if i.get("id") != image_id]
        self._save_state(project_id, tool_id, state)

    def robot_mark_instance(
        self,
        project_id: str,
        tool_id: str,
        *,
        class_id: str,
        image_id: str | None,
        image_bytes: bytes | None,
        box: dict,
    ) -> dict:
        """Mark a rotated box on an image and save a template instance crop."""
        self._require_tool(project_id, tool_id)
        if image_bytes:
            image = self._decode(image_bytes)
        elif image_id:
            image = self._decode(self.robot_register_image_path(project_id, tool_id, image_id).read_bytes())
        else:
            # Fall back to current scene image
            files = current_images.read(project_id)
            if not files:
                raise ValueError("Add a register image or store a scene image first.")
            image = self._decode(files[0][1])
            image_id = "current"

        class_folder = self._robot_classes_dir(project_id, tool_id) / class_id
        if not class_folder.exists():
            raise FileNotFoundError("Class not found. Add a class first.")
        meta = _read_json(class_folder / "meta.json")
        crop = self._crop_rotated(image, box)
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            raise ValueError("Draw a larger region around the object.")
        inst_id = uuid.uuid4().hex[:10]
        inst_dir = class_folder / "instances"
        inst_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(inst_dir / f"{inst_id}.png"), crop)
        record = {
            "id": inst_id,
            "class_id": class_id,
            "class_name": meta.get("name") or class_id,
            "image_id": image_id or "",
            "width": int(crop.shape[1]),
            "height": int(crop.shape[0]),
            "box": box,
            "added": _now(),
        }
        _write_json(inst_dir / f"{inst_id}.json", record)
        return record

    def robot_delete_instance(self, project_id: str, tool_id: str, class_id: str, instance_id: str) -> None:
        folder = self._robot_classes_dir(project_id, tool_id) / class_id / "instances"
        for path in folder.glob(f"{instance_id}.*"):
            path.unlink(missing_ok=True)

    def robot_list_templates(self, project_id: str, tool_id: str) -> list[dict]:
        """Load all instance template images for matching."""
        out: list[dict] = []
        root = self._robot_classes_dir(project_id, tool_id)
        if not root.exists():
            return out
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            meta_path = folder / "meta.json"
            meta = _read_json(meta_path) if meta_path.exists() else {"id": folder.name, "name": folder.name}
            inst_dir = folder / "instances"
            if not inst_dir.exists():
                continue
            for png in sorted(inst_dir.glob("*.png")):
                img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                out.append(
                    {
                        "class_id": meta.get("id") or folder.name,
                        "class_name": meta.get("name") or folder.name,
                        "instance_id": png.stem,
                        "image": img,
                    }
                )
        return out

    def robot_calib(self, project_id: str, tool_id: str) -> dict:
        path = self._robot_root(project_id, tool_id) / "calib.json"
        if not path.exists():
            return {"ready": False, "points": [], "H": None}
        try:
            data = _read_json(path)
        except Exception:
            return {"ready": False, "points": [], "H": None}
        return data if isinstance(data, dict) else {"ready": False, "points": [], "H": None}

    def robot_save_calib(self, project_id: str, tool_id: str, points: list[dict]) -> dict:
        self._require_tool(project_id, tool_id)
        if len(points) < 4:
            raise ValueError("Need 4 pixel/robot point pairs.")
        src = np.array([[float(p["px"]), float(p["py"])] for p in points[:4]], dtype=np.float32)
        dst = np.array([[float(p["rx"]), float(p["ry"])] for p in points[:4]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise ValueError("Could not compute homography from those points.")
        data = {
            "ready": True,
            "points": points[:4],
            "H": H.tolist(),
            "updated": _now(),
        }
        _write_json(self._robot_root(project_id, tool_id) / "calib.json", data)
        return data

    def robot_clear_calib(self, project_id: str, tool_id: str) -> None:
        path = self._robot_root(project_id, tool_id) / "calib.json"
        path.unlink(missing_ok=True)

    def robot_homography(self, project_id: str, tool_id: str) -> np.ndarray | None:
        calib = self.robot_calib(project_id, tool_id)
        if not calib.get("ready") or not calib.get("H"):
            return None
        return np.asarray(calib["H"], dtype=np.float64)

    def robot_convert_point(self, project_id: str, tool_id: str, px: float, py: float) -> dict:
        H = self.robot_homography(project_id, tool_id)
        if H is None:
            raise ValueError("Calibration is not ready.")
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return {"px": float(px), "py": float(py), "rx": float(out[0, 0, 0]), "ry": float(out[0, 0, 1])}

    @staticmethod
    def _crop_rotated(image: np.ndarray, box: dict) -> np.ndarray:
        h, w = image.shape[:2]
        cx = float(box.get("cx", 0.5))
        cy = float(box.get("cy", 0.5))
        bw = float(box.get("w", 0.2))
        bh = float(box.get("h", 0.2))
        angle = float(box.get("angle", 0))
        # Normalized vs pixels
        if max(cx, cy, bw, bh) <= 1.5:
            cx *= w
            cy *= h
            bw *= w
            bh *= h
        bw = max(8.0, bw)
        bh = max(8.0, bh)
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x0 = int(round(cx - bw / 2))
        y0 = int(round(cy - bh / 2))
        x1 = int(round(cx + bw / 2))
        y1 = int(round(cy + bh / 2))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return rotated[y0:y1, x0:x1]

    def train(self, project_id: str, tool_id: str) -> dict:
        with _lock:
            tool = self._require_tool(project_id, tool_id)
            if tool["type"] != "classify":
                raise ValueError("Only ML Judgment tools are trained from samples.")
            state = self._load_state(project_id, tool_id)
            roi = (tool.get("config") or {}).get("roi")
            items = [(self.load_sample_image(project_id, tool_id, sample["id"]), sample["label"]) for sample in state["samples"]]
            model = SampleModel()
            stats = model.fit(items, roi)
            model.save(_tool_dir(project_id, tool_id) / "model" / "model.npz")
            self._models[self._model_key(project_id, tool_id)] = model
            state["model"] = {
                "trained_at": stats["trained_at"],
                "ok_count": stats["ok_count"],
                "ng_count": stats["ng_count"],
                "stale": False,
                "train_accuracy": stats["train_accuracy"],
                "ok_radius": stats["ok_radius"],
                "elapsed_ms": stats["elapsed_ms"],
            }
            self._save_state(project_id, tool_id, state)
            return {**stats, "stale": False}

    def get_model(self, project_id: str, tool_id: str) -> SampleModel:
        key = self._model_key(project_id, tool_id)
        with _lock:
            cached = self._models.get(key)
            state = self._load_state(project_id, tool_id)
            if cached and cached.ready and not state.get("model", {}).get("stale"):
                return cached
            model = SampleModel.load(_tool_dir(project_id, tool_id) / "model" / "model.npz")
            self._models[key] = model
            return model

    def _apply_flow(self, project_id: str, data: dict, flow: dict) -> dict:
        old = {tool["id"]: tool for tool in (data.get("flow") or {}).get("tools", [])}
        old_roi = {tool["id"]: (tool.get("config") or {}).get("roi") for tool in old.values()}
        kept: list[dict] = []
        for item in flow.get("tools") or []:
            try:
                tool = normalize_config(item)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            if tool["id"] not in old and not item.get("id"):
                tool = new_tool(tool["type"], tool["name"])
            previous = old.get(tool["id"])
            if previous and (previous.get("config") or {}).get("roi") != tool["config"].get("roi"):
                state = self._load_state(project_id, tool["id"])
                state["model"]["stale"] = True
                self._save_state(project_id, tool["id"], state)
                self._models.pop(self._model_key(project_id, tool["id"]), None)
            elif tool["id"] in old_roi and old_roi[tool["id"]] != tool["config"].get("roi"):
                state = self._load_state(project_id, tool["id"])
                state["model"]["stale"] = True
                self._save_state(project_id, tool["id"], state)
                self._models.pop(self._model_key(project_id, tool["id"]), None)
            self._ensure_tool(project_id, tool["id"])
            kept.append(tool)
        kept_ids = {tool["id"] for tool in kept}
        for tool_id in old:
            if tool_id not in kept_ids:
                self._delete_tool_data(project_id, tool_id)
        return {"logic": flow.get("logic") or "all", "tools": kept}

    def _public(self, data: dict) -> dict:
        public = json.loads(json.dumps(data))
        tools = []
        for tool in public.get("flow", {}).get("tools", []):
            try:
                item = normalize_config(tool)
            except KeyError:
                continue
            state = self._load_state(data["id"], item["id"])
            if item["type"] == "robotic":
                try:
                    state = self.robot_state(data["id"], item["id"])
                except Exception:
                    pass
            item["state"] = state
            tools.append(item)
        public.setdefault("flow", default_flow())["tools"] = tools
        return public

    def _raw(self, project_id: str) -> dict:
        path = _project_dir(project_id) / "project.json"
        if not path.exists():
            raise FileNotFoundError("Project not found.")
        return _read_json(path)

    def _require_tool(self, project_id: str, tool_id: str) -> dict:
        data = self._raw(project_id)
        tool = next((item for item in data.get("flow", {}).get("tools", []) if item.get("id") == tool_id), None)
        if not tool:
            raise FileNotFoundError("Tool not found.")
        return normalize_config(tool)

    def _ensure_tool(self, project_id: str, tool_id: str) -> None:
        folder = _tool_dir(project_id, tool_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "state.json"
        if not path.exists():
            _write_json(path, _empty_state())

    def _load_state(self, project_id: str, tool_id: str) -> dict:
        self._ensure_tool(project_id, tool_id)
        path = _tool_dir(project_id, tool_id) / "state.json"
        state = _read_json(path)
        state.setdefault("samples", [])
        state.setdefault("model", _empty_state()["model"])
        return state

    def _save_state(self, project_id: str, tool_id: str, state: dict) -> None:
        _write_json(_tool_dir(project_id, tool_id) / "state.json", state)

    def _delete_tool_data(self, project_id: str, tool_id: str) -> None:
        self._models.pop(self._model_key(project_id, tool_id), None)
        folder = _tool_dir(project_id, tool_id)
        if folder.exists():
            shutil.rmtree(folder)

    def _drop_project_models(self, project_id: str) -> None:
        prefix = project_id + ":"
        for key in list(self._models):
            if key.startswith(prefix):
                self._models.pop(key, None)

    @staticmethod
    def _model_key(project_id: str, tool_id: str) -> str:
        return f"{project_id}:{tool_id}"

    @staticmethod
    def _summary(data: dict) -> dict:
        return {
            "id": data["id"],
            "name": data["name"],
            "created": data.get("created"),
            "updated": data.get("updated"),
            "tools": len((data.get("flow") or {}).get("tools") or []),
        }

    @staticmethod
    def _roi_pixels(image: np.ndarray, roi: dict | None) -> tuple[int, int, int, int]:
        h, w = image.shape[:2]
        if not roi:
            return 0, 0, w, h
        x = float(roi.get("x", 0))
        y = float(roi.get("y", 0))
        rw = float(roi.get("w", 1))
        rh = float(roi.get("h", 1))
        if max(x, y, rw, rh) > 1.5:
            x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
        else:
            x0, y0 = int(round(x * w)), int(round(y * h))
            x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return x0, y0, x1, y1

    @staticmethod
    def _decode(raw: bytes) -> np.ndarray:
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("File is not a readable image.")
        return image

    @staticmethod
    def _thumb(image: np.ndarray, size: int = 160) -> np.ndarray:
        h, w = image.shape[:2]
        scale = size / float(max(h, w))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def _safe_name(name: str, index: int) -> str:
    stem = Path(name or "image").name
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._")
    if not stem:
        stem = "image"
    return f"{index:02d}_{stem}"[:120]


class CurrentImageStore:
    def dir(self, project_id: str) -> Path:
        return _project_dir(project_id) / "current"

    def clear(self, project_id: str) -> None:
        path = self.dir(project_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def save(self, project_id: str, files: list[tuple[str, bytes]]) -> list[dict]:
        if not files:
            raise ValueError("No image selected.")
        self.clear(project_id)
        dest = self.dir(project_id)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for index, (name, raw) in enumerate(files):
            if not raw:
                continue
            stored = _safe_name(name, index)
            (dest / stored).write_bytes(raw)
            saved.append({"name": stored, "filename": name or stored, "index": index})
        if not saved:
            raise ValueError("No image selected.")
        _write_json(dest / "manifest.json", {"files": saved})
        return saved

    def files(self, project_id: str) -> list[dict]:
        path = self.dir(project_id) / "manifest.json"
        if not path.exists():
            return []
        try:
            raw = _read_json(path).get("files") or []
        except (OSError, json.JSONDecodeError):
            return []
        found = []
        for item in raw:
            stored = Path(str(item.get("name") or "")).name
            if not stored or stored in {"manifest.json", "result.json"}:
                continue
            if (self.dir(project_id) / stored).is_file():
                found.append({"name": stored, "filename": item.get("filename") or stored, "index": item.get("index", len(found))})
        return found

    def read(self, project_id: str) -> list[tuple[str, bytes]]:
        items = []
        for item in self.files(project_id):
            raw = (self.dir(project_id) / item["name"]).read_bytes()
            items.append((item["filename"], raw))
        return items

    def path(self, project_id: str, index: int) -> Path:
        files = self.files(project_id)
        if index < 0 or index >= len(files):
            raise FileNotFoundError("Selected image is not saved.")
        return self.dir(project_id) / files[index]["name"]

    def save_result(self, project_id: str, result: dict) -> None:
        dest = self.dir(project_id)
        dest.mkdir(parents=True, exist_ok=True)
        _write_json(dest / "result.json", result)

    def result(self, project_id: str) -> dict | None:
        path = self.dir(project_id) / "result.json"
        if not path.exists():
            return None
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None


current_images = CurrentImageStore()
store = ProjectStore()
