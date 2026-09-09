from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.core.classifier import SampleModel
from app.tools.registry import get_tool, new_tool, normalize_config

_lock = threading.RLock()


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


def default_settings() -> dict:
    return {"layout": "inspect", "flow_open": False}


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
    return settings


def save_settings(layout: str, flow_open: bool = False) -> dict:
    settings = {
        "layout": "edit" if layout == "edit" else "inspect",
        "flow_open": bool(flow_open) and layout == "edit",
    }
    _write_json(_settings_path(), settings)
    return settings


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
