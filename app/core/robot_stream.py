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

import asyncio
import json
import threading
import time
from typing import Any

from fastapi import WebSocket


class RobotStreamHub:
    """Fan-out WebSocket hub for robotic coordinate streams."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clients: dict[str, set[WebSocket]] = {}
        self._loops: dict[str, threading.Thread] = {}
        self._stop: dict[str, threading.Event] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def key(project_id: str, tool_id: str) -> str:
        return f"{project_id}:{tool_id}"

    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    async def connect(self, project_id: str, tool_id: str, ws: WebSocket) -> None:
        await ws.accept()
        k = self.key(project_id, tool_id)
        with self._lock:
            self._clients.setdefault(k, set()).add(ws)

    def disconnect(self, project_id: str, tool_id: str, ws: WebSocket) -> None:
        k = self.key(project_id, tool_id)
        with self._lock:
            clients = self._clients.get(k)
            if not clients:
                return
            clients.discard(ws)
            if not clients:
                self._clients.pop(k, None)

    async def _safe_send(self, ws: WebSocket, text: str, project_id: str, tool_id: str) -> None:
        try:
            await ws.send_text(text)
        except Exception:
            self.disconnect(project_id, tool_id, ws)

    def broadcast_text(self, project_id: str, tool_id: str, payload: dict[str, Any]) -> None:
        k = self.key(project_id, tool_id)
        with self._lock:
            clients = list(self._clients.get(k) or [])
        if not clients:
            return
        text = json.dumps(payload, default=str)
        loop = self._main_loop
        if loop is None or not loop.is_running():
            return
        for ws in clients:
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, text, project_id, tool_id), loop)

    def publish_from_measure(self, project_id: str, result: dict) -> None:
        try:
            from app.storage.store import store

            project = store.get_project(project_id)
            cfg_by_id = {
                t.get("id"): (t.get("config") or {})
                for t in (project.get("flow") or {}).get("tools") or []
                if t.get("type") == "robotic"
            }
        except Exception:
            cfg_by_id = {}
        for item in result.get("results") or [result]:
            for tool in item.get("tools") or []:
                if tool.get("type") != "robotic":
                    continue
                tid = tool.get("id") or ""
                if not tid:
                    continue
                mode = str((cfg_by_id.get(tid) or {}).get("stream_mode") or "on_trigger")
                if mode != "on_trigger":
                    continue
                self.broadcast_text(
                    project_id,
                    tid,
                    {
                        "ts": time.time(),
                        "scene": project_id,
                        "tool": tid,
                        "count": tool.get("count", 0),
                        "objects": tool.get("objects") or [],
                        "stream_mode": "on_trigger",
                        "angle_enable": tool.get("angle_enable", True),
                        "model_name": tool.get("model_name") or "",
                        "judgment": tool.get("judgment"),
                    },
                )

    def start_continuous(self, project_id: str, tool_id: str, hz: float = 5.0) -> None:
        k = self.key(project_id, tool_id)
        with self._lock:
            existing = self._loops.get(k)
            if existing and existing.is_alive():
                return
            stop = threading.Event()
            self._stop[k] = stop
            thread = threading.Thread(
                target=self._continuous_loop,
                args=(project_id, tool_id, max(1.0, float(hz)), stop),
                daemon=True,
                name=f"robot-stream-{k}",
            )
            self._loops[k] = thread
            thread.start()

    def stop_continuous(self, project_id: str, tool_id: str) -> None:
        k = self.key(project_id, tool_id)
        with self._lock:
            stop = self._stop.pop(k, None)
            self._loops.pop(k, None)
        if stop:
            stop.set()

    def stop_all(self) -> None:
        with self._lock:
            keys = list(self._stop.keys())
        for k in keys:
            pid, tid = k.split(":", 1)
            self.stop_continuous(pid, tid)

    def sync_continuous_from_project(self, project: dict) -> None:
        pid = project.get("id") or ""
        wanted: set[str] = set()
        for tool in (project.get("flow") or {}).get("tools") or []:
            if tool.get("type") != "robotic":
                continue
            cfg = tool.get("config") or {}
            if str(cfg.get("stream_mode") or "on_trigger") != "continuous":
                continue
            tid = tool.get("id") or ""
            wanted.add(self.key(pid, tid))
            self.start_continuous(pid, tid, float(cfg.get("stream_hz") or 5))
        with self._lock:
            running = list(self._loops.keys())
        for k in running:
            if k not in wanted:
                p, t = k.split(":", 1)
                self.stop_continuous(p, t)

    def _continuous_loop(self, project_id: str, tool_id: str, hz: float, stop: threading.Event) -> None:
        from app.core.camera import camera_from_project
        from app.core.engine import decode_image
        from app.storage.store import current_images, store
        from app.tools.robotic import RoboticTool

        period = 1.0 / hz
        robotic = RoboticTool()
        while not stop.is_set():
            started = time.perf_counter()
            try:
                project = store.get_project(project_id)
                tool = next(
                    (t for t in (project.get("flow") or {}).get("tools") or [] if t.get("id") == tool_id),
                    None,
                )
                if not tool or tool.get("type") != "robotic":
                    break
                cfg = tool.get("config") or {}
                if str(cfg.get("stream_mode") or "") != "continuous":
                    break
                image = None
                cam = camera_from_project(project)
                if cam.source in {"usb", "gige"} and cam.status().connected:
                    try:
                        image = cam.grab()
                    except Exception:
                        image = None
                if image is None:
                    stored = current_images.read(project_id)
                    if stored:
                        image = decode_image(stored[0][1])
                if image is not None:
                    result, _ = robotic.run(image, tool, project_id)
                    self.broadcast_text(
                        project_id,
                        tool_id,
                        {
                            "ts": time.time(),
                            "scene": project_id,
                            "tool": tool_id,
                            "count": result.get("count", 0),
                            "objects": result.get("objects") or [],
                            "stream_mode": "continuous",
                            "angle_enable": result.get("angle_enable", True),
                            "model_name": result.get("model_name") or "",
                            "judgment": result.get("judgment"),
                        },
                    )
            except Exception as exc:
                print(f"[RobotStream] continuous error: {exc}", flush=True)
            elapsed = time.perf_counter() - started
            if stop.wait(max(0.05, period - elapsed)):
                break


robot_stream = RobotStreamHub()
