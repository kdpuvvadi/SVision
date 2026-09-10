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

import threading
import time

from app.core.camera import camera_from_project
from app.core.engine import decode_image, inspect_array, inspect_many
from app.core.handshake import MeasureJob, handshake
from app.storage.store import current_images, store


def _jpeg_url(raw: bytes) -> str:
    if not raw:
        return ""
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _public_batch(result: dict) -> dict:
    result = dict(result)
    rows = []
    for item in result.get("results") or []:
        item = dict(item)
        item["preview"] = _jpeg_url(item.pop("preview_jpeg", b"") or b"")
        tools = []
        for tool in item.get("tools") or []:
            tool = dict(tool)
            tool["preview"] = _jpeg_url(tool.pop("preview_jpeg", b"") or b"")
            tools.append(tool)
        item["tools"] = tools
        rows.append(item)
    result["results"] = rows
    return result


class MeasureService:
    """Dedicated orchestrator thread. Comm adapters only enqueue jobs."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="svision-measure", daemon=True)
        self._thread.start()
        print("[Measure] worker thread started", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = handshake.trigger_queue.try_get(timeout=0.2)
            if job is None:
                self._refresh_ready()
                continue
            self._run_job(job)

    def _refresh_ready(self) -> None:
        pid = handshake.production_project_id()
        ready = False
        if pid:
            try:
                project = store.get_project(pid)
                cam = camera_from_project(project)
                if cam.source == "upload":
                    ready = bool(current_images.files(pid))
                else:
                    ready = cam.status().connected
            except Exception:
                ready = False
        handshake.set_ready(ready and not handshake.is_busy())

    def _run_job(self, job: MeasureJob) -> None:
        started = time.perf_counter()
        print(f"[Measure] start scene={job.project_id} source={job.source} file={job.filename}", flush=True)
        handshake.begin_cycle()
        handshake.set_ready(False)
        try:
            stored = current_images.files(job.project_id)
            print(f"[Measure] stored images={len(stored)}", flush=True)
            project = store.get_project(job.project_id)
            result = self._inspect(project, job)
            elapsed = (time.perf_counter() - started) * 1000.0
            # Prefer first/overall batch result for handshake bits
            overall = result
            if "results" in result and result["results"]:
                overall = result["results"][0]
                overall_judgment = result.get("judgment") or overall.get("judgment")
            else:
                overall_judgment = result.get("judgment")
            job.result = result
            try:
                current_images.save_result(job.project_id, _public_batch(result))
            except Exception:
                pass
            if overall_judgment == "OK":
                handshake.finish_ok(elapsed, result)
            else:
                handshake.finish_ng(elapsed, result, result.get("message") or "NG")
            print(
                f"[Measure] done judgment={overall_judgment} elapsed_ms={elapsed:.1f} "
                f"message={result.get('message') or ''}",
                flush=True,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            job.error = str(exc)
            handshake.finish_error(str(exc), elapsed)
            print(f"[Measure] ERROR after {elapsed:.1f} ms: {exc}", flush=True)
        finally:
            job.done_event.set()
            self._refresh_ready()

    def _inspect(self, project: dict, job: MeasureJob) -> dict:
        cam = camera_from_project(project)
        source = job.source
        stored = current_images.read(job.project_id)
        if source == "auto":
            # Prefer the selected/stored image until a live camera is actually connected.
            if stored:
                source = "stored"
            elif cam.source in {"usb", "gige"} and cam.status().connected:
                source = "live"
            else:
                source = "stored"

        if job.image_bytes:
            image = decode_image(job.image_bytes)
            item = inspect_array(job.project_id, image, job.filename or "image")
            return {
                "count": 1,
                "ok": 1 if item.get("judgment") == "OK" else 0,
                "ng": 0 if item.get("judgment") == "OK" else 1,
                "elapsed_ms": item.get("elapsed_ms") or 0,
                "results": [item],
                "judgment": item.get("judgment"),
                "message": item.get("message") or "",
            }

        if source == "live":
            frame = cam.grab()
            item = inspect_array(job.project_id, frame, f"live-{cam.source}")
            return {
                "count": 1,
                "ok": 1 if item.get("judgment") == "OK" else 0,
                "ng": 0 if item.get("judgment") == "OK" else 1,
                "elapsed_ms": item.get("elapsed_ms") or 0,
                "results": [item],
                "judgment": item.get("judgment"),
                "message": item.get("message") or "",
            }

        if not stored:
            raise ValueError("No stored image. Choose an image in SVision, then trigger again.")
        batch = inspect_many(job.project_id, stored)
        batch["judgment"] = "OK" if batch.get("ng", 1) == 0 else "NG"
        batch["message"] = "All tools passed." if batch["judgment"] == "OK" else "One or more tools failed."
        return batch


measure_service = MeasureService()
