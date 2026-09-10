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
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any


JUDGMENT_IDLE = 0
JUDGMENT_OK = 1
JUDGMENT_NG = 2
JUDGMENT_ERROR = 3


@dataclass
class HandshakeSnapshot:
    trigger: bool = False
    ready: bool = False
    busy: bool = False
    done: bool = False
    ok: bool = False
    ng: bool = False
    error: bool = False
    elapsed_ms: int = 0
    judgment: int = JUDGMENT_IDLE
    last_trigger_at: float = 0.0
    last_error: str = ""


@dataclass
class MeasureJob:
    project_id: str
    source: str = "auto"  # auto | stored | live | upload
    filename: str = "image"
    image_bytes: bytes | None = None
    done_event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str = ""


class TriggerQueue:
    """Depth-1 queue: one pending measure while busy drops new triggers."""

    def __init__(self) -> None:
        self._queue: Queue[MeasureJob] = Queue(maxsize=1)
        self._lock = threading.Lock()

    def offer(self, job: MeasureJob) -> bool:
        with self._lock:
            try:
                self._queue.put_nowait(job)
                return True
            except Full:
                return False

    def get(self, timeout: float | None = None) -> MeasureJob:
        return self._queue.get(timeout=timeout)

    def try_get(self, timeout: float = 0.2) -> MeasureJob | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break


class HandshakeState:
    """Thread-safe PLC handshake bits shared by Modbus, OPC UA, and MeasureService."""

    def __init__(self, trigger_queue: TriggerQueue | None = None) -> None:
        self._lock = threading.RLock()
        self._trigger = False
        self._prev_trigger = False
        self._ready = False
        self._busy = False
        self._done = False
        self._ok = False
        self._ng = False
        self._error = False
        self._elapsed_ms = 0
        self._judgment = JUDGMENT_IDLE
        self._last_trigger_at = 0.0
        self._last_error = ""
        self._done_pulse_ms = 80
        self._production_project_id = ""
        self._last_result: dict | None = None
        self._measure_seq = 0
        self.trigger_queue = trigger_queue or TriggerQueue()
        self._done_timer: threading.Timer | None = None

    def configure(self, *, done_pulse_ms: int = 80, production_project_id: str = "") -> None:
        with self._lock:
            self._done_pulse_ms = max(10, int(done_pulse_ms))
            self._production_project_id = (production_project_id or "").strip()

    def production_project_id(self) -> str:
        with self._lock:
            return self._production_project_id

    def done_pulse_ms(self) -> int:
        with self._lock:
            return self._done_pulse_ms

    def set_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = bool(ready)

    def set_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = bool(busy)

    def set_trigger(self, value: bool, *, project_id: str | None = None) -> bool:
        """Set TRIGGER level. Rising edge enqueues a measure when not busy. Returns True if enqueued."""
        pulse_error: str | None = None
        with self._lock:
            level = bool(value)
            rising = level and not self._prev_trigger
            if level:
                print(
                    f"[PLC] TRIGGER write=1 rising={rising} busy={self._busy} "
                    f"scene={self._production_project_id or '-'}",
                    flush=True,
                )
            self._trigger = level
            self._prev_trigger = level
            if not level:
                return False
            if not rising:
                print("[PLC] TRIGGER ignored (need rising edge: write 0 then 1, or wait for clear)", flush=True)
                return False
            if self._busy:
                print("[PLC] TRIGGER ignored (BUSY)", flush=True)
                return False
            pid = (project_id or self._production_project_id or "").strip()
            if not pid:
                try:
                    from app.storage.store import load_settings

                    pid = str((load_settings().get("plc") or {}).get("production_project_id") or "").strip()
                    self._production_project_id = pid
                except Exception:
                    pid = ""
            if not pid:
                self._trigger = False
                self._prev_trigger = False
                pulse_error = "No production scene selected. Open PLC settings and Save."
                print(f"[PLC] TRIGGER failed: {pulse_error}", flush=True)
            else:
                self._last_trigger_at = time.time()
                self._error = False
                self._last_error = ""
                job = MeasureJob(project_id=pid, source="stored")
                if not self.trigger_queue.offer(job):
                    self._trigger = False
                    self._prev_trigger = False
                    pulse_error = "Measure queue full (busy)."
                    print(f"[PLC] TRIGGER failed: {pulse_error}", flush=True)
                else:
                    print(f"[PLC] TRIGGER accepted → measure queued scene={pid} source=stored", flush=True)
                    self._trigger = False
                    self._prev_trigger = False
                    return True
        if pulse_error:
            self.finish_error(pulse_error, 0.0)
        return False

    def request_measure(
        self,
        project_id: str,
        *,
        source: str = "auto",
        filename: str = "image",
        image_bytes: bytes | None = None,
    ) -> MeasureJob:
        print(f"[PLC] UI/API measure request scene={project_id} source={source}", flush=True)
        job = MeasureJob(
            project_id=project_id,
            source=source,
            filename=filename,
            image_bytes=image_bytes,
        )
        if self.is_busy() or not self.trigger_queue.offer(job):
            job.error = "Measure is busy."
            print("[PLC] UI/API measure rejected (busy)", flush=True)
            job.done_event.set()
            return job
        with self._lock:
            self._last_trigger_at = time.time()
        return job

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def begin_cycle(self) -> None:
        with self._lock:
            if self._done_timer is not None:
                self._done_timer.cancel()
                self._done_timer = None
            self._busy = True
            self._done = False
            self._ok = False
            self._ng = False
            self._error = False
            self._judgment = JUDGMENT_IDLE
            self._last_error = ""

    def finish_ok(self, elapsed_ms: float, result: dict | None = None) -> None:
        with self._lock:
            self._ok = True
            self._ng = False
            self._error = False
            self._judgment = JUDGMENT_OK
            self._elapsed_ms = int(round(elapsed_ms))
            self._last_result = result
            self._measure_seq += 1
            self._busy = False
        self._pulse_outputs()

    def finish_ng(self, elapsed_ms: float, result: dict | None = None, message: str = "") -> None:
        with self._lock:
            self._ok = False
            self._ng = True
            self._error = False
            self._judgment = JUDGMENT_NG
            self._elapsed_ms = int(round(elapsed_ms))
            self._last_result = result
            self._measure_seq += 1
            if message:
                self._last_error = message
            self._busy = False
        self._pulse_outputs()

    def finish_error(self, message: str, elapsed_ms: float = 0.0) -> None:
        with self._lock:
            self._ok = False
            self._ng = False
            self._error = True
            self._judgment = JUDGMENT_ERROR
            self._elapsed_ms = int(round(elapsed_ms))
            self._last_error = message or "Measure failed."
            self._measure_seq += 1
            self._busy = False
        self._pulse_outputs()

    def _pulse_outputs(self) -> None:
        """Raise DONE + OK/NG/ERROR for done_pulse_ms, then clear those coils.

        Holding registers (elapsed_ms / judgment) stay until the next cycle so the PLC
        can still read them after the coil pulse ends.
        """
        with self._lock:
            if self._done_timer is not None:
                self._done_timer.cancel()
            self._done = True
            ms = self._done_pulse_ms

        def _clear() -> None:
            with self._lock:
                self._done = False
                self._ok = False
                self._ng = False
                self._error = False
                self._done_timer = None

        timer = threading.Timer(ms / 1000.0, _clear)
        timer.daemon = True
        with self._lock:
            self._done_timer = timer
        timer.start()

    def last_result(self) -> dict | None:
        with self._lock:
            return self._last_result

    def snapshot(self) -> HandshakeSnapshot:
        with self._lock:
            return HandshakeSnapshot(
                trigger=self._trigger,
                ready=self._ready,
                busy=self._busy,
                done=self._done,
                ok=self._ok,
                ng=self._ng,
                error=self._error,
                elapsed_ms=self._elapsed_ms,
                judgment=self._judgment,
                last_trigger_at=self._last_trigger_at,
                last_error=self._last_error,
            )

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trigger": self._trigger,
                "ready": self._ready,
                "busy": self._busy,
                "done": self._done,
                "ok": self._ok,
                "ng": self._ng,
                "error": self._error,
                "elapsed_ms": self._elapsed_ms,
                "judgment": self._judgment,
                "last_trigger_at": self._last_trigger_at,
                "last_error": self._last_error,
                "measure_seq": self._measure_seq,
                "production_project_id": self._production_project_id,
                "done_pulse_ms": self._done_pulse_ms,
            }


handshake = HandshakeState()
