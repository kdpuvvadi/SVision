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
import logging
import threading
from typing import Any

from app.core.handshake import handshake

log = logging.getLogger("svision.opcua")


class OpcUaAdapter:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._status: dict[str, Any] = {"running": False, "error": "", "endpoint": ""}

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def start(self, endpoint: str = "opc.tcp://0.0.0.0:4840/svision/") -> None:
        self.stop()
        self._stop.clear()
        self._status = {"running": False, "error": "", "endpoint": endpoint}
        self._thread = threading.Thread(
            target=self._thread_main,
            args=(endpoint,),
            name="svision-opcua",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._status["running"] = False

    def _thread_main(self, endpoint: str) -> None:
        try:
            from asyncua import Server
        except ImportError:
            self._status["error"] = "asyncua is missing. Run: pip install asyncua"
            log.error(self._status["error"])
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _serve() -> None:
            server = Server()
            await server.init()
            server.set_endpoint(endpoint)
            server.set_server_name("SVision")
            uri = "urn:svision:handshake"
            idx = await server.register_namespace(uri)
            objects = server.get_objects_node()
            root = await objects.add_object(idx, "SVision")
            hs = await root.add_object(idx, "Handshake")

            trigger = await hs.add_variable(idx, "Trigger", False)
            await trigger.set_writable()
            ready = await hs.add_variable(idx, "Ready", False)
            busy = await hs.add_variable(idx, "Busy", False)
            done = await hs.add_variable(idx, "Done", False)
            ok = await hs.add_variable(idx, "Ok", False)
            ng = await hs.add_variable(idx, "Ng", False)
            error = await hs.add_variable(idx, "Error", False)
            elapsed = await hs.add_variable(idx, "ElapsedMs", 0)
            judgment = await hs.add_variable(idx, "Judgment", 0)

            self._status["running"] = True
            self._status["error"] = ""
            async with server:
                last_trigger = False
                while not self._stop.is_set():
                    snap = handshake.snapshot()
                    await ready.write_value(snap.ready)
                    await busy.write_value(snap.busy)
                    await done.write_value(snap.done)
                    await ok.write_value(snap.ok)
                    await ng.write_value(snap.ng)
                    await error.write_value(snap.error)
                    await elapsed.write_value(int(snap.elapsed_ms))
                    await judgment.write_value(int(snap.judgment))
                    try:
                        value = bool(await trigger.get_value())
                    except Exception:
                        value = False
                    if value and not last_trigger:
                        print("[OPC UA] Trigger=true received", flush=True)
                        handshake.set_trigger(True)
                        try:
                            await trigger.write_value(False)
                        except Exception:
                            pass
                        last_trigger = False
                    else:
                        last_trigger = value
                    await asyncio.sleep(0.05)

        try:
            loop.create_task(_serve())
            while not self._stop.is_set():
                loop.run_forever()
                break
        except Exception as exc:
            self._status["error"] = str(exc)
            log.exception("OPC UA server failed")
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()
            self._status["running"] = False
            self._loop = None


opcua_adapter = OpcUaAdapter()
