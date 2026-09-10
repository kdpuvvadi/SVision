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

log = logging.getLogger("svision.modbus")

# PDU / 0-based addresses (Open ModScan 1-based coil 000001 == address 0 here).
COIL_TRIGGER = 0
COIL_READY = 1
COIL_BUSY = 2
COIL_DONE = 3
COIL_OK = 4
COIL_NG = 5
COIL_ERROR = 6
HR_ELAPSED = 0
HR_JUDGMENT = 1

# Large enough for ModScan default polls (Length 50+) without Illegal Data Address.
BLOCK_SIZE = 256


class ModbusTcpAdapter:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._status: dict[str, Any] = {"running": False, "error": "", "host": "", "port": 0}

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def start(self, host: str = "0.0.0.0", port: int = 1502, unit_id: int = 1) -> None:
        self.stop()
        self._status = {
            "running": False,
            "error": "",
            "host": host,
            "port": int(port),
            "unit_id": int(unit_id),
        }
        self._thread = threading.Thread(
            target=self._thread_main,
            args=(host, int(port), int(unit_id)),
            name="svision-modbus",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                from pymodbus.server import ServerAsyncStop

                fut = asyncio.run_coroutine_threadsafe(ServerAsyncStop(), loop)
                try:
                    fut.result(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._status["running"] = False

    def _thread_main(self, host: str, port: int, unit_id: int) -> None:
        try:
            from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext

            try:
                from pymodbus.datastore import ModbusSlaveContext as DeviceContext
            except ImportError:
                from pymodbus.datastore import ModbusDeviceContext as DeviceContext
        except ImportError:
            self._status["error"] = "pymodbus is missing. Run: pip install pymodbus"
            log.error(self._status["error"])
            return

        class HandshakeBlock(ModbusSequentialDataBlock):
            def __init__(self, kind: str) -> None:
                # Address 0 + BLOCK_SIZE values so FC01 length 50 from ModScan is legal.
                super().__init__(0, [0] * BLOCK_SIZE)
                self.kind = kind

            def validate(self, address, count=1):  # noqa: N802
                address = int(address)
                count = int(count)
                return address >= 0 and count >= 1 and address + count <= BLOCK_SIZE

            def setValues(self, address, values):  # noqa: N802
                address = int(address)
                values = list(values)
                end = min(BLOCK_SIZE, address + len(values))
                if address < BLOCK_SIZE:
                    clipped = values[: max(0, end - address)]
                    if clipped:
                        super().setValues(address, clipped)
                if self.kind != "coils":
                    return
                for offset, value in enumerate(values):
                    # Support both 0-based (zero_mode) and 1-based datastore addressing.
                    raw = address + offset
                    logical = raw - 1 if address >= 1 else raw
                    if logical == COIL_TRIGGER and int(value):
                        print(
                            f"[Modbus] TRIGGER coil write raw={raw} logical={logical} value=1",
                            flush=True,
                        )
                        handshake.set_trigger(True)
                    elif logical == COIL_TRIGGER and not int(value):
                        handshake.set_trigger(False)

            def getValues(self, address, count=1):  # noqa: N802
                snap = handshake.snapshot()
                start = int(address)
                count = int(count)
                # If datastore is 1-based, reads often start at address 1.
                base = 1 if start >= 1 else 0
                if self.kind == "coils":
                    bits = [
                        snap.trigger,
                        snap.ready,
                        snap.busy,
                        snap.done,
                        snap.ok,
                        snap.ng,
                        snap.error,
                    ]
                    out = []
                    for i in range(count):
                        idx = start + i - base
                        out.append(1 if 0 <= idx < len(bits) and bits[idx] else 0)
                    return out
                if self.kind == "holding":
                    regs = [snap.elapsed_ms & 0xFFFF, snap.judgment & 0xFFFF]
                    out = []
                    for i in range(count):
                        idx = start + i - base
                        out.append(regs[idx] if 0 <= idx < len(regs) else 0)
                    return out
                return [0] * count

        device_kwargs = dict(
            di=HandshakeBlock("discrete"),
            co=HandshakeBlock("coils"),
            hr=HandshakeBlock("holding"),
            ir=HandshakeBlock("input"),
        )
        try:
            device = DeviceContext(**device_kwargs, zero_mode=True)
        except TypeError:
            device = DeviceContext(**device_kwargs)

        context = None
        for kwargs in (
            {"slaves": {unit_id: device}, "single": False},
            {"devices": {unit_id: device}, "single": False},
            {"slaves": device, "single": True},
            {"devices": device, "single": True},
        ):
            try:
                context = ModbusServerContext(**kwargs)
                break
            except TypeError:
                continue
        if context is None:
            self._status["error"] = "Unsupported pymodbus ServerContext API"
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _serve() -> None:
            try:
                from pymodbus.server import StartAsyncTcpServer
            except ImportError:
                from pymodbus.server.async_io import StartAsyncTcpServer  # type: ignore

            self._status["running"] = True
            self._status["error"] = ""
            print(f"[Modbus] listening on {host}:{port} unit={unit_id}", flush=True)
            await StartAsyncTcpServer(context=context, address=(host, port))

        try:
            loop.run_until_complete(_serve())
        except Exception as exc:
            if self._status.get("running") and "cancelled" in str(exc).lower():
                pass
            else:
                self._status["error"] = str(exc)
                log.exception("Modbus server failed")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._status["running"] = False
            self._loop = None


modbus_adapter = ModbusTcpAdapter()
