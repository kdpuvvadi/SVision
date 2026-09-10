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

from typing import Any

from app.core.handshake import handshake
from app.core.measure_service import measure_service
from app.core.modbus_server import modbus_adapter
from app.core.opcua_server import opcua_adapter
from app.storage.store import default_plc_settings, load_settings


class PlcRuntime:
    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        measure_service.start()
        self.apply_settings(load_settings().get("plc") or default_plc_settings())
        self._started = True

    def stop(self) -> None:
        modbus_adapter.stop()
        opcua_adapter.stop()
        measure_service.stop()
        self._started = False

    def apply_settings(self, plc: dict[str, Any] | None) -> dict[str, Any]:
        cfg = {**default_plc_settings(), **(plc or {})}
        modbus = {**default_plc_settings()["modbus"], **(cfg.get("modbus") or {})}
        opcua = {**default_plc_settings()["opcua"], **(cfg.get("opcua") or {})}
        handshake.configure(
            done_pulse_ms=int(cfg.get("done_pulse_ms") or 80),
            production_project_id=str(cfg.get("production_project_id") or ""),
        )
        if modbus.get("enabled"):
            modbus_adapter.start(
                host=str(modbus.get("host") or "0.0.0.0"),
                port=int(modbus.get("port") or 1502),
                unit_id=int(modbus.get("unit_id") or 1),
            )
        else:
            modbus_adapter.stop()
        if opcua.get("enabled"):
            opcua_adapter.start(endpoint=str(opcua.get("endpoint") or "opc.tcp://0.0.0.0:4840/svision/"))
        else:
            opcua_adapter.stop()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "handshake": handshake.as_dict(),
            "modbus": modbus_adapter.status(),
            "opcua": opcua_adapter.status(),
            "measure_running": True,
        }


plc_runtime = PlcRuntime()
