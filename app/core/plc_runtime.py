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
from app.core.mqtt_broker import mqtt_bridge
from app.core.opcua_server import opcua_adapter
from app.core.socket_server import result_socket
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
        result_socket.stop()
        mqtt_bridge.stop()
        measure_service.stop()
        self._started = False

    def apply_settings(self, plc: dict[str, Any] | None) -> dict[str, Any]:
        defaults = default_plc_settings()
        cfg = {**defaults, **(plc or {})}
        modbus = {**defaults["modbus"], **(cfg.get("modbus") or {})}
        opcua = {**defaults["opcua"], **(cfg.get("opcua") or {})}
        sock = {**defaults["socket"], **(cfg.get("socket") or {})}
        mqtt = {**defaults["mqtt"], **(cfg.get("mqtt") or {})}
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
        if sock.get("enabled"):
            result_socket.start(
                host=str(sock.get("host") or "0.0.0.0"),
                port=int(sock.get("port") or 9100),
            )
        else:
            result_socket.stop()
        if mqtt.get("enabled"):
            mqtt_bridge.start(
                host=str(mqtt.get("host") or "0.0.0.0"),
                port=int(mqtt.get("port") or 1883),
                topic=str(mqtt.get("topic") or "svision/results"),
            )
        else:
            mqtt_bridge.stop()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "handshake": handshake.as_dict(),
            "modbus": modbus_adapter.status(),
            "opcua": opcua_adapter.status(),
            "socket": result_socket.status(),
            "mqtt": mqtt_bridge.status(),
            "measure_running": True,
        }

    def publish_result(self, project_id: str, result: dict[str, Any] | None, error: str | None = None) -> None:
        from app.core.result_payload import build_result_payload

        payload = build_result_payload(project_id, result, error=error)
        if result_socket.status().get("running"):
            result_socket.publish(payload)
        if mqtt_bridge.status().get("running"):
            mqtt_bridge.publish(payload)


plc_runtime = PlcRuntime()
