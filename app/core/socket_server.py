# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any

from app.core.handshake import handshake
from app.config import lan_addresses

log = logging.getLogger("svision.socket")


class ResultSocketServer:
    """TCP server: push newline-delimited JSON results; accept trigger commands."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._last_payload: dict[str, Any] | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "error": "",
            "host": "",
            "port": 0,
            "clients": 0,
            "connect": [],
        }

    def status(self) -> dict[str, Any]:
        with self._clients_lock:
            clients = len(self._clients)
        out = dict(self._status)
        out["clients"] = clients
        return out

    def start(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        self.stop()
        self._stop.clear()
        host = str(host or "0.0.0.0")
        port = int(port or 9100)
        connect = [f"127.0.0.1:{port}"] + [f"{ip}:{port}" for ip in lan_addresses()]
        self._status = {
            "running": False,
            "error": "",
            "host": host,
            "port": port,
            "clients": 0,
            "connect": connect,
            "protocol": "tcp-json-lines",
            "hint": 'Connect with a TCP client. Each measure pushes one JSON object + newline. Send {"cmd":"trigger"} or TRIGGER to start a measure.',
        }
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(16)
            sock.settimeout(0.5)
            self._sock = sock
        except OSError as exc:
            self._status["error"] = str(exc)
            log.error("socket bind failed: %s", exc)
            return
        self._thread = threading.Thread(target=self._serve, name="svision-socket", daemon=True)
        self._thread.start()
        self._status["running"] = True
        log.info("result socket listening on %s:%s", host, port)

    def stop(self) -> None:
        self._stop.set()
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._status["running"] = False
        self._status["clients"] = 0

    def publish(self, payload: dict[str, Any]) -> None:
        self._last_payload = payload
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        dead: list[socket.socket] = []
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sendall(line)
            except OSError:
                dead.append(client)
        if dead:
            with self._clients_lock:
                for client in dead:
                    if client in self._clients:
                        self._clients.remove(client)
                    try:
                        client.close()
                    except OSError:
                        pass

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(0.5)
            with self._clients_lock:
                self._clients.append(client)
            log.info("socket client connected %s", addr)
            threading.Thread(
                target=self._client_loop,
                args=(client, addr),
                name=f"svision-sock-{addr[1]}",
                daemon=True,
            ).start()
            if self._last_payload is not None:
                try:
                    client.sendall((json.dumps(self._last_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                except OSError:
                    pass

    def _client_loop(self, client: socket.socket, addr: tuple[Any, ...]) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self._handle_line(raw.decode("utf-8", errors="replace").strip())
        finally:
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                client.close()
            except OSError:
                pass
            log.info("socket client disconnected %s", addr)

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        cmd = ""
        try:
            if line.upper() == "TRIGGER":
                cmd = "trigger"
            else:
                data = json.loads(line)
                cmd = str(data.get("cmd") or data.get("command") or "").lower()
        except json.JSONDecodeError:
            if "trigger" in line.lower():
                cmd = "trigger"
        if cmd == "trigger":
            try:
                ok = handshake.set_trigger(True)
                log.info("socket trigger %s", "accepted" if ok else "ignored")
            except Exception as exc:
                log.warning("socket trigger failed: %s", exc)


result_socket = ResultSocketServer()
