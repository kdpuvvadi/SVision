# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

"""WebSocket ↔ TCP result-socket bridge for the Socket Test web UI."""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.core.logutil import log
from app.core.socket_server import result_socket
from app.storage.store import load_settings


def _target_from_settings(host: str | None = None, port: int | None = None) -> tuple[str, int, dict[str, Any]]:
    plc = (load_settings().get("plc") or {})
    sock = plc.get("socket") if isinstance(plc.get("socket"), dict) else {}
    status = result_socket.status()
    # Prefer the live listener when running; fall back to saved settings.
    bind_host = str(status.get("host") or sock.get("host") or "0.0.0.0")
    if port is not None:
        bind_port = int(port)
    elif status.get("running") and status.get("port"):
        bind_port = int(status.get("port") or 9100)
    else:
        bind_port = int(sock.get("port") or 9100)
    # Clients never connect to 0.0.0.0 — use loopback.
    connect_host = (host or "").strip() or ("127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host)
    info = {
        "enabled": bool(sock.get("enabled") or status.get("running")),
        "running": bool(status.get("running")),
        "bind": f"{bind_host}:{bind_port}",
        "connect": f"{connect_host}:{bind_port}",
        "host": connect_host,
        "port": bind_port,
        "error": status.get("error") or "",
        "clients": status.get("clients") or 0,
    }
    return connect_host, bind_port, info


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def socket_test_session(ws: WebSocket, host: str | None = None, port: int | None = None) -> None:
    await ws.accept()
    connect_host, connect_port, info = _target_from_settings(host, port)
    log(f"[SocketTest] browser connected → target {connect_host}:{connect_port} running={info['running']}")
    await _send(ws, {"type": "status", "connected": False, "server": info, "message": "Browser linked. Click Connect to open the TCP socket."})
    await _send(
        ws,
        {
            "type": "log",
            "level": "info",
            "message": f"Server status: enabled={info['enabled']} running={info['running']} bind={info['bind']}",
        },
    )
    if not info["running"]:
        await _send(
            ws,
            {
                "type": "log",
                "level": "warn",
                "message": "TCP Socket server is not running. Open Settings → enable TCP Socket → Save, then Connect.",
            },
        )

    tcp: socket.socket | None = None
    reader_task: asyncio.Task | None = None
    stop = asyncio.Event()

    async def close_tcp() -> None:
        nonlocal tcp, reader_task
        stop.set()
        if reader_task and not reader_task.done():
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
        reader_task = None
        if tcp is not None:
            try:
                tcp.close()
            except OSError:
                pass
            tcp = None
        stop.clear()

    async def read_tcp(sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        buf = b""
        try:
            while not stop.is_set():
                try:
                    chunk = await loop.run_in_executor(None, sock.recv, 65536)
                except OSError as exc:
                    await _send(ws, {"type": "log", "level": "error", "message": f"TCP read error: {exc}"})
                    break
                if not chunk:
                    await _send(ws, {"type": "log", "level": "warn", "message": "TCP server closed the connection"})
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        await _send(ws, {"type": "log", "level": "warn", "message": f"Non-JSON line: {line[:120]}"})
                        continue
                    log(f"[SocketTest] result judgment={payload.get('judgment')} scene={(payload.get('scene') or {}).get('name')}")
                    await _send(ws, {"type": "result", "payload": payload})
                    await _send(
                        ws,
                        {
                            "type": "log",
                            "level": "ok",
                            "message": f"Result {payload.get('judgment')} · {payload.get('elapsed_ms')} ms · {(payload.get('scene') or {}).get('name') or ''}",
                        },
                    )
        finally:
            await _send(ws, {"type": "status", "connected": False, "server": _target_from_settings(host, port)[2], "message": "Disconnected from TCP socket"})
            log("[SocketTest] TCP session ended")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, {"type": "log", "level": "warn", "message": f"Bad browser message: {raw[:80]}"})
                continue
            action = str(msg.get("action") or msg.get("cmd") or "").lower()
            if action == "status":
                _, _, info = _target_from_settings(host, port)
                await _send(ws, {"type": "status", "connected": tcp is not None, "server": info, "message": "Status refreshed"})
            elif action == "connect":
                await close_tcp()
                connect_host, connect_port, info = _target_from_settings(msg.get("host"), msg.get("port"))
                if msg.get("host"):
                    connect_host = str(msg.get("host")).strip() or connect_host
                if msg.get("port"):
                    try:
                        connect_port = int(msg.get("port"))
                    except (TypeError, ValueError):
                        pass
                await _send(ws, {"type": "log", "level": "info", "message": f"Connecting TCP {connect_host}:{connect_port}…"})
                log(f"[SocketTest] connect TCP {connect_host}:{connect_port}")
                loop = asyncio.get_running_loop()
                try:
                    sock = await loop.run_in_executor(
                        None,
                        lambda: socket.create_connection((connect_host, connect_port), timeout=3.0),
                    )
                    sock.settimeout(0.5)
                except OSError as exc:
                    await _send(
                        ws,
                        {
                            "type": "status",
                            "connected": False,
                            "server": info,
                            "message": f"Connect failed: {exc}",
                        },
                    )
                    await _send(ws, {"type": "log", "level": "error", "message": f"Connect failed: {exc}"})
                    log(f"[SocketTest] connect FAILED {exc}")
                    continue
                tcp = sock
                stop.clear()
                reader_task = asyncio.create_task(read_tcp(sock))
                await _send(
                    ws,
                    {
                        "type": "status",
                        "connected": True,
                        "server": {**info, "host": connect_host, "port": connect_port, "connect": f"{connect_host}:{connect_port}"},
                        "message": f"Connected {connect_host}:{connect_port}",
                    },
                )
                await _send(ws, {"type": "log", "level": "ok", "message": f"Connected to TCP {connect_host}:{connect_port}"})
                log(f"[SocketTest] connected TCP {connect_host}:{connect_port}")
            elif action == "disconnect":
                await close_tcp()
                await _send(ws, {"type": "log", "level": "info", "message": "Disconnected by user"})
                log("[SocketTest] disconnect requested")
            elif action == "trigger":
                if tcp is None:
                    await _send(ws, {"type": "log", "level": "warn", "message": "Not connected — Connect first, then Trigger"})
                    continue
                try:
                    tcp.sendall(b'{"cmd":"trigger"}\n')
                    await _send(ws, {"type": "log", "level": "info", "message": 'Sent {"cmd":"trigger"}'})
                    log("[SocketTest] trigger sent")
                except OSError as exc:
                    await _send(ws, {"type": "log", "level": "error", "message": f"Trigger failed: {exc}"})
                    log(f"[SocketTest] trigger FAILED {exc}")
            else:
                await _send(ws, {"type": "log", "level": "warn", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        log("[SocketTest] browser disconnected")
    finally:
        await close_tcp()
