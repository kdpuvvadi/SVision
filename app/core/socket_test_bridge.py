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


def _normalize_endpoint(host: Any = None, port: Any = None) -> tuple[str, int]:
    """Sanitize host/port from the browser. Rejects corrupted values like 91080800."""
    h = str(host or "").strip() or "127.0.0.1"
    p_raw = port
    if isinstance(h, str) and ":" in h and not h.startswith("["):
        left, right = h.rsplit(":", 1)
        if right.isdigit():
            h = left.strip() or "127.0.0.1"
            if p_raw in (None, "", 0, "0"):
                p_raw = right
    try:
        # Digits only — avoids "9100" + "8080" style corruption surviving as int.
        digits = "".join(ch for ch in str(p_raw if p_raw is not None else "9100") if ch.isdigit())
        p = int(digits or "9100")
    except (TypeError, ValueError):
        p = 9100
    if p < 1 or p > 65535 or p in {91080800, 91008080}:
        p = 9100
    if h in {"0.0.0.0", "::", ""}:
        h = "127.0.0.1"
    return h, p


def _target_from_settings(host: str | None = None, port: int | None = None) -> tuple[str, int, dict[str, Any]]:
    plc = (load_settings().get("plc") or {})
    sock = plc.get("socket") if isinstance(plc.get("socket"), dict) else {}
    status = result_socket.status()
    # Prefer the live listener when running; fall back to saved settings.
    bind_host = str(status.get("host") or sock.get("host") or "0.0.0.0")
    if port is not None:
        _, bind_port = _normalize_endpoint("127.0.0.1", port)
    elif status.get("running") and status.get("port"):
        bind_port = int(status.get("port") or 9100)
    else:
        bind_port = int(sock.get("port") or 9100)
    if bind_port < 1 or bind_port > 65535:
        bind_port = 9100
    connect_host, bind_port = _normalize_endpoint(
        (host or "").strip() or ("127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host),
        bind_port,
    )
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
                except socket.timeout:
                    # Idle is normal — results only arrive after a measure/trigger.
                    continue
                except OSError as exc:
                    if stop.is_set():
                        break
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
            if not stop.is_set():
                await _send(
                    ws,
                    {
                        "type": "status",
                        "connected": False,
                        "server": _target_from_settings(host, port)[2],
                        "message": "Disconnected from TCP socket",
                    },
                )
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
                connect_host, connect_port = _normalize_endpoint(msg.get("host"), msg.get("port"))
                _, _, info = _target_from_settings(connect_host, connect_port)
                if not info.get("running"):
                    # Start the TCP result server so Connect works without a manual Settings trip.
                    from app.core.plc_runtime import plc_runtime
                    from app.storage.store import save_settings

                    await _send(ws, {"type": "log", "level": "warn", "message": "TCP server off — starting it…"})
                    log(f"[SocketTest] auto-start TCP {connect_host}:{connect_port}")
                    plc = load_settings().get("plc") or {}
                    plc = dict(plc)
                    plc["socket"] = {
                        **(plc.get("socket") if isinstance(plc.get("socket"), dict) else {}),
                        "enabled": True,
                        "host": "0.0.0.0",
                        "port": connect_port,
                    }
                    save_settings(plc=plc)
                    plc_runtime.apply_settings(plc)
                    await asyncio.sleep(0.35)
                    _, _, info = _target_from_settings(connect_host, connect_port)
                await _send(ws, {"type": "log", "level": "info", "message": f"Connecting TCP {connect_host}:{connect_port}…"})
                log(f"[SocketTest] connect TCP {connect_host}:{connect_port}")
                loop = asyncio.get_running_loop()
                try:
                    sock = await loop.run_in_executor(
                        None,
                        lambda h=connect_host, p=connect_port: socket.create_connection((h, p), timeout=3.0),
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
                await _send(
                    ws,
                    {
                        "type": "log",
                        "level": "info",
                        "message": "Listening for results — click Trigger or run Measure in SVision.",
                    },
                )
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
