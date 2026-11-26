# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

"""Small TCP result viewer for SVision socket communication."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import tkinter as tk
from tkinter import ttk


DEFAULT_HOST = os.environ.get("SVISION_SOCKET_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SVISION_SOCKET_PORT", "9100") or 9100)


class SocketTestApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SVision · Socket test")
        self.root.geometry("760x520")
        self.root.minsize(560, 360)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._connected = False
        self._last: dict | None = None

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Host").pack(side="left")
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(top, textvariable=self.host_var, width=18).pack(side="left", padx=6)
        ttk.Label(top, text="Port").pack(side="left")
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        ttk.Entry(top, textvariable=self.port_var, width=8).pack(side="left", padx=6)
        self.connect_btn = ttk.Button(top, text="Connect", command=self.toggle_connect)
        self.connect_btn.pack(side="left", padx=6)
        self.trigger_btn = ttk.Button(top, text="Trigger", command=self.send_trigger, state="disabled")
        self.trigger_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=10)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=1)
        body.add(right, weight=2)

        self.scene_var = tk.StringVar(value="Scene: —")
        self.judgment_var = tk.StringVar(value="—")
        self.message_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.scene_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.judgment_label = ttk.Label(left, textvariable=self.judgment_var, font=("Segoe UI", 28, "bold"))
        self.judgment_label.pack(anchor="w", pady=(8, 4))
        ttk.Label(left, textvariable=self.elapsed_var).pack(anchor="w")
        ttk.Label(left, textvariable=self.message_var, wraplength=240).pack(anchor="w", pady=(4, 10))

        ttk.Label(left, text="Tools").pack(anchor="w")
        cols = ("name", "judgment", "ms")
        self.tools = ttk.Treeview(left, columns=cols, show="headings", height=12)
        self.tools.heading("name", text="Tool")
        self.tools.heading("judgment", text="Result")
        self.tools.heading("ms", text="ms")
        self.tools.column("name", width=140)
        self.tools.column("judgment", width=70, anchor="center")
        self.tools.column("ms", width=60, anchor="e")
        self.tools.pack(fill="both", expand=True)

        ttk.Label(right, text="JSON").pack(anchor="w")
        self.json_text = tk.Text(right, wrap="none", font=("Consolas", 10))
        yscroll = ttk.Scrollbar(right, orient="vertical", command=self.json_text.yview)
        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.json_text.xview)
        self.json_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.json_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._reader: threading.Thread | None = None
        # Auto-connect shortly after launch so run.bat feels ready.
        self.root.after(600, self.connect)

    def toggle_connect(self) -> None:
        if self._connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        self.disconnect()
        host = self.host_var.get().strip() or "127.0.0.1"
        try:
            port = int(self.port_var.get().strip() or "9100")
        except ValueError:
            self.status_var.set("Invalid port")
            return
        self.status_var.set(f"Connecting {host}:{port}…")
        self._stop.clear()

        def worker() -> None:
            while not self._stop.is_set():
                try:
                    sock = socket.create_connection((host, port), timeout=2.0)
                    sock.settimeout(0.5)
                    self._sock = sock
                    self._connected = True
                    self.root.after(0, lambda: self._set_connected(True, f"Connected {host}:{port}"))
                    self._read_loop(sock)
                except OSError as exc:
                    self._connected = False
                    self._sock = None
                    msg = f"Waiting… ({exc})"
                    self.root.after(0, lambda m=msg: self._set_connected(False, m))
                    self._stop.wait(1.5)
            self.root.after(0, lambda: self._set_connected(False, "Disconnected"))

        self._reader = threading.Thread(target=worker, name="socket-test-reader", daemon=True)
        self._reader.start()

    def disconnect(self) -> None:
        self._stop.set()
        sock = self._sock
        self._sock = None
        self._connected = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._set_connected(False, "Disconnected")

    def _set_connected(self, ok: bool, text: str) -> None:
        self.status_var.set(text)
        self.connect_btn.configure(text="Disconnect" if ok else "Connect")
        self.trigger_btn.configure(state="normal" if ok else "disabled")

    def _read_loop(self, sock: socket.socket) -> None:
        buf = b""
        while not self._stop.is_set() and self._sock is sock:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
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
                    continue
                self.root.after(0, lambda p=payload: self.show_payload(p))
        if self._sock is sock:
            self._connected = False
            self._sock = None

    def send_trigger(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall(b'{"cmd":"trigger"}\n')
            self.status_var.set("Trigger sent")
        except OSError as exc:
            self.status_var.set(f"Trigger failed: {exc}")

    def show_payload(self, payload: dict) -> None:
        self._last = payload
        scene = payload.get("scene") or {}
        self.scene_var.set(f"Scene: {scene.get('name') or scene.get('id') or '—'}")
        judgment = str(payload.get("judgment") or "—")
        self.judgment_var.set(judgment)
        color = "#1f9d55" if judgment == "OK" else "#c0392b" if judgment in {"NG", "ERROR"} else "#333"
        self.judgment_label.configure(foreground=color)
        elapsed = payload.get("elapsed_ms")
        self.elapsed_var.set(f"{float(elapsed):.0f} ms" if elapsed is not None else "")
        self.message_var.set(str(payload.get("message") or payload.get("error") or ""))
        for item in self.tools.get_children():
            self.tools.delete(item)
        for tool in payload.get("tools") or []:
            self.tools.insert(
                "",
                "end",
                values=(
                    tool.get("name") or tool.get("type") or tool.get("id") or "?",
                    tool.get("judgment") or "—",
                    f"{float(tool.get('elapsed_ms') or 0):.0f}",
                ),
            )
        self.json_text.delete("1.0", "end")
        self.json_text.insert("1.0", json.dumps(payload, indent=2))

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> None:
    # Wait briefly so SVision can bind first when launched from run.bat.
    time.sleep(0.2)
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    SocketTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
