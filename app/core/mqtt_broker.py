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
import struct
import threading
from typing import Any

from app.config import lan_addresses

log = logging.getLogger("svision.mqtt")

# MQTT 3.1.1 packet types
CONNECT = 1
CONNACK = 2
PUBLISH = 3
SUBSCRIBE = 8
SUBACK = 9
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14


def _encode_remaining(length: int) -> bytes:
    out = bytearray()
    while True:
        digit = length % 128
        length //= 128
        if length:
            digit |= 0x80
        out.append(digit)
        if not length:
            break
    return bytes(out)


def _decode_remaining(data: bytes, start: int) -> tuple[int, int]:
    multiplier = 1
    value = 0
    pos = start
    while True:
        if pos >= len(data):
            raise ValueError("incomplete remaining length")
        encoded = data[pos]
        pos += 1
        value += (encoded & 127) * multiplier
        if not (encoded & 128):
            break
        multiplier *= 128
        if multiplier > 128 * 128 * 128:
            raise ValueError("remaining length too large")
    return value, pos


def _encode_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("!H", len(raw)) + raw


def _topic_match(filter_topic: str, topic: str) -> bool:
    if filter_topic == "#":
        return True
    f_parts = filter_topic.split("/")
    t_parts = topic.split("/")
    for i, part in enumerate(f_parts):
        if part == "#":
            return True
        if i >= len(t_parts):
            return False
        if part == "+":
            continue
        if part != t_parts[i]:
            return False
    return len(f_parts) == len(t_parts)


class MqttResultBridge:
    """Minimal MQTT 3.1.1 broker + result publisher (QoS 0)."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._clients_lock = threading.Lock()
        self._clients: dict[socket.socket, set[str]] = {}
        self._topic = "svision/results"
        self._last_payload: dict[str, Any] | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "error": "",
            "host": "",
            "port": 0,
            "topic": self._topic,
            "clients": 0,
            "connect": [],
        }

    def status(self) -> dict[str, Any]:
        with self._clients_lock:
            clients = len(self._clients)
        out = dict(self._status)
        out["clients"] = clients
        out["topic"] = self._topic
        return out

    def start(self, host: str = "0.0.0.0", port: int = 1883, topic: str = "svision/results") -> None:
        self.stop()
        self._stop.clear()
        host = str(host or "0.0.0.0")
        port = int(port or 1883)
        self._topic = str(topic or "svision/results").strip() or "svision/results"
        connect = [f"127.0.0.1:{port}"] + [f"{ip}:{port}" for ip in lan_addresses()]
        self._status = {
            "running": False,
            "error": "",
            "host": host,
            "port": port,
            "topic": self._topic,
            "clients": 0,
            "connect": connect,
            "protocol": "mqtt-3.1.1",
            "hint": f"Subscribe with any MQTT client to topic '{self._topic}'. QoS 0 JSON payloads on each measure.",
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
            log.error("mqtt bind failed: %s", exc)
            return
        self._thread = threading.Thread(target=self._serve, name="svision-mqtt", daemon=True)
        self._thread.start()
        self._status["running"] = True
        log.info("mqtt broker listening on %s:%s topic=%s", host, port, self._topic)

    def stop(self) -> None:
        self._stop.set()
        with self._clients_lock:
            for client in list(self._clients):
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
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        packet = self._publish_packet(self._topic, body)
        dead: list[socket.socket] = []
        with self._clients_lock:
            items = list(self._clients.items())
        for client, subs in items:
            if any(_topic_match(sub, self._topic) for sub in subs):
                try:
                    client.sendall(packet)
                except OSError:
                    dead.append(client)
        if dead:
            with self._clients_lock:
                for client in dead:
                    self._clients.pop(client, None)
                    try:
                        client.close()
                    except OSError:
                        pass

    @staticmethod
    def _publish_packet(topic: str, payload: bytes) -> bytes:
        variable = _encode_str(topic) + payload
        return bytes([PUBLISH << 4]) + _encode_remaining(len(variable)) + variable

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(1.0)
            with self._clients_lock:
                self._clients[client] = set()
            log.info("mqtt client connected %s", addr)
            threading.Thread(
                target=self._client_loop,
                args=(client, addr),
                name=f"svision-mqtt-{addr[1]}",
                daemon=True,
            ).start()

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
                while True:
                    if len(buf) < 2:
                        break
                    try:
                        remaining, pos = _decode_remaining(buf, 1)
                    except ValueError:
                        break
                    if len(buf) < pos + remaining:
                        break
                    packet = buf[: pos + remaining]
                    buf = buf[pos + remaining :]
                    self._handle_packet(client, packet)
        finally:
            with self._clients_lock:
                self._clients.pop(client, None)
            try:
                client.close()
            except OSError:
                pass
            log.info("mqtt client disconnected %s", addr)

    def _handle_packet(self, client: socket.socket, packet: bytes) -> None:
        if not packet:
            return
        ptype = packet[0] >> 4
        remaining, pos = _decode_remaining(packet, 1)
        body = packet[pos : pos + remaining]
        if ptype == CONNECT:
            # CONNACK accepted
            client.sendall(bytes([CONNACK << 4, 2, 0, 0]))
            if self._last_payload is not None:
                with self._clients_lock:
                    # wait until they subscribe
                    pass
        elif ptype == SUBSCRIBE:
            if len(body) < 2:
                return
            packet_id = struct.unpack("!H", body[:2])[0]
            idx = 2
            codes = bytearray()
            while idx + 2 <= len(body):
                tlen = struct.unpack("!H", body[idx : idx + 2])[0]
                idx += 2
                topic = body[idx : idx + tlen].decode("utf-8", errors="replace")
                idx += tlen
                if idx >= len(body):
                    break
                idx += 1  # requested QoS
                with self._clients_lock:
                    self._clients.setdefault(client, set()).add(topic)
                codes.append(0)  # QoS 0 granted
                if self._last_payload is not None and _topic_match(topic, self._topic):
                    try:
                        body_json = json.dumps(self._last_payload, ensure_ascii=False).encode("utf-8")
                        client.sendall(self._publish_packet(self._topic, body_json))
                    except OSError:
                        pass
            client.sendall(bytes([SUBACK << 4]) + _encode_remaining(2 + len(codes)) + struct.pack("!H", packet_id) + bytes(codes))
        elif ptype == PINGREQ:
            client.sendall(bytes([PINGRESP << 4, 0]))
        elif ptype == DISCONNECT:
            raise OSError("client disconnect")
        elif ptype == PUBLISH:
            # Ignore client publishes for now (results flow SVision → clients).
            return


mqtt_bridge = MqttResultBridge()
