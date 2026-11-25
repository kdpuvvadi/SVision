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

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.cameras.base import CameraBackend, CameraStatus
from app.core.cameras.registry import register_backend

_XML_CACHE: dict[str, Path] = {}
_DEFAULT_CTI_HINTS = [
    Path(r"C:\Program Files\Common Files\OMRON_SENTECH\GenTL"),
    Path(r"C:\Program Files\Common Files\Sentech\GenTL"),
    Path(r"C:\Program Files\Sentech\GenTL"),
]


def _make_genapi_port(gentl_port):
    """Director port bridging GenTL Device.remote_port to GenApi NodeMap."""
    from genicam.genapi import AbstractPort, EAccessMode

    class GenApiPort(AbstractPort):
        def __init__(self, port) -> None:
            super().__init__()
            self._port = port

        def is_open(self) -> bool:
            return self._port is not None

        def write(self, address, value):
            self._port.write(address, value)

        def read(self, address, length):
            return self._port.read(address, length)[1]

        def get_access_mode(self):
            return EAccessMode.RW if self.is_open() else EAccessMode.NA

    return GenApiPort(gentl_port)


def discover_cti_files(extra: list[str] | None = None) -> list[dict[str, str]]:
    roots: list[Path] = []
    for raw in (os.environ.get("GENICAM_GENTL64_PATH") or "").split(os.pathsep):
        raw = raw.strip().strip('"')
        if raw:
            roots.append(Path(raw))
    roots.extend(_DEFAULT_CTI_HINTS)
    for item in extra or []:
        if item:
            roots.append(Path(item))

    found: dict[str, dict[str, str]] = {}
    for root in roots:
        try:
            if root.is_file() and root.suffix.lower() == ".cti":
                candidates = [root]
            elif root.is_dir():
                candidates = sorted(root.rglob("*.cti"))
            else:
                continue
        except OSError:
            continue
        for path in candidates:
            key = str(path.resolve()).lower()
            if key in found:
                continue
            found[key] = {
                "path": str(path),
                "name": path.name,
                "vendor_hint": path.parent.parent.name if path.parent else "",
            }
    def sort_key(item: dict[str, str]) -> tuple:
        path = item["path"].lower()
        preferred = 0 if ("omron_sentech" in path or "sentech" in path) else 1
        return (preferred, path)

    return sorted(found.values(), key=sort_key)


def list_gentl_devices(cti_path: str) -> list[dict[str, str]]:
    cti = (cti_path or "").strip()
    if not cti:
        raise ValueError("Choose a GenTL .cti producer file first.")
    if not Path(cti).is_file():
        raise FileNotFoundError(f"CTI file not found: {cti}")
    try:
        from genicam.gentl import GenTLProducer
    except ImportError as exc:
        raise RuntimeError("GenTL support needs the genicam package. Run install.bat.") from exc

    producer = GenTLProducer.create_producer()
    producer.open(cti)
    system = producer.create_system()
    system.open()
    try:
        system.update_interface_info_list(3000)
        devices: list[dict[str, str]] = []
        for iface_info in system.interface_info_list:
            iface = iface_info.create_interface()
            iface.open()
            try:
                iface.update_device_info_list(4000)
                for info in iface.device_info_list:
                    devices.append(
                        {
                            "id": str(getattr(info, "id_", "") or ""),
                            "display_name": str(getattr(info, "display_name", "") or ""),
                            "model": str(getattr(info, "model", "") or ""),
                            "vendor": str(getattr(info, "vendor", "") or ""),
                            "serial": str(getattr(info, "serial_number", "") or ""),
                            "tl_type": str(getattr(info, "tl_type", "") or ""),
                            "interface": str(getattr(iface_info, "display_name", "") or ""),
                        }
                    )
            finally:
                iface.close()
        return devices
    finally:
        system.close()
        producer.close()


def _load_remote_nodemap(device):
    from genicam.genapi import NodeMap

    remote = device.remote_port
    urls = list(remote.url_info_list)
    if not urls:
        raise RuntimeError("Camera remote port has no GenICam XML URL.")
    url = urls[0].url
    cache_key = f"{device.id_}|{url}"
    path = _XML_CACHE.get(cache_key)
    if path is None or not path.is_file():
        _location, others = url.split(":", 1)
        file_name, address_s, size_s = others.split(";")
        file_name = file_name.lstrip("/")
        address = int(address_s, 16)
        size = int(size_s, 16)
        _, payload = remote.read(address, size)
        folder = Path(tempfile.mkdtemp(prefix="svision-gentl-"))
        path = folder / (file_name or "device.zip")
        path.write_bytes(bytes(payload))
        _XML_CACHE[cache_key] = path

    node_map = NodeMap()
    try:
        node_map.load_xml_from_zip_file(str(path))
    except Exception:
        node_map.load_xml_from_file(str(path))
    node_map.connect(_make_genapi_port(remote), remote.name)
    return node_map


def _to_bgr(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    fmt = (pixel_format or "").upper()
    if frame.ndim == 2 or fmt in {"MONO8", "MONO10", "MONO12", "MONO16"}:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if "BAYERRG" in fmt:
        return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
    if "BAYERGR" in fmt:
        return cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
    if "BAYERGB" in fmt:
        return cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
    if "BAYERBG" in fmt:
        return cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
    if fmt in {"RGB8", "RGB8PACKED"}:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if fmt in {"BGR8", "BGR8PACKED"}:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


class GenTLBackend(CameraBackend):
    """GigE / USB3 Vision cameras through a vendor GenTL .cti producer."""

    source = "gentl"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.cti_path = (self.config.get("cti_path") or "").strip()
        self.device_id = (self.config.get("device_id") or "").strip()
        self.gige_ip = (self.config.get("gige_ip") or "").strip()
        self._lock = threading.RLock()
        self._producer = None
        self._system = None
        self._iface = None
        self._device = None
        self._stream = None
        self._node_map = None
        self._handles: list = []
        self._keep: list[bytes] = []
        self._acquiring = False
        self._width = 0
        self._height = 0
        self._pixel_format = "Mono8"
        self._payload = 0
        self._label = ""

    def status(self) -> CameraStatus:
        with self._lock:
            if not self.cti_path:
                return CameraStatus(self.source, False, "Choose a GenTL .cti producer file on the Camera tool.")
            try:
                self._ensure_open()
            except Exception as exc:
                self._safe_close()
                return CameraStatus(self.source, False, str(exc))
            label = self._label or self.device_id or self.gige_ip or Path(self.cti_path).name
            return CameraStatus(self.source, True, f"GenTL camera {label} ready.")

    def grab(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            assert self._stream is not None and self._node_map is not None
            from genicam.gentl import EVENT_TYPE_LIST, EventManagerNewBuffer

            manager = EventManagerNewBuffer(self._stream.register_event(EVENT_TYPE_LIST.EVENT_NEW_BUFFER))
            manager.update_event_data(5000)
            buffer = manager.buffer
            ctx = buffer.context
            if isinstance(ctx, int) and 0 <= ctx < len(self._keep):
                data = self._keep[ctx]
            else:
                data = bytes(buffer.raw_buffer)
            mono = np.frombuffer(data, dtype=np.uint8, count=self._width * self._height).reshape(
                self._height, self._width
            )
            frame = _to_bgr(mono.copy(), self._pixel_format)
            try:
                self._stream.queue_buffer(buffer)
            except Exception:
                pass
            return frame

    def close(self) -> None:
        with self._lock:
            self._safe_close()

    def _ensure_open(self) -> None:
        if self._acquiring and self._device is not None and self._device.is_open():
            return
        self._safe_close()
        self._open_session()

    def _open_session(self) -> None:
        from genicam.gentl import (
            ACQ_START_FLAGS_LIST,
            DEVICE_ACCESS_FLAGS_LIST as DAF,
            BufferToken,
            DataStream,
            GenTLProducer,
        )

        if not Path(self.cti_path).is_file():
            raise FileNotFoundError(f"CTI file not found: {self.cti_path}")

        producer = GenTLProducer.create_producer()
        producer.open(self.cti_path)
        system = producer.create_system()
        system.open()
        system.update_interface_info_list(3000)

        target = None
        iface = None
        display = ""
        wanted = self.device_id
        for iface_info in system.interface_info_list:
            candidate = iface_info.create_interface()
            candidate.open()
            candidate.update_device_info_list(4000)
            for info in candidate.device_info_list:
                device_id = str(getattr(info, "id_", "") or "")
                serial = str(getattr(info, "serial_number", "") or "")
                name = str(getattr(info, "display_name", "") or "")
                if wanted and wanted not in {device_id, serial, name}:
                    continue
                target = info
                iface = candidate
                display = name or device_id or serial
                break
            if target is not None:
                break
            candidate.close()

        if target is None:
            system.close()
            producer.close()
            hint = wanted or self.gige_ip or "any"
            raise RuntimeError(f"No GenTL camera matched ({hint}). Scan devices after choosing the CTI file.")

        assert iface is not None
        device = target.create_device()
        try:
            device.open(DAF.DEVICE_ACCESS_EXCLUSIVE)
        except Exception:
            device.open(DAF.DEVICE_ACCESS_CONTROL)

        node_map = _load_remote_nodemap(device)
        width = int(node_map.Width.value)
        height = int(node_map.Height.value)
        pixel = str(node_map.PixelFormat.value)
        payload = int(node_map.PayloadSize.value)

        if hasattr(node_map, "TriggerMode"):
            try:
                node_map.TriggerMode.value = "Off"
            except Exception:
                pass
        if hasattr(node_map, "GevSCPSPacketSize"):
            try:
                node_map.GevSCPSPacketSize.value = min(9000, int(node_map.GevSCPSPacketSize.max))
            except Exception:
                pass

        stream = DataStream(device)
        stream.open(list(device.data_stream_ids)[0])
        handles: list = []
        keep: list[bytes] = []
        for index in range(4):
            raw = bytes(payload)
            keep.append(raw)
            handle = stream.announce_buffer(BufferToken(raw, index))
            handles.append(handle)
            stream.queue_buffer(handle)

        stream.start_acquisition(ACQ_START_FLAGS_LIST.ACQ_START_FLAGS_DEFAULT)
        if hasattr(node_map, "TLParamsLocked"):
            node_map.TLParamsLocked.value = 1
        node_map.AcquisitionStart.execute()

        self._producer = producer
        self._system = system
        self._iface = iface
        self._device = device
        self._stream = stream
        self._node_map = node_map
        self._handles = handles
        self._keep = keep
        self._acquiring = True
        self._width = width
        self._height = height
        self._pixel_format = pixel
        self._payload = payload
        self._label = display
        if not self.device_id:
            self.device_id = str(getattr(target, "id_", "") or "")

    def _safe_close(self) -> None:
        from contextlib import suppress

        with suppress(Exception):
            if self._node_map is not None and hasattr(self._node_map, "AcquisitionStop"):
                self._node_map.AcquisitionStop.execute()
        with suppress(Exception):
            if self._node_map is not None and hasattr(self._node_map, "TLParamsLocked"):
                self._node_map.TLParamsLocked.value = 0
        with suppress(Exception):
            if self._stream is not None:
                from genicam.gentl import ACQ_STOP_FLAGS_LIST

                self._stream.stop_acquisition(ACQ_STOP_FLAGS_LIST.ACQ_STOP_FLAGS_KILL)
        for handle in self._handles:
            with suppress(Exception):
                if self._stream is not None:
                    self._stream.revoke_buffer(handle)
        with suppress(Exception):
            if self._stream is not None:
                self._stream.close()
        with suppress(Exception):
            if self._device is not None:
                self._device.close()
        with suppress(Exception):
            if self._iface is not None:
                self._iface.close()
        with suppress(Exception):
            if self._system is not None:
                self._system.close()
        with suppress(Exception):
            if self._producer is not None:
                self._producer.close()
        self._producer = None
        self._system = None
        self._iface = None
        self._device = None
        self._stream = None
        self._node_map = None
        self._handles = []
        self._keep = []
        self._acquiring = False


register_backend(GenTLBackend, "gige")
