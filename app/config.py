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
import socket
import sys
from pathlib import Path


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


ROOT = _app_root()
_DEFAULT_DATA = Path.home() / ".config" / "svision" / "data"
DATA_DIR = Path(os.environ.get("SVISION_DATA") or _DEFAULT_DATA).expanduser()
HOST = os.environ.get("SVISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("SVISION_PORT", "8080"))
MAX_UPLOAD_MB = int(os.environ.get("SVISION_MAX_UPLOAD_MB", "40"))
THREAD_COUNT = max(2, min(int(os.environ.get("SVISION_THREADS", "0") or (os.cpu_count() or 4)), 16))

FEATURE_VERSION = 2
FEATURE_SIZE = 64


def app_version() -> str:
    path = ROOT / ".version"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text.splitlines()[0].strip() if text else ""


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)


def lan_addresses() -> list[str]:
    ips: list[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.4)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            ips.append(ip)
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips
