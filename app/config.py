from __future__ import annotations

import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = Path.home() / ".config" / "svision" / "data"
DATA_DIR = Path(os.environ.get("SVISION_DATA") or _DEFAULT_DATA).expanduser()
HOST = os.environ.get("SVISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("SVISION_PORT", "8080"))
MAX_UPLOAD_MB = int(os.environ.get("SVISION_MAX_UPLOAD_MB", "40"))
THREAD_COUNT = max(2, min(int(os.environ.get("SVISION_THREADS", "0") or (os.cpu_count() or 4)), 16))

FEATURE_VERSION = 2
FEATURE_SIZE = 64


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
