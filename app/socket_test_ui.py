# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

"""Open the Socket Test web UI (same stack as the main app)."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import webbrowser

from app.config import PORT
from app.core.logutil import log


def _wait_ready(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if 200 <= int(resp.status) < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.4)
    return False


def main() -> None:
    port = int(os.environ.get("SVISION_PORT", str(PORT)) or PORT)
    page = f"http://127.0.0.1:{port}/socket-test"
    health = f"http://127.0.0.1:{port}/api/health"
    log(f"[SocketTest] waiting for SVision at {health}")
    if not _wait_ready(health):
        log("[SocketTest] SVision did not become ready — opening page anyway")
    else:
        log(f"[SocketTest] opening {page}")
    webbrowser.open_new(page)


if __name__ == "__main__":
    main()
