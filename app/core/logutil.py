# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# Timestamped console logging for Measure / GenTL / PLC lines.

from __future__ import annotations

from datetime import datetime


def log(message: str, *, flush: bool = True) -> None:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{stamp} {message}", flush=flush)
