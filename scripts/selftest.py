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

"""Train on synthetic OK/NG images and inspect both classes."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ensure_dirs
from app.core.engine import inspect_array
from app.storage.store import store


def make_ok() -> np.ndarray:
    img = np.full((240, 320, 3), 230, np.uint8)
    cv2.rectangle(img, (120, 40), (200, 200), (40, 40, 40), -1)
    for i in range(8):
        cv2.line(img, (128, 55 + i * 16), (192, 55 + i * 16), (250, 250, 250), 6)
    return img


def make_ng(seed: int) -> np.ndarray:
    img = make_ok()
    rng = np.random.default_rng(seed)
    x = int(rng.integers(130, 180))
    y = int(rng.integers(60, 170))
    cv2.circle(img, (x, y), 10, (20, 20, 20), -1)
    return img


def encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def main() -> None:
    ensure_dirs()
    project = store.create_project("Self test")
    pid = project["id"]
    store.add_flow_tool(pid, "camera")
    tool = store.add_flow_tool(pid, "classify")
    tid = tool["id"]
    store.add_samples(pid, tid, "OK", [(f"ok{i}.png", encode(make_ok())) for i in range(4)])
    store.add_samples(pid, tid, "NG", [(f"ng{i}.png", encode(make_ng(i))) for i in range(4)])
    stats = store.train(pid, tid)
    ok_result = inspect_array(pid, make_ok(), "ok.png")
    ng_result = inspect_array(pid, make_ng(99), "ng.png")
    print("train_ms", round(stats["elapsed_ms"], 2), "fit", round(stats["train_accuracy"], 3))
    print("ok", ok_result["judgment"], f"{ok_result['elapsed_ms']:.2f} ms", ok_result["tools"])
    print("ng", ng_result["judgment"], f"{ng_result['elapsed_ms']:.2f} ms", ng_result["tools"])
    store.delete_project(pid)
    if ok_result["judgment"] != "OK" or ng_result["judgment"] != "NG":
        raise SystemExit("Classifier did not separate the synthetic samples.")
    print("selftest ok")


if __name__ == "__main__":
    main()
