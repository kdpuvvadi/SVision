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

import time

import cv2
import numpy as np

from app.core.features import _roi_box
from app.tools.base import Tool


def _crop(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return image[y0:y1, x0:x1].copy()


class PositionCompensationTool(Tool):
    type = "position_compensation"
    title = "Position Compensation"
    summary = "Crop the image to the selected region so later tools see fewer pixels."
    transforms_image = True

    def fields(self) -> list[dict]:
        return [
            {"key": "roi", "label": "Crop region", "kind": "roi"},
            {
                "key": "align",
                "label": "Align to Shape Search",
                "kind": "select",
                "options": [
                    {"value": "yes", "label": "Yes, follow found position"},
                    {"value": "no", "label": "No, crop the fixed region"},
                ],
            },
        ]

    def default_config(self) -> dict:
        return {"roi": None, "align": "yes"}

    def status(self) -> str:
        return "Optional. Place after Shape Search if the crop should follow the part. Tools after this one run on the cropped image."

    def run(self, image: np.ndarray, tool: dict, project_id: str, ctx: dict | None = None) -> tuple[dict, np.ndarray | None]:
        started = time.perf_counter()
        cfg = tool.get("config") or {}
        name = tool.get("name") or self.title
        roi = cfg.get("roi")
        if not roi:
            return {
                "id": tool.get("id"),
                "type": self.type,
                "name": name,
                "judgment": "NG",
                "message": "Draw the region to crop.",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }, None

        pose = (ctx or {}).get("pose") if (cfg.get("align") or "yes") == "yes" else None
        before = int(image.shape[0] * image.shape[1])
        angle = 0.0
        dx = 0.0
        dy = 0.0

        if pose:
            angle = float(pose.get("angle") or 0.0)
            cx = float(pose["x"])
            cy = float(pose["y"])
            rotated = image
            if abs(angle) >= 0.5:
                matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(
                    image,
                    matrix,
                    (image.shape[1], image.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            _, _, x1, y1 = _roi_box(image, roi)
            x0, y0, _, _ = _roi_box(image, roi)
            rw = max(1, x1 - x0)
            rh = max(1, y1 - y0)
            dx = cx - ((x0 + x1) / 2.0)
            dy = cy - ((y0 + y1) / 2.0)
            nx0 = int(round(cx - rw / 2.0))
            ny0 = int(round(cy - rh / 2.0))
            cropped = _crop(rotated, nx0, ny0, nx0 + rw, ny0 + rh)
        else:
            x0, y0, x1, y1 = _roi_box(image, roi)
            cropped = _crop(image, x0, y0, x1, y1)

        after = int(cropped.shape[0] * cropped.shape[1])
        if ctx is not None:
            ctx["image"] = cropped
        preview = cropped
        long_edge = max(cropped.shape[:2])
        if long_edge > 640:
            scale = 640.0 / long_edge
            preview = cv2.resize(
                cropped,
                (max(1, int(cropped.shape[1] * scale)), max(1, int(cropped.shape[0] * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        followed = "followed Shape Search" if pose else "fixed region"
        return {
            "id": tool.get("id"),
            "type": self.type,
            "name": name,
            "judgment": "OK",
            "width": int(cropped.shape[1]),
            "height": int(cropped.shape[0]),
            "pixels_before": before,
            "pixels_after": after,
            "angle": round(angle, 1),
            "offset_x": round(dx, 1),
            "offset_y": round(dy, 1),
            "message": f"Crop {cropped.shape[1]}×{cropped.shape[0]}  {after}/{before} px  {followed}",
            "elapsed_ms": elapsed,
        }, preview
