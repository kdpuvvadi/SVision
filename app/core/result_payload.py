# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

from __future__ import annotations

import time
from typing import Any

from app.storage.store import store


def _tool_public(tool: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in tool.items() if k not in {"preview", "preview_jpeg"}}
    return out


def _item_public(item: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in item.items() if k not in {"preview", "preview_jpeg"}}
    out["tools"] = [_tool_public(dict(t)) for t in (item.get("tools") or [])]
    return out


def build_result_payload(project_id: str, result: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    """Compact JSON for TCP / MQTT clients (no base64 previews)."""
    scene_name = ""
    tools_flow: list[dict[str, Any]] = []
    try:
        project = store.get_project(project_id)
        scene_name = str(project.get("name") or "")
        for tool in (project.get("flow") or {}).get("tools") or []:
            tools_flow.append(
                {
                    "id": tool.get("id"),
                    "type": tool.get("type"),
                    "name": tool.get("name") or tool.get("type"),
                }
            )
    except Exception:
        pass

    result = result or {}
    items = [_item_public(dict(item)) for item in (result.get("results") or [])]
    if not items and result.get("tools") is not None:
        items = [_item_public(dict(result))]

    tools_used: list[dict[str, Any]] = []
    if items:
        tools_used = list(items[0].get("tools") or [])

    judgment = result.get("judgment")
    if not judgment and items:
        judgment = items[0].get("judgment")
    if error:
        judgment = "ERROR"

    return {
        "type": "svision.result",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scene": {"id": project_id, "name": scene_name},
        "flow": tools_flow,
        "judgment": judgment or "UNKNOWN",
        "message": error or result.get("message") or "",
        "elapsed_ms": result.get("elapsed_ms"),
        "count": result.get("count", len(items)),
        "ok": result.get("ok"),
        "ng": result.get("ng"),
        "tools": tools_used,
        "results": items,
        "error": error or "",
    }
