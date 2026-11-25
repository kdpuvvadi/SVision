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

import importlib
import pkgutil
import uuid
from typing import Any

from app.tools.base import Tool

_TOOLS: dict[str, Tool] = {}
_DISCOVERED = False
_SKIP_MODULES = {"base", "registry", "__init__"}


def register(tool: Tool) -> Tool:
    if not tool.type:
        raise ValueError("Tool.type is required")
    _TOOLS[tool.type] = tool
    return tool


def get_tool(tool_type: str) -> Tool:
    discover()
    tool = _TOOLS.get(tool_type)
    if tool is None:
        raise KeyError(f"Unknown tool: {tool_type}")
    return tool


def all_tools() -> list[Tool]:
    discover()
    return list(_TOOLS.values())


def catalog() -> list[dict]:
    return [tool.catalog() for tool in all_tools()]


def new_tool(tool_type: str, name: str | None = None) -> dict:
    spec = get_tool(tool_type)
    return {
        "id": uuid.uuid4().hex[:10],
        "type": spec.type,
        "name": (name or spec.title).strip() or spec.title,
        "config": spec.default_config(),
    }


def normalize_config(tool: dict) -> dict:
    spec = get_tool(tool["type"])
    raw = dict(tool.get("config") or {})
    for key in spec.default_config():
        if key not in raw and key in tool:
            raw[key] = tool[key]
    config = spec.default_config()
    config.update({key: raw[key] for key in spec.default_config() if key in raw})
    return {
        "id": tool["id"],
        "type": spec.type,
        "name": (tool.get("name") or spec.title).strip() or spec.title,
        "config": config,
    }


def mount_all(app: Any) -> None:
    for tool in all_tools():
        tool.mount(app)


def notify_project_updated(project: dict[str, Any]) -> None:
    for tool in all_tools():
        try:
            tool.on_project_updated(project)
        except Exception:
            pass


def notify_project_deleted(project_id: str, tools: list[dict[str, Any]]) -> None:
    for tool in all_tools():
        try:
            tool.on_project_deleted(project_id, tools)
        except Exception:
            pass


def notify_measure_done(project_id: str, result: dict[str, Any]) -> None:
    for tool in all_tools():
        try:
            tool.on_measure_done(project_id, result)
        except Exception:
            pass


def discover() -> None:
    """Import every app.tools.* module so tools can self-register."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import app.tools as tools_pkg

    for module in pkgutil.iter_modules(tools_pkg.__path__):
        name = module.name
        if name in _SKIP_MODULES or name.startswith("_") or name.endswith("_routes"):
            continue
        importlib.import_module(f"app.tools.{name}")
