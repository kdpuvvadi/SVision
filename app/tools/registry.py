from __future__ import annotations

import uuid

from app.tools.base import Tool
from app.tools.camera import CameraTool
from app.tools.classify import ClassifyTool
from app.tools.code import CodeTool
from app.tools.ocr import OcrTool
from app.tools.position_compensation import PositionCompensationTool
from app.tools.shape_search import ShapeSearchTool

_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _TOOLS[tool.type] = tool


def get_tool(tool_type: str) -> Tool:
    tool = _TOOLS.get(tool_type)
    if tool is None:
        raise KeyError(f"Unknown tool: {tool_type}")
    return tool


def catalog() -> list[dict]:
    return [tool.catalog() for tool in _TOOLS.values()]


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


register(CameraTool())
register(ShapeSearchTool())
register(PositionCompensationTool())
register(ClassifyTool())
register(OcrTool())
register(CodeTool())
