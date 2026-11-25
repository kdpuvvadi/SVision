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

from typing import Any, Type

from app.core.cameras.base import CameraBackend

_BACKENDS: dict[str, Type[CameraBackend]] = {}
_ALIASES: dict[str, str] = {}


def register_backend(backend: Type[CameraBackend], *aliases: str) -> Type[CameraBackend]:
    if not backend.source:
        raise ValueError("Camera backend needs a source id")
    _BACKENDS[backend.source] = backend
    for alias in aliases:
        _ALIASES[alias.strip().lower()] = backend.source
    return backend


def resolve_source(source: str) -> str:
    key = (source or "upload").strip().lower()
    return _ALIASES.get(key, key)


def create_backend(source: str, config: dict[str, Any] | None = None) -> CameraBackend:
    key = resolve_source(source)
    cls = _BACKENDS.get(key)
    if cls is None:
        raise KeyError(f"Unknown camera source: {source}")
    return cls(config)


def list_backends() -> list[str]:
    return sorted(_BACKENDS)
