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

"""Pluggable camera backends used by the Camera tool."""

from app.core.cameras.base import CameraBackend, CameraStatus
from app.core.cameras.registry import create_backend, list_backends, register_backend, resolve_source

# Import backends so they self-register.
from app.core.cameras import gentl as _gentl  # noqa: F401
from app.core.cameras import upload as _upload  # noqa: F401
from app.core.cameras import usb as _usb  # noqa: F401

__all__ = [
    "CameraBackend",
    "CameraStatus",
    "create_backend",
    "list_backends",
    "register_backend",
    "resolve_source",
]
