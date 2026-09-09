#!/bin/bash
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
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python was not found. Install Python 3.10 or newer and try again."
  exit 1
fi

if [ ! -f ".venv/bin/activate" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv || exit 1
  source ".venv/bin/activate"
  python -m pip install --upgrade pip
  pip install -r requirements.txt || exit 1
else
  source ".venv/bin/activate"
fi

python -m app.main
