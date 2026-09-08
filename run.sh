#!/bin/bash
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
