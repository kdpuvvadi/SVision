@echo off
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer and try again.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
.venv\Scripts\pip install "rapidocr-onnxruntime>=1.2.3,<1.3"
echo.
echo Installed. Start the server with run.bat
pause
