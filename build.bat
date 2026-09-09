@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer, then run this again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "rapidocr-onnxruntime>=1.2.3,<1.3"
pip install pyinstaller

pyinstaller --noconfirm --clean svision.spec
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Self-contained app:
echo   dist\SVision\SVision.exe
echo.
echo Copy the whole dist\SVision folder to the IPC. No Python install is required there.
echo Close the console window to stop the server.
endlocal
