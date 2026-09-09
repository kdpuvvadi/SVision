@echo off
rem SVision inspection software
rem Copyright (C) 2026 KD Puvvadi
rem This program is free software; you can redistribute it and/or
rem modify it under the terms of the GNU General Public License
rem as published by the Free Software Foundation; either version 2
rem of the License, or (at your option) any later version.
rem
rem This program is distributed in the hope that it will be useful,
rem but WITHOUT ANY WARRANTY; without even the implied warranty of
rem MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
rem GNU General Public License for more details.
rem
rem You should have received a copy of the GNU General Public License
rem along with this program; if not, write to the Free Software
rem Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
pip install pyinstaller

pyinstaller --noconfirm --clean svision.spec
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)
copy /Y LICENSE dist\SVision\LICENSE >nul
if errorlevel 1 (
  echo Failed to copy LICENSE into the build.
  exit /b 1
)

echo.
echo Self-contained app:
echo   dist\SVision\SVision.exe
echo.
echo Copy the whole dist\SVision folder to the IPC. No Python install is required there.
echo Close the console window to stop the server.
endlocal
