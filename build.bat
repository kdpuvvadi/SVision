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

set "APP_VERSION="
if not exist ".version" (
  echo .version was not found.
  exit /b 1
)
for /f "usebackq delims=" %%V in (".version") do (
  if not defined APP_VERSION set "APP_VERSION=%%V"
)
if not defined APP_VERSION (
  echo .version is empty.
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

set "OUT_DIR=dist\SVision-%APP_VERSION%"
if exist "%OUT_DIR%" rmdir /s /q "%OUT_DIR%"
if errorlevel 1 (
  echo Failed to replace %OUT_DIR%.
  exit /b 1
)
move "dist\SVision" "%OUT_DIR%" >nul
if errorlevel 1 (
  echo Failed to save the build as %OUT_DIR%.
  exit /b 1
)

echo.
echo Self-contained app:
echo   %OUT_DIR%\SVision.exe
echo.
echo Copy the whole %OUT_DIR% folder to the IPC. No Python install is required there.
echo Close the console window to stop the server.
endlocal
