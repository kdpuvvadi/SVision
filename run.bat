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
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

rem Free a leftover SVision bind on the configured port (default 8080).
set "SVISION_PORT_CHECK=%SVISION_PORT%"
if "%SVISION_PORT_CHECK%"=="" set "SVISION_PORT_CHECK=8080"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port = %SVISION_PORT_CHECK%;" ^
  "$pids = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique);" ^
  "foreach ($procId in $pids) {" ^
  "  if (-not $procId) { continue }" ^
  "  $p = Get-CimInstance Win32_Process -Filter (\"ProcessId=$procId\") -ErrorAction SilentlyContinue;" ^
  "  $cmd = [string]$p.CommandLine;" ^
  "  if ($cmd -match 'app\.main' -or $cmd -match 'SVision' -or $cmd -match 'uvicorn') {" ^
  "    Write-Host (\"Stopping leftover SVision on port $port (PID $procId)...\");" ^
  "    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue;" ^
  "  } else {" ^
  "    Write-Host (\"Port $port is already in use by PID $procId.\");" ^
  "    Write-Host 'Close that program, or set SVISION_PORT to another port, then retry.';" ^
  "    exit 2;" ^
  "  }" ^
  "}" ^
  "Start-Sleep -Milliseconds 400;" ^
  "exit 0"
if errorlevel 2 (
  pause
  exit /b 1
)

python -m app.main
pause
