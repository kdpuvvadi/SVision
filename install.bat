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
echo.
rem OCR GPU is optional: requirements.txt installs CPU onnxruntime via rapidocr-onnxruntime.
rem Offer CUDA OCR only when an NVIDIA GPU is present; skip quietly otherwise.
where nvidia-smi >nul 2>&1
if errorlevel 1 goto after_gpu_ocr
echo.
echo NVIDIA GPU detected.
echo OCR GPU needs onnxruntime-gpu (replaces CPU onnxruntime). Shape Search already uses OpenCL.
set /p GPU_OCR="Install onnxruntime-gpu for CUDA OCR? [y/N] "
if /i not "%GPU_OCR%"=="y" if /i not "%GPU_OCR%"=="yes" goto after_gpu_ocr
echo Installing onnxruntime-gpu...
pip uninstall -y onnxruntime >nul 2>&1
pip install onnxruntime-gpu
if errorlevel 1 (
  echo onnxruntime-gpu install failed — OCR will keep using CPU. Shape Search OpenCL is unchanged.
) else (
  echo OCR CUDA provider installed.
)
:after_gpu_ocr
echo.
echo Installed.
echo Note: default install is CPU OCR. OpenCL GPU for Shape Search needs no extra package.
set /p RUN_NOW="Start SVision now with run.bat? [y/N] "
if /i "%RUN_NOW%"=="y" goto run_app
if /i "%RUN_NOW%"=="yes" goto run_app
echo Start later with run.bat
pause
exit /b 0

:run_app
call "%~dp0run.bat"
