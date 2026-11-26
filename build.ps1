# SVision inspection software
# Copyright (C) 2026 KD Puvvadi
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

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Fail([string]$Message) {
    Write-Host $Message
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python was not found. Install Python 3.10 or newer, then run this again."
}

$versionPath = Join-Path $PSScriptRoot ".version"
if (-not (Test-Path -LiteralPath $versionPath)) {
    Fail ".version was not found."
}
$appVersion = (Get-Content -LiteralPath $versionPath -TotalCount 1).Trim()
if (-not $appVersion) {
    Fail ".version is empty."
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv .venv
}

$activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
. $activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean svision.spec
if ($LASTEXITCODE -ne 0) {
    Fail "Build failed."
}

$builtDir = Join-Path $PSScriptRoot "dist\SVision"
$licenseSrc = Join-Path $PSScriptRoot "LICENSE"
Copy-Item -LiteralPath $licenseSrc -Destination (Join-Path $builtDir "LICENSE") -Force

$outDir = Join-Path $PSScriptRoot "dist\SVision-$appVersion"
if (Test-Path -LiteralPath $outDir) {
    Remove-Item -LiteralPath $outDir -Recurse -Force
}
Move-Item -LiteralPath $builtDir -Destination $outDir

Write-Host ""
Write-Host "Self-contained app:"
Write-Host "  $outDir\SVision.exe"
Write-Host ""
Write-Host "Copy the whole $outDir folder to the IPC. No Python install is required there."
Write-Host "Close the console window to stop the server."
