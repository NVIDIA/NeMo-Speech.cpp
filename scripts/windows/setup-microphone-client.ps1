# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param([string]$Python)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Requirements = Join-Path $RepoRoot 'examples\python\requirements-microphone.txt'
$Venv = Join-Path $RepoRoot '.tools\microphone-client-venv'
if (-not $Python) { $Python = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source) }
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python 3.11 or newer was not found. Pass -Python with its executable path.' }
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required.' }
& $Python -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw 'Unable to create the virtual environment.' }
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Unable to upgrade pip.' }
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'Unable to install microphone client dependencies.' }
Write-Host 'Microphone client environment is ready.' -ForegroundColor Green
Write-Host '& ".\.tools\microphone-client-venv\Scripts\python.exe" `'
Write-Host '  ".\examples\python\microphone_http.py" `'
Write-Host '  --language en `'
Write-Host '  --show-words'
