# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [string]$Executable,
    [ValidatePattern('^cuda(:[0-9]+)?$')] [string]$Device = 'cuda:0',
    [ValidateRange(1, 65535)] [int]$Port = 8081
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Executable) { $Executable = Join-Path $RepoRoot 'build-pascal-cuda-http\bin\nemo-speech.exe' }
foreach ($item in @(@{ Name = 'Model'; Path = $Model }, @{ Name = 'Executable'; Path = $Executable })) {
    if (-not (Test-Path -LiteralPath $item.Path -PathType Leaf)) { throw "$($item.Name) not found: $($item.Path)" }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use. No process was stopped." }
$baseUrl = "http://127.0.0.1:$Port"
$arguments = @('serve', '--asr-model', $Model, '--device', $Device, '--host', '127.0.0.1', '--port', $Port, '--skinny-q8', 'auto', '--suppress-cuda-graph-log')
Write-Host "Server:`n$baseUrl`nReady:`n$baseUrl/ready`nTranscriptions:`n$baseUrl/v1/audio/transcriptions" -ForegroundColor Cyan
Write-Host "`nRun this in another PowerShell after preparing the client:" -ForegroundColor Yellow
Write-Host '& ".\.tools\microphone-client-venv\Scripts\python.exe" `'
Write-Host '  ".\examples\python\microphone_http.py" `'
Write-Host ('  --url "{0}/v1/audio/transcriptions" `' -f $baseUrl)
Write-Host '  --language en `'
Write-Host '  --show-words'
Write-Host ('`nRunning: & "{0}" {1}' -f $Executable, (($arguments | ForEach-Object { '"{0}"' -f $_ }) -join ' '))
& $Executable @arguments
exit $LASTEXITCODE
