# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [string]$Executable,
    [ValidatePattern('^(cpu|cuda(:[0-9]+)?)$')] [string]$Device = 'cuda:0'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Wav = Join-Path $RepoRoot 'test_files\fork\asr\teste-en.wav'
if (-not $Executable) { $Executable = Join-Path $RepoRoot 'build-pascal-cuda-http\bin\nemo-speech.exe' }
foreach ($item in @(@{ Name = 'Model'; Path = $Model }, @{ Name = 'WAV fixture'; Path = $Wav }, @{ Name = 'Executable'; Path = $Executable })) {
    if (-not (Test-Path -LiteralPath $item.Path -PathType Leaf)) { throw "$($item.Name) not found: $($item.Path)" }
}
$arguments = @('transcribe', $Wav, '--model', $Model, '--device', $Device, '--skinny-q8', 'auto', '--suppress-cuda-graph-log', '--format', 'json')
Write-Host 'Running:' -ForegroundColor Cyan
Write-Host ('& "{0}" {1}' -f $Executable, (($arguments | ForEach-Object { '"{0}"' -f $_ }) -join ' '))
& $Executable @arguments
exit $LASTEXITCODE
