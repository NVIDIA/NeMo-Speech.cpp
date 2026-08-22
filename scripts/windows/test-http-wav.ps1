# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$Url = 'http://127.0.0.1:8081/v1/audio/transcriptions',
    [ValidateRange(1, 600)] [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Wav = Join-Path $RepoRoot 'test_files\fork\asr\teste-en.wav'
$ExpectedPath = Join-Path $RepoRoot 'test_files\fork\asr\teste-en.txt'
foreach ($item in @(@{ Name = 'WAV fixture'; Path = $Wav }, @{ Name = 'Expected transcript'; Path = $ExpectedPath })) {
    if (-not (Test-Path -LiteralPath $item.Path -PathType Leaf)) { throw "$($item.Name) not found: $($item.Path)" }
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe is required but was not found on PATH.' }
$baseUrl = ([uri]$Url).GetLeftPart([System.UriPartial]::Authority)
$ready = "$baseUrl/ready"
Write-Host "Checking readiness: $ready" -ForegroundColor Cyan
& curl.exe --silent --show-error --fail --max-time $TimeoutSeconds $ready | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$responseFile = New-TemporaryFile
try {
    $metric = & curl.exe --silent --show-error --output $responseFile --write-out '%{http_code}|%{time_total}' --max-time $TimeoutSeconds --form "file=@$Wav;type=audio/wav" --form 'model=default' --form 'response_format=verbose_json' $Url
    $exitCode = $LASTEXITCODE
    $parts = (($metric | Out-String).Trim()) -split '\|', 2
    if ($exitCode -ne 0) { exit $exitCode }
    if ($parts.Count -ne 2 -or [int]$parts[0] -lt 200 -or [int]$parts[0] -ge 300) { throw "HTTP request failed (curl result: $($parts -join '|')). Response: $(Get-Content -LiteralPath $responseFile -Raw)" }
    $raw = Get-Content -LiteralPath $responseFile -Raw
    try { $payload = $raw | ConvertFrom-Json } catch { throw "Server response was not JSON: $raw" }
    $payload | ConvertTo-Json -Depth 20
    $expected = (Get-Content -LiteralPath $ExpectedPath -Raw).Trim(); $actual = ([string]$payload.text).Trim()
    Write-Host ("`nHTTP + inference: {0:N2} ms" -f (([double]$parts[1]) * 1000)) -ForegroundColor Green
    Write-Host "Expected: $expected"; Write-Host "Received: $actual"
    if ($actual -eq $expected) { Write-Host 'Transcript comparison: exact match.' -ForegroundColor Green } else { Write-Host 'Transcript comparison: visually review the expected and received text above.' -ForegroundColor Yellow }
}
finally { Remove-Item -LiteralPath $responseFile -Force -ErrorAction SilentlyContinue }
