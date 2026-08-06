# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [ValidateRange(1, 10000)] [int]$Runs = 10,
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
$commandText = '& "{0}" {1}' -f $Executable, (($arguments | ForEach-Object { '"{0}"' -f $_ }) -join ' ')
function Invoke-Measurement {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $null = & $Executable @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $watch.Stop()
    if ($exitCode -ne 0) { throw "nemo-speech exited with $exitCode" }
    return [Math]::Round($watch.Elapsed.TotalMilliseconds, 3)
}
function Get-Percentile([double[]]$Values, [double]$Percentile) {
    $ordered = @($Values | Sort-Object); $index = ($ordered.Count - 1) * $Percentile
    $lower = [Math]::Floor($index); $upper = [Math]::Ceiling($index)
    if ($lower -eq $upper) { return $ordered[$lower] }
    return $ordered[$lower] + (($ordered[$upper] - $ordered[$lower]) * ($index - $lower))
}

Write-Host "Warm-up: $commandText" -ForegroundColor Cyan
$warmupMs = Invoke-Measurement
Write-Host ("Warm-up completed in {0:N3} ms (excluded)." -f $warmupMs)
$measurements = [System.Collections.Generic.List[double]]::new()
for ($i = 1; $i -le $Runs; $i++) { $elapsed = Invoke-Measurement; $measurements.Add($elapsed); Write-Host ("Run {0}/{1}: {2:N3} ms" -f $i, $Runs, $elapsed) }

$values = [double[]]$measurements.ToArray(); $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$resultsDir = Join-Path $RepoRoot 'benchmark-results'; New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
$jsonPath = Join-Path $resultsDir "pascal-wav-$timestamp.json"; $markdownPath = Join-Path $resultsDir "pascal-wav-$timestamp.md"
$gpu = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
$cpu = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Name)
$commit = (& git -C $RepoRoot rev-parse HEAD 2>$null).Trim()
$stats = [ordered]@{ minimum_ms = [Math]::Round(($values | Measure-Object -Minimum).Minimum, 3); maximum_ms = [Math]::Round(($values | Measure-Object -Maximum).Maximum, 3); mean_ms = [Math]::Round(($values | Measure-Object -Average).Average, 3); median_ms = [Math]::Round((Get-Percentile $values 0.5), 3); p95_ms = [Math]::Round((Get-Percentile $values 0.95), 3) }
$result = [ordered]@{ timestamp = (Get-Date).ToString('o'); repository_commit = $commit; model = $Model; wav = $Wav; executable = $Executable; device = $Device; command = $commandText; warmup_ms = $warmupMs; runs_ms = $values; statistics = $stats; hardware = [ordered]@{ cpu = $cpu; gpu = $gpu; os = [Environment]::OSVersion.VersionString } }
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $values | ForEach-Object -Begin { $number = 0 } -Process { $number++; "| $number | $([Math]::Round($_, 3)) |" }
@('# Pascal WAV benchmark', '', "- Timestamp: $($result.timestamp)", "- Commit: $commit", "- Model: ``$Model``", "- Device: ``$Device``", "- WAV: ``$Wav``", "- Command: ``$commandText``", "- Warm-up excluded: $warmupMs ms", "- CPU: $cpu", "- GPU: $($gpu -join '; ')", '', '## Summary', '', '| Minimum (ms) | Maximum (ms) | Mean (ms) | Median (ms) | P95 (ms) |', '| ---: | ---: | ---: | ---: | ---: |', "| $($stats.minimum_ms) | $($stats.maximum_ms) | $($stats.mean_ms) | $($stats.median_ms) | $($stats.p95_ms) |", '', '## Runs', '', '| Run | Wall time (ms) |', '| ---: | ---:|') + $rows | Set-Content -LiteralPath $markdownPath -Encoding utf8
Write-Host "Saved JSON: $jsonPath" -ForegroundColor Green
Write-Host "Saved Markdown: $markdownPath" -ForegroundColor Green
