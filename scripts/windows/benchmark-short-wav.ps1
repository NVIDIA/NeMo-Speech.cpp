# SPDX-License-Identifier: Apache-2.0
<#[
.SYNOPSIS
    Benchmark HTTP plus inference for the same short English WAV.

.DESCRIPTION
    Start either the default or custom server first. This script never starts, stops, or swaps a
    server; -Runtime records the selected runtime in the result metadata.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [ValidateSet('default', 'custom')] [string]$Runtime,
    [ValidateRange(1, 10000)] [int]$Runs = 20,
    [string]$Url = 'http://127.0.0.1:8081/v1/audio/transcriptions',
    [string]$Audio
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Audio) { $Audio = Join-Path $RepoRoot 'test_files\fork\asr\short-en.wav' }
if (-not (Test-Path -LiteralPath $Audio -PathType Leaf)) {
    throw "Short WAV fixture not found: $Audio. It is intentionally pending a reviewed redistribution license; see test_files\fork\asr\README.md."
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe is required but was not found on PATH.' }

function Get-WavDurationSeconds([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $reader = [System.IO.BinaryReader]::new($stream)
    try {
        if ([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'RIFF') { throw 'Expected a RIFF WAV file.' }
        $null = $reader.ReadUInt32()
        if ([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'WAVE') { throw 'Expected a WAVE file.' }
        $byteRate = 0
        while ($stream.Position -lt $stream.Length) {
            $chunk = [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $size = [int64]$reader.ReadUInt32()
            if ($chunk -eq 'fmt ') {
                if ($size -lt 16) { throw 'Invalid fmt chunk.' }
                $format = $reader.ReadUInt16(); $channels = $reader.ReadUInt16(); $sampleRate = $reader.ReadUInt32(); $byteRate = $reader.ReadUInt32()
                $null = $reader.ReadUInt16(); $bits = $reader.ReadUInt16()
                if ($format -ne 1 -or $channels -ne 1 -or $sampleRate -ne 16000 -or $bits -ne 16) { throw 'Expected PCM16 mono 16 kHz WAV.' }
                $stream.Position += $size - 16 + ($size % 2)
            } elseif ($chunk -eq 'data') {
                if ($byteRate -le 0) { throw 'WAV fmt chunk was missing.' }
                return $size / [double]$byteRate
            } else {
                $stream.Position += $size + ($size % 2)
            }
        }
        throw 'WAV data chunk was not found.'
    } finally { $reader.Dispose(); $stream.Dispose() }
}
function Get-Percentile([double[]]$Values, [double]$Percentile) {
    $ordered = @($Values | Sort-Object); $index = ($ordered.Count - 1) * $Percentile
    $lower = [Math]::Floor($index); $upper = [Math]::Ceiling($index)
    if ($lower -eq $upper) { return $ordered[$lower] }
    return $ordered[$lower] + (($ordered[$upper] - $ordered[$lower]) * ($index - $lower))
}
function Invoke-HttpInference([string]$RequestUrl, [string]$WavPath) {
    $response = New-TemporaryFile
    try {
        $metric = & curl.exe --silent --show-error --output $response --write-out '%{http_code}|%{time_total}' --form "file=@$WavPath;type=audio/wav" --form 'model=default' --form 'response_format=verbose_json' $RequestUrl
        $exitCode = $LASTEXITCODE
        $parts = (($metric | Out-String).Trim()) -split '\|', 2
        if ($exitCode -ne 0) { throw "curl.exe exited with $exitCode" }
        if ($parts.Count -ne 2 -or [int]$parts[0] -lt 200 -or [int]$parts[0] -ge 300) { throw "HTTP request failed: $($parts -join '|')" }
        return [double]$parts[1] * 1000
    } finally { Remove-Item -LiteralPath $response -Force -ErrorAction SilentlyContinue }
}

$baseUrl = ([uri]$Url).GetLeftPart([System.UriPartial]::Authority)
Write-Host "Checking readiness: $baseUrl/ready" -ForegroundColor Cyan
& curl.exe --silent --show-error --fail "$baseUrl/ready" | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$durationSeconds = Get-WavDurationSeconds $Audio
Write-Host "Warm-up ($Runtime): $Url" -ForegroundColor Cyan
$warmupMs = Invoke-HttpInference $Url $Audio
Write-Host ("Warm-up completed in {0:N2} ms (excluded)." -f $warmupMs)
$measurements = [System.Collections.Generic.List[double]]::new()
for ($i = 1; $i -le $Runs; $i++) {
    $milliseconds = Invoke-HttpInference $Url $Audio
    $measurements.Add($milliseconds)
    $rtf = ($milliseconds / 1000) / $durationSeconds
    Write-Host ("Run {0}/{1}: {2:N2} ms | RTF {3:N4} | {4:N3}x realtime" -f $i, $Runs, $milliseconds, $rtf, (1 / $rtf))
}

$values = [double[]]$measurements.ToArray()
$stats = [ordered]@{ minimum_ms = [Math]::Round(($values | Measure-Object -Minimum).Minimum, 3); maximum_ms = [Math]::Round(($values | Measure-Object -Maximum).Maximum, 3); mean_ms = [Math]::Round(($values | Measure-Object -Average).Average, 3); median_ms = [Math]::Round((Get-Percentile $values 0.5), 3); p95_ms = [Math]::Round((Get-Percentile $values 0.95), 3) }
$medianRtf = ($stats.median_ms / 1000) / $durationSeconds
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'; $resultsDir = Join-Path $RepoRoot 'benchmark-results'; New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
$jsonPath = Join-Path $resultsDir "short-wav-$Runtime-$timestamp.json"; $markdownPath = Join-Path $resultsDir "short-wav-$Runtime-$timestamp.md"
$gpu = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
$commit = (& git -C $RepoRoot rev-parse HEAD 2>$null).Trim()
$result = [ordered]@{ timestamp = (Get-Date).ToString('o'); runtime = $Runtime; model = $Model; audio = $Audio; url = $Url; audio_duration_seconds = $durationSeconds; warmup_ms = $warmupMs; runs_ms = $values; statistics = $stats; median_rtf = $medianRtf; median_realtime_speed = (1 / $medianRtf); repository_commit = $commit; gpu = $gpu }
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $values | ForEach-Object -Begin { $number = 0 } -Process { $number++; "| $number | $([Math]::Round($_, 3)) |" }
@('# Short WAV HTTP benchmark', '', "- Runtime: $Runtime", "- Model: $Model", "- Audio: $Audio", "- Server: $Url", "- Commit: $commit", "- Audio duration: $([Math]::Round($durationSeconds, 6)) s", "- Warm-up excluded: $warmupMs ms", "- GPU: $($gpu -join '; ')", '', '| Minimum (ms) | Maximum (ms) | Mean (ms) | Median (ms) | P95 (ms) | Median RTF | Speed |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |', "| $($stats.minimum_ms) | $($stats.maximum_ms) | $($stats.mean_ms) | $($stats.median_ms) | $($stats.p95_ms) | $([Math]::Round($medianRtf, 4)) | $([Math]::Round((1 / $medianRtf), 3))x |", '', '## Runs', '', '| Run | HTTP + inference (ms) |', '| ---: | ---: |') + $rows | Set-Content -LiteralPath $markdownPath -Encoding utf8
Write-Host "Saved JSON: $jsonPath" -ForegroundColor Green
Write-Host "Saved Markdown: $markdownPath" -ForegroundColor Green
