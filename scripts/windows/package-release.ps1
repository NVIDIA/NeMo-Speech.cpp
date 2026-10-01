# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Package a Windows build for the binary installer.
.DESCRIPTION
    Installs -BuildDir into a directory named after the archive, checks that the
    package is self-contained, and writes
    nemo-speech-<version>-windows-<arch>-<backend>.zip and its .sha256 to
    -OutputDir. Every DLL a packaged binary imports must ship in bin\ or be a
    Windows or GPU driver library.
.PARAMETER BuildDir
    Build tree produced by scripts\windows\build.ps1.
.PARAMETER Backend
    cpu, vulkan, or cuda.
.PARAMETER Version
    Release version, or nightly. Defaults to the one in VERSION.
.PARAMETER OutputDir
    Destination for the archive and checksum (default: release-artifacts).
.EXAMPLE
    pwsh scripts\windows\package-release.ps1 -BuildDir build\release-cuda -Backend cuda
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BuildDir,
    [Parameter(Mandatory)]
    [ValidateSet('cpu', 'vulkan', 'cuda')]
    [string]$Backend,
    [string]$Version,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot 'release-artifacts' }
if (-not $Version) {
    $Version = ((Get-Content (Join-Path $RepoRoot 'VERSION')) -match '^NEMO_SPEECH_VERSION:' |
        Select-Object -First 1) -replace '^NEMO_SPEECH_VERSION:\s*', ''
}
if ($Version -ne 'nightly' -and $Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$') {
    throw "invalid release version '$Version'"
}
$arch = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
    'X64' { 'x86_64' }
    'Arm64' { 'aarch64' }
    default { throw "unsupported architecture: $_" }
}

$name = "nemo-speech-$Version-windows-$arch-$Backend"
$staging = Join-Path ([System.IO.Path]::GetTempPath()) "nemo-speech-package-$PID"
$root = Join-Path $staging $name
New-Item -ItemType Directory -Force -Path $root, $OutputDir | Out-Null
try {
    & cmake --install $BuildDir --config Release --prefix $root
    if ($LASTEXITCODE -ne 0) { throw "cmake --install failed ($LASTEXITCODE)" }

    $bin = Join-Path $root 'bin'
    foreach ($required in @('bin\nemo-speech.exe', 'share\licenses\nemo-speech\LICENSE',
                            'share\licenses\nemo-speech\THIRD_PARTY_NOTICES.md')) {
        if (-not (Test-Path (Join-Path $root $required))) { throw "package is missing $required" }
    }
    if (Get-ChildItem -Recurse $root -Include 'riva_server.exe', 'grpc*.dll') {
        throw 'release contains Riva gRPC payload'
    }
    if ($Backend -eq 'cuda') {
        if (-not (Get-ChildItem $bin -Filter 'ggml-cuda.dll')) { throw 'CUDA package is missing ggml-cuda.dll' }
        if (-not (Get-ChildItem $bin -Filter 'cublas64_*.dll')) {
            throw 'CUDA package is missing the cuBLAS shim; build with -CublasShim'
        }
    }

    # Every imported DLL must be bundled or provided by Windows or the GPU driver.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $dumpbin = & $vswhere -latest -products * -find '**\Hostx64\x64\dumpbin.exe' | Select-Object -First 1
    if (-not $dumpbin) { throw 'dumpbin.exe was not found (Visual Studio C++ tools are required)' }
    $system = '^(api-ms-win-.*|ext-ms-.*|kernel32|kernelbase|user32|gdi32|advapi32|shell32|ole32|oleaut32|' +
              'ws2_32|wsock32|mswsock|bcrypt|crypt32|ncrypt|secur32|dbghelp|shlwapi|winmm|ntdll|userenv|' +
              'psapi|version|iphlpapi|setupapi|cfgmgr32|comctl32|comdlg32|rpcrt4|dnsapi|powrprof|winhttp|' +
              'wininet|normaliz|avrt|ucrtbase|nvcuda|nvml|vulkan-1)\.dll$'
    $bundled = @{}
    Get-ChildItem $bin -Filter '*.dll' | ForEach-Object { $bundled[$_.Name.ToLowerInvariant()] = $true }
    $missing = @()
    foreach ($file in Get-ChildItem $bin -Include '*.dll', '*.exe' -Recurse) {
        $dependencies = & $dumpbin /nologo /dependents $file.FullName | ForEach-Object {
            if ($_ -match '^\s+(\S+\.dll)\s*$') { $Matches[1].ToLowerInvariant() }
        }
        foreach ($dependency in $dependencies) {
            if (-not $bundled.ContainsKey($dependency) -and $dependency -notmatch $system) {
                $missing += "$($file.Name): $dependency"
            }
        }
    }
    if ($missing) {
        throw "the package is not self-contained:`n  $($missing -join "`n  ")"
    }

    $zip = Join-Path (Resolve-Path $OutputDir) "$name.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $root, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText("$zip.sha256", "$hash  $name.zip`n")
    Write-Host "Created: $zip"
    Write-Host "SHA-256: $zip.sha256"
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $staging
}
