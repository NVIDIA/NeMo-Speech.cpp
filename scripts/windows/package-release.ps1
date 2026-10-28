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
.PARAMETER Architecture
    Architecture the build targets: auto (the host), x64, or arm64. Set it for a
    cross-compiled build; every packaged binary must match it.
.EXAMPLE
    pwsh scripts\windows\package-release.ps1 -BuildDir build\release-cuda -Backend cuda
    pwsh scripts\windows\package-release.ps1 -BuildDir build\release-cuda-arm64 -Backend cuda -Architecture arm64
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BuildDir,
    [Parameter(Mandatory)]
    [ValidateSet('cpu', 'vulkan', 'cuda')]
    [string]$Backend,
    [string]$Version,
    [string]$OutputDir,
    [ValidateSet('auto', 'x64', 'arm64')]
    [string]$Architecture = 'auto'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot 'release-artifacts' }
if (-not $Version) {
    # @() keeps -match filtering lines: on a single-line file it would return a bool.
    $Version = (@(Get-Content (Join-Path $RepoRoot 'VERSION')) -match '^NEMO_SPEECH_VERSION:' |
        Select-Object -First 1) -replace '^NEMO_SPEECH_VERSION:\s*', ''
}
if ($Version -ne 'nightly' -and $Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$') {
    throw "invalid release version '$Version'"
}
if ($Architecture -eq 'auto') {
    $Architecture = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
        'X64' { 'x64' }
        'Arm64' { 'arm64' }
        default { throw "unsupported architecture: $_" }
    }
}
$arch = if ($Architecture -eq 'arm64') { 'aarch64' } else { 'x86_64' }

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
    # A -CublasShim build installs its cublas64_*.dll; otherwise the package carries
    # NVIDIA's cuBLAS and cuBLASLt for the target architecture, and the CUDA EULA.
    $cublasShim = $false
    if ($Backend -eq 'cuda') {
        if (-not (Get-ChildItem $bin -Filter 'ggml-cuda.dll')) { throw 'CUDA package is missing ggml-cuda.dll' }
        $cublasShim = [bool](Get-ChildItem $bin -Filter 'cublas64_*.dll')
        if (-not $cublasShim) {
            $cache = Get-Content (Join-Path $BuildDir 'CMakeCache.txt')
            $toolkitBin = ($cache | Where-Object { $_ -match '^CUDAToolkit_BIN_DIR:PATH=(.+)$' } |
                ForEach-Object { $Matches[1] } | Select-Object -First 1)
            if (-not $toolkitBin) { throw 'CUDAToolkit_BIN_DIR is not in the build cache' }
            $toolkit = Split-Path -Parent $toolkitBin
            $cublasDir = @("$toolkit\bin\$Architecture", "$toolkit\bin") |
                Where-Object { Get-ChildItem $_ -Filter 'cublasLt64_*.dll' -ErrorAction SilentlyContinue } |
                Select-Object -First 1
            if (-not $cublasDir) { throw "no cuBLAS DLLs for $Architecture under $toolkit\bin" }
            Copy-Item (Join-Path $cublasDir 'cublas64_*.dll'), (Join-Path $cublasDir 'cublasLt64_*.dll') $bin
            $eula = Join-Path $root 'share\licenses\nemo-speech\nvidia\cuda-runtime'
            New-Item -ItemType Directory -Force -Path $eula | Out-Null
            Copy-Item (Join-Path $toolkit 'EULA.txt') $eula
            Write-Host "Bundled cuBLAS and cuBLASLt from $cublasDir"
        }
    }

    # Every imported DLL must be bundled or provided by Windows or the GPU driver.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $dumpbin = & $vswhere -latest -products * -find '**\Hostx64\x64\dumpbin.exe' | Select-Object -First 1
    if (-not $dumpbin) { throw 'dumpbin.exe was not found (Visual Studio C++ tools are required)' }
    # A failed inspection would otherwise read as "no dependencies".
    function Invoke-Dumpbin([string]$Option, [string]$Path) {
        $output = & $dumpbin /nologo $Option $Path
        if ($LASTEXITCODE -ne 0) { throw "dumpbin $Option failed for $Path ($LASTEXITCODE)" }
        $output
    }
    $system = '^(api-ms-win-.*|ext-ms-.*|kernel32|kernelbase|user32|gdi32|advapi32|shell32|ole32|oleaut32|' +
              'ws2_32|wsock32|mswsock|bcrypt|crypt32|ncrypt|secur32|dbghelp|shlwapi|winmm|ntdll|userenv|' +
              'psapi|version|iphlpapi|setupapi|cfgmgr32|comctl32|comdlg32|rpcrt4|dnsapi|powrprof|winhttp|' +
              'wininet|normaliz|avrt|ucrtbase|nvcuda|nvml|vulkan-1)\.dll$'
    $bundled = @{}
    Get-ChildItem $bin -Filter '*.dll' | ForEach-Object { $bundled[$_.Name.ToLowerInvariant()] = $true }
    $missing = @()
    foreach ($file in Get-ChildItem $bin -Include '*.dll', '*.exe' -Recurse) {
        $dependencies = Invoke-Dumpbin /dependents $file.FullName | ForEach-Object {
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

    # A cross-compiled package must not mix in host binaries. The ARM64 Visual C++
    # runtime's DLLs are ARM64X, which report AA64 (vcruntime140_1.dll, which reports
    # x64, is for x64 code and is not installed).
    $machine = if ($Architecture -eq 'arm64') { '^\s*AA64 machine' } else { '^\s*8664 machine \(x64\)\s*$' }
    $wrongMachine = foreach ($file in Get-ChildItem $bin -Include '*.dll', '*.exe' -Recurse) {
        $header = Invoke-Dumpbin /headers $file.FullName | Where-Object { $_ -match 'machine \(' } | Select-Object -First 1
        if ($header -notmatch $machine) { "$($file.Name): $($header.Trim())" }
    }
    if ($wrongMachine) {
        throw "binaries that do not target ${Architecture}:`n  $($wrongMachine -join "`n  ")"
    }

    # The cuBLAS shim implements only the calls ggml makes; a call added by a
    # llama.cpp update must be added to kernels\ before it can ship.
    if ($cublasShim) {
        $shim = (Get-ChildItem $bin -Filter 'cublas64_*.dll' | Select-Object -First 1)
        $exported = @{}
        Invoke-Dumpbin /exports $shim.FullName | ForEach-Object {
            if ($_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)') { $exported[$Matches[1]] = $true }
        }
        $dll = $null
        $unexported = @()
        Invoke-Dumpbin /imports (Join-Path $bin 'ggml-cuda.dll') | ForEach-Object {
            if ($_ -match '^\s{4}(\S+\.dll)\s*$') {
                $dll = $Matches[1]
            } elseif ($dll -ieq $shim.Name -and $_ -match '^\s+[0-9A-Fa-f]+\s+(\S+)\s*$' -and
                      -not $exported.ContainsKey($Matches[1])) {
                $unexported += $Matches[1]
            }
        }
        if (-not $exported.Count) { throw "dumpbin listed no exports for $($shim.Name)" }
        if ($unexported) {
            throw "the cuBLAS shim does not export these functions ggml-cuda.dll calls:`n  $($unexported -join "`n  ")"
        }
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
