# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$Version = "latest",
    [ValidateSet("stable", "nightly")][string]$Channel = "stable",
    [string]$Prefix = "$env:LOCALAPPDATA\Programs\NeMoSpeech",
    [ValidateSet("auto", "cpu", "cuda", "vulkan")][string]$Backend = "auto",
    [switch]$Source,
    [switch]$BinaryOnly,
    [switch]$NoModifyPath,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$releaseBase = if ($env:NEMO_SPEECH_RELEASE_BASE_URL) {
    $env:NEMO_SPEECH_RELEASE_BASE_URL.TrimEnd('/')
} else {
    ""
}
$sourceUrl = if ($env:NEMO_SPEECH_SOURCE_URL) {
    $env:NEMO_SPEECH_SOURCE_URL
} else {
    (Get-Location).Path
}
if ($Source -and $BinaryOnly) { throw "-Source and -BinaryOnly are mutually exclusive" }

function Assert-SourcePrerequisites {
    param([string]$SelectedBackend, [string]$Architecture)

    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath;$env:Path"

    $installHint = "Install Git, CMake 3.26+, and Ninja with winget, then install Visual Studio 2022 Build Tools with the Desktop development with C++ workload."
    foreach ($tool in @("git", "cmake", "ninja")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool is required to build from source. $installHint Use -BinaryOnly to disable source fallback."
        }
    }

    $cmakeLine = (& cmake --version | Select-Object -First 1)
    if ($cmakeLine -notmatch 'cmake version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)' -or
        [version]$Matches[1] -lt [version]'3.26') {
        throw "CMake 3.26 or newer is required; found '$cmakeLine'. $installHint"
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio 2022 Build Tools were not found. Install the Desktop development with C++ workload."
    }
    $toolset = if ($Architecture -eq 'aarch64') {
        'Microsoft.VisualStudio.Component.VC.Tools.ARM64'
    } else {
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    }
    $vsPath = & $vswhere -latest -products * -requires $toolset -property installationPath
    if (-not $vsPath) {
        throw "Visual Studio 2022 is missing the C++ toolset for $Architecture. Modify the installation and add the Desktop development with C++ workload and the $Architecture build tools."
    }

    if ($SelectedBackend -eq 'cuda' -and -not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
        throw "The CUDA compiler (nvcc.exe) was not found. Install a supported NVIDIA CUDA Toolkit, or select -Backend cpu."
    }
    if ($SelectedBackend -eq 'vulkan') {
        $vulkanSdk = if ($env:VULKAN_SDK) {
            $env:VULKAN_SDK
        } else {
            [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
        }
        if (-not $vulkanSdk -or -not (Test-Path (Join-Path $vulkanSdk 'Bin\glslc.exe'))) {
            throw "The Vulkan SDK (including glslc and SPIR-V headers) was not found. Install the LunarG Vulkan SDK, or select -Backend cpu."
        }
    }
    if ($Architecture -eq 'aarch64') {
        $clangCandidates = @(
            (Join-Path $vsPath 'VC\Tools\Llvm\ARM64\bin\clang-cl.exe'),
            (Join-Path $vsPath 'VC\Tools\Llvm\x64\bin\clang-cl.exe')
        )
        $hasClang = (Get-Command clang-cl.exe -ErrorAction SilentlyContinue) -or
            @($clangCandidates | Where-Object { Test-Path $_ }).Count -gt 0
        if (-not $hasClang) {
            throw "Windows ARM64 builds require clang-cl. Install LLVM or the Visual Studio C++ Clang tools component."
        }
    }
}

$arch = switch ([System.Runtime.InteropServices.RuntimeInformation,mscorlib]::OSArchitecture) {
    "X64" { "x86_64" }
    "Arm64" { "aarch64" }
    default { throw "Unsupported Windows architecture: $_" }
}
if ($Backend -eq "auto") {
    $Backend = if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
}
$binaryCandidate = -not $Source -and [bool]$releaseBase
if ($BinaryOnly -and -not $releaseBase) {
    throw "-BinaryOnly requires NEMO_SPEECH_RELEASE_BASE_URL"
}
if ($Version -eq "latest") {
    if ($Channel -eq "nightly") {
        $Version = "nightly"
    } elseif ($Source) {
        $Version = "source"
    } elseif (-not $binaryCandidate) {
        $Version = "source"
        Write-Host "No release endpoint is configured; building from the current source branch."
    } else {
        try {
            $response = Invoke-WebRequest -Uri "$releaseBase/latest" -MaximumRedirection 10
            $location = if ($response.BaseResponse.RequestMessage) {
                $response.BaseResponse.RequestMessage.RequestUri.AbsoluteUri
            } else {
                $response.BaseResponse.ResponseUri.AbsoluteUri
            }
            $Version = ($location.TrimEnd('/') -split '/')[-1]
        } catch {
            if ($BinaryOnly) { throw "Could not resolve the latest release. $($_.Exception.Message)" }
            $binaryCandidate = $false
            $Version = "source"
            Write-Host "Latest release is unavailable; falling back to the main source branch."
        }
    }
}
$tag = if ($Version -eq "nightly") { "nightly" } elseif ($Version.StartsWith('v')) { $Version } else { "v$Version" }
$releaseVersion = $Version.TrimStart('v')
$archive = "nemo-speech-$releaseVersion-windows-$arch-$Backend.zip"
$url = "$releaseBase/download/$tag/$archive"
$sourceRef = if ($env:NEMO_SPEECH_SOURCE_REF) {
    $env:NEMO_SPEECH_SOURCE_REF
} elseif ($releaseVersion -eq 'source' -and
          (Get-Command git -ErrorAction SilentlyContinue) -and
          (Test-Path (Join-Path $sourceUrl '.git'))) {
    $localSourceRef = (& git -C $sourceUrl symbolic-ref --quiet --short HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $localSourceRef) { $localSourceRef } else { 'main' }
} elseif ($releaseVersion -in @("nightly", "source")) {
    "main"
} else {
    $tag
}

Write-Host "NeMo-Speech.cpp $releaseVersion (windows/$arch, $Backend)"
if (-not $Source -and $binaryCandidate) { Write-Host "Artifact: $url" }
if (-not $BinaryOnly) { Write-Host "Source:   $sourceUrl#$sourceRef ($Backend speech server fallback)" }
Write-Host "Prefix:   $Prefix"
if ($DryRun) { return }

$installIdentity = "$releaseVersion windows $arch $Backend"
$sourceIdentity = "$installIdentity source:$sourceRef profile:speech-server"
$installMetadata = Join-Path $Prefix ".nemo-speech-install"
$installedBinary = Join-Path $Prefix "bin\nemo-speech.exe"
if (-not $Source -and $binaryCandidate -and (Test-Path $installedBinary) -and (Test-Path $installMetadata) -and
    ((Get-Content $installMetadata -Raw).Trim() -eq $installIdentity)) {
    Write-Host "Already installed."
    & $installedBinary --version
    return
}

$temp = Join-Path ([IO.Path]::GetTempPath()) ("nemo-speech-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $archivePath = Join-Path $temp $archive
    $binaryReady = $false
    if (-not $Source -and $binaryCandidate) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $archivePath
            Invoke-WebRequest -Uri "$url.sha256" -OutFile "$archivePath.sha256"
            $binaryReady = $true
        } catch {
            if ($BinaryOnly) { throw "Release artifact or checksum is unavailable. $($_.Exception.Message)" }
            Remove-Item -Force -ErrorAction SilentlyContinue $archivePath, "$archivePath.sha256"
            Write-Host "Release artifact is unavailable; building from source."
        }
    }

    if ($binaryReady) {
        $expected = ((Get-Content "$archivePath.sha256" -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        if ($expected -notmatch '^[0-9a-f]{64}$') { throw "Release checksum is not a SHA-256 digest" }
        $actual = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "SHA-256 verification failed" }
        $extract = Join-Path $temp extract
        Expand-Archive -Path $archivePath -DestinationPath $extract
        $entries = @(Get-ChildItem -Force $extract)
        $root = if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) { $entries[0].FullName } else { $extract }
    } else {
        if ((Test-Path $installedBinary) -and (Test-Path $installMetadata) -and
            ((Get-Content $installMetadata -Raw).Trim() -eq $sourceIdentity)) {
            Write-Host "Already installed from source."
            & $installedBinary --version
            return
        }
        Assert-SourcePrerequisites -SelectedBackend $Backend -Architecture $arch

        $sourceDir = Join-Path $temp source
        Write-Host "Cloning $sourceUrl at $sourceRef..."
        & git clone --depth 1 --single-branch --branch $sourceRef $sourceUrl $sourceDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)" }

        if (Test-Path (Join-Path $sourceDir ".gitmodules")) {
            foreach ($submodule in @("ggml", "third_party/cpp-httplib")) {
                $configuredPath = & git -C $sourceDir config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>$null |
                    ForEach-Object { ($_ -split '\s+', 2)[1] } |
                    Where-Object { $_ -eq $submodule }
                if ($configuredPath) {
                    & git -C $sourceDir submodule update --init $submodule
                    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed for $submodule" }
                }
            }
        }
        $buildDir = Join-Path $temp "b"
        $buildScript = Join-Path $sourceDir "scripts\windows\build.ps1"
        & $buildScript -Backend $Backend -BuildDir $buildDir -Config Release -Http
        if ($LASTEXITCODE -ne 0) { throw "Source build failed ($LASTEXITCODE)" }
        $root = Join-Path $temp source-install
        & cmake --install $buildDir --config Release --prefix $root
        if ($LASTEXITCODE -ne 0) { throw "Source install failed ($LASTEXITCODE)" }
        $installIdentity = $sourceIdentity
    }
    if (-not (Test-Path (Join-Path $root "bin\nemo-speech.exe"))) {
        throw "Installation does not contain bin\nemo-speech.exe"
    }
    Set-Content -Path (Join-Path $root ".nemo-speech-install") -Value $installIdentity -NoNewline
    $parent = Split-Path -Parent $Prefix
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $next = "$Prefix.new"
    $old = "$Prefix.old"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $next, $old
    Move-Item $root $next
    if (Test-Path $Prefix) { Move-Item $Prefix $old }
    try {
        Move-Item $next $Prefix
    } catch {
        if (Test-Path $old) { Move-Item $old $Prefix }
        throw "Could not activate the new installation; the previous version was restored. $($_.Exception.Message)"
    }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $old

    $bin = Join-Path $Prefix bin
    if (-not $NoModifyPath) {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $parts = @($userPath -split ';' | Where-Object { $_ })
        if ($parts -notcontains $bin) {
            [Environment]::SetEnvironmentVariable("Path", (($parts + $bin) -join ';'), "User")
            Write-Host "Added $bin to the user PATH; open a new terminal to use it."
        }
    }
    & (Join-Path $bin "nemo-speech.exe") --version
    Write-Host "Next: download a model, then run 'nemo-speech transcribe' or 'nemo-speech serve' (see README.md)."
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $temp
}
