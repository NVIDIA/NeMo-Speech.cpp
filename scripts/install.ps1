# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$Version = "latest",
    [ValidateSet("stable", "nightly")][string]$Channel = "stable",
    [string]$Prefix = "$env:LOCALAPPDATA\Programs\NeMoSpeech",
    [ValidateSet("auto", "cpu", "cuda", "vulkan")][string]$Backend = "auto",
    [ValidateSet("core", "asr", "server", "full")][string]$Profile = "server",
    [switch]$Grpc,
    [switch]$Nmt,
    [switch]$Flashlight,
    [switch]$TtsJa,
    [switch]$TtsZh,
    [switch]$Http,
    [switch]$HttpTls,
    [string]$CudaArch = "native",
    [string]$VcpkgRoot,
    [string]$VcpkgTriplet,
    [switch]$Source,
    [switch]$BinaryOnly,
    [switch]$NoModifyPath,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$releaseBase = if ($env:NEMO_SPEECH_RELEASE_BASE_URL) {
    $env:NEMO_SPEECH_RELEASE_BASE_URL.TrimEnd('/')
} else {
    "https://github.com/NVIDIA/NeMo-Speech.cpp/releases"
}
$sourceUrl = if ($env:NEMO_SPEECH_SOURCE_URL) {
    $env:NEMO_SPEECH_SOURCE_URL
} else {
    "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
}
$versionUrl = if ($env:NEMO_SPEECH_VERSION_URL) {
    $env:NEMO_SPEECH_VERSION_URL
} else {
    "https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/VERSION"
}
if ($Source -and $BinaryOnly) { throw "-Source and -BinaryOnly are mutually exclusive" }
$profileIncludesTts = $Profile -ne 'asr'
if ($Grpc -and -not $profileIncludesTts) {
    throw 'The gRPC server requires both ASR and TTS. Use -Profile server or full.'
}
if (($TtsJa -or $TtsZh) -and -not $profileIncludesTts) {
    throw 'The Japanese and Mandarin tokenizers require TTS. Use a profile that includes TTS.'
}

function Invoke-DownloadWithRetry {
    param([string]$Uri, [string]$OutFile)

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile
            return
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds $attempt
        }
    }
}

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
            $machineVulkan = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
            if ($machineVulkan) { $machineVulkan } else { [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'User') }
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
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$env:Path = "$machinePath;$userPath;$env:Path"
if ($Backend -eq "auto") {
    $Backend = "cpu"
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        $gpu = & $smi.Source --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) {
            $Backend = "cuda"
        } else {
            Write-Host 'nvidia-smi is present but the NVIDIA driver is not working; selecting the CPU backend.'
        }
    }
}
$customSourceProfile = $Profile -ne 'server' -or
    $Grpc -or $Nmt -or $Flashlight -or $TtsJa -or $TtsZh -or
    $HttpTls
if ($BinaryOnly -and $customSourceProfile) {
    throw '-BinaryOnly supports only the server profile without extra components.'
}
$binaryCandidate = -not $Source -and -not $customSourceProfile -and [bool]$releaseBase
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
            $manifest = (Invoke-WebRequest -UseBasicParsing -Uri $versionUrl).Content
            if ($manifest -notmatch '(?m)^NEMO_SPEECH_VERSION:\s*([^\s]+)\s*$') {
                throw "VERSION does not contain NEMO_SPEECH_VERSION"
            }
            $Version = $Matches[1]
        } catch {
            if ($BinaryOnly) { throw "Could not resolve the latest release. $($_.Exception.Message)" }
            $binaryCandidate = $false
            $Version = "source"
            Write-Host "Latest release is unavailable ($($_.Exception.Message)); falling back to the main source branch."
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
          ($sourceUrl -notmatch '^[A-Za-z][A-Za-z0-9+.-]*://') -and
          (Test-Path -LiteralPath $sourceUrl -PathType Container) -and
          (Test-Path -LiteralPath (Join-Path $sourceUrl '.git'))) {
    $localSourceRef = (& git -C $sourceUrl symbolic-ref --quiet --short HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $localSourceRef) { $localSourceRef } else { 'main' }
} elseif ($releaseVersion -in @("nightly", "source")) {
    "main"
} else {
    $tag
}

Write-Host "NeMo-Speech.cpp $releaseVersion (windows/$arch, $Backend, $Profile)"
if (-not $Source -and $binaryCandidate) { Write-Host "Artifact: $url" }
if (-not $BinaryOnly) { Write-Host "Source:   $sourceUrl#$sourceRef ($Backend/$Profile)" }
Write-Host "Prefix:   $Prefix"
if ($DryRun) { return }
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "curl.exe was not found. Installation can continue, but automatic model downloads will be unavailable; local and already cached models still work. Windows 10 and 11 normally include curl.exe in %SystemRoot%\\System32."
}

$installIdentity = "$releaseVersion windows $arch $Backend"
$extraComponents = [Collections.Generic.List[string]]::new()
foreach ($component in @(
    @{ Name = 'grpc'; Enabled = $Grpc }, @{ Name = 'nmt'; Enabled = $Nmt },
    @{ Name = 'flashlight'; Enabled = $Flashlight }, @{ Name = 'tts-ja'; Enabled = $TtsJa },
    @{ Name = 'tts-zh'; Enabled = $TtsZh }, @{ Name = 'http'; Enabled = $Http },
    @{ Name = 'http-tls'; Enabled = $HttpTls }
)) {
    if ($component.Enabled) { $extraComponents.Add($component.Name) }
}
$componentIdentity = if ($extraComponents.Count) { " components:$($extraComponents -join ',')" } else { '' }
$sourceIdentity = "$installIdentity source:$sourceRef profile:$Profile$componentIdentity"
$installMetadata = Join-Path $Prefix ".nemo-speech-install"
$installedBinary = Join-Path $Prefix "bin\nemo-speech.exe"
if (-not $Source -and $binaryCandidate -and (Test-Path $installedBinary) -and (Test-Path $installMetadata) -and
    ((Get-Content $installMetadata -Raw).Trim() -eq $installIdentity)) {
    Write-Host "Already installed."
    & $installedBinary --version
    return
}

$temp = Join-Path ([IO.Path]::GetTempPath()) ("nsi-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $archivePath = Join-Path $temp $archive
    $binaryReady = $false
    if (-not $Source -and $binaryCandidate) {
        try {
            Invoke-DownloadWithRetry -Uri $url -OutFile $archivePath
            Invoke-DownloadWithRetry -Uri "$url.sha256" -OutFile "$archivePath.sha256"
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

        $buildDir = Join-Path $temp "b"
        $buildScript = Join-Path $sourceDir "scripts\windows\build.ps1"
        $buildParameters = @{
            Backend = $Backend
            Profile = $Profile
            BuildDir = $buildDir
            Config = 'Release'
            CudaArch = $CudaArch
        }
        foreach ($switchParameter in @(
            @{ Name = 'Grpc'; Enabled = $Grpc }, @{ Name = 'Nmt'; Enabled = $Nmt },
            @{ Name = 'Flashlight'; Enabled = $Flashlight }, @{ Name = 'TtsJa'; Enabled = $TtsJa },
            @{ Name = 'TtsZh'; Enabled = $TtsZh }, @{ Name = 'Http'; Enabled = $Http },
            @{ Name = 'HttpTls'; Enabled = $HttpTls }
        )) {
            if ($switchParameter.Enabled) { $buildParameters[$switchParameter.Name] = $true }
        }
        if ($VcpkgRoot) { $buildParameters.VcpkgRoot = $VcpkgRoot }
        if ($VcpkgTriplet) { $buildParameters.VcpkgTriplet = $VcpkgTriplet }
        & $buildScript @buildParameters
        if ($LASTEXITCODE -ne 0) { throw "Source build failed ($LASTEXITCODE)" }
        $root = Join-Path $temp source-install
        & cmake --install $buildDir --config Release --prefix $root
        if ($LASTEXITCODE -ne 0) { throw "Source install failed ($LASTEXITCODE)" }
        $installIdentity = $sourceIdentity
    }
    $stagedBinary = Join-Path $root "bin\nemo-speech.exe"
    if (-not (Test-Path $stagedBinary)) {
        throw "Installation does not contain bin\nemo-speech.exe"
    }
    & $stagedBinary --version
    if ($LASTEXITCODE -ne 0) { throw "Installed binary failed its version check ($LASTEXITCODE)" }
    & $stagedBinary --json doctor
    if ($LASTEXITCODE -ne 0) { throw "Installed binary failed its runtime health check ($LASTEXITCODE)" }
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
