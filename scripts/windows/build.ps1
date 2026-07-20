# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Configure + build nemo-speech on Windows (MSVC + Ninja).

.DESCRIPTION
    One-stop build driver for Windows. It:
      1. Refreshes PATH/VULKAN_SDK from the registry (installers update the
         Machine/User scope, not an already-open shell).
      2. Imports the MSVC x64 dev environment (vcvars64.bat) so cl.exe / nvcc
         find the toolset. MSVC is required: nvcc on Windows only supports cl.exe
         as the CUDA host compiler.
      3. For a CUDA build, applies the CUDA-only ggml patches.
      4. Configures with CMake (Ninja) and builds.

.PARAMETER Backend
    cuda | vulkan | cpu. CUDA and Vulkan are separate build trees (different ggml
    config), so use a distinct -BuildDir for each.

.PARAMETER Grpc
    Also build the Riva-compatible riva_server. Requires grpc+protobuf from
    vcpkg; pass -VcpkgRoot (default C:\vcpkg). Install once with:
        C:\vcpkg\vcpkg.exe install grpc protobuf portaudio --triplet x64-windows

.PARAMETER Nmt
    Also build the NMT translation component (links llama.cpp; checks out the
    llama.cpp submodule if needed). Combine with -Grpc for the gRPC TranslateText
    service. Convert the model separately - see docs/development/windows-build.md.

.PARAMETER Flashlight
    Build LM-fused CTC decoding with Flashlight and dynamically linked KenLM
    DLLs. Requires SentencePiece from vcpkg.

.PARAMETER TtsJa
    Build the optional Japanese TTS tokenizer (Open JTalk, MeCab, and the NAIST
    dictionary).

.PARAMETER TtsZh
    Build the optional Mandarin TTS tokenizer (cppjieba and limonp).

.PARAMETER AsrOnly
    Build only the CLI, ASR, and diarization. Useful for a minimal ASR build.

.PARAMETER Http
    Build the HTTP API, realtime WebSocket endpoint, and browser playground.
    The default component set includes ASR, diarization, and TTS; combine with
    -AsrOnly only when TTS is not wanted.

.PARAMETER CudaArch
    CUDA architectures for ggml-cuda (default: native = the local GPU). Examples:
    "89" (Ada/RTX 40xx), "86" (Ampere/RTX 30xx), "120" (Blackwell). Set a concrete
    value (not native) when building to ship to other GPUs.

.PARAMETER Compiler
    C/C++ compiler: auto (default), msvc, or clang-cl. auto picks cl on x64 and
    clang-cl on ARM64 (ggml's ARM CPU backend rejects MSVC). nvcc always uses
    cl.exe as its CUDA host compiler regardless of this setting.

.PARAMETER Architecture
    Target architecture: auto (the host), x64, or arm64.

.EXAMPLE
    pwsh scripts\windows\build.ps1 -Backend cuda -Grpc
    pwsh scripts\windows\build.ps1 -Backend cpu -Flashlight
    pwsh scripts\windows\build.ps1 -Backend vulkan
#>
[CmdletBinding()]
param(
    [ValidateSet('cuda', 'vulkan', 'cpu')]
    [string]$Backend = 'cuda',
    [switch]$Grpc,
    [switch]$Nmt,
    [switch]$Flashlight,
    [switch]$TtsJa,
    [switch]$TtsZh,
    [switch]$AsrOnly,
    [switch]$Http,
    [string]$BuildDir,
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Config = 'Release',
    [string]$CudaArch = 'native',
    [string]$VcpkgRoot = 'C:\vcpkg',
    [string]$VcpkgTriplet,
    [ValidateSet('auto', 'x64', 'arm64')]
    [string]$Architecture = 'auto',
    # C/C++ compiler. 'auto' = cl on x64, clang-cl on ARM64 (ggml's ARM CPU
    # backend rejects MSVC). nvcc's CUDA host compiler is cl.exe in all cases.
    [ValidateSet('auto', 'msvc', 'clang-cl')]
    [string]$Compiler = 'auto',
    [int]$Jobs = 0
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $BuildDir) {
    $buildName = if ($Architecture -eq 'auto') { "build-$Backend" } else { "build-$Backend-$Architecture" }
    $BuildDir = Join-Path $RepoRoot $buildName
}
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

$HostArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$TargetArch = if ($Architecture -eq 'auto') {
    if ($HostArch -eq 'ARM64') { 'arm64' } else { 'x64' }
} else {
    $Architecture
}
if ($Compiler -eq 'auto') { $Compiler = if ($TargetArch -eq 'arm64') { 'clang-cl' } else { 'msvc' } }
if ($TargetArch -eq 'arm64' -and $Compiler -eq 'msvc') {
    throw "ggml's ARM CPU backend does not compile with MSVC; use -Compiler clang-cl on ARM64."
}
$CrossCompiling = ($HostArch -eq 'ARM64') -ne ($TargetArch -eq 'arm64')
if ($Backend -eq 'cuda' -and $CrossCompiling) {
    throw 'CUDA cross-compilation is not supported by this driver; build CUDA natively on the target architecture.'
}
$VcpkgArch = $TargetArch
if (-not $VcpkgTriplet) {
    $VcpkgTriplet = if ($Flashlight -or ($Grpc -and $Compiler -eq 'clang-cl')) {
        "$VcpkgArch-windows-static-md"
    } else {
        "$VcpkgArch-windows"
    }
}

Write-Host "==> nemo-speech Windows build" -ForegroundColor Cyan
Write-Host "    backend=$Backend grpc=$($Grpc.IsPresent) http=$($Http.IsPresent) nmt=$($Nmt.IsPresent) flashlight=$($Flashlight.IsPresent) tts-ja=$($TtsJa.IsPresent) tts-zh=$($TtsZh.IsPresent) config=$Config compiler=$Compiler host=$HostArch target=$TargetArch build=$BuildDir jobs=$Jobs"

# --- 1. Refresh environment from registry (choco/installers land there) ---------
$machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$user    = [Environment]::GetEnvironmentVariable('Path', 'User')
$env:Path = "$machine;$user"
$vk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
if ($vk) { $env:VULKAN_SDK = $vk }

# --- 2. Import the MSVC dev environment (arch-specific vcvars) ------------------
# cl.exe is always needed: it is nvcc's only supported CUDA host compiler on
# Windows, even when clang-cl compiles the C/C++.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found; install VS 2022 Build Tools (VC++ workload)." }
$vcToolset = if ($TargetArch -eq 'arm64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' }
             else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
$vsPath = & $vswhere -latest -products * -requires $vcToolset -property installationPath
if (-not $vsPath) { throw "No VS install with the C++ toolset for $TargetArch found." }
$vcvarsBat = if ($HostArch -eq 'ARM64' -and $TargetArch -eq 'x64') {
    'vcvarsarm64_x64.bat'
} elseif ($HostArch -ne 'ARM64' -and $TargetArch -eq 'arm64') {
    'vcvarsamd64_arm64.bat'
} elseif ($TargetArch -eq 'arm64') {
    'vcvarsarm64.bat'
} else {
    'vcvars64.bat'
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\$vcvarsBat"
if (-not (Test-Path $vcvars)) { throw "$vcvarsBat not found at $vcvars" }
Write-Host "==> importing MSVC env from $vsPath ($vcvarsBat)"
cmd /c "`"$vcvars`" >NUL 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

foreach ($tool in 'cl', 'cmake', 'ninja') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH after env setup." }
}
if ($Compiler -eq 'clang-cl' -and -not (Get-Command clang-cl -ErrorAction SilentlyContinue)) {
    throw "clang-cl not on PATH; install LLVM (choco install llvm) or the VS 'C++ Clang tools' component."
}
if ($Backend -eq 'cuda'   -and -not (Get-Command nvcc  -ErrorAction SilentlyContinue)) { throw "nvcc not found; install the CUDA Toolkit." }
if ($Backend -eq 'vulkan' -and -not $env:VULKAN_SDK) { throw "VULKAN_SDK not set; install the Vulkan SDK." }

# --- 3. CUDA-only: apply the ggml patches ---------------------------------------
if ($Backend -eq 'cuda') {
    Write-Host "==> applying ggml patches (CUDA)"
    & (Join-Path $PSScriptRoot 'apply-ggml-patches.ps1')
}

# --- 3b. NMT: ensure the llama.cpp submodule is checked out ---------------------
if ($Nmt -and -not (Test-Path (Join-Path $RepoRoot 'llama.cpp\CMakeLists.txt'))) {
    Write-Host "==> initializing llama.cpp submodule (NMT)"
    & git -C $RepoRoot submodule update --init llama.cpp
    if ($LASTEXITCODE -ne 0) { throw "git submodule update --init llama.cpp failed" }
}

# --- 3c. HTTP: ensure cpp-httplib is checked out -------------------------------
if ($Http -and -not (Test-Path (Join-Path $RepoRoot 'third_party\cpp-httplib\httplib.h'))) {
    Write-Host "==> initializing third_party/cpp-httplib (HTTP and playground)"
    & git -C $RepoRoot submodule update --init third_party/cpp-httplib
    if ($LASTEXITCODE -ne 0) { throw "git submodule update --init third_party/cpp-httplib failed" }
}

# --- 3d. Flashlight: ensure the decoder and KenLM sources are checked out -------
if ($Flashlight) {
    foreach ($submodule in 'third_party/flashlight-text', 'third_party/kenlm') {
        if (-not (Test-Path (Join-Path $RepoRoot "$submodule\CMakeLists.txt"))) {
            Write-Host "==> initializing $submodule (Flashlight)"
            & git -C $RepoRoot submodule update --init $submodule
            if ($LASTEXITCODE -ne 0) { throw "git submodule update --init $submodule failed" }
        }
    }
}

# --- 3e. Optional TTS tokenizers: ensure their sources are checked out ---------
if ($TtsJa -and -not (Test-Path (Join-Path $RepoRoot 'third_party\open_jtalk\src\CMakeLists.txt'))) {
    Write-Host "==> initializing third_party/open_jtalk (Japanese TTS tokenizer)"
    & git -C $RepoRoot submodule update --init third_party/open_jtalk
    if ($LASTEXITCODE -ne 0) { throw "git submodule update --init third_party/open_jtalk failed" }
}
if ($TtsZh -and -not (Test-Path (Join-Path $RepoRoot 'third_party\cppjieba\CMakeLists.txt'))) {
    Write-Host "==> initializing third_party/cppjieba (Mandarin TTS tokenizer)"
    & git -C $RepoRoot submodule update --init --recursive third_party/cppjieba
    if ($LASTEXITCODE -ne 0) { throw "git submodule update --init --recursive third_party/cppjieba failed" }
}

# --- 4. Configure + build -------------------------------------------------------
$cmakeArgs = @('-S', $RepoRoot, '-B', $BuildDir, '-G', 'Ninja', "-DCMAKE_BUILD_TYPE=$Config")
if ($Compiler -eq 'clang-cl') {
    $cmakeArgs += '-DCMAKE_C_COMPILER=clang-cl'
    $cmakeArgs += '-DCMAKE_CXX_COMPILER=clang-cl'
    if ($CrossCompiling) {
        $llvmTarget = if ($TargetArch -eq 'arm64') { 'arm64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
        $cmakeArgs += "-DCMAKE_C_COMPILER_TARGET=$llvmTarget"
        $cmakeArgs += "-DCMAKE_CXX_COMPILER_TARGET=$llvmTarget"
        $cmakeArgs += '-DCMAKE_SYSTEM_NAME=Windows'
        $cmakeArgs += "-DCMAKE_SYSTEM_PROCESSOR=$TargetArch"
    }
    if ($Backend -eq 'cuda') {
        # nvcc only supports cl.exe as its host compiler on Windows; pin it
        # explicitly so it never inherits clang-cl.
        $cmakeArgs += '-DCMAKE_CUDA_HOST_COMPILER=cl'
    }
}
switch ($Backend) {
    'cuda'   { $cmakeArgs += '-DGGML_CUDA=ON'; $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" }
    'vulkan' {
        $cmakeArgs += '-DGGML_VULKAN=ON'
        $cmakeArgs += '-DNEMO_SPEECH_GGML_PATCHED=OFF'
        # ggml-vulkan hard-requires the SPIRV-Headers CMake package; the Vulkan
        # SDK ships its config, but not on CMake's default search path.
        $spirvDir = Join-Path $env:VULKAN_SDK 'Lib\cmake\SPIRV-Headers'
        if (Test-Path $spirvDir) { $cmakeArgs += "-DSPIRV-Headers_DIR=$spirvDir" }
    }
    'cpu'    { $cmakeArgs += '-DNEMO_SPEECH_GGML_PATCHED=OFF' }
}
if ($Grpc -or $Flashlight) {
    $toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
    if (-not (Test-Path $toolchain)) { throw "vcpkg toolchain not found at $toolchain (bootstrap vcpkg first)." }
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
    $cmakeArgs += "-DVCPKG_TARGET_TRIPLET=$VcpkgTriplet"
}
if ($Grpc) {
    $cmakeArgs += '-DNEMO_SPEECH_WITH_GRPC=ON'
}
if ($Nmt) {
    $cmakeArgs += '-DNEMO_SPEECH_WITH_NMT=ON'
}
if ($Flashlight) {
    $cmakeArgs += '-DNEMO_SPEECH_WITH_FLASHLIGHT=ON'
}
if ($TtsJa) {
    $cmakeArgs += '-DNEMO_SPEECH_TTS_WITH_JA=ON'
}
if ($TtsZh) {
    $cmakeArgs += '-DNEMO_SPEECH_TTS_WITH_ZH=ON'
}
if ($AsrOnly) {
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_ASR=ON'
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_DIAR=ON'
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_TTS=OFF'
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_NMT=OFF'
    $cmakeArgs += '-DNEMO_SPEECH_WITH_NMT=OFF'
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_HTTP=OFF'
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_GRPC=OFF'
    $cmakeArgs += '-DNEMO_SPEECH_WITH_GRPC=OFF'
}
if ($Http) {
    $cmakeArgs += '-DNEMO_SPEECH_BUILD_HTTP=ON'
}

Write-Host "==> cmake $($cmakeArgs -join ' ')"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

Write-Host "==> building"
& cmake --build $BuildDir --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }

Write-Host "==> done. Binaries in $BuildDir\bin" -ForegroundColor Green
