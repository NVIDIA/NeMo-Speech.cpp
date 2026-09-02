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
      3. Provisions required C++ dependencies.
      4. For a CUDA build, applies the CUDA-only ggml patches.
      5. Configures with CMake (Ninja) and builds.

.PARAMETER Backend
    cuda | vulkan | cpu. CUDA and Vulkan are separate build trees (different ggml
    config), so use a distinct -BuildDir for each.

.PARAMETER Profile
    Component preset: core, asr, server (core + NMT + HTTP), full, or developer
    (full + tests, examples, and tools). Component switches add features.

.PARAMETER Grpc
    Also build the Riva-compatible riva_server. gRPC and protobuf are installed
    from the vcpkg manifest automatically.

.PARAMETER Nmt
    Also build the NMT translation component (links llama.cpp; checks out the
    llama.cpp submodule if needed). Combine with -Grpc for the gRPC TranslateText
    service. Convert the model separately - see docs/development/windows-build.md.

.PARAMETER Flashlight
    Build LM-fused CTC decoding with Flashlight and dynamically linked KenLM
    DLLs. Its required libraries are installed automatically.

.PARAMETER HttpTls
    Enable TLS in the HTTP server and install OpenSSL automatically.

.PARAMETER VcpkgRoot
    Existing vcpkg checkout to use. By default the driver bootstraps its pinned
    version under %LOCALAPPDATA%\NeMoSpeech.

.PARAMETER TtsJa
    Build the optional Japanese TTS tokenizer (Open JTalk, MeCab, and the NAIST
    dictionary).

.PARAMETER TtsZh
    Build the optional Mandarin TTS tokenizer (cppjieba and limonp).

.PARAMETER Tests
    Also build the first-party test executables (NEMO_SPEECH_BUILD_TESTS) so
    ctest can run against the build tree. Implied by -Profile developer.

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

.PARAMETER CublasShim
    Build the app-local cuBLAS replacement used by portable CUDA packages.

.PARAMETER Compiler
    C/C++ compiler: auto (default), msvc, or clang-cl. auto picks cl on x64 and
    clang-cl on ARM64 (ggml's ARM CPU backend rejects MSVC). nvcc always uses
    cl.exe as its CUDA host compiler regardless of this setting.

.PARAMETER Architecture
    Target architecture: auto (the host), x64, or arm64.

.EXAMPLE
    pwsh scripts\windows\build.ps1 -Backend cuda -Profile server
    pwsh scripts\windows\build.ps1 -Backend cpu -Profile full
    pwsh scripts\windows\build.ps1 -Backend vulkan -Profile developer
#>
[CmdletBinding()]
param(
    [ValidateSet('cuda', 'vulkan', 'cpu')]
    [string]$Backend = 'cuda',
    [ValidateSet('core', 'asr', 'server', 'full', 'developer')]
    [string]$Profile = 'core',
    [switch]$Grpc,
    [switch]$Nmt,
    [switch]$Flashlight,
    [switch]$TtsJa,
    [switch]$TtsZh,
    [switch]$Tests,
    [switch]$AsrOnly,
    [switch]$Http,
    [switch]$HttpTls,
    [string]$BuildDir,
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Config = 'Release',
    [string]$CudaArch = 'native',
    [switch]$CublasShim,
    [string]$VcpkgRoot,
    [string]$VcpkgTriplet,
    [ValidateSet('auto', 'x64', 'arm64')]
    [string]$Architecture = 'auto',
    # C/C++ compiler. 'auto' = cl on x64, clang-cl on ARM64 (ggml's ARM CPU
    # backend rejects MSVC). nvcc's CUDA host compiler is cl.exe in all cases.
    [ValidateSet('auto', 'msvc', 'clang-cl')]
    [string]$Compiler = 'auto',
    [int]$Jobs = 0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VcpkgBootstrapTag = '2026.07.29'
$VcpkgBootstrapCommit = '9e593bb18ea69cc5095e012465dcd675a822ed0d'

if ($AsrOnly) {
    if ($Profile -ne 'core' -and $Profile -ne 'asr') {
        throw '-AsrOnly cannot be combined with a non-ASR -Profile.'
    }
    $Profile = 'asr'
}
$BuildAsr = $false
$BuildDiar = $false
$BuildTts = $false
$BuildNmt = $false
$BuildHttp = $false
$BuildGrpc = $false
$BuildFlashlight = $false
$BuildTtsJa = $false
$BuildTtsZh = $false
$BuildTests = $false
$BuildExamples = $false
$BuildTools = $false

switch ($Profile) {
    'core'      { $BuildAsr = $true; $BuildDiar = $true; $BuildTts = $true }
    'asr'       { $BuildAsr = $true; $BuildDiar = $true }
    'server'    {
        $BuildAsr = $true; $BuildDiar = $true; $BuildTts = $true
        $BuildNmt = $true; $BuildHttp = $true
    }
    'full'      {
        $BuildAsr = $true; $BuildDiar = $true; $BuildTts = $true; $BuildNmt = $true
        $BuildHttp = $true; $BuildGrpc = $true; $BuildFlashlight = $true
        $BuildTtsJa = $true; $BuildTtsZh = $true
    }
    'developer' {
        $BuildAsr = $true; $BuildDiar = $true; $BuildTts = $true; $BuildNmt = $true
        $BuildHttp = $true; $BuildGrpc = $true; $BuildFlashlight = $true
        $BuildTtsJa = $true; $BuildTtsZh = $true
        $BuildTests = $true; $BuildExamples = $true; $BuildTools = $true
    }
}

$BuildGrpc = $BuildGrpc -or $Grpc.IsPresent
$BuildNmt = $BuildNmt -or $Nmt.IsPresent
$BuildFlashlight = $BuildFlashlight -or $Flashlight.IsPresent
$BuildTtsJa = $BuildTtsJa -or $TtsJa.IsPresent
$BuildTtsZh = $BuildTtsZh -or $TtsZh.IsPresent
$BuildHttp = $BuildHttp -or $Http.IsPresent -or $HttpTls.IsPresent
$BuildHttpTls = $HttpTls.IsPresent
$BuildTests = $BuildTests -or $Tests.IsPresent
if ($BuildGrpc -and (-not $BuildAsr -or -not $BuildTts)) {
    throw 'The gRPC server currently requires both ASR and TTS. Use -Profile server, full, or developer.'
}
if ($BuildFlashlight -and -not $BuildAsr) {
    throw 'The Flashlight decoder requires ASR. Use -Profile asr, server, full, or developer.'
}
if (($BuildTtsJa -or $BuildTtsZh) -and -not $BuildTts) {
    throw 'The Japanese and Mandarin tokenizers require TTS. Use a profile that includes TTS.'
}

if (-not $BuildDir) {
    $profileSuffix = if ($Profile -eq 'core') { '' } else { "-$Profile" }
    $archSuffix = if ($Architecture -eq 'auto') { '' } else { "-$Architecture" }
    $buildName = "build-$Backend$profileSuffix$archSuffix"
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
if ($CublasShim -and $Backend -ne 'cuda') {
    throw '-CublasShim requires -Backend cuda.'
}
$VcpkgArch = $TargetArch
if (-not $VcpkgTriplet) {
    # Link vcpkg libraries statically so installed binaries do not depend on the
    # build tree. The Visual C++ runtime remains dynamically linked.
    $VcpkgTriplet = "$VcpkgArch-windows-static-md"
}

$VcpkgFeatures = [Collections.Generic.List[string]]::new()
if ($BuildAsr -or $BuildDiar) { $VcpkgFeatures.Add('asr') }
if ($BuildGrpc) { $VcpkgFeatures.Add('grpc') }
if ($BuildFlashlight) { $VcpkgFeatures.Add('flashlight') }
if ($BuildHttpTls) { $VcpkgFeatures.Add('http-tls') }
if ($BuildExamples) { $VcpkgFeatures.Add('examples') }

Write-Host "==> nemo-speech Windows build" -ForegroundColor Cyan
Write-Host "    backend=$Backend profile=$Profile asr=$BuildAsr diar=$BuildDiar tts=$BuildTts nmt=$BuildNmt http=$BuildHttp grpc=$BuildGrpc flashlight=$BuildFlashlight tts-ja=$BuildTtsJa tts-zh=$BuildTtsZh tests=$BuildTests examples=$BuildExamples tools=$BuildTools"
Write-Host "    config=$Config compiler=$Compiler host=$HostArch target=$TargetArch build=$BuildDir jobs=$Jobs cublas-shim=$($CublasShim.IsPresent)"
Write-Host "    vcpkg=$($VcpkgFeatures -join ',') triplet=$VcpkgTriplet"
if ($DryRun) { return }

# --- 1. Refresh environment from registry (choco/installers land there) ---------
$machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$user    = [Environment]::GetEnvironmentVariable('Path', 'User')
$process = $env:Path
$env:Path = "$machine;$user;$process"
$vk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
if (-not $vk) { $vk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'User') }
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

foreach ($tool in 'git', 'cl', 'cmake', 'ninja') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH after env setup." }
}
if ($Compiler -eq 'clang-cl' -and -not (Get-Command clang-cl -ErrorAction SilentlyContinue)) {
    throw "clang-cl not on PATH; install LLVM (choco install llvm) or the VS 'C++ Clang tools' component."
}
if ($Backend -eq 'cuda') {
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        throw 'nvcc not found; install a supported NVIDIA CUDA Toolkit.'
    }
    if ($CudaArch -eq 'native') {
        $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if (-not $smi) {
            throw 'CUDA architecture auto-detection requires a working NVIDIA driver (nvidia-smi). Install the driver or pass -CudaArch explicitly for a cross-build.'
        }
        $gpu = & $smi.Source --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $gpu) {
            throw 'The NVIDIA driver is installed but not working. Repair the driver, or pass -CudaArch explicitly for a build host without a GPU.'
        }
    }
}
if ($Backend -eq 'vulkan') {
    if (-not $env:VULKAN_SDK) { throw 'VULKAN_SDK not set; install the LunarG Vulkan SDK.' }
    foreach ($required in 'Bin\glslc.exe', 'Lib\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake') {
        if (-not (Test-Path (Join-Path $env:VULKAN_SDK $required))) {
            throw "The Vulkan SDK is incomplete: $required was not found under $env:VULKAN_SDK. Reinstall the LunarG Vulkan SDK."
        }
    }
}

# --- 3. Provision manifest dependencies -----------------------------------------
$toolchain = $null
if ($VcpkgFeatures.Count -gt 0) {
    if (-not $VcpkgRoot) {
        $VcpkgRoot = Join-Path $env:LOCALAPPDATA "NeMoSpeech\vcpkg-$VcpkgBootstrapTag"
    }
    $toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
    $vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
    if ((Test-Path $toolchain) -and -not (Test-Path (Join-Path $VcpkgRoot '.git'))) {
        throw "vcpkg manifest mode requires a Git checkout, but $VcpkgRoot has no .git directory. Use the automatic user-local checkout or pass a valid -VcpkgRoot."
    }
    if (-not (Test-Path $toolchain)) {
        if ((Test-Path $VcpkgRoot) -and @(Get-ChildItem -Force $VcpkgRoot).Count -gt 0) {
            throw "vcpkg root exists but is incomplete: $VcpkgRoot. Pass -VcpkgRoot with a valid checkout or remove that incomplete directory."
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VcpkgRoot) | Out-Null
        Write-Host "==> bootstrapping vcpkg $VcpkgBootstrapTag in $VcpkgRoot"
        & git clone --depth 1 --branch $VcpkgBootstrapTag https://github.com/microsoft/vcpkg.git $VcpkgRoot
        if ($LASTEXITCODE -ne 0) { throw "vcpkg clone failed ($LASTEXITCODE)" }
    }
    if (-not $PSBoundParameters.ContainsKey('VcpkgRoot')) {
        $managedHead = (& git -C $VcpkgRoot rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -ne 0 -or $managedHead -ne $VcpkgBootstrapCommit) {
            throw "The managed vcpkg checkout at $VcpkgRoot is not the expected $VcpkgBootstrapTag revision. Remove that directory and rerun, or pass a valid -VcpkgRoot explicitly."
        }
    }
    if (-not (Test-Path $vcpkg)) {
        $bootstrap = Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat'
        if (-not (Test-Path $bootstrap)) { throw "vcpkg bootstrap script not found at $bootstrap" }
        Write-Host '==> building vcpkg'
        & $bootstrap -disableMetrics
        if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed ($LASTEXITCODE)" }
    }
    $env:VCPKG_ROOT = $VcpkgRoot
    $env:VCPKG_DISABLE_METRICS = '1'
    Write-Host "==> vcpkg features: $($VcpkgFeatures -join ', ') ($VcpkgTriplet)"
}

# --- 4. Initialize exactly the required submodules ------------------------------
function Initialize-RequiredSubmodule {
    param([string]$Path, [string]$Sentinel, [switch]$Recursive)
    if (Test-Path (Join-Path $RepoRoot "$Path\$Sentinel")) { return }
    if (-not (Test-Path (Join-Path $RepoRoot '.gitmodules'))) {
        throw "Required source dependency '$Path' is missing and this is not a Git checkout."
    }
    Write-Host "==> initializing $Path"
    $arguments = @('-C', $RepoRoot, 'submodule', 'update', '--init')
    if ($Recursive) { $arguments += '--recursive' }
    $arguments += $Path
    & git @arguments
    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed for $Path" }
    if (-not (Test-Path (Join-Path $RepoRoot "$Path\$Sentinel"))) {
        throw "Submodule $Path completed without the required file $Sentinel"
    }
}

Initialize-RequiredSubmodule 'ggml' 'CMakeLists.txt'
if ($BuildNmt) {
    Initialize-RequiredSubmodule 'llama.cpp' 'CMakeLists.txt'
} elseif ($BuildAsr) {
    Initialize-RequiredSubmodule 'llama.cpp' 'vendor\miniaudio\miniaudio.h'
}
if ($BuildHttp) { Initialize-RequiredSubmodule 'third_party\cpp-httplib' 'httplib.h' }
if ($BuildGrpc) { Initialize-RequiredSubmodule 'proto\riva-common' 'LICENSE' }
if ($BuildFlashlight) {
    Initialize-RequiredSubmodule 'third_party\flashlight-text' 'CMakeLists.txt'
    Initialize-RequiredSubmodule 'third_party\kenlm' 'CMakeLists.txt'
}
if ($BuildTtsJa) { Initialize-RequiredSubmodule 'third_party\open_jtalk' 'src\CMakeLists.txt' }
if ($BuildTtsZh) {
    Initialize-RequiredSubmodule 'third_party\cppjieba' 'deps\limonp\CMakeLists.txt' -Recursive
    $lfsPointers = @(Get-ChildItem (Join-Path $RepoRoot 'src\tts\tokenizer\mandarin_data') -File |
        Where-Object { (Get-Content $_.FullName -TotalCount 1) -eq 'version https://git-lfs.github.com/spec/v1' })
    if ($lfsPointers.Count -gt 0) {
        & git lfs version *> $null
        if ($LASTEXITCODE -ne 0) {
            throw 'Mandarin TTS requires Git LFS data. Install Git for Windows with Git LFS, then rerun the build.'
        }
        Write-Host '==> downloading Mandarin tokenizer data with Git LFS'
        & git -C $RepoRoot lfs pull --include='src/tts/tokenizer/mandarin_data/*'
        if ($LASTEXITCODE -ne 0) { throw 'git lfs pull failed for Mandarin tokenizer data' }
        foreach ($pointer in $lfsPointers) {
            if ((Get-Content $pointer.FullName -TotalCount 1) -eq 'version https://git-lfs.github.com/spec/v1') {
                throw "Git LFS did not materialize $($pointer.FullName)"
            }
        }
    }
}

# --- 5. CUDA-only: apply the ggml patches ---------------------------------------
if ($Backend -eq 'cuda') {
    Write-Host "==> applying ggml patches (CUDA)"
    & (Join-Path $PSScriptRoot 'apply-ggml-patches.ps1')
}

# --- 6. Configure + build -------------------------------------------------------
function ConvertTo-CMakeBool([bool]$Value) {
    if ($Value) { return 'ON' }
    return 'OFF'
}

$cmakeArgs = @(
    '-S', $RepoRoot, '-B', $BuildDir, '-G', 'Ninja', "-DCMAKE_BUILD_TYPE=$Config",
    "-DNEMO_SPEECH_BUILD_ASR=$(ConvertTo-CMakeBool $BuildAsr)",
    "-DNEMO_SPEECH_BUILD_DIAR=$(ConvertTo-CMakeBool $BuildDiar)",
    "-DNEMO_SPEECH_BUILD_TTS=$(ConvertTo-CMakeBool $BuildTts)",
    "-DNEMO_SPEECH_BUILD_NMT=$(ConvertTo-CMakeBool $BuildNmt)",
    "-DNEMO_SPEECH_WITH_NMT=$(ConvertTo-CMakeBool $BuildNmt)",
    "-DNEMO_SPEECH_BUILD_HTTP=$(ConvertTo-CMakeBool $BuildHttp)",
    "-DNEMO_SPEECH_HTTP_TLS=$(ConvertTo-CMakeBool $BuildHttpTls)",
    "-DNEMO_SPEECH_BUILD_GRPC=$(ConvertTo-CMakeBool $BuildGrpc)",
    "-DNEMO_SPEECH_WITH_GRPC=$(ConvertTo-CMakeBool $BuildGrpc)",
    "-DNEMO_SPEECH_WITH_FLASHLIGHT=$(ConvertTo-CMakeBool $BuildFlashlight)",
    '-DNEMO_SPEECH_WITH_NORM=OFF',
    "-DNEMO_SPEECH_TTS_WITH_JA=$(ConvertTo-CMakeBool $BuildTtsJa)",
    "-DNEMO_SPEECH_TTS_WITH_ZH=$(ConvertTo-CMakeBool $BuildTtsZh)",
    "-DNEMO_SPEECH_BUILD_TESTS=$(ConvertTo-CMakeBool $BuildTests)",
    "-DBUILD_TESTING=$(ConvertTo-CMakeBool $BuildTests)",
    "-DNEMO_SPEECH_BUILD_EXAMPLES=$(ConvertTo-CMakeBool $BuildExamples)",
    "-DNEMO_SPEECH_BUILD_TOOLS=$(ConvertTo-CMakeBool $BuildTools)"
)
if ($VcpkgFeatures.Count -gt 0) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
    $cmakeArgs += "-DVCPKG_TARGET_TRIPLET=$VcpkgTriplet"
    $cmakeArgs += "-DVCPKG_MANIFEST_FEATURES=$($VcpkgFeatures -join ';')"
    $cmakeArgs += "-DVCPKG_INSTALLED_DIR=$(Join-Path $BuildDir 'vcpkg_installed')"
}
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
if ($TargetArch -eq 'arm64') {
    # Avoid an additional OpenMP runtime DLL in ARM64 packages.
    $cmakeArgs += '-DGGML_OPENMP=OFF'
}
switch ($Backend) {
    'cuda'   {
        $cmakeArgs += '-DGGML_CUDA=ON'
        $cmakeArgs += '-DGGML_VULKAN=OFF'
        $cmakeArgs += "-DNEMO_SPEECH_CUBLAS_SHIM=$(ConvertTo-CMakeBool $CublasShim.IsPresent)"
        $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    }
    'vulkan' {
        $cmakeArgs += '-DGGML_CUDA=OFF'
        $cmakeArgs += '-DGGML_VULKAN=ON'
        $cmakeArgs += '-DNEMO_SPEECH_GGML_PATCHED=OFF'
        # ggml-vulkan hard-requires the SPIRV-Headers CMake package; the Vulkan
        # SDK ships its config, but not on CMake's default search path.
        $spirvDir = Join-Path $env:VULKAN_SDK 'Lib\cmake\SPIRV-Headers'
        if (Test-Path $spirvDir) { $cmakeArgs += "-DSPIRV-Headers_DIR=$spirvDir" }
    }
    'cpu'    {
        $cmakeArgs += '-DGGML_CUDA=OFF'
        $cmakeArgs += '-DGGML_VULKAN=OFF'
        $cmakeArgs += '-DNEMO_SPEECH_GGML_PATCHED=OFF'
    }
}

Write-Host "==> cmake $($cmakeArgs -join ' ')"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

Write-Host "==> building"
& cmake --build $BuildDir --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }

Write-Host "==> done. Binaries in $BuildDir\bin" -ForegroundColor Green
