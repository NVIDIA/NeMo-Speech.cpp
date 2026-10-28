# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Install the Windows build prerequisites that a bare machine lacks.
.DESCRIPTION
    Installs only what is missing, from the publishers' own servers: CMake and Ninja
    from their GitHub releases, the Python embeddable package from python.org and the
    Vulkan SDK from LunarG, each checked against a pinned SHA-256, and the Visual Studio
    2022 Build Tools (C++ workload) with Microsoft's bootstrapper, whose signature is
    checked. Downloads are retried, rate limits (HTTP 429) and stalls included. Under
    GitHub Actions the new tool directories are added to GITHUB_PATH (and VULKAN_SDK to
    GITHUB_ENV) for later steps.
.PARAMETER Install
    What to provide: cmake, ninja, msvc, python, vulkan (default: cmake, ninja, msvc).
.PARAMETER Architecture
    Target architecture of the build: x64, or arm64, which also needs the ARM64 MSVC
    tools (nvcc's host compiler) and clang-cl.
.PARAMETER VsVersion
    vswhere version range of the Visual Studio instance to use, for example
    '[17.0,18.0)' for 2022 (default: any). A missing instance is installed as 2022.
.PARAMETER ToolsDir
    Where CMake, Ninja and Python are unpacked (default: RUNNER_TOOL_CACHE, else
    %LOCALAPPDATA%\NeMoSpeech\tools).
.EXAMPLE
    pwsh scripts\windows\install-build-prerequisites.ps1 -Architecture arm64
    pwsh scripts\windows\install-build-prerequisites.ps1 -Install python
    pwsh scripts\windows\install-build-prerequisites.ps1 -Install vulkan
#>
[CmdletBinding()]
param(
    [ValidateSet('cmake', 'ninja', 'msvc', 'python', 'vulkan')]
    [string[]]$Install = @('cmake', 'ninja', 'msvc'),
    [ValidateSet('x64', 'arm64')]
    [string]$Architecture = 'x64',
    [string]$VsVersion,
    [string]$ToolsDir
)

$ErrorActionPreference = 'Stop'
if (-not $ToolsDir) {
    $ToolsDir = if ($env:RUNNER_TOOL_CACHE) { $env:RUNNER_TOOL_CACHE }
                else { Join-Path $env:LOCALAPPDATA 'NeMoSpeech\tools' }
}

# Pinned archives: change a version and its digest together.
$Archives = @{
    cmake  = @{
        Url    = 'https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-x86_64.zip'
        Sha256 = 'bace36e94b31c68ab6fa295f26dfa11219e0701cf7c94b0284a7d1cb13dac536'
        Bin    = 'cmake-4.4.4-windows-x86_64\bin'
    }
    ninja  = @{
        Url    = 'https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip'
        Sha256 = '07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65'
        Bin    = '.'
    }
    python = @{
        Url    = 'https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-amd64.zip'
        Sha256 = '91d828c2da3a029b41699e918674a0cb379c02cf20dab9c501306885f837402a'
        Bin    = '.'
    }
}
$VulkanSdk = @{
    Version = '1.4.363.0'
    Url     = 'https://sdk.lunarg.com/sdk/download/1.4.363.0/windows/vulkansdk-windows-X64-1.4.363.0.exe'
    Sha256  = '94a82d378f7a5e3e54c9db7d2fb7016af136e14ac0a18dbf0f2f67a36352d141'
}
$VsBootstrapper = 'https://aka.ms/vs/17/release/vs_buildtools.exe'

function Get-Download([string]$Url, [string]$Path) {
    # A transfer slower than 100 kB/s for a minute, or still running after 10 minutes, is
    # abandoned and retried (a Chocolatey download of the Vulkan SDK once hung for 45).
    curl.exe -fsSL --retry 4 --retry-all-errors --retry-delay 20 --connect-timeout 30 `
        --speed-limit 100000 --speed-time 60 --max-time 600 -o $Path $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed ($LASTEXITCODE): $Url" }
}

function Assert-Sha256([string]$Path, [string]$Expected, [string]$Name) {
    $digest = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
    if ($digest -ne $Expected) { throw "$Name checksum mismatch: got $digest" }
}

function Add-ToolPath([string]$Directory) {
    $env:Path = "$Directory;$env:Path"
    if ($env:GITHUB_PATH) { Add-Content -Path $env:GITHUB_PATH -Value $Directory }
    Write-Host "    on PATH: $Directory"
}

function Install-Archive([string]$Name) {
    $archive = $Archives[$Name]
    $root = Join-Path $ToolsDir "$Name-$($archive.Sha256.Substring(0, 12))"
    if (-not (Test-Path (Join-Path $root '.complete'))) {
        Write-Host "==> $Name from $($archive.Url)"
        $zip = Join-Path ([IO.Path]::GetTempPath()) "$Name-$PID.zip"
        Get-Download $archive.Url $zip
        Assert-Sha256 $zip $archive.Sha256 $Name
        if (Test-Path $root) { Remove-Item -Recurse -Force $root }
        Expand-Archive -Path $zip -DestinationPath $root
        Remove-Item $zip
        New-Item -ItemType File (Join-Path $root '.complete') | Out-Null
    }
    Add-ToolPath (Resolve-Path (Join-Path $root $archive.Bin)).Path
}

# A tool already on PATH is kept; the Microsoft Store's python.exe alias is not Python.
foreach ($name in 'cmake', 'ninja', 'python') {
    if ($Install -notcontains $name) { continue }
    $found = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notmatch '\\WindowsApps\\' } | Select-Object -First 1
    if ($found) { Write-Host "==> ${name}: $($found.Source)" }
    else { Install-Archive $name }
}

if ($Install -contains 'msvc') {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $components = @('Microsoft.VisualStudio.Component.VC.Tools.x86.x64')
    if ($Architecture -eq 'arm64') {
        $components += 'Microsoft.VisualStudio.Component.VC.Tools.ARM64',
                       'Microsoft.VisualStudio.Component.VC.Llvm.Clang'
    }
    $versionRange = if ($VsVersion) { @('-version', $VsVersion) } else { @() }
    function Find-VisualStudio([string[]]$Requires) {
        if (-not (Test-Path $vswhere)) { return $null }
        & $vswhere -latest -products * @versionRange -requires @Requires -property installationPath
    }

    $vs = Find-VisualStudio $components
    if ($vs) {
        Write-Host "==> Visual Studio: $vs"
    } else {
        $existing = Find-VisualStudio @('Microsoft.VisualStudio.Component.VC.Tools.x86.x64')
        $add = $components | ForEach-Object { '--add', $_ }
        if ($existing) {
            # The instance's own installer adds the missing components, whatever its product.
            $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
            $arguments = @('modify', '--installPath', "`"$existing`"") + $add + @('--quiet', '--norestart')
        } else {
            $installer = Join-Path ([IO.Path]::GetTempPath()) "vs_buildtools-$PID.exe"
            Get-Download $VsBootstrapper $installer
            $signature = Get-AuthenticodeSignature $installer
            if ($signature.Status -ne 'Valid' -or
                $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
                throw "the Visual Studio bootstrapper is not signed by Microsoft ($($signature.Status))"
            }
            $arguments = @('--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended') +
                         $add + @('--quiet', '--wait', '--norestart', '--nocache')
        }
        Write-Host "==> Visual Studio Build Tools: $(Split-Path -Leaf $installer) $($arguments -join ' ')"
        $process = Start-Process $installer -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -notin 0, 3010) {
            throw "the Visual Studio installer failed ($($process.ExitCode))"
        }
        $vs = Find-VisualStudio $components
        if (-not $vs) { throw "Visual Studio still lacks $($components -join ', ')" }
        Write-Host "==> Visual Studio: $vs"
    }
}

if ($Install -contains 'vulkan') {
    $sdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
    if (-not ($sdk -and (Test-Path (Join-Path $sdk 'Bin\glslc.exe')))) {
        Write-Host "==> Vulkan SDK $($VulkanSdk.Version) from $($VulkanSdk.Url)"
        $installer = Join-Path ([IO.Path]::GetTempPath()) "vulkansdk-$PID.exe"
        Get-Download $VulkanSdk.Url $installer
        Assert-Sha256 $installer $VulkanSdk.Sha256 'Vulkan SDK'
        # LunarG's unattended install; it sets the machine VULKAN_SDK.
        $root = Join-Path $env:SystemDrive "VulkanSDK\$($VulkanSdk.Version)"
        $arguments = @('--root', "`"$root`"", '--accept-licenses', '--default-answer',
                       '--confirm-command', 'install')
        $process = Start-Process $installer -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "the Vulkan SDK installer failed ($($process.ExitCode))" }
        Remove-Item $installer
        $sdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
        if (-not $sdk) { throw 'VULKAN_SDK was not set by the installer' }
    }
    Write-Host "==> Vulkan SDK: $sdk"
    $env:VULKAN_SDK = $sdk
    if ($env:GITHUB_ENV) { Add-Content -Path $env:GITHUB_ENV -Value "VULKAN_SDK=$sdk" }
    Add-ToolPath (Join-Path $sdk 'Bin')
}
