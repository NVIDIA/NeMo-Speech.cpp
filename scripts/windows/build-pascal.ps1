# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$BuildDir = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'build-pascal-cuda-http'),
    [int]$Jobs = 0
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1') -Backend cuda -CudaArch 61 -AsrOnly -Http -BuildDir $BuildDir -Jobs $Jobs
exit $LASTEXITCODE
