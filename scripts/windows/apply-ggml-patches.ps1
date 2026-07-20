# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Apply the in-tree ggml patches onto the vendored ggml submodule (Windows).

.DESCRIPTION
    PowerShell port of scripts/apply-ggml-patches.sh. Applies ggml-patches/*.patch
    (filename order) via `git apply`, keeping the submodule pinned to clean
    upstream. The patches are CUDA-only, so they are only needed for a CUDA build
    (-DGGML_CUDA=ON with the default -DNEMO_SPEECH_GGML_PATCHED=ON).

    Patches that still reverse-apply cleanly are skipped. Apply the complete
    series to a clean submodule for deterministic setup; later patches may
    refine lines introduced by earlier patches, making reverse detection
    ambiguous on a fully patched tree. Each patch is normalized to LF before
    `git apply` - on Windows with core.autocrlf=true the .patch files
    check out CRLF, and git apply rejects some hunks ("corrupt patch") on mixed
    endings (notably 0007). Normalizing makes it work regardless of checkout EOL.

    Control flow is driven off $LASTEXITCODE rather than $ErrorActionPreference:
    `git apply --reverse --check` writes to stderr on the expected "not applied
    yet" path, which under -ErrorAction Stop would otherwise abort the script.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$Root    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # scripts\windows\..\..
$Ggml    = Join-Path $Root 'ggml'
$Patches = Join-Path $Root 'ggml-patches'

if (-not (Test-Path (Join-Path $Ggml 'src'))) {
    throw "ggml submodule not initialized at $Ggml. Run: git submodule update --init ggml"
}
if (-not (Test-Path -LiteralPath $Patches -PathType Container)) {
    throw "ggml-patches directory not found: $Patches"
}

# Run `git apply <args> <patch>` against an LF-normalized copy of the patch.
# Returns git's exit code; stderr is swallowed (callers decide off the code).
function Invoke-GitApply {
    param([string[]] $ApplyArgs, [string] $PatchFile)
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        # Strip ALL CR bytes (same as `tr -d '\r'`), not just CRLF pairs: a
        # worktree checked out before .gitattributes landed can carry stray CRs
        # that git apply rejects as "corrupt patch".
        $content = [System.IO.File]::ReadAllText($PatchFile) -replace "`r", ""
        [System.IO.File]::WriteAllText($tmp, $content)  # UTF-8, no BOM
        & git -C $Ggml apply @ApplyArgs $tmp 2>&1 | Out-Null
        return $LASTEXITCODE
    }
    finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

# Enumerate up front with -ErrorAction Stop (so a read failure isn't swallowed by
# the 'Continue' preference above) and fail if there are no patches to apply.
$patchFiles = @(Get-ChildItem -LiteralPath $Patches -Filter '*.patch' -File -ErrorAction Stop |
    Sort-Object Name)
if ($patchFiles.Count -eq 0) {
    throw "no .patch files found in $Patches"
}

# Later patches may modify hunks introduced by earlier patches, so individual
# reverse checks cannot always recognize a fully applied series. Build the
# expected tree in a temporary index and compare it with the patched paths in
# the worktree. The caller's real Git index is never modified.
function Test-PatchSeriesApplied {
    param([System.IO.FileInfo[]] $PatchFiles)

    & git -C $Ggml rev-parse --git-dir 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
    [System.IO.Directory]::CreateDirectory($tmpDir) | Out-Null
    $expectedIndex = Join-Path $tmpDir 'expected.index'
    $currentIndex = Join-Path $tmpDir 'current.index'
    $previousIndex = $env:GIT_INDEX_FILE

    try {
        $env:GIT_INDEX_FILE = $expectedIndex
        & git -C $Ggml read-tree HEAD 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw 'failed to initialize temporary ggml index'
        }
        foreach ($patch in $PatchFiles) {
            if ((Invoke-GitApply @('--cached') $patch.FullName) -ne 0) {
                throw "$($patch.Name) does not apply to the pinned ggml commit"
            }
        }
        $paths = @(& git -C $Ggml diff --cached --name-only HEAD)
        $expectedTree = (& git -C $Ggml write-tree).Trim()

        $env:GIT_INDEX_FILE = $currentIndex
        & git -C $Ggml read-tree HEAD 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw 'failed to initialize temporary current ggml index'
        }
        $currentPaths = @()
        foreach ($path in $paths) {
            & git -C $Ggml cat-file -e "HEAD:$path" 2>&1 | Out-Null
            if ((Test-Path -LiteralPath (Join-Path $Ggml $path)) -or $LASTEXITCODE -eq 0) {
                $currentPaths += $path
            }
        }
        if ($currentPaths.Count -ne 0) {
            & git -C $Ggml add -A -- @currentPaths 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw 'failed to populate temporary current ggml index'
            }
        }
        $currentTree = (& git -C $Ggml write-tree).Trim()
        return $currentTree -eq $expectedTree
    }
    finally {
        if ($null -eq $previousIndex) {
            Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue
        }
        else {
            $env:GIT_INDEX_FILE = $previousIndex
        }
        Remove-Item -LiteralPath $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (Test-PatchSeriesApplied $patchFiles) {
    Write-Host '[ggml-patch] current series already applied'
    Write-Host '[ggml-patch] done'
    exit 0
}

$applied = 0
$skipped = 0
foreach ($patch in $patchFiles) {
    $name = $patch.Name
    if ((Invoke-GitApply @('--reverse', '--check') $patch.FullName) -eq 0) {
        Write-Host "[ggml-patch] ${name}: already applied"
        $skipped++
        continue
    }
    if ((Invoke-GitApply @('--check') $patch.FullName) -ne 0) {
        throw "[ggml-patch] ${name}: does NOT apply cleanly to current ggml (the tree is modified or contains a stale patch series; restore the pinned ggml submodule, then retry)"
    }
    if ((Invoke-GitApply @() $patch.FullName) -ne 0) {
        throw "[ggml-patch] ${name}: git apply failed"
    }
    Write-Host "[ggml-patch] ${name}: applied"
    $applied++
}

Write-Host "[ggml-patch] done (applied $applied, skipped $skipped)"
