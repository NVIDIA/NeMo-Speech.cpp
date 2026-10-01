# llama.cpp and ggml patches

ggml comes from the `llama.cpp` submodule, which is pinned to an upstream commit
and never modified. This directory holds the project's changes to it: one patch
file per change, applied in the order `series` lists them.

## Builds

CMake applies the series automatically. With `NEMO_SPEECH_GGML_PATCHED=ON` (the
default, and the `cuda-*` and `cpu-*` presets) it copies the submodule to
`<build>/_deps/llama.cpp`, applies these patches, and builds ggml and llama.cpp
from the copy. It refreshes the copy only when a patch or the pinned commit
changes, and rewrites only the files whose content changed, so an edit to one
patch recompiles only what that edit affects.

- `-DNEMO_SPEECH_GGML_PATCHED=OFF` builds the pristine submodule (the `metal-*`
  and `vulkan-*` presets do this).
- `-DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=<dir>` builds `<dir>` as-is.

The environment variables the patches read are listed in
[docs/development/diagnostics.md](../docs/development/diagnostics.md).

## Patch files

Each patch is a commit exported with `git format-patch`, trimmed to the author,
the subject, the description and the diff:

```
From: Prabhsimran Singh <prabhsimrans@nvidia.com>
Subject: ggml-cuda: flatten PAD launches for large batches

PAD launched ne1 x (ne2*ne3) blocks, so large batches or long inputs exceeded
the 65535 grid.y/grid.z limit. Flatten the launch into grid.x.

Upstream: candidate (add test_pad cases beyond 65535).

diff --git a/ggml/src/ggml-cuda/pad.cu b/ggml/src/ggml-cuda/pad.cu
index 31cd00f77816fe6f5ffdda0428289e2aa1a25f9b..40b1b496b011eb4f85c35af3d1c29d223e2639e4 100644
...
```

- `git am` reads the header; `git apply`, which builds use, skips everything
  before the first `diff`. `export` writes the same bytes for the same commits.
- The `index` lines carry full blob IDs. Abbreviated IDs lengthen as a
  repository grows, so full ones keep the export identical in every clone.
  They also let `git am -3` or `git apply -3` fall back to a three-way merge
  when a patch file no longer applies cleanly, for example when bringing in
  patches from another branch.
- The file name is the subject in lowercase, with every character other than
  a letter, digit, `.` or `_` replaced by `-`, cut at 100 characters.
- The subject starts with the area: `ggml:`, `ggml-cuda:`, `ggml-cpu:`, or
  `llama:`. The description says what the change does and why this project
  needs it, and ends with an `Upstream:` line: a pull request link,
  `candidate` (with what it still needs), or why it stays here. Credit merged
  work with `Co-authored-by:` lines after it.
- `series` lists the files in apply order. A patch file that `series` does not
  list, or a listed file that does not exist, fails the build and the check.

Don't edit patch files by hand. Change the commits and run `export`: CI runs
`scripts/llama-patches.sh check`, which fails when the files differ from what
`export` writes.

## Working on the patches

`scripts/llama-patches.sh edit` creates `.deps/llama.cpp-patches`, a worktree of
the `llama.cpp` submodule at the pinned commit with one commit per patch. Build
against it and use ordinary git there; its commits never leave the worktree.

```sh
scripts/llama-patches.sh edit
cmake --preset cuda-asr -DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=$PWD/.deps/llama.cpp-patches
cd .deps/llama.cpp-patches
```

On Windows, run the script from Git Bash, which Git for Windows installs; it
prints the worktree path in the `C:/...` form CMake expects. Builds need no
script on any platform.

Then, in the worktree:

- **Change a patch:** edit, build, test, then
  `git commit -a --fixup=<commit of that patch>` and
  `git rebase -i --autosquash <pinned commit>`.
- **Add a patch:** commit on top (or move it with `git rebase -i`). Its commit
  message becomes the patch description; follow the conventions above.
- **Reword, reorder, split, merge or drop patches:** `git rebase -i <pinned commit>`.
- **Finish:** from the repository root, `scripts/llama-patches.sh export`
  rewrites `patches/` and `series`, and `scripts/llama-patches.sh done` removes
  the worktree (it refuses while the worktree has unexported changes). Commit
  the `patches/` changes together with the source changes that need them.

`scripts/llama-patches.sh diff` prints a `git range-diff` between the committed
series and the working tree, patch by patch, which is easier to review than a
diff of patch files.

When two branches both add a patch, both edit the end of `series`, so the merge
conflicts there on purpose: keep both lines in the order that applies, then run
`scripts/llama-patches.sh check`.

## Updating llama.cpp

```sh
scripts/llama-patches.sh rebase <commit or tag>   # pins the submodule there
# resolve conflicts with git in .deps/llama.cpp-patches, build, test
scripts/llama-patches.sh export
```

Patches that upstream has absorbed become empty and are dropped; `export` lists
what was removed. Update at least monthly: the longer the pin sits, the more the
series conflicts.

A clean rebase can still change behavior: when upstream narrows a backend's
`supports_op`, a patched op silently falls back to the CPU. After an update, run
`test-backend-ops` and
[`check_backend_coverage`](../docs/development/diagnostics.md#backend-coverage),
and compare outputs against the previous pin.
