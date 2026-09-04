# ggml-patches

Edge-only changes to the vendored `ggml` submodule, kept as patches so the
submodule stays pinned to clean upstream (`ggml-org/ggml`). Apply them after
checking out / updating the submodule:

```sh
git submodule update --init ggml
scripts/apply-ggml-patches.sh        # applies patches in filename order
```

Patched CUDA and Metal builds expect the patches before CMake configuration.
CPU, Vulkan, and stock-CUDA builds do not. `apply-ggml-patches.sh` uses `git
apply`, skips patches that are already applied, and applies new patches in
filename order. Later patches may build on files changed by earlier patches;
0006 carries the dispatch wiring for the ops/kernels introduced by
0001/0003/0005. Docker builds and `scripts/configure.sh` apply the series
automatically; apply it explicitly before a raw CUDA or Metal CMake
configuration.

Metal requires only 0018, the one patch that touches the Metal backend. The
`metal-*` presets apply the whole series because `apply-ggml-patches.sh`
requires all of them or it fails. Once the submodule moves past upstream ggml
`33c9ea5`, drop 0018 as described below and remove `metal-*` from the `case` in
`scripts/configure.sh`.

## Building against patched vs stock ggml

`NEMO_SPEECH_GGML_PATCHED` tells the build whether the vendored ggml contains
this patch series. It defaults to `ON` because the ASR encoder directly uses
the fused relative-position attention op from 0001 and the F16 depthwise-conv
behavior from 0004.

```sh
# Patched ggml (default): apply the patches before configuring CMake.
scripts/apply-ggml-patches.sh
cmake -S . -B build -DGGML_CUDA=ON

# Pristine upstream ggml: do not apply the patches and opt out explicitly.
cmake -S . -B build -DGGML_CUDA=ON -DNEMO_SPEECH_GGML_PATCHED=OFF
```

With `NEMO_SPEECH_GGML_PATCHED=OFF`, the encoder uses stock ggml operations:
unfused relative-position attention and `ggml_conv_1d_dw` im2col lowering.
These paths remain correct across backends but cost latency on CUDA. The
dependent options below default to `ON` only when both `GGML_CUDA` and
`NEMO_SPEECH_GGML_PATCHED` are enabled, and are otherwise forced `OFF`:

| option | effect when enabled |
|---|---|
| `NEMO_SPEECH_FUSED_RELPOS_ATTN` | emit the fused relative-position attention CUDA op |
| `NEMO_SPEECH_DIRECT_DW_CONV` | use the direct CUDA depthwise-convolution kernel |
| `NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS` | emit patched sigmoid-GLU and BF16-fusion graph patterns |

The options can be disabled independently for correctness or performance
bisection even in a patched build. Other patches optimize ordinary ggml
operations through their own eligibility checks and may still activate when
`NEMO_SPEECH_GGML_PATCHED=OFF` if the patched sources are present. A genuine
stock comparison therefore requires both a pristine ggml checkout and
`NEMO_SPEECH_GGML_PATCHED=OFF`.

## Patches

- **0001-fused-relpos-attn.patch** - adds relative-position fused attention for
  CUDA, including stride-aware inputs, F16 K/V/P storage, head-merged output,
  and the `NEMO_SPEECH_FUSED_RELPOS_ATTN` encoder path.

- **0002-nvfp4-warp-quantizer.patch** - adds warp-cooperative NVFP4 activation
  quantization with configurable scale search and an optional self-check.

- **0003-norm-mul-add-fusion.patch** - fuses affine LayerNorm with row-vector
  scale and optional bias.

- **0004-conv2d-dw-f16-kernel.patch** - supports F16 weights in the direct
  depthwise-convolution CUDA kernel with F32 input and output.

- **0005-skinny-q8-gemm.patch** - adds a Q8_0 x F32 GEMM for skinny streaming
  activations, including planar weights, deterministic K-split reduction, and
  an optional bias epilogue. `GGML_SKINNY_Q8` controls dispatch;
  `GGML_SKINNY_Q8_INPLACE=0` uses separate repack storage when required by a
  multi-stream scheduler.

- **0006-cuda-dispatch-wiring.patch** - wires fused attention, affine
  LayerNorm, skinny-Q8, planar-Q8, and narrow bias/SiLU epilogues into the CUDA
  backend.

- **0007-magpietts-nanocodec.patch** - adds grouped transposed convolution and
  Snake for MagpieTTS and NanoCodec, two-column MMVF epilogues for paired CFG,
  bounded CUDA graph caching, and CUDA architecture handling.

- **0008-cublas-bf16-projections.patch** - flattens contiguous outer activation
  dimensions into shared-weight cuBLAS GEMMs and folds supported BF16 projection
  epilogues into output conversion.

- **0009-fastconformer-cuda-fusions.patch** - adds sigmoid GLU, Macaron
  residual, affine LayerNorm conversion, and BF16 projection fusions for
  FastConformer.

- **0010-cuda-pad-large-batch-grid.patch** - flattens CUDA PAD launches into
  `grid.x` so large batch dimensions do not exceed the `grid.z` limit.

- **0011-cuda-graph-shape-key.patch** - keys cached CUDA graph executables by
  graph identity and structural shape so allocator reuse cannot select an
  incompatible executable.

- **0012-cuda-streaming-cache-copies.patch** - adds pitched cache-tail copies
  and vectorized indexed state-arena transfers for streaming K/V state.

- **0013-cuda-cached-f16-cublas.patch** - adds an optional cached-F16/cuBLAS
  path for skinny Q8 projections while retaining F32 accumulation and output.

- **0014-cuda-fused-attention-extensions.patch** - generalizes fused attention
  to standard and relative-position modes, persistent circular K/V caches,
  active-length state, streaming FastConformer shapes, and Magpie's cached
  single-query shape.

- **0015-cuda-ctc-batch-fusions.patch** - adds BatchNorm, transpose/SiLU,
  affine LayerNorm, and cached-F16 projection epilogues used by batched
  FastConformer and Magpie.

- **0016-fix-batched-conv1d-layout.patch** - restores batch and output-channel
  axes after flattened Conv1D matrix multiplication.

- **0017-cuda-stream-interop.patch** - exposes borrowed access to the active
  CUDA stream and stable graph templates for external graph composition.

- **0018-metal-tensor-api-dynamic-k.patch** - backports upstream ggml `33c9ea5`
  (llama/27450), which stops the Metal tensor-API matmul reading `src1` out of
  bounds when `K % 32 != 0`. Drop it once the submodule passes that commit,
  keeping the `K = 1296` test case.

## Regenerating after editing ggml

Several patches touch the same ggml files, so regenerating a patch from the
fully patched submodule can accidentally fold later changes into it. Edit and
diff at the patch's actual point in the series:
Most base files belong to one patch; 0013 and 0014 are explicit layered
exceptions. Do not regenerate 0001 from a fully patched live tree without first
removing the 0014 delta, or the circular-cache extension will be folded into it.

```sh
# Create a disposable worktree at the pinned upstream commit.
git -C ggml worktree add "$PWD/ggml-patch-work" HEAD
cd ggml-patch-work

# Apply every patch before the one being edited, then stage that baseline.
target=0007-magpietts-nanocodec.patch
for patch in ../ggml-patches/*.patch; do
    [ "$(basename "$patch")" = "$target" ] && break
    git apply "$patch"
done
git add -A

# Apply the target patch, edit it, and capture only its delta.
git apply "../ggml-patches/$target"
# Edit and test the affected files.
git add -N src/ggml-cuda/<new-file>  # only when the patch adds a new file
git diff --binary > "../ggml-patches/$target"

# Return to the repository root, check the patch, and remove the worktree.
cd ..
git diff --check -- "ggml-patches/$target"
git -C ggml worktree remove --force "$PWD/ggml-patch-work"
```

Adjust paths when the disposable worktree is placed elsewhere. If an edited
patch changes context used by later patches, rebase those later patches in the
same way.

Patch 0013 is intentionally layered on top of 0005. To regenerate it without
folding the generic skinny-Q8 implementation into the cached-F16 patch, use a
temporary ggml worktree, apply and stage patches 0001 through 0012 as the
baseline, then copy in only the cached-F16 changes and the CUDA architecture
target correction, and run `git diff` against that staged baseline for
`CMakeLists.txt` and `skinny-q8.cu`.

Patch 0014 is intentionally layered on top of 0001 and 0013. Apply and stage
patches 0001 through 0013 in a temporary worktree, copy the edited attention
API, dispatch, and CUDA implementation into that worktree, then generate 0014
with `git diff` against the staged baseline.

Generate patches with `git diff` only (GNU `diff`/editors can strip the
leading space on blank context lines, which `git apply` rejects as corrupt).

To verify the complete series, apply every patch in order to a fresh worktree at
the pinned ggml commit, then recursively compare that tree with the live patched
submodule and confirm that all files match.
