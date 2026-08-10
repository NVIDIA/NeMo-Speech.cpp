# ggml-patches

Edge-only changes to the vendored `ggml` submodule, kept as patches so the
submodule stays pinned to clean upstream (`ggml-org/ggml`). Apply them after
checking out / updating the submodule:

```sh
git submodule update --init ggml
scripts/apply-ggml-patches.sh        # applies patches in filename order
```

Patched CUDA builds expect the patches before CMake configuration; CPU, Metal,
Vulkan, and stock-CUDA builds do not. `apply-ggml-patches.sh` uses `git apply`,
skips patches that are already applied, and applies new patches in filename
order. Later patches may build on files changed by earlier patches; 0006 carries
the dispatch wiring for the ops/kernels introduced by 0001/0003/0005. Docker
builds and `scripts/configure.sh` apply the series automatically; apply it
explicitly before a raw CUDA CMake configuration.

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

- **0001-fused-relpos-attn.patch** - adds `GGML_OP_FUSED_RELPOS_ATTN`, a fused
  FastConformer relative-position attention op (content + position-with-rel-shift
  + scale + mask + softmax + context in one CUDA kernel). Touches `ggml.h`,
  `ggml.c`, the CPU backend (unsupported stub + `supports_op` false); adds
  `ggml-cuda/fused-relpos-attn.{cu,cuh}`. The kernel takes K/V/P in F32 or F16
  (it is re-read per query, so F16 halves the dominant traffic; math stays F32)
  and uses a warp-cooperative, vectorized score loop for d_k == 128 (the
  thread-per-key scalar loop is load-issue-bound and left as the generic
  fallback). For the SM100 streaming shape (`d_k=128`, `q=2`, `kv=72`), an
  occupancy query selects between one block per query and a two-query block
  that reuses K/V after the grid exceeds one resident wave. Other CUDA shapes
  retain the generic fused kernel.
  The op is stride-general: Q/K/V/P may be non-contiguous views
  (only d_k rows must be contiguous; the CUDA op derives all addressing from
  tensor `nb[]`), `merge_heads` emits a head-merged output layout whose
  permute view is a contiguous `(n_feat, q, batch)` matrix, and the rel-pos
  table length assert is `>=` so one precomputed table serves shorter tail
  chunks — together this lets the cache-aware streaming encoder call the
  kernel directly on the fused-QKV output and feat-major K/V windows with
  zero staging copies. Wired into the encoder behind
  `NEMO_SPEECH_FUSED_RELPOS_ATTN`.

- **0002-nvfp4-warp-quantizer.patch** - reworks the CUDA NVFP4 MMQ activation
  quantizer. Upstream uses one thread per 16-element sub-block (serial) plus a
  5-candidate scale search, which dominates small-batch streaming GEMMs. Adds a
  single-pass warp-cooperative kernel (~8x threads, MXFP4-style) gated by
  `NEMO_SPEECH_NVFP4_WARP`, a `NEMO_SPEECH_NVFP4_SCALE_SEARCH` width knob (1..5), and a
  byte-exact self-check (`NEMO_SPEECH_NVFP4_SELFCHECK`). The native-FP4 MMQ path is
  Blackwell-1200-only; these knobs are inert on architectures using the generic
  path.

- **0003-norm-mul-add-fusion.patch** - fused LayerNorm kernel
  (`norm_mul_add_f32`): `GGML_OP_NORM` + row-vector gamma `MUL` + optional
  row-vector beta `ADD` in one launch (upstream only fuses `RMS_NORM`).
  Restricted to the classic affine pattern (ne0-length contiguous vectors).
  Eligibility + dispatch live in patch 0006. Graph code must emit non-inplace
  mul/add for the fusion to match (inplace ops are views).

- **0004-conv2d-dw-f16-kernel.patch** - `conv2d-dw.cu` accepts an F16 kernel
  (weights) with F32 input/output (templated kernel type). Lets the encoder's
  depthwise convs run the direct CUDA kernel instead of im2col + GEMM while
  keeping converter-produced F16 conv weights.

- **0005-skinny-q8-gemm.patch** - adds `ggml-cuda/skinny-q8.{cu,cuh}`: a
  Q8_0 x F32 GEMM specialized for skinny activations (9 <= N <= 64, the
  streaming-encoder shape where mul_mat_q runs latency-bound). int8 tensor-core
  `mma.m16n8k32` with per-q8-block scaling, K128 two-buffer cp.async pipeline,
  once-per-tensor weight repack into aligned planes (cached; weight buffers
  only), warp-coalesced activation quantizer, deterministic K-split reduction
  for small-M shapes, one grid.z launch for all 64-column outer-batch tiles,
  and an optional fused row-vector bias epilogue. It is enabled by default; logical
  per-sequence-width dispatch prevents outer batch size from selecting different
  math (`GGML_SKINNY_Q8_OUTER_BATCH=1` opts into dense outer-batch flattening -
  use with `GGML_SKINNY_Q8_INPLACE=0` under a multi-stream scheduler). Accepts
  serialized tensor-planar Q8 weights
  (`GGML_TENSOR_FLAG_Q8_PLANAR`, see 0006) without a runtime repack. Kill
  switch: `GGML_SKINNY_Q8=0`. The repack is in-place by default (reuses the
  weight buffer, saving the ~1.07 GB cudaMalloc duplicate on parakeet-xxl),
  which is correct and fast for the streaming-ASR encoder runtime. Two
  caveats for the llama.cpp NMT decoder, which the NMT pipeline handles by
  setting an env var process-wide before any service warms up: (1) the
  in-place D2D memcpy is a stream-ordering hazard under llama.cpp's
  multi-stream graph-split scheduling (it corrupts the GEMM even though the
  repacked bytes are correct), and (2) the kernel is tuned for the encoder's
  N=9..64 shape, so for the decoder's N=1 decode it is ~17% slower than stock
  mmvq. So: NMT without ASR sets `GGML_SKINNY_Q8=0` (disable skinny entirely;
  correct and faster); NMT alongside ASR sets `GGML_SKINNY_Q8_INPLACE=0`
  (keep the encoder's skinny, force the safe separate-buffer layout).

- **0006-cuda-dispatch-wiring.patch** - `ggml-cuda.cu` + `common.cuh` +
  `mmvq.cu`/`vecdotq.cuh` + the `ggml.h` flag: the CUDA-backend wiring for
  the patches above (fused-relpos op dispatch and `supports_op`,
  `GGML_OP_NORM` fusion acceptance + eval-loop dispatch, skinny-q8 mul_mat
  dispatch hook and the skinny GEMM+bias fusion), plus the streaming Q8
  weight/epilogue work: the serialized tensor-planar Q8 layout
  (`GGML_TENSOR_FLAG_Q8_PLANAR`; planar Q8_0 vec-dot + planar MMVQ dispatch,
  `VDR_Q8_0_Q8_1_MMVQ` 2->4; GGUFs produced with
  `convert_model.py --outtype q8_0 --q8-layout planar`) and the Q8
  narrow bias/SiLU epilogue fusion for the two-frame streaming chunk
  (`GGML_CUDA_Q8_NARROW_EPILOGUE`, default on: MUL_MAT+UNARY and
  MUL_MAT+ADD+SILU fusion with broadcast Linear-bias support, plus
  flattened-outer-batch MMVQ eligibility). Also carries the documented
  sm_110 finding on the Blackwell FP4 gate in `common.cuh`.

- **0007-magpietts-nanocodec.patch** - adds the CUDA operations used by
  MagpieTTS and NanoCodec, including grouped transposed convolution and Snake;
  bounds the keyed CUDA graph cache with configurable sweep and idle-eviction
  intervals; and adds SM110/Jetson Thor architecture handling.

- **0008-cublas-bf16-projections.patch** - recognizes shared F32/F16/BF16
  weights broadcast over contiguous outer activation dimensions and presents
  `[K,T,B,...]` as one `[K,T*B*...]` cuBLAS GEMM. For BF16 projections it also
  folds row bias and optional SiLU/rounding into the required output conversion.
  Native BF16 epilogues dispatch only on NVIDIA SM80+; older architectures keep
  the established conversion and elementwise paths.

- **0009-fastconformer-cuda-fusions.patch** - adds the CUDA sigmoid GLU used by
  the convolution module, fuses Macaron `residual + scale * ff`, and lets fused
  affine LayerNorm write the BF16 projection input directly. The graph rewrites
  are behind `NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS`; BF16 output requires
  NVIDIA SM80+, and the 256-thread specialization for 1024-wide LayerNorm rows
  is selected only on SM90+.

- **0010-cuda-pad-large-batch-grid.patch** - flattens CUDA PAD's tensor-slice
  launch into `grid.x`. Upstream maps `ne2 * ne3` onto `grid.z`, which exceeds
  CUDA's 65,535-block z-dimension limit for Nemotron's 256-channel causal
  subsampling tensors at batch sizes of 256 or larger. The flattened launch
  preserves the same indexing while allowing the large batches required for
  throughput sweeps.

- **0011-cuda-graph-shape-key.patch** - keys cached CUDA graph executables by
  the host graph identity plus a structural signature containing node count and
  endpoint tensor descriptors. This prevents allocator reuse from associating
  a new batch shape/topology with an incompatible executable while avoiding a
  full graph scan on every dispatch.

- **0012-cuda-streaming-cache-copies.patch** - recognizes inner-contiguous F32
  cache-tail views and materializes all batch planes with one pitched
  `cudaMemcpy2DAsync`; adds aligned float4 fast paths for gathering and
  scattering large indexed state-arena rows, including multi-plane K/V arenas.
  The shape/alignment guards keep all other COPY, GET_ROWS, and SET_ROWS cases
  on their existing kernels.

- **0013-cuda-cached-f16-cublas.patch** - optional cached-F16/cuBLAS path for
  skinny Q8 projections on NVIDIA SM80+. It expands immutable Q8 weights once,
  converts only the live activation, and retains FP32 accumulation/output.
  Runtime selection is controlled by `GGML_SKINNY_Q8_CUBLAS_F16` and its
  minimum-N threshold; cuBLAS chooses the implementation for the active GPU.
  Keep it opt-in because the F16 cache consumes additional device memory and
  the performance crossover depends on the GPU and physical batch size.

- **0014-cuda-relpos-extensions.patch** - extends fused relative-position
  attention for the cache-aware and offline FastConformer paths. The
  cache-aware path reads persistent K/V rows directly by state slot and
  circular head, then overwrites only the rows replaced by the current chunk.
  Register-resident NVIDIA SM80+ kernels cover the common R=0, 1, 3, 6, and 13
  streaming shapes for both Nemotron cache geometries, with exact-shape kernels
  retained where they are faster. The offline mask contract accepts both
  per-batch key-padding masks and `[key, query]` L/R masks. Set
  `GGML_CUDA_RELPOS_REGISTER_RESIDENT=0` before process start to
  disable the register-resident specializations without changing the direct
  circular-cache path.

- **0015-cuda-ctc-batch-fusions.patch** - reduces large-batch FastConformer
  overhead by fusing BatchNorm and BatchNorm+transpose+SiLU graph patterns,
  extending SiLU and affine LayerNorm output conversion to F16, and folding
  bias and residual addition into the cached-F16 cuBLASLt projection. The
  eligibility checks preserve the unfused path for unsupported layouts,
  precisions, and GPUs.

- **0016-fix-batched-conv1d-layout.patch** - restores the batch and output-channel
  axes after the flattened Conv1D matrix multiplication. The upstream direct
  reshape interleaves those axes for batches larger than one; batch one keeps
  its original zero-copy path.

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
baseline, then copy in only the cached-F16 changes and the SM100 CMake target
correction and run `git diff` against that staged baseline for `CMakeLists.txt`
and `skinny-q8.cu`.

Patch 0014 is intentionally layered on top of 0001 and 0013. Apply and stage
patches 0001 through 0013 in a temporary worktree, copy the edited
`include/ggml.h`, `src/ggml.c`, and `src/ggml-cuda/fused-relpos-attn.cu` into
that worktree, then generate 0014 with `git diff` against the staged baseline.

Generate patches with `git diff` only (GNU `diff`/editors can strip the
leading space on blank context lines, which `git apply` rejects as corrupt).

To verify the complete series, apply every patch in order to a fresh worktree at
the pinned ggml commit, then recursively compare that tree with the live patched
submodule and confirm that all files match.
