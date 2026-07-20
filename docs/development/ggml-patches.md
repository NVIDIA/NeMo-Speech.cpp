# ggml patches

Edge-only changes to the vendored `ggml` submodule are kept as patches under
[`ggml-patches/`](../../ggml-patches/) so the submodule stays pinned to clean
upstream (`ggml-org/ggml`). For a patched CUDA build, apply them after checking
out or updating the submodule and **before** configuring CMake:

```bash
git submodule update --init ggml
scripts/apply-ggml-patches.sh        # applies patches in filename order
```

`apply-ggml-patches.sh` uses `git apply`, skips patches it can identify as
already applied, and applies new patches in filename order. Later patches may
build on files changed by earlier patches; 0006 carries the dispatch wiring for
the ops/kernels added by 0001/0003/0005. Docker builds and
`scripts/configure.sh` apply them automatically; run the script explicitly
before a raw CUDA CMake configuration.

| patch | what it adds | gate |
|---|---|---|
| `0001-fused-relpos-attn.patch` | `GGML_OP_FUSED_RELPOS_ATTN` - fused FastConformer relative-position attention (content + rel-shift + scale + mask + softmax + context in one CUDA kernel); stride-general operands + `merge_heads` output for the copy-free cache-aware streaming path | `NEMO_SPEECH_FUSED_RELPOS_ATTN` |
| `0002-nvfp4-warp-quantizer.patch` | single-pass warp-cooperative NVFP4 MMQ activation quantizer (Blackwell-1200 native-FP4 path) | `NEMO_SPEECH_NVFP4_WARP` |
| `0003-norm-mul-add-fusion.patch` | fused LayerNorm (`norm_mul_add_f32`): NORM + affine gamma/beta in one launch | dispatch in 0006 |
| `0004-conv2d-dw-f16-kernel.patch` | F16-kernel depthwise conv2d (direct CUDA kernel vs im2col+GEMM) | - |
| `0005-skinny-q8-gemm.patch` | default-on `skinny-q8` Q8_0×F32 GEMM specialized for streaming-encoder skinny activations (9 ≤ N ≤ 64); accepts tensor-planar Q8 weights, opt-in dense outer-batch dispatch | disable with `GGML_SKINNY_Q8=0`; `GGML_SKINNY_Q8_OUTER_BATCH` |
| `0006-cuda-dispatch-wiring.patch` | eligibility + dispatch wiring for the ops/kernels above; tensor-planar Q8 layout (`GGML_TENSOR_FLAG_Q8_PLANAR`, planar MMVQ vec-dot) and the Q8 narrow bias/SiLU epilogue fusion | `GGML_CUDA_Q8_NARROW_EPILOGUE` (default on) |
| `0007-magpietts-nanocodec.patch` | MagpieTTS / NanoCodec CUDA ops + keyed multi-CUDA-graph cache with idle eviction | `GGML_CUDA_GRAPH_SWEEP_MS`, `GGML_CUDA_GRAPH_EVICT_AFTER_MS` |
| `0008-cublas-bf16-projections.patch` | flattens shared-weight outer dimensions into one cuBLAS GEMM and fuses BF16 bias/SiLU conversion epilogues | shape eligibility; native BF16 epilogues require NVIDIA SM80+ |
| `0009-fastconformer-cuda-fusions.patch` | sigmoid GLU, Macaron scale-add, BF16 LayerNorm output, and the 1024-wide LayerNorm specialization | `NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS`; BF16 SM80+, LayerNorm specialization SM90+ |
| `0010-cuda-pad-large-batch-grid.patch` | flattens CUDA PAD launches into `grid.x`, avoiding the 65,535-block `grid.z` limit at ASR batch sizes ≥256 | always active on CUDA PAD |
| `0011-cuda-graph-shape-key.patch` | separates CUDA graph executables by host graph identity plus structural shape/topology signature | CUDA graphs enabled; architecture-neutral correctness fix |
| `0012-cuda-streaming-cache-copies.patch` | pitched F32 cache-tail copies plus aligned float4 gather/scatter for large indexed state rows and paired K/V planes | guarded CUDA COPY/GET_ROWS/SET_ROWS shapes; all other layouts keep the generic kernels |
| `0013-blackwell-sm100-cached-f16.patch` | cached-F16 real-cuBLAS skinny-Q8 path | compiled only for an SM100a target; exact-SM100 runtime check |

See [`ggml-patches/README.md`](../../ggml-patches/README.md) for the full
per-patch design notes.

## Building against patched vs stock ggml

Whether the build assumes a patched ggml is a CMake option, **`NEMO_SPEECH_GGML_PATCHED`**
(default `ON`). It exists because the ASR encoder directly references two
patch-only constructs - the fused rel-pos attention op (`0001`) and F16
depthwise-conv behaviour (`0004`) - so the build has to know whether those
symbols / semantics are present.

```bash
# Patched ggml (default): apply the patches first, then build.
scripts/apply-ggml-patches.sh
cmake -S . -B build -DGGML_CUDA=ON               # NEMO_SPEECH_GGML_PATCHED=ON

# Stock upstream ggml (patches NOT applied): opt out.
cmake -S . -B build -DGGML_CUDA=ON -DNEMO_SPEECH_GGML_PATCHED=OFF
```

With `NEMO_SPEECH_GGML_PATCHED=OFF` the encoder uses only stock ggml ops
(unfused rel-pos attention, `ggml_conv_1d_dw` im2col lowering) - correct on every
backend and on unpatched ggml, at some latency cost on the streaming path. Two
dependent options follow it (all `ON` only when `GGML_CUDA` **and**
`NEMO_SPEECH_GGML_PATCHED` are on, else forced `OFF`):

| option | default | effect when ON |
|---|---|---|
| `NEMO_SPEECH_GGML_PATCHED` | `ON` | build expects the ASR patches (fused rel-pos op, F16 dw-conv) |
| `NEMO_SPEECH_FUSED_RELPOS_ATTN` | `ON`¹ | encoder uses the fused rel-pos attention CUDA op |
| `NEMO_SPEECH_DIRECT_DW_CONV` | `ON`¹ | encoder uses the direct CUDA depthwise-conv kernel |
| `NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS` | `ON`¹ | encoder emits patched CUDA sigmoid-GLU and BF16-fusion graph patterns |

¹ Defaults `ON` only with `GGML_CUDA=ON` + `NEMO_SPEECH_GGML_PATCHED=ON`;
forced `OFF` otherwise (a STATUS line reports the fallback). They can be toggled
`OFF` independently for debugging or output-comparison bisection even on a
patched build.

The remaining patches (NVFP4/skinny-q8 kernels, norm and BF16 fusions) do not
create direct source-level dependencies in the ASR runtime. When the patched
ggml sources are present, however, their own eligibility checks may still
optimize ordinary ggml operations even if `NEMO_SPEECH_GGML_PATCHED=OFF`.
Use a pristine ggml checkout plus `NEMO_SPEECH_GGML_PATCHED=OFF` for a true
stock-upstream comparison.

`skinny-q8` is enabled by default. Its tensor-core accumulation order differs
from stock MMQ, so eligibility is based on the logical per-sequence width, not
the flattened outer batch size; singleton and dynamic-batch execution therefore
cannot select different kernels solely because B changed. Set
`GGML_SKINNY_Q8=0` for performance/correctness bisection. For high-concurrency
CUDA ASR, the preferred path is a GGUF converted with `--q8-layout planar`;
this covers ordinary projections and fused attention QKV. Wider outer batches
then dispatch to skinny-Q8 automatically. `GGML_SKINNY_Q8_OUTER_BATCH=1` is the
opt-in experiment for a portable block-layout GGUF, not a requirement for
planar Q8. See the
[CUDA throughput workflow](asr-batching.md#cuda-throughput-workflow).

The Blackwell cached-F16 path is absent from builds whose CUDA architecture
list does not contain feature-specific `100a`, and additionally checks that the
active device is exactly SM100 before dispatch.
