# Build switches, runtime knobs, and diagnostics

Every switch that changes which code path runs, in one place. Defaults are what
you want; the rest exist for bisection and debugging.

## Build

| CMake option | Default | Effect |
|---|---|---|
| `NEMO_SPEECH_GGML_PATCHED` | `ON` (`cuda-*`, `cpu-*` presets), `OFF` (`metal-*`, `vulkan-*`) | Apply [`patches/`](../../patches/README.md) to llama.cpp and ggml and, on CUDA, use the fused kernels. `OFF` builds pristine upstream llama.cpp with stock ggml operations. |
| `NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR` | unset | Build ggml and llama.cpp from this tree as-is (for example a `scripts/llama-patches.sh edit` worktree). |
| `GGML_CUDA_GRAPHS` | `ON` | Capture and replay CUDA graphs (upstream option; this project turns it on). |
| `GGML_LLAMAFILE` | `ON` | Tiled CPU matrix multiplication (upstream option; this project turns it on). |
| `NEMO_SPEECH_CUBLAS_SHIM` | `OFF` | Build the drop-in cuBLAS replacement ([cuBLAS shim](cublas-shim.md)). |

Component options (`NEMO_SPEECH_BUILD_*`, `NEMO_SPEECH_WITH_*`) are listed in
[the build guide](../build.md). `NEMO_SPEECH_WITH_NMT` and `NEMO_SPEECH_WITH_GRPC`
are deprecated spellings of `NEMO_SPEECH_BUILD_NMT` and `NEMO_SPEECH_BUILD_GRPC`.

## Runtime (patched CUDA backend)

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_GRAPH_EVICT_AFTER_MS` | `10000`; `0` once a server loads TTS, and for VoiceChat | Drop CUDA graphs idle for this long; `0` keeps them. Read when a CUDA backend is created. |
| `GGML_SKINNY_Q8=0` | enabled | Disable the skinny Q8_0 GEMM for block Q8_0 weights. Planar Q8 weights always use it. |
| `GGML_SKINNY_Q8_INPLACE=0` | in place; `0` when ASR and NMT share a process | Keep repacked skinny-Q8 weights in a separate buffer. |

Upstream switches that are useful for bisecting a CUDA problem:
`GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_DISABLE_FUSION=1`, and
`GGML_SCHED_DEBUG=2`.

## Other variables

- `NEMO_SPEECH_MODEL_DIR`, `NEMO_SPEECH_MODEL_INDEX`, `NEMO_SPEECH_HF_BASE_URL`: the CLI model store.
- `NEMO_SPEECH_<CONFIG_KEY>`: overrides one configuration key.
- `S2S_*`: VoiceChat tuning; see [VoiceChat configuration](../s2s/configuration.md).
- `MAGPIETTS_LOGIT_DUMP` and `MAGPIETTS_FORCE_CODES`: file paths used for MagpieTTS parity testing.
- `EDGE_SHIM_TRACE_SHAPES`: logs every call in the cuBLAS shim.

## Backend coverage

`check_backend_coverage` loads an ASR GGUF and exercises the frontend and
encoder Sessions used by its CTC or streaming-transducer path, including the
compact CTC head, RNNT/TDT predictor and joint, and cache-aware encoder when
applicable. It then prints their per-op backend assignment.
Use it to catch **silent CPU fallbacks** when enabling a new GPU backend or
updating llama.cpp - a single fallback op mid-graph adds a GPU↔CPU roundtrip per
audio chunk and can significantly increase streaming latency. The lazy offline
transducer path is outside this diagnostic's coverage.

```bash
scripts/configure.sh cuda-asr -DNEMO_SPEECH_BUILD_TOOLS=ON
cmake --build --preset cuda-asr --target check_backend_coverage
build/cuda-asr/bin/check_backend_coverage \
  nemotron-speech-streaming-en-0.6b.q8_0.gguf --gpu 0
# --gpu N    GPU device index (default 0). -1 forces CPU.
```

Append `--diar diar_streaming_sortformer_4spk-v2.q8_0.gguf` to exercise the
optional Sortformer Session in the same run.

Sample output:

```
== CacheStreamRunner cache-aware encoder Session (RNNT) ==
backend summary:
  CUDA0: 2837 nodes
  CPU:     0 nodes
sample (first 20 nodes):
  RESHAPE  encoder.pre_encode.conv.5.weight (reshaped)  -> CUDA0
  ...
CPU-fallback ops (none) ✓
```

Exit code is 0 when no GPU-targeted Session has ops on CPU and nonzero
otherwise, so scripts can use it as a gate.
