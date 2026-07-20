# ASR batching

ASR supports bounded dynamic microbatching across concurrent recognition calls.
Compatible neural work is combined into wider GPU graphs while each caller
keeps its own stream, decoder state, transcript, and result. Batching is opt-in
because collecting a batch adds queueing latency to otherwise uncontended local
inference.

## CUDA throughput workflow

High-throughput CUDA operation involves three separate choices: the build, the
model layout, and the runtime batching policy.

### 1. Build the patched CUDA runtime

The `cuda-asr` preset enables CUDA graphs and the patched FastConformer CUDA
optimizations:

```bash
scripts/configure.sh cuda-asr
cmake --build --preset cuda-asr
```

Use `cuda-server` instead when the HTTP API and playground are needed. The
server preset builds the same ASR runtime without gRPC; use `cuda-full` or set
`NEMO_SPEECH_BUILD_GRPC=ON` when `riva_server` is also needed. Pass
`-DCMAKE_CUDA_ARCHITECTURES=...` during configuration only when targeting a
specific GPU architecture.

### 2. Use planar Q8 encoder weights

The default Q8 layout is portable across backends. For a CUDA-only throughput
artifact, convert encoder weights to the tensor-planar layout used directly by
the patched batched kernels:

```bash
python3 convert_model.py model.nemo \
  --outfile model.planar.q8_0.gguf \
  --outtype q8_0 \
  --q8-layout planar
```

Planar Q8 requires the patched CUDA runtime. Keep a default block-layout model
for CPU, Metal, Vulkan, or unpatched ggml builds.

### 3. Enable and size batching

Configure batching under `asr.batching`. For example:

```yaml
asr:
  batching:
    enabled: true
    max_batch_size: 32
    max_queue_delay_us: 5000
    max_queue_depth: 2048
    ingress_cohort_delay_us: 20000
    state_arena_slots: 32
```

| Key | Tuning effect |
|---|---|
| `enabled` | Enables batching; leave off for the lowest single-stream latency. |
| `max_batch_size` | Caps one physical neural microbatch, not total active streams. |
| `max_queue_delay_us` | Trades per-stage latency for more opportunities to combine compatible work. |
| `max_queue_depth` | Bounds pending work and provides backpressure. |
| `ingress_cohort_delay_us` | Aligns streaming audio arrivals before frontend and encoder work. |
| `state_arena_slots` | Reserves recurrent/cache state rows; provision at least the maximum concurrent stateful streams. |

More streams than `max_batch_size` are processed in multiple waves. Increasing
the cap or either delay does not guarantee better throughput; tune them on the
target GPU with the expected request cadence and audio chunk size.

## Measure the result

The unified benchmark loads one recognizer and automatically enables and sizes
batching for the highest requested concurrency:

```bash
build/cuda-asr/bin/nemo-speech bench asr audio.wav \
  --model model.planar.q8_0.gguf \
  --mode stream \
  --concurrency 1,8,16,32 \
  --json
```

Compare RTFx and utterances per second across concurrency levels. The command
also reports transcript mismatches against the first result observed for each
input.

`bench_asr_batching` remains available with
`NEMO_SPEECH_BUILD_TOOLS=ON` when paced chunk latency or per-stage batch
metrics are needed for runtime development. A sustained realtime workload must
keep paced chunk latency below the incoming chunk duration; otherwise work
accumulates. The developer tool is not required for normal throughput tuning.

## Runtime constraints

- Only work with the same graph-shaping dimensions and options can share a
  microbatch. Incompatible work remains queued for a separate graph.
- Stateful RNNT/TDT encoder and decoder caches, plus VAD recurrent state, use
  indexed device rows. Exhausting `state_arena_slots` rejects new stateful work
  instead of silently reusing another stream's state.
- CUDA streaming and offline recognition use the GPU frontend even when
  batching is disabled. Batching combines compatible frontend work as well as
  encoder, predictor, joint, VAD, and PnC work.
- Backend submission remains serialized. Throughput improves by submitting
  fewer, wider graphs rather than running unrelated ggml graphs concurrently.
- Disabling batching keeps the scalar path and avoids its queue-delay cost.

For planar-Q8 kernel behavior, diagnostic environment switches, and patched
versus stock ggml builds, see [ggml patches](ggml-patches.md). The complete
runtime key reference is in [ASR configuration](../asr/configuration.md).
