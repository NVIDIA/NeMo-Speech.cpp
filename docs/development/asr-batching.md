# ASR batching

ASR supports bounded dynamic microbatching across concurrent recognition calls.
Compatible neural work is combined into wider GPU graphs while each caller
keeps its own stream, decoder state, transcript, and result. Batching is opt-in
because collecting a batch adds queueing latency to otherwise uncontended local
inference.

RNNT/TDT decoder stages use a work-conserving scheduler because their next
operation is conditional. Predictor requests split by recurrent-state bank,
and joint requests split by remaining frame count and optional bias. The
scheduler releases useful exact-compatible work without treating every active
decoder as eligible for every operation.

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
    offline_bucket_ms: 1000
```

| Key | Tuning effect |
|---|---|
| `enabled` | Enables batching; leave off for the lowest single-stream latency. |
| `max_batch_size` | Caps one physical neural microbatch, not total active streams. |
| `max_queue_delay_us` | Trades per-stage latency for more opportunities to combine compatible work. |
| `max_queue_depth` | Bounds pending work and provides backpressure. |
| `ingress_cohort_delay_us` | Aligns streaming audio arrivals before frontend and encoder work. |
| `state_arena_slots` | Reserves recurrent/cache state rows; provision at least the maximum concurrent stateful streams. |
| `offline_bucket_ms` | Silence-pads offline utterances to a duration multiple so similar lengths can share graph shapes; `0` disables bucketing. |

More streams than `max_batch_size` are processed in multiple waves. Increasing
the cap or either delay does not guarantee better throughput; tune them on the
target GPU with the expected request cadence and audio chunk size.
Offline bucketing is useful only for concurrent offline workloads, and its
padding cost grows with the bucket size. Start with a modest value such as
`1000` ms and measure it on the expected duration distribution.

The gRPC and HTTP streaming adapters opt into ingress coordination. Direct
library streams and benchmark calls do not, so they avoid the transport
alignment delay and continue to measure the native pipeline.

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
metrics are needed for runtime development; the tool prints its stage metrics
unconditionally. A sustained realtime workload must keep paced chunk latency
below the incoming chunk duration; otherwise work accumulates. For the gRPC
server, set `NEMO_SPEECH_BATCH_METRICS=1` before process start to print
formed-batch, release-reason, queue-wait, compatibility, and execution
summaries.

## Runtime constraints

- Only work with the same graph-shaping dimensions and options can share a
  microbatch. Incompatible work remains queued for a separate graph.
- A transducer decode wave is admitted once at the start of `step()`. The
  predictor/joint scheduler dispatches when one exact-compatible group reaches
  physical capacity, all admitted decoders are queued or selected, or the
  oldest compatible group reaches its deadline. It selects the largest ready
  group instead of waiting for an impossible batch of all active decoders.
- Stateful RNNT/TDT encoder and decoder caches, plus VAD recurrent state, use
  indexed device rows. Exhausting `state_arena_slots` rejects new stateful work
  instead of silently reusing another stream's state. Released rows are marked
  dirty and cleared in coalesced ranges before their next use.
- CUDA streaming and offline recognition use the GPU frontend even when
  batching is disabled. Batching combines compatible frontend work as well as
  encoder, predictor, joint, VAD, and PnC work.
- Backend submission remains serialized. Bounded decoder execution slots
  overlap host packing and dependency wakeups, but the backend compute mutex
  still serializes each ggml graph. Throughput improves through fewer, wider
  submissions rather than concurrent access to ggml's shared scheduler.
- Disabling batching keeps the scalar path and avoids its queue-delay cost.

For planar-Q8 kernel behavior, diagnostic environment switches, and patched
versus stock ggml builds, see [ggml patches](ggml-patches.md). The complete
runtime key reference is in [ASR configuration](../asr/configuration.md).
