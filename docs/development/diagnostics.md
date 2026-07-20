# Backend coverage diagnostic

`check_backend_coverage` loads an ASR GGUF, runs one inference step through every
Session (encoder, RNNT predictor + joint, cache-aware encoder), and prints the
per-op backend assignment. Use it to catch **silent CPU fallbacks** when
enabling a new backend (Vulkan / Metal / CPU) - a single fallback op
mid-graph is a GPU↔CPU roundtrip per audio chunk, which kills streaming latency.

```bash
scripts/configure.sh cuda-asr -DNEMO_SPEECH_BUILD_TOOLS=ON
cmake --build --preset cuda-asr --target check_backend_coverage
build/cuda-asr/bin/check_backend_coverage \
  nemotron-speech-streaming-0.6b.gguf --gpu 0
# --gpu N    GPU device index (default 0). -1 forces CPU.
```

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
