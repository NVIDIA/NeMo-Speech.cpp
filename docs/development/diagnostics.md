# Backend coverage diagnostic

`check_backend_coverage` loads an ASR GGUF and exercises the frontend and
encoder Sessions used by its CTC or streaming-transducer path, including the
compact CTC head, RNNT/TDT predictor and joint, and cache-aware encoder when
applicable. It then prints their per-op backend assignment.
Use it to catch **silent CPU fallbacks** when enabling a new GPU backend - a
single fallback op mid-graph adds a GPU↔CPU roundtrip per audio chunk and can
significantly increase streaming latency. The lazy offline transducer path is
outside this diagnostic's coverage.

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
