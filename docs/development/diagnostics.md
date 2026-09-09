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

Append `--diar diarization.gguf` to exercise an optional Sortformer V2 or V3
Session in the same run.

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

## Sortformer parity and benchmarking

`test_sortformer_parity` shares one shape-aware reference format for V2 and V3.
Generate references in a NeMo environment with the matching checkpoint and a
16 kHz mono WAV. The neural-step fixture covers cold, FIFO-only, and full-cache
inputs, including native/coarse probabilities and chunk embeddings:

```bash
python scripts/asr/dump_sortformer_reference.py \
  models/Nemotron-3-Diarization.nemo sample.wav reference.npz \
  --device cuda --steady-state --chunk 13 --lc 0 --rc 1 \
  --spkcache 264 --fifo 80 --update-period 40
python scripts/asr/export_ref_bins.py reference.npz reference-bins
build/cuda-asr/bin/test_sortformer_parity \
  models/Nemotron-3-Diarization.f32.gguf reference-bins --gpu
# Add --q8 when testing the Q8_0 conversion of the same checkpoint.
```

Reference generation disables TF32. V3 embedding comparisons use maximum absolute
error divided by the reference's maximum magnitude (floored at one): `1e-5`
for CPU F32, `1e-3` for CUDA F32 weights with F16 convolution, and `2e-2` for
Q8. Native/coarse probabilities use absolute limits of `5e-4`, `2e-3`, and
`2e-2`, respectively. Every comparison rejects non-finite values and incompatible
shapes. V2 F32 retains its existing `1e-2` relative-embedding and absolute-probability
gates. These numerical gates are not substitutes for dataset accuracy scoring.

Omit `--steady-state` for the existing V2 multi-chunk graph/state/frontend
reference. V3 full-stream correctness is covered separately by
`test_diar_streaming --batching-check`; neural-step parity alone does not measure
DER or cpWER. Model-dependent tests are explicit commands, not automatic CTests.

With `NEMO_SPEECH_BUILD_TOOLS=ON`, `bench_diar` measures a synthetic neural
step at explicit geometry and batch size:

```bash
build/cuda-asr/bin/bench_diar models/Nemotron-3-Diarization.q8_0.gguf \
  --gpu --batch-size 64 --warmups 5 --reps 20 \
  --chunk 13 --lc 0 --rc 1 --spkcache 264 --fifo 80
```

`median_ms` is host-observed whole-batch latency, including dispatch and output
copies, not GPU-event time. `per_item_ms` divides that latency by batch size;
it is not individual request latency. Report geometry, device, weight type,
backend patches, and observed batch size alongside results. RTFx here excludes
the audio frontend, state update, and ASR.

`eval_diarization` exports full native probability timelines for external DER
scoring. Its tab-separated manifest contains `id`, `wav`, `offset_seconds`, and
`duration_seconds`, with no header; WAV paths are relative to the process's
working directory unless absolute. IDs must be unique filename components.

```bash
build/cuda-asr/bin/eval_diarization MODEL.gguf inputs.tsv new-output-dir \
  13 1 264 80 40 32
# Positional values: chunk, right context, cache, FIFO, update, concurrency.
# Optional: --lc N (default 0), --cpu (default GPU).
```

Each successful recording publishes `ID.f32` atomically. `metadata.tsv` records
the native frame period, speaker count, geometry, and effective concurrency.
The tool rejects out-of-bounds audio regions rather than silently truncating
them, refuses existing output directories, and returns nonzero on failures.
It retains each active recording's full timeline for rescoring; memory therefore
grows with recording length and concurrency. Dataset labels, reference mapping,
collars, and DER/cpWER scoring belong to the evaluation harness, not this exporter.
