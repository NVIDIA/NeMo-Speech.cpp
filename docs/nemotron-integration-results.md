# Nemotron native backend integration results

This report records local integration checks and performance samples from
October 15, 2026. The C++ frontend successfully serves the retained native
Nemotron backend, including concurrent streaming and cancellation. B512
startup latency remains higher than the legacy frontend in the same-host
control. These samples do not establish production readiness, acoustic
quality, or a performance improvement over the legacy deployment.

Use the [build and run guide](nemotron-megakernel.md) for reproducible setup and
the [measurement instructions](nemotron-megakernel.md#validate-and-measure) for
client commands and metric definitions. Model weights, tokenizer assets,
normalization grammars, and generated validation receipts are supplied
separately from the repository.

## Architecture and provenance

The request path is C++ HTTP frontend → C++ synthesizer → Python bridge →
retained Python native controller → CUDA megakernels. The legacy Rust frontend
is not part of this request path. The bridge runs outside the sealed native
package, so frontend integration does not alter that package's source inventory.

The C++ frontend multiplexes requests over one Unix socket using bounded
per-request audio queues. The bridge validates the tokenizer against the model
seal, applies the retained native English normalizer when configured, tokenizes
without added special tokens, and appends EOS `131073`. It submits the resulting
text and IDs through the controller's existing MessagePack transport. Native
PCM slabs are validated and copied before acknowledgment; the default bridge
path converts them to PCM16 with the legacy asymmetric scaling, rounding, and
saturation rules. The native controller retains GPU state, scheduling, kernel
dispatch, codec execution, and slot lifecycle ownership.

| Component | Revision or identity |
|---|---|
| C++ repository base | `083861c241061562b8ec29eb06d7f6cdc3af8775` |
| Retained native source | `2cc27a4e6b06e5e6bf33020ce4faa1fc00bfd25b` |
| Model checkpoint SHA-256 | `8bb6bfeb9cd4335f014a70bf87059af58945399be454a4c4e353a983d09eeffa` |
| Tokenizer SHA-256 | `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7` |
| Ten-text benchmark corpus SHA-256 | `8bae1d20bfd5d17bee363afdbd6d25e0167d4bcc5c11ea6663d75c3b8d9450a2` |

A fresh byte comparison verified **579 imported files unchanged** against the
retained source revision: 468 kernel files including the recipe inventory, 65
Python runtime files, 38 native normalizer files, and eight deployment/build
files. The two additional copies of the retained root license also match their
source. The inventory counts each destination once and preserves the vendor
licenses within the imported trees. Recompilation does not imply byte-identical
shared libraries.

These components retain their original terms; see the
[license scope](../THIRD_PARTY_NOTICES.md#imported-nemotron-native-deployment-closure).
Local qualification does not resolve redistribution approval.

The tested GPU was an H100 80 GB HBM3 with 132 SMs, compute capability 9.0,
without MIG, using `sm_90a` kernels. The v3 runtime reported CUDA 13.2,
PyTorch `2.12.0a0+5aff3928d8.nv26.05`, tokenizers 0.23.1, MessagePack 1.2.2,
and NumPy 2.3.5. TensorRT and TensorRT-LLM imports were unavailable in that
runtime. The [Docker recipe](../docker/Dockerfile.nemotron) pins its base images
and builds the frontend, normalizer, kernels, and native package from source.

## Qualification completed

- The native worker reached qualification readiness at widths 1, 32, and 512.
  Its readiness receipt explicitly reports `production_enabled: false`.
- B1 returned two successful HTTP PCM streams with identical bytes and sample
  counts on repeated input. This is repeatability within the new stack, not a
  cross-frontend acoustic comparison.
- At B32 and B512, native lifecycle checks passed EOS completion, cancellation
  after the first chunk, exact replay after slot reuse, and replay in the last
  physical slot. The replay PCM hashes and chunk sample counts matched within
  each width.
- Eighteen bridge/launcher unit tests passed, covering multiplexing, token
  preparation, copy-before-ACK ownership, malformed output, slow-consumer
  failure, cancellation, tokenizer identity, PCM conversion, and bounded child
  shutdown. HTTP tests additionally exercise streaming before completion,
  WAV metadata, disconnect cancellation, and backend failure handling.
- The v6 container build passed all 13 registered CTest checks.
- The actual tokenizer produced identical IDs for 38 corpus and edge cases
  when changing from `encode` to singleton `encode_batch`. All 35 cases within
  the token budget also produced byte-identical MessagePack submit payloads.

The tests and lifecycle checks do not provide new MOS, intelligibility, speaker
similarity, ASR error-rate, or other acoustic scores. No such scores are claimed.
Cross-frontend exact audio parity also remains separate from the timing samples
below: lane ordering and effective fixed-cohort seeds differed between runs.

## v3 performance samples

Each v3 mode below had **one server boot and three measured trials**, with text
normalization enabled and the retained ten-text corpus. Fixed-cohort trials
submitted one synchronized wave of 512 requests; in-flight trials submitted
2,048 requests at concurrency 512 with refills. Each client used two warmup
rounds. The two load patterns and client timing definitions differ and should
not be compared as equivalent latency experiments.

Values below are the equal mean of the three trial statistics, not percentiles
of a pooled request set. Peak VRAM is the maximum observed device memory usage.

| B512 sample | Successful requests | TTFA p50 | TTFA p95 | TTFA p99 | RTFx | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Legacy same-host fixed-cohort control | 1,536 / 1,536 | 180.00 ms | 191.70 ms | 196.63 ms | 532.30 | 28,615 MiB |
| C++/bridge v3 fixed-cohort | 1,536 / 1,536 | 276.33 ms | 298.18 ms | 306.37 ms | 481.18 | 28,587 MiB |
| C++/bridge v3 in-flight | 6,144 / 6,144 | 146.70 ms | 349.25 ms | 374.34 ms | 1,267.48 | 24,673 MiB |

The fixed-cohort p95 difference is **106.48 ms** in this sample. The legacy
control used the same host, native package, normalizer library, corpus, and CPU
role allocations, with a frozen legacy frontend binary. All ten normalized
strings, their token IDs, and per-cohort multiplicities matched. The legacy
frontend performed an additional B1 readiness synthesis, and effective seeds
and lane ordering differed; generated durations therefore differed too. These
controls help locate host overhead but do not establish an audio-matched
throughput comparison. No same-host legacy in-flight control was collected.

The v3 in-flight client recorded five playback underruns in its first trial and
none in the other two. The largest reported audio-buffer deficit was 18.64 ms.
Zero request failures should not be interpreted as zero playback interruption.

Earlier diagnostic receipts localized most of the startup difference to request
admission: the legacy worker received a B512 cohort over a median 116.35 ms,
versus 186.79 ms for one bridge diagnostic trial. Backend startup after the final
submission was about 2–3 ms in both arms; backend first-chunk time was about
64–67 ms for the legacy control and 70.46 ms for that diagnostic. These numbers
describe the diagnostic run, not a decomposition of the three-trial v3 table.

An isolated CPU preparation check on four physical cores with SMT and 16
preparation threads measured 140.03 ms for 512 requests with `encode`, and
134.18 ms with singleton `encode_batch`. It used the real tokenizer and grammar,
with `TOKENIZERS_PARALLELISM=false`. Reducing the preparation pool or retaining
the GIL during normalizer cleanup did not improve total preparation time.
Moving the bridge onto frontend cores also worsened the HTTP diagnostic.
No further host optimization is promoted by these checks.

## v4 optional float32 forwarding screen

The v4 screen enabled `NEMOTRON_BRIDGE_FORWARD_F32=1`. This optional path forwards
native float32 samples to C++ for PCM16 conversion; the public HTTP format
remains PCM16. The default bridge still performs PCM16 conversion in Python.
Each mode again used one server boot and three measured trials.

| B512 v4 sample | Successful requests | TTFA p50 | TTFA p95 | TTFA p99 | RTFx | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-cohort | 1,536 / 1,536 | 267.34 ms | 283.33 ms | 289.81 ms | 636.16 | 28,583 MiB |
| In-flight | 6,144 / 6,144 | 149.00 ms | 323.73 ms | 359.19 ms | 1,245.80 | 24,673 MiB |

The in-flight client reported no playback deficit responses in these three
trials. Fixed-cohort RTFx ranged from 470.34 to 795.44 across trials; the mean
alone does not establish a throughput gain. Fixed-cohort p95 TTFA remained
91.62 ms above the same-host legacy control. This screen does not promote the
optional transport to the default or establish acoustic equivalence.

## Final representative samples

The v6 image includes CLI seed propagation and bounded preparation cancellation.
All six arms completed with text normalization enabled and optional
`NEMOTRON_BRIDGE_FORWARD_F32=1`. The bridge default remains PCM16 forwarding.
The final tests used the image's freshly built CUDA libraries; all 15 packaged
shared-library hashes match the tested in-flight package. The fixed-cohort
package was assembled from the same build using the fixed serving policy.

Each arm used one fresh boot and three measured trials. Fixed-cohort B1/B32
used seven synchronized waves per trial, and B512 used one. In-flight trials
contained 32/128/2048 requests at concurrency 1/32/512. Warmups were excluded.
Numbers are equal means of trial statistics; latency and ITL are milliseconds.

| B | Mode | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Fixed cohort | 16.73 | 2.14 | 20.25 | 20.95 | 20.95 | 4.40 | 4.59 | 5,191 | 21 | 0 | — |
| 1 | In-flight | 17.50 | 2.51 | 19.88 | 20.86 | 20.99 | 4.39 | 4.54 | 5,199 | 96 | 0 | 0 |
| 32 | Fixed cohort | 259.35 | 34.61 | 32.18 | 48.82 | 49.54 | 3.92 | 4.73 | 10,335 | 672 | 0 | — |
| 32 | In-flight | 301.40 | 42.30 | 25.31 | 43.32 | 44.51 | 7.47 | 12.59 | 6,835 | 384 | 0 | 0 |
| 512 | Fixed cohort | 511.06 | 67.62 | 265.75 | 279.29 | 285.44 | 19.25 | 23.31 | 28,587 | 1,536 | 0 | — |
| 512 | In-flight | 1238.79 | 170.87 | 149.06 | 336.34 | 361.50 | 28.67 | 37.00 | 24,673 | 6,144 | 0 | 0 |

All **8,853 measured requests completed**. The in-flight client recorded zero
playback-deficit responses across its 6,624 measured requests. The historical
client does not report that metric; its dashes are not evidence of zero deficits.
The repository portable client additionally completed 64 B32 requests in each
mode, with no failures or measured playback deficits. Those smoke tests have
different timing definitions and are excluded from this table.

B512 fixed-cohort p95 remains **87.59 ms above** the legacy control from the
earlier same-host diagnostic comparison. This integration therefore does not
yet reproduce the legacy frontend's B512 latency. The diagnostic admission
gap is the main identified remaining host-path cost. The control and final
samples were not collected as matched boot pairs; no confidence interval or
performance-parity claim is made. No fresh acoustic campaign was performed.

The host used one H100 SXM 80 GB (132 SMs), driver 595.91.07, and an AMD EPYC
7413 CPU. The native worker and Python bridge shared physical cores 0–3 plus
SMT siblings 24–27; C++ HTTP used 4–7/28–31; clients used 8–11/32–35.
No GPU clock or power settings were changed. GPU telemetry and source/binary
identities accompany the raw observations.

Image index: `sha256:af3ce40c4f1ee973630da290fcf4f6d1ec96fb4b9f1c2d6ab0de7b03a5fb1474`.
Image config: `sha256:93e9f1ba8ce1e7ef094f13554bcf8147203f825dcfb391096630605ea1b19daf`.
The image contains no TensorRT or TensorRT-LLM installation used by this backend.
Its build passed 13 CTests, including the 18-case bridge/launcher suite.
Scalar/AVX2 PCM conversion also passed independent boundary/random-value and
ASan/UBSan checks. Remote CI, gRPC builds, other GPU geometries, and wider
batch configurations have not been validated by this campaign.

## Evidence inventory

The following names identify receipts relative to the external validation
directory. They are evidence identifiers, not bundled downloads or links to
machine-specific paths.

| Evidence | Receipt |
|---|---|
| Imported-file byte comparison | `frozen-import-inventory.json` |
| Qualified runtime imports | `runtime-v3.json`, `runtime-v6.json` |
| Native readiness and lifecycle | `b1-qualification/worker/NATIVE_READY.json`, `b32-qualification/worker/LIFECYCLE.json`, `b512-qualification/worker/LIFECYCLE.json` |
| B1 HTTP repeatability | `b1-execution.json` |
| Final registration and summary | `HTTP_REGISTRATION-final.json`, `final-summary.json`, `final-summary.md` |
| Final measurements and portable smoke tests | `http-final/b{1,32,512}-{fixed_cohort,inflight}/` |
| Image/tested library identity | `image-tested-package-v6.json`, `kernels-v6/BUILD.json` |
| v3 experiment registration | `HTTP_REGISTRATION-r3.json` |
| v3 fixed-cohort measurements | `http-r3/b512-fixed_cohort/historical.json` |
| v3 in-flight measurements | `http-r3/b512-inflight/inflight-{0,1,2}.json` |
| v4 optional-transport screen | `http-r4/b512-fixed_cohort/historical.json`, `http-r4/b512-inflight/inflight-{0,1,2}.json` |
| Legacy control and provenance | `http-legacy-control/COMPARISON.json`, `http-legacy-control/historical.json` |
| Admission diagnostics | `http-diag2/b512-fixed_cohort/bridge-trace.jsonl` and worker cohort receipts |
| CPU preparation and normalization controls | `tokenizer-qualification/encode-comparison.json`, `thread-sweep.json`, `normalizer-profile.json` |

Every timing arm retains its launch command, image identity, worker receipts,
client records, GPU telemetry, and shutdown result. Repeating a performance
claim requires additional independent boots and matched measurement conditions;
the current one-boot samples have no confidence intervals.
