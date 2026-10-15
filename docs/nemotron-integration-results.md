# Nemotron native backend integration results

This report records local integration checks and performance samples from
October 15, 2026. The C++ frontend successfully serves the retained native
Nemotron backend, including concurrent streaming and cancellation. Initial
B512 host overhead and its subsequent repairs are retained below. The latest
matched direct-worker comparison measures 205.04 ms in-flight TTFA p95; it does
not establish the 200-ms target, production readiness, or fresh acoustic quality.

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

## Subsequent B512 host-path repair

The following campaign supersedes the v6 performance samples for the repaired
host path; the earlier observations above remain unchanged. With the same native
CUDA package, moving normalization and tokenization into the C++ frontend's
native preparation path reduced B512 HTTP TTFA p95 from **274.60 to 224.30 ms**
for fixed cohorts and from **335.23 to 229.95 ms** for in-flight requests. This
repairs most of the port's measured host overhead, but does not establish parity
with the legacy frontend or a change in neural execution speed.

### Repair and executable scope

The repaired path is C++ HTTP → native normalization/tokenization → validated
prepared-request transport → Python bridge → unchanged native controller and
CUDA. Preparation uses the retained English normalizer and a native tokenizer
binding, with the same trimming, punctuation, tokenizer assets, token budget,
and EOS rules. The bridge negotiates prepared-input support and validates token
IDs before submission. Its event loop no longer performs each request's text
preparation. This backend-specific path does not replace preparation for other
models.

The tested bundle also enables bounded bridge event-loop yielding, TCP_NODELAY
on this frontend's HTTP connections, and float32 forwarding with exact PCM16
conversion in C++. Both comparison arms already used float32 forwarding.
Consequently, the table attributes the improvement to the complete tested host
bundle; it does not assign independent HTTP gains to these mechanisms. Worker
affinity changes and process-pool preparation were investigated but are not
enabled by the selected configuration.

There are **no CUDA kernel, native controller, state-format, generation, or
chunk-schedule changes** in this comparison. Both arms use the exact retained v6
native package, built for `sm_90a` at widths 1, 32, and 512. Its kernel source
manifest SHA-256 is
`3976dd371b717d3a5c3b7bb3c2ec1e8dbdada2ffec6f4b9f5cee7f542f3ed222`.
The previously verified 15 packaged shared libraries are reused, rather than
substituted with independently rebuilt binaries. The fixed-cohort and in-flight
serving policies remain distinct, with the same policy package used by both
arms of each comparison.

| Tested image | Image index SHA-256 |
|---|---|
| v6 baseline | `af3ce40c4f1ee973630da290fcf4f6d1ec96fb4b9f1c2d6ab0de7b03a5fb1474` |
| v7 repaired host | `dea07919df447b9ca57ab15a390d5b592c95a3287d35fe8301554b4f8da6433d` |

The v7 image rebuilt the C++ frontend and native preparation component while
copying `/package` from v6. The C++ build passed all 13 CTests. The Python suite
ran 44 tests with three fixture-dependent skips, and a separate real-fixture
qualification ran nine tests successfully. The latter checks normalization and
token-ID agreement, native allocation ownership, and concurrent preparation.
The normalizer/tokenizer comparison includes the corpus and edge cases with
normalization both enabled and disabled. No fresh acoustic scoring was performed.

The complete source-build recipe remains `docker/Dockerfile.nemotron`, including
CUDA compilation. This host-only campaign deliberately canceled its redundant
CUDA rebuild and reused the qualified v6 package; it does not certify a newly
completed clean build of every component at v7. Remote CI, gRPC, other GPU
architectures, and widths beyond the recorded cases remain outside this result.

### Matched HTTP comparison

Each B512 arm and mode used **two fresh server boots and three measured trials
per boot**, ordered baseline/repaired/repaired/baseline. Fixed-cohort trials used
one synchronized 512-request wave; in-flight trials used 2,048 requests at
concurrency 512 with refills. Two warmup rounds were excluded. The ten-text
corpus, normalization, voice, generation cap, 2/3/4/5 → 8 chunk schedule, and
publication behavior were unchanged. The clients' existing timers were
preserved, including admission delay; no interval was subtracted.

CPU placement remained four physical cores plus SMT for the worker/bridge
(0–3/24–27), another four for C++ HTTP and preparation (4–7/28–31), and four for
the client (8–11/32–35). This is a **4+4 server-core comparison**. It does not
include the separate shared-eight-core frontend-placement control.

Statistics are equal means of per-trial values across both boots, not pooled
request percentiles. Latency and ITL are in milliseconds; peak memory is the
maximum telemetry after readiness, including warmups.

| B | Host | Mode | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | v6 baseline | Fixed cohort | 471.21 | 62.43 | 259.61 | 274.60 | 281.01 | 19.64 | 23.36 | 28,585 | 3,072 | 0 | — |
| 512 | v7 repaired | Fixed cohort | 543.67 | 72.74 | 203.77 | 224.30 | 231.48 | 19.51 | 23.39 | 28,621 | 3,072 | 0 | — |
| 512 | v6 baseline | In-flight | 1252.60 | 172.77 | 146.15 | 335.23 | 364.24 | 28.38 | 36.95 | 24,673 | 12,288 | 0 | 0 |
| 512 | v7 repaired | In-flight | 1263.37 | 174.26 | 147.26 | 229.95 | 249.26 | 28.23 | 36.61 | 24,673 | 12,288 | 0 | 0 |

All **30,720 measured B512 requests completed**. Both in-flight arms recorded
zero playback-deficit responses. The historical fixed-cohort client does not
measure playback deficits; the dashes do not imply zero.

| Mode | TTFA p95 change | Repaired / baseline | 95% matched interval for ratio |
|---|---:|---:|---:|
| Fixed cohort | −50.30 ms (−18.32%) | 0.8168 | [0.7576, 0.8621] |
| In-flight | −105.28 ms (−31.41%) | 0.6859 | [0.6453, 0.7216] |

Intervals use 20,000 paired hierarchical bootstrap draws, resampling matched
boot pairs and then trial positions within each pair. Two boot pairs on one GPU
provide limited uncertainty evidence, not independent hardware replication.
In-flight p50 changed little: this repair primarily reduced startup/tail
preparation delay in that workload.

Fixed-cohort RTFx must not be interpreted as a kernel speedup. Asynchronous
preparation changes lane order and therefore batch-seeded generated trajectories.
The baseline's six measured cohorts all reached 511 producer steps; repaired
cohorts ran for 483/511/511 and 435/511/275 steps. Length-limit terminations were
8/3 for baseline/repaired, respectively, with the remaining requests ending at
EOS. All 12,288 requests in each B512 in-flight arm ended at EOS. These
differences are retained in the result rather than removed from throughput.

### Legacy residual and B32 check

A fresh same-host legacy fixed-cohort control used one boot and three trials,
with the same four-plus-four server-core assignment and retained native package.
Its TTFA p95 was **204.19 ms**, versus the repaired port's **224.30 ms**: a
remaining **20.10 ms, or 9.84%**, difference. The legacy control is not a matched
two-boot arm and has different lane/seed trajectories, so this comparison has
no matched confidence interval and does not demonstrate performance parity.
The earlier 191.70-ms legacy result remains a historical observation above.

The repaired B32 regression check used one boot and three trials per mode.
Fixed cohorts retained seven waves per trial; in-flight trials contained 128
requests. It is a smoke/regression sample, not a new matched improvement claim.

| B | Mode | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | Fixed cohort | 233.72 | 31.28 | 32.68 | 45.16 | 46.35 | 3.90 | 4.73 | 10,335 | 672 | 0 | — |
| 32 | In-flight | 298.11 | 41.84 | 26.08 | 39.98 | 41.34 | 7.51 | 13.05 | 6,835 | 384 | 0 | 0 |

There are no new CER, WER, speaker-similarity, or other acoustic figures for
these arms. Exact text/token preparation tests and unchanged neural binaries
do not replace acoustic evaluation. Additional CPU-placement controls should
be reported separately from this frozen 4+4 comparison.

### Repair evidence inventory

The following receipts are relative to the external host-repair evidence
directory. They supplement the earlier validation inventory.

| Evidence | Receipt |
|---|---|
| Frozen campaign and independent summary | `FINAL_REGISTRATION.json`, `FINAL_COMPARISON.json`, `FINAL_COMPARISON.md`, `summarize_final.py` |
| Per-boot clients, telemetry, commands, termination records | `http-final/` |
| Fresh legacy control | `legacy-fixed-refresh/REGISTRATION.json`, `legacy-fixed-refresh/RECEIPT.json`, `legacy-fixed-refresh/historical.json` |
| Host-only build scope and image | `HOST_BUILD_SCOPE.json`, `Dockerfile.host-only`, `build-v7-host-only.log` |
| C++ and Python test receipts | `build-v7.log`, `tests-frontend-preparation-all.log`, `native-v7-qualification.log` |
| Native preparation source/fixture qualification | `preparation-process/native-qualification-formatted.log` |
| Retained negative process-pool experiment | `preparation-process/result.json` |

## Subsequent shared-eight-core placement

A separate placement experiment allowed the C++ frontend and its native
preparation workers to use **all eight existing server cores**, 0–7/24–31,
while keeping the worker and bridge on 0–3/24–27. The total service allocation
remains eight physical cores plus SMT. This relaxes the earlier rigid split;
it does not add CPU cores or change GPU execution. The v7 image, native package,
client commands, normalization, and generation contract were unchanged.

After an exploratory placement screen, two fresh boots per mode each ran three
measured trials with the same warmups and request counts as the comparison
above. These measurements were collected **after** that comparison, without
interleaving another baseline. Differences against the earlier arms are
therefore descriptive chronological comparisons, not matched randomized
estimates of the placement's isolated effect.

| B | Mode | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | Fixed cohort | 621.76 | 83.45 | 160.36 | 176.80 | 184.06 | 19.70 | 24.52 | 28,613 | 3,072 | 0 | — |
| 512 | In-flight | 1310.30 | 180.73 | 140.95 | 224.45 | 239.73 | 27.03 | 36.06 | 24,673 | 12,288 | 0 | 0 |

Fixed-cohort boot means were **173.77 and 179.84 ms** TTFA p95; in-flight boot
means were **217.30 and 231.61 ms**. Within-arm hierarchical bootstrap intervals
from 20,000 draws were [170.21, 182.93] ms and [216.16, 235.40] ms, respectively.
These describe the limited two-boot sample on one GPU and do not account for
placement selection or chronological effects.

The fixed-cohort mean is 47.49 ms below the earlier repaired 4+4 sample and
27.39 ms below the fresh 204.19-ms legacy control. The in-flight mean is 5.49 ms
below the repaired 4+4 sample; its boot-to-boot variation prevents attributing
that small difference confidently to placement. Relative to the earlier v6
baseline, the observed complete-stack p95 differences are −97.80 ms fixed and
−110.78 ms in-flight. No paired confidence interval is claimed for these
chronological contrasts.

All **15,360 measured requests completed**. Fixed cohorts recorded 3,071 EOS
terminations and one length limit; cohort steps were 436/387/332 and
511/363/307. All 12,288 in-flight requests ended at EOS with zero measured
playback-deficit responses. As above, fixed-cohort RTFx varies with generated
trajectories and cap tails, and its client does not record playback deficits.

No fresh acoustic scores or measured cross-frontend PCM identity are claimed.
The in-flight population matches normalized texts, sample counts, and
termination reasons between the original baseline and repaired arms. Measured
request receipts do not contain PCM hashes; hashes in HTTP preparation receipts
belong to pre-serving qualification and cannot establish exact audio identity
for the timed requests.

This separate sample is recorded by
`HTTP_REGISTRATION-shared-cpu-final.json`, `shared-cpu-final/`,
`SHARED_CPU_COMPARISON.json`, `SHARED_CPU_COMPARISON.md`, and
`summarize_shared_cpu.py` in the external host-repair evidence directory. The
frozen 4+4 comparison above remains the matched evidence for the code repair.

## Fresh legacy in-flight control

A subsequent legacy Rust frontend control completed one fresh boot with three
unchanged `tts_perf_client` trials: 2,048 measured requests at concurrency 512
per trial, after two warmup rounds. Its TTFA p95 was **212.93 ms**, versus
**224.45 ms** for the selected C++ host path above. The remaining point-estimate
difference is **11.53 ms, or 5.41%**. The selected fixed-cohort point estimate
remains 176.80 ms versus the fresh legacy control's 204.19 ms.

| Frontend | Mode | Boots | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy Rust | Fixed cohort | 1 | 490.04 | 64.74 | 191.48 | 204.19 | 209.66 | 18.88 | 22.13 | 28,615 | 1,536 | 0 | — |
| Selected C++ | Fixed cohort | 2 | 621.76 | 83.45 | 160.36 | 176.80 | 184.06 | 19.70 | 24.52 | 28,613 | 3,072 | 0 | — |
| Legacy Rust | In-flight | 1 | 1308.91 | 180.54 | 141.35 | 212.93 | 224.95 | 27.14 | 35.26 | 24,673 | 6,144 | 0 | 0 |
| Selected C++ | In-flight | 2 | 1310.30 | 180.73 | 140.95 | 224.45 | 239.73 | 27.03 | 36.06 | 24,673 | 12,288 | 0 | 0 |

Both implementations used the same H100, native package, model, normalization
assets, corpus, and client timing definitions. Both had eight physical service
cores plus SMT. Legacy kept frontend and worker on disjoint four-core sets;
selected C++ allowed frontend preparation to share all eight cores while its
worker/bridge retained four. Legacy also performed one excluded readiness
synthesis before the registered client warmups. These are chronological samples
with **one legacy boot versus two C++ boots**, not interleaved matched pairs.
There is no resolved parity or superiority claim, and no matched confidence
interval is assigned to the legacy comparison.

All 6,144 measured legacy in-flight requests ended at EOS with no failures or
playback-deficit responses. Each selected C++ in-flight boot matched that
control's complete normalized-text/sample-count/termination population and
**44,544.72 seconds of generated audio** across 6,144 measured requests. This
comparison explicitly excludes legacy readiness and each client's warmups.
It verifies population and duration agreement, not PCM identity or acoustic
quality. Fixed-cohort lane-order/seed trajectories still differ, so their RTFx
difference cannot be attributed solely to frontend speed.

The additional receipts are `legacy-inflight-refresh/REGISTRATION.json`,
`legacy-inflight-refresh/RECEIPT.json`, the three client results under
`legacy-inflight-refresh/`, and the independently reconstructed
`LEGACY_COMPARISON.json`, `LEGACY_COMPARISON.md`, and `summarize_legacy.py` in
the external host-repair evidence directory.

### Launcher and small-width checks

The optional `--native-cpus` and `--frontend-cpus` controls were subsequently
added to the Nemotron launcher. Image `nemo-speech-cpp:nemotron-v7.1` changes
only that launcher relative to v7; CUDA, the C++ frontend, and native text
preparation binaries are identical. The launch options reproduce the selected
CPU masks before child threads initialize and record their effective affinity.
No placement is imposed when the options are absent.

Live checks using these options measured B1 fixed-cohort p95 **21.11 ms**
(21 measured requests) and B32 in-flight p95 **33.40 ms** (384 measured requests),
with three trials per fresh boot and normalization enabled. All requests
completed, and B32 recorded zero playback deficits. These are regression
samples, not matched cross-width performance claims. All existing child-thread
affinities matched the requested role masks. Commands, results, and launch
receipts are retained in `launcher-regression/`.

Validation includes 13 CMake tests, six Rust tests, and 51 Python controller,
bridge, and launcher tests (three optional model-fixture tests skipped in that
generic invocation). The separate nine-test model/native-library invocation
passed with those fixtures available. Formatting and repository hooks passed.
An initial test run inherited the image's F32 forwarding default while one
test expected PCM16; the test now explicitly selects its PCM16 control, while
the separate F32 test retains coverage of the selected path. Runtime behavior
was unchanged by that test correction. Logs are retained in `build-v7.log`,
`tests-final.log`, `tests-final-r2.log`, `native-v7-qualification.log`, and
`precommit-final.log`.

## Subsequent direct-worker transport

The opt-in `--direct-worker` path removes the Python compatibility bridge from
request submission and audio delivery. C++ prepares and validates text, sends
the native controller's existing MessagePack protocol, and copies shared-memory
audio into owned PCM16 before acknowledging slab reuse. The native controller
still owns admission, request slots, GPU work and retirement. No CUDA kernel,
arithmetic, RNG, state format, generation policy or chunk schedule changed.
The default remains the compatibility bridge; other models are unaffected.

This section reports the selected direct **base** implementation. Experimental
descriptor batching, publication-before-ACK and other development candidates
are excluded. Both arms retain normalization and the shared-eight-core placement:
frontend/preparation on 0–7/24–31 and native worker on 0–3/24–27. The baseline's
bridge shares the worker's mask. The total service allocation is eight physical
cores plus SMT, with the client on separate cores 8–11/32–35.

### Matched B512 in-flight result

The comparison used two fresh boots per arm in baseline/direct/direct/baseline
order, three measured trials per boot, 2,048 requests per trial at concurrency
512, and two excluded warmup rounds. Both arms used the exact retained external
`tts_perf_client` binary, SHA-256
`d7713f0bbd418c95395959ebd7497bc3c1110c5313da41f0b0dacf019730abbf`.
The client stopwatch, arrivals, preparation and admission accounting were
unchanged. Values are equal means of six per-trial statistics, not pooled
request percentiles; latency and ITL are milliseconds.

| B | Transport | RTFx | req/s | TTFA p50 | p95 | p99 | ITL avg | ITL p95 | Peak MiB | Requests | Failures | Deficit responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | Compatibility bridge | 1332.86 | 183.84 | 138.40 | 218.53 | 235.37 | 26.60 | 35.79 | 24,673 | 12,288 | 0 | 0 |
| 512 | Direct worker | 1353.61 | 186.70 | 135.59 | 205.04 | 217.24 | 26.06 | 34.94 | 24,673 | 12,288 | 0 | 0 |

Direct transport reduces the primary statistic by **13.49 ms (6.17%)**. The
95% matched bootstrap interval is **[−20.36, −4.22] ms**, with a direct/baseline
ratio interval of **[0.9068, 0.9801]**. The calculation uses 20,000 draws of
matched boot blocks and paired trial positions. Only two blocks on one GPU
were measured, so this is limited single-device evidence.

Baseline boot means were 211.50 and 225.57 ms; direct boot means were 199.59
and 210.50 ms. The direct mean's within-arm interval is [191.20, 214.58] ms.
Consequently, **TTFA p95 ≤200 ms is not established**. Development screens and
diagnostics do not enter this comparison. The older 212.93-ms legacy Rust
sample was collected separately and is not a paired arm of this experiment.

### Separate supporting checks

The direct path also completed the following single-boot, three-trial samples.
They are regression/smoke measurements, not matched improvement estimates.

| B | Mode | RTFx | req/s | TTFA p50 | p95 | p99 | Requests | Failures | Deficit responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | In-flight | 17.54 | 2.51 | 18.32 | 19.25 | 19.33 | 96 | 0 | 0 |
| 512 | Fixed cohort | 485.26 | 63.87 | 154.16 | 171.09 | 177.34 | 1,536 | 0 | — |

Fixed-cohort lane ordering and batch-seeded output lengths differ across
separate runs; the RTFx above is not evidence of a kernel or transport throughput
gain. Its historical client does not measure playback deficits.

B32 received a separate same-seed HTTP qualification, not a fresh timing claim.
Each arm served 64 requests, including four client cancellations and four slow
readers. All 60 uncancelled requests completed; the four cancellations were
intentional. All 64 corresponding received PCM hashes and sample counts matched
between bridge and direct paths, including the cancelled prefixes. Native
receipts checked physical B32, request seeds, slot generations, cancellations
and completed sample totals. These checks establish this fixture's exact
output/lifecycle agreement, not fresh CER, WER or speaker-similarity scores.

A clean CPU-only CMake build passed all **14 CTests**, including 121 native
transport checks and the 59-test bridge/launcher suite (three optional fixture
tests skipped). It does not rebuild or requalify CUDA kernels. Live execution
reused the same qualified native package; remote CI and additional widths or
GPU architectures remain outside this result.

### Direct-worker evidence and reproduction

The following receipts are relative to the external in-flight-r2 evidence
directory. The tested direct frontend library SHA-256 is
`5577751830e46b0defdcd903eeae13a86207d308d41b59558dcd1f3f5c7dd8de`;
the B512 in-flight serving manifest SHA-256 is
`f7d92e89925ddca8ee8f419700456e2700e6d82b629a67364c6776f76a8caee5`.

| Evidence | Receipt |
|---|---|
| Registration, identities and matched statistics | `FROZEN_CLIENT_COMPARISON.json`, `DIRECT_COMPARISON.json` |
| Matched client observations, telemetry and launch commands | `screens/{baseline_frozen_client,direct_frozen_client}-boot{1,2}/` |
| B1 and fixed-cohort supporting samples | `screens/direct_b1-boot1/`, `screens/direct_fixed512-boot1/` |
| B32 exact PCM and lifecycle comparison | `screens/{qual_baseline,qual_direct}-boot1/pcm-qualification/RESULT.json` |
| Clean CPU build and full CTest results | `ci-build/RECEIPT.json`, `ci-build/ctest.log` |

Follow [direct-worker B512 reproduction](nemotron-megakernel.md#direct-worker-b512-comparison)
with the sealed package, model, normalization assets and frozen client. Remove
only `--direct-worker` for the bridge control, keep all other settings matched,
and use fresh server boots and new receipt directories for each arm.
