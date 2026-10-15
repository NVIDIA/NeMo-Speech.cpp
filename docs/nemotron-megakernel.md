# Nemotron megakernel TTS backend

The `nemotron` backend connects the NeMo-Speech.cpp C++ frontend to the retained
Nemotron native controller and CUDA megakernels. It supports the pinned
EasyMagpie checkpoint, `english_sample` voice, English text, and mono 22050 Hz
PCM16 output. It is a different model from the default MagpieTTS/NanoCodec GGUF
backend.

The Python controller is retained (Python 3.11 or newer; the pinned image uses Python 3.12). Python owns native package loading, GPU
state, in-flight scheduling, and the bridge's tokenization and normalization.
The C++ frontend owns public HTTP handling and multiplexes requests over one
Unix socket. This integration is not a pure C++/CUDA rewrite of that controller.
There is no TensorRT-LLM execution dependency in the retained native path.

The vendored controller, kernels, and native normalizer retain their source
notices. Their licensing and redistribution review is pending; a successful
local build does not authorize publishing those sources or model assets. The
imported closure is not relicensed by the repository's Apache-2.0 license; see
the [retained license scope](../THIRD_PARTY_NOTICES.md#imported-nemotron-native-deployment-closure).
The container installs the project and imported component notices under
`/opt/nemo-speech/share/licenses/nemo-speech/`. The retained normalizer README
describes its historical Rust integration and build paths. Use this document
for the current frontend, Docker recipe, and Python bridge setup.

## Build

The container recipe builds the frontend, native normalizer, selected CUDA
kernels, and controller package from source, using pinned builder/runtime
images. Model and grammar assets are mounted separately at runtime:

```bash
docker build -f docker/Dockerfile.nemotron \
  --build-arg NATIVE_WIDTHS=1,32,512 \
  --build-arg NATIVE_BATCHING_MODE=inflight \
  --build-arg NATIVE_INFLIGHT_PROFILE=retained \
  -t nemo-speech:nemotron .
```

The batching mode is fixed when the native package is assembled. To build a
separate fixed-cohort image, use `--build-arg NATIVE_BATCHING_MODE=fixed_cohort`
and a separate tag such as `nemo-speech:nemotron-fixed`. Start that image with
`--mode fixed_cohort`; start the default image with `--mode inflight`.
The launcher rejects a mode that differs from the package manifest. Fixed-cohort
execution needs a full wave of the selected width, so a lone request at B32 or
B512 waits for the rest of its cohort.

Initialize the repository's submodules before configuring CMake. A CPU frontend
build can serve the GPU controller; the controller and native kernels have
their own CUDA runtime requirements.

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DNEMO_SPEECH_BUILD_TTS=ON -DNEMO_SPEECH_BUILD_HTTP=ON \
  -DNEMO_SPEECH_BUILD_ASR=OFF -DNEMO_SPEECH_BUILD_DIAR=OFF \
  -DNEMO_SPEECH_BUILD_NMT=OFF -DNEMO_SPEECH_BUILD_S2S=OFF \
  -DNEMO_SPEECH_BUILD_GRPC=OFF -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
  -DGGML_CUDA=OFF
cmake --build build-native --target nemo_speech_cli -j 4
```

Use the retained CUDA/PyTorch runtime for native compilation and execution.
The kernel recipes target Hopper `sm_90a` and the qualified H100 configuration
has 132 SMs. An H100 80 GB PCIe card can have fewer SMs; its name and memory
capacity alone do not establish compatibility with these launch recipes.
Changing the GPU geometry, architecture, or runtime requires separate
qualification. The bridge additionally uses the `tokenizers`
package and NumPy. Python support dependencies are pinned in
[`scripts/nemotron_native/requirements.txt`](../scripts/nemotron_native/requirements.txt);
PyTorch comes from the qualified CUDA image and is not replaced by that file.
For a local virtual environment inside that CUDA/PyTorch environment, retain
the qualified system PyTorch installation:

```bash
python3 -m venv --system-site-packages .venv-nemotron
. .venv-nemotron/bin/activate
python3 -m pip install -r scripts/nemotron_native/requirements.txt
```

Validate the runtime's actual imports and shared libraries:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 scripts/nemotron_native/runtime_dependencies.py \
  --include-backend --output /tmp/nemotron-runtime-dependencies.json
python3 scripts/nemotron_native/build_kernels.py --validate
python3 scripts/nemotron_native/build_kernels.py \
  --widths 1,32,512 --output /tmp/nemotron-kernels --jobs 2
python3 scripts/nemotron_native/build_package.py \
  --kernels /tmp/nemotron-kernels --output /tmp/nemotron-package \
  --widths 1,32,512 --batching-mode inflight --inflight-profile retained
```

For a local fixed-cohort package, reuse the compiled kernels but choose a fresh
output directory and `--batching-mode fixed_cohort`. Select that directory with
the launcher's `--package` and pass `--mode fixed_cohort`.

Build outputs and package directories must be fresh. Compilation produces an
unqualified package; it does not establish audio correctness or performance.
Keep the model directory and its seal outside the repository. The expected
checkpoint and file hashes are recorded in
[`deploy/nemotron_native/model.lock.json`](../deploy/nemotron_native/model.lock.json).
Do not substitute the default `nemo-speech pull magpie` assets.

## Run

Start the controller and bridge in the qualified Python/CUDA environment:

```bash
python3 scripts/nemotron_native/launch.py \
  --package /tmp/nemotron-package \
  --model /models/nemotron --model-seal /models/model-seal.json \
  --width 32 --mode inflight \
  --listen /tmp/nemotron-b32.sock \
  --receipts /tmp/nemotron-b32-receipts \
  --http-host 127.0.0.1 --http-port 8080 --http-threads 40 \
  --frontend-binary build-native/bin/nemo-speech
```

The launcher verifies the selected package's width and mode, starts its native
worker, waits for readiness, starts the bridge, then starts and supervises the
C++ HTTP frontend when `--http-port` is present. The receipts directory must be
new or empty, and existing socket paths are never replaced. Keep the launcher
running while serving. It sets `TOKENIZERS_PARALLELISM=false`; concurrent requests
use the bounded preparation pool instead of nested tokenizer worker pools.

The Docker image uses this launcher as its entrypoint, with the package at
`/package` and `nemo-speech` on `PATH`. Pass `--http-port`, `--http-host`, and
`--http-threads` to start the complete service in one container. Mount model,
model-seal, normalizer assets, configuration, and receipt paths explicitly.

For text normalization, add `--normalization-config /config/normalization.json`.
The configuration uses the retained normalizer ABI and identifies its native
library and grammar directories. Its `pool_size`, `append_punctuation`, input
limits, and normalization settings are preserved. For comparisons with the
TN-enabled historical runs, use the same grammar hashes and configuration,
including pool size 16 and `NEMOTRON_TN_STRING_MATCHER=1`. Omitting the
configuration disables normalization and preserves trimmed input text.

This complete Docker example starts B32 in-flight serving with normalization.
Set the three asset paths to absolute paths on your machine. Model assets must
match the model seal; the grammar root must contain the matching `en-US`
directory, including `tokenize_and_classify.far` and `verbalize.far`. These
assets are user-supplied and are not downloaded by the image.

```bash
NEMOTRON_MODEL_DIR=/absolute/path/to/converted-model
NEMOTRON_MODEL_SEAL=/absolute/path/to/model-seal.json
NEMOTRON_GRAMMAR_DIR=/absolute/path/to/normalizer-assets
NEMOTRON_RUN_DIR="$(mktemp -d)"
mkdir -p "$NEMOTRON_RUN_DIR/config" "$NEMOTRON_RUN_DIR/receipts"
cat > "$NEMOTRON_RUN_DIR/config/normalization.json" <<'JSON'
{
  "enabled": true,
  "native_library_root": "/opt/nemotron-speech/lib/processors",
  "native_processor": "/opt/nemotron-speech/lib/processors/libnemotron_speech_tts_text_preprocessor.so",
  "normalizer_root": "/normalizer",
  "normalizer_path": "en-US",
  "language_code": "en-US",
  "direct_grammar": true,
  "append_punctuation": true,
  "pool_size": 16,
  "max_input_bytes": 65536,
  "max_normalized_bytes": 262144,
  "max_segments": 256,
  "bounded_preparation": true
}
JSON
docker run --rm --gpus device=0 --shm-size=1g --stop-timeout=120 \
  -p 127.0.0.1:8080:8080 \
  --mount "type=bind,src=$NEMOTRON_MODEL_DIR,dst=/model,readonly" \
  --mount "type=bind,src=$NEMOTRON_MODEL_SEAL,dst=/model-seal.json,readonly" \
  --mount "type=bind,src=$NEMOTRON_GRAMMAR_DIR,dst=/normalizer,readonly" \
  --mount "type=bind,src=$NEMOTRON_RUN_DIR/config,dst=/config,readonly" \
  --mount "type=bind,src=$NEMOTRON_RUN_DIR/receipts,dst=/receipts" \
  nemo-speech:nemotron \
  --package /package --model /model --model-seal /model-seal.json \
  --width 32 --mode inflight --listen /tmp/nemotron-b32.sock \
  --receipts /receipts --normalization-config /config/normalization.json \
  --http-host 0.0.0.0 --http-port 8080 --http-threads 40
```

Run the readiness and synthesis requests below from another terminal. Receipts
remain in `$NEMOTRON_RUN_DIR/receipts` after the container stops. Use a fresh run
directory for another launch. For serving without normalization, omit the
grammar/config mounts and `--normalization-config`. The container's native
worker still requires the qualified model and GPU configuration.

To reproduce the optional float32-forwarding benchmark candidate, add
`-e NEMOTRON_BRIDGE_FORWARD_F32=1` before the image name in the `docker run`
command above. This moves float32-to-PCM16 conversion to the C++ frontend;
public HTTP audio remains PCM16. Leave the variable unset for the default
PCM16 bridge path, and record the selected value with benchmark receipts.
The [results report](nemotron-integration-results.md) distinguishes this
candidate from the default and records its qualification limits.

Alternatively omit `--http-port` from the launcher and start the frontend in
another terminal:

```bash
build-native/bin/nemo-speech serve \
  --tts-backend nemotron --native-socket /tmp/nemotron-b32.sock \
  --host 127.0.0.1 --port 8080 --http-threads 40 --no-warmup
curl --fail http://127.0.0.1:8080/ready
curl --fail http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-tts","input":"Hello from Nemotron.","voice":"english_sample","response_format":"wav"}' \
  --output /tmp/nemotron.wav
```

`response_format: "pcm"` streams S16LE audio with `x-audio-sample-rate: 22050`.
`wav` buffers the completed audio before returning its WAV header. Worker errors
after PCM response headers have been sent abort the HTTP stream; clients must
treat truncated chunked responses as failures. A clean HTTP EOF and nonempty,
sample-aligned audio are required for success. Client disconnects abort the
corresponding backend request.

The HTTP worker pool must accommodate the intended request concurrency plus
control requests. Suggested initial settings for B1/B32/B512 are 8/40/520 HTTP
workers. Changing HTTP concurrency does not change the package's physical
width. Keep TTS preemption disabled for throughput measurements.

For diagnostic runs only, set `NEMOTRON_BRIDGE_DIAGNOSTICS` to a new JSONL
path. The bridge records request receipt, preparation, native submission and
first-audio delivery timestamps from one monotonic clock. This does not replace
client TTFA or GPU event timing. Leave this unset during performance collection.

## Validate and measure

The mock tests require no model or GPU:

```bash
python3 -B tests/nemotron_http_test.py --binary build-native/bin/nemo-speech
python3 -B tests/nemotron_normalization_test.py
python3 -B -m unittest discover -s scripts/nemotron_native/tests -v
```

The HTTP suite covers first audio before completion, overlapping requests, WAV
metadata, worker failure, recovery, disconnect cancellation, and malformed PCM.
The normalizer test uses the real library when
`NEMOTRON_NORMALIZATION_CONFIG=/config/normalization.json` is set; otherwise its
native ABI test is reported as skipped.

For GPU qualification, compare the old and new frontend at the same physical
width, package, text, normalized text/token IDs, context, seed, sampling
parameters, precision, and chunk schedule. Compare exact PCM hashes and sample
counts, EOS/length-cap termination, and replay after cancellation and slot
reuse. Include overlapping short/long requests and refills. Fixed-cohort and
in-flight sampling have different RNG contracts, so exact PCM equality is not
an appropriate gate between those two modes.

The portable benchmark is included in this repository and requires only the
Python standard library. It sends requests directly to the supplied endpoint
and does not start, stop, or reconfigure a server:

```bash
python3 scripts/nemotron_benchmark.py \
  --url http://127.0.0.1:8080/v1/audio/speech \
  --mode closed-loop --width 32 --concurrency 32 --num-requests 128 \
  --warmup-waves 2 --trials 3 --corpus-tsv /data/corpus.tsv \
  --seed 0 --output /results/portable-b32.json
```

`--mode fixed-cohort` gates each wave before connecting and waits for every
request in that wave to terminate before starting the next. `closed-loop`
maintains the configured concurrency through worker-thread refill. `--width`
records the declared physical backend width; it cannot verify that width from
HTTP. The output path must be new. All raw warmup and measured request records,
failures, observed samples, PCM hashes, and read arrivals are retained.

The default request stopwatch includes TCP/TLS connection establishment.
`--exclude-connect` excludes connection time from request latencies while
keeping it in trial wall time. Trial wall time includes final response drain.
TTFA is arrival of the first complete PCM16 sample. ITL is reconstructed from
the configured 2/3/4/5-frame startup and 8-frame steady boundaries, with 1764
samples per frame; incomplete terminal chunks contribute audio duration but
not ITL. Read coalescing can produce zero reconstructed intervals. These are
client observations, not server codec timestamps.

`--timeout` is the per-request budget including connection establishment,
checked at socket-operation and read boundaries. It still includes connection
time when `--exclude-connect` changes the latency stopwatch. Requests already
submitted are drained before the final report is written, including failures
and partial PCM from interrupted streams. An interrupted run retains submitted
requests and records how many planned requests were not submitted.

Reports contain per-trial nearest-rank percentiles and the equal mean of trial
statistics, including p50/p95/p99 TTFA and completion latency. `rtfx` counts all
received sample-derived audio, including partial failed responses;
`completed_rtfx` counts successful responses only. Failed-request timings and
playback deficits remain visible. GPU memory is not measured by this client;
collect telemetry separately. Its timing and scheduling are distinct from the
historical clients even when connection timing is excluded.

For comparison against historical results, retain the original external client
and its configuration. The historical clients are not included in this image;
the command below assumes a separately supplied binary at the displayed path. The following `tts_perf_client` B512 example measures
four refill cohorts after two warmup rounds:

```bash
/opt/nemotron-speech/bin/tts_perf_client \
  --base-url http://127.0.0.1:8080 --transport speech-http \
  --concurrency 512 --num-requests 2048 --warmup-rounds 2 \
  --model easymagpie --voice english_sample --corpus-tsv /data/corpus.tsv \
  --legacy-speech-request --disable-http-pooling \
  --request-timeout-seconds 300 --record-audio-arrivals \
  --codec-frame-samples 1764 --startup-chunk-frames 2,3,4,5 \
  --steady-chunk-frames 8 --json-output /results/inflight-b512.json
```

For a small width sweep, boot separate B1/B32/B512 controllers and use 32/128/2048
measured requests respectively. Freeze client binaries and the corpus; the
historical ten-text corpus hash is
`8bae1d20bfd5d17bee363afdbd6d25e0167d4bcc5c11ea6663d75c3b8d9450a2`.
Retain failures, slow requests, audio arrival traces, playback deficits, VRAM
telemetry, and launch/package/model/normalization receipts. Run compared arms
sequentially on the same idle GPU with matched CPU allocations.

Historical `measure_public_api.py` burst measurements and `tts_perf_client`
closed-loop measurements have different timers and arrival patterns. Report
them separately. The built-in `nemo-speech bench tts` measures synthesis
callbacks without the HTTP transport and must also be labelled separately.
Neither successful streaming nor a performance run establishes acoustic
quality; preserve WAVs and run the same intelligibility and acoustic checks on
both implementations.

The [integration results](nemotron-integration-results.md) record the completed
checks, measured samples, and remaining limitations separately from these
reproduction instructions.
