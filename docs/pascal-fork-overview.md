# Pascal community fork overview

> [!NOTE]
> This is an unofficial community fork of NeMo-Speech.cpp. It is not affiliated with,
> maintained by, or officially supported by NVIDIA.

The original code, authorship, notices, credits, and license remain those of the
[NeMo-Speech.cpp project](https://github.com/NVIDIA/NeMo-Speech.cpp). The attribution below is
limited to the modifications, tests, scripts, and documentation introduced by this fork.

## Fork-specific author and maintenance

The fork-specific work was carried out by:

- **GitHub:** [UNDER192103](https://github.com/UNDER192103)
- **Name/project:** Under Nouzen

## Objective

This fork was started to improve the compatibility, configuration, and user experience of
NeMo-Speech.cpp on NVIDIA Pascal GPUs. It is initially aimed at the GeForce GTX 10 series:

```text
GTX 1050
GTX 1050 Ti
GTX 1060
GTX 1070
GTX 1080
GTX 1080 Ti
```

The first hardware used for development and validation was an NVIDIA GeForce GTX 1060 6 GB,
Compute Capability 6.1. Practical tests have only been performed on that GTX 1060. Community
validation is still required before claiming operation on any other Pascal GPU.

The first stage adds safe execution controls, automatic architecture detection, and better
diagnostics. It does not contain a newly designed Pascal-specific kernel.

## Initial changes

- Added `--skinny-q8 auto|on|off`.
- Added automatic CUDA Compute Capability detection.
- Added a safe fallback for GPUs below SM 8.0.
- Added a controlled error when Skinny Q8 is forced on incompatible hardware.
- Added `--suppress-cuda-graph-log`.
- Added selective suppression of the repeated CUDA Graph architecture message.
- Preserved other logs, warnings, and errors.
- Added Windows build and execution scripts.
- Added GTX 1060/Pascal-specific documentation.

### Runtime flags in detail

`--skinny-q8` is a global CLI option and may appear before or after a subcommand. The value is
validated and converted into an internal process environment setting before the backend is created.

| Mode | Behavior on a GPU below SM 8.0 |
| --- | --- |
| `auto` | Sets the existing `GGML_SKINNY_Q8=0` fallback and continues. |
| `off` | Explicitly sets `GGML_SKINNY_Q8=0` and continues. |
| `on` | Stops before inference with an actionable compatibility error. |

For SM 8.0 or newer, `auto` leaves Skinny Q8 available and `on` explicitly enables it. The code
queries CUDA through `ggml_backend_cuda_get_device_compute_capability()` and compares the GGML
architecture identifier with `800`; a GTX 1060 reports `610` (SM 6.1).

`--skinny-q8 auto` **does not** implement a Pascal Q8 kernel. It selects the existing compatible
CUDA fallback on Pascal. It does not modify the model, precision, or transcription mathematics.

`--suppress-cuda-graph-log` sets a process-local control that hides only the repeated debug
message explaining why CUDA Graphs are disabled on pre-Ampere GPUs. It does not enable CUDA Graphs,
change kernels, affect memory use, or improve inference speed.

## Tested environment

| Item | Value |
| --- | --- |
| Operating system | Windows 11 |
| GPU | NVIDIA GeForce GTX 1060 6 GB |
| Architecture | Pascal |
| Compute Capability | 6.1 |
| CPU | Intel Xeon E5-2660 v2 |
| RAM | 32 GB |
| CUDA Toolkit | 12.6 |
| Model | Nemotron 3.5 ASR Streaming 0.6B Q8 GGUF |
| Mode | Persistent HTTP server |

## Performance observations

The source of truth for the current numerical results is
[Pascal performance observations](pascal-performance-observations.md). It records both the
reproducible 11-second JFK comparison and the preliminary two-to-three-second microphone comparison,
including raw values, methodology, and limitations.

The custom runtime is effectively equivalent to default for the measured 11-second sample. The
manual short-request observations are promising but not yet a controlled same-WAV benchmark. The
main expected low-latency factors are persistent serving and CUDA execution; the fork flags are for
compatibility, safety, diagnostics, and log readability. `--suppress-cuda-graph-log` is not a speed
optimization, and `--skinny-q8 auto` selects the existing Pascal-compatible fallback rather than a
new Pascal kernel.

## Reproduction

Build a CUDA configuration compatible with the target hardware, then run a persistent local
server. This public example deliberately uses generic paths:

```powershell
.\build\bin\nemo-speech.exe serve `
  --asr-model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf" `
  --gpu 0 `
  --host 127.0.0.1 `
  --port 8081 `
  --skinny-q8 auto `
  --suppress-cuda-graph-log
```

With the server running, send audio to the local transcription endpoint:

```text
http://127.0.0.1:8081/v1/audio/transcriptions
```

Use `--skinny-q8 auto` for the portable, safe choice. On the tested GTX 1060, expected startup
diagnostics include a CUDA backend selection and the automatic `skinny-q8=off` fallback due to
Compute Capability 6.1.

## Current status

Validated on the GTX 1060 6 GB:

- CUDA SM 6.1 build;
- file transcription;
- persistent HTTP server;
- `/ready` endpoint;
- `/v1/audio/transcriptions` endpoint;
- CPU execution;
- CUDA execution;
- automatic Skinny Q8 fallback;
- controlled error for `--skinny-q8 on`;
- selective CUDA Graph log suppression.

Not yet implemented:

- Pascal-specific Q8 kernel;
- DP4A-based optimization;
- CUDA Graphs for Pascal;
- tests on other GTX 10 GPUs;
- formal reproducible same-WAV short-request benchmark between upstream and this fork.

## Community contributions

Pascal users are invited to test and report:

- GPU model and Compute Capability;
- operating system and CUDA Toolkit version;
- build command;
- GGUF model;
- audio duration and measured latency;
- relevant logs; and
- transcription result.

Please do not publish models or audio protected by licenses. A report that includes the same test
audio, warm-up conditions, repeat count, median, and P95 is especially useful for the planned
controlled benchmark.

## Next steps

1. Add a reviewed, redistributable short English WAV and run the reproducible upstream-versus-fork
   benchmark protocol.
2. Gather community validation across the GTX 10 family.
3. Investigate Pascal-appropriate optimizations, including DP4A, only after measurement identifies
   a meaningful bottleneck.
4. Keep all compatibility behavior explicit and safe for unsupported GPU architectures.
