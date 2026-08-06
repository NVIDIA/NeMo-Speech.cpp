# NeMo-Speech.cpp

> [!NOTE]
> This is an unofficial community fork of NeMo-Speech.cpp. It is not affiliated with,
> maintained by, or officially supported by NVIDIA.

The original code belongs to the [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
project. This fork preserves the original notices, credits, and licenses; its fork-specific
credits apply only to the modifications, tests, scripts, and documentation added here.

## About this fork

This fork was started to improve compatibility, configuration, and day-to-day usability of
NeMo-Speech.cpp on NVIDIA Pascal GPUs, especially the GeForce GTX 10 series. Initial development
and validation used an NVIDIA GeForce GTX 1060 6 GB (Compute Capability 6.1).

Its initial target family includes GTX 1050, GTX 1050 Ti, GTX 1060, GTX 1070, GTX 1080, and
GTX 1080 Ti. So far, practical testing has been performed only on the GTX 1060 6 GB; community
validation is required before claiming support for the other Pascal GPUs.

It adds safe runtime controls and clearer diagnostics. It does **not** yet include a new kernel
optimized specifically for Pascal.

### Fork-specific author and maintenance

The fork-specific changes, tests, and documentation were made by:

- **GitHub:** [UNDER192103](https://github.com/UNDER192103)
- **Name/project:** Under Nouzen

This attribution does not apply to the original NeMo-Speech.cpp codebase.

### Initial fork changes

- `--skinny-q8 auto|on|off` runtime control.
- Automatic CUDA Compute Capability detection.
- Safe fallback for GPUs below SM 8.0, plus a controlled error if an incompatible Skinny Q8
  mode is forced.
- `--suppress-cuda-graph-log` to selectively hide the repeated CUDA Graph architecture message.
- Windows build and execution scripts, plus Pascal/GTX 1060 documentation.

`--skinny-q8 auto` does not add a Pascal kernel: it disables the incompatible Skinny Q8 path and
uses the existing CUDA fallback. `--suppress-cuda-graph-log` does not enable CUDA Graphs and does
not make inference faster. Neither change alters model precision, model contents, or transcription
math.

### Tested environment

- OS: Windows 11
- GPU: NVIDIA GeForce GTX 1060 6 GB (Pascal, Compute Capability 6.1)
- CPU: Intel Xeon E5-2660 v2; RAM: 32 GB; CUDA Toolkit: 12.6
- Model: Nemotron 3.5 ASR Streaming 0.6B Q8 GGUF
- Mode: persistent HTTP server

## Pascal performance observations

On the tested GTX 1060 6 GB, the custom runtime showed performance similar to the default runtime
for the included 11-second JFK sample.

For manually recorded short requests around two to three seconds, four of the custom observations
were around 72–80 ms, while one first custom observation was 620.39 ms. The default observations
were generally above 130 ms and also included a large latency spike.

These short-request results are preliminary. A fully reproducible benchmark using the same short
English WAV is being prepared, pending a redistributable fixture.

| Test | Default median | Custom median | Observation |
| --- | ---: | ---: | --- |
| 11-second JFK WAV | 181.41 ms | 178.26 ms | Similar performance |
| Short local speech | 179.16 ms | 74.29 ms | Large preliminary median difference |

See [Pascal performance observations](docs/pascal-performance-observations.md) for the complete
methodology, raw values, limitations, and reproduction instructions.

### Current status

Validated on the GTX 1060 6 GB: CUDA SM 6.1 build, file transcription, persistent HTTP server,
`/ready`, `/v1/audio/transcriptions`, CPU execution, CUDA execution, automatic Skinny Q8 fallback,
the controlled `--skinny-q8 on` error, and selective CUDA Graph log suppression.

Not yet implemented: a Pascal-specific Q8 kernel, DP4A optimization, CUDA Graphs on Pascal,
testing on other GTX 10 GPUs, or a controlled reproducible upstream-versus-fork benchmark.

Example server command (paths are intentionally generic):

```powershell
.\build\bin\nemo-speech.exe serve `
  --asr-model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf" `
  --gpu 0 `
  --host 127.0.0.1 `
  --port 8081 `
  --skinny-q8 auto `
  --suppress-cuda-graph-log
```

Community testing on Pascal GPUs is welcome. Please report GPU model, Compute Capability, operating
system, CUDA Toolkit, build command, GGUF model, audio duration, latency, logs, and transcription
result. Do not publish licensed models or protected audio.

## Reproducible test audio

The fork test fixture is intended to live at `test_files/fork/asr/teste-en.wav`, with the expected
transcript in [`test_files/fork/asr/teste-en.txt`](test_files/fork/asr/teste-en.txt):

```text
Ask not what your country can do for you. Ask what you can do for your country.
```

The WAV itself is currently **not included** because its redistribution license still needs manual
review. Do not publish it until that review is complete. Once a reviewed copy is present, the three
main commands are:

```powershell
.\scripts\windows\test-pascal-wav.ps1 -Model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
.\scripts\windows\run-pascal-server.ps1 -Model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
.\scripts\windows\test-http-wav.ps1
```

For the microphone client, setup, WAV metadata, license-review checklist, and complete testing
workflow, see [Testing with audio and microphone](docs/testing-with-audio-and-microphone.md).

A lightweight native C++ runtime for NVIDIA Nemotron Speech models built on ggml. Runs speech models in realtime and in batch mode across platforms/backends.

## Contents

- [Installation](#installation)
- [About this fork](#about-this-fork)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Local server and playground](#local-server-and-playground)
- [Native SDK](#native-sdk)
- [Build from source](#build-from-source)
- [Documentation](#documentation)
- [License](#license)
- [Contributing](#contributing)

## Installation

From a source checkout, install the CLI, HTTP API, and browser playground for
the detected platform and backend:

```bash
scripts/install.sh --source
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
```

The source build requires Git, CMake 3.26 or newer, Ninja, a C++17 compiler,
and the toolkit for the selected GPU backend. See
[Installation](docs/install.md) for platform-specific prerequisites and
options. The same installer will support native release archives once their
public URL is configured.

## Quick start

The runtime consumes GGUF models. Until the preconverted GGUF is published,
use the included converter to download the public `.nemo` checkpoint and
produce a portable Q8 model. Complete the one-time
[conversion setup](docs/model-conversion.md) first; it does not install NeMo.

```bash
python3 convert_model.py nvidia/nemotron-speech-streaming-en-0.6b \
  --outfile nemotron-speech-streaming-en-0.6b.q8_0.gguf

nemo-speech transcribe test_files/asr/wav/test/jfk.wav \
  --model nemotron-speech-streaming-en-0.6b.q8_0.gguf
```

The converter downloads only the `.nemo` checkpoint through the standard
Hugging Face cache. The CLI selects an available backend and handles common
mono or stereo PCM WAV sample rates automatically. Substitute your own WAV
file after verifying the bundled sample.

## Command line

The CLI is the primary interface. Run `nemo-speech --help` to see the
capabilities included in your build. The [CLI guide](docs/cli.md) covers model
selection, GPU controls, directory transcription, subtitles, diarization,
translation, synthesis, structured output, and benchmarking when you need them.

## Local server and playground

Start the same runtime as a local HTTP service and open the playground:

```bash
nemo-speech serve \
  --asr-model nemotron-speech-streaming-en-0.6b.q8_0.gguf \
  --open
```

The server binds to <http://127.0.0.1:8080> by default and also provides a
documented OpenAI-compatible audio API subset and realtime WebSocket
transcription. A separately built `riva_server` binary provides the
Riva-compatible gRPC interface. See the [server guide](docs/server.md) when you
are ready to integrate either interface.

## Native SDK

Release archives include stable C headers, shared libraries, and an exported
CMake package. An installed application can link only the capability it uses:

```cmake
find_package(NeMoSpeech REQUIRED COMPONENTS ASR)
target_link_libraries(my_app PRIVATE NeMoSpeech::ASR)
```

See [native SDK integration](docs/sdk.md) for in-process C/C++ usage, or
[client integration](docs/clients.md) for OpenAI SDK, curl, and Riva-compatible
gRPC usage.

## Build from source

For a CUDA ASR and TTS server with the playground from an initialized checkout:

Requires CMake 3.26 or newer, Ninja, C and C++17 compilers, and a supported CUDA
toolkit.

```bash
git submodule update --init ggml third_party/cpp-httplib
scripts/configure.sh cuda-server
cmake --build --preset cuda-server
```

The configuration helper validates required submodules and applies the pinned
ggml patch series for CUDA builds. CPU, Metal, Vulkan, server, component,
Windows, and container instructions are in
[Build from source](docs/build.md).

## Documentation

| Start here | What it covers |
|---|---|
| [Installation](docs/install.md) | Native releases, Windows, upgrades, and manual verification |
| [CLI guide](docs/cli.md) | Transcription, subtitles, directories, diarization, NMT, TTS, and tooling |
| [Model conversion](docs/model-conversion.md) | Convert NeMo and Hugging Face checkpoints to runtime GGUF files |
| [Servers](docs/server.md) | HTTP playground/realtime serving and the separate Riva-compatible gRPC server |
| [Native SDK](docs/sdk.md) | CMake components, C ABI lifetimes, threading, and examples |
| [Client integration](docs/clients.md) | OpenAI SDKs, curl, and Riva gRPC clients |
| [Troubleshooting](docs/troubleshooting.md) | `doctor` output and common runtime failures |
| [Build from source](docs/build.md) | Presets, optional components, dependencies, containers, and artifacts |
| [Pascal performance](docs/pascal-performance-observations.md) | GTX 1060 short/long latency observations and reproducible benchmark instructions |
| [All documentation](docs/README.md) | ASR, TTS, NMT, configuration, and developer references |

## License

NVIDIA-authored code is released under the [Apache License 2.0](LICENSE), with
the project copyright notice in [NOTICE](NOTICE). Third-party components retain
their respective terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Contributing

External contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the contribution terms and Developer Certificate of Origin sign-off process.
