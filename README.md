# NeMo-Speech.cpp

A lightweight native C++ runtime for running the NVIDIA Nemotron Speech model family locally, with broad hardware support. It supports multilingual speech recognition, speaker diarization, translation, and speech synthesis in real-time and batch modes.

NeMo-Speech.cpp is NVIDIA's official local speech inference solution, with day-0 support for our latest speech models. It builds on models from [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech), with native inference powered by [ggml](https://github.com/ggml-org/ggml).

## Models and applications

| Application | Supported models |
|---|---|
| Speech recognition | [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), [Nemotron Speech Streaming 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), and [Parakeet CTC 1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b) |
| Speaker diarization | [Streaming Sortformer 4-speaker v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), standalone or combined with ASR |
| Text and speech translation | [Riva Translate 4B Instruct v2](https://huggingface.co/nvidia/Riva-Translate-4B-Instruct-v2), with composed ASR-to-NMT-to-TTS speech translation |
| Speech synthesis | [MagpieTTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) with [NeMo NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps) |
| Speech processing | [Silero VAD](https://github.com/snakers4/silero-vad), punctuation and capitalization, endpointing, text normalization, and subtitles |

## Contents

- [Models and applications](#models-and-applications)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Local server and playground](#local-server-and-playground)
- [Native SDK](#native-sdk)
- [Build from source](#build-from-source)
- [Documentation](#documentation)
- [License](#license)
- [Contributing](#contributing)

## Installation

Install the `nemo-speech` CLI for the detected platform and backend:

On Linux or macOS, run:

```bash
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
```

On Windows, run from PowerShell:

```powershell
irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 | iex
```

Open a new PowerShell window after installation so the updated user `PATH`
takes effect.

The installer prefers a verified native release and falls back to a source
build when an artifact is unavailable. A source build requires Git, CMake 3.26
or newer, Ninja, a C++17 compiler, SentencePiece development files, and the
toolchain required by the selected backend, if any. See
[Installation](docs/install.md) for platform-specific prerequisites and
options.

## Quick start

Transcribe a local WAV file. On first use, the CLI downloads the pinned default
Nemotron 3.5 GGUF from Hugging Face and verifies its size and SHA-256:

```bash
nemo-speech transcribe /path/to/audio.wav
```

Source checkouts can use `test_files/asr/wav/test/jfk.wav` as a smoke-test
input.

The same command can transcribe the default microphone on builds that include
live capture:

```bash
nemo-speech transcribe --live
```

Run `nemo-speech model list` to see defaults, short names, and which command
uses each model. For example, `nemo-speech pull nemotron-en` downloads the
English-only model ahead of time, and `--model nemotron-en` selects it. Local
GGUF paths continue to work without downloading anything. The CLI selects an
available backend and handles common mono or stereo PCM WAV sample rates
automatically. See the [CLI model guide](docs/cli.md#models-and-cache) and
[model conversion](docs/model-conversion.md) for custom checkpoints.

## Command line

The CLI is the primary interface. Run `nemo-speech --help` to see the
capabilities included in your build. The [CLI guide](docs/cli.md) covers model
selection, GPU controls, directory transcription, subtitles, diarization,
translation, synthesis, structured output, and benchmarking when you need them.

## Local server and playground

Start the same runtime as a local HTTP service and open the playground:

```bash
nemo-speech serve \
  --asr-model nemotron-3.5 \
  --open
```

The server binds to <http://127.0.0.1:8080> by default. Its transcription and
speech routes expose documented OpenAI-compatible subsets, alongside realtime
WebSocket transcription. A separately built `riva_server` binary provides the
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

Requires CMake 3.26 or newer, Ninja, C and C++17 compilers, SentencePiece
development files, and the toolchain required by the selected backend, if any.
For a CUDA ASR and TTS server with the playground:

```bash
git submodule update --init ggml llama.cpp third_party/cpp-httplib
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
| [HTTP API reference](docs/api.md) | Every endpoint's request fields, responses, and the realtime protocol |
| [Native SDK](docs/sdk.md) | CMake components, C ABI lifetimes, threading, and examples |
| [Client integration](docs/clients.md) | OpenAI SDKs, curl, and Riva gRPC clients |
| [Troubleshooting](docs/troubleshooting.md) | `doctor` output and common runtime failures |
| [Build from source](docs/build.md) | Presets, optional components, dependencies, containers, and artifacts |
| [All documentation](docs/README.md) | ASR, TTS, NMT, configuration, and developer references |

## License

NVIDIA-authored code is released under the
[Apache License 2.0](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/LICENSE),
with the project copyright notice in
[NOTICE](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/NOTICE). Third-party
components retain their respective terms; see
[Third-Party Notices](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/THIRD_PARTY_NOTICES.md).
Release archives also include these files under `share/licenses/nemo-speech/`.

## Contributing

External contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the contribution terms and Developer Certificate of Origin sign-off process.
