# NeMo-Speech.cpp

A lightweight native C++ runtime for NVIDIA Nemotron Speech models built on ggml. Runs speech models in realtime and in batch mode across platforms/backends.

## Contents

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

From a source checkout, install the CLI, HTTP API, and browser playground for
the detected platform and backend:

```bash
scripts/install.sh --source
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
```

The source build requires Git, CMake 3.26 or newer, Ninja, a C++17 compiler,
and the toolkit for the selected GPU backend. See
[Installation](docs/install.md) for platform-specific prerequisites, options,
and the native release-archive flow.

## Quick start

Download the ready-to-run Q8 GGUF from the model's Hugging Face repository,
then transcribe the bundled sample:

```bash
hf download nvidia/nemotron-speech-streaming-en-0.6b \
  nemotron-speech-streaming-en-0.6b.q8_0.gguf \
  --local-dir models

nemo-speech transcribe test_files/asr/wav/test/jfk.wav \
  --model models/nemotron-speech-streaming-en-0.6b.q8_0.gguf
```

Install the `hf` command with `pip install -U huggingface_hub` if needed. The
CLI selects an available backend and handles common mono or stereo PCM WAV
sample rates automatically. Substitute your own WAV file after verifying the
bundled sample. See [ASR models](docs/asr/models.md) for the other published
GGUFs and [model conversion](docs/model-conversion.md) for custom checkpoints.

## Command line

The CLI is the primary interface. Run `nemo-speech --help` to see the
capabilities included in your build. The [CLI guide](docs/cli.md) covers model
selection, GPU controls, directory transcription, subtitles, diarization,
translation, synthesis, structured output, and benchmarking when you need them.

## Local server and playground

Start the same runtime as a local HTTP service and open the playground:

```bash
nemo-speech serve \
  --asr-model models/nemotron-speech-streaming-en-0.6b.q8_0.gguf \
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

Requires CMake 3.26 or newer, Ninja, C and C++17 compilers, and a supported
CUDA toolkit. For a CUDA ASR and TTS server with the playground:

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
| [HTTP API reference](docs/api.md) | Every endpoint's request fields, responses, and the realtime protocol |
| [Native SDK](docs/sdk.md) | CMake components, C ABI lifetimes, threading, and examples |
| [Client integration](docs/clients.md) | OpenAI SDKs, curl, and Riva gRPC clients |
| [Troubleshooting](docs/troubleshooting.md) | `doctor` output and common runtime failures |
| [Build from source](docs/build.md) | Presets, optional components, dependencies, containers, and artifacts |
| [All documentation](docs/README.md) | ASR, TTS, NMT, configuration, and developer references |

## License

NVIDIA-authored code is released under the [Apache License 2.0](LICENSE), with
the project copyright notice in [NOTICE](NOTICE). Third-party components retain
their respective terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Contributing

External contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the contribution terms and Developer Certificate of Origin sign-off process.
