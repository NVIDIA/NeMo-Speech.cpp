# Build from source

Published binaries are the shortest path to a working CLI. Build from source
when developing the runtime, enabling optional components, or targeting a
platform without a suitable release artifact.

## Requirements

- Git;
- CMake 3.26 or newer and Ninja;
- a C compiler and a C++17-capable compiler compatible with the selected
  backend and, when applicable, its toolkit;
- SentencePiece development files for ASR and diarization; and
- any toolkit required by the selected backend.

CUDA 12 and 13 are supported. Outside the Windows build driver, gRPC builds
additionally need compatible gRPC, Protobuf compiler/runtime, and Abseil
development packages from a mutually compatible package set.

### Install the basic tools

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
  libsentencepiece-dev
```

Fedora:

```bash
sudo dnf install -y gcc gcc-c++ cmake ninja-build git pkgconf-pkg-config \
  sentencepiece-devel
```

macOS with Homebrew:

```bash
xcode-select --install  # skip if the Command Line Tools are already installed
brew install cmake ninja sentencepiece
```

Windows from an elevated PowerShell with Chocolatey:

```powershell
choco install -y git cmake ninja visualstudio2022-workload-vctools
```

Confirm that `cmake --version` reports 3.26 or newer. Distribution repositories
that ship an older CMake require a newer package before configuration. CUDA,
Vulkan, Metal, gRPC, and optional language frontend dependencies are installed
only when selecting those features; see the platform sections below.

### Backend toolkits

- CPU builds need no accelerator toolkit.
- CUDA builds need a supported CUDA 12 or 13 toolkit (including `nvcc`) at
  build time and a compatible NVIDIA driver at run time. Set
  `CMAKE_CUDA_ARCHITECTURES` when the output must run on GPUs other than the
  build host.
- Vulkan builds need Vulkan headers and loader development files, `glslc`, and
  SPIR-V headers at build time, plus a vendor Vulkan driver at run time. On
  Ubuntu these are available as `libvulkan-dev`, `glslc`, and `spirv-headers`.
- Metal builds require macOS on Apple Silicon and the Xcode Command Line Tools;
  no separate Metal SDK is needed.

## Prepare the checkout

Initialize the submodules needed by the selected components:

```bash
git submodule update --init ggml
git submodule update --init third_party/cpp-httplib  # HTTP server only
git submodule update --init llama.cpp                # ASR live capture or NMT
git submodule update --init proto/riva-common        # gRPC only
git submodule update --init third_party/flashlight-text third_party/kenlm  # Flashlight only
git submodule update --init third_party/open_jtalk   # Japanese TTS only
git submodule update --init --recursive third_party/cppjieba  # Mandarin TTS only
```

`scripts/configure.sh` checks required submodules before CMake runs. CUDA
presets also apply the pinned patches from `ggml-patches/` in order. Mandarin
TTS also requires the Git LFS files under `src/tts/tokenizer/mandarin_data/`;
the helper reports any files that are still LFS pointers. Materialize them with
`git lfs pull --include='src/tts/tokenizer/mandarin_data/*'`.

## Configure and build

Use a supported preset:

```bash
scripts/configure.sh cuda-asr
cmake --build --preset cuda-asr
```

Presets follow `<backend>-<component>` for the `cpu`, `cuda`, `metal`, and
`vulkan` backends and the `asr`, `diar`, `tts`, `nmt`, `speech`, and `server`
component sets. Run `cmake --list-presets` for the exhaustive list. A few useful
landmarks are:

| Preset | Components |
|---|---|
| `cpu-asr` | CPU ASR and diarization |
| `cuda-speech` | CUDA ASR, diarization, NMT, and TTS |
| `metal-nmt` | Metal-enabled NMT build |
| `vulkan-diar` | Vulkan standalone diarization build |
| `<backend>-server` | ASR, diarization, NMT, TTS, HTTP API, realtime WebSocket, and playground |
| `cuda-full` | CUDA server plus normalization, Flashlight, and language frontends (prepare the [optional dependencies](#optional-dependencies) first) |
| `developer` | CPU speech components plus HTTP, gRPC, examples, tests, and tools |

The `<backend>-server` presets include NMT but not protobuf or gRPC. Use
`cuda-full`, `developer`, or explicit CMake options when the Riva-compatible
adapters and separate `riva_server` executable are needed.

The preset selects which components and ggml backend are compiled.

Additional CMake definitions can follow the preset:

```bash
scripts/configure.sh cuda-server -DNEMO_SPEECH_WITH_NORM=ON
cmake --build --preset cuda-server
```

For a manual configuration, explicitly disable the patched CUDA paths when
building against stock ggml:

```bash
cmake -S . -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DNEMO_SPEECH_GGML_PATCHED=OFF
cmake --build build -j"$(nproc)"
```

See [ggml patches](development/ggml-patches.md) for the patched and stock
runtime tradeoffs.

## Components

| CMake option | Default | Purpose |
|---|---:|---|
| `NEMO_SPEECH_BUILD_ASR` | ON | ASR runtime, C ABI, and CLI |
| `NEMO_SPEECH_BUILD_DIAR` | ON | Standalone and ASR-integrated diarization |
| `NEMO_SPEECH_BUILD_TTS` | ON | TTS runtime, C ABI, and CLI |
| `NEMO_SPEECH_BUILD_NMT` | OFF | NMT through llama.cpp |
| `NEMO_SPEECH_BUILD_CLI` | ON | Unified `nemo-speech` executable |
| `NEMO_SPEECH_BUILD_MIC_CAPTURE` | ON | Microphone capture in the CLI and live example |
| `NEMO_SPEECH_BUILD_HTTP` | OFF | HTTP, realtime WebSocket, and playground |
| `NEMO_SPEECH_BUILD_GRPC` | OFF | Riva-compatible gRPC adapters |
| `NEMO_SPEECH_BUILD_EXAMPLES` | OFF | Public in-process C ABI examples |
| `NEMO_SPEECH_BUILD_TESTS` | OFF | First-party correctness tests |
| `NEMO_SPEECH_BUILD_TOOLS` | OFF | Developer diagnostics and microbenchmarks |
| `NEMO_SPEECH_WITH_FLASHLIGHT` | OFF | Flashlight and dynamically linked KenLM decoding |
| `NEMO_SPEECH_WITH_NORM` | OFF | Sparrowhawk/OpenFST text normalization |
| `NEMO_SPEECH_TTS_WITH_JA` | OFF | Japanese TTS text frontend |
| `NEMO_SPEECH_TTS_WITH_ZH` | OFF | Mandarin TTS text frontend |

Component builds initialize and link only their required optional dependencies.
Japanese and Mandarin tokenizer dependencies remain disabled unless their
frontends are selected.

## Optional dependencies

The HTTP server uses the `third_party/cpp-httplib` submodule and OpenSSL
development files for optional TLS (`libssl-dev` on Ubuntu/Debian). CLI-only
builds require neither dependency.

On Ubuntu/Debian, the optional gRPC adapters can use one compatible system
package set:

```bash
sudo apt-get install -y libgrpc++-dev libprotobuf-dev protobuf-compiler \
  protobuf-compiler-grpc libabsl-dev
```

Flashlight decoding requires the `third_party/flashlight-text` and
`third_party/kenlm` submodules. KenLM is built as a replaceable dynamic library
(`libkenlm` or `kenlm.dll`).

Linux ITN/TN builds additionally require Autotools, Protobuf headers and
`protoc`, RE2 development files, and GCC 12. For example, on Ubuntu:

```bash
sudo apt-get install -y autoconf automake bison libtool gcc-12 g++-12 \
  libprotobuf-dev protobuf-compiler libre2-dev
```

Build the pinned OpenFST and Sparrowhawk dependencies into the project-local
prefix without `sudo`:

```bash
CC=gcc-12 CXX=g++-12 scripts/build_itn_deps.sh
```

On Linux, normalization builds also require the static SentencePiece
dependency:

```bash
scripts/build_sentencepiece_static.sh
```

See the component model and configuration guides for runtime companion assets.

## Install the SDK

For native C/C++ integration, install the SDK into a staging prefix. Another
CMake project can then consume its public headers, runtime libraries, and
exported targets without referencing the NeMo-Speech.cpp source tree:

```bash
cmake --install build/cuda-asr --prefix "$PWD/install"
cmake -S my-app -B my-app/build -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build my-app/build
```

The prefix includes the CLI, selected runtime libraries, stable C headers,
CMake targets, documentation, configuration examples, and applicable license
notices. See [Native SDK integration](sdk.md) for consuming the package.

## Windows

Native Windows builds use Visual Studio 2022 Build Tools. `cl.exe` is the CUDA
host compiler on x64; ARM64 uses `clang-cl`. The build driver handles backend,
vcpkg, and patch setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cuda -Http
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend vulkan -Http
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Flashlight
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -AsrOnly
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Grpc
```

See [Windows builds](development/windows-build.md) for prerequisites and backend
notes.

## Containers

`docker/Dockerfile` provides a minimal `runtime` target and a `builder`
target containing the toolchain and sources. Put an ASR GGUF in
`$PWD/models` before starting the model-free runtime image:

```bash
docker build --platform=linux/amd64 -f docker/Dockerfile --target runtime \
  -t nemo-speech-runtime:x86_64 .

docker run --gpus all -p 8080:8080 \
  -v "$PWD/models:/models:ro" nemo-speech-runtime:x86_64 \
  serve --host 0.0.0.0 \
  --asr-model /models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
```

The runtime image entry point is `nemo-speech`; models remain outside the
image. HTTP, gRPC, Flashlight, normalization, NMT, and TTS language frontends
are controlled by Docker build arguments documented in the Dockerfile.

## Build outputs

Selected artifacts are written under the configured build directory's `bin/`:

| Artifact | Purpose |
|---|---|
| `nemo-speech` | Unified CLI and optional HTTP/WebSocket host |
| `riva_server` | Optional Riva-compatible gRPC server |
| `libnemo_speech_asr`, `libnemo_speech_asr_c` | ASR/diarization runtime and public C ABI |
| `libnemo_speech_tts` | TTS runtime and public C ABI |
| `libnemo_speech_nmt`, `libnemo_speech_nmt_c` | Optional NMT runtime and public C ABI |
| `libkenlm` / `kenlm.dll` | Optional dynamically linked language-model runtime |
| `transcribe_file`, `diarize_file` | Native ASR and diarization examples |
| `translate_text`, `synthesize_text` | Native NMT and TTS examples |

Examples, tests, and developer tools are emitted only when their corresponding
build options are enabled.
