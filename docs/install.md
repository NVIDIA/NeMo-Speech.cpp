# Install NeMo-Speech.cpp

The installer selects a backend-matched native release containing the ASR,
diarization, translation, and TTS CLI, HTTP API, realtime WebSocket endpoint,
browser playground, SDK, and notices. It builds from source when a matching
archive is unavailable. Models are distributed separately; inference commands
download missing indexed defaults on first use, while the server downloads a
model only when explicitly enabled with an indexed name. See [models and
cache](cli.md#models-and-cache).

## Linux and macOS

Inspect
[`scripts/install.sh`](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/scripts/install.sh),
then run:

```bash
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
nemo-speech --version
```

With no version argument, the installer reads the current release identifier
from the repository's `VERSION` file, including prerelease identifiers.
Native Linux archives require glibc 2.31 or newer (Ubuntu 20.04 or equivalent).

Prebuilt CPU archives require no GPU toolkit. Linux CUDA archives include the
required user-space CUDA libraries but still need a compatible NVIDIA driver.
Vulkan archives use the host's Vulkan loader and vendor driver. The Linux
x86_64 CUDA archive supports Turing-class GPUs (compute capability 7.5,
including RTX 20-series) and newer. On an older GPU, select `--backend cpu` or
`--backend vulkan`, or build from source with a compatible CUDA toolkit.

The installer selects CUDA when `nvidia-smi` is available, Metal on Apple
Silicon, and CPU otherwise. Override the backend or force a source build:

```bash
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh |
  sh -s -- --backend cpu
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh |
  sh -s -- --source
```

On Linux aarch64, the CUDA release is selected by platform and driver:
`cuda12` for Jetson Orin and `cuda13` for Jetson Thor or DGX Spark. Set
`NEMO_SPEECH_CUDA_SERIES=12` or `13` only when automatic detection is not
available.

It installs without `sudo` and links the CLI into `~/.local/bin`. Run `--help`
to see prefix, backend, PATH, and dry-run options. Downloaded archives are
verified against their published SHA-256 files; a present archive with an
invalid or mismatched checksum always fails rather than falling back to source.

The source fallback requires Git, CMake 3.26 or newer, Ninja, a C++17 compiler,
SentencePiece development files, and any toolkit required by the selected
backend. On Ubuntu/Debian install `libsentencepiece-dev`; on Fedora install
`sentencepiece-devel`; on macOS install `sentencepiece` with Homebrew.

Source installs use `main` by default. To install a fork or the current
checkout, set `NEMO_SPEECH_SOURCE_URL` to its URL or path (`$PWD` on Linux or
macOS, or `(Get-Location).Path` in PowerShell). Set
`NEMO_SPEECH_SOURCE_REF` to select another branch or tag. Local installs clone
the committed branch state and do not include uncommitted changes.

Automatic model pulls require the `curl` executable. The Linux/macOS installer
also uses it for release downloads; the Windows installer uses PowerShell's
HTTPS support. macOS and current Windows releases include `curl`, while Linux
users can install it with their distribution package manager. `nemo-speech
doctor` reports whether model downloads are available. Existing local model
paths and already cached models still work if `curl` later becomes unavailable.

## Windows

Inspect
[`scripts/install.ps1`](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/scripts/install.ps1),
then run from PowerShell:

```powershell
irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 | iex
```

The installer updates the current user's `PATH`. Open a new PowerShell window,
then run `nemo-speech --version`.

Select a backend explicitly when needed:

```powershell
irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 `
  -OutFile .\install-nemo-speech.ps1
powershell -ExecutionPolicy Bypass -File .\install-nemo-speech.ps1 `
  -Source -Backend cuda
```

Select the components to install:

```powershell
# ASR and diarization only
powershell -ExecutionPolicy Bypass -File .\install-nemo-speech.ps1 `
  -Source -Backend cpu -Profile asr

# Full runtime profile (add -HttpTls for TLS)
powershell -ExecutionPolicy Bypass -File .\install-nemo-speech.ps1 `
  -Source -Backend cuda -Profile full
```

| Profile | Components |
|---|---|
| `core` | ASR, diarization, and TTS |
| `asr` | ASR and diarization |
| `server` (default) | `core` plus NMT, the HTTP API, and playground |
| `full` | `server` plus gRPC, Flashlight, and JA/ZH tokenizers |

Use `-Grpc`, `-Nmt`, `-Flashlight`, `-TtsJa`, `-TtsZh`, `-Http`, or `-HttpTls`
to customize a profile. Binary installation is limited to `server`; other
selections build from source. Contributors can run
`.\scripts\windows\build.ps1 -Backend cpu -Profile developer` to build `full`
plus tests, examples, and diagnostic tools.

The default prefix is `%LOCALAPPDATA%\Programs\NeMoSpeech`, and the installer
updates only the current user's PATH. A Windows source build requires Git,
CMake, Ninja, Visual Studio 2022 Build Tools, and the selected backend toolkit.
It includes the same CLI, HTTP API, and playground as the Linux and macOS source
installation.

The source installer downloads required C++ libraries automatically. Use
`-VcpkgRoot` to override its vcpkg location.

CUDA requires the NVIDIA CUDA Toolkit and driver. Vulkan requires the LunarG
Vulkan SDK and a vendor driver. Text normalization is not supported on Windows.

A later installer run checks for the published binary archive first and
replaces an existing source build automatically once the matching archive is
available.

## Manual verification

Download the archive and adjacent `.sha256` file from the release page. On
Linux use `sha256sum --check`; on macOS use `shasum -a 256`; on Windows compare
`Get-FileHash -Algorithm SHA256` with the published value. Extract the archive
anywhere and run `bin/nemo-speech --version`.

Release archives use this naming contract:

```text
nemo-speech-<version>-<linux|macos>-<x86_64|aarch64>-<backend>.tar.gz
nemo-speech-<version>-windows-<x86_64|aarch64>-<backend>.zip
```

Linux aarch64 CUDA archives use `cuda12` or `cuda13` as the backend suffix.
On Apple Silicon, automatic selection installs `macos-aarch64-metal`; pass
`--backend cpu` to install the smaller `macos-aarch64-cpu` archive.

To uninstall on Linux or macOS, remove the prefix printed during installation
and `~/.local/bin/nemo-speech`; remove the two-line NeMo-Speech.cpp PATH
entry from the shell startup file if the installer added it. On Windows,
remove `%LOCALAPPDATA%\Programs\NeMoSpeech` (or the selected prefix) and that
prefix's `bin` directory from the current-user PATH. The model cache is stored
separately and is not removed with the runtime: `~/Library/Caches/NeMoSpeech/models`
on macOS, `${XDG_CACHE_HOME:-$HOME/.cache}/nemo-speech/models` on Linux, and
`%LOCALAPPDATA%\NeMoSpeech\models` on Windows.
