# Install NeMo-Speech.cpp

The installer selects a backend-matched native release containing the ASR,
diarization, translation, and TTS CLI, HTTP API, realtime WebSocket endpoint,
browser playground, SDK, and notices. It builds from source when a matching
archive is unavailable. Models are distributed separately and are never
downloaded when the server starts. Ready-to-run GGUFs are available from the
linked Hugging Face repositories in the [ASR](asr/models.md) and
[TTS](tts/models.md) model guides.

## Linux and macOS

Inspect [`scripts/install.sh`](../scripts/install.sh), then run:

```bash
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
nemo-speech --version
```

With no version argument, the installer reads the current release identifier
from the repository's `VERSION` file, including prerelease identifiers.
Native Linux archives require glibc 2.31 or newer (Ubuntu 20.04 or equivalent).

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
and the toolkit for the selected GPU backend. It clones only the submodules
needed by the CLI and playground. When run from a checkout without an
explicit version, it builds that checkout's current branch. Override the source
for a fork or local mirror with `NEMO_SPEECH_SOURCE_URL` and
`NEMO_SPEECH_SOURCE_REF`.

## Windows

Inspect [`scripts/install.ps1`](../scripts/install.ps1), then run from
PowerShell:

```powershell
irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 | iex
nemo-speech --version
```

Select a backend explicitly when needed:

```powershell
.\scripts\install.ps1 -Source -Backend cuda
```

Select the components to install:

```powershell
# ASR and diarization only
.\scripts\install.ps1 -Source -Backend cpu -Profile asr

# Full runtime profile (add -HttpTls for TLS)
.\scripts\install.ps1 -Source -Backend cuda -Profile full
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

To uninstall on Linux or macOS, remove the prefix printed during installation
and `~/.local/bin/nemo-speech`; remove the two-line NeMo-Speech.cpp PATH
entry from the shell startup file if the installer added it. On Windows,
remove `%LOCALAPPDATA%\Programs\NeMoSpeech` (or the selected prefix) and that
prefix's `bin` directory from the current-user PATH. Models downloaded through
Hugging Face or another artifact tool are stored separately and are not removed
by uninstalling the runtime.
