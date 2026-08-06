# Install NeMo-Speech.cpp

The current public installation path builds and installs the backend-matched
ASR, diarization, and TTS CLI, HTTP API, realtime WebSocket endpoint, and browser
playground from a source checkout. Models are distributed separately and are
never downloaded when the server starts. Ready-to-run GGUFs are available from
the linked Hugging Face repositories in the [ASR](asr/models.md) and
[TTS](tts/models.md) model guides.

The installers also contain the native release flow. Once a public release URL
is configured, they prefer release archives containing the CLI, runtime
libraries, headers, CMake package files, and license notices, with source as the
fallback.

## Linux and macOS

Inspect [`scripts/install.sh`](../scripts/install.sh), then run:

```bash
scripts/install.sh --source
export PATH="$HOME/.local/bin:$PATH"  # current shell; future shells are updated
nemo-speech --version
```

The installer selects CUDA when `nvidia-smi` is available, Metal on Apple
Silicon, and CPU otherwise. Override that decision for a source build:

```bash
scripts/install.sh --source --backend cpu
```

It installs without `sudo` and links the CLI into `~/.local/bin`. Run `--help`
to see prefix, backend, PATH, and dry-run options. Downloaded archives are
verified against their published SHA-256 files; an archive with an invalid or
mismatched checksum always fails rather than falling back to source.

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
.\scripts\install.ps1 -Source
nemo-speech --version
```

Select a backend explicitly when needed:

```powershell
.\scripts\install.ps1 -Source -Backend cuda
```

The default prefix is `%LOCALAPPDATA%\Programs\NeMoSpeech`, and the installer
updates only the current user's PATH. A Windows source build requires Git,
CMake, Ninja, Visual Studio 2022 Build Tools, and the selected backend toolkit.
It includes the same CLI, HTTP API, and playground as the Linux and macOS source
installation.

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

To uninstall on Linux or macOS, remove the prefix printed during installation
and `~/.local/bin/nemo-speech`; remove the two-line NeMo-Speech.cpp PATH
entry from the shell startup file if the installer added it. On Windows,
remove `%LOCALAPPDATA%\Programs\NeMoSpeech` (or the selected prefix) and that
prefix's `bin` directory from the current-user PATH. Models downloaded through
Hugging Face or another artifact tool are stored separately and are not removed
by uninstalling the runtime.
