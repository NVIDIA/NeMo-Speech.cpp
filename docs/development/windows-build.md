# Building on Windows (CUDA/Vulkan)

Native Windows build with **MSVC + Ninja**, covering the CUDA, Vulkan, and CPU
backends plus the optional Riva-compatible gRPC server. For other platforms,
see [Build from source](../build.md).

## Toolchain

One policy across Windows architectures. Visual Studio 2022 Build Tools are
required in every configuration (`cl.exe` is `nvcc`'s only supported CUDA host
compiler on Windows).

| Host arch | C/C++ compiler | CUDA host compiler |
|---|---|---|
| x64 | `cl` (default; `clang-cl` selectable) | `cl` |
| ARM64 (e.g. Tegra) | `clang-cl` (required - ggml's ARM CPU backend rejects MSVC) | `cl` |

`build.ps1` picks this automatically (`-Compiler auto`); override with
`-Compiler msvc|clang-cl`. Notes:

- `clang-cl` targets the same MSVC ABI (same STL/CRT/linker), so the Windows
  handling in the tree (DLL export via `WINDOWS_EXPORT_ALL_SYMBOLS`, no POSIX
  APIs) applies to both compilers. Install it via the VS "C++ Clang tools"
  component or `choco install llvm`. MinGW (gcc/clang in GNU mode) is not
  supported - it cannot drive `nvcc`.
- Warning flags are scoped to C/C++ compilation
  (`$<COMPILE_LANGUAGE:C,CXX>`) so nvcc's `cl` host never receives them - with
  clang-cl as CXX they would otherwise leak into the CUDA compile (`D8021`).
- **ARM64 specifics.** CUDA 13.4+/CCCL 3.4 requires the `cuda/iterator` include
  fix present in the current ggml pin. On integrated GPUs (e.g. Tegra), Vulkan
  devices enumerate as iGPU (`GGML_BACKEND_DEVICE_TYPE_IGPU`), which the ASR
  backend picker accepts.

## Prerequisites

The example commands use [Chocolatey](https://chocolatey.org/) from an
**elevated** shell. You can install the same components manually instead.

| Component | Why | Install |
|---|---|---|
| VS 2022 Build Tools (VC++ workload) | MSVC `cl.exe` + Windows SDK; CUDA host compiler | `choco install -y visualstudio2022-workload-vctools` |
| LLVM (`clang-cl`) - *ARM64 only; optional on x64* | C/C++ compiler on ARM64 (see [Toolchain](#toolchain)) | `choco install -y llvm` (or the VS "C++ Clang tools" component) |
| CMake ≥ 3.26 | build system | `choco install -y cmake` |
| Ninja | generator | `choco install -y ninja` |
| CUDA Toolkit 12.x/13.x | CUDA backend (`nvcc`, cuBLAS) - install **after** VS | `choco install -y cuda` |
| Vulkan SDK | Vulkan backend (`glslc`, loader, headers) | `choco install -y vulkan-sdk` |
| Git | sources + submodules + patch apply | `choco install -y git` |
| A recent NVIDIA driver | runs both CUDA and Vulkan | - |

ARM64 builds also require the Visual Studio
`Microsoft.VisualStudio.Component.VC.Tools.ARM64` and
`Microsoft.VisualStudio.Component.VC.Llvm.Clang` individual components.

The build driver downloads required vcpkg dependencies under
`%LOCALAPPDATA%\NeMoSpeech`. Use `-VcpkgRoot` or `-VcpkgTriplet` to override the
defaults.

> **PATH note:** installers update the *machine* `PATH`, which an already-open
> shell won't see. Open a new terminal afterward (or let `build.ps1` refresh the
> environment from the registry, which it does automatically).

## Get the sources

```powershell
git submodule update --init ggml                            # required (all backends)
git submodule update --init proto/riva-common               # gRPC server
git submodule update --init llama.cpp                       # NMT (-DNEMO_SPEECH_BUILD_NMT=ON)
git submodule update --init third_party/flashlight-text third_party/kenlm # only for LM-fused CTC decoding
git submodule update --init third_party/open_jtalk          # optional TTS JA tokenizer (-TtsJa)
git submodule update --init --recursive third_party/cppjieba  # optional TTS ZH tokenizer (-TtsZh)
# or: git submodule update --init --recursive
```

The JA/ZH TTS tokenizers are gated by `NEMO_SPEECH_TTS_WITH_JA` /
`NEMO_SPEECH_TTS_WITH_ZH` (both default OFF). JA builds Open JTalk/MeCab from
the `third_party/open_jtalk` submodule and compiles its dictionary during the build
(`build-<backend>\open_jtalk_dic`); ZH uses the header-only `third_party/cppjieba` submodule
(recursive - it nests `limonp`). Pass `-TtsJa` / `-TtsZh` to `build.ps1`, or
enable the corresponding CMake options directly.

## Build with `build.ps1` (recommended)

`scripts/windows/build.ps1` imports the MSVC environment, applies the ggml
patches for CUDA, and configures + builds. CUDA and Vulkan are **separate build
trees** (different ggml config), so each gets its own directory.

```powershell
# CUDA + gRPC server (RTX 40xx = Ada; -CudaArch native auto-detects the local GPU)
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cuda -Grpc

# CUDA + gRPC + NMT translation (also checks out the llama.cpp submodule)
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cuda -Grpc -Nmt

# Vulkan (cross-vendor GPU) + gRPC server
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend vulkan -Grpc

# CPU-only, no server
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu

# CPU ASR + TTS + HTTP API, realtime WebSocket, and playground
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Profile server

# Full runtime profile (add -HttpTls for TLS)
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Profile full

# Full profile plus tests, examples, and diagnostic tools
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Profile developer

# CPU + Flashlight decoder + dynamically linked KenLM
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Backend cpu -Flashlight

```

Key parameters: `-Backend cuda|vulkan|cpu`,
`-Profile core|asr|server|full|developer`,
`-Architecture auto|x64|arm64`,
`-Grpc`, `-Nmt`, `-AsrOnly`, `-Http`, `-HttpTls`, `-Flashlight`, `-TtsJa`,
`-TtsZh`,
`-Config Release|RelWithDebInfo|Debug`, `-CudaArch <native|89|86|120|…>`,
`-VcpkgRoot C:\vcpkg`, `-VcpkgTriplet <triplet>`, `-BuildDir <path>`, `-Jobs N`.
Binaries land in `build-<backend>[-<profile>][-<architecture>]\bin`; the default
`core` and `auto` suffixes are omitted.

## Build with raw CMake

If you prefer to drive CMake yourself, run from an **x64 Native Tools** prompt
(or after `vcvars64.bat`), with CMake/Ninja/CUDA/Vulkan on `PATH`:

```powershell
# CUDA: apply the CUDA-only ggml patches first
powershell -ExecutionPolicy Bypass -File scripts\windows\apply-ggml-patches.ps1

cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native `
    -DNEMO_SPEECH_BUILD_GRPC=ON `
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
    -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-cuda --parallel

# Vulkan: stock ggml (the project's ggml patches are CUDA-only). ggml-vulkan requires the
# SPIRV-Headers CMake package; the Vulkan SDK ships it under Lib\cmake.
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=ON -DNEMO_SPEECH_GGML_PATCHED=OFF `
    -DSPIRV-Headers_DIR="$env:VULKAN_SDK\Lib\cmake\SPIRV-Headers" `
    -DNEMO_SPEECH_BUILD_GRPC=ON `
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
    -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-vulkan --parallel
```

### Windows-specific build behavior

- **cuBLAS shim is auto-skipped** on Windows (it's a Linux `.so` size hack using
  a GNU-ld version script). ggml-cuda links the toolkit's real cuBLAS instead;
  `-DNEMO_SPEECH_CUBLAS_SHIM=ON` is a no-op here.
- **ggml patches are CUDA-only.** A Vulkan/CPU build uses stock ggml; pass
  `-DNEMO_SPEECH_GGML_PATCHED=OFF` (the encoder uses the portable op path).
- DLLs export their symbols via `WINDOWS_EXPORT_ALL_SYMBOLS` (the C ABI libs use
  `__declspec(dllexport)`); at runtime, dependent DLLs must be next to the `.exe`
  or on `PATH` (Ninja places them together in `build-<backend>\bin`).
- Flashlight builds the replaceable `kenlm.dll` from a runtime-only source
  allowlist. Flashlight and its static-md vcpkg dependencies remain private in
  the ASR DLL.
- **`NOMINMAX` is defined globally on Windows** so `<windows.h>` (pulled in by the
  CUDA headers when `GGML_CUDA=ON`) doesn't clobber `std::min`/`std::max`.
- **ggml patch files are forced to LF** via `.gitattributes`. On a CRLF checkout,
  `git apply` rejects some hunks as "corrupt patch"; the Windows patch script
  (`scripts/windows/apply-ggml-patches.ps1`) also strips all CR bytes defensively.
- **Open JTalk/MeCab** (TTS tokenizer, vendored via the `third_party/open_jtalk` submodule)
  builds on MSVC with adjusted flags in `src/tts/tokenizer/CMakeLists.txt`: the
  POSIX `HAVE_*` defines are swapped for `HAVE_WINDOWS_H` (mecab's own Win32
  mmap/dirent paths), and `<functional>` is force-included with
  `_HAS_AUTO_PTR_ETC=1` so `std::binary_function` (removed in C++17) resolves.
  No vendored sources are modified.

### After pulling new changes

If `apply-ggml-patches` reports a patch "does NOT apply cleanly" after a pull
(commonly a mixed line-ending or half-applied state in the ggml worktree), reset
the submodule to pristine and re-apply:

```powershell
git -C ggml reset -q
git -C ggml checkout -- .
git -C ggml clean -fd src        # removes patch-created files
powershell -ExecutionPolicy Bypass -File scripts\windows\apply-ggml-patches.ps1
```

### Backend status

| Backend | ASR | TTS | NMT |
|---|---|---|---|
| **CPU** | ✅ Supported | ✅ Supported | ✅ Supported |
| **CUDA** | ✅ Supported | ✅ Supported | ✅ Supported |
| **Vulkan** | ✅ Supported | ✅ Supported | ✅ Supported |

NMT runs through llama.cpp linked against the in-tree patched ggml, so it inherits the build's
backend. TTS uses generic ggml graphs on Vulkan; CUDA builds additionally use
specialized sampling and local-transformer kernels.

Use the unified `nemo-speech` CLI for local ASR, NMT, and TTS commands; see
the [CLI guide](../cli.md). Stock Riva clients work against `riva_server` when
the build includes `-Grpc`. A CPU-only build selects the CPU automatically, or
you can pass `--device cpu` explicitly.

**Vulkan graph-optimization workaround.** ggml-vulkan's graph-optimization pass
reorders graph nodes in a way that breaks this runtime's **in-place persistent
cache tensors** (the streaming FastConformer/RNNT K/V/conv cache): the decode
degenerates into a single repeated token after the first chunk. The runtime
auto-sets `GGML_VK_DISABLE_GRAPH_OPTIMIZE` for Vulkan builds
(`src/runtime/ggml/backend.cpp`), which fixes it at a small Vulkan-perf cost.
This is a platform-independent ggml-vulkan issue, not a Windows one. An explicit
user-set value of the env var takes precedence over the auto-set.

### Optional features (flashlight, ITN)

| Feature (CMake flag) | Windows status |
|---|---|
| **Flashlight** (`-DNEMO_SPEECH_WITH_FLASHLIGHT=ON`) | ✅ Builds replaceable `kenlm.dll`; SentencePiece and compression libraries are provisioned automatically. |
| **ITN/TN** (`-DNEMO_SPEECH_WITH_NORM=ON`) | ❌ Not supported on Windows. Requires the OpenFST 1.8 / Sparrowhawk WFST stack, which `scripts/build_itn_deps.sh` builds via Linux autotools (neither is in vcpkg). |

Automatic dependencies use one architecture-matched `*-windows-static-md`
triplet.

## Next steps

Model conversion and runtime commands are platform-neutral. Continue with:

- [Model conversion](../model-conversion.md)
- [CLI workflows](../cli.md)
- [Server configuration](../server.md)
- [Client integration](../clients.md)

> **Run-time PATH:** CUDA build-tree binaries need the CUDA Toolkit `bin` on
> `PATH`. Installed packages place the project DLLs and Visual C++ runtime next
> to the executable.
