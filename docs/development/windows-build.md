# Building on Windows

Native Windows build with **MSVC + Ninja**, covering the CUDA, Vulkan, and CPU
backends plus the optional Riva-compatible gRPC server. For other platforms,
see [Build from source](../build.md).

## Toolchain

Visual Studio 2022 Build Tools are required in every configuration (`cl.exe` is
`nvcc`'s supported CUDA host compiler on Windows).

| Host arch | C/C++ compiler | CUDA host compiler |
|---|---|---|
| x64 | `cl` (default; `clang-cl` selectable) | `cl` |
| ARM64 (e.g. Tegra) | `clang-cl` (required - ggml's ARM CPU backend rejects MSVC) | `cl` |

`build.ps1` selects the compiler with `-Compiler auto`; override it with
`-Compiler msvc|clang-cl`. `clang-cl` uses the MSVC ABI. MinGW is not supported
and cannot drive `nvcc`.

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
git submodule update --init llama.cpp                       # ASR live capture or NMT
git submodule update --init third_party/flashlight-text third_party/kenlm # only for LM-fused CTC decoding
git submodule update --init third_party/open_jtalk          # optional TTS JA tokenizer (-TtsJa)
git submodule update --init --recursive third_party/cppjieba  # optional TTS ZH tokenizer (-TtsZh)
# or: git submodule update --init --recursive
```

The JA/ZH TTS tokenizers are disabled by default. Enable them with `-TtsJa` or
`-TtsZh`; the Mandarin dependency must be initialized recursively as shown
above.

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

# CPU ASR + NMT + TTS + HTTP API, realtime WebSocket, and playground
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
`-Config Release|RelWithDebInfo|Debug`, `-CudaArch <native|75|86|89|120|…>`,
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

- **The cuBLAS shim is optional.** Pass `-CublasShim` to the build driver for an
  app-local `cublas64_<major>.dll` that avoids shipping cuBLAS and cuBLASLt.
- **ggml patches are CUDA-only.** A Vulkan/CPU build uses stock ggml; pass
  `-DNEMO_SPEECH_GGML_PATCHED=OFF` (the encoder uses the portable op path).
- Dependent DLLs must be next to the executable or on `PATH`. Ninja places them
  together in `build-<backend>\bin`.
- Flashlight builds install the replaceable `kenlm.dll` alongside the runtime
  libraries.

### Reset a partially patched ggml checkout

If `apply-ggml-patches` reports that a patch does not apply cleanly, reset the
submodule and re-apply it:

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

Use the unified `nemo-speech` CLI for local ASR, NMT, and TTS commands; see
the [CLI guide](../cli.md). Stock Riva clients work against `riva_server` when
the build includes `-Grpc`. A CPU-only build selects the CPU automatically, or
you can pass `--device cpu` explicitly.

**Vulkan graph optimization.** The runtime disables ggml-vulkan graph
optimization because it is incompatible with the persistent caches used by
streaming ASR. An explicit user value for `GGML_VK_DISABLE_GRAPH_OPTIMIZE`
takes precedence; enabling the pass can produce incorrect streaming output.

### Optional features (flashlight, ITN)

| Feature (CMake flag) | Windows status |
|---|---|
| **Flashlight** (`-DNEMO_SPEECH_WITH_FLASHLIGHT=ON`) | ✅ Builds replaceable `kenlm.dll`; SentencePiece and compression libraries are provisioned automatically. |
| **ITN/TN** (`-DNEMO_SPEECH_WITH_NORM=ON`) | ❌ Not supported on Windows. Requires the OpenFST 1.8 / Sparrowhawk WFST stack, which `scripts/build_itn_deps.sh` builds via Linux autotools (neither is in vcpkg). |

## Next steps

Model conversion and runtime commands are platform-neutral. Continue with:

- [Model conversion](../model-conversion.md)
- [CLI workflows](../cli.md)
- [Server configuration](../server.md)
- [Client integration](../clients.md)

> **Run-time PATH:** CUDA build-tree binaries need the CUDA Toolkit `bin` on
> `PATH`. Installed packages place the project DLLs and Visual C++ runtime next
> to the executable.
