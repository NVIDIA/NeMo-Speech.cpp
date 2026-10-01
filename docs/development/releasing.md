# Releasing

[`.github/workflows/release.yml`](../../.github/workflows/release.yml) builds
the binary archives that `scripts/install.sh` and `scripts/install.ps1`
download, checks them, and publishes them to GitHub Releases.

| Platform | Archives | Built on |
|---|---|---|
| Linux x86_64 | `cpu`, `vulkan`, `cuda` | `docker/Dockerfile.release-linux` |
| Linux aarch64 | `cpu`, `vulkan`, `cuda12` (Orin), `cuda13` (Thor, DGX Spark) | `docker/Dockerfile.release-linux` |
| macOS | `aarch64-cpu`, `aarch64-metal`, `x86_64-cpu` | `macos-15`, `macos-15-intel` |
| Windows x86_64 | `cpu`, `vulkan`, `cuda` | `scripts/windows/build.ps1` |

Archives are named `nemo-speech-<version>-<os>-<arch>-<backend>.tar.gz`
(`.zip` on Windows), contain a single directory of the same name, and ship
with a `.sha256` file.

Linux and macOS archives include ITN and TN: `scripts/build_itn_deps.sh` with
`STATIC=1` builds OpenFST, Sparrowhawk, Protobuf, and RE2 as static archives,
and `libnemo_speech_text_normalization` links them privately. The grammars,
`itn_configs.tar.bz2` and `tn_configs.tar.bz2`, are not built by the workflow;
each run downloads them from the release pinned in the `grammars` job, checks
their SHA-256 digests, and publishes them again. To ship new grammars, attach
them to a release and update the tag and digests there.

## Cutting a release

1. Set `NEMO_SPEECH_VERSION` in `VERSION` and merge it to `main`.
2. Push the matching tag: `git tag v0.2.0 && git push origin v0.2.0`. The
   workflow fails if the tag and `VERSION` differ.
3. Approve the `release` environment when the build and checks finish. The
   workflow creates a draft release with the archives, `SHA256SUMS`, build
   provenance attestations, and the text-normalization grammars.
4. Review the draft and publish it.

A daily scheduled run publishes the `nightly` prerelease, which
`install.sh --channel nightly` installs. It is skipped when `main` has not
moved since the current nightly; running the workflow manually with `nightly`
always rebuilds.

Pull requests that change release packaging (the workflow, the release
Dockerfile and packagers, the dependency build scripts, the CMake files that
link the bundled dependencies, or the model smoke test)
run the workflow as a dry run once copy-pr-bot mirrors them to a
`pull-request/<number>` branch. Run the workflow manually with
`dry-run` to build and check everything without publishing.

## What every run checks

- **CPU baseline:** x86_64 archives are built with `GGML_NATIVE=OFF` and must
  not contain instructions beyond x86-64-v3 (AVX2, FMA, F16C, BMI2);
  `scripts/release/check_release.py` disassembles every x86_64 binary. The
  Linux x86_64 CPU archive also transcribes audio under QEMU's Haswell model,
  which has no AVX-512.
- **Self-contained packages:** Linux archives need glibc 2.31 or newer and find
  their bundled libraries through `DT_RPATH`, which `LD_LIBRARY_PATH` cannot
  override. Vulkan archives use the host's `libstdc++` and `libgcc_s`, which
  the host's Vulkan drivers also need. macOS archives need macOS 13.3 or newer, link only system
  libraries, and are ad-hoc signed. Windows archives bundle every DLL they
  import except Windows and GPU driver libraries.
- **Smoke tests:** CPU archives run `tests/ci/model_smoke.py` on their runner.
  The x86_64 Linux and Windows CUDA archives run it on L4 GPUs; the Windows run
  blocks only tagged releases. Linux Vulkan archives only start (`--version`);
  the aarch64 CUDA and Windows Vulkan archives are not run. On Linux and macOS
  the test also synthesizes digits through TN and transcribes them back through
  ITN.

## Runners

CUDA builds are the slowest jobs. Set these repository variables to run them
on larger runners:

| Variable | Default |
|---|---|
| `RELEASE_LINUX_X64_CUDA_RUNNER` | `ubuntu-24.04` |
| `RELEASE_LINUX_ARM64_CUDA_RUNNER` | `ubuntu-24.04-arm` |
| `RELEASE_WINDOWS_CUDA_RUNNER` | `windows-2022` |

## Building an archive locally

```sh
# Linux, from the repository root (CUDA: --target cuda-artifact)
docker build --platform=linux/amd64 -f docker/Dockerfile.release-linux \
    --build-arg BACKEND=cpu --target artifact \
    --output type=local,dest=release-artifacts .

# macOS, after scripts/build_sentencepiece_static.sh, scripts/build_itn_deps.sh,
# and installing a preset configured with -DNEMO_SPEECH_WITH_NORM=ON
# -DGGML_NATIVE=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3, all with
# MACOSX_DEPLOYMENT_TARGET=13.3 in the environment
scripts/release/package-macos.sh --install-prefix <prefix> --backend metal --arch aarch64
```

```powershell
# Windows, after scripts\windows\build.ps1 -Backend cpu -Profile server -CMakeArgs '-DGGML_NATIVE=OFF'
scripts\windows\package-release.ps1 -BuildDir <build dir> -Backend cpu
```
