# Developer guide

Implementation and performance notes for contributors. End users configuring
the server want [ASR configuration](../asr/configuration.md),
[TTS configuration](../tts/configuration.md), or
[Server configuration](../server.md#engine-and-listener-configuration) instead.

## Contents

- [`diagnostics.md`](diagnostics.md) - `check_backend_coverage`, catching silent
  CPU fallbacks on a new backend.
- [`asr-batching.md`](asr-batching.md) - exact-shape neural microbatching and
  indexed streaming-state arenas.
- [`ggml-patches.md`](ggml-patches.md) - the project-specific ggml patches and how
  they are applied at build setup.
- [`cublas-shim.md`](cublas-shim.md) - the in-tree drop-in cuBLAS replacement
  under `kernels/` and where the custom GPU kernels live.
- [Windows build notes](windows-build.md)
