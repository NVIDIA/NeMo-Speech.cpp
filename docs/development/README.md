# Developer guide

Implementation and performance notes for contributors. End users configuring
the server want [ASR configuration](../asr/configuration.md),
[TTS configuration](../tts/configuration.md), or
[Server configuration](../server.md#engine-and-listener-configuration) instead.

## Contents

- [`diagnostics.md`](diagnostics.md) - build switches, runtime knobs, and
  `check_backend_coverage` for catching silent CPU fallbacks on a new backend.
- [`asr-batching.md`](asr-batching.md) - exact-shape neural microbatching and
  indexed streaming-state arenas.
- [`patches/README.md`](../../patches/README.md) - the project's llama.cpp and ggml
  patches, how builds apply them, and how to edit them.
- [`cublas-shim.md`](cublas-shim.md) - the in-tree drop-in cuBLAS replacement
  under `kernels/` and where the custom GPU kernels live.
- [Windows build notes](windows-build.md)
