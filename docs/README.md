# Documentation

User and developer documentation for NeMo-Speech.cpp. The shortest path
to a transcript lives in the [root README](../README.md); everything below is
the detailed manual.

Start with:

- [Installation](install.md)
- [Command-line workflows](cli.md)
- [Configuration examples](../config/README.md)
- [Model conversion](model-conversion.md)
- [HTTP/realtime and optional gRPC server](server.md)
- [HTTP API reference](api.md)
- [Client integration](clients.md)
- [Native SDK integration](sdk.md)
- [Troubleshooting](troubleshooting.md)
- [Build from source](build.md)

## ASR

- [Models](asr/models.md) - ready-to-run GGUF downloads, custom checkpoint
  conversion, and quantization options.
- [Configuration](asr/configuration.md) - full `asr.*` key reference plus the
  decoding / word-boosting / VAD / endpointing / diarization / postprocessing knobs.
- [Feature matrix and customization](asr/customization.md) - which models
  support which features, and request-time vs startup customization.

## TTS

- [Models](tts/models.md) - ready-to-run MagpieTTS and NanoCodec GGUF downloads,
  tokenizer setup, and custom conversion.
- [Configuration](tts/configuration.md) - `tts.*` key reference and serving.

## NMT

- [Models and conversion](nmt/models.md) - obtain and convert Riva-Translate
  models for the llama.cpp runtime.
- [Configuration](nmt/configuration.md) - `nmt.*` key reference and serving.

## Developer guide

- [Overview](development/README.md) - implementation and performance internals.
- [Diagnostics](development/diagnostics.md) - `check_backend_coverage`.
- [ASR batching](development/asr-batching.md) - neural microbatching and
  streaming-state arenas.
- [ggml patches](development/ggml-patches.md) - the project-specific ggml changes.
- [cuBLAS shim](development/cublas-shim.md) - the in-tree drop-in cuBLAS
  replacement.
