# NMT configuration

Configuration for Riva-Translate NMT. `nemo-speech serve` hosts HTTP; the
separate `riva_server` hosts Riva-compatible gRPC.
Either process can load ASR, TTS, and NMT together. NMT is built only with
`-DNEMO_SPEECH_BUILD_NMT=ON`, which links llama.cpp. For how keys are set
(YAML, environment, and CLI precedence), see
[Server configuration](../server.md#engine-and-listener-configuration). To
obtain and convert the model, see [NMT models](models.md).

## Serving

```bash
nemo-speech serve --config config/nmt.example.yaml
```

or with flags for HTTP:

```bash
nemo-speech serve \
    --nmt.model.path riva-translate-4b-instruct-v2.q8_0.gguf \
    --nmt.backend.gpu 0 --host 127.0.0.1 --port 8080
```

For Riva-compatible gRPC, start the separate binary:

```bash
riva_server \
    --nmt.model.path riva-translate-4b-instruct-v2.q8_0.gguf \
    --nmt.backend.gpu 0 --bind 0.0.0.0:50051
```

NMT auto-enables when `nmt.model.path` is set; force with `nmt.enabled`.

## Config keys

| key | default | meaning |
|---|---|---|
| `nmt.backend.gpu` | `0` (`-1` on CPU-only builds) | GPU device index, `-1` for CPU; alias `--gpu` / `-g` |
| `nmt.model.path` | (none) | path to the Riva-Translate GGUF |
| `nmt.model.n_ctx` | `1024` | decode context length in tokens (model max `8192`) |
| `nmt.generation.max_new_tokens` | `256` | cap per input text |
| `nmt.pool.contexts` | `1` | concurrent decode contexts (one request per context) |
| `nmt.verbose` | `false` | verbose llama.cpp logs for direct/gRPC use; use global `--verbose` with `nemo-speech` |

## Memory and concurrency

Each decode context holds one KV cache sized by `n_ctx` (about `0.13 MiB/token`,
so `~136 MiB` at the default `1024`). The defaults target the common case:
sentence-level translation with one request per context.

- **Longer inputs:** raise `nmt.model.n_ctx` (up to the model's `8192`). The
  model translates a sentence/short paragraph well but degrades on inputs far
  longer than that regardless of `n_ctx`, so there is little gain past a few
  thousand tokens. A prompt that exceeds `n_ctx` is rejected with a clear error.
- **Concurrency:** `translate()` is thread-safe and serves concurrent requests;
  `nmt.pool.contexts` caps how many decode in parallel (extra callers block
  until a context frees). Each added context costs one more `n_ctx`-sized KV
  cache, so raise it only as far as concurrent load needs.

## RPCs

The optional `riva_server` supports `RivaTranslation.TranslateText` and
`ListSupportedLanguagePairs`.

`TranslateText` takes a batch of `texts` plus `source_language` and
`target_language` and returns one `Translation` per input. The pair is resolved
to the model's tag (`en-de`, `en-zh-cn`, ...); pass the two codes
(`source_language: en`, `target_language: de`) or put the full tag in either
field and leave the other empty. Every supported pair has English on one side;
unsupported pairs return `INVALID_ARGUMENT`. Send plain text.

`ListSupportedLanguagePairs` returns the supported `source -> target` pairs keyed
by model name.

When ASR is loaded, `riva_server` also exposes `StreamingTranslateSpeechToText`;
loading TTS enables
`StreamingTranslateSpeechToSpeech`.

See [Client integration](../clients.md) for HTTP and Riva-compatible gRPC
examples.
