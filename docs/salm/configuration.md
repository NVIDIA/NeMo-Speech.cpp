# SALM configuration

The CLI and both servers also take a YAML file (`--config salm.yaml`); flags such as `--asr.salm.max_sequences 64`
override it. For how keys are set (YAML, environment, and CLI precedence), see
[Server configuration](../server.md#engine-and-listener-configuration).

```yaml
asr:
  model:
    path: salm.gguf
    name: salm                     # the model id reported by /v1/models
  salm:
    max_sequences: 64              # requests decoded at the same time
  batching:
    max_queue_depth: 2048          # requests waiting for a free slot
```

## Keys

| Key (`asr.salm.*`) | Default | Meaning |
|---|---|---|
| `encoder_path` | the decoder's encoder file | encoder GGUF |
| `max_sequences` | 16 | requests decoded at the same time; the main memory setting |
| `n_ctx_per_sequence` | 8192 | KV cache cells per decode slot, pooled for all requests (see below) |
| `max_context` | 73728 | longest request in tokens: one hour of audio needs the default |
| `n_batch` | 2048 | tokens and audio frames per prefill pass; at least `max_sequences x (mtp_draft + 1)` |
| `encoder_max_batch` | 16 | encoder windows per encoder pass |
| `max_new_tokens` | 1024 | generated tokens of a request without an output budget of its own |
| `encoder_ahead_seconds` | 20000 | audio encoded ahead of the decoder, in seconds; it waits on the GPU (about 134 KB per second of audio) until a decode slot frees, so lower it if long requests fail with `pee: out of device memory for the encoder output`; 0 removes the limit |
| `mtp_draft` | 7 | speculative decoding, tokens drafted per step (up to 15), with the model's MTP head; more memory; 0 turns it off |
| `dflash_path` | none | DFlash2 draft model GGUF: drafts the `mtp_draft` tokens instead of the MTP head (batched path) |

CPU threads are `asr.backend.threads`, as for the other models.

## GPU memory

With the NVFP4 decoder the defaults fit a 32 GB GPU and hold one hour of audio per request. For other sizes, for
example:

| GPU memory | `max_sequences` | `n_ctx_per_sequence` | Longest request | Other changes |
|---|---|---|---|---|
| 24 GB | 4 | 8192 | 29 minutes (`max_context: 32768`) | IQ4_NL model; `mtp_draft: 0`, `n_batch: 1024`, `encoder_max_batch: 4` |
| 32 GB | 16 | 8192 | one hour | none (the defaults) |
| 80 GB or more | 240 | 2048 | one hour | `n_batch: 16384`, `encoder_max_batch: 128`: many requests at once |
| 128 GB, unified | 240 | 2048 | one hour | `n_batch: 8192`, `encoder_max_batch: 64`, `mtp_draft: 3` |

The requests decoding at the same time share a pool of `n_ctx_per_sequence x max_sequences` tokens, which must hold
the longest request: 12.5 tokens per second of audio plus up to 6 per second of transcript. The batched path grows
each request's cells as it decodes; CPU, Vulkan, Metal and the other [upstream llama.cpp](models.md#platforms) paths
reserve them all (prompt and whole transcript budget) when it starts, so fewer long requests decode at once there.

On those paths each slot also holds about 48 MiB of recurrent state whatever the request's length, so on a laptop or a
small GPU lower `max_sequences` and raise `n_ctx_per_sequence` to keep the pool: for example `max_sequences: 4` and
`n_ctx_per_sequence: 18432` hold one hour (73,728 tokens, about 0.5 GiB of KV cache).

## Concurrency and errors

One loaded model serves every caller and decodes up to `max_sequences` requests at once; others wait in a queue of
`batching.max_queue_depth`. For throughput, send requests concurrently (CLI `-c N` for directories). The HTTP server
blocks one worker per request, so it starts `max_sequences + 8` workers, at least 256, unless `http.threads` is set.
A connection beyond the workers waits for a free one, and a keep-alive connection keeps its worker for up to 100
requests, so keep `http.threads` above the number of clients.

| Error | HTTP | gRPC | CLI exit |
|---|---|---|---|
| invalid request (bad field, malformed audio header, audio too long) | 400 | `INVALID_ARGUMENT` | 2 |
| audio that cannot be decoded (unsupported format) | 422 | `INVALID_ARGUMENT` | 2 |
| queue full | 429 | `RESOURCE_EXHAUSTED` | 1 |
| server shutting down | 503 | `UNAVAILABLE` | 1 |
| anything else | 500 | `INTERNAL` | 1 |

A directory transcription lists each failed file and exits 2 when every failure was an invalid request, otherwise 1.
