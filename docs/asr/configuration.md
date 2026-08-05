# ASR configuration

Feature-level ASR configuration shared by `nemo-speech serve` and the
separate `riva_server`. For how keys are set (YAML / env / CLI precedence,
`--config`, listeners, service enablement), see
[Server configuration](../server.md#engine-and-listener-configuration).

- [Key reference](#key-reference)
- [CTC decoding: greedy vs flashlight](#ctc-decoding-greedy-vs-flashlight)
- [Word boosting](#word-boosting)
- [VAD feature masking](#vad-feature-masking)
- [Endpointing](#endpointing)
- [Postprocessing: profanity, ITN, PnC](#postprocessing-profanity-itn-pnc)
- [Riva parity: known exclusions](#riva-parity-known-exclusions)

## Key reference

All keys nest under `asr.`. Defaults shown; every key is optional. The CLI alias
column lists the short flag where one exists - the dotted form
(`--asr.<key>`) always works and is what YAML uses.

| YAML / dotted key | CLI alias | default | meaning |
|---|---|---|---|
| `asr.enabled` | - | `auto` | `true` / `false` / `auto` |
| `asr.backend.gpu` | `--gpu`, `-g` | `0` | GPU index; `-1` = CPU |
| `asr.model.path` | - | - | model GGUF path (required for ASR) |
| `asr.model.name` | - | derived | display-name override |
| `asr.streaming.chunk_size` | `--chunk-sec` | `0.16` | CTC buffered window (s) |
| `asr.streaming.ctc_left_padding` | `--left-pad-sec` | `1.92` | CTC left context (s) |
| `asr.streaming.ctc_right_padding` | `--right-pad-sec` | `1.92` | CTC right context (s) |
| `asr.streaming.rnnt_right_context` | - | `1` | cache-aware R; `-1` = model max |
| `asr.batching.enabled` | - | `false` | opt in to batching compatible neural work |
| `asr.batching.max_batch_size` | - | `1024` | maximum items in one neural microbatch |
| `asr.batching.max_queue_delay_us` | - | `5000` | bounded wait for compatible work (µs) |
| `asr.batching.max_queue_depth` | - | `2048` | pending-job backpressure per neural stage |
| `asr.batching.ingress_cohort_delay_us` | - | `20000` | streaming ingress-wave alignment window (µs) |
| `asr.batching.state_arena_slots` | - | `16` | concurrent RNNT/VAD state rows resident on device |
| `asr.decoder.kind` | - | `greedy` | `greedy` / `flashlight` |
| `asr.decoder.lm_path` | `--lm-path` | - | KenLM `.bin`/`.arpa` (implies flashlight) |
| `asr.decoder.lexicon_path` | `--lexicon` | - | flashlight lexicon TSV |
| `asr.decoder.tokenizer_path` | `--tokenizer` | - | SentencePiece, for OOV boosting |
| `asr.decoder.beam_size` | `--beam-size` | `32` | flashlight beam width |
| `asr.decoder.beam_size_token` | - | `16` | flashlight per-token beam |
| `asr.decoder.beam_threshold` | `--beam-threshold` | `20.0` | beam pruning threshold |
| `asr.decoder.lm_weight` | `--lm-weight` | `0.8` | LM rescoring weight |
| `asr.decoder.word_insertion_score` | `--word-score` | `1.0` | word insertion bonus |
| `asr.decoder.max_boost` | `--max-boost` | `10.0` | CTC: max per-word boost magnitude |
| `asr.decoder.boosting_tree_alpha` | `--boosting-tree-alpha` | `1.0` | RNNT word-boosting weight (0 disables) |
| `asr.decoder.boosting_max_boost` | `--boosting-max-boost` | `5.0` | RNNT boost clamp (greedy-safe) |
| `asr.decoder.boosting_depth_scaling` | `--boosting-depth-scaling` | `2.0` | RNNT per-token depth scaling |
| `asr.vad.model_path` | `--vad-model` | - | Silero VAD GGUF (empty = no VAD) |
| `asr.vad.masker.mask_enable` | `--vad-masking` | `false` | mask silence mel frames |
| `asr.vad.masker.onset` | `--vad-onset` | `0.5` | prob > onset → enter speech |
| `asr.vad.masker.offset` | `--vad-offset` | `0.3` | prob < offset → leave speech |
| `asr.vad.masker.pad_ms` | `--vad-pad-ms` | `200` | extend both segment edges (ms) |
| `asr.vad.masker.pad_onset_ms` | - | `200` | extend segment start only (ms) |
| `asr.vad.masker.pad_offset_ms` | - | `200` | extend segment end only (ms) |
| `asr.vad.masker.min_duration_off_ms` | `--vad-min-duration-off-ms` | `500` | min silence to mask (ms) |
| `asr.vad.masker.stddev_floor` | `--vad-stddev-floor` | `1e-5` | normalization denom floor |
| `asr.vad.masker.mask_value` | `--vad-mask-value` | `-16.635` | log-mel fill for masked frames |
| `asr.diar.model_path` | `--diar-model` | - | Sortformer diarizer GGUF (empty = diarization unavailable) |
| `asr.diar.preset` | `--diar-preset` | `streaming` | geometry preset: `streaming` \| `offline` (long-form batch mode: 8 s chunks + bigger AOSC caches); replaces the individual keys below |
| `asr.diar.chunk` | `--diar-chunk` | `20` | chunk length (80 ms frames) |
| `asr.diar.right_context` | `--diar-rc` | `0` | chunk right context (frames) |
| `asr.diar.left_context` | `--diar-lc` | `0` | chunk left context (frames) |
| `asr.diar.fifo` | `--diar-fifo` | `80` | FIFO length (frames) |
| `asr.diar.spkcache` | `--diar-spkcache` | `160` | speaker cache length (frames) |
| `asr.diar.update_period` | `--diar-update-period` | `80` | speaker cache update period (frames) |
| `asr.endpointing.enable` | `--endpointing` | `false` | mid-stream EOU (multiple finals) |
| `asr.endpointing.vad_based` | `--vad-based-eou` | `false` | ride VAD timeline vs token-silence |
| `asr.endpointing.stop_history_eou_ms` | `--stop-history-eou-ms` | `800` | trailing-silence EOU (ms) |
| `asr.postproc.profanity_list_path` | `--profanity-list` | - | profanity word list |
| `asr.postproc.itn_model_dir` | `--itn-model-dir` | - | Sparrowhawk grammar dir, or a root containing per-language dirs |
| `asr.postproc.pnc_model_path` | `--pnc-model` | - | PnC BERT GGUF |
| `asr.postproc.cpu_workers` | - | `2` | bounded final-result worker count |
| `asr.postproc.max_queue_depth` | - | `64` | maximum queued postprocessing results |

The examples below use short aliases for brevity; each maps to the dotted key and
works identically in YAML.

Batching is off by default to preserve local B=1 latency. High-concurrency workloads
can enable it with `--asr.batching.enabled true`; the default 5 ms neural queue
window then coalesces exact-shape CTC, RNNT, TDT, VAD, and PnC work while
preserving bounded backpressure. Concurrent RecognitionStreams use a separate
20 ms ingress cohort window by default so callers reach FE and the acoustic
stage as a small number of useful waves instead of fragmenting the exact-shape
graph cache. This behavior belongs to each `RecognitionStream`, independent of
the interface that created it. Set
`NEMO_SPEECH_INGRESS_COHORT=0` only for diagnostics. Batching also switches
streaming feature extraction to a shared batched graph when CUDA is available.
Without request batching, CUDA streaming still uses the GPU frontend but bypasses
the ingress queue; `NEMO_SPEECH_STREAM_GPU_FE=0` selects the CPU diagnostic
path. Full-utterance CTC, RNNT, and TDT also use GPU feature extraction whenever
CUDA is available.

## CTC decoding: greedy vs flashlight

CTC heads run greedy argmax by default. Set `asr.decoder.lm_path` (KenLM
`.bin`/`.arpa`) + `asr.decoder.lexicon_path` (flashlight lexicon TSV) to enable
the flashlight LexiconDecoder with n-gram rescoring. Requires a
`-DNEMO_SPEECH_WITH_FLASHLIGHT=ON` build; greedy is always available.

That build dynamically links `libkenlm` on Unix or `kenlm.dll` on Windows. The
library must be available on the shared-library search path; the build layouts
stage it beside the other runtime libraries. Flashlight itself remains private
in the ASR library.

```bash
riva_server \
    --asr.model.path parakeet-ctc-1.1b.gguf --bind 0.0.0.0:50051 --gpu 0 \
    --lm-path /path/to/lm.bin --lexicon /path/to/lexicon.txt
```

LM artifacts are model-specific - source your own KenLM binary/ARPA + a matching
lexicon TSV. Beam tunables: `asr.decoder.beam_size` (32),
`asr.decoder.beam_threshold` (20.0), `asr.decoder.lm_weight` (0.8),
`asr.decoder.word_insertion_score` (1.0). Tune these values for the selected
language model, lexicon, and audio domain.

## Word boosting

Per-request word boosting biases the decoder toward caller-supplied phrases
(names, jargon). Sent via `RecognitionConfig.speech_contexts` (`{phrases[],
boost}`); stock clients expose `--boosted_words` + `--boosted_words_score`.
The same name and shape apply on every surface: gRPC, HTTP form field, realtime
`session.update` (`[{"phrases": ["..."], "boost": N}]`), and the CLI
`--speech-context` flags. HTTP `prompt` is an OpenAI-compat shim: one phrase at
boost 10. Requests are portable between CTC and cache-aware RNNT; each clamps
to its safe range. Heads without boosting support (greedy CTC without an LM,
Parakeet TDT) ignore `speech_contexts` with a one-time warning.

**Tokenizer (both heads):** boost phrases are tokenized with the SentencePiece
tokenizer **embedded in the GGUF** (`asr.tokenizer.spm_model`), which is
guaranteed to match the model's vocab. GGUFs converted before the embed:
re-convert with `convert_model.py`, or (CTC only) pass an external
`asr.decoder.tokenizer_path` as an override.

**CTC (flashlight, beam)** - requires the flashlight LM decoder
(`asr.decoder.lm_path` + `asr.decoder.lexicon_path`); greedy CTC with no LM
ignores `speech_contexts` (warns once). The boost is a per-word bump inside the
beam-search LM score: competing hypotheses push back, so scores run high.
Clamp `asr.decoder.max_boost` (default `10`); typical requests 8-10.

```bash
nemo-speech serve \
    --asr.model.path parakeet-ctc-1.1b.gguf --gpu 0 \
    --lm-path lm.bin --lexicon lexicon.txt

riva_streaming_asr_client --riva_uri=localhost:50051 \
    --audio_file=audio.wav --language_code=en-US \
    --boosted_words="nvidia,parakeet,nemotron" --boosted_words_score=8.0
```

**Cache-aware RNNT (greedy)** - built in, no flashlight or LM artifacts:
phrases compile into a context-biasing tree (Aho-Corasick shallow fusion, the
approach NeMo uses). Greedy has no competing hypothesis to push back, so
each point of score is far more potent than on CTC - roughly 3x. Clamp `asr.decoder.boosting_max_boost`
(default `5.0`); typical requests 2-3. Tune deployment-wide strength with
`asr.decoder.boosting_tree_alpha` (default `1.0`, `0` disables);
`asr.decoder.boosting_depth_scaling` (default `2.0`) shapes how the score grows
along a phrase match.

```bash
riva_streaming_asr_client --riva_uri=localhost:50051 \
    --audio_file=audio.wav --language_code=en-US \
    --boosted_words="Kowalczyk,Nemotron" --boosted_words_score=3.0
```

A CTC-scale request against an RNNT head is safe: the score clamps to 5.0 and
stays in the stable region (no repeats or insertions).

## VAD feature masking

Optional Silero VAD floors silence mel frames before the encoder, matching
NVIDIA Riva's VAD feature-masking behavior. **Off by default even with a VAD model loaded** -
the default Riva configuration uses VAD for endpointing without feature masking
(`mask_features=false`). Enable masking with `asr.vad.masker.mask_enable`
(`--vad-masking 1`). Always compiled; composes with greedy or LM decoding on both
CTC and RNNT.

The VAD model is a separate GGUF (`general.architecture="vad"`), not bundled into
the ASR model:

```bash
pip install "silero-vad==6.2.0"
python3 convert_model.py silero --outfile models/silero-v6.2.0.gguf

nemo-speech serve \
    --asr.model.path parakeet-ctc-1.1b.gguf --gpu 0 \
    --lm-path lm.bin --lexicon lexicon.txt \
    --vad-model models/silero-v6.2.0.gguf --vad-masking 1
```

See the `asr.vad.masker.*` keys in the [reference](#key-reference) for onset /
offset / padding / mask-value tunables. The decision follows the same
`vad_postprocessor` sequence used by Riva (binarize → pad → merge short
silences) before feature masking.

## Endpointing

Off by default, the server emits one final when the client closes the stream.
With `asr.endpointing.enable` (`--endpointing`) it detects end-of-utterance
mid-stream and emits a final per utterance (multiple `is_final=true` per stream),
matching Riva. Works on both heads; the per-utterance reset clears
transcript/token buffers. RNNT endpoints flush through the normal EOS path and
then reset encoder and predictor state while preserving the stream timeline, so
delayed tokens cannot leak into the next utterance.

EOU fires when trailing silence reaches `asr.endpointing.stop_history_eou_ms`
(default 800), then re-arms on the next speech. Both silence signals run on the
**decode clock** (audio actually decoded), never the raw-audio frontier - the
buffered CTC runner decodes ~2 s behind arriving audio, and firing on arrival
time would finalize before the tail is decoded.

- **token-silence (default)** - gap since the decoder's last non-blank frame;
  works with greedy and flashlight, no VAD model needed.
- **VAD-driven** (`asr.endpointing.vad_based` + a VAD model) - silence from the
  Silero timeline (Riva's `vad_based_eou=true`). Falls back to token-silence with
  a warning if no VAD model.

```bash
# token-silence EOU (default, no VAD), CTC, 1 s threshold:
nemo-speech serve --asr.model.path parakeet-ctc-1.1b.gguf --gpu 0 \
    --endpointing --stop-history-eou-ms 1000

# VAD-driven EOU, RNNT:
nemo-speech serve --asr.model.path nemotron-speech-streaming-en-0.6b.q8_0.gguf --gpu 0 \
    --endpointing --vad-based-eou 1 --vad-model models/silero-v6.2.0.gguf
```

Masking and endpointing are independent and share the *same* Silero inference, so
you can run masking only, token-silence endpointing only, VAD-driven endpointing
without masking (the default), or both. Per-request Riva knobs:
`custom_configuration["stop_history_eou"]` overrides the threshold for a stream;
a `runtime_config["force_eou"] = "true"` message finalizes the current utterance
immediately. Onset endpointing (`start_history`/`start_threshold`) and
`residue_tokens` / segment-gap variants are not implemented.

## Postprocessing: profanity, ITN, PnC

Postprocessing runs on the final transcript in a fixed order: **profanity → ITN →
PnC**. Each stage is enabled by giving the server its artifact and gated per
request; stages with no artifact are skipped at zero cost.

| stage | config key | build flag | per-request gate |
|---|---|---|---|
| Profanity filter | `asr.postproc.profanity_list_path` | none (always compiled) | request sets `profanity_filter` |
| ITN (Sparrowhawk) | `asr.postproc.itn_model_dir` | `-DNEMO_SPEECH_WITH_NORM=ON` | runs unless `verbatim_transcripts` |
| PnC (BERT punct + caps) | `asr.postproc.pnc_model_path` | none (always compiled) | request sets `enable_automatic_punctuation` |

```bash
nemo-speech serve \
    --asr.model.path parakeet-ctc-1.1b.gguf --gpu 0 \
    --profanity-list profanity.txt \
    --itn-model-dir /path/to/sparrowhawk_grammars \
    --pnc-model pnc-bert-base-en.q8_0.gguf
```

- **Profanity** - one word per line, case-insensitive; a match is masked keeping
  the first letter (`damn` → `d***`), trailing punctuation preserved.
- **ITN** - spoken-form → written-form ("twenty twenty four" → "2024", "three
  dollars" → "$3"). The grammar dir requires `tokenize_and_classify.far` and
  `verbalize.far`. `pre_process.far` and `post_process.far` are detected and
  enabled when present; ASCII proto sidecars are not required.
  A direct grammar directory preserves the single-language behavior. For
  Nemotron 3.5, point the option at a parent whose children are named `en`,
  `es`, `de`, and so on; the explicit request language selects the child, while
  `language_code=auto` uses the model-detected language. Locale codes such as
  `es-ES` fall back to `es`. Grammars load lazily on first use. Requires a
  `-DNEMO_SPEECH_WITH_NORM=ON` build.
- **PnC** - restores casing + `. , ?` with a BERT token classifier, for models
  that emit lowercase unpunctuated text (e.g. Parakeet CTC). Convert with
  `convert_model.py`.

These request gates are independent.

## Riva parity: known exclusions

Intentional differences a side-by-side with riva-speech will surface:

- **LINEAR_PCM only.** Mono 16-bit PCM from 8-96 kHz is accepted and resampled
  to the model rate with a streaming anti-alias filter. FLAC, µ-law, A-law, and
  Opus still require client-side transcoding.
- **N-best output is not implemented.** `max_alternatives` is accepted, but
  current decoders return one alternative per result.
- **No utterance-onset gating** and no `min_duration_on` short-speech deletion:
  pre-speech noise Riva suppresses can surface as partials. Empty-transcript
  finals are swallowed server-side.
- **Confidence:** interim alternatives report 0.0. Final greedy CTC alternatives
  report mean token posterior (with per-word minima when timestamps are
  requested); RNNT and beam-decoded results currently report 1.0.
