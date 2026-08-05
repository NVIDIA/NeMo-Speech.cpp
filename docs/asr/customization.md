# ASR customization

ASR behavior can be changed per request or when the engine starts. This page
helps choose the right mechanism; the exact keys and defaults are in
[ASR configuration](configuration.md).

## Feature matrix

Support depends mostly on the head type. Word boosting works on CTC (flashlight
LM decoder) and cache-aware RNNT (built-in context-biasing tree);
postprocessing and diarization are head-independent.

| Model | Languages | Word boosting | VAD masking | Endpointing | Profanity | ITN | Auto punctuation | Language ID | Diarization |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [Parakeet CTC 1.1B](models.md#parakeet-ctc-11b-offline--buffered-streaming) | en | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes |
| [Nemotron-Speech 0.6B](models.md#nemotron-speech-streaming-06b-cache-aware-rnnt) | en | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes |
| [Nemotron 3.5 0.6B](models.md#nemotron-35-06b-multilingual-prompt-conditioned-rnnt) | 40+ | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| [Parakeet TDT 0.6B v3](models.md#parakeet-tdt-06b-v3-multilingual-offline-transducer) | 25 | No | No | No | Yes | Yes | Yes | No | Yes |


## Request-time options

These options can vary between requests without restarting the server.

### Word boosting

`RecognitionConfig.speech_contexts` biases recognition toward caller-supplied
phrases (names, jargon); stock Riva clients expose `--boosted_words` +
`--boosted_words_score`. Phrases are tokenized with the GGUF-embedded
tokenizer - re-convert pre-embed GGUFs with `convert_model.py` (CTC also
accepts an `asr.decoder.tokenizer_path` override).

CTC needs the flashlight LM decoder at startup and takes typical scores of
8-10; cache-aware RNNT boosts during greedy decoding with no extra artifacts,
and each point is ~3x more potent (typical 2-3). Mechanism, clamps, and field
syntax per surface: [word boosting](configuration.md#word-boosting).

### Transcript postprocessing

- `enable_automatic_punctuation` preserves punctuation from self-punctuating
  models or runs a loaded PnC model for plain-text models.
- `verbatim_transcripts=true` skips ITN when a grammar is loaded.
- `profanity_filter=true` masks words found in the configured profanity list.

The artifacts, build requirements, and processing order are documented under
[Postprocessing](configuration.md#postprocessing-profanity-itn-pnc).

### Speaker diarization

`RecognitionConfig.diarization_config.enable_speaker_diarization` adds a
1-based speaker tag to each final word. The server must have a Sortformer model
configured through `asr.diar.model_path`; requests are rejected if diarization
is requested without one. Word timestamps are enabled automatically. See
[ASR configuration](configuration.md#key-reference) and
[Sortformer models](models.md#sortformer-speaker-diarization).

For diarization without ASR, use `nemo-speech diarize` or the standalone
`nemo_speech_diar_*` C API.

### Language selection

Nemotron 3.5 accepts a request language such as `en-US` or `es-ES`, or `auto`
for model-based detection. The selected language is returned on the transcript
and words. Other listed ASR models do not perform language identification.

### Force an endpoint

For Riva-compatible streaming clients,
`runtime_config["force_eou"] = "true"` finalizes the current utterance.
`custom_configuration["stop_history_eou"]` overrides the silence threshold for
that stream.

## Startup options

These settings affect the loaded engine and therefore require a restart.

- **CTC decoder:** greedy decoding is always available. A Flashlight-enabled
  build can load KenLM and a lexicon for beam search and word boosting. See
  [CTC decoding](configuration.md#ctc-decoding-greedy-vs-flashlight).
- **VAD masking:** a Silero VAD model can suppress silence features before the
  encoder. Loading the model alone does not enable masking. See
  [VAD feature masking](configuration.md#vad-feature-masking).
- **Endpointing:** silence-based endpointing emits multiple final utterances on
  one stream. It can use the decoder timeline or a loaded VAD model. See
  [Endpointing](configuration.md#endpointing).
- **Streaming context:** `asr.streaming.rnnt_right_context` trades RNNT latency
  for accuracy. Parakeet CTC instead uses its chunk and left/right padding
  settings.

## Compatibility notes

The Riva-compatible gRPC adapter intentionally does not implement every Riva
codec and recognition option. See
[Riva parity: known exclusions](configuration.md#riva-parity-known-exclusions).
