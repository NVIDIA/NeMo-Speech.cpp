# SALM transcripts and translation

## Transcripts

- **Punctuation and casing** come from the model. Punctuation options (`--no-punctuation`,
  `automatic_punctuation=false`, the Riva flag) have no effect; inverse text normalization and the profanity filter
  stay opt-in.
- **Language.** The model does not detect the language. A language code (`-l`, HTTP `language`, Riva
  `language_code`) adds "The speech is in <language>." to the default instruction; a custom instruction is used as
  written, so include the language there if it matters. Responses report the requested language.
- **Length.** One request takes from 20 ms (320 samples at 16 kHz) up to one hour of audio.
- **Silence and noise.** Audio without a voice returns an empty transcript without running the language model:
  near-digital silence (peak below -60 dBFS, RMS below -90 dBFS), and audio such as noise or a tone in which the
  encoder's speaker branch hears no one. Stretches without speech inside a recording, such as noise or music, can
  still be transcribed as invented text.
- **Custom instructions.** `POST /v1/chat/completions` and the CLI's `--instruction` take an instruction with the
  audio ("Transcribe with speaker tags", "Translate the speech into German", ...); see the
  [API reference](../api.md#post-v1chatcompletions) and [Prompts](prompts.md).
- **Streamed output.** `POST /v1/chat/completions` with `stream: true` sends the transcript as it is generated
  (server-sent events); word timestamps and speakers come with the last event.

## Translation

Without a translation (NMT) model, SALM translates speech itself into any language it knows: HTTP
`/v1/audio/translations` (`target_language`), gRPC `StreamingTranslateSpeechToText` (`target_language_code`), the CLI's
`--translate-to` (text or JSON, without word timestamps), or an instruction on chat completions. The target defaults
to English; codes such as `de`, `fr`, `ja` or `hi` select another, and an unknown code is refused. Pass the source
language as well (HTTP `language`, gRPC `source_language_code`, CLI `-l`). gRPC `ListSupportedLanguagePairs` lists the
targets. With an NMT model loaded, translation runs through it instead.
