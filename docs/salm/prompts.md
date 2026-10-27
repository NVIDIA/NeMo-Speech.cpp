# SALM prompts

## Default prompts

Without a custom instruction, the server sends the model one of these prompts:

| Request | Prompt |
|---|---|
| Transcription | Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate. |
| Transcription with a language code the model knows | the same, then "The speech is in \<language\>." |
| Speakers (`diarization=true`, `--diarize`, gRPC speaker diarization) | Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as \<spk:0\>, \<spk:1\>, \<spk:2\>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged transcript. |
| Translation | Translate the audio into English. (or the target language's name) |
| A transcription (no custom instruction, no translation) that came back empty, once, without word timestamps | Transcribe the following. Write all numbers in words. Ignore any music or background sounds: |

## Custom instructions

A custom instruction (chat completions, CLI `--instruction`) replaces the prompt and is used as written, so name the
language in it.
These example instructions cover common needs; check the output on representative recordings, since a prompt does
not guarantee a format.

| Use | Instruction |
|---|---|
| Verbatim, with fillers | Write an exact transcript of this recording in English. Retain fillers, repeated words, and false starts. Add punctuation and capitalization, but do not rewrite or summarize the speech. Return only the transcript. (Add "Write numbers as spoken words." to keep numbers as said.) |
| Readable, without fillers | Create a readable English transcript of this recording with punctuation and capitalization. Omit verbal fillers and accidental repetitions, but retain the remaining wording and meaning. Return only the transcript. |
| Vocabulary hints | Transcribe this English technical discussion with punctuation and capitalization. Vocabulary hints: CUDA, TensorRT, and NVLink. Use these spellings only when supported by the speech; do not add a term merely because it appears in this list. Return only the transcript. |
| Speaker tags | the speakers prompt above |

Vocabulary hints in the instruction are this model's way to steer spellings; the phrase-boosting fields are not
supported (see [Not supported](README.md#not-supported)). Word timestamps and subtitles are requested with API options,
not in the instruction, and work with any of these instructions
([Word timestamps and speakers](timestamps-and-speakers.md)).
