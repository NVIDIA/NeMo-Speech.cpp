# SALM: speech-LLM transcription

SALM (Speech-Augmented Language Model) pairs a speech encoder with a language model that writes the transcript. The
model is [Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), which you
convert once into the GGUF files NeMo-Speech.cpp runs.

Give it a recording of up to one hour and it returns a transcript with punctuation and capitalization. It can also
[time every word and say who spoke it](#word-timestamps-and-speakers), [follow your own
instruction](#your-own-instructions), such as a readable transcript without fillers or a list of names to spell right,
and [translate the speech](#translate-speech) into another language.

- [Get started](#get-started)
- [Transcribe](#transcribe)
- [Word timestamps and speakers](#word-timestamps-and-speakers)
- [Your own instructions](#your-own-instructions)
- [Translate speech](#translate-speech)
- [Stream the transcript](#stream-the-transcript)
- [What SALM does not do](#what-salm-does-not-do)
- Reference: [Models and platforms](models.md), [Configuration](configuration.md)

## Get started

SALM runs through the same entrypoints as the other ASR models: `nemo-speech transcribe`, the HTTP server
(`nemo-speech serve`) and the Riva gRPC server (`riva_server`). The release archives, the `*-server` presets and the
container include it; other source builds need `NEMO_SPEECH_BUILD_SPEECHLM=ON` ([Build from source](../build.md)).

**1. Get the model.** Download the BF16 checkpoint and convert it to a decoder of the precision for your hardware, as
[Models and platforms](models.md#getting-the-model) describes. The conversion writes the decoder and, next to it, its
encoder file.

**2. Transcribe.** Every command takes the decoder; its encoder file is loaded with it:

```bash
MODEL=models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4.gguf
nemo-speech transcribe meeting.wav --model $MODEL
```

The default settings fit a GPU with 32 GB of memory and take recordings of up to one hour.
[Configuration](configuration.md#gpu-memory) has settings for smaller and larger GPUs. The examples below use
`$MODEL` for the decoder you converted.

## Transcribe

From the command line, or from the HTTP server with the OpenAI transcription request:

```bash
nemo-speech transcribe meeting.wav --model $MODEL
nemo-speech transcribe recordings/ --model $MODEL -c 8 --output-dir transcripts

nemo-speech serve --asr-model $MODEL --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav
```

The Riva gRPC server takes the same path: `riva_server --asr.model.path $MODEL`.

**Language.** Without a language, the model works out the language from the speech. Short recordings and less common
languages can then come out in the wrong language, or translated, so pass the language when you know it: `-l fr` on
the CLI, `language=fr` over HTTP, `language_code` over gRPC. For a language the model knows, the server then adds "The
speech is in French." to its prompt. Responses report the language you sent (`und` when you sent none); the model does
not report the language it heard.

**Punctuation.** The model writes punctuation and capitalization itself, so the punctuation switches of the other ASR
models (`--no-punctuation`, `automatic_punctuation=false`, the Riva flag) have no effect. Inverse text normalization
and the profanity filter stay available, off unless a request turns them on.

**Length, silence and noise.** A request takes from 20 ms (320 samples at 16 kHz) up to one hour of audio. Chat
completions carry the audio base64-encoded in a JSON body of at most 256 MiB (`http.chat.max-request-bytes`), about 35
minutes of 48 kHz mono 16-bit WAV, and refuse a larger body with HTTP 413; send such recordings compressed, at 16 kHz,
or to `/v1/audio/transcriptions` (uploads up to 512 MiB, `--max-upload-mb`). Audio without a voice returns an empty
transcript without running the language model: near-digital silence (peak below -60 dBFS, RMS below -90 dBFS), and
audio such as noise or a tone in which the encoder's speaker branch hears no one. Stretches without speech inside a
recording, such as noise or music, can still be transcribed as invented text, so trim them when you can.

## Word timestamps and speakers

After the model has written the transcript, the encoder file's CTC timestamp head lines each word up with the audio,
and its speaker branch tells the voices apart, up to eight speakers. Together they give every word a start time, an
end time and a speaker label, in every language; Chinese, Japanese and Thai are timed per character.

```bash
# JSON with word times and speakers
nemo-speech transcribe meeting.wav --model $MODEL --format json --diarize

# subtitles
nemo-speech transcribe meeting.wav --model $MODEL --format srt

# HTTP: verbose JSON with word times and speakers
curl http://127.0.0.1:8080/v1/audio/transcriptions \
    -F file=@meeting.wav -F response_format=verbose_json -F diarization=true
```

| Entrypoint | Word timestamps | Speakers |
|---|---|---|
| CLI | `--word-times`, `--format json`, SRT and VTT output | `--diarize` |
| HTTP `/v1/audio/transcriptions` | `response_format=verbose_json` (or `srt`, `vtt`) | `diarization=true` |
| HTTP `/v1/chat/completions` | `timestamp_granularities: ["word"]` | included with the timestamps |
| gRPC `Recognize` | `enable_word_time_offsets` | `diarization_config.enable_speaker_diarization` |
| HTTP `/v1/audio/diarizations` | | speaker turns alone, without a transcript |

- **Times** are in seconds over HTTP and in CLI JSON, and in milliseconds over gRPC.
- **Speakers** are numbered from 1, and from `spk:0` in chat completions. A speaker keeps the same label across a
  long recording.
- **Speaker requests change the prompt.** `--diarize` and `diarization=true` send the server's speaker prompt, which
  asks for lowercase text without punctuation and numbers as spoken. For a punctuated transcript with speakers, send
  the speaker instruction from [Your own instructions](#your-own-instructions) and ask for word timestamps.
- **When the alignment fails**, the transcript still comes back, without word times, rather than with made-up ones;
  chat completions reports why in `word_timestamp_alignment`.
- **Speaker turns alone**: `/v1/audio/diarizations` uses a separately loaded diarization model (`--diar-model`) when
  there is one, and the speech-LLM's speaker branch otherwise.

Every published encoder file includes the timestamp head; with an encoder file converted without it, requests for
word timestamps or speakers are refused. The [API reference](../api.md#post-v1audiotranscriptions) has the response
fields.

## Your own instructions

The model reads a text prompt along with the audio, and the prompt decides what it writes. Without an instruction the
server picks the prompt (below). To ask for something else, send an instruction with the CLI's `--instruction` or as
the `text` part of a chat completions request:

```bash
nemo-speech transcribe call.wav --model $MODEL \
    --instruction "Transcribe with punctuation and capitalization. Context terms: CUDA, TensorRT, NVLink."

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @- <<JSON
{"model": "salm", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "Transcribe with punctuation and capitalization. Context terms: CUDA, TensorRT, NVLink."},
  {"type": "input_audio", "input_audio": {"data": "$(base64 -w0 call.wav)", "format": "wav"}}]}]}
JSON
```

These instructions come from the
[Prompt Guide](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#prompt-guide) of the model card:

| You want | Instruction |
|---|---|
| The words as spoken | Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only. |
| Readable text | Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only. |
| Names and terms spelled right | Transcribe with punctuation and capitalization. Context terms: \<names and terms\>. |
| Who said what, with punctuation | Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags \<spk:0\> \<spk:1\> \<spk:2\> \<spk:3\> on the transcription. |
| Numbers and spelled letters as spoken | Transcribe word for word without punctuation. Write numbers as spoken words and keep individually spoken letters separate. |
| Emails, URLs, paths, commands and IDs | Transcribe as a clean written transcript. Preserve exact values and punctuation/separators in emails, URLs, file paths, commands, and identifiers. Use conventional written forms. |
| Several languages in one recording | Transcribe verbatim in the original languages. Preserve hesitations and false starts. Write \<language\> in \<script\> and English in Latin script. Return only the transcript. |

Replace the parts in angle brackets. For more than four speakers, extend the tag list up to \<spk:7\>, and ask for
word timestamps to get each word's speaker.

- **Keep one task per instruction.** An instruction is followed as written, and asking for something other than a
  transcript, such as a summary, can make the model answer instead of transcribe.
- **Name the language** when you know it: the server's language sentence is not added to your instruction.
- **Leave timing out of the instruction.** Word timestamps and subtitles are request options and work with any of
  these instructions.
- **Check the output on your own recordings.** An instruction steers the model; it does not guarantee a format.
- An instruction can include the speaker, event and style tags the model writes, such as `<spk:0>`, but not the
  tokenizer's other special tokens, such as `<|im_end|>`. Names and terms in an instruction are how this model is
  steered toward spellings; the phrase-boosting fields are not supported.

Without an instruction, the server sends one of these prompts:

| Request | Prompt |
|---|---|
| Transcription | Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate. |
| Transcription with a language the model knows | the same, followed by "The speech is in \<language\>." |
| Speakers (`--diarize`, `diarization=true`, gRPC speaker diarization) | Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as \<spk:0\>, \<spk:1\>, \<spk:2\>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged transcript. |
| Translation | Translate the audio into English. (or the target language's name) |
| Retry of a transcription that came back empty (once, without word timestamps) | Transcribe the following. Write all numbers in words. Ignore any music or background sounds: |

## Translate speech

Without a translation (NMT) model loaded, SALM translates the speech itself, into any language it knows:

```bash
nemo-speech transcribe interview.wav --model $MODEL -l fr --translate-to en
curl http://127.0.0.1:8080/v1/audio/translations -F file=@interview.wav -F language=fr -F target_language=de
```

| Entrypoint | How |
|---|---|
| CLI | `--translate-to de` (text or JSON output, without word timestamps or speakers) |
| HTTP `/v1/audio/translations` | `target_language=de` |
| HTTP `/v1/chat/completions` | an instruction, such as "Please translate the given speech to German." |
| gRPC `StreamingTranslateSpeechToText` | `target_language_code` |

The target defaults to English; codes such as `de`, `fr`, `ja` or `hi` choose another, and an unknown code is
refused. Pass the source language too (CLI `-l`, HTTP `language`, gRPC `source_language_code`): it helps for the same
reason as in transcription. gRPC `ListSupportedLanguagePairs` lists the targets. With an NMT model loaded, translation
runs through it instead.

## Stream the transcript

With `"stream": true`, chat completions sends the transcript as server-sent events while the model writes it, in the
OpenAI chunk format. Word timestamps and speakers, when requested, arrive with the last event, once the transcript is
complete. The audio is still uploaded whole; only the output streams. The
[API reference](../api.md#post-v1chatcompletions) lists every chat completions field.

## What SALM does not do

- **Live audio.** SALM transcribes complete recordings. The CLI's `--stream` and `--live`, the HTTP realtime socket
  and gRPC `StreamingRecognize` refuse SALM with an error. Chat completions can still
  [stream the transcript](#stream-the-transcript) while it is written.
- **Phrase boosting.** `speech_contexts` phrases and the HTTP `prompt` field are not supported. Put the names and
  terms in an [instruction](#your-own-instructions) instead.
- **Text-only translation without an NMT model** (`/v1/translations`, gRPC `TranslateText`). Speech translation needs
  no NMT model: SALM [translates the audio itself](#translate-speech).
