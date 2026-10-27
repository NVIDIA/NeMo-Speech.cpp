# SALM (speech-LLM transcription)

SALM (Speech-Augmented Language Model) is a speech-LLM: a speech encoder feeding a language model that writes the transcript. It
transcribes complete recordings of up to one hour with punctuation and casing, and can add word timestamps and
speaker labels, translate into other languages, or follow a custom instruction. It runs through the usual entrypoints:
`nemo-speech transcribe`, the HTTP server (`nemo-speech serve`) and the Riva gRPC server (`riva_server`).

The release archives, the `*-server` presets and the container build it; other source builds need
`NEMO_SPEECH_BUILD_SPEECHLM=ON` ([Build from source](../build.md)).

## Quick start

The model you pass is the decoder GGUF; its encoder file sits next to it ([Models and platforms](models.md)).

```bash
# the decoder (its encoder file next to it), JSON output
nemo-speech transcribe audio.wav --model salm.gguf --format json

# HTTP server with word timestamps and speakers
nemo-speech serve --asr-model salm.gguf --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/audio/transcriptions \
    -F file=@audio.wav -F response_format=verbose_json -F diarization=true

# Riva gRPC server
riva_server --asr.model.path salm.gguf --bind 127.0.0.1:50051
```

## Guides

- [Models and platforms](models.md) - the model files and which builds run the batched path.
- [Configuration](configuration.md) - `asr.salm.*` keys, settings for smaller GPUs, concurrency and errors.
- [Transcripts and translation](transcripts.md) - language, length, silence, custom instructions, streamed output,
  speech translation.
- [Prompts](prompts.md) - the prompts the server sends and example custom instructions.
- [Word timestamps and speakers](timestamps-and-speakers.md) - how each entrypoint requests them, and speaker turns
  alone.

## Not supported

- Streaming recognition (audio sent while it is recorded): the CLI's `--stream` and `--live`, the HTTP realtime socket
  and gRPC `StreamingRecognize` refuse with an error. Send complete audio instead; chat completions can stream the
  output.
- Phrase boosting (`speech_contexts` with phrases, the HTTP `prompt` field).
- Translating text (text in, text out: `/v1/translations`, gRPC `TranslateText`) without an NMT model. Speech
  translation needs none: the model translates the audio itself ([Translation](transcripts.md#translation)).
