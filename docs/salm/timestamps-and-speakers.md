# SALM word timestamps and speakers

These need the CTC timestamp head, which is part of the encoder file; with an encoder file without it, requests for
them are refused. They cover every language; Chinese, Japanese and Thai transcripts
are timed per character (see the [API reference](../api.md#post-v1audiotranscriptions)).

| Entrypoint | Word timestamps | Speakers |
|---|---|---|
| CLI | `--word-times`, `--format json`, SRT and VTT output | `--diarize` |
| HTTP `/v1/audio/transcriptions` | `response_format=verbose_json` (or `srt`, `vtt`) | `diarization=true` |
| HTTP `/v1/chat/completions` | `timestamp_granularities: ["word"]` | included with the timestamps |
| gRPC `Recognize` | `enable_word_time_offsets` | `diarization_config.enable_speaker_diarization` |
| HTTP `/v1/audio/diarizations` | | speaker turns alone, without a transcript |

Times are seconds over HTTP and in CLI JSON, milliseconds over gRPC. Speakers are numbered from 1 (`spk:0` onwards
in chat completions). Speaker labels stay consistent across a long recording. If the alignment fails, the transcript
comes back without word times rather than with made-up ones.

`/v1/audio/diarizations` uses a separately loaded diarization model (`--diar-model`) when there is one, and the
speech-LLM's speaker branch otherwise.
