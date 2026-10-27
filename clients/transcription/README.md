# Transcription client

`transcription_client.py` exercises the transcription API of `nemo-speech serve`: transcription, translation,
speech-LLM chat completions with an audio part, and JSONL batches. Every request is made by the official `openai`
Python SDK, so it is also a check that the server works with OpenAI clients; server extensions travel in the SDK's
`extra_body`.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r clients/transcription/requirements.txt
python clients/transcription/transcription_client.py --base-url http://localhost:8080/v1 models
```

The base URL defaults to `$NEMO_SPEECH_BASE_URL` or `http://127.0.0.1:8080/v1`, the API key to
`$NEMO_SPEECH_API_KEY` or `$OPENAI_API_KEY` (a server started without `--api-key` needs none), and the model to the
one `GET /v1/models` lists (a transcription model first). Audio files are uploaded unchanged; the server decides
which formats it decodes.

## Commands

| Command | Endpoint | Main options |
| --- | --- | --- |
| `models` | `GET /v1/models` | |
| `transcribe AUDIO` | `POST /v1/audio/transcriptions` | `--language`, `--prompt`, `--format json\|verbose_json\|text\|srt\|vtt`, `--timestamps word` / `segment` (repeatable; implies `verbose_json`), `--diarize`, `--max-speakers`, `--temperature` |
| `translate AUDIO` | `POST /v1/audio/translations` | `--language`, `--target-language`, `--prompt`, `--format json\|verbose_json\|text`, `--temperature` |
| `chat AUDIO` | `POST /v1/chat/completions` | `--instruction` (the text part; omitted, the server's prompt applies), `--language`, `--timestamps word` / `segment`, `--seed`, `--temperature`, `--audio-format` (default: the file extension) |
| `batch MANIFEST` | transcribe or chat | `--output`, `--mode transcribe\|chat`, `--workers`, `--warmup`, `--stats-json`, `--include-response`, `--overwrite`, plus the request options |
| `bench AUDIO...` | transcribe or chat | `--out`, `--conc`, `--min-requests`, `--warmup`, `--mode chat\|transcribe`, plus the request options |

`transcribe`, `translate` and `chat` print the transcript (subtitles for `srt` / `vtt`); `--json` prints the whole
response, `--show-words` adds the timed words (with speakers when diarized), `--output FILE` writes to a file.
`--stats` prints the request's numbers to stderr (see [Latency and throughput](#latency-and-throughput)).
`--extra KEY=VALUE` (repeatable) adds any other request field, for example `--extra verbatim=true` or
`--extra 'speech_contexts=[{"phrases":["NeMo"],"boost":20}]'`; a value that parses as JSON is sent as JSON.

```bash
C="python clients/transcription/transcription_client.py"
$C transcribe meeting.wav --timestamps word --diarize --show-words
$C transcribe talk.flac --format srt --output talk.srt
$C chat call.wav --instruction "Transcribe this audio with speaker tags such as <spk:0>." --timestamps word --json
$C batch manifest.jsonl --output results.jsonl --workers 16 --format verbose_json
$C bench clips/ --out bench/ --conc 1 8 64 256 --min-requests 64
```

## Batches

The manifest has one JSON object per line with an `audio_filepath` (relative paths are relative to the manifest);
a row's `language`, `prompt` (transcribe mode) or `instruction` (chat mode) overrides the command line, and every
other field is kept. The output has one line per row, in manifest order:

```json
{"line": 1, "input": {"audio_filepath": "a.wav", "id": "a"}, "status": "ok", "latency_ms": ..., "text": "...", "words": [], "duration": 3.2}
{"line": 2, "input": {"audio_filepath": "b.wav"}, "status": "error", "error": {"status": 400, "code": "invalid_request", "param": null, "message": "..."}}
```

`duration` is the audio length the server reports (`verbose_json` and chat responses; `null` otherwise);
`audio_seconds` is the length the client uses for RTFx (the WAV header, else `duration`). A JSON summary goes to
stderr at the end (and with `--stats-json FILE` to that file):

```json
{"rows": 64, "ok": 64, "error": 0, "wall_seconds": ..., "latency_p50_ms": ..., "latency_p90_ms": ..., "latency_p99_ms": ..., "latency_max_ms": ..., "audio_seconds": 1920.0, "rtfx": ...}
```

`--warmup N` first sends N untimed requests (the first rows, repeated if the manifest is shorter); they are not
written to the output.

## Latency and throughput

- **latency** (ms): from sending the request to the end of the response.
- **audio seconds**: the WAV header's duration; for other formats the duration the server reports
  (`verbose_json`, chat). Without one, RTFx is left out.
- **RTFx**: audio seconds / seconds. For one request the seconds are its latency; for `batch` and `bench`, the wall
  time from the first send to the last response.

Single requests (`--stats` on `transcribe`, `translate`, `chat`) print one line to stderr; stdout stays the
transcript:

```
stats: latency ... ms | audio 30.0 s | rtfx ... | tokens ...
```

`bench` measures a server under load: for each `--conc N` it keeps exactly N requests in flight (each finished
request starts the next) until `max(2 N, --min-requests)` have been sent, with the audio files used round robin
(continuing from one level to the next), after one untimed warm-up wave of `--warmup` requests. Directories give their audio files in name order; the files are read once before the run so
that no disk read falls in a timed window. Each level prints a row, appends it to `OUT/points.tsv` and writes
`OUT/c<N>.json` with every request:

```
conc  requests  errors  audio_s  wall_s  rtfx   latency_p50_ms  latency_p90_ms  latency_p99_ms  finish_length
1     64        0       1920.0   ...     ...    ...             ...             ...             0
8     64        0       1920.0   ...     ...    ...             ...             ...             0
```

`finish_length` counts responses cut at the token limit. The client runs every request on its own thread; check
that the machine running it is not the bottleneck at high concurrency (it should be mostly idle).

## Errors

The client does not duplicate the server's value rules (ranges, field combinations, formats): it sends what it is
given and prints the server's error, for example
`error: HTTP 400 invalid_request: seed must be a non-negative integer (param: seed)`. Exit status: 0 success,
1 a request or a batch row failed, 2 usage error. SDK retries are off by default (`--max-retries`) so that errors
are seen, not retried.

## Tests

```bash
python -m unittest discover -s clients/transcription -p 'test_*.py' -v
```

The tests run the client against a local fake of the API and check the requests the SDK sends; they need only the
`openai` package.
