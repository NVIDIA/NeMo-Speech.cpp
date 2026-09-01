# VoiceChat client integration

`nemo-speech serve` exposes VoiceChat at `/v1/realtime`, with `/realtime` as a
compatibility alias for realtime SDK integrations. The included reference
client connects directly to this WebSocket API and covers microphone or file
audio, playback, transcripts, instructions, tools, output capture, and
concurrent streams. Its `--server` option accepts the base server URL and adds
the VoiceChat endpoint path automatically.

## Install

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r clients/voicechat/requirements.txt
```

Microphone capture and live playback additionally require PyAudio and a system
PortAudio installation. File input with `--no-playback` does not.

## Live conversation

Start a VoiceChat server, then run:

```bash
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080
```

Press Ctrl+C to end the session. The client writes received audio, user and
assistant transcripts, function calls, and a JSONL conversation log.

## File input

Stream a WAV file and save the response without opening an audio device:

```bash
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080 \
  --input-file input.wav \
  --audio-output response.wav \
  --no-playback
```

The client accepts WAV input from 16-48 kHz, downmixes stereo input, and sends
the source rate for server-side resampling. It appends trailing silence so the
model can finish its response; adjust this with `--trailing-silence`.

## Instructions, tools, and concurrency

Pass system instructions inline or as a text file:

```bash
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080 \
  --instructions "Answer in one concise sentence."
```

`--tools` accepts an OpenAI-format JSON array inline or from a file. The client
executes its included demonstration tools and sends their results back to the
server. Run with `--ignore-tool-calls` to log calls without executing them.

For load testing, file input can be repeated across independent sessions:

```bash
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080 \
  --input-file input.wav \
  --num-streams 4 \
  --output-dir outputs
```

The server's `s2s.max_streams` must be at least the requested concurrency.

## Protocol overview

The server sends `session.created` immediately after the WebSocket upgrade.
Configure the session before sending audio:

```json
{
  "type": "session.update",
  "session": {
    "audio": {
      "input": {"format": {"type": "audio/pcm", "rate": 24000}},
      "output": {"format": "pcm16"}
    },
    "instructions": "You are a concise voice assistant.",
    "tools": []
  }
}
```

Input may be sent as base64 PCM16 in `input_audio_buffer.append` events or as
binary little-endian PCM16 frames. Input rates from 16-48 kHz are resampled to
the model rate. Output arrives as base64 PCM16 in
`response.output_audio.delta` events at 24 kHz, with 1,920 samples per packet.

The main server events are:

| Event | Meaning |
|---|---|
| `session.created`, `session.updated` | Session lifecycle and effective configuration |
| `input_audio_buffer.speech_started`, `.speech_stopped` | Detected user turn boundaries |
| `conversation.item.input_audio_transcription.delta`, `.completed` | User transcript |
| `response.created`, `response.done` | Assistant response lifecycle |
| `response.output_audio.delta`, `.done` | Assistant PCM audio |
| `response.output_audio_transcript.delta`, `.done` | Assistant transcript |
| `response.function_call_arguments.done` | Complete tool request |
| `error` | Recoverable protocol or inference error |
| `session.end` | Final stream counters |

Send `session.close` after the final input frame. The server drains a residual
partial processing step, emits remaining response events and `session.end`,
then closes the conversation. Continue sending input or trailing silence while
waiting for a spoken response; inference advances in 160 ms audio steps.

### Tool responses

Tools use the OpenAI Realtime shape, either directly or inside a `function`
object. A tool can optionally define acknowledgement messages:

```json
[
  {
    "name": "lookup_order",
    "description": "Look up an order by identifier.",
    "parameters": {
      "type": "object",
      "properties": {"order_id": {"type": "string"}},
      "required": ["order_id"]
    },
    "ack_messages": ["Let me check that order."]
  }
]
```

When the server emits `response.function_call_arguments.done`, execute the
named tool and send its result on the same socket:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "status": "completed",
    "call_id": "call_123",
    "output": "{\"status\":\"shipped\"}"
  }
}
```

See the [realtime API reference](../api.md#voicechat-websocket-v1realtime-and-realtime)
for the complete server contract.
