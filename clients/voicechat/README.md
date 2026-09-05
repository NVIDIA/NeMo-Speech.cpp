# VoiceChat reference client

`nemotron-voicechat-client.py` exercises the realtime VoiceChat WebSocket API
exposed by `nemo-speech serve`. It supports microphone or WAV input, playback,
transcript and audio capture, instructions, tools, and concurrent file-based
sessions.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r clients/voicechat/requirements.txt
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080
```

Microphone capture and playback additionally require PortAudio and PyAudio.
For a run without audio-device dependencies, pass `--input-file INPUT.wav` and
`--no-playback`.

See the [VoiceChat guide](../../docs/s2s/README.md) for the complete conversion,
build, server, and client workflow.
