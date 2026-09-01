# Configuration examples

`nemo-speech serve` hosts any configured combination of ASR, diarization,
NMT, TTS, and VoiceChat over HTTP, plus realtime WebSocket APIs. The separately
built `riva_server` remains the Riva-compatible gRPC endpoint.
Use `config/asr.example.yaml` for ASR-only, `config/diar.example.yaml` for
standalone diarization, `config/tts.example.yaml` for TTS-only,
`config/nmt.example.yaml` for NMT-only, or
`config/server.example.yaml` for a combined server. Use
`config/voicechat.yaml` for the VoiceChat realtime WebSocket server.
Capabilities are enabled automatically when their required model paths are
present. The `asr.enabled`, `nmt.enabled`, `tts.enabled`, and `s2s.enabled` keys
may be set to `true`, `false`, or `auto`.

See [Server configuration](../docs/server.md#engine-and-listener-configuration)
for YAML, environment-variable, and CLI precedence.
