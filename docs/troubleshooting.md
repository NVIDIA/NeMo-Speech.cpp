# Troubleshooting

## CLI exit codes

All commands use the same stable process status contract. With `--json`, an
error is written to stderr as one JSON object containing the same `exit_code`
and a machine-readable `type`; stdout remains reserved for successful results.

| Code | Meaning | JSON error type |
|---:|---|---|
| 0 | success | - |
| 1 | runtime, inference, network, or partial batch failure | `runtime_error` |
| 2 | invalid command, option, input, or configuration | `invalid_argument` |
| 3 | required model or companion artifact is missing | `missing_model` |
| 4 | capability is not compiled or supported by the selected model | `unsupported_feature` |

Start with:

```bash
nemo-speech doctor
nemo-speech model info ./models/model.gguf
```

`doctor --json` is suitable for support bundles and automation. It reports the
runtime version, compiled capabilities, backend devices, and driver status.

## Common failures

| Symptom | Check | Resolution |
|---|---|---|
| Command is absent | `nemo-speech --help`; `doctor` features | Install/build an archive containing that component. |
| No suitable model | `model list`; `model info FILE` | Use an indexed model name, run `model pull NAME`, or pass a local GGUF path. |
| Automatic model download is unavailable | `doctor --json` `model_download` field; `curl --version` | Install `curl` and ensure it is on `PATH`, or use an existing local model. |
| Missing companion model | command error | Download the reported component and pass it explicitly or set it in YAML. |
| GPU requested but unavailable | `doctor --json` devices | Install the matching backend build/driver or use `--device cpu`. |
| Server exits before listening | server stderr | Correct the reported model/component path or compatibility error. |
| HTTP 401 | request `Authorization` header | Send `Authorization: Bearer $NEMO_SPEECH_HTTP_API_KEY`. |
| HTTP 413 | upload size | Raise `--max-upload-mb` intentionally or split the input. |
| gRPC UNAVAILABLE | `riva_server` listener and port | Start `riva_server --bind 0.0.0.0:50051`, verify binding/firewall, and use plaintext unless TLS is terminated externally. |
| Unsupported WAV | file format | Use PCM16 or float32 WAV; the CLI handles mono/stereo and 8-96 kHz. |
| Out of memory | device inventory and composition | Use a smaller quantization, reduce concurrency/context, or load fewer capabilities. |

Keep diagnostics on stderr and command results on stdout when collecting logs.
Never include API keys, registry tokens, or signed model URLs in a report.
