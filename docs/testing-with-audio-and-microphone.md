# Testing with audio and microphone

This guide covers the Pascal-oriented Windows workflow: build the runtime, test one fixed WAV,
start a persistent HTTP server, and use the separate microphone client. The microphone client does
not load a model; the server owns the loaded model.

## Test-fixture licensing status

The expected fixture path is `test_files/fork/asr/teste-en.wav`, but the WAV is intentionally not
present yet. Its candidate source and redistribution license require manual review. Do not add,
commit, release, or claim redistribution rights for that audio until the review documents the real
source and license. The expected transcript is already present in `teste-en.txt`.

When an approved copy is available, copy it without modifying its bytes and compare SHA-256 before
adding it. The proposed candidate hash is
`148B936B43CE7C546A866E64DA059F0458AEE2D65E617F16E9D94F06E8D99ED6`; see the fixture
[README](../test_files/fork/asr/README.md) for its measured format and the pending-review notice.

## 1. Compilar

```powershell
.\scripts\windows\build-pascal.ps1
```

This creates `build-pascal-cuda-http\bin\nemo-speech.exe` with CUDA architecture 6.1 and the HTTP
server. It requires the Windows build prerequisites documented in `scripts/windows/build.ps1`.

## 2. Testar por arquivo

```powershell
.\scripts\windows\test-pascal-wav.ps1 `
  -Model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
```

The script uses the real CLI options `--device cuda:0`, `--skinny-q8 auto`, and
`--suppress-cuda-graph-log`. Use `-Device cpu` for a CPU run or `-Executable` to point to another
compiled binary.

## 3. Preparar cliente de microfone

```powershell
.\scripts\windows\setup-microphone-client.ps1
```

It creates `.tools\microphone-client-venv` and installs only `numpy`, `requests`, and
`sounddevice`. It does not install PyTorch, NeMo, or CUDA.

## 4. Iniciar servidor

```powershell
.\scripts\windows\run-pascal-server.ps1 `
  -Model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
```

Leave this PowerShell open. The process starts a local server at `http://127.0.0.1:8081`, exposes
`/ready` and `/v1/audio/transcriptions`, and keeps the model loaded between requests.

## 5. Testar o WAV via HTTP

```powershell
.\scripts\windows\test-http-wav.ps1
```

The script checks `/ready`, posts the WAV as multipart form data with `response_format=verbose_json`,
formats the response, measures HTTP-plus-inference time, and prints the expected and returned text
for visual comparison. It uses `curl.exe` and does not require Python.

## 6. Testar microfone em outro PowerShell

```powershell
& ".\.tools\microphone-client-venv\Scripts\python.exe" `
  ".\examples\python\microphone_http.py" `
  --language en `
  --show-words
```

Press Enter to start capture and Enter again to stop. The client captures 16 kHz mono audio, creates
an in-memory PCM16 WAV, posts it to the persistent server, and prints capture, preparation,
HTTP-plus-inference, RTF, and realtime-speed measurements. It does not save microphone audio by default.

## 7. Listar microfones

```powershell
& ".\.tools\microphone-client-venv\Scripts\python.exe" `
  ".\examples\python\microphone_http.py" `
  --list-devices
```

## 8. Escolher dispositivo

```powershell
& ".\.tools\microphone-client-venv\Scripts\python.exe" `
  ".\examples\python\microphone_http.py" `
  --device "Microphone" `
  --language en `
  --show-words
```

`--device` accepts an input-device index or a case-insensitive part of its name. If the name matches
more than one device, the client asks for an index. Other client controls are `--url` and
`--timeout` (default: 120 seconds).

## File benchmark

After the reviewed WAV exists, run:

```powershell
.\scripts\windows\benchmark-pascal-wav.ps1 `
  -Model "C:\Models\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf" `
  -Runs 10
```

It performs an excluded warm-up, records wall-clock time for each subsequent run, and saves JSON
and Markdown with minimum, maximum, mean, median, P95, hardware, model, commit, and command in
`benchmark-results/`. That directory is ignored by Git.

## Publishing checklist

Before publishing the WAV, confirm its actual source and redistribution license, preserve its bytes,
compare the SHA-256 hash, and document any required attribution. Do not add models, builds,
executables, virtual environments, caches, benchmark output, personal paths, or additional audio to
the repository.
