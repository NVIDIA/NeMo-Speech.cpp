# Pascal performance observations

This document records observed default-versus-custom behavior on one local Pascal system. It is the source of truth for the fork's performance numbers; it does not claim an upstream-wide performance improvement.

## Hardware used

| Item | Value |
| --- | --- |
| Operating system | Windows 11 |
| GPU | NVIDIA GeForce GTX 1060 6 GB |
| Architecture | Pascal |
| Compute Capability | 6.1 |
| CPU | Intel Xeon E5-2660 v2 |
| RAM | 32 GB |
| CUDA Toolkit | 12.6 |
| Model | Nemotron 3.5 ASR Streaming 0.6B Q8 GGUF |
| Server | Persistent HTTP server |
| Endpoint | `/v1/audio/transcriptions` |

All tests and observations below were run and manually verified by **UNDER192103 / Under Nouzen** ([GitHub](https://github.com/UNDER192103)). This attribution applies to the fork-specific tests and documentation, not to the original NeMo-Speech.cpp project.

## Long-audio result

The reproducible long-audio test used `test_files/asr/wav/test/jfk.wav`.

| Property | Value |
| --- | --- |
| Duration | 11 seconds |
| Sample rate | 16 kHz |
| Channels | mono |
| Bit depth | 16 bits |
| Warm-up | 1 execution |
| Measured executions | 10 |
| Language | English |

| Runtime | Minimum | Maximum | Mean | Median | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 178.47 ms | 197.39 ms | 183.00 ms | 181.41 ms | 197.39 ms |
| Custom | 175.08 ms | 193.77 ms | 179.64 ms | 178.26 ms | 193.77 ms |

The observed median difference is approximately **3.15 ms**, or an observed relative reduction of approximately **1.7%**. This is small and may be within normal system variation. For this 11-second test, default and custom performance were practically equivalent; this result does not support a claim of a significant long-audio speed improvement.

## Short-speech result

These preliminary observations came from the HTTP microphone client while repeatedly speaking:

```text
Olá! Como que você está hoje?
```

Each capture was approximately 2.3–2.9 seconds including leading and trailing silence.

### Default observed

```text
670.57 ms
195.10 ms
151.69 ms
163.22 ms
620.39 ms
```

Minimum: 151.69 ms; maximum: 670.57 ms; median: 195.10 ms.

### Custom observed

```text
73.63 ms
77.19 ms
74.29 ms
72.31 ms
```

Minimum: 72.31 ms; maximum: 77.19 ms; median: approximately 73.96 ms.

| Short scenario | Samples | Minimum | Maximum | Median |
| --- | ---: | ---: | ---: | ---: |
| Default | 5 | 151.69 ms | 670.57 ms | 195.10 ms |
| Custom | 4 | 72.31 ms | 77.19 ms | 73.96 ms |

The preliminary median comparison is 195.10 ms versus 73.96 ms: an observed difference of approximately 121.14 ms and an observed relative reduction of approximately 62%.

> [!IMPORTANT]
> The short-speech results are preliminary observations made with the local microphone. Although the same phrase, computer, microphone, HTTP client, and model were used, every recording has small differences in duration, silence, intensity, and pronunciation.
>
> Therefore, the approximately 62% reduction represents behavior observed in this environment. It is not yet a scientific benchmark or performance guarantee.

The microphone does not execute inference. It creates an in-memory mono PCM16 WAV and sends it to the same HTTP endpoint; inference runs entirely in whichever default or custom server is open. The capture interval is not included in the reported `HTTP + inference` time. The pattern was consistent enough to warrant investigation, but it still needs repeated testing with exactly the same short WAV on both runtimes.

## Current interpretation

The current evidence suggests that the custom runtime does not significantly change throughput for longer recordings, such as the 11-second JFK sample.

However, on the tested GTX 1060 6 GB, the custom runtime showed substantially lower and more consistent latency for short requests around two to three seconds.

This may indicate that the fork changes affect fixed per-request overhead, backend initialization, buffer preparation, kernel selection, synchronization, or another short-request execution path. The exact cause has not yet been isolated.

Do not attribute the observation to one flag. `--suppress-cuda-graph-log` only suppresses a log; it does not enable CUDA Graphs or improve inference speed. `--skinny-q8 auto` does not introduce a new Pascal kernel: on the GTX 1060 it selects the existing compatible fallback.

## Reproducible short-WAV benchmark

The planned fixture is `test_files/fork/asr/short-en.wav`, a two-to-three-second English PCM16, mono, 16 kHz WAV with a documented phrase. It has **not** been added: no appropriate local short English WAV with a verified redistribution license was found. Consequently there is no source, license, duration, transcript, or SHA-256 to publish yet, and the short comparison is not fully reproducible.

Do not add a personal microphone capture, a Portuguese fixture, or any WAV without confirmed redistribution rights. When a suitable audio source is approved, document its exact transcript in `short-en.txt`, copy it byte-for-byte to `test_files/fork/asr/short-en.wav`, compare SHA-256, and record its source, license, format, duration, and attribution in the fixture README.

### Benchmark commands

Start either server yourself; the benchmark deliberately does not start or switch a server.

```powershell
.\scripts\windows\benchmark-short-wav.ps1 `
  -Runtime default `
  -Model "C:\Models\nemotron.gguf" `
  -Runs 20
```

```powershell
.\scripts\windows\benchmark-short-wav.ps1 `
  -Runtime custom `
  -Model "C:\Models\nemotron.gguf" `
  -Runs 20
```

The script checks `/ready`, warms up once, sends the exact same WAV for every measured request, measures HTTP plus inference only, reports each run plus minimum, maximum, mean, median, P95, RTF, and realtime speed, and writes ignored JSON and Markdown results under `benchmark-results/`.

The public microphone client can be used separately for exploratory testing:

```powershell
python .\examples\python\microphone_http.py `
  --url "http://127.0.0.1:8081/v1/audio/transcriptions" `
  --language en `
  --show-words
```
