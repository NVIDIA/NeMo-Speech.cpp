# Benchmarks

## Speech synthesis (MagpieTTS)

### NVIDIA GeForce RTX 4090

MagpieTTS Multilingual 357M v2607 (Q8_0), streaming, one stream:

| Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|:---:|:---:|:---:|
| **8.6** · 9.4 | **2.9** · 3.1 | **59.8×** |

By input length:

| Input | Audio (s) | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| Short (8 words) | 2.6 | **8.4** · 9.8 | **56.5×** |
| Medium (55 words) | 22.8 | **9.0** · 10.3 | **54.3×** |
| Long (258 words) | 92.6 | **10.1** · 11.0 | **56.0×** |

### NVIDIA DGX Spark (GB10)

MagpieTTS Multilingual 357M v2607 (Q8_0), streaming, one stream:

| Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|:---:|:---:|:---:|
| **16.8** · 18.1 | **5.8** · 25.5 | **30.0×** |

By input length:

| Input | Audio (s) | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| Short (8 words) | 3.1 | **18.6** · 23.2 | **27.9×** |
| Medium (55 words) | 22.0 | **16.3** · 17.8 | **27.5×** |
| Long (258 words) | 87.0 | **18.4** · 19.2 | **28.3×** |

Latencies are measured at the client from the streaming audio callback;
throughput is audio duration divided by wall time. The first table uses the 10
LJSpeech sentences of the Riva TTS performance reports
([`test_files/tts/ljs_audio_text_test_filelist_small.txt`](test_files/tts/ljs_audio_text_test_filelist_small.txt),
20 requests), the second [`test_files/tts/bench`](test_files/tts/bench)
(5 requests per input). Values are averages over three trials.

### Setup

| | RTX 4090 | DGX Spark |
|---|---|---|
| System | GeForce RTX 4090 (128 SMs, 24 GB), Intel Core i7-11700K (16 threads), 128 GB RAM | NVIDIA DGX Spark: GB10 GPU (48 SMs), 20-core Arm CPU, 128 GB unified memory |
| Software | Ubuntu 24.04, NVIDIA driver 595.84, CUDA 13.2 | Ubuntu 24.04, NVIDIA driver 580.126.09, CUDA 13.0 |
| Build | Release, `-DCMAKE_CUDA_ARCHITECTURES=89` | Release, `-DCMAKE_CUDA_ARCHITECTURES=121` |
| Models | MagpieTTS Multilingual 357M v2607 (Q8_0 GGUF), NeMo NanoCodec 22 kHz (F16 GGUF) | same |
| Synthesis | `en-US`, default voice, seed 1, 22.05 kHz audio in 186 ms chunks (4 codec frames) | same |

### Reproduce

```bash
nemo-speech bench tts --text-file test_files/tts/ljs_audio_text_test_filelist_small.txt \
  --per-stream 20 --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf \
  --tokenizer-dir tokenizer/

nemo-speech bench tts test_files/tts/bench -n 5 \
  --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf --tokenizer-dir tokenizer/
```
