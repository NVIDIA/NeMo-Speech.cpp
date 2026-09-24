# Benchmarks

## Speech synthesis (MagpieTTS)

### NVIDIA DGX Spark (GB10)

MagpieTTS Multilingual 357M v2607 (Q8_0), streaming, one stream:

| Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|:---:|:---:|:---:|
| **22.9** · 30.9 | **5.9** · 25.8 | **28.7×** |

By input length:

| Input | Audio (s) | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| Short (8 words) | 3.1 | **21.9** · 26.3 | **26.8×** |
| Medium (55 words) | 22.0 | **23.2** · 28.0 | **26.6×** |
| Long (258 words) | 87.0 | **24.7** · 27.6 | **27.5×** |

Latencies are measured at the client from the streaming audio callback;
throughput is audio duration divided by wall time. The first table uses the 10
LJSpeech sentences of the Riva TTS performance reports
([`test_files/tts/ljs_audio_text_test_filelist_small.txt`](test_files/tts/ljs_audio_text_test_filelist_small.txt),
20 requests), the second [`test_files/tts/bench`](test_files/tts/bench)
(5 requests per input). Values are averages over three trials.

### Setup

| | |
|---|---|
| System | NVIDIA DGX Spark: GB10 GPU (48 SMs), 20-core Arm CPU, 128 GB unified memory |
| Software | Ubuntu 24.04, NVIDIA driver 580.126.09, CUDA 13.0 |
| Build | Release, `-DCMAKE_CUDA_ARCHITECTURES=121` |
| Models | MagpieTTS Multilingual 357M v2607 (Q8_0 GGUF), NeMo NanoCodec 22 kHz (F16 GGUF) |
| Synthesis | `en-US`, default voice, seed 1, 22.05 kHz audio in 186 ms chunks (4 codec frames) |

### Reproduce

```bash
nemo-speech bench tts --text-file test_files/tts/ljs_audio_text_test_filelist_small.txt \
  --per-stream 20 --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf \
  --tokenizer-dir tokenizer/

nemo-speech bench tts test_files/tts/bench -n 5 \
  --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf --tokenizer-dir tokenizer/
```
