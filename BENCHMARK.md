# Benchmarks

## Speech recognition (Nemotron Speech Streaming)

Nemotron Speech Streaming EN 0.6B, cache-aware streaming, LibriSpeech
test-clean (2,620 utterances, 5.4 h).

### GeForce RTX 4090

| Engine | Chunk | Compute per chunk (ms)<br>avg · p99 | Throughput (RTFX) | WER |
|---|:---:|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | 1.12 s | **3.4** · 4.6 | **238.4×** | 2.51% |
| NeMo (FP32) | 1.12 s | 21.2 · 23.1 | 49.3× | 2.32% |
| NeMo-Speech.cpp (Q8_0) | 160 ms | **2.3** · 2.8 | **66.6×** | 2.66% |
| NeMo (FP32) | 160 ms | 20.7 · 22.7 | 7.6× | 2.69% |

### DGX Spark (GB10)

| Engine | Chunk | Compute per chunk (ms)<br>avg · p99 | Throughput (RTFX) | WER |
|---|:---:|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | 1.12 s | **6.6** · 8.6 | **120.4×** | 2.50% |
| NeMo (FP32) | 1.12 s | 20.4 · 21.7 | 50.9× | 2.32% |
| NeMo-Speech.cpp (Q8_0) | 160 ms | **4.7** · 5.4 | **32.2×** | 2.64% |
| NeMo (FP32) | 160 ms | 18.9 · 19.7 | 8.4× | 2.69% |

### CPU

Intel Core i7-11700K, 8 threads for both engines, on a 100-utterance subset of test-clean (13.3 min).

| Engine | Chunk | Compute per chunk (ms)<br>avg · p99 | Throughput (RTFX) | WER |
|---|:---:|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | 1.12 s | **42.7** · 56.4 | **19.2×** | 3.10% |
| NeMo (FP32) | 1.12 s | 187.9 · 211.6 | 5.6× | 2.96% |
| NeMo-Speech.cpp (Q8_0) | 160 ms | **27.0** · 28.9 | **5.6×** | 3.00% |
| NeMo (FP32) | 160 ms | 103.5 · 112.5 | 1.5× | 3.14% |

## Speech synthesis (MagpieTTS)

MagpieTTS Multilingual 357M v2607, streaming.

### GeForce RTX 4090

| Engine | Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | **8.6** · 9.4 | **2.9** · 3.1 | **59.8×** |
| NeMo (FP32) | 129.3 · 137.0 | 123.5 · 131.7 | 1.5× |

By input length:

| Input | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) | NeMo (FP32): time to first audio (ms)<br>avg · p99 | NeMo (FP32): throughput (RTFX) |
|---|:---:|:---:|:---:|:---:|
| Short (8 words) | **8.4** · 9.8 | **56.5×** | 126.0 · 127.9 | 1.5× |
| Medium (55 words) | **9.0** · 10.3 | **54.3×** | 127.1 · 131.1 | 1.5× |
| Long (258 words) | **10.1** · 11.0 | **56.0×** | 127.1 · 128.2 | 1.5× |

### DGX Spark (GB10)

| Engine | Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | **16.8** · 18.1 | **5.8** · 25.5 | **30.0×** |
| NeMo (FP32) | 87.3 · 95.5 | 86.7 · 92.8 | 2.1× |

By input length:

| Input | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) | NeMo (FP32): time to first audio (ms)<br>avg · p99 | NeMo (FP32): throughput (RTFX) |
|---|:---:|:---:|:---:|:---:|
| Short (8 words) | **18.6** · 23.2 | **27.9×** | 86.6 · 87.3 | 2.2× |
| Medium (55 words) | **16.3** · 17.8 | **27.5×** | 90.4 · 91.8 | 2.0× |
| Long (258 words) | **18.4** · 19.2 | **28.3×** | 90.1 · 91.2 | 2.0× |

### CPU

Intel Core i7-11700K, 8 threads for both engines.

| Engine | Time to first audio (ms)<br>avg · p99 | Inter-chunk latency (ms)<br>avg · p99 | Throughput (RTFX) |
|---|:---:|:---:|:---:|
| NeMo-Speech.cpp (Q8_0) | **203.4** · 238.2 | **63.8** · 86.8 | **2.72×** |
| NeMo (FP32) | 659.8 · 679.1 | 665.0 · 761.6 | 0.28× |

By input length:

| Input | Time to first audio (ms)<br>avg · p99 | Throughput (RTFX) | NeMo (FP32): time to first audio (ms)<br>avg · p99 | NeMo (FP32): throughput (RTFX) |
|---|:---:|:---:|:---:|:---:|
| Short (8 words) | **191.0** · 208.2 | **2.58×** | 659.7 · 663.8 | 0.28× |
| Medium (55 words) | **199.2** · 233.1 | **2.60×** | 663.3 · 666.5 | 0.27× |
| Long (258 words) | **216.4** · 229.9 | **2.61×** | 686.5 · 697.9 | 0.27× |

## Devices

### GeForce RTX 4090

| | |
|---|---|
| System | GeForce RTX 4090 (128 SMs, 24 GB), Intel Core i7-11700K (16 threads), 128 GB RAM |
| Software | Ubuntu 24.04, NVIDIA driver 595.84, CUDA 13.2 |
| Build | Release, `-DCMAKE_CUDA_ARCHITECTURES=89` |
| NeMo | NeMo 3.1 (`main`), PyTorch 2.12.1 (CUDA 13.2), TF32 matmuls |

### CPU

Intel Core i7-11700K, the RTX 4090 host's CPU, with the same build and software, run with `--device cpu`.

| | |
|---|---|
| System | Intel Core i7-11700K (8 cores, 16 threads, AVX-512), 128 GB RAM |
| Threads | 8 for both engines: `--asr.backend.threads 8` / `--tts.threads 8`; NeMo `torch.set_num_threads(8)` |

### DGX Spark (GB10)

| | |
|---|---|
| System | GB10 GPU (48 SMs), 20-core Arm CPU, 128 GB unified memory |
| Software | Ubuntu 24.04, NVIDIA driver 580.126.09, CUDA 13.0 |
| Build | Release, `-DCMAKE_CUDA_ARCHITECTURES=121` |
| NeMo | NeMo 3.1, PyTorch 2.11 (CUDA 13.0), TF32 matmuls |

## Methodology

Precision is shown per engine. NeMo-Speech.cpp runs Q8_0 GGUF weights; NeMo runs
FP32, which was faster than BF16 here.

### Speech recognition

| | |
|---|---|
| Model | Nemotron Speech Streaming EN 0.6B |
| Chunks | 1.12 s (`--asr.streaming.rnnt_right_context 13`), 160 ms (`1`) |
| Metrics | Compute per chunk is the time to process one chunk; NeMo's excludes feature extraction, which its streaming example runs per utterance. Throughput is audio duration over wall time; NeMo's likewise excludes feature extraction. WER uses the Whisper English normalizer. |
| CPU subset | Every 26th utterance of test-clean (100 utterances, 40 speakers, 13.3 min) |
| Trials | RTX 4090 and CPU: NeMo-Speech.cpp average of three, NeMo one |

```bash
OUT_DIR=datasets scripts/asr/prepare_librispeech.sh test-clean

nemo-speech bench asr datasets/librispeech-test-clean -r --mode stream -c 1 -n 1 \
  --model nemotron-speech-streaming-en-0.6b.q8_0.gguf \
  --asr.streaming.rnnt_right_context 13 --save hyp/
```

Score `hyp/` against `datasets/librispeech-test-clean/transcripts.json` with the
Whisper English normalizer.

### Speech synthesis

| | |
|---|---|
| Models | MagpieTTS Multilingual 357M v2607 (Q8_0, [converted locally](docs/tts/models.md#magpietts-token-generator)), NeMo NanoCodec 22 kHz (F16) |
| Synthesis | `en-US`, default voice, seed 1, 22.05 kHz audio in 186 ms chunks (4 codec frames) |
| Inputs | The 10 LJSpeech sentences of the Riva TTS performance reports ([`ljs_audio_text_test_filelist_small.txt`](test_files/tts/ljs_audio_text_test_filelist_small.txt), 20 requests); by length, [`test_files/tts/bench`](test_files/tts/bench) (5 requests per input) |
| Metrics | Latencies at the client from the streaming audio callback; throughput is audio duration over wall time |
| Trials | NeMo-Speech.cpp: average of three; NeMo: one |

NeMo audio is decoded in 4-frame chunks as codes are produced.

```bash
nemo-speech bench tts --text-file test_files/tts/ljs_audio_text_test_filelist_small.txt \
  --per-stream 20 --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf \
  --tokenizer-dir tokenizer/

nemo-speech bench tts test_files/tts/bench -n 5 \
  --magpie-model magpie.q8_0.gguf --codec-model nanocodec.gguf --tokenizer-dir tokenizer/
```
