# Native SDK integration

Use the native SDK when inference must run inside an existing process. For
shell workflows use the [CLI](cli.md); for language-neutral or remote access
use the [HTTP or gRPC interfaces](clients.md).

The supported embedding contract is the C ABI under
[`include/nemo_speech/`](../include/nemo_speech/). The headers are valid in
C and C++, expose opaque handles rather than implementation types, and do not
allow exceptions or C++ standard-library types to cross the ABI.

## Link an installed SDK

Install a selected build into a prefix as described in
[Build from source](build.md#install-the-sdk), then request only the components
the application uses:

```cmake
cmake_minimum_required(VERSION 3.26)
project(my_speech_app LANGUAGES CXX)

find_package(NeMoSpeech REQUIRED COMPONENTS ASR)

add_executable(my_speech_app main.cpp)
target_link_libraries(my_speech_app PRIVATE NeMoSpeech::ASR)
```

Pass the install prefix through `CMAKE_PREFIX_PATH` when configuring the
application:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/nemo-speech
cmake --build build
```

| Component | CMake target | Header | Capability |
|---|---|---|---|
| `ASR` | `NeMoSpeech::ASR` | `nemo_speech/asr.h` | offline and streaming recognition |
| `Diarization` | `NeMoSpeech::Diarization` | `nemo_speech/diar.h` | standalone speaker diarization |
| `NMT` | `NeMoSpeech::NMT` | `nemo_speech/nmt.h` | batched text translation |
| `TTS` | `NeMoSpeech::TTS` | `nemo_speech/tts.h` | text or token synthesis to PCM16 |

A component is available only when it was included in the installed build.
`find_package(... REQUIRED COMPONENTS ...)` reports a missing component during
application configuration.

## Runtime files and models

Keep the installed `bin` and `lib` directories together when packaging an
application; the installed libraries use a relative runtime search path for
their selected ggml backend and other runtime dependencies. On Windows, put the
SDK `bin` directory beside the application or on `PATH`.

Models and companion assets are not embedded in the SDK. The application owns
their storage and passes local paths when it creates an ASR recognizer,
diarization model, NMT translator, or TTS synthesizer. A GPU build also requires
the matching system driver and backend runtime on the target machine.

## ABI conventions

- Initialize every input struct to zero and set its `size` to `sizeof(struct)`.
  Use the provided `*_default()` functions where available; they also set
  `size` and populate library defaults.
- Model and startup configuration is copied by each `*_create` call. Opaque
  model handles must outlive streams or jobs created from them.
- A successful call that returns a result handle transfers that handle to the
  caller. Release it with the matching `*_result_destroy` or stream-close
  function.
- Strings returned by result accessors are owned by that result and become
  invalid when it is destroyed. PCM passed to a TTS callback is valid only for
  the duration of the callback.
- Failed calls return a status enum. The matching `*_last_error()` string is
  thread-local and remains valid until the next call in that API family on the
  same thread.
- The size-prefixed structs are append-only. Code built against an older header
  can pass a shorter struct and the runtime defaults fields added later.

## Minimal offline ASR

The SDK consumes mono float32 samples. ASR and diarization accept input rates
from 8 through 96 kHz and resample internally.

```cpp
#include <cstdio>
#include "nemo_speech/asr.h"

int transcribe(const float* samples, size_t count, int sample_rate) {
    nemo_speech_asr_backend_config backend{};
    backend.size = sizeof(backend);
    backend.gpu = 0;  // -1 for CPU

    nemo_speech_asr_model_config model{};
    model.size = sizeof(model);
    model.path = "models/asr.q8_0.gguf";

    nemo_speech_asr_recognizer_config config{};
    config.size = sizeof(config);
    config.backend = &backend;
    config.model = &model;

    nemo_speech_asr_recognizer* recognizer = nullptr;
    if (nemo_speech_asr_create(&config, &recognizer) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(stderr, "%s\n", nemo_speech_asr_last_error());
        return 1;
    }

    nemo_speech_asr_recognition_options options = nemo_speech_asr_recognition_options_default();
    options.enable_word_time_offsets = true;

    nemo_speech_asr_result* result = nullptr;
    const nemo_speech_asr_status status = nemo_speech_asr_recognize_f32(
        recognizer, &options, samples, count, sample_rate, &result);
    if (status != NEMO_SPEECH_ASR_OK) {
        std::fprintf(stderr, "%s\n", nemo_speech_asr_last_error());
        nemo_speech_asr_destroy(recognizer);
        return 1;
    }

    if (result && nemo_speech_asr_result_alternative_count(result) != 0)
        std::printf("%s\n", nemo_speech_asr_result_transcript(result, 0));

    nemo_speech_asr_result_destroy(result);
    nemo_speech_asr_destroy(recognizer);
    return 0;
}
```

For streaming ASR, create a stream with `nemo_speech_asr_streaming_recognize`, alternate
`nemo_speech_asr_stream_push_f32` with calls to `nemo_speech_asr_stream_next`, call
`nemo_speech_asr_stream_finish`, drain the remaining results with `next`, and release the
stream with `nemo_speech_asr_stream_close`.

One ASR recognizer can serve independent calls from multiple threads. Drive an
individual stream from one thread. Enabling `nemo_speech_asr_batching_config` on the
shared recognizer allows compatible work from concurrent utterances to form GPU
microbatches without creating additional model copies.

## Other capability flows

- **Diarization:** create `nemo_speech_diar_model`, use `nemo_speech_diar_offline_f32` or a
  `nemo_speech_diar_stream`, query `nemo_speech_diar_segments` first for its count and then with
  a caller-owned buffer, close the job, and destroy the model. Independent
  streams may use different threads; each stream is single-threaded.
- **NMT:** create `nemo_speech_nmt_translator`, pass one or more strings to
  `nemo_speech_nmt_translate`, read the returned translations, destroy the result, and
  destroy the translator. Calls may be concurrent; `nemo_speech_nmt_pool_config.contexts`
  sets the number of decode contexts available in parallel.
- **TTS:** create `nemo_speech_tts_synthesizer`, call `nemo_speech_tts_synthesize_text` or
  `nemo_speech_tts_synthesize_tokens`, consume each signed PCM16 chunk in the callback,
  and destroy the synthesizer. Returning `false` from the callback cancels the
  request.

## Complete examples

Build the examples with `NEMO_SPEECH_BUILD_EXAMPLES=ON`. Each example uses
only the stable C headers and component libraries:

| Example | Demonstrates |
|---|---|
| [`transcribe_file.cpp`](../examples/transcribe_file.cpp) | WAV loading, offline ASR, word timestamps, and optional speaker tags |
| [`transcribe_live.cpp`](../examples/transcribe_live.cpp) | microphone capture and streaming ASR |
| [`diarize_file.cpp`](../examples/diarize_file.cpp) | streaming or full-attention standalone diarization |
| [`translate_text.cpp`](../examples/translate_text.cpp) | batched NMT and result ownership |
| [`synthesize_text.cpp`](../examples/synthesize_text.cpp) | TTS callbacks, PCM collection, and WAV output |
| [`speech_translate_file.cpp`](../examples/speech_translate_file.cpp) | ASR → NMT → TTS composition through the three stable APIs |

The internal C++ engine classes are used by the CLI and servers but are not the
stable binary integration surface. Applications that need C++ ergonomics can
wrap the C handles in local RAII types without depending on runtime internals.
