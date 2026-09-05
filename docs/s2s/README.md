# Nemotron Labs VoiceChat

Nemotron Labs VoiceChat is a full-duplex speech-to-speech model with realtime
audio, transcripts, and tool calling. This guide converts the model, starts a
local server, and connects a microphone client.

## What you need

- An NVIDIA GPU with CUDA and at least 16 GB of GPU-accessible memory for the
  recommended Q4_K_M model
- The CUDA toolkit
- Git, Python 3.10+, CMake 3.26+, Ninja, and a C++17 compiler

On Ubuntu/Debian, install the remaining system packages with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git curl pkg-config libsentencepiece-dev \
  libsndfile1 portaudio19-dev python3-venv
```

See [Build from source](../build.md) for other platforms and toolchains.

## 1. Set up the source checkout

```bash
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git
cd NeMo-Speech.cpp
git submodule update --init ggml llama.cpp third_party/cpp-httplib

python3 -m venv .venv
. .venv/bin/activate
python -m pip install \
  -r requirements.txt \
  -r clients/voicechat/requirements.txt
```

PyAudio is only needed for microphone capture and playback.

## 2. Convert the model

```bash
python convert_model.py \
  nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --outfile models/voicechat
```

This creates the recommended Q4_K_M bundle. Downloads use the Hugging Face
cache, so unchanged files are reused. If access is denied, accept the model
terms on Hugging Face and run `hf auth login` before trying again.

See [Model conversion](models.md) for NVFP4, BF16, local checkpoints, and other
options.

## 3. Build and start the server

```bash
scripts/configure.sh cuda-s2s
cmake --build --preset cuda-s2s
```

Start a local server for one conversation:

```bash
build/cuda-s2s/bin/nemo-speech serve \
  --s2s-model-dir models/voicechat \
  --s2s-max-streams 1
```

The server listens on `localhost:8080`. When model loading finishes, its
readiness endpoint is available at
`http://localhost:8080/v1/realtime/health`.

## 4. Talk to VoiceChat

In another terminal:

```bash
cd NeMo-Speech.cpp
. .venv/bin/activate
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080
```

Press Ctrl+C to end the session. The client saves the response audio,
transcripts, tool calls, and conversation log.

To use a WAV file without microphone or playback dependencies:

```bash
python clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:8080 \
  --input-file input.wav \
  --audio-output response.wav \
  --no-playback
```

## Next steps

- [Client options and tool calling](clients.md)
- [Server configuration, authentication, and concurrency](configuration.md)
- [Realtime API reference](../api.md#voicechat-websocket-v1realtime-and-realtime)
- [Optional Docker deployment](docker.md)
