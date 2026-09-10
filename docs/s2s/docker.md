# VoiceChat with Docker

The runtime image contains the VoiceChat server but not model artifacts. Build
the image once, then mount a converted model directory when starting the
container.

## Prerequisites

Install Docker with BuildKit and configure the NVIDIA Container Toolkit so
containers can access CUDA. Convert the model as described in
[Models and conversion](models.md) before starting the server.

The Docker builder stage installs the compiler, CUDA headers, CMake, Ninja,
SentencePiece, and other native build dependencies inside the image; they do
not need to be installed on the host.

Initialize the repository submodules used by the container build:

```bash
git submodule update --init --recursive
```

## Build the image

Build the focused S2S runtime without the unrelated gRPC, normalization, or
Flashlight components:

```bash
docker build -f docker/Dockerfile --target runtime \
  --build-arg ENABLE_S2S=ON \
  --build-arg ENABLE_GRPC=OFF \
  --build-arg ENABLE_HTTP=ON \
  --build-arg ENABLE_FLASHLIGHT=OFF \
  --build-arg ENABLE_NORM=OFF \
  -t nemo-speech-voicechat .
```

The resulting image uses `nemo-speech` as its entry point.

## Run the realtime server

Mount the converted model directory read-only and publish the realtime API on
the host's loopback interface only:

```bash
docker run --rm --gpus all --name nemo-voicechat \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:9000:9000 \
  -v "$PWD/models/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF:/models/voicechat:ro" \
  nemo-speech-voicechat \
  serve --host 0.0.0.0 --port 9000 \
  --s2s-model-dir /models/voicechat \
  --s2s-max-streams 1
```

`--user` runs the server as the host user instead of the image's baked-in
`uid 1000`. On some NFS exports, read access to a bind-mounted model
directory is restricted to its owning UID (and root) regardless of file
mode bits, so a mismatched in-image UID gets `Permission denied` reading the
GGUF files even when they are world-readable. Matching the host UID avoids
that; it is harmless on local disk too, where it simply keeps any files the
container writes owned by the host user instead of root.

Although the server listens on all interfaces inside the container, this port
mapping makes it reachable only from the Docker host. For remote access, place
it behind a TLS reverse proxy that authenticates clients; do not expose this
cleartext, unauthenticated command on a public or shared network.

Increase `--s2s-max-streams` only after sizing the deployment for its intended
concurrency. Model files remain on the host and can be replaced without
rebuilding the image.

From another terminal, verify readiness:

```bash
curl http://127.0.0.1:9000/v1/realtime/health
```

Then start a live conversation with the reference client:

```bash
python3 clients/voicechat/nemotron-voicechat-client.py \
  --server ws://localhost:9000
```

See [VoiceChat clients](clients.md) for file input, tool calling, concurrent
streams, and protocol integration.
