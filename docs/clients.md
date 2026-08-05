# Client integration

Start a local server with the models needed by your application:

```bash
nemo-speech serve --asr-model models/asr.q8_0.gguf
```

The HTTP server implements the OpenAI audio API subset documented in the
[HTTP API reference](api.md).
An API key is only required when the server was started with `--api-key`; SDKs
still require a nonempty placeholder locally.

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
with open("recording.wav", "rb") as audio:
    result = client.audio.transcriptions.create(model="default", file=audio)
print(result.text)
```

## OpenAI JavaScript SDK

```javascript
import { createReadStream } from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "local",
});
const result = await client.audio.transcriptions.create({
  model: "default",
  file: createReadStream("recording.wav"),
});
console.log(result.text);
```

Browser code should use the playground's realtime WebSocket protocol rather
than placing an API key in a public page.

## curl

```bash
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@recording.wav -F model=default -F response_format=verbose_json

curl -s http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","voice":"default","input":"Hello","response_format":"wav"}' \
  -o hello.wav
```

## Riva-compatible gRPC clients

Start the Riva-compatible listener:

```bash
riva_server --asr.model.path models/asr.q8_0.gguf --bind 0.0.0.0:50051
```

Use NVIDIA Riva's existing
[Python clients](https://github.com/nvidia-riva/python-clients) or
[C++ clients](https://github.com/nvidia-riva/cpp-clients) against
`127.0.0.1:50051`. No NeMo-Speech.cpp-specific client is required.

## In-process C and C++

For embedding the runtime directly, see [Native SDK integration](sdk.md). It
covers the installed CMake components, stable C ABI, object lifetimes,
threading, and the complete checked-in examples.
