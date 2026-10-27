# SALM models and platforms

## Getting the model

Convert the checkpoint with `convert_model.py` ([Model conversion](../model-conversion.md#speech-llm-salm)): it writes
the decoder GGUF and its encoder file next to it. Every entrypoint takes the decoder's path and loads the encoder file
from the same directory:

```bash
nemo-speech transcribe audio.wav --model models/salm.gguf
nemo-speech serve --asr-model models/salm.gguf
riva_server --asr.model.path models/salm.gguf
```

The DFlash2 draft model converts on its own ([DFlash2 draft model](../model-conversion.md#dflash2-draft-model)) and
is passed as `asr.salm.dflash_path`.

## Model files

A model is two GGUF files in one directory: the language-model decoder, which is the model you pass, and the encoder
file it names (`<checkpoint>-Encoder-F16.gguf`): the speech encoder with its speaker branch, and the CTC timestamp head
(word timestamps and speaker labels). Decoders of every precision share one encoder file, and a decoder refuses an
encoder file of another checkpoint. The DFlash2 draft model is a separate optional file.
[Model conversion](../model-conversion.md#speech-llm-salm) makes the files from a checkpoint. `--salm-encoder` (or
`asr.salm.encoder_path`) loads another encoder file than the one the decoder names.

## Platforms

| Build | Path | Notes |
|---|---|---|
| CUDA with `patches/` (the default), NVIDIA GPU of compute capability 8.0 or newer | batched serving | many requests decoded together, MTP or DFlash2 speculative decoding; needs CUDA 12.8 or newer |
| CPU, Vulkan, Metal, CUDA without `patches/` (`NEMO_SPEECH_GGML_PATCHED=OFF`) or older GPUs | upstream llama.cpp | up to 256 requests at once, no speculative decoding; IQ4_NL decoder |
