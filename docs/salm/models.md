# SALM models and platforms

## Getting the model

[Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) is published as a BF16 checkpoint. Download it with
the Hugging Face CLI, then convert it with `convert_model.py` from a source checkout with the converter's Python
requirements ([Model conversion](../model-conversion.md#quick-start)): once per decoder precision you need, each
run writing the decoder and, next to it, the encoder file they share. Choose the decoder precision for your hardware:

| Hardware | `--outtype` |
|---|---|
| NVIDIA Blackwell GPU | `nvfp4` |
| Other NVIDIA GPUs, the CPU, Vulkan | `iq4_nl` |

For example:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --local-dir checkpoints/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
python3 convert_model.py checkpoints/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --architecture salm \
    --outtype nvfp4 --outfile models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4.gguf
```

The example writes the NVFP4 decoder; for another precision, change `--outtype` and the file name. Every entrypoint
takes the path of the decoder you converted:

```bash
MODEL=models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4.gguf
nemo-speech transcribe audio.wav --model $MODEL
nemo-speech serve --asr-model $MODEL
riva_server --asr.model.path $MODEL
```

The optional [DFlash2 draft model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2) converts on its own, with the checkpoint it drafts for
([DFlash2 draft model](../model-conversion.md#dflash2-draft-model)), and is passed as `asr.salm.dflash_path`:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2 --local-dir checkpoints/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2
python3 convert_model.py checkpoints/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2 --architecture dflash2 \
    --salm-target checkpoints/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --outfile models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2-BF16.gguf
nemo-speech serve --asr-model $MODEL \
    --asr.salm.dflash_path models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2-BF16.gguf
```

## What the files are

A model is two GGUF files in one directory: the language-model decoder, which is the model you pass, and the encoder
file it names (`<checkpoint>-Encoder-F16.gguf`): the speech encoder with its speaker branch, and the CTC timestamp head
(word timestamps and speaker labels). Decoders of every precision share one encoder file, and a decoder refuses an
encoder file of another checkpoint. The DFlash2 draft model is a separate optional file.
[Model conversion](../model-conversion.md#speech-llm-salm) makes the files from a checkpoint. `--salm-encoder` (or
`asr.salm.encoder_path`) loads another encoder file than the one the decoder names.

## Platforms

The default CUDA build decodes many requests together in one batch and drafts several tokens ahead (speculative
decoding), so it is the one to serve with. The other builds run llama.cpp's standard path: it works on the CPU and
with Vulkan and decodes requests side by side without speculative decoding.

| Build | Path | Notes |
|---|---|---|
| CUDA with `patches/` (the default), NVIDIA GPU of compute capability 8.0 or newer | batched serving | many requests decoded together, MTP or DFlash2 speculative decoding; needs CUDA 12.8 or newer |
| CPU, Vulkan, CUDA without `patches/` (`NEMO_SPEECH_GGML_PATCHED=OFF`) or older GPUs | upstream llama.cpp | up to 256 requests at once, no speculative decoding; IQ4_NL decoder |
