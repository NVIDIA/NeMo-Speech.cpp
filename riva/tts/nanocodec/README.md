# NeMo NanoCodec GGUF

This runner converts the NVIDIA NeMo NanoCodec 22 kHz checkpoint to a GGUF
decoder and runs the token-to-audio path used by MagpieTTS.

Model: <https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps>

## Convert

```bash
python convert_model.py models/nemo_nano_codec_22khz_1.89kbps_21.5fps/extracted \
  --outfile models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --metadata-json models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.gguf.json
```

The converter writes only the inference decoder needed for codec tokens to
audio. It folds PyTorch weight norm into plain convolution weights, stores
Conv1d tensors in GGML layout, expands grouped ConvTranspose1d kernels into
GGML-compatible dense kernels, and records the deterministic FSQ quantizer
metadata/codebook.

## Decode

Build the decoder:

```bash
cmake -S . -B build -DNEMO_SPEECH_BUILD_TOOLS=ON
cmake --build build --target nanocodec -j
```

Provide a text file containing codec tokens, then run:

```bash
build/bin/nanocodec \
  -m models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --codes magpie_codes.txt \
  -o magpie.wav
```

The token file is plain text with 8 integer codebook IDs per frame.
