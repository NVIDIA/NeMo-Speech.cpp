# NeMo NanoCodec GGUF

This directory covers converting the NVIDIA NeMo NanoCodec 22 kHz checkpoint
to a GGUF decoder and running the token-to-audio path used by MagpieTTS.

Model: <https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps>

## Convert

Run the converter from a source checkout after following
[`docs/model-conversion.md`](../../../docs/model-conversion.md):

```bash
python convert_model.py models/nano-codec/extracted \
  --outfile models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --metadata-json models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.gguf.json
```

The converter writes the inference decoder and FSQ codebook metadata required
to turn codec tokens into audio.

## Decode

Build the decoder:

```bash
scripts/configure.sh cpu-tts -DNEMO_SPEECH_BUILD_TOOLS=ON
cmake --build --preset cpu-tts --target nanocodec
```

Provide a text file containing codec tokens, then run:

```bash
build/cpu-tts/bin/nanocodec \
  -m models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --codes magpie_codes.txt \
  -o magpie.wav
```

The token file is plain text with 8 integer codebook IDs per frame.
