# TTS models

The TTS pipeline loads two GGUFs: a **MagpieTTS** token generator and a **NeMo
NanoCodec** decoder. Ready-to-run F16 GGUFs are published in their Hugging Face
repositories. Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

## MagpieTTS token generator

Hugging Face: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)

```bash
# Download the v2602 GGUF and the original archive containing its tokenizer.
hf download nvidia/magpie_tts_multilingual_357m \
    --include magpie_tts_multilingual_357m.v2602.f16.gguf \
    --include magpie_tts_multilingual_357m.nemo \
    --local-dir models/magpie-tts

# Extract the tokenizer assets loaded by the runtime.
mkdir -p models/magpie-tts/extracted
tar -xf models/magpie-tts/magpie_tts_multilingual_357m.nemo \
    -C models/magpie-tts/extracted
```

**Tokenizer.** MagpieTTS's tokenizer assets live *inside* the `.nemo` archive -
they are not part of the GGUF. Extract the `.nemo` and pass that directory to
the server as `--tts.tokenizer-model-dir` (here
`models/magpie-tts/extracted`).
The model-specific IPA/text tokenizer assets are loaded from this directory.
Japanese tokenization requires a build with `NEMO_SPEECH_TTS_WITH_JA=ON`
(disabled by default), which builds Open JTalk, MeCab, and the NAIST dictionary.
Mandarin requires `NEMO_SPEECH_TTS_WITH_ZH=ON` (disabled by default) and
additionally uses cppjieba plus pypinyin-compatible tables bundled with the
native runtime. Those tables are stored in Git LFS, so run `git lfs install`
once and `git lfs pull` before configuring this feature. No Python environment
is needed when serving `zh` or `zh-CN`. Run `git lfs pull` in the checkout
before `docker build` as well, because the build context does not include
`.git`.

**Text normalization.** TTS can optionally run Sparrowhawk TN before Magpie
tokenization. Build with `-DNEMO_SPEECH_WITH_NORM=ON`, install the WFST
dependencies with `scripts/build_itn_deps.sh`, and pass a TN grammar directory
such as `models/tn_configs` through `--tts.tn-model-dir`. The expected
multilingual layout matches ASR ITN: immediate language-named children such as
`en/`, `fr/`, and `vi/`, each containing `tokenize_and_classify.far`,
`verbalize.far`, and optionally `post_process.far`. A direct single-language
grammar directory and the older split `classify/` and `verbalize/` layout remain
supported. See [TTS text normalization](configuration.md#text-normalization) for
server, YAML, and offline runner examples.

## NanoCodec decoder

Hugging Face: [nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
(no tokenizer is needed for the codec decoder).

```bash
hf download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
    nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
    --local-dir models/nano-codec
```

Run both models together:

```bash
nemo-speech synthesize "Hello from Magpie Multilingual." \
    --magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
    --codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
    --tokenizer-dir models/magpie-tts/extracted \
    --output output.wav
```

## Converting custom TTS checkpoints

The unified [`convert_model.py`](../../convert_model.py) entry point accepts
compatible local `.nemo` archives and extracted NeMo checkpoints. It defaults
to `--outtype f16` for MagpieTTS and NanoCodec; pass `--outtype f32` to retain
full precision.

```bash
pip install -r requirements.txt
python3 convert_model.py custom-magpie.nemo --outfile custom-magpie.f16.gguf
```

The converter reads `.nemo` archives directly with PyTorch and does not require
`nemo_toolkit`. The optional `scripts/tts/tokenize-magpietts.py` debugging
helper does use NeMo's Python tokenizer implementation.

## Notes

For CUDA builds, the MagpieTTS and NanoCodec operations require the ggml
patches applied by `scripts/configure.sh`; see
[ggml patches](../development/ggml-patches.md).

Once converted, point the server at them - see
[TTS configuration](configuration.md).
