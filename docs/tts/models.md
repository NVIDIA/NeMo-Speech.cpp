# TTS models and conversion

The TTS pipeline loads two GGUFs: a **MagpieTTS** token generator and a **NeMo
NanoCodec** decoder. Both are converted from NeMo checkpoints - either public
models from Hugging Face or compatible local NeMo checkpoints (an extracted
checkpoint or `.nemo` archive can be passed to the converter).

The unified [`convert_model.py`](../../convert_model.py) entry point defaults to
**`--outtype f16`** for both models; pass `--outtype f32` to keep full precision.

The converters read `.nemo` archives directly with PyTorch and do not require
`nemo_toolkit`. The optional `scripts/tts/tokenize-magpietts.py` debugging
helper does use NeMo's Python tokenizer implementation.

## MagpieTTS token generator

Hugging Face: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
(the repo is a single `.nemo` archive).

```bash
# 1. download the .nemo
hf download nvidia/magpie_tts_multilingual_357m --local-dir magpie-tts

# 2. extract it - this directory holds the tokenizer the server loads at runtime
mkdir -p magpie-tts/extracted
tar -xf magpie-tts/magpie_tts_multilingual_357m.nemo -C magpie-tts/extracted

# 3. convert to GGUF (the converter also accepts the .nemo path directly)
python3 convert_model.py magpie-tts/extracted \
    --outfile magpie-tts/magpie.f16.gguf --outtype f16
```

**Tokenizer.** MagpieTTS's tokenizer assets live *inside* the `.nemo` archive -
they are not part of the GGUF. Extract the `.nemo` (step 2) and pass that
directory to the server as `--tts.tokenizer-model-dir` (here `magpie-tts/extracted`).
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
(a single `.nemo` archive; no tokenizer needed - it's a codec decoder).

```bash
hf download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps --local-dir nano-codec
python3 convert_model.py \
    nano-codec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
    --outfile nano-codec/nano-codec.decoder.f16.gguf
```

## Notes

For CUDA builds, the MagpieTTS and NanoCodec operations require the ggml
patches applied by `scripts/configure.sh`; see
[ggml patches](../development/ggml-patches.md).

Once converted, point the server at them - see
[TTS configuration](configuration.md).
