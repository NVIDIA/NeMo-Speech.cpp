# TTS models

The TTS pipeline loads two GGUFs: a **MagpieTTS** token generator and a **NeMo
NanoCodec** decoder. The CLI downloads the complete default stack, including
Magpie's tokenizer assets, with one command:

```bash
nemo-speech pull magpie
nemo-speech synthesize "Hello from Magpie Multilingual." --output output.wav
```

`synthesize` performs the same verified pull automatically when its model
options are omitted.

## MagpieTTS token generator

Hugging Face: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)

**Tokenizer.** MagpieTTS's tokenizer assets live *inside* the `.nemo` archive -
they are not part of the GGUF. The built-in pull extracts only the required,
pinned tokenizer members and verifies each one. For a custom Magpie checkpoint,
extract its `.nemo` archive and pass that directory as `--tokenizer-dir` or
`--tts.tokenizer-model-dir`.
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

Pull it independently with `nemo-speech pull nano-codec`. Pulling `magpie`
does this automatically because the two models must run together.

Run the default stack:

```bash
nemo-speech synthesize "Hello from Magpie Multilingual." --output output.wav
```

## Converting custom TTS checkpoints

The unified [`convert_model.py`](../../convert_model.py) entry point accepts
compatible local `.nemo` archives and extracted NeMo checkpoints. It defaults
to `--outtype f16` for MagpieTTS and NanoCodec; pass `--outtype f32` to retain
full precision. The converter is a source-tree Python tool and is not included
in native release archives; see [Model conversion](../model-conversion.md) for
environment setup.

```bash
python3 convert_model.py custom-magpie.nemo --outfile custom-magpie.f16.gguf
```

Conversion does not require `nemo_toolkit`. The optional
`scripts/tts/tokenize-magpietts.py` debugging helper does.

Once converted, point the server at them; see
[TTS configuration](configuration.md).
