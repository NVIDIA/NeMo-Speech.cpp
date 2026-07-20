# Mandarin G2P data

These generated runtime tables reproduce the Mandarin frontend used by the
MagpieTTS NeMo configuration. They are generated from Jieba 0.42.1, pypinyin
0.55.0, and pypinyin-dict 0.9.0 by
`scripts/tts/generate_mandarin_tokenizer_data.py`.

CppJieba, Jieba, pypinyin, and pypinyin-dict are distributed under the MIT
license. CppJieba and its limonp dependency retain their license files in their
submodule directories. Project-level attribution is consolidated in the root
`THIRD_PARTY_NOTICES.md`. Source versions and generated-file checksums are
recorded in `manifest.json`.

Regenerate the runtime tables with the pinned Python packages installed:

```bash
python3 scripts/tts/generate_mandarin_tokenizer_data.py \
  --output-dir riva/tts/tokenizer/mandarin_data
```
