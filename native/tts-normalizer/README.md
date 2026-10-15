# Native TTS normalizer

This package is the source-owned build boundary for the CPU-only WFST text
normalizer used by the Rust EasyMagpie adapter. It is intentionally independent
of a NemotronSpeech checkout and exposes only processor ABI v1 from
`include/processor_abi.h`.

The final deployment still contains a shared library because the stable native
processor ABI is the isolation boundary between the Rust adapter and the C++
normalizer. The library is now built and tested from this directory rather than
stored as a Git LFS binary.

## Build and test

The supported build runs in `docker/Dockerfile.frontend`. The checked-in English
test grammars under `testdata/en-US` make a matching local Bazel 6.2.1 build
hermetic:

```bash
bazel test --config=release \
    //:normalize_test \
    //:speech_hints_test \
    //:text_preprocessor_test \
    //:text_preprocessor_c_test
bazel build --config=release //:text_preprocessor_c
```

The shared library is produced as
`bazel-bin/libtext_preprocessor_c.so`. The container installs it as
`/opt/nemotron-speech/lib/processors/libnemotron_speech_tts_text_preprocessor.so`.

## Ownership boundary

This directory owns normalization, sentence splitting, bounded SSML handling,
abbreviation expansion, and the processor ABI wrapper. It does not own model
tokenization, session state, streaming boundary detection, language pool sizing,
or vLLM transport; those remain in the Rust adapter and frontend.

The English FAR/proto/abbreviation files under `testdata/en-US` are test fixtures
only. They are consumed by the native Bazel tests and are not copied into the
runtime image. Production deployments use the external multilingual asset root
configured in `rust/config/example.json` (or an explicit deploy-time override);
the source build does not generate or package production grammars.

All external source archives are revision- and SHA-256-pinned. An uncached build
therefore needs network access, but it does not depend on mutable Git branches or
an adjacent source checkout.

See `LICENSE.nemotron_speech` and `THIRD_PARTY_NOTICES.md` for licensing information.
