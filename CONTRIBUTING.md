# Contributing

We welcome external contributions to NeMo-Speech.cpp.

## Development checks

Follow the [source-build guide](docs/build.md) for prerequisites and submodules.
For a model-independent CPU ASR test build:

```bash
git submodule update --init ggml llama.cpp
scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TESTS=ON
cmake --build --preset cpu-asr
ctest --test-dir build/cpu-asr --output-on-failure
```

Install [pre-commit](https://pre-commit.com/) and run the same formatting,
license-header, and static file checks used by CI:

```bash
pre-commit run --all-files
```

Use the closest matching CUDA, Metal, Vulkan, server, or component preset when
the change affects code outside the CPU ASR path. Include the commands and
results relevant to the change in the pull request.

## Contribution license and provenance

Unless a file states otherwise, contributions are submitted under the
[Apache License 2.0](LICENSE). Preserve all existing copyright, license, and
attribution notices.

If a contribution contains or is derived from third-party code, data, generated
content, or model artifacts, identify its source, revision, copyright holder,
and license in the pull request. Preserve its original notices and update
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) when applicable. Do not submit
material under terms that are incompatible with this project. Contributions
may be held for provenance and license review before acceptance.

## Signing off your work

Every commit must be signed off. The sign-off certifies that you have the right
to submit the contribution under the license indicated in the file. Commits
without a `Signed-off-by` line will not be accepted.

Use Git's `--signoff` (or `-s`) option:

```bash
git commit --signoff -m "Add a feature"
```

This appends:

```text
Signed-off-by: Your Name <your@email.com>
```

The full, unmodified [Developer Certificate of Origin
1.1](https://developercertificate.org/) follows:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
