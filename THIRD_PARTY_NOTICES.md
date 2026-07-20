# Third-Party Notices

This file covers third-party source and data incorporated into or modified by
the NeMo-Speech.cpp source tree. The Git submodule revisions are pinned by
the repository's gitlinks. Their upstream notices remain in the submodule
checkouts and are summarized here.

## Git submodules

### ggml

- Source: [`ggml-org/ggml`](https://github.com/ggml-org/ggml)
- Path: `ggml`
- Copyright (c) 2023-2026 The ggml authors
- License: MIT; upstream text: [`ggml/LICENSE`](ggml/LICENSE)

### llama.cpp

- Source: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- Path: `llama.cpp`
- Copyright (c) 2023-2026 The ggml authors
- License: MIT; upstream text: [`llama.cpp/LICENSE`](llama.cpp/LICENSE)

The pinned llama.cpp checkout also carries this notice for gguf-py, which is
used by its model-conversion tooling:

- gguf-py: Copyright (c) 2023 Georgi Gerganov; MIT;
  [`llama.cpp/gguf-py/LICENSE`](llama.cpp/gguf-py/LICENSE)

### Riva common protobuf definitions

- Source: [`nvidia-riva/common`](https://github.com/nvidia-riva/common)
- Path: `proto/riva-common`
- Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- License: MIT; upstream text:
  [`proto/riva-common/LICENSE`](proto/riva-common/LICENSE)

### Flashlight Text

- Source: [`flashlight/text`](https://github.com/flashlight/text)
- Path: `third_party/flashlight-text`
- Copyright (c) Facebook, Inc. and its affiliates.
- License: MIT; upstream text:
  [`third_party/flashlight-text/LICENSE`](third_party/flashlight-text/LICENSE)

### KenLM

- Source: [`kpu/kenlm`](https://github.com/kpu/kenlm)
- Path: `third_party/kenlm`
- Upstream attribution: language-model inference code by Kenneth Heafield;
  additional file-level copyright notices are retained and summarized below
- License: primarily GNU Lesser General Public License v2.1 or later;
  upstream notice: [`third_party/kenlm/LICENSE`](third_party/kenlm/LICENSE);
  full terms: [`COPYING`](third_party/kenlm/COPYING),
  [`COPYING.3`](third_party/kenlm/COPYING.3), and
  [`COPYING.LESSER.3`](third_party/kenlm/COPYING.LESSER.3)

File-level exceptions used by the runtime include MurmurHash (Austin Appleby,
public domain/MIT), StringPiece (Google and RE2 authors, BSD 3-Clause),
double-conversion (V8 authors, BSD 3-Clause), and integer conversion code (Milo
Yip and Wojciech Muła, MIT/BSD). Their upstream notices remain intact.

KenLM's AT&T-licensed `util/getopt.*` files are explicitly excluded from this
project's runtime source allowlist and are not compiled or linked.

### Open JTalk

- Source: [`r9y9/open_jtalk`](https://github.com/r9y9/open_jtalk)
- Path: `third_party/open_jtalk`
- License: BSD 3-Clause
- Open JTalk: Copyright (c) 2008-2016 Nagoya Institute of Technology,
  Department of Computer Science;
  [`src/COPYING`](third_party/open_jtalk/src/COPYING)
- Bundled MeCab: Copyright (c) 2001-2008 Taku Kudo and Copyright (c) 2004-2008
  Nippon Telegraph and Telephone Corporation;
  [`src/mecab/COPYING`](third_party/open_jtalk/src/mecab/COPYING)
- Bundled NAIST Japanese dictionary: Copyright (c) 2009 Nara Institute of
  Science and Technology, Japan;
  [`src/mecab-naist-jdic/COPYING`](third_party/open_jtalk/src/mecab-naist-jdic/COPYING)

### CppJieba and limonp

- Source: [`yanyiwu/cppjieba`](https://github.com/yanyiwu/cppjieba)
- Path: `third_party/cppjieba`
- Copyright (c) 2013
- License: MIT; upstream text:
  [`third_party/cppjieba/LICENSE`](third_party/cppjieba/LICENSE)
- Nested submodule: [`yanyiwu/limonp`](https://github.com/yanyiwu/limonp),
  path `third_party/cppjieba/deps/limonp`, Copyright (c) 2013, MIT;
  upstream text: `third_party/cppjieba/deps/limonp/LICENSE`

### cpp-httplib

- Source: [`yhirose/cpp-httplib`](https://github.com/yhirose/cpp-httplib)
- Path: `third_party/cpp-httplib`
- Copyright (c) 2017 yhirose
- License: MIT; upstream text:
  [`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)

### OpenSSL

- Source: [OpenSSL](https://www.openssl.org/)
- Copyright The OpenSSL Project Authors and other contributors
- License: Apache License 2.0

OpenSSL is dynamically linked for optional HTTP-server TLS. It is not
incorporated into the source tree; binary and container distributions include
the applicable OpenSSL attribution and Apache 2.0 terms.

## Other incorporated third-party code and data

### whisper.cpp sample audio

The ASR quick-start fixture at `test_files/asr/wav/test/jfk.wav` is copied from
[`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp), revision
`23ee03506a91ac3d3f0071b40e66a430eebdfa1d`.

Copyright (c) 2023-2026 The ggml authors. License: MIT; the license text is
reproduced below.

### parakeet.cpp

Portions of `riva/runtime/ggml/`, `riva/asr/encoder/fastconformer.*`, and
`riva/asr/encoder/rel_pos_attention.*` are derived from
[`parakeet.cpp`](https://github.com/jason-ni/parakeet.cpp), revision
`1675ee5b612b629645e3bb4b42f671da27a76d5b`.

Copyright (c) 2025 Jason Ni

### Mandarin tokenizer

The generated Mandarin tokenizer tables contain data derived from:

- Jieba 0.42.1, Copyright Sun Junyi and contributors
- pypinyin 0.55.0, Copyright 2016 mozillazg and 闲耘
- pypinyin-dict 0.9.0, Copyright 2021 mozillazg

The CppJieba and limonp sources used to consume these tables are listed above.

## MIT License

The following text applies to the MIT-licensed components and derived material
identified above. Component-specific copyright notices remain as listed with
each component.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## NVIDIA ggml patches

The `ggml-patches/` directory contains NVIDIA-authored changes applied to the
pinned MIT-licensed ggml source. New source files created by those patches
carry the NVIDIA Apache-2.0 header; existing ggml files retain their upstream
notices. The resulting combined source and binaries retain ggml's MIT notice.
