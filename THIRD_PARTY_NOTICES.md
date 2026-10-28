# Third-Party Notices

This file covers third-party source and data incorporated into or modified by
the NeMo-Speech.cpp source tree. The Git submodule revisions are pinned by
the repository's gitlinks. Their upstream notices remain in the submodule
checkouts and are summarized here.

## Git submodules

### ggml

- Source: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (`ggml/`,
  developed and vendored in llama.cpp)
- Path: `llama.cpp/ggml`
- Copyright (c) 2023-2026 The ggml authors
- License: MIT; upstream text: [`llama.cpp/LICENSE`](llama.cpp/LICENSE)

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

The runtime uses these file-level exceptions:

- `util/murmur_hash.cc`: Austin Appleby, MIT option
- `util/string_piece.cc` and `util/string_piece.hh`: Google and RE2 authors,
  BSD 3-Clause
- `util/double-conversion/*`: V8 authors, BSD 3-Clause
- `util/integer_to_string.cc` and `util/integer_to_string.hh`: Milo Yip and
  Wojciech Muła, MIT/BSD

All other files in the project's explicit KenLM runtime source allowlist are
LGPL 2.1 or later. KenLM's AT&T-licensed `util/getopt.c` and `util/getopt.hh`
are explicitly excluded and are not compiled or linked.

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

OpenSSL is linked only when HTTP-server TLS is enabled. Windows packages include
its license. The default Linux container does not include OpenSSL.

### Windows vcpkg dependencies

Windows builds obtain optional dependencies through vcpkg. Installed packages
include each selected dependency's license.

### Container system libraries

The Linux container copies the shared libraries required by its executables
from the CUDA/Ubuntu builder into a `FROM scratch` runtime. This includes the
GNU C Library, GCC runtime libraries, c-ares, zlib, bzip2, and XZ Utils/liblzma,
subject to the exact feature set and package revisions in the built image.

The container records the package copyright file for every copied
Debian/Ubuntu shared library under
`/opt/nemo-speech/share/licenses/nemo-speech/third_party/system/`. The complete
project notice set is installed under
`/opt/nemo-speech/share/licenses/nemo-speech/`.

### NVIDIA CUDA Runtime

- Source: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Copyright (c) NVIDIA CORPORATION & AFFILIATES
- License: NVIDIA Software License Agreement and CUDA Supplement

CUDA release archives include the redistributable CUDA runtime library, and the
Windows ones also cuBLAS and cuBLASLt. The applicable agreement is installed with
the archive under
`share/licenses/nemo-speech/nvidia/cuda-runtime/`.

## Other incorporated third-party code and data

### miniaudio

- Source: [`mackron/miniaudio`](https://github.com/mackron/miniaudio), version
  0.11.25 at commit `9634bedb5b5a2ca38c1ee7108a9358a4e233f14d`, vendored by
  the pinned llama.cpp checkout
- Path: `llama.cpp/vendor/miniaudio/miniaudio.h`
- Copyright 2026 David Reid
- License: Public Domain (Unlicense) or MIT No Attribution (MIT-0); upstream
  text is reproduced at
  [`third_party/miniaudio/LICENSE`](third_party/miniaudio/LICENSE)

The command-line microphone capture layer compiles miniaudio directly into
`nemo-speech`, and the audio ingest (`src/common/audio_miniaudio.cpp`) compiles
its embedded dr_flac and dr_mp3 decoders into every build, with all miniaudio
symbols internal to that translation unit. Release archives install its license
under `share/licenses/nemo-speech/third_party/miniaudio/`.

### SentencePiece

- Source: [`google/sentencepiece`](https://github.com/google/sentencepiece),
  revision `31646a467d2051eb904e0b45de3a73e91fe1c1e3`
- Copyright 2018 Google Inc.
- License: Apache License 2.0

Default Windows ASR, macOS, and Linux release builds statically link the
SentencePiece runtime and its bundled Abseil, protobuf-lite, and Darts-clone
components. Their Apache 2.0 and BSD license texts are installed under
`share/licenses/nemo-speech/third_party/sentencepiece/`.

### Text normalization runtime

`scripts/build_itn_deps.sh` builds these pinned sources for ITN and TN:

- OpenFST: [`sarane22/openfst`](https://github.com/sarane22/openfst), revision
  `fc23b4cf529429284b874a26f28b15c6cc94f404`; Copyright 2005-2024 Google LLC;
  Apache License 2.0
- Sparrowhawk: [`sarane22/sparrowhawk`](https://github.com/sarane22/sparrowhawk),
  revision `8b082acc507312077a096be8398584a13832c490`; Copyright 2015 and
  onwards Google, Inc.; Apache License 2.0
- Protocol Buffers:
  [`protocolbuffers/protobuf`](https://github.com/protocolbuffers/protobuf)
  v21.12; Copyright 2008 Google Inc.; BSD 3-Clause License
- RE2: [`google/re2`](https://github.com/google/re2) 2023-03-01; Copyright (c)
  2009 The RE2 Authors; BSD 3-Clause License

Linux and macOS release archives statically link all four into
`libnemo_speech_text_normalization`. Builds that use system Protocol Buffers and
RE2 link those instead. The license texts are installed under
`share/licenses/nemo-speech/third_party/`.

### whisper.cpp sample audio

The ASR quick-start fixture at `test_files/asr/wav/test/jfk.wav` is copied from
[`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp), revision
`23ee03506a91ac3d3f0071b40e66a430eebdfa1d`.

Copyright (c) 2023-2026 The ggml authors. License: MIT; the license text is
reproduced below.

### parakeet.cpp

Portions of `src/runtime/ggml/`, `src/asr/encoder/fastconformer.*`, and
`src/asr/encoder/rel_pos_attention.*` are derived from
[`parakeet.cpp`](https://github.com/jason-ni/parakeet.cpp), revision
`1675ee5b612b629645e3bb4b42f671da27a76d5b`.

Copyright (c) 2025 Jason Ni

### Mandarin tokenizer

The generated Mandarin tokenizer tables contain data derived from:

- Jieba 0.42.1, Copyright Sun Junyi and contributors
- pypinyin 0.55.0, Copyright 2016 mozillazg and 闲耘
- pypinyin-dict 0.9.0, Copyright 2021 mozillazg

The CppJieba and limonp sources used to consume these tables are listed above.

### SciPy, NumPy and PyTorch numerics

The speech-LLM word aligner (`src/speechlm/pee/timestamp_alignment.cpp`) reproduces
these routines so that its alignments equal NeMo's Python aligner exactly:

- SciPy's rectangular linear sum assignment
  (`scipy/optimize/rectangular_lsap/rectangular_lsap.cpp`, the shortest augmenting
  path algorithm after D. F. Crouse; author P. M. Larsen):
  [`scipy/scipy`](https://github.com/scipy/scipy); Copyright (c) 2001-2002
  Enthought, Inc. 2003, SciPy Developers; BSD 3-Clause License
- NumPy's float32 logarithm kernel (`simd_log_FLOAT`, including its polynomial
  coefficients) and its pairwise summation:
  [`numpy/numpy`](https://github.com/numpy/numpy); Copyright (c) 2005-2025, NumPy
  Developers; BSD 3-Clause License
- PyTorch's float32 `linspace` arithmetic:
  [`pytorch/pytorch`](https://github.com/pytorch/pytorch); Copyright (c) 2016-
  Facebook, Inc (Adam Paszke) and the other contributors listed in its `LICENSE`;
  BSD 3-Clause License

The BSD 3-Clause license text is reproduced below.

### Unicode Character Database

The generated tables `src/common/grapheme_break_table.inc` (ICU 74, Unicode 15.1),
`src/speechlm/pee/timestamp_unit_tables.inc` (Unicode 15.1) and
`src/speechlm/letter_mark_table.inc` (Unicode 15.0) contain data derived from the
Unicode Character Database (`scripts/gen_grapheme_tables.py`,
`scripts/gen_timestamp_unit_tables.py`, `scripts/gen_letter_mark_table.py`).

Copyright © 1991-2026 Unicode, Inc. License: Unicode License v3; the license text
is reproduced below.

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

## BSD 3-Clause License

The following text applies to the BSD-licensed material identified above, with
the copyright notices listed for each.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Unicode License v3

UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

## NVIDIA llama.cpp and ggml patches

The `patches/` directory contains NVIDIA-authored changes applied to the pinned
MIT-licensed llama.cpp and ggml source. New source files created by those patches
carry the NVIDIA Apache-2.0 header; existing files retain their upstream
notices. The resulting combined source and binaries retain the MIT notice.
