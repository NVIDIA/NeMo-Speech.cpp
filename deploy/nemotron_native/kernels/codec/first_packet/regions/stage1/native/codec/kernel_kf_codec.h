#pragma once
#include <cstddef>
struct CodecArgs {
    const void *inp[41];
    const void *st[26];
    void *ns[26];
    void *waveform;
    void *scratch;
    int B;
    int F;
};
void codec_launch(const CodecArgs &a, void *stream);
size_t codec_scratch_bytes(int B, int F);
