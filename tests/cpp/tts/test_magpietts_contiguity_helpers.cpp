// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Covers the as_contig/as_f32_contig helpers in src/tts/magpietts/graph.h.
//
// The helpers replace ggml_cont/ggml_cont_2d/ggml_cont_3d/ggml_cast at sites
// where the tensor usually already satisfies what is being asked for. They have
// to be two things at once, and the two pull in opposite directions:
//
//   1. an optimization -- when the input already qualifies, nothing that touches
//      memory is emitted, because ggml_cont of an already-contiguous tensor is a
//      full device-to-device copy that computes nothing;
//   2. never a change in behaviour -- when the input does NOT qualify, the
//      result must be bit-identical to what the ggml builtin would produce.
//
// Checking only (1) would pass a helper that skipped a copy it actually needed,
// which is the dangerous direction. Checking only (2) would pass a helper that
// just called ggml_cont every time, which is the thing being removed. So every
// case below asserts both, against the builtin it replaces: the bytes, and the
// number of nodes that materialise memory.
//
// Raw node count is the wrong measure here. The helpers fall back to
// ggml_reshape_*, which is still a node but a pure view -- ggml computes nothing
// for it. What the optimization removes is CONT/CPY/DUP, so those are what get
// counted. That also makes the F32 cases honest: as_f32_contig drops a redundant
// CPY even when the input is permuted and a real CONT is still required, so it
// saves work in both the qualifying and the non-qualifying case, just different
// amounts.
//
// The non-contiguous inputs are built with ggml_permute, which is how they
// arise at the attention sites these helpers are applied to.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "tts/magpietts/graph.h"

namespace {

using nemo_speech::tts::as_contig;
using nemo_speech::tts::as_contig_2d;
using nemo_speech::tts::as_contig_3d;
using nemo_speech::tts::as_f32_contig;

constexpr int64_t kD = 16;  // d_head
constexpr int64_t kH = 4;   // n_head
constexpr int64_t kT = 6;   // tokens

using Build = ggml_tensor* (*)(ggml_context*, ggml_tensor*);

ggml_tensor*
identity(ggml_context*, ggml_tensor* x) {
    return x;
}

// Contiguous in memory, but not in ggml's index order, so ggml_is_contiguous is
// false and a real copy is required.
ggml_tensor*
permuted(ggml_context* ctx, ggml_tensor* x) {
    return ggml_permute(ctx, ggml_reshape_3d(ctx, x, kD, kH, kT), 0, 2, 1, 3);
}

ggml_tensor*
bi_cont(ggml_context* ctx, ggml_tensor* t) {
    return ggml_cont(ctx, t);
}
ggml_tensor*
he_cont(ggml_context* ctx, ggml_tensor* t) {
    return as_contig(ctx, t);
}
ggml_tensor*
bi_cont2(ggml_context* ctx, ggml_tensor* t) {
    return ggml_cont_2d(ctx, t, kD * kH, kT);
}
ggml_tensor*
he_cont2(ggml_context* ctx, ggml_tensor* t) {
    return as_contig_2d(ctx, t, kD * kH, kT);
}
ggml_tensor*
bi_cont3(ggml_context* ctx, ggml_tensor* t) {
    return ggml_cont_3d(ctx, t, kD, kH, kT);
}
ggml_tensor*
he_cont3(ggml_context* ctx, ggml_tensor* t) {
    return as_contig_3d(ctx, t, kD, kH, kT);
}
// F16 inputs, which is what makes the cast in as_f32_contig a real one rather
// than a no-op. The cast to F16 is itself a materialising node, so it is counted
// in both arms and the expected totals below include it.
ggml_tensor*
as_f16(ggml_context* ctx, ggml_tensor* x) {
    return ggml_cast(ctx, x, GGML_TYPE_F16);
}
ggml_tensor*
as_f16_permuted(ggml_context* ctx, ggml_tensor* x) {
    return ggml_permute(ctx, ggml_reshape_3d(ctx, as_f16(ctx, x), kD, kH, kT), 0, 2, 1, 3);
}

ggml_tensor*
bi_f32(ggml_context* ctx, ggml_tensor* t) {
    return ggml_cont(ctx, ggml_cast(ctx, t, GGML_TYPE_F32));
}
ggml_tensor*
he_f32(ggml_context* ctx, ggml_tensor* t) {
    return as_f32_contig(ctx, t);
}

struct Case {
    const char* name;
    Build prepare;
    Build builtin;
    Build helper;
    int want_materializing;  // nodes the helper may spend on copies
};

// CONT, CPY and DUP are the ops that actually move bytes; RESHAPE, VIEW,
// PERMUTE and TRANSPOSE are views ggml computes nothing for.
bool
materializes(ggml_op op) {
    return op == GGML_OP_CONT || op == GGML_OP_CPY || op == GGML_OP_DUP;
}

bool
run_arm(
    Build prepare, Build build, const std::vector<float>& in, std::vector<uint8_t>& out_bytes,
    int& out_nodes) {
    ggml_init_params p = {
        /*.mem_size   =*/64u * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context* ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kD * kH, kT);
    std::memcpy(x->data, in.data(), in.size() * sizeof(float));

    ggml_tensor* out = build(ctx, prepare(ctx, x));

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    out_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (materializes(ggml_graph_node(gf, i)->op)) {
            ++out_nodes;
        }
    }
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    out_bytes.assign((const uint8_t*)out->data, (const uint8_t*)out->data + ggml_nbytes(out));
    ggml_free(ctx);
    return true;
}

}  // namespace

int
main() {
    std::vector<float> in((size_t)kD * kH * kT);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = (float)i * 0.125f - 3.0f;
    }

    const Case cases[] = {
        // Already contiguous, already F32: nothing should touch memory at all.
        {"as_contig / contiguous", identity, bi_cont, he_cont, 0},
        {"as_contig_2d / contiguous", identity, bi_cont2, he_cont2, 0},
        {"as_contig_3d / contiguous", identity, bi_cont3, he_cont3, 0},
        {"as_f32_contig / F32", identity, bi_f32, he_f32, 0},
        // Permuted: exactly one real copy, and bytes equal to ggml's.
        {"as_contig / permuted", permuted, bi_cont, he_cont, 1},
        {"as_contig_3d / permuted", permuted, bi_cont3, he_cont3, 1},
        {"as_f32_contig / permuted", permuted, bi_f32, he_f32, 1},
        // F16 in: the cast is required, so the helper must still emit it. Counts
        // include the one materialising node the F16 input itself costs.
        {"as_f32_contig / F16", as_f16, bi_f32, he_f32, 2},
        {"as_f32_contig / F16 permuted", as_f16_permuted, bi_f32, he_f32, 2},
    };

    int failures = 0;
    for (const Case& c : cases) {
        std::vector<uint8_t> want, got;
        int want_nodes = 0, got_nodes = 0;
        if (!run_arm(c.prepare, c.builtin, in, want, want_nodes) ||
            !run_arm(c.prepare, c.helper, in, got, got_nodes)) {
            std::fprintf(stderr, "FAIL %-28s could not build the graph\n", c.name);
            ++failures;
            continue;
        }

        if (want.size() != got.size() || std::memcmp(want.data(), got.data(), want.size()) != 0) {
            std::fprintf(
                stderr, "FAIL %-28s output differs from the ggml builtin (%zu vs %zu bytes)\n",
                c.name, got.size(), want.size());
            ++failures;
        } else if (got_nodes != c.want_materializing) {
            std::fprintf(
                stderr,
                "FAIL %-28s materialises %d node(s), expected %d (the builtin it replaces "
                "materialises %d)\n",
                c.name, got_nodes, c.want_materializing, want_nodes);
            ++failures;
        } else if (got_nodes > want_nodes) {
            std::fprintf(
                stderr, "FAIL %-28s materialises %d node(s), more than the builtin's %d\n", c.name,
                got_nodes, want_nodes);
            ++failures;
        } else {
            std::fprintf(
                stderr, "ok   %-28s bytes match, %d copy node(s) against the builtin's %d\n",
                c.name, got_nodes, want_nodes);
        }
    }

    std::fprintf(
        stderr, "%s: %d of %d cases failed\n", failures ? "FAILED" : "PASSED", failures,
        (int)(sizeof(cases) / sizeof(cases[0])));
    return failures == 0 ? 0 : 1;
}
