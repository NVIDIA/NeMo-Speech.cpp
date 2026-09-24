// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "model_logging.h"
#include "nvtx_utils.h"

static constexpr int NANO_CODEC_MAX_NODES = 32768;

using nc_hparams = nemo_speech::tts::nanocodec::NanoCodecHParams;

static bool
is_default_graph_node_name(const ggml_tensor* tensor) {
    if (!tensor) {
        return false;
    }
    const char* name = ggml_get_name(tensor);
    return !name || name[0] == '\0' || std::strncmp(name, "node_", 5) == 0;
}

static void
tag_graph_first_node(ggml_cgraph* gf) {
    const int n_nodes = gf ? ggml_graph_n_nodes(gf) : 0;
    ggml_tensor* first = n_nodes > 0 ? ggml_graph_node(gf, 0) : nullptr;
    if (!first || !is_default_graph_node_name(first)) {
        return;
    }
    const char* label = ggml_get_name(ggml_graph_node(gf, n_nodes - 1));
    if (label && label[0]) {
        ggml_set_name(first, label);
    }
}

struct nc_activation {
    ggml_tensor* alpha = nullptr;
    ggml_tensor* alpha_inv = nullptr;
};

struct nc_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
    // Optional F32 copy of `w` prepared at load time for ops that require F32 weights
    // (transposed convolutions), replacing a per-call in-graph cast.
    ggml_tensor* w_f32 = nullptr;
    // Optional F16 weights repacked as [cout_pad, cin_pad, K] for the fused CUDA conv op.
    ggml_tensor* w_packed = nullptr;
    int stride = 1;
    int dilation = 1;
};

struct nc_res_block {
    nc_activation input_act;
    nc_activation skip_act;
    nc_conv input_conv;
    nc_conv skip_conv;
};

// The three kernel-size branches of a residual stage batched into one fused op per conv slot:
// weights of the branches packed back to back, biases / snake parameters concatenated.
struct nc_grouped_conv {
    ggml_tensor* w = nullptr;      // F16 [cin_pad, cout_pad, sum K_g]
    ggml_tensor* b = nullptr;      // F32 [groups * cout]
    ggml_tensor* alpha = nullptr;  // F32 [groups * snake] or nullptr
    ggml_tensor* alpha_inv = nullptr;
    int groups = 0;
    int K[3] = {1, 1, 1};
    int d[3] = {1, 1, 1};
    int cout = 0;
    int snake = 0;
};
struct nc_res_layer {
    std::vector<std::vector<nc_res_block>> by_kernel;
    // grouped fused path (CUDA): one op per residual-block slot across the kernel-size branches
    std::vector<nc_grouped_conv> grouped_in;    // input convs of block d (d = dilation index)
    std::vector<nc_grouped_conv> grouped_skip;  // skip convs of block d
    bool grouped = false;
};

struct nc_model {
    nc_hparams hparams;

    gguf_context* gguf = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    // Derived tensors (F32 transposed-conv weights) created after the GGUF load.
    ggml_context* aux_ctx = nullptr;
    ggml_backend_buffer_t aux_buffer = nullptr;
    ggml_context* pack_ctx = nullptr;  // fused-conv packed weights
    ggml_backend_buffer_t pack_buffer = nullptr;
    ggml_context* group_ctx =
        nullptr;  // grouped fused-conv tensors (weights, biases, snake params)
    ggml_backend_buffer_t group_buffer = nullptr;
    bool fused_conv = false;

    nc_conv pre_conv;
    std::vector<nc_activation> activations;
    std::vector<nc_conv> up_convs;
    std::vector<nc_res_layer> res_layers;
    nc_activation post_activation;
    nc_conv post_conv;
};

namespace nemo_speech::tts::nanocodec {

struct NanoCodecModel::Impl {
    nc_model model;
    bool loaded = false;
};

}  // namespace nemo_speech::tts::nanocodec

static int32_t
gguf_i32(const gguf_context* ctx, const char* key, int32_t def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    const gguf_type t = gguf_get_kv_type(ctx, id);
    if (t == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id);
    }
    if (t == GGUF_TYPE_UINT32) {
        return (int32_t)gguf_get_val_u32(ctx, id);
    }
    if (t == GGUF_TYPE_INT64) {
        return (int32_t)gguf_get_val_i64(ctx, id);
    }
    if (t == GGUF_TYPE_UINT64) {
        return (int32_t)gguf_get_val_u64(ctx, id);
    }
    return def;
}

static std::vector<int32_t>
gguf_i32_array(const gguf_context* ctx, const char* key, const std::vector<int32_t>& def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        return def;
    }

    const size_t n = gguf_get_arr_n(ctx, id);
    const gguf_type t = gguf_get_arr_type(ctx, id);
    const void* data = gguf_get_arr_data(ctx, id);
    std::vector<int32_t> out(n);

    if (t == GGUF_TYPE_INT32) {
        const int32_t* p = (const int32_t*)data;
        for (size_t i = 0; i < n; ++i) {
            out[i] = p[i];
        }
        return out;
    }
    if (t == GGUF_TYPE_INT64) {
        const int64_t* p = (const int64_t*)data;
        for (size_t i = 0; i < n; ++i) {
            out[i] = (int32_t)p[i];
        }
        return out;
    }
    if (t == GGUF_TYPE_UINT32) {
        const uint32_t* p = (const uint32_t*)data;
        for (size_t i = 0; i < n; ++i) {
            out[i] = (int32_t)p[i];
        }
        return out;
    }
    if (t == GGUF_TYPE_UINT64) {
        const uint64_t* p = (const uint64_t*)data;
        for (size_t i = 0; i < n; ++i) {
            out[i] = (int32_t)p[i];
        }
        return out;
    }
    return def;
}

static ggml_tensor*
require_tensor(const nc_model& model, const std::string& name) {
    ggml_tensor* t = ggml_get_tensor(model.ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", name.c_str());
        std::exit(1);
    }
    return t;
}

static nc_activation
load_activation(const nc_model& model, const std::string& prefix) {
    nc_activation act;
    act.alpha = require_tensor(model, prefix + ".alpha");
    act.alpha_inv = require_tensor(model, prefix + ".alpha_inv");
    return act;
}

static nc_conv
load_conv(const nc_model& model, const std::string& prefix, int stride = 1, int dilation = 1) {
    nc_conv conv;
    conv.w = require_tensor(model, prefix + ".w");
    conv.b = require_tensor(model, prefix + ".b");
    conv.stride = stride;
    conv.dilation = dilation;
    return conv;
}

// Batch the kernel-size branches of every residual stage: for each block slot d, the input convs
// of the G branches become one grouped op (and likewise the skip convs), so a stage runs 2*D
// launches instead of 2*D*G, each with G times the tiles. Requires the per-conv packed weights.
static void
nc_pack_grouped_convs(nc_model& model, bool verbose) {
    struct plan_item {
        nc_grouped_conv* out;
        std::vector<const nc_conv*> convs;
        std::vector<const nc_activation*> acts;
    };
    std::vector<plan_item> items;
    for (nc_res_layer& layer : model.res_layers) {
        const size_t G = layer.by_kernel.size();
        if (G < 1 || G > 3)
            continue;
        const size_t D = layer.by_kernel[0].size();
        bool ok = D > 0;
        for (const auto& stack : layer.by_kernel) ok = ok && stack.size() == D;
        if (!ok)
            continue;
        for (const auto& stack : layer.by_kernel)
            for (const nc_res_block& blk : stack)
                ok = ok && blk.input_conv.w_packed && blk.skip_conv.w_packed &&
                     blk.input_act.alpha && blk.skip_act.alpha &&
                     blk.input_conv.w_packed->ne[0] == blk.skip_conv.w_packed->ne[0];
        if (!ok)
            continue;
        layer.grouped_in.assign(D, nc_grouped_conv{});
        layer.grouped_skip.assign(D, nc_grouped_conv{});
        for (size_t d = 0; d < D; ++d) {
            plan_item in{&layer.grouped_in[d], {}, {}}, sk{&layer.grouped_skip[d], {}, {}};
            for (size_t g = 0; g < G; ++g) {
                const nc_res_block& blk = layer.by_kernel[g][d];
                in.convs.push_back(&blk.input_conv);
                in.acts.push_back(&blk.input_act);
                sk.convs.push_back(&blk.skip_conv);
                sk.acts.push_back(&blk.skip_act);
            }
            items.push_back(in);
            items.push_back(sk);
        }
        layer.grouped = true;
    }
    if (items.empty())
        return;
    ggml_init_params params = {ggml_tensor_overhead() * (items.size() * 4 + 1), nullptr, true};
    model.group_ctx = ggml_init(params);
    for (plan_item& it : items) {
        const int G = (int)it.convs.size();
        const int64_t cin_pad = it.convs[0]->w_packed->ne[0];
        const int64_t cout_pad = it.convs[0]->w_packed->ne[1];
        int64_t ksum = 0;
        for (int g = 0; g < G; ++g) {
            it.out->K[g] = (int)it.convs[g]->w_packed->ne[2];
            it.out->d[g] = it.convs[g]->dilation;
            ksum += it.out->K[g];
        }
        it.out->groups = G;
        it.out->cout = (int)it.convs[0]->w->ne[2];
        it.out->snake = (int)it.acts[0]->alpha->ne[1];
        it.out->w = ggml_new_tensor_3d(model.group_ctx, GGML_TYPE_F16, cin_pad, cout_pad, ksum);
        it.out->b = ggml_new_tensor_1d(model.group_ctx, GGML_TYPE_F32, (int64_t)G * it.out->cout);
        it.out->alpha =
            ggml_new_tensor_1d(model.group_ctx, GGML_TYPE_F32, (int64_t)G * it.out->snake);
        it.out->alpha_inv =
            ggml_new_tensor_1d(model.group_ctx, GGML_TYPE_F32, (int64_t)G * it.out->snake);
    }
    model.group_buffer = ggml_backend_alloc_ctx_tensors(model.group_ctx, model.backend);
    if (!model.group_buffer) {
        fprintf(
            stderr,
            "warning: could not allocate grouped NanoCodec conv tensors; using per-branch convs\n");
        ggml_free(model.group_ctx);
        model.group_ctx = nullptr;
        for (nc_res_layer& layer : model.res_layers) layer.grouped = false;
        return;
    }
    std::vector<ggml_fp16_t> wbuf;
    std::vector<float> fbuf;
    for (plan_item& it : items) {
        const int G = (int)it.convs.size();
        size_t off = 0;
        for (int g = 0; g < G; ++g) {
            const ggml_tensor* src = it.convs[g]->w_packed;
            const size_t n = (size_t)ggml_nelements(src);
            wbuf.resize(n);
            ggml_backend_tensor_get(src, wbuf.data(), 0, n * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(
                it.out->w, wbuf.data(), off * sizeof(ggml_fp16_t), n * sizeof(ggml_fp16_t));
            off += n;
        }
        auto concat_f32 = [&](ggml_tensor* dst, auto getter, int64_t per) {
            fbuf.resize((size_t)per);
            for (int g = 0; g < G; ++g) {
                const ggml_tensor* src = getter(g);
                if ((int64_t)ggml_nelements(src) != per) {
                    throw std::runtime_error("nanocodec grouped conv: inconsistent branch shapes");
                }
                ggml_backend_tensor_get(src, fbuf.data(), 0, (size_t)per * sizeof(float));
                ggml_backend_tensor_set(
                    dst, fbuf.data(), (size_t)g * per * sizeof(float), (size_t)per * sizeof(float));
            }
        };
        concat_f32(it.out->b, [&](int g) { return it.convs[g]->b; }, it.out->cout);
        concat_f32(it.out->alpha, [&](int g) { return it.acts[g]->alpha; }, it.out->snake);
        concat_f32(it.out->alpha_inv, [&](int g) { return it.acts[g]->alpha_inv; }, it.out->snake);
    }
    if (verbose) {
        fprintf(
            stderr, "nanocodec: grouped residual-branch convolutions enabled (%zu ops)\n",
            items.size());
    }
}

// Repack the stride-1 convolution weights ([K, Cin, Cout] F16, ne0 = K) into the fused CUDA
// conv layout [cout_pad, cin_pad, K] (ne0 = cout, both channel paddings multiples of 16, zero
// filled) when the backend supports ggml_conv1d_fused.
#if defined(NEMO_SPEECH_GGML_PATCHED)
// The fused kernel has per-shape limits (kernel size, dilated window, channel padding): ask the
// backend about every convolution with its real shape instead of one generic probe.
static bool
nc_fused_conv_supported(ggml_backend_t backend, const nc_conv& c) {
    const int64_t K = c.w->ne[0], cin = c.w->ne[1], cout = c.w->ne[2];
    const int64_t cin_pad = (cin + 15) / 16 * 16, cout_pad = (cout + 15) / 16 * 16;
    const int64_t left_pad = (K - 1) * c.dilation;
    ggml_init_params probe_params = {ggml_tensor_overhead() * 8, nullptr, true};
    ggml_context* probe = ggml_init(probe_params);
    if (!probe) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_3d(probe, GGML_TYPE_F32, 64, cin, 1);
    ggml_tensor* cache =
        left_pad > 0 ? ggml_new_tensor_3d(probe, GGML_TYPE_F32, left_pad, cin, 1) : nullptr;
    ggml_tensor* w = ggml_new_tensor_3d(probe, GGML_TYPE_F16, cin_pad, cout_pad, K);
    ggml_tensor* op = ggml_conv1d_fused(
        probe, x, cache, w, nullptr, nullptr, nullptr, (int)K, c.dilation, (int)cout, 0, 0.01f);
    const bool ok = ggml_backend_supports_op(backend, op);
    ggml_free(probe);
    return ok;
}
#endif

static void
nc_pack_fused_conv_weights(nc_model& model, bool verbose) {
#if !defined(NEMO_SPEECH_GGML_PATCHED)
    // ggml_conv1d_fused is provided by the patched ggml series only.
    (void)model;
    (void)verbose;
#else
    if (!model.backend) {
        return;
    }
    {
        // Probe backend support with a tiny op description.
        ggml_init_params probe_params = {ggml_tensor_overhead() * 8, nullptr, true};
        ggml_context* probe = ggml_init(probe_params);
        ggml_tensor* x = ggml_new_tensor_3d(probe, GGML_TYPE_F32, 64, 16, 1);
        ggml_tensor* w = ggml_new_tensor_3d(probe, GGML_TYPE_F16, 16, 16, 1);
        ggml_tensor* op =
            ggml_conv1d_fused(probe, x, nullptr, w, nullptr, nullptr, nullptr, 1, 1, 16, 0, 0.01f);
        const bool ok = ggml_backend_supports_op(model.backend, op);
        ggml_free(probe);
        if (!ok) {
            return;
        }
    }
    std::vector<nc_conv*> convs;
    convs.push_back(&model.pre_conv);
    for (nc_res_layer& layer : model.res_layers) {
        for (auto& stack : layer.by_kernel) {
            for (nc_res_block& block : stack) {
                convs.push_back(&block.input_conv);
                convs.push_back(&block.skip_conv);
            }
        }
    }
    convs.push_back(&model.post_conv);
    std::vector<nc_conv*> pending;
    for (nc_conv* c : convs) {
        if (c->w && c->w->type == GGML_TYPE_F16 && c->stride == 1 && ggml_is_contiguous(c->w) &&
            nc_fused_conv_supported(model.backend, *c)) {
            pending.push_back(c);
        }
    }
    if (pending.empty()) {
        return;
    }
    ggml_init_params params = {ggml_tensor_overhead() * (pending.size() + 1), nullptr, true};
    model.pack_ctx = ggml_init(params);
    std::vector<ggml_tensor*> packed;
    for (nc_conv* c : pending) {
        const int64_t K = c->w->ne[0], cin = c->w->ne[1], cout = c->w->ne[2];
        const int64_t cin_pad = (cin + 15) / 16 * 16, cout_pad = (cout + 15) / 16 * 16;
        ggml_tensor* dst = ggml_new_tensor_3d(model.pack_ctx, GGML_TYPE_F16, cin_pad, cout_pad, K);
        const std::string name = std::string(ggml_get_name(c->w)) + ".packed";
        ggml_set_name(dst, name.c_str());
        packed.push_back(dst);
    }
    model.pack_buffer = ggml_backend_alloc_ctx_tensors(model.pack_ctx, model.backend);
    if (!model.pack_buffer) {
        fprintf(
            stderr,
            "warning: could not allocate packed NanoCodec conv weights; using im2col "
            "convolutions\n");
        ggml_free(model.pack_ctx);
        model.pack_ctx = nullptr;
        return;
    }
    std::vector<ggml_fp16_t> src, dstv;
    for (size_t i = 0; i < pending.size(); ++i) {
        nc_conv* c = pending[i];
        const int64_t K = c->w->ne[0], cin = c->w->ne[1], cout = c->w->ne[2];
        const int64_t cin_pad = packed[i]->ne[0], cout_pad = packed[i]->ne[1];
        src.resize((size_t)ggml_nelements(c->w));
        ggml_backend_tensor_get(c->w, src.data(), 0, src.size() * sizeof(ggml_fp16_t));
        dstv.assign((size_t)(cout_pad * cin_pad * K), ggml_fp32_to_fp16(0.0f));
        for (int64_t co = 0; co < cout; ++co) {
            for (int64_t ci = 0; ci < cin; ++ci) {
                for (int64_t k = 0; k < K; ++k) {
                    dstv[(size_t)((k * cout_pad + co) * cin_pad + ci)] =
                        src[(size_t)((co * cin + ci) * K + k)];
                }
            }
        }
        ggml_backend_tensor_set(packed[i], dstv.data(), 0, dstv.size() * sizeof(ggml_fp16_t));
        c->w_packed = packed[i];
    }
    model.fused_conv = true;
    if (verbose) {
        fprintf(
            stderr, "nanocodec: fused tensor-core convolutions enabled for %zu layers\n",
            pending.size());
    }
    nc_pack_grouped_convs(model, verbose);
#endif
}

static bool
nc_model_load(
    const std::string& fname, nc_model& model, bool force_cpu = false, bool verbose = false) {
    const ggml_nvtx::range nvtx_range("nanocodec_model_load");
    nemo_speech::common::ensure_ggml_logging(verbose);

    gguf_init_params params = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&model.ctx,
    };
    model.gguf = gguf_init_from_file(fname.c_str(), params);
    if (!model.gguf || !model.ctx) {
        fprintf(stderr, "failed to load GGUF: %s\n", fname.c_str());
        return false;
    }

    nc_hparams& h = model.hparams;
    h.sample_rate = gguf_i32(model.gguf, "nano_codec.sample_rate", h.sample_rate);
    h.samples_per_frame = gguf_i32(model.gguf, "nano_codec.samples_per_frame", h.samples_per_frame);
    h.num_codebooks = gguf_i32(model.gguf, "nano_codec.num_codebooks", h.num_codebooks);
    h.codebook_size = gguf_i32(model.gguf, "nano_codec.codebook_size", h.codebook_size);
    h.latent_dim = gguf_i32(model.gguf, "nano_codec.latent_dim", h.latent_dim);
    h.group_dim = gguf_i32(model.gguf, "nano_codec.codebook_dim_per_group", h.group_dim);
    h.levels = gguf_i32_array(model.gguf, "nano_codec.quantizer.num_levels_per_group", h.levels);
    h.base = gguf_i32_array(model.gguf, "nano_codec.quantizer.dim_base_index", h.base);
    h.scale = gguf_i32_array(model.gguf, "nano_codec.quantizer.scale", h.scale);
    h.offset = gguf_i32_array(model.gguf, "nano_codec.quantizer.offset", h.offset);
    h.up_rates = gguf_i32_array(model.gguf, "nano_codec.decoder.up_sample_rates", h.up_rates);
    h.res_kernels =
        gguf_i32_array(model.gguf, "nano_codec.decoder.resblock_kernel_sizes", h.res_kernels);
    h.res_dilations =
        gguf_i32_array(model.gguf, "nano_codec.decoder.resblock_dilation_sizes", h.res_dilations);

    if (h.num_codebooks <= 0 || h.group_dim <= 0 || h.latent_dim != h.num_codebooks * h.group_dim) {
        fprintf(stderr, "invalid FSQ dimensions in GGUF metadata\n");
        return false;
    }

    ggml_backend_load_all();
    if (!force_cpu) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!model.backend || force_cpu) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
#if defined(GGML_USE_CUDA) && defined(NEMO_SPEECH_GGML_PATCHED)
    if (model.backend && ggml_backend_is_cuda(model.backend)) {
        // Above default-priority side streams (longform chunk prefetch) and below the
        // latency-critical Magpie decoder stream, so codec blocks are scheduled ahead of
        // background work and the codec cannot fall behind the producer.
        ggml_backend_cuda_set_stream_priority(model.backend, -2);
    }
#endif
    if (!model.backend) {
        fprintf(stderr, "failed to initialize ggml backend\n");
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(model.backend);
    if (verbose) {
        fprintf(
            stderr, "NanoCodec backend: %s%s%s%s\n", ggml_backend_name(model.backend),
            dev ? " - " : "", dev ? ggml_backend_dev_description(dev) : "",
            force_cpu ? " (forced CPU)" : "");
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer) {
        fprintf(
            stderr, "failed to allocate NanoCodec tensors on backend %s\n",
            ggml_backend_name(model.backend));
        return false;
    }

    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s for tensor loading\n", fname.c_str());
        return false;
    }
    std::vector<uint8_t> read_buf(16 * 1024 * 1024);
    const int n_tensors = gguf_get_n_tensors(model.gguf);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(model.gguf, i);
        ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);
        if (!tensor) {
            continue;
        }

        const size_t tensor_offset =
            gguf_get_data_offset(model.gguf) + gguf_get_tensor_offset(model.gguf, i);
        if (fseek(f, (long)tensor_offset, SEEK_SET) != 0) {
            fprintf(stderr, "failed to seek tensor %s\n", name);
            fclose(f);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += read_buf.size()) {
            const size_t ncopy = std::min(read_buf.size(), nbytes - pos);
            if (fread(read_buf.data(), 1, ncopy, f) != ncopy) {
                fprintf(stderr, "failed to read tensor %s\n", name);
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(tensor, read_buf.data(), pos, ncopy);
        }
    }
    fclose(f);

    model.pre_conv = load_conv(model, "dec.pre");

    model.activations.resize(h.up_rates.size());
    model.up_convs.resize(h.up_rates.size());
    model.res_layers.resize(h.up_rates.size());
    for (size_t i = 0; i < h.up_rates.size(); ++i) {
        model.activations[i] = load_activation(model, "dec.act." + std::to_string(i));
        model.up_convs[i] = load_conv(model, "dec.up." + std::to_string(i), h.up_rates[i], 1);
    }
    // Transposed convolutions consume F32 weights; convert once here instead of casting
    // ~15 MB of F16 weights inside every decode graph evaluation.
    {
        std::vector<size_t> pending;
        for (size_t i = 0; i < model.up_convs.size(); ++i) {
            if (model.up_convs[i].w && model.up_convs[i].w->type == GGML_TYPE_F16) {
                pending.push_back(i);
            }
        }
        if (!pending.empty()) {
            ggml_init_params aux_params = {
                /*.mem_size   =*/ggml_tensor_overhead() * (pending.size() + 1),
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };
            model.aux_ctx = ggml_init(aux_params);
            std::vector<ggml_tensor*> copies;
            for (size_t i : pending) {
                ggml_tensor* src = model.up_convs[i].w;
                ggml_tensor* dst =
                    ggml_new_tensor(model.aux_ctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
                const std::string name = std::string(ggml_get_name(src)) + ".f32";
                ggml_set_name(dst, name.c_str());
                copies.push_back(dst);
            }
            model.aux_buffer = ggml_backend_alloc_ctx_tensors(model.aux_ctx, model.backend);
            if (model.aux_buffer) {
                std::vector<ggml_fp16_t> packed;
                std::vector<float> values;
                for (size_t k = 0; k < pending.size(); ++k) {
                    ggml_tensor* src = model.up_convs[pending[k]].w;
                    const int64_t n = ggml_nelements(src);
                    packed.resize((size_t)n);
                    values.resize((size_t)n);
                    ggml_backend_tensor_get(
                        src, packed.data(), 0, packed.size() * sizeof(ggml_fp16_t));
                    ggml_fp16_to_fp32_row(packed.data(), values.data(), n);
                    ggml_backend_tensor_set(
                        copies[k], values.data(), 0, values.size() * sizeof(float));
                    model.up_convs[pending[k]].w_f32 = copies[k];
                }
            } else {
                fprintf(
                    stderr,
                    "warning: could not allocate F32 NanoCodec up-conv weights; using in-graph "
                    "casts\n");
                ggml_free(model.aux_ctx);
                model.aux_ctx = nullptr;
            }
        }
    }
    for (size_t i = 0; i < h.up_rates.size(); ++i) {
        nc_res_layer& layer = model.res_layers[i];
        layer.by_kernel.resize(h.res_kernels.size());
        for (size_t ik = 0; ik < h.res_kernels.size(); ++ik) {
            layer.by_kernel[ik].resize(h.res_dilations.size());
            for (size_t id = 0; id < h.res_dilations.size(); ++id) {
                const std::string p = "dec.res." + std::to_string(i) + "." + std::to_string(ik) +
                                      "." + std::to_string(id);
                nc_res_block& block = layer.by_kernel[ik][id];
                block.input_act = load_activation(model, p + ".ia");
                block.skip_act = load_activation(model, p + ".sa");
                block.input_conv = load_conv(model, p + ".ic", 1, h.res_dilations[id]);
                block.skip_conv = load_conv(model, p + ".sc", 1, 1);
            }
        }
    }

    model.post_activation = load_activation(model, "dec.post_act");
    model.post_conv = load_conv(model, "dec.post");
    nc_pack_fused_conv_weights(model, verbose);

    if (verbose) {
        fprintf(
            stderr,
            "loaded NanoCodec GGUF: sample_rate=%d codebooks=%d codebook_size=%d frame=%d "
            "samples\n",
            h.sample_rate, h.num_codebooks, h.codebook_size, h.samples_per_frame);
    }
    return true;
}

static void
nc_model_free(nc_model& model) {
    if (model.group_buffer) {
        ggml_backend_buffer_free(model.group_buffer);
        model.group_buffer = nullptr;
    }
    if (model.group_ctx) {
        ggml_free(model.group_ctx);
        model.group_ctx = nullptr;
    }
    if (model.pack_buffer) {
        ggml_backend_buffer_free(model.pack_buffer);
        model.pack_buffer = nullptr;
    }
    if (model.pack_ctx) {
        ggml_free(model.pack_ctx);
        model.pack_ctx = nullptr;
    }
    if (model.aux_buffer) {
        ggml_backend_buffer_free(model.aux_buffer);
        model.aux_buffer = nullptr;
    }
    if (model.aux_ctx) {
        ggml_free(model.aux_ctx);
        model.aux_ctx = nullptr;
    }
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
    if (model.gguf) {
        gguf_free(model.gguf);
        model.gguf = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
}

static ggml_context*
new_graph_context() {
    const size_t buf_size = ggml_tensor_overhead() * NANO_CODEC_MAX_NODES +
                            ggml_graph_overhead_custom(NANO_CODEC_MAX_NODES, false);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ggml_init(params);
}

static ggml_tensor*
causal_conv1d(ggml_context* ctx, ggml_tensor* x, const nc_conv& conv) {
    const int kernel = (int)conv.w->ne[0];
    const int left_pad = (kernel - 1) * conv.dilation;
    ggml_tensor* padded = ggml_pad_ext(ctx, x, left_pad, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor* y = ggml_conv_1d(ctx, conv.w, padded, conv.stride, 0, conv.dilation);
    y = ggml_add(ctx, y, conv.b);
    return y;
}

static ggml_tensor*
causal_conv_transpose1d(ggml_context* ctx, ggml_tensor* x, const nc_conv& conv) {
    const int64_t out_len = x->ne[0] * conv.stride;
    ggml_tensor* weight =
        conv.w_f32 ? conv.w_f32 : ggml_cont(ctx, ggml_cast(ctx, conv.w, GGML_TYPE_F32));
    ggml_tensor* full = ggml_conv_transpose_1d(ctx, weight, x, conv.stride, 0, 1);
    ggml_tensor* cropped =
        ggml_view_3d(ctx, full, out_len, weight->ne[1], 1, full->nb[1], full->nb[2], 0);
    return ggml_add(ctx, cropped, conv.b);
}

static ggml_tensor*
half_snake(ggml_context* ctx, ggml_tensor* x, const nc_activation& act) {
    const int64_t len = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t snake_channels = act.alpha->ne[1];
    if (snake_channels <= 0 || snake_channels > channels) {
        fprintf(
            stderr, "invalid half_snake channels: alpha=%lld x=%lld\n", (long long)snake_channels,
            (long long)channels);
        std::exit(1);
    }

    ggml_tensor* x_snake = ggml_view_3d(ctx, x, len, snake_channels, 1, x->nb[1], x->nb[2], 0);
    ggml_tensor* x_lrelu = ggml_view_3d(
        ctx, x, len, channels - snake_channels, 1, x->nb[1], x->nb[2], snake_channels * x->nb[1]);

    ggml_tensor* ax = ggml_mul(ctx, x_snake, act.alpha);
    ggml_tensor* periodic = ggml_sqr(ctx, ggml_sin(ctx, ax));
    periodic = ggml_mul(ctx, periodic, act.alpha_inv);
    ggml_tensor* snake_out = ggml_add(ctx, x_snake, periodic);
    ggml_tensor* lrelu_out = ggml_leaky_relu(ctx, x_lrelu, 0.01f, false);
    return ggml_concat(ctx, snake_out, lrelu_out, 1);
}

static ggml_tensor*
residual_block(ggml_context* ctx, ggml_tensor* x, const nc_res_block& block) {
    ggml_tensor* y = half_snake(ctx, x, block.input_act);
    y = causal_conv1d(ctx, y, block.input_conv);
    y = half_snake(ctx, y, block.skip_act);
    y = causal_conv1d(ctx, y, block.skip_conv);
    return ggml_add(ctx, x, y);
}

static ggml_tensor*
hifigan_resblock_stack(ggml_context* ctx, ggml_tensor* x, const std::vector<nc_res_block>& blocks) {
    ggml_tensor* y = x;
    for (const nc_res_block& block : blocks) {
        y = residual_block(ctx, y, block);
    }
    return y;
}

static ggml_tensor*
hifigan_reslayer(ggml_context* ctx, ggml_tensor* x, const nc_res_layer& layer) {
    ggml_tensor* sum = nullptr;
    for (const auto& stack : layer.by_kernel) {
        ggml_tensor* y = hifigan_resblock_stack(ctx, x, stack);
        sum = sum ? ggml_add(ctx, sum, y) : y;
    }
    return ggml_scale(ctx, sum, 1.0f / (float)layer.by_kernel.size());
}

static int
nc_conv_kernel(const nc_conv& conv) {
    return conv.w ? std::max<int64_t>(1, conv.w->ne[0]) : 1;
}

static int64_t
nc_decoder_left_context_samples(const nc_model& model) {
    const nc_hparams& h = model.hparams;
    int64_t samples_per_step = std::max<int32_t>(1, h.samples_per_frame);
    int64_t context = (int64_t)(nc_conv_kernel(model.pre_conv) - 1) *
                      std::max(1, model.pre_conv.dilation) * samples_per_step;

    for (size_t i = 0; i < h.up_rates.size(); ++i) {
        const int rate = std::max(1, i < h.up_rates.size() ? h.up_rates[i] : 1);
        samples_per_step = std::max<int64_t>(1, samples_per_step / rate);

        if (i < model.up_convs.size()) {
            const int up_kernel = nc_conv_kernel(model.up_convs[i]);
            context += (int64_t)std::max(0, up_kernel - rate) * samples_per_step;
        }

        int64_t layer_context = 0;
        if (i < model.res_layers.size()) {
            const nc_res_layer& layer = model.res_layers[i];
            for (const std::vector<nc_res_block>& stack : layer.by_kernel) {
                int64_t stack_context = 0;
                for (const nc_res_block& block : stack) {
                    stack_context += (int64_t)(nc_conv_kernel(block.input_conv) - 1) *
                                     std::max(1, block.input_conv.dilation) * samples_per_step;
                    stack_context += (int64_t)(nc_conv_kernel(block.skip_conv) - 1) *
                                     std::max(1, block.skip_conv.dilation) * samples_per_step;
                }
                layer_context = std::max(layer_context, stack_context);
            }
        }
        context += layer_context;
    }

    context +=
        (int64_t)(nc_conv_kernel(model.post_conv) - 1) * std::max(1, model.post_conv.dilation);
    return std::max<int64_t>(0, context);
}

static int
nc_decoder_left_context_frames(const nc_model& model) {
    const int frame = std::max<int32_t>(1, model.hparams.samples_per_frame);
    const int64_t samples = nc_decoder_left_context_samples(model);
    return (int)((samples + frame - 1) / frame);
}

enum nc_stream_cache_kind {
    NC_STREAM_CACHE_CONV = 0,
    NC_STREAM_CACHE_DECONV = 1,
};

// Streaming state (causal conv histories and transposed-conv overlap tails) lives in a
// device buffer owned by the state. Stream graphs read the state tensors directly and
// write the next-chunk values back with in-graph copies, so a decode call needs no
// host round trips for state (only the latent H2D and the audio D2H).
struct nc_stream_graph_io {
    std::vector<ggml_tensor*> writebacks;
};

static constexpr size_t NC_STREAM_MAX_STATE_TENSORS = 256;

struct nc_stream_state;

struct nc_stream_decode_graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor* latent = nullptr;
    ggml_tensor* audio = nullptr;
    nc_stream_graph_io io;
    int chunk_frames = 0;
    size_t output_samples = 0;
    const nc_stream_state* state = nullptr;  // state instance the graph's cache tensors belong to
    uint64_t state_generation = 0;
    size_t samples_per_frame = 0;
    std::vector<float> latent_data;
    std::vector<float> audio_data;
};

struct nc_stream_state {
    ggml_backend_t backend = nullptr;
    ggml_context* ctx = nullptr;             // no_alloc context owning the state tensors
    ggml_backend_buffer_t buffer = nullptr;  // device buffer backing all state tensors
    std::vector<ggml_tensor*> conv_caches;   // F32 [left_pad, channels, 1]
    std::vector<ggml_tensor*> deconv_tails;  // F32 [tail_len, channels, 1]
    size_t conv_pos = 0;
    size_t deconv_pos = 0;
    // Changes whenever the device buffer is (re)allocated or released: stream graphs bind the
    // state tensors at build time and use this to notice a different or reallocated state.
    uint64_t generation = 0;

    nc_stream_state() = default;
    nc_stream_state(const nc_stream_state&) = delete;
    nc_stream_state& operator=(const nc_stream_state&) = delete;
    ~nc_stream_state() { release(); }

    void begin_graph() {
        conv_pos = 0;
        deconv_pos = 0;
    }

    bool allocated() const { return buffer != nullptr; }

    // Zero the streaming history (start of a new utterance). Keeps the device buffer.
    void clear() {
        if (buffer) {
            ggml_backend_buffer_clear(buffer, 0);
        }
        begin_graph();
    }

    void release() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        conv_caches.clear();
        deconv_tails.clear();
        backend = nullptr;
        generation = 0;
        begin_graph();
    }

    ggml_context* tensor_ctx() {
        if (!ctx) {
            ggml_init_params params = {
                /*.mem_size   =*/ggml_tensor_overhead() * NC_STREAM_MAX_STATE_TENSORS,
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };
            ctx = ggml_init(params);
        }
        return ctx;
    }

    // Allocate the device buffer for all state tensors created so far and zero it.
    bool allocate(ggml_backend_t be) {
        if (buffer) {
            return true;
        }
        if (!ctx) {
            return true;  // no state tensors (nothing to allocate)
        }
        backend = be;
        buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buffer) {
            fprintf(stderr, "failed to allocate NanoCodec stream state buffer\n");
            return false;
        }
        ggml_backend_buffer_clear(buffer, 0);
        static std::atomic<uint64_t> next_generation{1};
        generation = next_generation.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
};

namespace nemo_speech::tts::nanocodec {

struct NanoCodecStreamState::Impl {
    nc_stream_state state;
};

struct NanoCodecStreamGraph::Impl {
    nc_stream_decode_graph graph;
};

}  // namespace nemo_speech::tts::nanocodec

static void
nc_stream_decode_graph_free(nc_stream_decode_graph& graph) {
    if (graph.allocr) {
        ggml_gallocr_free(graph.allocr);
        graph.allocr = nullptr;
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
        graph.ctx = nullptr;
    }
    graph.gf = nullptr;
    graph.latent = nullptr;
    graph.audio = nullptr;
    graph.io = {};
    graph.chunk_frames = 0;
    graph.output_samples = 0;
    graph.samples_per_frame = 0;
    graph.latent_data.clear();
    graph.audio_data.clear();
}

// Return the persistent device tensor for cache `index` of `kind`, creating it in the
// state's tensor context on first use. Shapes are independent of the chunk size, so any
// stream graph built for this state (one per chunk size) shares the same tensors.
static ggml_tensor*
nc_stream_cache_tensor(
    ggml_context* /*graph_ctx*/, nc_stream_state& state, nc_stream_graph_io& /*io*/,
    nc_stream_cache_kind kind, size_t index, int64_t len, int64_t channels) {
    std::vector<ggml_tensor*>& caches =
        kind == NC_STREAM_CACHE_CONV ? state.conv_caches : state.deconv_tails;
    if (index < caches.size() && caches[index]) {
        ggml_tensor* existing = caches[index];
        if (existing->ne[0] != len || existing->ne[1] != channels) {
            throw std::runtime_error("NanoCodec stream state shape mismatch between graphs");
        }
        return existing;
    }
    if (state.allocated()) {
        throw std::runtime_error("NanoCodec stream state cannot grow after allocation");
    }
    if (index >= caches.size()) {
        caches.resize(index + 1, nullptr);
    }
    ggml_tensor* tensor = ggml_new_tensor_3d(state.tensor_ctx(), GGML_TYPE_F32, len, channels, 1);
    const std::string name = std::string(
                                 kind == NC_STREAM_CACHE_CONV ? "nanocodec_stream_conv_cache_"
                                                              : "nanocodec_stream_deconv_tail_") +
                             std::to_string(index);
    ggml_set_name(tensor, name.c_str());
    caches[index] = tensor;
    return tensor;
}

// Schedule an in-graph copy of `next` (a possibly strided view of this chunk's data)
// into the persistent state tensor `cache` for the next chunk.
static void
nc_stream_add_cache_writeback(
    ggml_context* ctx, nc_stream_graph_io& io, ggml_tensor* next, ggml_tensor* cache) {
    ggml_tensor* writeback = ggml_cpy(ctx, next, cache);
    io.writebacks.push_back(writeback);
}

static ggml_tensor*
nc_stream_causal_conv1d(
    ggml_context* ctx, ggml_tensor* x, const nc_conv& conv, nc_stream_state& state,
    nc_stream_graph_io& io) {
    const int kernel = (int)conv.w->ne[0];
    const int left_pad = (kernel - 1) * conv.dilation;
    ggml_tensor* conv_in = x;

    if (left_pad > 0) {
        const size_t cache_index = state.conv_pos++;
        ggml_tensor* cache = nc_stream_cache_tensor(
            ctx, state, io, NC_STREAM_CACHE_CONV, cache_index, left_pad, x->ne[1]);
        conv_in = ggml_concat(ctx, cache, x, 0);

        const int64_t tail_start = conv_in->ne[0] - left_pad;
        ggml_tensor* tail = ggml_view_3d(
            ctx, conv_in, left_pad, x->ne[1], 1, conv_in->nb[1], conv_in->nb[2],
            (size_t)tail_start * conv_in->nb[0]);
        nc_stream_add_cache_writeback(ctx, io, tail, cache);
    }

    ggml_tensor* y = ggml_conv_1d(ctx, conv.w, conv_in, conv.stride, 0, conv.dilation);
    y = ggml_add(ctx, y, conv.b);
    return y;
}

// Fused path: optional half-snake activation + causal conv + bias in one CUDA op reading the
// streaming cache prefix directly. The cache holds *pre-activation* values (the kernel activates
// whatever it loads), so the writeback copies the raw tail of x (or, when the chunk is shorter
// than the receptive field, the shifted [cache | x] window).
static ggml_tensor*
nc_stream_conv1d_act(
    ggml_context* ctx, ggml_tensor* x, const nc_activation* act, const nc_conv& conv,
    nc_stream_state& state, nc_stream_graph_io& io) {
    if (!conv.w_packed) {
        ggml_tensor* y = act ? half_snake(ctx, x, *act) : x;
        return nc_stream_causal_conv1d(ctx, y, conv, state, io);
    }
#if !defined(NEMO_SPEECH_GGML_PATCHED)
    GGML_ABORT("nanocodec: packed conv weights require the patched ggml series");
#else
    const int kernel = (int)conv.w->ne[0];
    const int left_pad = (kernel - 1) * conv.dilation;
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    const int cout = (int)conv.w->ne[2];
    ggml_tensor* cache = nullptr;
    if (left_pad > 0) {
        const size_t cache_index = state.conv_pos++;
        cache =
            nc_stream_cache_tensor(ctx, state, io, NC_STREAM_CACHE_CONV, cache_index, left_pad, C);
    }
    const int snake_channels = act ? (int)act->alpha->ne[1] : 0;
    ggml_tensor* y = ggml_conv1d_fused(
        ctx, x, cache, conv.w_packed, conv.b, act ? act->alpha : nullptr,
        act ? act->alpha_inv : nullptr, kernel, conv.dilation, cout, snake_channels, 0.01f);
    if (cache) {
        ggml_tensor* next = nullptr;
        if (T >= left_pad) {
            next = ggml_view_3d(
                ctx, x, left_pad, C, 1, x->nb[1], x->nb[2], (size_t)(T - left_pad) * x->nb[0]);
        } else {
            ggml_tensor* kept = ggml_view_3d(
                ctx, cache, left_pad - T, C, 1, cache->nb[1], cache->nb[2],
                (size_t)T * cache->nb[0]);
            next = ggml_concat(ctx, kept, x, 0);
        }
        nc_stream_add_cache_writeback(ctx, io, next, cache);
    }
    return y;
#endif
}

static ggml_tensor*
nc_stream_causal_conv_transpose1d(
    ggml_context* ctx, ggml_tensor* x, const nc_conv& conv, nc_stream_state& state,
    nc_stream_graph_io& io) {
    const int64_t out_len = x->ne[0] * conv.stride;
    ggml_tensor* weight =
        conv.w_f32 ? conv.w_f32 : ggml_cont(ctx, ggml_cast(ctx, conv.w, GGML_TYPE_F32));
    ggml_tensor* full = ggml_conv_transpose_1d(ctx, weight, x, conv.stride, 0, 1);
    const int64_t tail_len = std::max<int64_t>(0, full->ne[0] - out_len);

    ggml_tensor* current = nullptr;
    if (tail_len > 0) {
        const size_t tail_index = state.deconv_pos++;
        ggml_tensor* prev_tail = nc_stream_cache_tensor(
            ctx, state, io, NC_STREAM_CACHE_DECONV, tail_index, tail_len, full->ne[1]);

        const int64_t add_len = std::min<int64_t>(tail_len, out_len);
        ggml_tensor* prefix =
            ggml_view_3d(ctx, full, add_len, full->ne[1], 1, full->nb[1], full->nb[2], 0);
        ggml_tensor* prev_prefix = prev_tail;
        if (add_len != tail_len) {
            prev_prefix = ggml_view_3d(
                ctx, prev_tail, add_len, prev_tail->ne[1], 1, prev_tail->nb[1], prev_tail->nb[2],
                0);
        }
        prefix = ggml_add(ctx, prefix, prev_prefix);

        if (out_len > add_len) {
            ggml_tensor* suffix = ggml_view_3d(
                ctx, full, out_len - add_len, full->ne[1], 1, full->nb[1], full->nb[2],
                (size_t)add_len * full->nb[0]);
            current = ggml_concat(ctx, prefix, suffix, 0);
        } else {
            current = prefix;
        }

        ggml_tensor* next_tail = ggml_view_3d(
            ctx, full, tail_len, full->ne[1], 1, full->nb[1], full->nb[2],
            (size_t)out_len * full->nb[0]);
        nc_stream_add_cache_writeback(ctx, io, next_tail, prev_tail);
    } else {
        current = ggml_view_3d(ctx, full, out_len, conv.w->ne[1], 1, full->nb[1], full->nb[2], 0);
    }

    return ggml_add(ctx, current, conv.b);
}

// Grouped fused conv over the branches of a residual stage (see nc_grouped_conv). One streaming
// cache of the largest receptive field serves every branch; the kernel reads each branch's suffix.
static ggml_tensor*
nc_stream_grouped_conv(
    ggml_context* ctx, ggml_tensor* x, bool shared_input, const nc_grouped_conv& gc,
    ggml_tensor* residual, bool residual_shared, nc_stream_state& state, nc_stream_graph_io& io) {
#if !defined(NEMO_SPEECH_GGML_PATCHED)
    (void)ctx, (void)x, (void)shared_input, (void)gc, (void)residual, (void)residual_shared;
    (void)state, (void)io;
    GGML_ABORT("nanocodec: grouped convs require the patched ggml series");
#else
    int max_pad = 0;
    for (int g = 0; g < gc.groups; ++g) max_pad = std::max(max_pad, (gc.K[g] - 1) * gc.d[g]);
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor* cache = nullptr;
    if (max_pad > 0) {
        const size_t cache_index = state.conv_pos++;
        cache =
            nc_stream_cache_tensor(ctx, state, io, NC_STREAM_CACHE_CONV, cache_index, max_pad, C);
    }
    ggml_tensor* y = ggml_conv1d_fused_grouped(
        ctx, x, cache, gc.w, gc.b, gc.alpha, gc.alpha_inv, residual, gc.groups, gc.K, gc.d, gc.cout,
        gc.snake, 0.01f, shared_input, residual_shared);
    if (cache) {
        ggml_tensor* next = nullptr;
        if (T >= max_pad) {
            next = ggml_view_3d(
                ctx, x, max_pad, C, 1, x->nb[1], x->nb[2], (size_t)(T - max_pad) * x->nb[0]);
        } else {
            ggml_tensor* kept = ggml_view_3d(
                ctx, cache, max_pad - T, C, 1, cache->nb[1], cache->nb[2],
                (size_t)T * cache->nb[0]);
            next = ggml_concat(ctx, kept, x, 0);
        }
        nc_stream_add_cache_writeback(ctx, io, next, cache);
    }
    return y;
#endif
}

static ggml_tensor*
nc_stream_residual_block(
    ggml_context* ctx, ggml_tensor* x, const nc_res_block& block, nc_stream_state& state,
    nc_stream_graph_io& io) {
    ggml_tensor* y = nc_stream_conv1d_act(ctx, x, &block.input_act, block.input_conv, state, io);
    y = nc_stream_conv1d_act(ctx, y, &block.skip_act, block.skip_conv, state, io);
    return ggml_add(ctx, x, y);
}

static ggml_tensor*
nc_stream_hifigan_resblock_stack(
    ggml_context* ctx, ggml_tensor* x, const std::vector<nc_res_block>& blocks,
    nc_stream_state& state, nc_stream_graph_io& io) {
    ggml_tensor* y = x;
    for (const nc_res_block& block : blocks) {
        y = nc_stream_residual_block(ctx, y, block, state, io);
    }
    return y;
}

static ggml_tensor*
nc_stream_hifigan_reslayer(
    ggml_context* ctx, ggml_tensor* x, const nc_res_layer& layer, nc_stream_state& state,
    nc_stream_graph_io& io) {
    if (layer.grouped) {
        // y_g <- x; for each block d: y_g <- y_g + skip_g(act(input_g(act(y_g)))); out = mean_g y_g
        const int64_t T = x->ne[0];
        const int64_t C = x->ne[1];
        ggml_tensor* y3 = nullptr;
        for (size_t d = 0; d < layer.grouped_in.size(); ++d) {
            ggml_tensor* a = nc_stream_grouped_conv(
                ctx, d == 0 ? x : y3, /*shared_input=*/d == 0, layer.grouped_in[d], nullptr, false,
                state, io);
            y3 = nc_stream_grouped_conv(
                ctx, a, false, layer.grouped_skip[d], d == 0 ? x : y3, /*residual_shared=*/d == 0,
                state, io);
        }
        const int G = layer.grouped_in[0].groups;
        ggml_tensor* sum = nullptr;
        for (int g = 0; g < G; ++g) {
            ggml_tensor* part =
                ggml_view_3d(ctx, y3, T, C, 1, y3->nb[1], y3->nb[2], (size_t)g * C * y3->nb[1]);
            sum = sum ? ggml_add(ctx, sum, part) : part;
        }
        return ggml_scale(ctx, sum, 1.0f / (float)G);
    }
    ggml_tensor* sum = nullptr;
    for (const auto& stack : layer.by_kernel) {
        ggml_tensor* y = nc_stream_hifigan_resblock_stack(ctx, x, stack, state, io);
        sum = sum ? ggml_add(ctx, sum, y) : y;
    }
    return ggml_scale(ctx, sum, 1.0f / (float)layer.by_kernel.size());
}

static void
dequantize_tokens_into(
    const nc_hparams& h, const std::vector<std::array<int32_t, 8>>& frames, int latent_frames,
    std::vector<float>& latent) {
    const int n_frames = (int)frames.size();
    if (latent_frames < n_frames) {
        throw std::runtime_error("latent frame buffer is smaller than codec frame count");
    }
    latent.assign((size_t)latent_frames * h.latent_dim, 0.0f);

    for (int t = 0; t < n_frames; ++t) {
        for (int g = 0; g < h.num_codebooks; ++g) {
            const int32_t token = frames[t][g];
            if (token < 0 || token >= h.codebook_size) {
                throw std::runtime_error("codec token out of range");
            }
            for (int d = 0; d < h.group_dim; ++d) {
                const int32_t nonnegative = (token / h.base[d]) % h.levels[d];
                const float value = ((float)nonnegative - (float)h.offset[d]) / (float)h.scale[d];
                const int channel = g * h.group_dim + d;
                latent[(size_t)channel * latent_frames + t] = value;
            }
        }
    }
}

static void
dequantize_tokens(
    const nc_hparams& h, const std::vector<std::array<int32_t, 8>>& frames,
    std::vector<float>& latent) {
    const ggml_nvtx::range nvtx_range("nanocodec_dequantize_tokens");
    dequantize_tokens_into(h, frames, (int)frames.size(), latent);
}

static void
dequantize_tokens_padded(
    const nc_hparams& h, const std::vector<std::array<int32_t, 8>>& frames, int latent_frames,
    std::vector<float>& latent) {
    const ggml_nvtx::range nvtx_range("nanocodec_dequantize_tokens_padded");
    dequantize_tokens_into(h, frames, latent_frames, latent);
}

static bool
decode_eval(
    const nc_model& model, const std::vector<std::array<int32_t, 8>>& frames, int threads,
    std::vector<float>& audio) {
    const ggml_nvtx::range nvtx_range("nanocodec_decode_eval");
    const nc_hparams& h = model.hparams;
    std::vector<float> latent;
    try {
        dequantize_tokens(h, frames, latent);
    }
    catch (const std::exception& e) {
        fprintf(stderr, "failed to dequantize tokens: %s\n", e.what());
        return false;
    }

    ggml_context* ctx = nullptr;
    {
        const ggml_nvtx::range nvtx_build("nanocodec_new_graph_context");
        ctx = new_graph_context();
    }
    if (!ctx) {
        fprintf(stderr, "failed to allocate graph context\n");
        return false;
    }

    const int n_frames = (int)frames.size();
    ggml_tensor* inp = nullptr;
    ggml_tensor* x = nullptr;
    ggml_cgraph* gf = nullptr;
    {
        const ggml_nvtx::range nvtx_build("nanocodec_build_decoder_graph");
        inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_frames, h.latent_dim, 1);
        ggml_set_name(inp, "nanocodec_latent");

        x = causal_conv1d(ctx, inp, model.pre_conv);
        for (size_t i = 0; i < h.up_rates.size(); ++i) {
            x = half_snake(ctx, x, model.activations[i]);
            x = causal_conv_transpose1d(ctx, x, model.up_convs[i]);
            x = hifigan_reslayer(ctx, x, model.res_layers[i]);
        }

        x = half_snake(ctx, x, model.post_activation);
        x = causal_conv1d(ctx, x, model.post_conv);
        x = ggml_clamp(ctx, x, -1.0f, 1.0f);
        ggml_set_name(x, "nanocodec_audio");

        gf = ggml_new_graph_custom(ctx, NANO_CODEC_MAX_NODES, false);
        ggml_build_forward_expand(gf, x);
        tag_graph_first_node(gf);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        fprintf(stderr, "failed to create graph allocator\n");
        ggml_free(ctx);
        return false;
    }
    {
        const ggml_nvtx::range nvtx_alloc("nanocodec_graph_alloc");
        ggml_gallocr_alloc_graph(allocr, gf);
    }
    {
        const ggml_nvtx::range nvtx_inputs("nanocodec_graph_set_inputs");
        ggml_backend_tensor_set(inp, latent.data(), 0, latent.size() * sizeof(float));
    }

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, threads);
    }

    ggml_status status = GGML_STATUS_FAILED;
    {
        const ggml_nvtx::range nvtx_compute("nanocodec_graph_compute");
        status = ggml_backend_graph_compute(model.backend, gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    {
        const ggml_nvtx::range nvtx_output("nanocodec_graph_get_audio");
        audio.resize((size_t)ggml_nelements(x));
        ggml_backend_tensor_get(x, audio.data(), 0, audio.size() * sizeof(float));
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

static bool
nc_stream_decode_graph_init(
    const nc_model& model, nc_stream_state& state, int chunk_frames,
    nc_stream_decode_graph& graph) {
    const ggml_nvtx::range nvtx_range("nanocodec_stream_init_persistent_graph");
    nc_stream_decode_graph_free(graph);

    if (chunk_frames <= 0) {
        fprintf(stderr, "chunk_frames must be positive\n");
        return false;
    }

    const nc_hparams& h = model.hparams;
    graph.chunk_frames = chunk_frames;
    graph.ctx = new_graph_context();
    if (!graph.ctx) {
        fprintf(stderr, "failed to allocate persistent stream graph context\n");
        return false;
    }

    state.begin_graph();

    {
        const ggml_nvtx::range nvtx_build("nanocodec_stream_build_persistent_decoder_graph");
        graph.latent = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, chunk_frames, h.latent_dim, 1);
        ggml_set_name(graph.latent, "nanocodec_stream_latent");
        ggml_set_input(graph.latent);

        ggml_tensor* x =
            nc_stream_conv1d_act(graph.ctx, graph.latent, nullptr, model.pre_conv, state, graph.io);
        for (size_t i = 0; i < h.up_rates.size(); ++i) {
            x = half_snake(graph.ctx, x, model.activations[i]);
            x = nc_stream_causal_conv_transpose1d(graph.ctx, x, model.up_convs[i], state, graph.io);
            x = nc_stream_hifigan_reslayer(graph.ctx, x, model.res_layers[i], state, graph.io);
        }

        x = nc_stream_conv1d_act(
            graph.ctx, x, &model.post_activation, model.post_conv, state, graph.io);
        x = ggml_clamp(graph.ctx, x, -1.0f, 1.0f);
        ggml_set_name(x, "nanocodec_stream_audio");
        ggml_set_output(x);
        graph.audio = x;

        graph.gf = ggml_new_graph_custom(graph.ctx, NANO_CODEC_MAX_NODES, false);
        // Expand the audio path first so every reader of a state tensor is ordered before
        // the write-back that overwrites it for the next chunk.
        ggml_build_forward_expand(graph.gf, graph.audio);
        for (ggml_tensor* writeback : graph.io.writebacks) {
            ggml_build_forward_expand(graph.gf, writeback);
        }
        tag_graph_first_node(graph.gf);
    }

    // The state tensors must be resident before the graph allocator runs so that it
    // treats them as externally allocated (like weights) rather than graph temporaries.
    if (!state.allocate(model.backend)) {
        nc_stream_decode_graph_free(graph);
        return false;
    }

    graph.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!graph.allocr) {
        fprintf(stderr, "failed to create persistent stream graph allocator\n");
        nc_stream_decode_graph_free(graph);
        return false;
    }

    {
        const ggml_nvtx::range nvtx_alloc("nanocodec_stream_persistent_graph_alloc");
        if (!ggml_gallocr_alloc_graph(graph.allocr, graph.gf)) {
            fprintf(stderr, "failed to allocate persistent stream graph tensors\n");
            nc_stream_decode_graph_free(graph);
            return false;
        }
    }

    graph.output_samples = (size_t)ggml_nelements(graph.audio);
    if (graph.output_samples == 0 || graph.output_samples % (size_t)chunk_frames != 0) {
        fprintf(
            stderr, "unexpected persistent stream audio length: %zu samples for %d frames\n",
            graph.output_samples, chunk_frames);
        nc_stream_decode_graph_free(graph);
        return false;
    }

    graph.samples_per_frame = graph.output_samples / (size_t)chunk_frames;
    graph.latent_data.assign((size_t)chunk_frames * h.latent_dim, 0.0f);
    graph.audio_data.resize(graph.output_samples);
    graph.state = &state;
    graph.state_generation = state.generation;
    return true;
}

static bool
decode_eval_stream(
    const nc_model& model, nc_stream_state& state, nc_stream_decode_graph& graph,
    const std::vector<std::array<int32_t, 8>>& frames, int threads, std::vector<float>& audio) {
    const ggml_nvtx::range nvtx_range("nanocodec_decode_eval_stream");
    if (frames.empty()) {
        audio.clear();
        return true;
    }

    if (!graph.ctx || !graph.gf || !graph.allocr || !graph.latent || !graph.audio ||
        graph.chunk_frames <= 0) {
        fprintf(stderr, "persistent stream graph is not initialized\n");
        return false;
    }
    if ((int)frames.size() > graph.chunk_frames) {
        fprintf(
            stderr, "stream chunk has %zu frames, larger than fixed graph chunk_frames=%d\n",
            frames.size(), graph.chunk_frames);
        return false;
    }

    try {
        dequantize_tokens_padded(model.hparams, frames, graph.chunk_frames, graph.latent_data);
    }
    catch (const std::exception& e) {
        fprintf(stderr, "failed to dequantize tokens: %s\n", e.what());
        return false;
    }

    {
        const ggml_nvtx::range nvtx_inputs("nanocodec_stream_graph_set_inputs");
        ggml_backend_tensor_set(
            graph.latent, graph.latent_data.data(), 0, graph.latent_data.size() * sizeof(float));
    }
    (void)state;  // streaming history is device-resident and updated inside the graph

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, threads);
    }

    ggml_status status = GGML_STATUS_FAILED;
    {
        const ggml_nvtx::range nvtx_compute("nanocodec_stream_graph_compute");
        status = ggml_backend_graph_compute(model.backend, graph.gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml stream graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    {
        const ggml_nvtx::range nvtx_outputs("nanocodec_stream_graph_get_outputs");
        ggml_backend_tensor_get(
            graph.audio, graph.audio_data.data(), 0, graph.audio_data.size() * sizeof(float));
    }

    const size_t keep_samples =
        std::min(graph.audio_data.size(), frames.size() * graph.samples_per_frame);
    audio.assign(graph.audio_data.begin(), graph.audio_data.begin() + (ptrdiff_t)keep_samples);
    return true;
}

static bool
decode_eval_stream_all(
    const nc_model& model, const std::vector<std::array<int32_t, 8>>& frames, int chunk_frames,
    int threads, std::vector<float>& audio) {
    const ggml_nvtx::range nvtx_range("nanocodec_decode_eval_stream_all");
    if (chunk_frames <= 0) {
        fprintf(stderr, "chunk_frames must be positive\n");
        return false;
    }

    nc_stream_state state;
    state.clear();
    nc_stream_decode_graph graph;
    audio.clear();

    if (frames.empty()) {
        return true;
    }

    if (!nc_stream_decode_graph_init(model, state, chunk_frames, graph)) {
        return false;
    }

    for (size_t start = 0, chunk_index = 0; start < frames.size();
         start += (size_t)chunk_frames, ++chunk_index) {
        const size_t end = std::min(frames.size(), start + (size_t)chunk_frames);
        std::vector<std::array<int32_t, 8>> chunk(
            frames.begin() + (ptrdiff_t)start, frames.begin() + (ptrdiff_t)end);

        std::vector<float> chunk_audio;
        const int64_t t_start = ggml_time_us();
        if (!decode_eval_stream(model, state, graph, chunk, threads, chunk_audio)) {
            nc_stream_decode_graph_free(graph);
            return false;
        }
        const double elapsed_ms = (ggml_time_us() - t_start) / 1000.0;
        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());

        fprintf(
            stderr, "streamed codec chunk %zu: %zu frames -> %zu decoded samples in %.2f ms%s\n",
            chunk_index, chunk.size(), chunk_audio.size(), elapsed_ms,
            end == frames.size() ? " (final)" : "");
    }

    nc_stream_decode_graph_free(graph);
    return true;
}

namespace nemo_speech::tts::nanocodec {

namespace {

const NanoCodecHParams&
empty_hparams() {
    static const NanoCodecHParams hparams;
    return hparams;
}

bool
require_loaded(const NanoCodecModel* model) {
    if (!model || !model->loaded()) {
        fprintf(stderr, "NanoCodec model is not loaded\n");
        return false;
    }
    return true;
}

}  // namespace

NanoCodecModel::NanoCodecModel() : impl_(std::make_unique<Impl>()) {}

NanoCodecModel::~NanoCodecModel() {
    reset();
}

NanoCodecModel::NanoCodecModel(NanoCodecModel&& other) noexcept = default;

NanoCodecModel&
NanoCodecModel::operator=(NanoCodecModel&& other) noexcept {
    if (this != &other) {
        reset();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

bool
NanoCodecModel::load(const std::string& path, bool force_cpu, bool verbose) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    reset();
    impl_->loaded = nc_model_load(path, impl_->model, force_cpu, verbose);
    if (!impl_->loaded) {
        reset();
    }
    return impl_->loaded;
}

void
NanoCodecModel::reset() {
    if (!impl_) {
        return;
    }
    nc_model_free(impl_->model);
    impl_->model = nc_model{};
    impl_->loaded = false;
}

bool
NanoCodecModel::loaded() const {
    return impl_ && impl_->loaded && impl_->model.ctx && impl_->model.backend;
}

bool
NanoCodecModel::onAccelerator() const {
    return loaded() && !ggml_backend_is_cpu(impl_->model.backend);
}

const NanoCodecHParams&
NanoCodecModel::hparams() const {
    return impl_ ? impl_->model.hparams : empty_hparams();
}

int
NanoCodecModel::sampleRate() const {
    return hparams().sample_rate;
}

int
NanoCodecModel::samplesPerFrame() const {
    return hparams().samples_per_frame;
}

int
NanoCodecModel::numCodebooks() const {
    return hparams().num_codebooks;
}

int
NanoCodecModel::codebookSize() const {
    return hparams().codebook_size;
}

int
NanoCodecModel::decoderLeftContextFrames() const {
    return require_loaded(this) ? nc_decoder_left_context_frames(impl_->model) : 0;
}

int64_t
NanoCodecModel::decoderLeftContextSamples() const {
    return require_loaded(this) ? nc_decoder_left_context_samples(impl_->model) : 0;
}

NanoCodecStreamState::NanoCodecStreamState() : impl_(std::make_unique<Impl>()) {
    clear();
}

NanoCodecStreamState::~NanoCodecStreamState() = default;

NanoCodecStreamState::NanoCodecStreamState(NanoCodecStreamState&& other) noexcept = default;

NanoCodecStreamState&
NanoCodecStreamState::operator=(NanoCodecStreamState&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void
NanoCodecStreamState::clear() {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->state.clear();
}

NanoCodecStreamGraph::NanoCodecStreamGraph() : impl_(std::make_unique<Impl>()) {}

NanoCodecStreamGraph::~NanoCodecStreamGraph() {
    reset();
}

NanoCodecStreamGraph::NanoCodecStreamGraph(NanoCodecStreamGraph&& other) noexcept = default;

NanoCodecStreamGraph&
NanoCodecStreamGraph::operator=(NanoCodecStreamGraph&& other) noexcept {
    if (this != &other) {
        reset();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void
NanoCodecStreamGraph::reset() {
    if (impl_) {
        nc_stream_decode_graph_free(impl_->graph);
    }
}

bool
NanoCodecStreamGraph::initialized() const {
    return impl_ && impl_->graph.ctx && impl_->graph.gf && impl_->graph.allocr &&
           impl_->graph.latent && impl_->graph.audio && impl_->graph.chunk_frames > 0;
}

int
NanoCodecStreamGraph::chunkFrames() const {
    return impl_ ? impl_->graph.chunk_frames : 0;
}

NanoCodecDecoder::NanoCodecDecoder(const NanoCodecModel& model) : model_(&model) {}

bool
NanoCodecDecoder::decode(
    const NanoCodecFrames& frames, int threads, std::vector<float>& audio) const {
    if (!require_loaded(model_)) {
        return false;
    }
    return decode_eval(model_->impl_->model, frames, threads, audio);
}

bool
NanoCodecDecoder::initStreamGraph(
    NanoCodecStreamState& state, int chunk_frames, NanoCodecStreamGraph& graph) const {
    if (!require_loaded(model_)) {
        return false;
    }
    if (!state.impl_) {
        state.impl_ = std::make_unique<NanoCodecStreamState::Impl>();
    }
    if (!graph.impl_) {
        graph.impl_ = std::make_unique<NanoCodecStreamGraph::Impl>();
    }
    return nc_stream_decode_graph_init(
        model_->impl_->model, state.impl_->state, chunk_frames, graph.impl_->graph);
}

bool
NanoCodecDecoder::decodeStream(
    NanoCodecStreamState& state, NanoCodecStreamGraph& graph, const NanoCodecFrames& frames,
    int threads, std::vector<float>& audio) const {
    if (!require_loaded(model_) || !state.impl_ || !graph.impl_) {
        return false;
    }
    nc_stream_decode_graph& g = graph.impl_->graph;
    const nc_stream_state& s = state.impl_->state;
    if (g.state != &s || g.state_generation != s.generation) {
        // The graph was built for another (or since reallocated) state: rebind it.
        if (g.chunk_frames <= 0 ||
            !nc_stream_decode_graph_init(
                model_->impl_->model, state.impl_->state, g.chunk_frames, g)) {
            return false;
        }
    }
    return decode_eval_stream(
        model_->impl_->model, state.impl_->state, graph.impl_->graph, frames, threads, audio);
}

bool
NanoCodecDecoder::decodeStreamAll(
    const NanoCodecFrames& frames, int chunk_frames, int threads, std::vector<float>& audio) const {
    if (!require_loaded(model_)) {
        return false;
    }
    return decode_eval_stream_all(model_->impl_->model, frames, chunk_frames, threads, audio);
}

}  // namespace nemo_speech::tts::nanocodec
