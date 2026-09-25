// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "model_logging.h"
#include "nvtx_utils.h"

static constexpr int NANO_CODEC_MAX_NODES = 32768;
// Upper bound on persistent layer caches: one per causal conv plus one tail per upsampler.
static constexpr int NANO_CODEC_MAX_CACHES = 512;

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
    int stride = 1;
    int dilation = 1;
};

struct nc_res_block {
    nc_activation input_act;
    nc_activation skip_act;
    nc_conv input_conv;
    nc_conv skip_conv;
};

struct nc_res_layer {
    std::vector<std::vector<nc_res_block>> by_kernel;
};

struct nc_model {
    nc_hparams hparams;

    gguf_context* gguf = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // Holds F32 copies of the upsampler kernels on backends that need them; empty otherwise.
    ggml_context* aux_ctx = nullptr;
    ggml_backend_buffer_t aux_buffer = nullptr;

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

// Whether this backend can convolve with the stored F16 kernel without changing the result.
//
// The transposed convolutions are the one place the stored F16 weights meet an op whose
// backend behaviour varies. CUDA and Metal widen each kernel element to F32 and accumulate
// in F32, so an F16 kernel is bit-identical to an F32 copy of it. Vulkan rejects a
// non-F32 kernel outright, and ggml's CPU path pairs an F16 kernel with an F16 rounding of
// the F32 activations, which loses precision the old F32 copy kept. Those two get a
// converted copy instead -- made once here rather than on every graph evaluation.
static bool
nc_backend_takes_f16_deconv(ggml_backend_t backend, const nc_conv& conv) {
    if (!backend || !conv.w || ggml_backend_is_cpu(backend)) {
        return false;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, conv.w->ne[2], 1);
    ggml_tensor* probe = ggml_conv_transpose_1d(ctx, conv.w, input, conv.stride, 0, 1);
    const bool supported = probe && ggml_backend_supports_op(backend, probe);
    ggml_free(ctx);
    return supported;
}

static bool
nc_prepare_deconv_weights(nc_model& model, bool verbose) {
    if (model.up_convs.empty()) {
        return true;
    }
    if (nc_backend_takes_f16_deconv(model.backend, model.up_convs.front())) {
        if (verbose) {
            fprintf(stderr, "NanoCodec upsampler kernels used as stored (F16)\n");
        }
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * (model.up_convs.size() + 2),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model.aux_ctx = ggml_init(params);
    if (!model.aux_ctx) {
        fprintf(stderr, "failed to allocate NanoCodec auxiliary weight context\n");
        return false;
    }

    std::vector<ggml_tensor*> converted(model.up_convs.size(), nullptr);
    size_t pending = 0;
    for (size_t i = 0; i < model.up_convs.size(); ++i) {
        const ggml_tensor* src = model.up_convs[i].w;
        if (src->type == GGML_TYPE_F32) {
            continue;
        }
        // Reject an unsupported type before reserving anything for it.
        if (src->type != GGML_TYPE_F16) {
            fprintf(stderr, "unsupported upsampler kernel type %s\n", ggml_type_name(src->type));
            return false;
        }
        converted[i] =
            ggml_new_tensor_3d(model.aux_ctx, GGML_TYPE_F32, src->ne[0], src->ne[1], src->ne[2]);
        ggml_set_name(converted[i], (std::string(ggml_get_name(src)) + ".f32").c_str());
        ++pending;
    }

    // A backend that will not take F16 kernels but whose model already stores them as F32
    // leaves nothing to convert. ggml_backend_alloc_ctx_tensors returns NULL for an empty
    // context, which is success here, not a failure to allocate.
    if (pending == 0) {
        ggml_free(model.aux_ctx);
        model.aux_ctx = nullptr;
        if (verbose) {
            fprintf(stderr, "NanoCodec upsampler kernels already stored as F32\n");
        }
        return true;
    }

    model.aux_buffer = ggml_backend_alloc_ctx_tensors(model.aux_ctx, model.backend);
    if (!model.aux_buffer) {
        fprintf(stderr, "failed to allocate NanoCodec F32 upsampler kernels\n");
        return false;
    }

    std::vector<ggml_fp16_t> src_half;
    std::vector<float> dst_full;
    for (size_t i = 0; i < converted.size(); ++i) {
        if (!converted[i]) {
            continue;
        }
        ggml_tensor* src = model.up_convs[i].w;
        const size_t n = (size_t)ggml_nelements(src);
        src_half.resize(n);
        dst_full.resize(n);
        ggml_backend_tensor_get(src, src_half.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(src_half.data(), dst_full.data(), (int64_t)n);
        ggml_backend_tensor_set(converted[i], dst_full.data(), 0, n * sizeof(float));
        model.up_convs[i].w = converted[i];
    }

    if (verbose) {
        fprintf(
            stderr, "NanoCodec upsampler kernels converted to F32 (%.1f MiB) for backend %s\n",
            (double)ggml_backend_buffer_get_size(model.aux_buffer) / (1024.0 * 1024.0),
            ggml_backend_name(model.backend));
    }
    return true;
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

    if (!nc_prepare_deconv_weights(model, verbose)) {
        return false;
    }

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
    // The kernel is already in a type this backend convolves with directly; any conversion
    // the backend needed happened once at load time.
    ggml_tensor* full = ggml_conv_transpose_1d(ctx, conv.w, x, conv.stride, 0, 1);
    ggml_tensor* cropped =
        ggml_view_3d(ctx, full, out_len, conv.w->ne[1], 1, full->nb[1], full->nb[2], 0);
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

// One refresh of a layer cache. `barrier` is the last graph node that reads the cache;
// expanding it before `write` keeps the copy ordered after every read of the old value.
struct nc_stream_cache_write {
    ggml_tensor* barrier = nullptr;
    ggml_tensor* write = nullptr;
};

struct nc_stream_graph_io {
    std::vector<nc_stream_cache_write> cache_writes;
};

// Defined below; the graph only records which state owns its cache tensors.
struct nc_stream_state;

struct nc_stream_decode_graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor* latent = nullptr;
    ggml_tensor* audio = nullptr;
    nc_stream_graph_io io;
    // The state whose cache tensors these nodes point at. A different state with the same
    // cache count would otherwise pass the check below and read freed tensors.
    const nc_stream_state* owner = nullptr;
    int chunk_frames = 0;
    size_t output_samples = 0;
    size_t samples_per_frame = 0;
    std::vector<float> latent_data;
    std::vector<float> audio_data;
};

// Layer state lives in backend memory for the lifetime of the stream. The decode graph
// reads each cache in place and writes the next value back with ggml_cpy, so decoding a
// chunk never round-trips conv state through the host.
//
// The tensors are created while the graph is built and the graph nodes point straight at
// them, so a state backs exactly one graph: re-initializing a graph resets the state, and
// the graph must be destroyed before the state it was built against.
struct nc_stream_state {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> caches;

    nc_stream_state() = default;
    ~nc_stream_state() { free_tensors(); }

    nc_stream_state(const nc_stream_state&) = delete;
    nc_stream_state& operator=(const nc_stream_state&) = delete;

    void free_tensors() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        caches.clear();
    }

    // Reset the stream to silence without discarding the graph built against these tensors.
    void clear() {
        if (buffer) {
            ggml_backend_buffer_clear(buffer, 0);
        }
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
    graph.owner = nullptr;
    graph.chunk_frames = 0;
    graph.output_samples = 0;
    graph.samples_per_frame = 0;
    graph.latent_data.clear();
    graph.audio_data.clear();
}

// Allocate one persistent layer cache in the state's own context. The tensors are backed
// by a backend buffer once the whole graph is built, before the graph allocator runs, so
// the allocator treats them as pre-allocated leaves and never reuses their memory.
static ggml_tensor*
nc_stream_cache_tensor(
    nc_stream_state& state, nc_stream_cache_kind kind, int64_t len, int64_t channels) {
    ggml_tensor* tensor = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32, len, channels, 1);
    ggml_set_input(tensor);
    const std::string name = (kind == NC_STREAM_CACHE_CONV ? "nc_conv_cache_" : "nc_deconv_tail_") +
                             std::to_string(state.caches.size());
    ggml_set_name(tensor, name.c_str());
    state.caches.push_back(tensor);
    return tensor;
}

// Refresh `cache` from `next` once `barrier` -- the last node reading the old value -- has run.
static void
nc_stream_write_cache(
    ggml_context* ctx, nc_stream_graph_io& io, ggml_tensor* barrier, ggml_tensor* next,
    ggml_tensor* cache) {
    io.cache_writes.push_back({barrier, ggml_cpy(ctx, next, cache)});
}

static ggml_tensor*
nc_stream_causal_conv1d(
    ggml_context* ctx, ggml_tensor* x, const nc_conv& conv, nc_stream_state& state,
    nc_stream_graph_io& io) {
    const int kernel = (int)conv.w->ne[0];
    const int left_pad = (kernel - 1) * conv.dilation;
    ggml_tensor* conv_in = x;

    if (left_pad > 0) {
        ggml_tensor* cache =
            nc_stream_cache_tensor(state, NC_STREAM_CACHE_CONV, left_pad, x->ne[1]);
        conv_in = ggml_concat(ctx, cache, x, 0);

        const int64_t tail_start = conv_in->ne[0] - left_pad;
        ggml_tensor* tail = ggml_view_3d(
            ctx, conv_in, left_pad, x->ne[1], 1, conv_in->nb[1], conv_in->nb[2],
            (size_t)tail_start * conv_in->nb[0]);
        // The concat is the only reader of `cache`, so the refresh may follow it directly.
        nc_stream_write_cache(ctx, io, conv_in, tail, cache);
    }

    ggml_tensor* y = ggml_conv_1d(ctx, conv.w, conv_in, conv.stride, 0, conv.dilation);
    y = ggml_add(ctx, y, conv.b);
    return y;
}

static ggml_tensor*
nc_stream_causal_conv_transpose1d(
    ggml_context* ctx, ggml_tensor* x, const nc_conv& conv, nc_stream_state& state,
    nc_stream_graph_io& io) {
    const int64_t out_len = x->ne[0] * conv.stride;
    ggml_tensor* full = ggml_conv_transpose_1d(ctx, conv.w, x, conv.stride, 0, 1);
    const int64_t tail_len = std::max<int64_t>(0, full->ne[0] - out_len);

    ggml_tensor* current = nullptr;
    if (tail_len > 0) {
        ggml_tensor* prev_tail =
            nc_stream_cache_tensor(state, NC_STREAM_CACHE_DECONV, tail_len, full->ne[1]);

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
        // `current` descends from the add that consumes the old tail, so it is the barrier.
        nc_stream_write_cache(ctx, io, current, next_tail, prev_tail);
    } else {
        current = ggml_view_3d(ctx, full, out_len, conv.w->ne[1], 1, full->nb[1], full->nb[2], 0);
    }

    return ggml_add(ctx, current, conv.b);
}

static ggml_tensor*
nc_stream_residual_block(
    ggml_context* ctx, ggml_tensor* x, const nc_res_block& block, nc_stream_state& state,
    nc_stream_graph_io& io) {
    ggml_tensor* y = half_snake(ctx, x, block.input_act);
    y = nc_stream_causal_conv1d(ctx, y, block.input_conv, state, io);
    y = half_snake(ctx, y, block.skip_act);
    y = nc_stream_causal_conv1d(ctx, y, block.skip_conv, state, io);
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

    // The graph nodes will point straight at the state tensors, so the previous state is
    // discarded together with the previous graph and the stream restarts from silence.
    state.free_tensors();
    {
        ggml_init_params cache_params = {
            // One over the limit: nc_stream_cache_tensor can create the cache that trips
            // the post-build check, and an exhausted pool aborts rather than reporting.
            /*.mem_size   =*/ggml_tensor_overhead() * (NANO_CODEC_MAX_CACHES + 1),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        state.ctx = ggml_init(cache_params);
    }
    if (!state.ctx) {
        fprintf(stderr, "failed to allocate stream cache context\n");
        nc_stream_decode_graph_free(graph);
        return false;
    }

    {
        const ggml_nvtx::range nvtx_build("nanocodec_stream_build_persistent_decoder_graph");
        graph.latent = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, chunk_frames, h.latent_dim, 1);
        ggml_set_name(graph.latent, "nanocodec_stream_latent");
        ggml_set_input(graph.latent);

        ggml_tensor* x =
            nc_stream_causal_conv1d(graph.ctx, graph.latent, model.pre_conv, state, graph.io);
        for (size_t i = 0; i < h.up_rates.size(); ++i) {
            x = half_snake(graph.ctx, x, model.activations[i]);
            x = nc_stream_causal_conv_transpose1d(graph.ctx, x, model.up_convs[i], state, graph.io);
            x = nc_stream_hifigan_reslayer(graph.ctx, x, model.res_layers[i], state, graph.io);
        }

        x = half_snake(graph.ctx, x, model.post_activation);
        x = nc_stream_causal_conv1d(graph.ctx, x, model.post_conv, state, graph.io);
        x = ggml_clamp(graph.ctx, x, -1.0f, 1.0f);
        ggml_set_name(x, "nanocodec_stream_audio");
        ggml_set_output(x);
        graph.audio = x;

        graph.gf = ggml_new_graph_custom(graph.ctx, NANO_CODEC_MAX_NODES, false);
        // Expanding each barrier before its write keeps every cache refresh ordered after
        // the node that reads the old value, while still letting the refreshes interleave
        // with the forward pass so the allocator can recycle activation buffers early.
        for (const nc_stream_cache_write& refresh : graph.io.cache_writes) {
            ggml_build_forward_expand(graph.gf, refresh.barrier);
            ggml_build_forward_expand(graph.gf, refresh.write);
        }
        ggml_build_forward_expand(graph.gf, graph.audio);
        tag_graph_first_node(graph.gf);
    }

    if ((int)state.caches.size() > NANO_CODEC_MAX_CACHES) {
        fprintf(
            stderr, "stream graph needs %zu layer caches, over the %d limit\n", state.caches.size(),
            NANO_CODEC_MAX_CACHES);
        nc_stream_decode_graph_free(graph);
        return false;
    }

    // Back the caches before the graph allocator runs so it sees them as pre-allocated
    // leaves and never hands their memory to an intermediate tensor.
    state.buffer = ggml_backend_alloc_ctx_tensors(state.ctx, model.backend);
    if (!state.buffer) {
        fprintf(stderr, "failed to allocate stream layer caches on the backend\n");
        nc_stream_decode_graph_free(graph);
        return false;
    }
    state.clear();

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

    // Every node above points at this state's cache tensors; record it so a later decode
    // cannot pair the graph with a different state that happens to hold as many caches.
    graph.owner = &state;
    graph.samples_per_frame = graph.output_samples / (size_t)chunk_frames;
    graph.latent_data.assign((size_t)chunk_frames * h.latent_dim, 0.0f);
    graph.audio_data.resize(graph.output_samples);
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
    // The graph nodes point at this state's cache tensors; a state that was reset or paired
    // with a different graph would leave those reads dangling.
    if (!state.buffer || graph.owner != &state ||
        state.caches.size() != graph.io.cache_writes.size()) {
        fprintf(stderr, "stream state does not back the persistent stream graph\n");
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
        // The layer caches stay in backend memory across chunks, so the latent is the only
        // input the host has to upload.
        const ggml_nvtx::range nvtx_inputs("nanocodec_stream_graph_set_inputs");
        ggml_backend_tensor_set(
            graph.latent, graph.latent_data.data(), 0, graph.latent_data.size() * sizeof(float));
    }

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
        // The graph refreshed the layer caches in place, so only the audio comes back.
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
