// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include "fastconformer.h"

using namespace nemo_speech::asr;


#include <cmath>
#include <cstring>

#include "runtime.h"

// Portable equivalent of ggml_pad_ext that emits ops universally supported
// across CUDA / Vulkan / Metal / CPU. ggml_pad_ext with nonzero left-pad
// isn't supported on every backend (Metal's PAD kernel, for one, rejects
// it), so we emulate left-pad with `right-pad by (lp+rp) on each dim` +
// `ggml_roll(lp...)` to rotate the zero region to the front. Both ops are
// supported on all targeted backends, keeping the encoder graph
// GPU-resident regardless of target.
static inline ggml_tensor*
pad_ext_portable(
    ggml_context* ctx, ggml_tensor* t, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2,
    int lp3, int rp3) {
    ggml_tensor* right_padded =
        ggml_pad_ext(ctx, t, 0, lp0 + rp0, 0, lp1 + rp1, 0, lp2 + rp2, 0, lp3 + rp3);

    // ggml_roll asserts abs(shift) < ne[d]. When the source had ne[d]=0 and
    // rp[d]=0, right_padded->ne[d] == lp[d] and a shift of lp[d] would trip
    // the assertion. In that case the right-padded tensor is already
    // all-zero on dim d (there was no data to roll), so we clamp shift to 0.
    int s0 = (lp0 < right_padded->ne[0]) ? lp0 : 0;
    int s1 = (lp1 < right_padded->ne[1]) ? lp1 : 0;
    int s2 = (lp2 < right_padded->ne[2]) ? lp2 : 0;
    int s3 = (lp3 < right_padded->ne[3]) ? lp3 : 0;
    if (s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0) {
        return right_padded;
    }
    return ggml_roll(ctx, right_padded, s0, s1, s2, s3);
}

// Cache-aware Nemotron uses **causal** Conv2D subsampling: for every strided
// conv we pad (k-1, stride-1) on both axes instead of the standard symmetric
// (k-1)/2 each side. This grows the post-subsample feature length by one
// (e.g. 128 -> 65 -> 33 -> 17 vs. 128 -> 64 -> 32 -> 16) so the Linear's
// in_features match the GGUF's `pre_encode.out.weight` (256*17 for Nemotron).
// We keep the Conv modules themselves at padding=0 and emit explicit
// `ggml_pad_ext` ops in build_graph when cfg.conv_context == Causal.
static int
subsample_feature_out_len(int feat_in, int n_stages, bool causal) {
    int len = feat_in;
    for (int i = 0; i < n_stages; i++) {
        if (causal) {
            // pad_left=k-1=2, pad_right=stride-1=1, k=3, stride=2:
            // out = floor((L + 3 - 3)/2 + 1) = L/2 + 1
            len = len / 2 + 1;
        } else {
            // pad=1 each side, k=3, stride=2: out = ceil(L / 2).
            len = (len + 1) / 2;
        }
    }
    return len;
}

SubSampling::SubSampling(const std::string& name, const EncoderConfig& cfg)
    : name_(name), cfg_(cfg) {
    const int sampling_num = static_cast<int>(std::log2(cfg.subsampling_factor));
    const int conv_channels = cfg.subsampling_conv_channels;
    const int kernel_size = 3;
    const int stride = 2;
    const bool causal = (cfg.conv_context == ConvContext::Causal);
    // Symmetric path keeps Conv2D's internal padding=1. Causal path uses
    // padding=0 + explicit asymmetric pad in build_graph.
    const int padding = causal ? 0 : 1;

    conv_ = new ggml_runtime::SequenceModule(name_ + ".conv");
    int in_channels = 1;
    int layer_index = 0;

    // Layer 0: Conv2d
    conv_->modules.push_back(new ggml_runtime::Conv2D(
        name_ + ".conv." + std::to_string(layer_index), in_channels, conv_channels, kernel_size,
        stride, padding));
    layer_index++;
    in_channels = conv_channels;
    // Layer 1: ReLU
    conv_->modules.push_back(
        new ggml_runtime::ReLU(name_ + ".conv." + std::to_string(layer_index)));
    layer_index++;

    // (sampling_num - 1) repeats of [Conv2dDW, Conv2d 1x1, ReLU]
    for (int i = 0; i < sampling_num - 1; i++) {
        conv_->modules.push_back(new ggml_runtime::Conv2DDW(
            name_ + ".conv." + std::to_string(layer_index), in_channels, in_channels, kernel_size,
            stride, padding));
        layer_index++;
        conv_->modules.push_back(new ggml_runtime::Conv2D(
            name_ + ".conv." + std::to_string(layer_index), in_channels, conv_channels, 1, 1, 0));
        layer_index++;
        conv_->modules.push_back(
            new ggml_runtime::ReLU(name_ + ".conv." + std::to_string(layer_index)));
        layer_index++;
        in_channels = conv_channels;
    }

    const int sub_out_len = subsample_feature_out_len(
        cfg.feat_in, static_cast<int>(std::log2(cfg.subsampling_factor)), causal);
    out_ = new ggml_runtime::Linear(
        name_ + ".out", conv_channels * sub_out_len, cfg.d_model,
        /*use_bias=*/true);
}

SubSampling::~SubSampling() {
    delete conv_;
    delete out_;
}


void
SubSampling::define_tensors(ggml_runtime::Session* session) {
    conv_->define_tensors(session);
    out_->define_tensors(session);
}

ggml_runtime::TensorBag
SubSampling::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    const bool causal = (cfg_.conv_context == ConvContext::Causal);
    ggml_runtime::TensorBag conv_output;
    if (!causal) {
        // Symmetric convolutions use their native padding.
        conv_output = conv_->build_graph(session, input_tensors, session_tensor_container);
    } else {
        // Cache-aware Nemotron path. The SequenceModule was built with
        // padding=0 for the strided Conv2D / Conv2DDW; here we walk the
        // modules manually and inject `ggml_pad_ext(k-1, stride-1, ...)`
        // before each k=3,stride=2 conv (and Conv2DDW). This matches the
        // reference build_causal_conv2d / build_causal_dw_conv2d.
        const int kW = 3, kH = 3, sW = 2, sH = 2;
        const int pad_l = kW - 1;  // 2
        const int pad_r = sW - 1;  // 1
        const int pad_t = kH - 1;  // 2
        const int pad_b = sH - 1;  // 1
        auto in_t = input_tensors.get_tensor(0);
        auto bf_ctx_in = session_tensor_container->get_ctx_of_buffer_type(in_t.buft);
        ggml_runtime::TensorBag bag = input_tensors;
        for (auto* m : conv_->modules) {
            // Heuristic: pre-pad before stride-2 layers (Conv2D layer 0 and
            // each Conv2DDW). 1x1 Conv2D / ReLU pass through untouched.
            const bool is_conv2d = (dynamic_cast<ggml_runtime::Conv2D*>(m) != nullptr);
            const bool is_dw = (dynamic_cast<ggml_runtime::Conv2DDW*>(m) != nullptr);
            if (is_dw || (is_conv2d && m == conv_->modules.front())) {
                auto t = bag.get_tensor(0);
                auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(t.buft);
                auto padded =
                    pad_ext_portable(bf_ctx.ctx, t.tensor, pad_l, pad_r, pad_t, pad_b, 0, 0, 0, 0);
                bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(padded, t.buft));
            }
            bag = m->build_graph(session, bag, session_tensor_container);
        }
        conv_output = bag;
        (void)bf_ctx_in;
    }
    auto conv_ret_tensor = conv_output.get_tensor(0);
    auto buft_ctx = session_tensor_container->get_ctx_of_buffer_type(conv_ret_tensor.buft);
    // NeMo: x = x.view(b, -1, t) then Linear along feature dim.
    // Conv output here is (W=T_sub, H=F_sub, C=subsampling_conv_channels, B=1).
    // Feed Linear which does mul_mat with weight (in=C*F, out=d_model):
    // transpose (0,2,1,3): (W, C, H, B), then view as (C*H, W, B) = (in, T, B).
    auto transpose_1_2 = ggml_permute(buft_ctx.ctx, conv_ret_tensor.tensor, 0, 2, 1, 3);
    auto cont_tensor = ggml_cont(buft_ctx.ctx, transpose_1_2);
    auto reshape_tensor = ggml_reshape_4d(
        buft_ctx.ctx, cont_tensor, cont_tensor->ne[0] * cont_tensor->ne[1], cont_tensor->ne[2],
        cont_tensor->ne[3], 1);
    ggml_runtime::TensorBag output_tensors;
    output_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(reshape_tensor, buft_ctx.buft));
    return out_->build_graph(session, output_tensors, session_tensor_container);
}

void
SubSampling::set_data(ggml_runtime::Session* session) {
    conv_->set_data(session);
    out_->set_data(session);
}

ConformerFF::ConformerFF(const std::string& name, int d_model, int d_ff, bool use_bias)
    : name_(name) {
    linear1_ = new ggml_runtime::Linear(name_ + ".linear1", d_model, d_ff, use_bias);
    linear2_ = new ggml_runtime::Linear(name_ + ".linear2", d_ff, d_model, use_bias);
}

ConformerFF::~ConformerFF() {
    delete linear1_;
    delete linear2_;
}


void
ConformerFF::define_tensors(ggml_runtime::Session* session) {
    linear1_->define_tensors(session);
    linear2_->define_tensors(session);
}

ggml_runtime::TensorBag
ConformerFF::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    auto ff_input = input_tensors.get_tensor(0);
    if (session->params.use_gpu && ff_input.tensor->type == GGML_TYPE_F32 &&
        session->gguf_loader->get_tensor_type(name_ + ".linear1.weight") == GGML_TYPE_BF16) {
        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(ff_input.buft);
        auto cast = ggml_cast(bf_ctx.ctx, ff_input.tensor, GGML_TYPE_BF16);
        input_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(cast, ff_input.buft));
    }
    auto out_bag = linear1_->build_graph(session, input_tensors, session_tensor_container);
    auto tensor = out_bag.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(tensor.buft);
    auto silu = ggml_silu(bf_ctx.ctx, tensor.tensor);
    // The second BF16 projection already rounds this activation to BF16 in
    // the CUDA backend. Make that conversion graph-visible so CUDA can fuse
    // SiLU with the cast and let cuBLAS consume the result directly.
    if (session->params.use_gpu &&
        session->gguf_loader->get_tensor_type(name_ + ".linear2.weight") == GGML_TYPE_BF16) {
        silu = ggml_cast(bf_ctx.ctx, silu, GGML_TYPE_BF16);
    }
    out_bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(silu, bf_ctx.buft));
    return linear2_->build_graph(session, out_bag, session_tensor_container);
}

void
ConformerFF::set_data(ggml_runtime::Session* session) {
    linear1_->set_data(session);
    linear2_->set_data(session);
}

ConformerConv::ConformerConv(
    const std::string& name, int d_model, int kernel_size, bool use_bias, ConvNorm norm,
    ConvContext context)
    : name_(name), d_model_(d_model), kernel_size_(kernel_size), use_bias_(use_bias),
      norm_kind_(norm), context_(context), batch_norm_(nullptr), layer_norm_(nullptr) {
    // pointwise_conv1: Conv1d(d_model -> 2*d_model, k=1), bias=use_bias
    pointwise_conv1_ = new ggml_runtime::Conv1D(
        name_ + ".pointwise_conv1", d_model, d_model * 2, 1, 1, 0, 1, use_bias);
    // pointwise_conv2: Conv1d(d_model -> d_model, k=1), bias=use_bias
    pointwise_conv2_ = new ggml_runtime::Conv1D(
        name_ + ".pointwise_conv2", d_model, d_model, 1, 1, 0, 1, use_bias);
    // depthwise_conv: Conv1d(d_model -> d_model, k, groups=d_model), bias=use_bias.
    // Symmetric: pad = (k-1)/2 on both sides. Causal: pad fully on the left;
    // we pass padding=0 here and do the prepend ourselves (either via zero
    // pre-pad in build_graph or via the conv_cache_cur in build_graph_cached).
    const int pad = (context_ == ConvContext::Symmetric) ? (kernel_size - 1) / 2 : 0;
    depthwise_conv_ = new ggml_runtime::Conv1D(
        name_ + ".depthwise_conv", d_model, d_model, kernel_size, 1, pad, 1, use_bias,
        /*is_dw=*/true);

    if (norm_kind_ == ConvNorm::BatchNorm) {
        batch_norm_ = new ggml_runtime::BatchNorm1d(name_ + ".batch_norm", d_model);
    } else {
        // NeMo names the LN params *.batch_norm.{weight,bias} even when it's
        // actually a LayerNorm. Keep the same tensor names so the GGUF schema
        // doesn't fork.
        const int64_t ln_shape[GGML_MAX_DIMS] = {d_model, 1, 1, 1};
        layer_norm_ = new ggml_runtime::LayerNorm(name_ + ".batch_norm", ln_shape);
    }
}

ConformerConv::~ConformerConv() {
    delete pointwise_conv1_;
    delete pointwise_conv2_;
    delete depthwise_conv_;
    delete batch_norm_;
    delete layer_norm_;
}


void
ConformerConv::define_tensors(ggml_runtime::Session* session) {
    pointwise_conv1_->define_tensors(session);
    depthwise_conv_->define_tensors(session);
    if (norm_kind_ == ConvNorm::BatchNorm)
        batch_norm_->define_tensors(session);
    else
        layer_norm_->define_tensors(session);
    pointwise_conv2_->define_tensors(session);
}

// Internal helper: shared graph body from GLU output through pointwise_conv2.
// `x_glu` is (T, d_model, B) "channel-mid" (the layout depthwise_conv_ expects:
// ne[0]=T, ne[1]=d_model). When cache_cur is non-null we replace the depthwise
// conv's native left-pad with prepended cache state and emit cache_next =
// last (kernel-1) frames of the concatenated buffer.
ggml_runtime::TensorBag
ConformerConv::build_post_glu(
    ggml_runtime::Session* session, ggml_runtime::TensorContainer* session_tensor_container,
    ggml_runtime::ggml_bf_tensor x_glu, ggml_tensor* cache_cur, ggml_tensor** cache_next) {
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x_glu.buft);
    const int kernel = kernel_size_;
    const int cache_len = kernel - 1;

    ggml_tensor* dw_input = x_glu.tensor;  // (T, d_model, B)
    ggml_tensor* concat_buffer = nullptr;  // (cache_len + T, d_model, B) for cache_next slicing

    if (context_ == ConvContext::Causal) {
        if (cache_cur != nullptr) {
            // cache_cur is stored on host as (d_model_inner, cache_len_outer):
            // tensor[:, t] gives the d_model values of cache frame t and they
            // sit contiguously in memory. dw_input is in the opposite layout
            // - (T_inner, d_model_outer, B) - so we need a real PERMUTE
            // (with ggml_cont to materialize the transposed data), not a
            // reshape (which is a metadata-only view).
            auto cache_perm =
                ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, cache_cur, 1, 0, 2, 3));
            dw_input = ggml_concat(bf_ctx.ctx, cache_perm, dw_input, 0);
            concat_buffer = dw_input;
        } else {
            // First chunk OR non-streaming causal mode: zero-pad on the left
            // by kernel-1 frames so the conv keeps the same output length.
            dw_input = pad_ext_portable(
                bf_ctx.ctx, dw_input, cache_len, 0,  // dim 0 (time): left, right
                0, 0, 0, 0, 0, 0);
            concat_buffer = dw_input;
        }
    }

    // Run the depthwise conv. With Causal we pre-padded, so its native
    // padding is 0; with Symmetric the conv itself does (k-1)/2 each side.
    ggml_runtime::TensorBag dw_in_bag;
    dw_in_bag.add_tensor(ggml_runtime::ggml_bf_tensor(dw_input, x_glu.buft));
    auto dw_out_bag = depthwise_conv_->build_graph(session, dw_in_bag, session_tensor_container);
    auto dw_out = dw_out_bag.get_tensor(0);

    // Normalize first; apply the elementwise SiLU only after the one layout
    // transform needed by pointwise-conv2 so it can fuse with the BF16 handoff.
    ggml_runtime::ggml_bf_tensor y_normed(nullptr, nullptr);
    bool y_normed_is_ct = false;
    if (norm_kind_ == ConvNorm::BatchNorm) {
        dw_out_bag = batch_norm_->build_graph(session, dw_out_bag, session_tensor_container);
        y_normed = dw_out_bag.get_tensor(0);  // (T,C,B)
    } else {
        // (T,C,B) -> (C,T,B) for LayerNorm-over-features. Keep this layout:
        // pointwise-conv2 consumes it directly.
        auto y_ct = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, dw_out.tensor, 1, 0, 2, 3));
        ggml_runtime::TensorBag ln_in;
        ln_in.add_tensor(ggml_runtime::ggml_bf_tensor(y_ct, x_glu.buft));
        ln_in = layer_norm_->build_graph(session, ln_in, session_tensor_container);
        y_normed = ln_in.get_tensor(0);  // (C,T,B)
        y_normed_is_ct = true;
    }

    // Convert (T,C,B) to (C,T,B) once and retain it through the residual path.
    auto y_ct = y_normed_is_ct
                    ? y_normed.tensor
                    : ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, y_normed.tensor, 1, 0, 2, 3));
    auto w2 =
        session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv2.weight");
    y_ct = ggml_silu(bf_ctx.ctx, y_ct);
    if (session->params.use_gpu && y_ct->type == GGML_TYPE_F32 &&
        w2.tensor->type == GGML_TYPE_BF16) {
        y_ct = ggml_cast(bf_ctx.ctx, y_ct, GGML_TYPE_BF16);
    }
    ggml_tensor* w2_mat = w2.tensor;
    if (w2_mat->ne[0] == 1 && w2_mat->ne[2] > 1) {
        w2_mat = ggml_reshape_2d(bf_ctx.ctx, w2_mat, w2_mat->ne[1], w2_mat->ne[2]);
    }
    auto z_tensor = ggml_mul_mat(bf_ctx.ctx, w2_mat, y_ct);
    if (use_bias_) {
        auto b2 =
            session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv2.bias");
        z_tensor = session->params.use_gpu && w2_mat->type == GGML_TYPE_BF16 &&
                           y_ct->ne[2] * y_ct->ne[3] > 1
                       ? ggml_add(bf_ctx.ctx, z_tensor, b2.tensor)
                       : ggml_add_inplace(bf_ctx.ctx, z_tensor, b2.tensor);
    }
    ggml_runtime::TensorBag out_bag;
    out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(z_tensor, y_normed.buft));

    // Emit conv cache_next: last (kernel-1) frames of the buffer we fed into
    // the depthwise conv. concat_buffer layout is (cache_len+T, d_model, B).
    // We want (d_model, cache_len) - permute then view.
    if (cache_next != nullptr) {
        ggml_tensor* tail_src = nullptr;
        if (concat_buffer != nullptr) {
            // (cache_len+T, d_model, B) -> permute to (d_model, cache_len+T, B)
            auto buf_perm =
                ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, concat_buffer, 1, 0, 2, 3));
            const int64_t full = buf_perm->ne[1];
            const int64_t take = cache_len;
            const int64_t offset = full - take;
            tail_src = ggml_view_3d(
                bf_ctx.ctx, buf_perm, d_model_, take, buf_perm->ne[2], buf_perm->nb[1],
                buf_perm->nb[2], offset * buf_perm->nb[1]);
        } else {
            // Derive non-causal cache_next from x_glu in channel-last layout.
            auto xperm = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x_glu.tensor, 1, 0, 2, 3));
            const int64_t full = xperm->ne[1];
            const int64_t take = std::min<int64_t>(cache_len, full);
            const int64_t offset = full - take;
            tail_src = ggml_view_3d(
                bf_ctx.ctx, xperm, d_model_, take, xperm->ne[2], xperm->nb[1], xperm->nb[2],
                offset * xperm->nb[1]);
            tail_src = ggml_cont(bf_ctx.ctx, tail_src);
        }
        *cache_next = tail_src;
    }
    return out_bag;
}

ggml_runtime::TensorBag
ConformerConv::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    // Pointwise-conv1 is a Linear over d_model; preserve the input layout until
    // its output is materialized for GLU and depthwise convolution.
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    auto w1 =
        session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv1.weight");
    ggml_tensor* pointwise_input = input_tensor.tensor;
    if (session->params.use_gpu && pointwise_input->type == GGML_TYPE_F32 &&
        w1.tensor->type == GGML_TYPE_BF16) {
        pointwise_input = ggml_cast(bf_ctx.ctx, pointwise_input, GGML_TYPE_BF16);
    }
    ggml_tensor* w1_mat = w1.tensor;
    if (w1_mat->ne[0] == 1 && w1_mat->ne[2] > 1) {
        w1_mat = ggml_reshape_2d(bf_ctx.ctx, w1_mat, w1_mat->ne[1], w1_mat->ne[2]);
    }
    auto x_ct = ggml_mul_mat(bf_ctx.ctx, w1_mat, pointwise_input);
    if (use_bias_) {
        auto b1 =
            session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv1.bias");
        x_ct = session->params.use_gpu && w1_mat->type == GGML_TYPE_BF16 &&
                       input_tensor.tensor->ne[2] * input_tensor.tensor->ne[3] > 1
                   ? ggml_add(bf_ctx.ctx, x_ct, b1.tensor)
                   : ggml_add_inplace(bf_ctx.ctx, x_ct, b1.tensor);
    }
    ggml_runtime::TensorBag out_bag;
#ifdef NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS
    if (session->params.use_gpu) {
        // Split contiguous (2*C,T,B) in the CUDA GLU kernel and materialize
        // only the final (T,C,B) depthwise-convolution layout.
        auto x_glu_ct = ggml_glu(bf_ctx.ctx, x_ct, GGML_GLU_OP_SIGMOID, /*swapped=*/true);
        auto x_glu_tc = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x_glu_ct, 1, 0, 2, 3));
        out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(x_glu_tc, input_tensor.buft));
    } else
#endif
    {
        auto x_tc = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x_ct, 1, 0, 2, 3));
        out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(x_tc, input_tensor.buft));
        auto x = out_bag.get_tensor(0);  // (T, 2*d_model, B, 1)

        auto part_a = ggml_view_4d(
            bf_ctx.ctx, x.tensor, x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2],
            x.tensor->ne[3], x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3], 0);
        auto part_b = ggml_view_4d(
            bf_ctx.ctx, x.tensor, x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2],
            x.tensor->ne[3], x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3],
            x.tensor->nb[1] * (x.tensor->ne[1] / 2));
        part_a = ggml_cont(bf_ctx.ctx, part_a);
        part_b = ggml_cont(bf_ctx.ctx, part_b);
        auto part_b_sigmoid = ggml_sigmoid(bf_ctx.ctx, part_b);
        auto x_glu = ggml_mul(bf_ctx.ctx, part_a, part_b_sigmoid);
        out_bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_glu, input_tensor.buft));
    }

    return build_post_glu(
        session, session_tensor_container, out_bag.get_tensor(0), /*cache_cur=*/nullptr,
        /*cache_next=*/nullptr);
}

ggml_runtime::TensorBag
ConformerConv::build_graph_cached(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container, ggml_tensor* cache_cur,
    ggml_tensor** cache_next) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    auto w1 =
        session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv1.weight");
    ggml_tensor* w1_mat = w1.tensor;
    if (w1_mat->ne[0] == 1 && w1_mat->ne[2] > 1) {
        w1_mat = ggml_reshape_2d(bf_ctx.ctx, w1_mat, w1_mat->ne[1], w1_mat->ne[2]);
    }
    auto x_ct = ggml_mul_mat(bf_ctx.ctx, w1_mat, input_tensor.tensor);
    if (use_bias_) {
        auto b1 =
            session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv1.bias");
        x_ct = session->params.use_gpu && w1_mat->type == GGML_TYPE_BF16 &&
                       input_tensor.tensor->ne[2] * input_tensor.tensor->ne[3] > 1
                   ? ggml_add(bf_ctx.ctx, x_ct, b1.tensor)
                   : ggml_add_inplace(bf_ctx.ctx, x_ct, b1.tensor);
    }
    auto x_tc = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x_ct, 1, 0, 2, 3));
    ggml_runtime::TensorBag out_bag;
    out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(x_tc, input_tensor.buft));
    auto x = out_bag.get_tensor(0);

    auto part_a = ggml_view_4d(
        bf_ctx.ctx, x.tensor, x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2],
        x.tensor->ne[3], x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3], 0);
    auto part_b = ggml_view_4d(
        bf_ctx.ctx, x.tensor, x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2],
        x.tensor->ne[3], x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3],
        x.tensor->nb[1] * (x.tensor->ne[1] / 2));
    part_a = ggml_cont(bf_ctx.ctx, part_a);
    part_b = ggml_cont(bf_ctx.ctx, part_b);
    auto part_b_sigmoid = ggml_sigmoid(bf_ctx.ctx, part_b);
    auto x_glu = ggml_mul(bf_ctx.ctx, part_a, part_b_sigmoid);
    out_bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_glu, input_tensor.buft));

    return build_post_glu(
        session, session_tensor_container, out_bag.get_tensor(0), cache_cur, cache_next);
}

// Channels-inner cache-aware convolution. Input is norm_conv output in
// (d_model, T, B) and everything stays in that layout: the k=1 pointwise
// convs are plain mul_mat over the feature dim (no transposes), GLU gates
// dim-0 halves, the depthwise conv runs via the cwhn direct kernel over
// [cache | x], and LayerNorm sees features innermost natively.
ggml_runtime::TensorBag
ConformerConv::build_graph_cached_ct(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container, ggml_tensor* cache_cur,
    ggml_tensor** cache_next, ggml_tensor* w_ct) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    ggml_tensor* x = input_tensor.tensor;  // (C, T, B)
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    const int64_t cache_len = kernel_size_ - 1;
    const size_t es = ggml_element_size(x);

    auto weight = [&](const char* suffix) {
        ggml_tensor* w = session->model_tensor_container->get_tensor_by_name(name_ + suffix).tensor;
        // Older CTC GGUFs keep k=1 convs as [1, in, out]; same bytes as [in, out].
        if (w->ne[0] == 1 && w->ne[2] > 1) {
            w = ggml_reshape_2d(bf_ctx.ctx, w, w->ne[1], w->ne[2]);
        }
        return w;
    };

    // pointwise_conv1 (k=1) == Linear over features: (C, T, B) -> (2C, T, B).
    ggml_tensor* y = ggml_mul_mat(bf_ctx.ctx, weight(".pointwise_conv1.weight"), x);
    if (use_bias_) {
        auto b =
            session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv1.bias");
        y = ggml_add_inplace(bf_ctx.ctx, y, b.tensor);
    }

    // GLU on dim-0 halves: x_glu = a * sigmoid(b).
    auto part_a = ggml_view_3d(bf_ctx.ctx, y, C, T, B, y->nb[1], y->nb[2], 0);
    auto part_b =
        ggml_view_3d(bf_ctx.ctx, y, C, T, B, y->nb[1], y->nb[2], static_cast<size_t>(C) * es);
    auto gate = ggml_sigmoid(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, part_b));
    auto x_glu = ggml_mul(bf_ctx.ctx, part_a, gate);  // (C, T, B), contiguous

    // Depthwise causal conv on u = [cache | x_glu] via the cwhn direct
    // kernel: express (C, time) memory as a channels-inner (W=time, H=1, C, B)
    // view; the repacked weight w_ct is (C, k) memory = (k,1,1,C) logical.
    // This path is Causal-only (see supports_ct_layout).
    ggml_tensor* u = nullptr;
    if (cache_cur != nullptr) {
        u = ggml_concat(bf_ctx.ctx, cache_cur, x_glu, 1);  // (C, cache+T, B)
    } else {
        u = pad_ext_portable(bf_ctx.ctx, x_glu, 0, 0, (int)cache_len, 0, 0, 0, 0, 0);
    }
    const int64_t Tu = u->ne[1];
    auto u4 = ggml_reshape_4d(bf_ctx.ctx, u, C, Tu, 1, B);
    auto u_cwhn = ggml_permute(bf_ctx.ctx, u4, 2, 0, 1, 3);  // logical (Tu, 1, C, B)
    auto w4 = ggml_reshape_4d(bf_ctx.ctx, w_ct, C, kernel_size_, 1, 1);
    auto w_cwhn = ggml_permute(bf_ctx.ctx, w4, 3, 0, 1, 2);  // logical (k, 1, 1, C)
    auto y4 = ggml_conv_2d_dw_direct(bf_ctx.ctx, w_cwhn, u_cwhn, 1, 1, 0, 0, 1, 1);
    // cwhn output keeps channel-inner strides; permuted back it is contiguous.
    auto dw =
        ggml_reshape_3d(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, y4, 1, 2, 0, 3), C, y4->ne[0], B);
    if (use_bias_) {
        auto b =
            session->model_tensor_container->get_tensor_by_name(name_ + ".depthwise_conv.bias");
        dw = ggml_add(bf_ctx.ctx, dw, b.tensor);
    }

    // Norm + SiLU. Features are already innermost, so LayerNorm applies
    // directly (this path requires norm_kind_ == LayerNorm).
    ggml_runtime::TensorBag ln_in;
    ln_in.add_tensor(ggml_runtime::ggml_bf_tensor(dw, input_tensor.buft));
    ln_in = layer_norm_->build_graph(session, ln_in, session_tensor_container);
    auto y_silu = ggml_silu(bf_ctx.ctx, ln_in.get_tensor(0).tensor);

    // pointwise_conv2: (C, T, B) -> (C, T, B).
    ggml_tensor* out_t = ggml_mul_mat(bf_ctx.ctx, weight(".pointwise_conv2.weight"), y_silu);
    if (use_bias_) {
        auto b =
            session->model_tensor_container->get_tensor_by_name(name_ + ".pointwise_conv2.bias");
        out_t = ggml_add_inplace(bf_ctx.ctx, out_t, b.tensor);
    }

    // Updated conv cache: last (kernel-1) frames of u, already (C, time, B) —
    // a raw view; the encoder feedback copies or materializes it.
    if (cache_next != nullptr) {
        *cache_next = ggml_view_3d(
            bf_ctx.ctx, u, C, cache_len, B, u->nb[1], u->nb[2],
            static_cast<size_t>(Tu - cache_len) * u->nb[1]);
    }

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(out_t, input_tensor.buft));
    return out;
}

void
ConformerConv::set_data(ggml_runtime::Session* session) {
    pointwise_conv1_->set_data(session);
    depthwise_conv_->set_data(session);
    if (norm_kind_ == ConvNorm::BatchNorm)
        batch_norm_->set_data(session);
    else
        layer_norm_->set_data(session);
    pointwise_conv2_->set_data(session);
}

ConformerLayer::ConformerLayer(const std::string& name, const EncoderConfig& cfg)
    : name_(name), cfg_(cfg) {
    const int64_t input_shape[4] = {cfg.d_model, 1, 1, 1};
    norm_feed_forward1_ = new ggml_runtime::LayerNorm(name_ + ".norm_feed_forward1", input_shape);
    feed_forward1_ = new ConformerFF(name_ + ".feed_forward1", cfg.d_model, cfg.d_ff, cfg.use_bias);
    norm_self_att_ = new ggml_runtime::LayerNorm(name_ + ".norm_self_att", input_shape);
    self_attn_ = new ggml_runtime::RelPositionMultiHeadAttention(
        name_ + ".self_attn", cfg.n_heads, cfg.d_model, cfg.use_bias);
    norm_conv_ = new ggml_runtime::LayerNorm(name_ + ".norm_conv", input_shape);
    conv_ = new ConformerConv(
        name_ + ".conv", cfg.d_model, cfg.conv_kernel_size, cfg.use_bias, cfg.conv_norm,
        cfg.conv_context);
    norm_feed_forward2_ = new ggml_runtime::LayerNorm(name_ + ".norm_feed_forward2", input_shape);
    feed_forward2_ = new ConformerFF(name_ + ".feed_forward2", cfg.d_model, cfg.d_ff, cfg.use_bias);
    norm_out_ = new ggml_runtime::LayerNorm(name_ + ".norm_out", input_shape);
}

ConformerLayer::~ConformerLayer() {
    delete norm_feed_forward1_;
    delete feed_forward1_;
    delete norm_self_att_;
    delete self_attn_;
    delete norm_conv_;
    delete conv_;
    delete norm_feed_forward2_;
    delete feed_forward2_;
    delete norm_out_;
}


void
ConformerLayer::define_tensors(ggml_runtime::Session* session) {
    norm_feed_forward1_->define_tensors(session);
    feed_forward1_->define_tensors(session);
    norm_self_att_->define_tensors(session);
    self_attn_->define_tensors(session);
    norm_conv_->define_tensors(session);
    conv_->define_tensors(session);
    norm_feed_forward2_->define_tensors(session);
    feed_forward2_->define_tensors(session);
    norm_out_->define_tensors(session);
}

ggml_runtime::TensorBag
ConformerLayer::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    return build_graph(
        session, input_tensors, session_tensor_container,
        /*cache=*/nullptr);
}

ggml_runtime::TensorBag
ConformerLayer::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container, LayerCacheIO* cache) {
    // Inputs: (x, pos_emb) offline; just (x) in cache-aware mode, where the
    // MHA consumes the precomputed per-layer projection (LayerCacheIO::
    // pos_proj) instead of a raw pos_emb slice.
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);

    ggml_tensor* residual = input_tensor.tensor;

    // FF1
    auto bag = norm_feed_forward1_->build_graph(session, input_tensors, session_tensor_container);
    bag = feed_forward1_->build_graph(session, bag, session_tensor_container);
    auto ff1 = bag.get_tensor(0);
    residual = ggml_add(bf_ctx.ctx, residual, ggml_scale(bf_ctx.ctx, ff1.tensor, 0.5f));

    // MHA: non-cached uses RelPositionMultiHeadAttention; cached path builds
    // the same math manually with K/V concat'd from layer cache state.
    ggml_runtime::TensorBag mha_in;
    mha_in.add_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    mha_in = norm_self_att_->build_graph(session, mha_in, session_tensor_container);
    auto x_normed = mha_in.get_tensor(0);

    ggml_tensor* attn_tensor = nullptr;
    if (cache == nullptr) {
        mha_in.add_tensor(input_tensors.get_tensor(1));  // pos_emb
        // Offline attention mask is threaded as input_tensors[2] when the
        // FastConformerEncoder built one (offline_{left,right}_ctx set). If
        // not present, fall through to the unmasked path identical to before.
        ggml_tensor* offline_mask = nullptr;
        if (input_tensors.tensor_count() >= 3) {
            offline_mask = input_tensors.get_tensor(2).tensor;
        }
        auto mha_out =
            self_attn_->build_graph_masked(session, mha_in, session_tensor_container, offline_mask);
        attn_tensor = mha_out.get_tensor(0).tensor;
    } else {
        auto mha_out = build_mha_cached(session, session_tensor_container, x_normed, cache);
        attn_tensor = mha_out.get_tensor(0).tensor;
    }
    residual = ggml_add(bf_ctx.ctx, residual, attn_tensor);

    // Conv
    ggml_runtime::TensorBag conv_in;
    conv_in.add_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    conv_in = norm_conv_->build_graph(session, conv_in, session_tensor_container);
    if (cache == nullptr) {
        conv_in = conv_->build_graph(session, conv_in, session_tensor_container);
    } else if (cache->dw_conv_w_ct != nullptr && conv_->supports_ct_layout()) {
        // Channels-inner conv module: needs the repacked depthwise weight and
        // the cwhn direct kernel (patched CUDA), provided by the encoder.
        conv_in = conv_->build_graph_cached_ct(
            session, conv_in, session_tensor_container, cache->conv_cache_cur,
            &cache->conv_cache_next, cache->dw_conv_w_ct);
    } else {
        conv_in = conv_->build_graph_cached(
            session, conv_in, session_tensor_container, cache->conv_cache_cur,
            &cache->conv_cache_next);
    }
    auto conv_out = conv_in.get_tensor(0);
    residual = ggml_add(bf_ctx.ctx, residual, conv_out.tensor);

    // FF2
    ggml_runtime::TensorBag ff2_in;
    ff2_in.add_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    ff2_in = norm_feed_forward2_->build_graph(session, ff2_in, session_tensor_container);
    ff2_in = feed_forward2_->build_graph(session, ff2_in, session_tensor_container);
    auto ff2 = ff2_in.get_tensor(0);
    residual = ggml_add(bf_ctx.ctx, residual, ggml_scale(bf_ctx.ctx, ff2.tensor, 0.5f));

    // Final norm
    ggml_runtime::TensorBag out_in;
    out_in.add_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    out_in = norm_out_->build_graph(session, out_in, session_tensor_container);

    // Offline: pass through pos_emb (and the optional offline mask, if one
    // was threaded in at input_tensors[2]) so the next ConformerLayer can
    // find them. Cache-aware bags carry only x.
    ggml_runtime::TensorBag out;
    out.add_tensor(out_in.get_tensor(0));
    if (cache == nullptr) {
        out.add_tensor(input_tensors.get_tensor(1));
        if (input_tensors.tensor_count() >= 3) {
            out.add_tensor(input_tensors.get_tensor(2));
        }
    }
    return out;
}

// Cached relative-position multi-head attention, following NeMo's
// RelPositionMultiHeadAttention.forward with a cache (multi_head_attention.py):
// cache prepended to K/V, pos_emb widened to 2*kv_len-1, and matrix_bd
// truncated to the key length after the rel-shift. Uses our `Linear` module
// (y = x @ W^T via ggml_mul_mat(weight, x)) and applies the rel-pos shift via
// the same pad+view trick used by the non-cached path (see nn.cpp) - but with
// a kv_len-wide output strip so the cached case lines up with K @ Q^T (which
// has kv_len rows).
ggml_runtime::TensorBag
ConformerLayer::build_mha_cached(
    ggml_runtime::Session* session, ggml_runtime::TensorContainer* session_tensor_container,
    ggml_runtime::ggml_bf_tensor x, LayerCacheIO* cache) {
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
    const int n_head = cfg_.n_heads;
    const int n_feat = cfg_.d_model;
    const int d_k = n_feat / n_head;

    const std::string base = name_ + ".self_attn";
    // RelPositionMultiHeadAttention runs Q/K/V as ONE fused projection (the
    // three GGUF weights/biases are stacked along the output dim at load —
    // see rel_pos_attention.cpp); there are no per-projection tensors.
    auto wqkv = session->model_tensor_container->get_tensor_by_name(base + ".linear_qkv.weight");
    auto wo = session->model_tensor_container->get_tensor_by_name(base + ".linear_out.weight");
    auto bias_u = session->model_tensor_container->get_tensor_by_name(base + ".pos_bias_u");
    auto bias_v = session->model_tensor_container->get_tensor_by_name(base + ".pos_bias_v");
    auto bqkv = cfg_.use_bias
                    ? session->model_tensor_container->get_tensor_by_name(base + ".linear_qkv.bias")
                    : ggml_runtime::ggml_bf_tensor(nullptr, nullptr);
    auto bo = cfg_.use_bias
                  ? session->model_tensor_container->get_tensor_by_name(base + ".linear_out.bias")
                  : ggml_runtime::ggml_bf_tensor(nullptr, nullptr);

    // x: (d_model, chunk_len, B). chunk_len from input.
    auto add_bias = [&](ggml_tensor* y, const ggml_runtime::ggml_bf_tensor& b) {
        if (b.tensor == nullptr)
            return y;
        return ggml_add_inplace(bf_ctx.ctx, y, b.tensor);
    };

    // One fused GEMM (+ one bias add). Per-projection slices stay strided
    // views of the fused output: ggml_concat handles non-contiguous sources,
    // and Q is carved head-split directly below, so no slice copies.
    ggml_tensor* qkv = add_bias(ggml_mul_mat(bf_ctx.ctx, wqkv.tensor, x.tensor), bqkv);
    const size_t qkv_es = ggml_element_size(qkv);
    auto qkv_view = [&](int idx) {
        return ggml_view_3d(
            bf_ctx.ctx, qkv, n_feat, qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2],
            (size_t)idx * n_feat * qkv_es);
    };
    auto k_chunk = qkv_view(1);
    auto v_chunk = qkv_view(2);

    const int64_t chunk_len = x.tensor->ne[1];
    const int64_t cache_len = (cache->k_cache_cur != nullptr) ? cache->k_cache_cur->ne[1] : 0;

    // Concat with K/V cache along the time axis (ne[1]). NeMo prepends its
    // cache to key/value the same way (MultiHeadAttention.update_cache,
    // multi_head_attention.py) but caches the PRE-projection layer input and
    // re-projects it every chunk; we cache projected K/V instead, which is
    // per-frame equivalent and skips the recompute.
    ggml_tensor* k_full = nullptr;
    ggml_tensor* v_full = nullptr;
    if (cache->k_cache_cur != nullptr && cache_len > 0) {
        // k_chunk and the gathered cache are [d_model,time,B]. Concatenate the
        // stream-local histories on the time axis (output is contiguous).
        auto k_c3 =
            ggml_reshape_3d(bf_ctx.ctx, cache->k_cache_cur, n_feat, cache_len, x.tensor->ne[2]);
        auto v_c3 =
            ggml_reshape_3d(bf_ctx.ctx, cache->v_cache_cur, n_feat, cache_len, x.tensor->ne[2]);
        k_full = ggml_concat(bf_ctx.ctx, k_c3, k_chunk, 1);
        v_full = ggml_concat(bf_ctx.ctx, v_c3, v_chunk, 1);
    } else {
        // No cache prepend: materialize the strided slices so the multi-head
        // reshapes below see contiguous tensors.
        k_full = ggml_cont(bf_ctx.ctx, k_chunk);
        v_full = ggml_cont(bf_ctx.ctx, v_chunk);
    }
    const int64_t kv_len = cache_len + chunk_len;

    // Emit updated K/V caches: last `left_context` frames of the post-concat
    // K/V buffer. (Note that the host-side cache size is always
    // cache_left_ctx; on the first few chunks we still write left_context
    // frames - internal zero-prefixed positions get masked off by attn_mask.)
    const int64_t left_ctx = cache->left_context;
    const int64_t cache_keep_len = std::min<int64_t>(kv_len, left_ctx);
    const int64_t cache_keep_offset = kv_len - cache_keep_len;
    if (cache_keep_offset >= 0) {
        // Slice last cache_keep_len rows from k_full / v_full and pad-front
        // back to left_ctx so the host buffer shape stays constant.
        auto k_tail = ggml_view_3d(
            bf_ctx.ctx, k_full, n_feat, cache_keep_len, k_full->ne[2], k_full->nb[1], k_full->nb[2],
            cache_keep_offset * k_full->nb[1]);
        auto v_tail = ggml_view_3d(
            bf_ctx.ctx, v_full, n_feat, cache_keep_len, v_full->ne[2], v_full->nb[1], v_full->nb[2],
            cache_keep_offset * v_full->nb[1]);
        if (cache_keep_len < left_ctx) {
            // Left-pad with zeros so the cache slot's shape (d_model, left_ctx)
            // matches what the encoder graph allocated.
            const int64_t pad_left = left_ctx - cache_keep_len;
            cache->k_cache_next =
                pad_ext_portable(bf_ctx.ctx, k_tail, 0, 0, pad_left, 0, 0, 0, 0, 0);
            cache->v_cache_next =
                pad_ext_portable(bf_ctx.ctx, v_tail, 0, 0, pad_left, 0, 0, 0, 0, 0);
        } else {
            // Raw strided views; the encoder's cache feedback either copies
            // them straight into the arena row (single-slot) or materializes
            // them for the indexed scatter (multi-slot).
            cache->k_cache_next = k_tail;
            cache->v_cache_next = v_tail;
        }
    }

    // Positional projection: precomputed once per geometry (chunk-invariant)
    // as [d_k, kv+chunk-1, n_head] — see build_precompute_graph. Already
    // head-split, so it slots in as p_p / the fused kernel's P directly.
    GGML_ASSERT(cache->pos_proj != nullptr && "cached MHA requires precomputed pos_proj");
    ggml_tensor* p_p = cache->pos_proj;

    const int64_t batch = x.tensor->ne[2];
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    ggml_tensor* ctx_attn = nullptr;
#ifdef NEMO_SPEECH_FUSED_RELPOS_ATTN
    // Single-kernel path: bias adds, content+position scores, the rel-shift,
    // scale+mask, softmax and the attn@V context all happen inside
    // ggml_fused_relpos_attn (the same op the offline encoder uses; its
    // rel-shift indexing row=(q-1)+j-i is exactly this strip shift). The op
    // is stride-general, so Q reads directly from the fused-QKV output and
    // K/V head-split from the feat-major concat windows — no staging copies —
    // and merge_heads gives the (n_feat, chunk) context as a free view. Every
    // batch size takes this path (the mask carries one column per stream).
    // Singleton and batched graphs must remain numerically equivalent.
    // Runtime-gated like the offline fused path: the fused op is CUDA-only
    // (CPU supports_op is false), so a CPU session in a CUDA build must take
    // the GEMM formulation below for all batch sizes — equivalence holds
    // per-device.
    const bool mask_fusable = session->params.use_gpu && cache->attn_mask != nullptr;
    if (mask_fusable && (d_k & (d_k - 1)) == 0) {
        auto q_heads = ggml_view_4d(
            bf_ctx.ctx, qkv, d_k, chunk_len, n_head, batch, qkv->nb[1],
            static_cast<size_t>(d_k) * qkv_es, qkv->nb[2], 0);
        auto k_heads = ggml_view_4d(
            bf_ctx.ctx, k_full, d_k, kv_len, n_head, batch, k_full->nb[1],
            static_cast<size_t>(d_k) * ggml_element_size(k_full), k_full->nb[2], 0);
        auto v_heads = ggml_view_4d(
            bf_ctx.ctx, v_full, d_k, kv_len, n_head, batch, v_full->nb[1],
            static_cast<size_t>(d_k) * ggml_element_size(v_full), v_full->nb[2], 0);
        // (kv_len, B): one additive key-mask column per stream.
        auto mask_2d = ggml_reshape_2d(bf_ctx.ctx, cache->attn_mask, kv_len, batch);
        auto ctx_heads = ggml_fused_relpos_attn(
            bf_ctx.ctx, q_heads, k_heads, v_heads, p_p, bias_u.tensor, bias_v.tensor, mask_2d,
            scale, /*merge_heads=*/true);
        // Head-merged layout: this permute view is contiguous (n_feat, chunk).
        ctx_attn = ggml_reshape_3d(
            bf_ctx.ctx, ggml_permute(bf_ctx.ctx, ctx_heads, 0, 2, 1, 3), n_feat, chunk_len, batch);
    }
    if (ctx_attn == nullptr)
#endif
    {
        // Q in head-split layout [d_k, chunk_len, n_head, batch]: carve heads
        // straight out of the fused QKV output (one transpose copy total).
        auto q_heads = ggml_view_4d(
            bf_ctx.ctx, qkv, d_k, n_head, chunk_len, batch, static_cast<size_t>(d_k) * qkv_es,
            qkv->nb[1], qkv->nb[2], 0);
        auto q_p = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, q_heads, 0, 2, 1, 3));

        // Reshape K/V to multi-head and bring the head dim out.
        auto k_mh = ggml_reshape_4d(bf_ctx.ctx, k_full, d_k, n_head, kv_len, batch);
        auto v_mh = ggml_reshape_4d(bf_ctx.ctx, v_full, d_k, n_head, kv_len, batch);
        auto k_p = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, k_mh, 0, 2, 1, 3));
        auto v_p = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, v_mh, 0, 2, 1, 3));

        // Add pos biases to Q.
        auto bu_4d = ggml_reshape_4d(bf_ctx.ctx, bias_u.tensor, d_k, 1, n_head, 1);
        auto bv_4d = ggml_reshape_4d(bf_ctx.ctx, bias_v.tensor, d_k, 1, n_head, 1);
        auto qu = ggml_add(bf_ctx.ctx, q_p, bu_4d);
        auto qv = ggml_add(bf_ctx.ctx, q_p, bv_4d);

        // Content attention (NeMo's matrix_ac): K @ (q + pos_bias_u)
        //   -> (kv_len, chunk_len, n_head, 1)
        auto matrix_ac = ggml_mul_mat(bf_ctx.ctx, k_p, qu);

        // Position attention (NeMo's matrix_bd, pre-shift): p @ (q + pos_bias_v)
        //   -> (pos_len, chunk_len, n_head, batch).
        // p_p broadcasts its singleton batch dim over qv's.
        auto matrix_bd_raw = ggml_mul_mat(bf_ctx.ctx, p_p, qv);

        // Transformer-XL relative shift, NeMo reference implementation:
        // RelPositionMultiHeadAttention.rel_shift (pad-left by one, fold, drop
        // first row) followed by forward()'s truncation of matrix_bd to
        // matrix_ac's key length (both in NeMo multi_head_attention.py); the
        // two steps are fused here. In ggml (ne[0] = fastest dim) that becomes:
        //   1. pad-left on dim 0 by 1               -> (pos_len+1, chunk_len, n_head, B)
        //   2. reshape (swap ne[0], ne[1])          -> (chunk_len, pos_len+1, n_head, B)
        //   3. drop first "row" (offset = chunk_len)-> (chunk_len, pos_len, n_head, B)
        //   4. reshape back                         -> (pos_len, chunk_len, n_head, B)
        //   5. clip to (kv_len, chunk_len, n_head, B)
        // The arithmetic only needs pos_len >= kv_len + chunk_len - 1, so the
        // precomputed strip (sized for the configured chunk) also serves
        // shorter tail chunks.
        auto bd_pad = pad_ext_portable(bf_ctx.ctx, matrix_bd_raw, 1, 0, 0, 0, 0, 0, 0, 0);
        // step 2: swap dims via reshape (only safe because bd_pad is contiguous
        // after the pad). pos_len+1 = bd_pad->ne[0]; chunk_len = bd_pad->ne[1].
        auto bd_pad_c = ggml_cont(bf_ctx.ctx, bd_pad);
        auto bd_reshape =
            ggml_reshape_4d(bf_ctx.ctx, bd_pad_c, chunk_len, bd_pad_c->ne[0], n_head, batch);
        // step 3: drop first row by offsetting chunk_len elements into ne[1] axis.
        auto bd_drop = ggml_view_4d(
            bf_ctx.ctx, bd_reshape, chunk_len, bd_reshape->ne[1] - 1, n_head, batch,
            bd_reshape->nb[1], bd_reshape->nb[2], bd_reshape->nb[3],
            chunk_len * ggml_element_size(bd_reshape));
        auto bd_drop_c = ggml_cont(bf_ctx.ctx, bd_drop);
        // step 4: reshape back to (pos_len, chunk_len, n_head, batch).
        auto bd_back =
            ggml_reshape_4d(bf_ctx.ctx, bd_drop_c, bd_drop_c->ne[1], chunk_len, n_head, batch);
        // step 5: clip to (kv_len, chunk_len, n_head, batch).
        auto bd_clip = ggml_cont(
            bf_ctx.ctx, ggml_view_4d(
                            bf_ctx.ctx, bd_back, kv_len, chunk_len, n_head, batch, bd_back->nb[1],
                            bd_back->nb[2], bd_back->nb[3], 0));

        ggml_tensor* scores = ggml_add(bf_ctx.ctx, matrix_ac, bd_clip);
        scores = ggml_scale_inplace(bf_ctx.ctx, scores, scale);
        // Apply attention mask: (kv_len, 1) broadcast across chunk_len, n_head, batch.
        if (cache->attn_mask != nullptr) {
            auto mask_bcast = ggml_reshape_4d(bf_ctx.ctx, cache->attn_mask, kv_len, 1, 1, batch);
            scores = ggml_add(bf_ctx.ctx, scores, mask_bcast);
        }
        auto attn = ggml_soft_max_inplace(bf_ctx.ctx, scores);

        // ctx = v^T @ attn -> permute v_p to (kv_len, d_k, n_head, 1) then matmul.
        auto v_pt = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, v_p, 1, 0, 2, 3));
        ctx_attn = ggml_mul_mat(bf_ctx.ctx, v_pt, attn);  // (d_k, chunk_len, n_head, 1)
        ctx_attn = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, ctx_attn, 0, 2, 1, 3));
        ctx_attn = ggml_reshape_3d(bf_ctx.ctx, ctx_attn, n_feat, chunk_len, batch);
    }

    auto out = add_bias(ggml_mul_mat(bf_ctx.ctx, wo.tensor, ctx_attn), bo);

    ggml_runtime::TensorBag out_bag;
    out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(out, x.buft));
    return out_bag;
}

void
ConformerLayer::set_data(ggml_runtime::Session* session) {
    norm_feed_forward1_->set_data(session);
    feed_forward1_->set_data(session);
    norm_self_att_->set_data(session);
    self_attn_->set_data(session);
    norm_conv_->set_data(session);
    conv_->set_data(session);
    norm_feed_forward2_->set_data(session);
    feed_forward2_->set_data(session);
    norm_out_->set_data(session);
}

ggml_runtime::Session::WeightLoadHook
nemo_speech::asr::planar_q8_weight_load_hook() {
    return [](const std::string& gguf_key, ggml_runtime::ggml_bf_tensor& t,
              ggml_runtime::GGUFLoader* loader) {
        if (loader == nullptr || t.tensor->type != GGML_TYPE_Q8_0 ||
            gguf_key.rfind("encoder.", 0) != 0 ||
            loader->get_str("asr.encoder.q8_layout") != "tensor_planar_v1") {
            return;
        }
        const char* buft_name = ggml_backend_buft_name(t.buft);
        if (buft_name == nullptr || std::string(buft_name).rfind("CUDA", 0) != 0) {
            throw std::runtime_error(
                "load_weight(" + gguf_key + "): tensor_planar_v1 Q8 weights require a CUDA buffer");
        }
#ifdef NEMO_SPEECH_GGML_PATCHED
        t.tensor->flags |= GGML_TENSOR_FLAG_Q8_PLANAR;
#else
        throw std::runtime_error("tensor-planar Q8 weights require NEMO_SPEECH_GGML_PATCHED");
#endif
    };
}

FastConformerEncoder::FastConformerEncoder(const std::string& name, const EncoderConfig& cfg)
    : name_(name), cfg_(cfg), layers_(nullptr) {
    pre_encode_ = new SubSampling("encoder.pre_encode", cfg);
    pos_enc_ = new ggml_runtime::RelPositionalEncoding(
        "encoder.pos_enc", cfg.d_model, cfg.pos_emb_max_len);
    if (cfg.cache_mode == CacheMode::Disabled) {
        layers_ = new ggml_runtime::SequenceModule("encoder.layers");
        for (int i = 0; i < cfg.n_layers; i++) {
            layers_->modules.push_back(
                new ConformerLayer("encoder.layers." + std::to_string(i), cfg));
        }
    } else {
        layer_ptrs_.reserve(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; i++) {
            layer_ptrs_.push_back(new ConformerLayer("encoder.layers." + std::to_string(i), cfg));
        }
    }
}

FastConformerEncoder::~FastConformerEncoder() {
    delete pre_encode_;
    delete pos_enc_;
    delete layers_;
    for (auto* p : layer_ptrs_) delete p;
}


std::string
FastConformerEncoder::kv_cache_name(int l) const {
    return name_ + ".cache.kv." + std::to_string(l);
}
std::string
FastConformerEncoder::conv_cache_name(int l) const {
    return name_ + ".cache.conv." + std::to_string(l);
}
std::string
FastConformerEncoder::attn_mask_name() const {
    return name_ + ".cache.attn_mask";
}
std::string
FastConformerEncoder::k_cache_out_name(int l) const {
    return name_ + ".cache.k_out." + std::to_string(l);
}
std::string
FastConformerEncoder::v_cache_out_name(int l) const {
    return name_ + ".cache.v_out." + std::to_string(l);
}
std::string
FastConformerEncoder::conv_cache_out_name(int l) const {
    return name_ + ".cache.conv_out." + std::to_string(l);
}
std::string
FastConformerEncoder::pos_proj_name(int l) const {
    return name_ + ".cache.pos_proj." + std::to_string(l);
}
std::string
FastConformerEncoder::dw_ct_name(int l) const {
    return name_ + ".cache.dw_ct." + std::to_string(l);
}

// One-shot graph filling the chunk-invariant per-layer tensors: the
// positional projections (fixed cached pos_emb window through each layer's
// linear_pos, stored head-split) and the channel-inner repack of each
// depthwise-conv weight for the cwhn direct kernel. Run once by
// CacheAwareEncoder::ensure_session (dispatched by the kPrecomputeTrigger
// input name) before any chunk graph is built, so chunk graphs can treat
// the tensors as filled constants.
ggml_runtime::TensorBag
FastConformerEncoder::build_precompute_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorContainer* session_tensor_container) {
    auto pe_tensor = session->model_tensor_container->get_tensor_by_name(name_ + ".pos_enc.pe");
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(pe_tensor.buft);
    const int d_k = cfg_.d_model / cfg_.n_heads;
    const int chunk_frames = cfg_.cache_chunk_frames;
    const int kv_len = cfg_.cache_left_ctx + chunk_frames;
    // The PE window the cache-aware attention geometry addresses (kv keys x
    // chunk queries); constant per config, which is what makes this a
    // one-shot precompute.
    const int slice_len = 2 * kv_len - 1;
    const int start_pos = cfg_.pos_emb_max_len - kv_len;
    const size_t offset = static_cast<size_t>(start_pos) * pe_tensor.tensor->nb[1];
    auto pos_emb_slice = ggml_view_2d(
        bf_ctx.ctx, pe_tensor.tensor, pe_tensor.tensor->ne[0], slice_len, pe_tensor.tensor->nb[1],
        offset);

    ggml_runtime::TensorBag out;
    for (int l = 0; l < cfg_.n_layers; l++) {
        const std::string wp_name =
            name_ + ".layers." + std::to_string(l) + ".self_attn.linear_pos.weight";
        auto wp = session->model_tensor_container->get_tensor_by_name(wp_name);
        auto dst = session->model_tensor_container->get_tensor_by_name(pos_proj_name(l));
        // [d_model, slice_len] -> heads [d_k, head, slice] -> [d_k, slice, head],
        // keeping only the rows the rel-shift can address for q <= chunk_frames.
        auto pos = ggml_mul_mat(bf_ctx.ctx, wp.tensor, pos_emb_slice);
        auto pos_mh = ggml_reshape_3d(bf_ctx.ctx, pos, d_k, cfg_.n_heads, slice_len);
        auto pos_hs = ggml_permute(bf_ctx.ctx, pos_mh, 0, 2, 1, 3);
        auto pos_rows = ggml_view_3d(
            bf_ctx.ctx, pos_hs, d_k, kv_len + chunk_frames - 1, cfg_.n_heads, pos_hs->nb[1],
            pos_hs->nb[2], 0);
        auto store = ggml_cpy(bf_ctx.ctx, pos_rows, dst.tensor);
        out.add_tensor(ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }

    // Depthwise-conv weight repack: (k, 1, C) k-inner -> (C, k) channel-inner,
    // the memory order the cwhn direct kernel expects (w[j*C + c]).
    for (int l = 0; l < cfg_.n_layers; l++) {
        const std::string w_name =
            name_ + ".layers." + std::to_string(l) + ".conv.depthwise_conv.weight";
        auto w = session->model_tensor_container->get_tensor_by_name(w_name);
        auto dst = session->model_tensor_container->get_tensor_by_name(dw_ct_name(l));
        auto w2 =
            ggml_reshape_2d(bf_ctx.ctx, w.tensor, cfg_.conv_kernel_size, cfg_.d_model);  // (k, C)
        auto w_ct = ggml_permute(bf_ctx.ctx, w2, 1, 0, 2, 3);  // (C, k) view
        auto store = ggml_cpy(bf_ctx.ctx, w_ct, dst.tensor);
        out.add_tensor(ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }
    return out;
}

void
FastConformerEncoder::define_tensors(ggml_runtime::Session* session) {
    pre_encode_->define_tensors(session);
    pos_enc_->define_tensors(session);
    if (layers_ != nullptr) {
        layers_->define_tensors(session);
    } else {
        for (auto* p : layer_ptrs_) p->define_tensors(session);

        // Pre-allocate the per-layer cache slots and the global attention
        // mask as input tensors (the runner sets them each step). We use
        // create_tensor_2d for K/V/conv and 1d for the mask. They live in
        // model_tensor_container because tensor lifetime must outlast the
        // graph build (the inputs are read every chunk).
        const int d_model = cfg_.d_model;
        const int cache_len = cfg_.cache_left_ctx;
        const int conv_cache_len = cfg_.conv_kernel_size - 1;
        // Persistent slot arenas. K and V have identical row geometry, so keep
        // them as two planes of one arena. A batch can then gather/scatter both
        // planes with one indexed CUDA launch while retaining independent views
        // inside each layer.
        for (int l = 0; l < cfg_.n_layers; l++) {
            session->model_tensor_container->create_tensor_3d(
                kv_cache_name(l), GGML_TYPE_F32, d_model * cache_len, cfg_.cache_state_slots, 2);
            session->model_tensor_container->create_tensor_2d(
                conv_cache_name(l), GGML_TYPE_F32, d_model * conv_cache_len,
                cfg_.cache_state_slots);
        }

        // Chunk-invariant positional projections, one per layer, filled once
        // by build_precompute_graph. Head-split layout [d_k, rows, head]
        // with rows = kv + chunk - 1: exactly the rel-shift rows any chunk of
        // length <= cache_chunk_frames reads (row (q-1)+j-i <= kv+q-2).
        const int d_k = cfg_.d_model / cfg_.n_heads;
        const int kv_len = cfg_.cache_left_ctx + cfg_.cache_chunk_frames;
        const int pos_rows = kv_len + cfg_.cache_chunk_frames - 1;
        // F32: the fused attention kernel requires K/V/P in one type, and the
        // strided fused path reads K/V as F32 views of the concat windows.
        for (int l = 0; l < cfg_.n_layers; l++) {
            session->model_tensor_container->create_tensor_3d(
                pos_proj_name(l), GGML_TYPE_F32, d_k, pos_rows, cfg_.n_heads);
        }

        // Channel-inner depthwise-conv weights for the cwhn direct kernel
        // ((d_model, T)-layout conv module). Same dtype as the source weight;
        // filled by build_precompute_graph.
        for (int l = 0; l < cfg_.n_layers; l++) {
            const std::string src =
                name_ + ".layers." + std::to_string(l) + ".conv.depthwise_conv.weight";
            session->model_tensor_container->create_tensor_2d(
                dw_ct_name(l), session->gguf_loader->get_tensor_type(src), cfg_.d_model,
                cfg_.conv_kernel_size);
        }
    }
}

ggml_runtime::TensorBag
FastConformerEncoder::build_pre_encode(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    return pre_encode_->build_graph(session, input_tensors, session_tensor_container);
}

ggml_runtime::TensorBag
FastConformerEncoder::build_graph_from_embeddings(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    GGML_ASSERT(cfg_.cache_mode == CacheMode::Disabled);
    auto pre = input_tensors;

    // xscaling: NeMo's PositionalEncoding.forward does x = x * sqrt(d_model).
    if (cfg_.xscaling) {
        auto x = pre.get_tensor(0);
        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
        auto scale = std::sqrt(static_cast<float>(cfg_.d_model));
        auto x_scaled = ggml_scale(bf_ctx.ctx, x.tensor, scale);
        pre.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_scaled, x.buft));
    }

    {
        auto pe_out = pos_enc_->build_graph(session, pre, session_tensor_container);

        // Every layer projects the same positional embedding. With BF16
        // weights, round that shared input once rather than letting each of
        // the 42 cuBLAS calls repeat the identical F32-to-BF16 conversion.
        const std::string pos_weight0 = name_ + ".layers.0.self_attn.linear_pos.weight";
        if (session->params.use_gpu &&
            session->gguf_loader->get_tensor_type(pos_weight0) == GGML_TYPE_BF16) {
            auto x = pe_out.get_tensor(0);
            auto pos = pe_out.get_tensor(1);
            auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(pos.buft);
            auto pos_bf16 = ggml_cast(bf_ctx.ctx, pos.tensor, GGML_TYPE_BF16);
            ggml_runtime::TensorBag shared_pe;
            shared_pe.add_tensor(x);
            shared_pe.add_tensor(ggml_runtime::ggml_bf_tensor(pos_bf16, pos.buft));
            pe_out = shared_pe;
        }

        // Offline attention mask: only built when the model asked for a
        // limited attention window via `offline_{left,right}_ctx`. Existing
        // GGUFs leave these at the -1 sentinel → no mask is built → existing
        // bidirectional behaviour unchanged.
        if (cfg_.offline_left_ctx >= 0 || cfg_.offline_right_ctx >= 0) {
            auto x = pe_out.get_tensor(0);
            const int T = static_cast<int>(x.tensor->ne[1]);
            const int L = (cfg_.offline_left_ctx < 0) ? T : cfg_.offline_left_ctx;
            const int R = (cfg_.offline_right_ctx < 0) ? T : cfg_.offline_right_ctx;
            auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);

            // pos = arange(T): (T,) f32.
            ggml_tensor* pos = ggml_arange(bf_ctx.ctx, 0.0f, static_cast<float>(T), 1.0f);
            // Two interpretations of `pos`: (T, 1) gives the key position
            // along ne[0], and (1, T) the query position along ne[1].
            if (cfg_.chunked_limited_attention && R >= 0) {
                // NeMo chunked_limited attention is a window over CHUNKS, not
                // a per-frame [q-L,q+R] band. A query can see its complete
                // R+1 frame chunk plus L/(R+1) preceding chunks.
                pos = ggml_floor(
                    bf_ctx.ctx, ggml_scale(bf_ctx.ctx, pos, 1.0f / static_cast<float>(R + 1)));
            }
            ggml_tensor* pos_j_col = ggml_reshape_2d(bf_ctx.ctx, pos, T, 1);
            ggml_tensor* pos_i_row = ggml_reshape_2d(bf_ctx.ctx, pos, 1, T);
            ggml_tensor* target = ggml_new_tensor_2d(bf_ctx.ctx, GGML_TYPE_F32, T, T);
            ggml_tensor* pos_j = ggml_repeat(bf_ctx.ctx, pos_j_col, target);
            ggml_tensor* pos_i = ggml_repeat(bf_ctx.ctx, pos_i_row, target);
            ggml_tensor* left_pen = nullptr;
            ggml_tensor* right_pen = nullptr;
            if (cfg_.chunked_limited_attention && R >= 0) {
                // chunk_diff = query_chunk - key_chunk. Reject future chunks
                // (diff < 0) and chunks farther left than left_chunks.
                ggml_tensor* chunk_diff = ggml_sub(bf_ctx.ctx, pos_i, pos_j);
                const int left_chunks = L / (R + 1);
                left_pen = ggml_relu(
                    bf_ctx.ctx,
                    ggml_scale_bias(
                        bf_ctx.ctx, chunk_diff, 1.0f, -static_cast<float>(left_chunks)));
                right_pen = ggml_relu(bf_ctx.ctx, ggml_scale(bf_ctx.ctx, chunk_diff, -1.0f));
            } else {
                // Regular local attention: diff=key-query; reject keys left
                // of q-L or right of q+R.
                ggml_tensor* diff = ggml_sub(bf_ctx.ctx, pos_j, pos_i);
                left_pen = ggml_relu(
                    bf_ctx.ctx, ggml_scale_bias(bf_ctx.ctx, diff, -1.0f, -static_cast<float>(L)));
                right_pen = ggml_relu(
                    bf_ctx.ctx, ggml_scale_bias(bf_ctx.ctx, diff, 1.0f, -static_cast<float>(R)));
            }
            ggml_tensor* penalty = ggml_add(bf_ctx.ctx, left_pen, right_pen);
            // Additive mask: -1e9 wherever penalty > 0, else 0.
            ggml_tensor* mask = ggml_scale(bf_ctx.ctx, ggml_step(bf_ctx.ctx, penalty), -1e9f);
            // Reshape to (T_kv, T_q, 1, 1) so it broadcasts across n_head & batch.
            mask = ggml_reshape_4d(bf_ctx.ctx, mask, T, T, 1, 1);
            pe_out.add_tensor(ggml_runtime::ggml_bf_tensor(mask, bf_ctx.buft));
        }

        // Conformer layers (non-cached).
        auto enc_out = layers_->build_graph(session, pe_out, session_tensor_container);
        ggml_runtime::TensorBag out;
        out.add_tensor(enc_out.get_tensor(0));
        return out;
    }
}

ggml_runtime::TensorBag
FastConformerEncoder::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* session_tensor_container) {
    // 1. Subsampling: (feat_in, T_audio, 1, 1) -> (d_model, T_sub, 1, 1)
    auto pre = pre_encode_->build_graph(session, input_tensors, session_tensor_container);

    // 2. Offline: xscaling + positional encoding + conformer stack.
    if (cfg_.cache_mode == CacheMode::Disabled) {
        return build_graph_from_embeddings(session, pre, session_tensor_container);
    }

    // Cache-aware path. xscaling first (NeMo's PositionalEncoding.forward
    // does x = x * sqrt(d_model)).
    if (cfg_.xscaling) {
        auto x = pre.get_tensor(0);
        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
        auto scale = std::sqrt(static_cast<float>(cfg_.d_model));
        auto x_scaled = ggml_scale(bf_ctx.ctx, x.tensor, scale);
        pre.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_scaled, x.buft));
    }

    // Then drop the leading `cache_drop_extra` frames from the subsampled
    // output so the visible chunk length is exactly cache_chunk_frames
    // (= 1 + R). The dropped frames overlap with the tail of the previous
    // chunk (handled by the K/V cache).
    if (cfg_.cache_drop_extra > 0) {
        auto x = pre.get_tensor(0);
        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
        auto x_view = ggml_view_4d(
            bf_ctx.ctx, x.tensor, x.tensor->ne[0], x.tensor->ne[1] - cfg_.cache_drop_extra,
            x.tensor->ne[2], x.tensor->ne[3], x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3],
            cfg_.cache_drop_extra * x.tensor->nb[1]);
        pre.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_view, x.buft));
    }

    // Cache-aware layers consume the precomputed per-layer positional projection.
    ggml_runtime::TensorBag x_bag = pre;
    auto mask_t = input_tensors.get_tensor(1);
    auto slot_ids_kv = input_tensors.get_tensor(2);  // [B,2], identical K/V columns
    const int64_t batch = slot_ids_kv.tensor->ne[0];
    auto slot_ids = ggml_runtime::ggml_bf_tensor(
        ggml_view_1d(
            session_tensor_container->get_ctx_of_buffer_type(slot_ids_kv.buft).ctx,
            slot_ids_kv.tensor, batch, 0),
        slot_ids_kv.buft);
    // With a single state slot the slot id is statically 0, so per-layer
    // gathers degenerate to arena views and feedback uses in-place copies.
    // Multi-slot arenas retain indexed gather/scatter.
    const bool single_slot = cfg_.cache_state_slots == 1 && batch == 1;
    for (int l = 0; l < cfg_.n_layers; l++) {
        LayerCacheIO cache;
        cache.left_context = cfg_.cache_left_ctx;
        cache.right_context = cfg_.cache_right_ctx;
        auto kv_arena = session->model_tensor_container->get_tensor_by_name(kv_cache_name(l));
        auto c_arena = session->model_tensor_container->get_tensor_by_name(conv_cache_name(l));
        auto bf = session_tensor_container->get_ctx_of_buffer_type(kv_arena.buft);
        auto cache_cur = [&](ggml_tensor* arena, int64_t frames) {
            if (single_slot) {
                return ggml_reshape_3d(bf.ctx, arena, cfg_.d_model, frames, batch);
            }
            return ggml_reshape_3d(
                bf.ctx, ggml_get_rows(bf.ctx, arena, slot_ids.tensor), cfg_.d_model, frames, batch);
        };
        const int64_t kv_row = cfg_.d_model * cfg_.cache_left_ctx;
        if (single_slot) {
            auto k_plane =
                ggml_view_2d(bf.ctx, kv_arena.tensor, kv_row, 1, kv_arena.tensor->nb[1], 0);
            auto v_plane = ggml_view_2d(
                bf.ctx, kv_arena.tensor, kv_row, 1, kv_arena.tensor->nb[1], kv_arena.tensor->nb[2]);
            cache.k_cache_cur =
                ggml_reshape_3d(bf.ctx, k_plane, cfg_.d_model, cfg_.cache_left_ctx, batch);
            cache.v_cache_cur =
                ggml_reshape_3d(bf.ctx, v_plane, cfg_.d_model, cfg_.cache_left_ctx, batch);
        } else {
            auto kv_rows = ggml_get_rows(bf.ctx, kv_arena.tensor, slot_ids_kv.tensor);
            auto k_plane =
                ggml_view_3d(bf.ctx, kv_rows, kv_row, batch, 1, kv_rows->nb[1], kv_rows->nb[2], 0);
            auto v_plane = ggml_view_3d(
                bf.ctx, kv_rows, kv_row, batch, 1, kv_rows->nb[1], kv_rows->nb[2], kv_rows->nb[2]);
            cache.k_cache_cur =
                ggml_reshape_3d(bf.ctx, k_plane, cfg_.d_model, cfg_.cache_left_ctx, batch);
            cache.v_cache_cur =
                ggml_reshape_3d(bf.ctx, v_plane, cfg_.d_model, cfg_.cache_left_ctx, batch);
        }
        cache.conv_cache_cur = cache_cur(c_arena.tensor, cfg_.conv_kernel_size - 1);
        cache.attn_mask = mask_t.tensor;
        cache.pos_proj =
            session->model_tensor_container->get_tensor_by_name(pos_proj_name(l)).tensor;
#ifdef NEMO_SPEECH_DIRECT_DW_CONV
        // Repacked channel-inner dw weight enables the (d_model, T)-layout
        // conv module; the cwhn direct kernel is CUDA-only (patch 0004), so a
        // CPU session in a CUDA build leaves it null and the layer takes the
        // portable transpose-based conv path (runtime-gated in nn.cpp).
        if (session->params.use_gpu) {
            cache.dw_conv_w_ct =
                session->model_tensor_container->get_tensor_by_name(dw_ct_name(l)).tensor;
        }
#endif

        x_bag = layer_ptrs_[l]->build_graph(session, x_bag, session_tensor_container, &cache);

        // Cache outputs are compute-graph nodes, so register them in the
        // per-call container. The feedback branches below consume these names
        // and scatter them into the persistent device-resident arenas during
        // the same graph execution; no cache state is copied back to the host.
        if (cache.k_cache_next != nullptr) {
            session_tensor_container->cache_tensor(
                k_cache_out_name(l), ggml_runtime::ggml_bf_tensor(cache.k_cache_next, mask_t.buft));
        }
        if (cache.v_cache_next != nullptr) {
            session_tensor_container->cache_tensor(
                v_cache_out_name(l), ggml_runtime::ggml_bf_tensor(cache.v_cache_next, mask_t.buft));
        }
        if (cache.conv_cache_next != nullptr) {
            session_tensor_container->cache_tensor(
                conv_cache_out_name(l),
                ggml_runtime::ggml_bf_tensor(cache.conv_cache_next, mask_t.buft));
        }
    }

    ggml_runtime::TensorBag out;
    out.add_tensor(x_bag.get_tensor(0));
    // Device-resident cache feedback keeps state on the GPU between chunks.
    // Ordering is safe without extra synchronization: each feedback source
    // depends on every read of the corresponding current cache, and graph
    // nodes execute in expansion order. These side branches are not reachable
    // backward from the encoder output, so their terminal copy/set nodes must
    // be in the output bag for ggml_build_forward_expand to reach them.
    for (int l = 0; l < cfg_.n_layers; l++) {
        if (session_tensor_container->has_tensor_by_name(k_cache_out_name(l)) &&
            session_tensor_container->has_tensor_by_name(v_cache_out_name(l))) {
            auto ko = session_tensor_container->get_tensor_by_name(k_cache_out_name(l));
            auto vo = session_tensor_container->get_tensor_by_name(v_cache_out_name(l));
            auto kv_arena = session->model_tensor_container->get_tensor_by_name(kv_cache_name(l));
            auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(ko.buft);
            // One non-contiguous concat materializes both cache tails as
            // [d_model,left,B,2]; reshape to arena rows and scatter both planes
            // in one SET_ROWS operation.
            auto kv_next = ggml_concat(bf_ctx.ctx, ko.tensor, vo.tensor, 3);
            kv_next = ggml_reshape_3d(bf_ctx.ctx, kv_next, kv_arena.tensor->ne[0], batch, 2);
            auto set = single_slot ? ggml_cpy(bf_ctx.ctx, kv_next, kv_arena.tensor)
                                   : ggml_set_rows(
                                         bf_ctx.ctx, kv_arena.tensor, kv_next, slot_ids_kv.tensor);
            out.add_tensor(ggml_runtime::ggml_bf_tensor(set, ko.buft));
        }

        auto feedback = [&](const std::string& out_name, const std::string& in_name) {
            // Only registered when the layer produced this cache output (same
            // null guard as the cache_tensor() calls above); skip otherwise so
            // get_tensor_by_name can't throw on an unregistered name.
            if (!session_tensor_container->has_tensor_by_name(out_name))
                return;
            auto co = session_tensor_container->get_tensor_by_name(out_name);
            auto ci = session->model_tensor_container->get_tensor_by_name(in_name);
            auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(co.buft);
            if (single_slot) {
                // Slot 0 is the whole arena: one strided in-place copy replaces
                // the materializing cont + indexed scatter. Expansion order
                // keeps it after every read of the arena, same as set_rows.
                auto set = ggml_cpy(bf_ctx.ctx, co.tensor, ci.tensor);
                out.add_tensor(ggml_runtime::ggml_bf_tensor(set, co.buft));
                return;
            }
            auto flat = ggml_reshape_2d(
                bf_ctx.ctx, ggml_cont(bf_ctx.ctx, co.tensor), ci.tensor->ne[0], batch);
            auto set = ggml_set_rows(bf_ctx.ctx, ci.tensor, flat, slot_ids.tensor);
            out.add_tensor(ggml_runtime::ggml_bf_tensor(set, co.buft));
        };
        feedback(conv_cache_out_name(l), conv_cache_name(l));
    }
    return out;
}

void
FastConformerEncoder::set_data(ggml_runtime::Session* session) {
    pre_encode_->set_data(session);
    pos_enc_->set_data(session);
    if (layers_ != nullptr) {
        layers_->set_data(session);
    } else {
        for (auto* p : layer_ptrs_) p->set_data(session);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            for (const auto& name : {kv_cache_name(l), conv_cache_name(l)}) {
                auto t = session->model_tensor_container->get_tensor_by_name(name);
                ggml_backend_tensor_memset(t.tensor, 0, 0, ggml_nbytes(t.tensor));
            }
        }
    }
}
