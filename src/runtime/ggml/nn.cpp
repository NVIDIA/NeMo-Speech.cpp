// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include "nn.h"

#include <math.h>

#include <cstdlib>

#include "runtime.h"

namespace ggml_runtime {

ggml_tensor*
cached_q8_input(
    Session* session, TensorContainer* session_tensor_container, const ggml_bf_tensor& weight,
    ggml_tensor* input) {
#ifdef NEMO_SPEECH_GGML_PATCHED
    static const bool enabled = [] {
        const char* value = std::getenv("GGML_SKINNY_Q8_CUBLAS_F16");
        return value != nullptr && value[0] != '0';
    }();
    static const int min_columns = [] {
        const char* value = std::getenv("GGML_SKINNY_Q8_CUBLAS_F16_MIN_N");
        const int parsed = value != nullptr ? std::atoi(value) : 128;
        return parsed > 0 ? parsed : 1;
    }();
    const int64_t columns = ggml_nelements(input) / input->ne[0];
    const bool planar = (weight.tensor->flags & GGML_TENSOR_FLAG_Q8_PLANAR) != 0;
    const bool eligible_block_q8 = std::string(weight.tensor->name).rfind("encoder.", 0) == 0 &&
                                   input->ne[1] > 8 && input->ne[1] <= 64;
    if (enabled && session->params.use_gpu && weight.tensor->type == GGML_TYPE_Q8_0 &&
        (planar || eligible_block_q8) && input->type == GGML_TYPE_F32 && columns >= min_columns) {
        const auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight.buft);
        return ggml_cast(bf_ctx.ctx, input, GGML_TYPE_F16);
    }
#else
    (void)session;
    (void)session_tensor_container;
    (void)weight;
#endif
    return input;
}

void
Conv1D::define_tensors(Session* session) {
    // The converter squeezes pointwise kernels to 2D so they can be quantized.
    ggml_type stored = session->gguf_loader->get_tensor_type(weight_name);
    int stored_n_dims = session->gguf_loader->get_tensor_n_dims(weight_name);
    is_pointwise_2d = (kernel_size == 1 && !is_dw && stored_n_dims == 2);

    if (kernel_size == 1 && !is_dw) {
        // 2D weight: store at the GGUF's dtype (Q8_0 / F16 / F32 / ...)
        // so ggml_mul_mat sees the original quant blocks.
        this->weight = session->model_tensor_container->create_tensor_2d(
            weight_name, stored, in_channels, out_channels);
    } else if (is_dw) {
        this->weight = session->model_tensor_container->create_tensor_3d(
            weight_name, GGML_TYPE_F16, kernel_size, 1, in_channels);
    } else {
        this->weight = session->model_tensor_container->create_tensor_3d(
            weight_name, GGML_TYPE_F16, kernel_size, in_channels, out_channels);
    }
    if (use_bias) {
        const int64_t bias_size = is_dw ? in_channels : out_channels;
        this->bias =
            session->model_tensor_container->create_tensor_1d(bias_name, GGML_TYPE_F32, bias_size);
    }
}

TensorBag
Conv1D::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);

    auto weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    ggml_tensor* out_tensor = nullptr;
    if (kernel_size == 1 && !is_dw) {
        // Pointwise (k=1) with quantized 2D weight (in, out). Caller's
        // input layout is (T, in, B); mul_mat needs (in, T, B) on the
        // right-hand side. Permute → mul_mat → permute back so the
        // output keeps the same (T, out, B) layout the caller expects
        // from ggml_conv_1d.
        auto x_in =
            ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, input_tensor.tensor, 1, 0, 2, 3));
        x_in = cached_q8_input(session, session_tensor_container, weight_tensor, x_in);
        // Older CTC GGUFs keep the k=1 conv as [1,in,out]; newer quantized
        // pointwise tensors are [in,out].  Both are byte-identical after
        // dropping the unit kernel dimension.
        auto matmul = ggml_mul_mat(bf_ctx.ctx, weight_tensor.tensor, x_in);
        out_tensor = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, matmul, 1, 0, 2, 3));
    } else if (is_dw) {
        bool direct_dw = false;
#ifdef NEMO_SPEECH_DIRECT_DW_CONV
        // Patch 0004 adds the F16 direct depthwise kernel only for CUDA; other
        // backends require the portable path below.
        direct_dw = session->params.use_gpu;
#endif
        if (direct_dw) {
            ggml_tensor* x4 = ggml_reshape_4d(
                bf_ctx.ctx, input_tensor.tensor, input_tensor.tensor->ne[0], 1,
                input_tensor.tensor->ne[1], input_tensor.tensor->ne[2]);
            ggml_tensor* w4 = ggml_reshape_4d(
                bf_ctx.ctx, weight_tensor.tensor, weight_tensor.tensor->ne[0], 1, 1,
                weight_tensor.tensor->ne[2]);
            ggml_tensor* y4 =
                ggml_conv_2d_dw_direct(bf_ctx.ctx, w4, x4, stride, 1, padding, 0, dilation, 1);
            // (OL, 1, C, B) -> (OL, C, B), matching ggml_conv_1d_dw's output.
            out_tensor = ggml_reshape_3d(bf_ctx.ctx, y4, y4->ne[0], y4->ne[2], y4->ne[3]);
        } else if (input_tensor.tensor->ne[2] > 1) {
            // ggml's portable 1-D im2col requires ne[3] == 1. For non-CUDA
            // backends, retain the per-item fallback and join along ne2.
            const int64_t T = input_tensor.tensor->ne[0];
            const int64_t C = input_tensor.tensor->ne[1];
            const int64_t B = input_tensor.tensor->ne[2];
            for (int64_t b = 0; b < B; ++b) {
                auto item = ggml_view_3d(
                    bf_ctx.ctx, input_tensor.tensor, T, C, 1, input_tensor.tensor->nb[1],
                    input_tensor.tensor->nb[2],
                    static_cast<size_t>(b) * input_tensor.tensor->nb[2]);
                item = ggml_cont(bf_ctx.ctx, item);
                ggml_tensor* item_out = nullptr;
                item_out = ggml_conv_1d_dw(
                    bf_ctx.ctx, weight_tensor.tensor, item, stride, padding, dilation);
                out_tensor = out_tensor == nullptr
                                 ? item_out
                                 : ggml_concat(bf_ctx.ctx, out_tensor, item_out, 2);
            }
        } else {
            // Portable path: ggml_conv_1d_dw lowers via im2col + mul_mat, which
            // handles F16 kernel weights on every backend (CPU/Metal/Vulkan/CUDA).
            out_tensor = ggml_conv_1d_dw(
                bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, padding, dilation);
        }
    } else {
        out_tensor = ggml_conv_1d(
            bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, padding, dilation);
    }
    if (use_bias) {
        auto bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        // Reshape (OC) to (1, OC, 1) for broadcast over output length and batch.
        auto bias_reshape =
            ggml_reshape_3d(bf_ctx.ctx, bias_tensor.tensor, 1, bias_tensor.tensor->ne[0], 1);
        out_tensor = ggml_add(bf_ctx.ctx, out_tensor, bias_reshape);
    }
    TensorBag output_tensors;
    output_tensors.add_tensor(ggml_bf_tensor(out_tensor, bf_ctx.buft));
    return output_tensors;
}

void
Conv1D::set_data(Session* session) {
    session->load_weight(weight_name);
    if (use_bias) {
        session->load_weight(bias_name);
    }
}


void
Conv2D::define_tensors(Session* session) {
    // Metal depthwise convolution requires F16 weights.
    this->weight = session->model_tensor_container->create_tensor_4d(
        weight_name, GGML_TYPE_F16, kernel_size, kernel_size, in_channels, out_channels);

    this->bias = session->model_tensor_container->create_tensor_4d(
        bias_name, GGML_TYPE_F32, 1, 1, out_channels, 1);
}

TensorBag
Conv2D::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
    ggml_tensor* conv2d_ret = ggml_conv_2d(
        bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, stride, padding, padding,
        dilation, dilation);

    ggml_tensor* output_tensor =
        ggml_add(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, conv2d_ret), bias_tensor.tensor);
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
Conv2D::set_data(Session* session) {
    session->load_weight(weight_name);
    session->load_weight(bias_name);
}


void
ReLU::define_tensors(Session* session) {}

TensorBag
ReLU::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    ggml_tensor* relu_tensor = ggml_relu_inplace(bf_ctx.ctx, input_tensor.tensor);
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(relu_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
ReLU::set_data(Session* session) {}


void
Conv2DDW::define_tensors(Session* session) {
    this->weight = session->model_tensor_container->create_tensor_4d(
        weight_name, GGML_TYPE_F16, kernel_size, kernel_size, 1, out_channels);

    this->bias = session->model_tensor_container->create_tensor_4d(
        bias_name, GGML_TYPE_F32, 1, 1, out_channels, 1);
}

TensorBag
Conv2DDW::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);

    ggml_tensor* conv2d_ret = nullptr;
    bool direct_dw = false;
#ifdef NEMO_SPEECH_DIRECT_DW_CONV
    // The direct kernel preserves the explicit N dimension (the stock im2col
    // lowering does not support this operator with ne[3] > 1) and reads the
    // F16 weights correctly only on the patched CUDA backend. Same runtime
    // gate as Conv1D's depthwise fast path above: a CPU-only session in a
    // CUDA build must take the portable lowering or it silently misreads the
    // F16 kernel as F32 (garbage subsampling output).
    direct_dw = session->params.use_gpu;
#endif
    if (direct_dw) {
        conv2d_ret = ggml_conv_2d_dw_direct(
            bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, stride, padding, padding,
            dilation, dilation);
    } else {
        conv2d_ret = ggml_conv_2d_dw(
            bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, stride, padding, padding,
            dilation, dilation);
    }

    ggml_tensor* output_tensor =
        ggml_add(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, conv2d_ret), bias_tensor.tensor);
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
Conv2DDW::set_data(Session* session) {
    session->load_weight(weight_name);
    session->load_weight(bias_name);
}


void
SequenceModule::define_tensors(Session* session) {
    for (auto& module : modules) {
        module->define_tensors(session);
    }
}

TensorBag
SequenceModule::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    TensorBag output_tensors;
    for (auto& module : modules) {
        output_tensors = module->build_graph(session, input_tensors, session_tensor_container);
        input_tensors = output_tensors;
    }
    return output_tensors;
}

void
SequenceModule::set_data(Session* session) {
    for (auto& module : modules) {
        module->set_data(session);
    }
}


void
Linear::define_tensors(Session* session) {
    // Honor the stored quantization type so quantized GGUFs load with
    // matching byte sizes. Bias always stays at the stored type too
    // (typically F32 or F16 — we never quantize biases in the converter).
    ggml_type weight_type = session->gguf_loader->get_tensor_type(weight_name);
    this->weight = session->model_tensor_container->create_tensor_4d(
        weight_name, weight_type, in_features, out_features, 1, 1);
    if (use_bias) {
        ggml_type bias_type = session->gguf_loader->get_tensor_type(bias_name);
        this->bias = session->model_tensor_container->create_tensor_4d(
            bias_name, bias_type, out_features, 1, 1, 1);
    }
}

TensorBag
Linear::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);

    ggml_tensor* matmul_input =
        cached_q8_input(session, session_tensor_container, weight_tensor, input_tensor.tensor);
    ggml_tensor* matmul_ret = ggml_mul_mat(bf_ctx.ctx, weight_tensor.tensor, matmul_input);

    ggml_tensor* output_tensor = nullptr;
    if (use_bias) {
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        const bool fuse_bf16_bias = session->params.use_gpu &&
                                    weight_tensor.tensor->type == GGML_TYPE_BF16 &&
                                    input_tensor.tensor->ne[2] * input_tensor.tensor->ne[3] > 1;
        output_tensor = fuse_bf16_bias
                            ? ggml_add(bf_ctx.ctx, matmul_ret, bias_tensor.tensor)
                            : ggml_add_inplace(bf_ctx.ctx, matmul_ret, bias_tensor.tensor);
    } else {
        output_tensor = matmul_ret;
    }
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
Linear::set_data(Session* session) {
    session->load_weight(weight_name);
    if (use_bias) {
        // Bias must be F32: the graph adds it to an F32 activation.
        ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
        if (bias_tensor_type != GGML_TYPE_F32) {
            throw std::runtime_error(
                "linear(" + name + ") bias " + bias_name + " is not f32 (got type " +
                std::to_string(int(bias_tensor_type)) + ")");
        }
        session->load_weight(bias_name);
    }
}


void
LayerNorm::define_tensors(Session* session) {
    session->model_tensor_container->create_tensor_4d(
        weight_name, GGML_TYPE_F32, input_shape[0], input_shape[1], input_shape[2], 1);
    session->model_tensor_container->create_tensor_4d(
        bias_name, GGML_TYPE_F32, input_shape[0], 1, 1, 1);
}

TensorBag
LayerNorm::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    auto bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);

    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
    // In-place views prevent CUDA from fusing NORM+MUL+ADD.
    auto norm_tensor = ggml_norm(bf_ctx.ctx, input_tensor.tensor, 1e-5);
    auto scaled_tensor = ggml_mul(bf_ctx.ctx, norm_tensor, weight_tensor.tensor);
    auto output_tensor = ggml_add(bf_ctx.ctx, scaled_tensor, bias_tensor.tensor);
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
LayerNorm::set_data(Session* session) {
    // γ and β must be F32: the graph multiplies/adds them against an F32 norm.
    ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
    ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
    if (weight_tensor_type != GGML_TYPE_F32 || bias_tensor_type != GGML_TYPE_F32) {
        throw std::runtime_error(
            "layer norm(" + name + ") weight " + weight_name + " or bias " + bias_name +
            " is not f32 (got types " + std::to_string(int(weight_tensor_type)) + "/" +
            std::to_string(int(bias_tensor_type)) + ")");
    }
    session->load_weight(weight_name);
    session->load_weight(bias_name);
}


void
BatchNorm1d::define_tensors(Session* session) {
    if (!affine) {
        throw std::runtime_error("BatchNorm1d without affine is not supported");
    }
    session->model_tensor_container->create_tensor_2d(weight_name, GGML_TYPE_F32, 1, num_features);
    session->model_tensor_container->create_tensor_2d(bias_name, GGML_TYPE_F32, 1, num_features);
    session->model_tensor_container->create_tensor_2d(
        running_mean_name, GGML_TYPE_F32, 1, num_features);
    session->model_tensor_container->create_tensor_2d(
        running_var_name, GGML_TYPE_F32, 1, num_features);
    session->model_tensor_container->create_tensor_1d(eps_name, GGML_TYPE_F32, 1);
}

TensorBag
BatchNorm1d::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);

    auto running_mean = session->model_tensor_container->get_tensor_by_name(running_mean_name);
    auto running_var = session->model_tensor_container->get_tensor_by_name(running_var_name);
    auto weight = session->model_tensor_container->get_tensor_by_name(weight_name);
    auto bias = session->model_tensor_container->get_tensor_by_name(bias_name);
    auto eps = session->model_tensor_container->get_tensor_by_name(eps_name);

    auto p1 = ggml_sub(bf_ctx.ctx, input_tensor.tensor, running_mean.tensor);
    auto p2 = ggml_sqrt_inplace(bf_ctx.ctx, ggml_add(bf_ctx.ctx, running_var.tensor, eps.tensor));
    auto p3 = ggml_mul_inplace(bf_ctx.ctx, ggml_div(bf_ctx.ctx, p1, p2), weight.tensor);
    auto x = ggml_add_inplace(bf_ctx.ctx, p3, bias.tensor);
    auto output_tensors = TensorBag();
    output_tensors.add_tensor(ggml_bf_tensor(x, bf_ctx.buft));
    return output_tensors;
}

void
BatchNorm1d::set_data(Session* session) {
    for (const auto& tensor_name : {weight_name, bias_name, running_mean_name, running_var_name}) {
        session->load_weight(tensor_name);
    }
    // eps is not a GGUF tensor: upload the standard 1e-5 numerical guard so
    // sqrt(running_var + eps) never reads uninitialized device memory.
    const float eps_value = 1e-5f;
    auto eps = session->model_tensor_container->get_tensor_by_name(eps_name);
    ggml_backend_tensor_set(eps.tensor, &eps_value, 0, sizeof(eps_value));
}

}  // namespace ggml_runtime
