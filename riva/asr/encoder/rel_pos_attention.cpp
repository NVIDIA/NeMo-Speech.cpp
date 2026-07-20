// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
// Implementations for RelPositionalEncoding + RelPositionMultiHeadAttention.
// Extracted from runtime/ggml/nn.cpp - these are FastConformer-specific.
#include "rel_pos_attention.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <vector>

namespace ggml_runtime {


void
RelPositionalEncoding::define_tensors(Session* session) {
    session->model_tensor_container->create_tensor_2d(
        name + ".pe", GGML_TYPE_F32, d_model, max_len * 2 - 1);
}

ggml_bf_tensor
RelPositionalEncoding::get_pe_tensor(Session* session, TensorContainer* session_tensor_container) {
    ggml_bf_tensor div_term_tensor =
        session->model_tensor_container->get_tensor_by_name(name + ".div_term");
    ggml_bf_tensor positions_tensor =
        session->model_tensor_container->get_tensor_by_name(name + ".positions");
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(div_term_tensor.buft);
    auto grid_tensor = ggml_out_prod(bf_ctx.ctx, div_term_tensor.tensor, positions_tensor.tensor);

    auto cos_grid_tensor = ggml_cos(bf_ctx.ctx, grid_tensor);
    auto sin_grid_tensor = ggml_sin(bf_ctx.ctx, grid_tensor);

    auto interleave_1_tensor =
        session->model_tensor_container->get_tensor_by_name(name + ".interleave_1");
    auto interleave_2_tensor =
        session->model_tensor_container->get_tensor_by_name(name + ".interleave_2");

    auto cos_grid_unsqueeze_tensor =
        ggml_reshape_4d(bf_ctx.ctx, cos_grid_tensor, 1, 1, grid_tensor->ne[0], grid_tensor->ne[1]);

    auto sin_grid_unsqueeze_tensor =
        ggml_reshape_4d(bf_ctx.ctx, sin_grid_tensor, 1, 1, grid_tensor->ne[0], grid_tensor->ne[1]);

    auto cos_grid_interleave_1_tensor =
        ggml_out_prod(bf_ctx.ctx, interleave_1_tensor.tensor, cos_grid_unsqueeze_tensor);

    auto sin_grid_interleave_2_tensor =
        ggml_out_prod(bf_ctx.ctx, interleave_2_tensor.tensor, sin_grid_unsqueeze_tensor);

    auto cos_grid_interleave_1_compact_tensor = ggml_reshape_2d(
        bf_ctx.ctx, cos_grid_interleave_1_tensor,
        cos_grid_interleave_1_tensor->ne[0] * cos_grid_interleave_1_tensor->ne[1] *
            cos_grid_interleave_1_tensor->ne[2],
        cos_grid_interleave_1_tensor->ne[3]);

    auto sin_grid_interleave_2_compact_tensor = ggml_reshape_2d(
        bf_ctx.ctx, sin_grid_interleave_2_tensor,
        sin_grid_interleave_2_tensor->ne[0] * sin_grid_interleave_2_tensor->ne[1] *
            sin_grid_interleave_2_tensor->ne[2],
        sin_grid_interleave_2_tensor->ne[3]);

    auto pe_tensor = ggml_add_inplace(
        bf_ctx.ctx, cos_grid_interleave_1_compact_tensor, sin_grid_interleave_2_compact_tensor);

    auto pe_bf_tensor = ggml_bf_tensor(pe_tensor, bf_ctx.buft);
    session->model_tensor_container->cache_tensor(pe_name, pe_bf_tensor);
    return pe_bf_tensor;
}


TensorBag
RelPositionalEncoding::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    int feature_len = input_tensor.tensor->ne[1];

    auto pe_tensor = session->model_tensor_container->get_tensor_by_name(name + ".pe");
    // Build the per-call view of the PE weight in the per-run container, not
    // model_tensor_container: it must be freed with the run (and sized by
    // probe-then-fit), else it accumulates in the persistent weight arena.
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(pe_tensor.buft);
    auto center_pos = max_len;
    auto start_pos = center_pos - feature_len;
    auto end_pos = center_pos + feature_len - 1;
    auto new_ne0 = pe_tensor.tensor->ne[0];
    auto new_ne1 = end_pos - start_pos;
    auto new_nb1 = input_tensor.tensor->nb[1];
    auto offset = start_pos * input_tensor.tensor->nb[1];
    auto pos_embd = ggml_view_2d(bf_ctx.ctx, pe_tensor.tensor, new_ne0, new_ne1, new_nb1, offset);
    input_tensors.add_tensor(ggml_bf_tensor(pos_embd, bf_ctx.buft));

    return input_tensors;
}

void
RelPositionalEncoding::set_data(Session* session) {
    auto pe_tensor = session->model_tensor_container->get_tensor_by_name(name + ".pe");
    auto data_size = ggml_nbytes(pe_tensor.tensor);
    // Prefer GGUF-embedded PE (produced by convert_model.py). Fall back
    // to external pe.bin for legacy tdt-0.6b flow.
    if (session->gguf_loader != nullptr && session->gguf_loader->has_tensor(name + ".pe")) {
        auto data = session->gguf_loader->get_tensor_file_data(name + ".pe", data_size);
        ggml_backend_tensor_set(pe_tensor.tensor, data, 0, data_size);
        return;
    }
    std::vector<char> buffer(data_size);
    if (!session->params.pe_bin_path || session->params.pe_bin_path[0] == '\0') {
        GGMLF_LOG_ERROR("pos_enc: no GGUF pe tensor and no pe_bin_path set");
        return;
    }
    auto file = ggml_fopen(session->params.pe_bin_path, "rb");
    if (file == nullptr) {
        GGMLF_LOG_ERROR("open pe.bin failed");
        return;
    }
    auto read_size = fread(buffer.data(), 1, data_size, file);
    if (read_size != data_size) {
        GGMLF_LOG_ERROR("read pe.bin failed");
    }
    fclose(file);
    ggml_backend_tensor_set(pe_tensor.tensor, buffer.data(), 0, data_size);
}


void
RelPositionMultiHeadAttention::define_tensors(Session* session) {
    session->model_tensor_container->create_tensor_4d(
        pos_bias_u_name, GGML_TYPE_F32, d_k, n_head, 1, 1);
    session->model_tensor_container->create_tensor_4d(
        pos_bias_v_name, GGML_TYPE_F32, d_k, n_head, 1, 1);
    // Fused QKV weight: Q/K/V stacked along the output dim at the stored
    // quantization type (rows are independent for all ggml quant formats, so
    // stacking is a plain byte concat — see set_data).
    const ggml_type qkv_type = session->gguf_loader->get_tensor_type(name + ".linear_q.weight");
    session->model_tensor_container->create_tensor_4d(
        qkv_weight_name, qkv_type, n_feat, 3 * n_feat, 1, 1);
    if (use_bias) {
        session->model_tensor_container->create_tensor_4d(
            qkv_bias_name, GGML_TYPE_F32, 3 * n_feat, 1, 1, 1);
    }
    linear_pos->define_tensors(session);
    linear_out->define_tensors(session);
}

TensorBag
RelPositionMultiHeadAttention::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    return build_graph_masked(
        session, input_tensors, session_tensor_container, /*attn_mask=*/nullptr);
}

TensorBag
RelPositionMultiHeadAttention::build_graph_masked(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container,
    ggml_tensor* attn_mask) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);

    auto pos_emb_tensor = input_tensors.get_tensor(1);

    // Q/K/V weights share one persistent allocation and all batch sizes use
    // the same fused projection, saving two activation quantizations and two
    // launches per layer while preserving the outer batch dimension in the
    // strided projection views below.
    ggml_bf_tensor qkv_weight =
        session->model_tensor_container->get_tensor_by_name(qkv_weight_name);
    ggml_tensor* qkv_input = input_tensor.tensor;
    if (session->params.use_gpu && qkv_input->type == GGML_TYPE_F32 &&
        qkv_weight.tensor->type == GGML_TYPE_BF16) {
        qkv_input = ggml_cast(bf_ctx.ctx, qkv_input, GGML_TYPE_BF16);
    }
    const int64_t qlen = input_tensor.tensor->ne[1];
    const int64_t batch = input_tensor.tensor->ne[2];
    ggml_tensor* q_multi_head = nullptr;
    ggml_tensor* k_multi_head = nullptr;
    ggml_tensor* v_reshaped = nullptr;
    {
        auto qkv = ggml_mul_mat(bf_ctx.ctx, qkv_weight.tensor, qkv_input);
        if (use_bias) {
            auto bias = session->model_tensor_container->get_tensor_by_name(qkv_bias_name);
            qkv = session->params.use_gpu && qkv_weight.tensor->type == GGML_TYPE_BF16 &&
                          input_tensor.tensor->ne[2] * input_tensor.tensor->ne[3] > 1
                      ? ggml_add(bf_ctx.ctx, qkv, bias.tensor)
                      : ggml_add_inplace(bf_ctx.ctx, qkv, bias.tensor);
        }
        const size_t es = ggml_element_size(qkv);
        auto slice = [&](int idx) {
            return ggml_view_4d(
                bf_ctx.ctx, qkv, d_k, n_head, qlen, batch, static_cast<size_t>(d_k) * es,
                qkv->nb[1], qkv->nb[2], static_cast<size_t>(idx) * n_feat * es);
        };
        q_multi_head = slice(0);
        k_multi_head = ggml_permute(bf_ctx.ctx, slice(1), 0, 2, 1, 3);
        v_reshaped = slice(2);
    }

    auto linear_pos_input_bag = TensorBag();
    // Keep pos_len in ne[1] (not a singleton-padded ne[2]) so the projection
    // dispatches as one GEMM over pos_len columns rather than a per-position
    // batched GEMV. Same memory layout; only the mul_mat path changes.
    auto pos_emb_reshaped = ggml_reshape_4d(
        bf_ctx.ctx, pos_emb_tensor.tensor, pos_emb_tensor.tensor->ne[0],
        pos_emb_tensor.tensor->ne[1], pos_emb_tensor.tensor->ne[2], 1);
    linear_pos_input_bag.add_tensor(ggml_bf_tensor(pos_emb_reshaped, bf_ctx.buft));
    auto pos_linear_out =
        linear_pos->build_graph(session, linear_pos_input_bag, session_tensor_container);
    auto pos_linear_out_tensor = pos_linear_out.get_tensor(0);
    auto p = ggml_reshape_3d(
        bf_ctx.ctx, pos_linear_out_tensor.tensor, d_k, n_head, pos_linear_out_tensor.tensor->ne[1]);

    ggml_bf_tensor pos_bias_u_tensor =
        session->model_tensor_container->get_tensor_by_name(pos_bias_u_name);
    ggml_bf_tensor pos_bias_v_tensor =
        session->model_tensor_container->get_tensor_by_name(pos_bias_v_name);

    auto scale_factor = 1.0f / sqrtf(d_k);

    // attn: attention context in [d_k, n_head, q_len, batch] layout, ready for
    // the shared merge-heads reshape below.
    ggml_tensor* attn = nullptr;
    bool attn_heads_merged = false;
#ifdef NEMO_SPEECH_FUSED_RELPOS_ATTN
    // The fused CUDA op handles a per-key mask only (broadcast over queries); a
    // banded / per-(key,query) offline mask (ne[1] > 1) can't be expressed and
    // falls back to the unfused graph below. This entire branch is compiled
    // ONLY with a patched ggml: ggml_fused_relpos_attn is a patch-only symbol,
    // so a stock-ggml build (NEMO_SPEECH_GGML_PATCHED=OFF) must not reference
    // it — it takes the unfused path unconditionally.
    //
    // Runtime gate too, not just compile-time: the op is CUDA-only (the CPU
    // backend reports it unsupported), so a CPU-only session (use_gpu=false ->
    // no GPU backend in the scheduler) must build the unfused graph or
    // ggml_backend_sched_split_graph aborts with no backend able to take the
    // node. use_gpu=true guarantees the GPU backend exists (BackendManager
    // throws otherwise), and the scheduler places the op there.
    const bool mask_fusable =
        attn_mask == nullptr ||
        (attn_mask->ne[1] == 1 && attn_mask->ne[2] == 1 && attn_mask->ne[3] == 1);
    const bool use_fused = session->params.use_gpu && mask_fusable;
    if (use_fused) {
        // Canonical contiguous [d_k, len, n_head, batch] operands. Bias add and
        // rel-shift happen inside the kernel, so q_can is passed pre-bias.
        //
        // K/V/P use F16 storage to reduce bandwidth; the kernel upconverts and
        // performs arithmetic in F32.
        // Q remains a stride-only view of the fused-QKV projection. The CUDA
        // kernel consumes arbitrary q_sq/q_sh/q_sb strides, so staging a
        // contiguous F32 copy here only adds a full activation round trip.
        auto q_can = ggml_permute(bf_ctx.ctx, q_multi_head, 0, 2, 1, 3);
        auto k_can = ggml_cast(bf_ctx.ctx, k_multi_head, GGML_TYPE_F16);
        auto v_can =
            ggml_cast(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, v_reshaped, 0, 2, 1, 3), GGML_TYPE_F16);
        auto p_can = ggml_cast(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, p, 0, 2, 1, 3), GGML_TYPE_F16);
        auto out = ggml_fused_relpos_attn(
            bf_ctx.ctx, q_can, k_can, v_can, p_can, pos_bias_u_tensor.tensor,
            pos_bias_v_tensor.tensor, attn_mask, scale_factor, /*merge_heads=*/true);
        attn = ggml_permute(bf_ctx.ctx, out, 0, 2, 1, 3);  // [d_k, n_head, q, batch]
        attn_heads_merged = true;
    }
    if (!use_fused)
#endif
    {
        // Batch-preserving formulation: Q/K/V [dk,T,H,B], P [dk,2T-1,H].
        // Every reshape and view must retain the B stride.
        const int64_t T = input_tensor.tensor->ne[1];
        const int64_t B = input_tensor.tensor->ne[2];
        auto qh = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, q_multi_head, 0, 2, 1, 3));
        auto kh = ggml_cont(bf_ctx.ctx, k_multi_head);
        auto vh = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, v_reshaped, 0, 2, 1, 3));
        auto ph = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, p, 0, 2, 1, 3));
        auto bu = ggml_reshape_4d(bf_ctx.ctx, pos_bias_u_tensor.tensor, d_k, 1, n_head, 1);
        auto bv = ggml_reshape_4d(bf_ctx.ctx, pos_bias_v_tensor.tensor, d_k, 1, n_head, 1);
        auto qu = ggml_add(bf_ctx.ctx, qh, bu);
        auto qv = ggml_add(bf_ctx.ctx, qh, bv);

        auto matrix_ac = ggml_mul_mat(bf_ctx.ctx, kh, qu);  // [T,T,H,B]
        auto matrix_bd = ggml_mul_mat(bf_ctx.ctx, ph, qv);  // [2T-1,T,H,B]

        // Portable one-row left pad followed by the Transformer-XL relative shift.
        auto first_row = ggml_view_4d(
            bf_ctx.ctx, matrix_bd, 1, T, n_head, B, matrix_bd->nb[1], matrix_bd->nb[2],
            matrix_bd->nb[3], 0);
        auto zero_row = ggml_scale(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, first_row), 0.0f);
        matrix_bd = ggml_concat(bf_ctx.ctx, zero_row, matrix_bd, 0);  // [2T,T,H,B]
        matrix_bd = ggml_reshape_4d(bf_ctx.ctx, matrix_bd, T, 2 * T, n_head, B);
        matrix_bd = ggml_view_4d(
            bf_ctx.ctx, matrix_bd, T, 2 * T - 1, n_head, B, matrix_bd->nb[1], matrix_bd->nb[2],
            matrix_bd->nb[3], matrix_bd->nb[1]);
        matrix_bd = ggml_cont(bf_ctx.ctx, matrix_bd);
        matrix_bd = ggml_reshape_4d(bf_ctx.ctx, matrix_bd, 2 * T - 1, T, n_head, B);
        matrix_bd = ggml_view_4d(
            bf_ctx.ctx, matrix_bd, T, T, n_head, B, matrix_bd->nb[1], matrix_bd->nb[2],
            matrix_bd->nb[3], 0);
        matrix_bd = ggml_cont(bf_ctx.ctx, matrix_bd);

        auto scores = ggml_scale_inplace(
            bf_ctx.ctx, ggml_add(bf_ctx.ctx, matrix_ac, matrix_bd), scale_factor);

        // Optional attention mask: caller passes an additive mask shaped to
        // broadcast onto (kv_len, q_len, n_head, batch). Valid positions are 0,
        // disallowed positions are -large (e.g. -1e30). Used by the
        // FastConformerEncoder offline path to enforce models trained with
        // limited attention context (`att_context_size=[L, R]`).
        if (attn_mask != nullptr) {
            scores = ggml_add(bf_ctx.ctx, scores, attn_mask);
        }

        auto soft = ggml_soft_max_inplace(bf_ctx.ctx, scores);
        auto vtk = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, vh, 1, 0, 2, 3));
        auto ctxh = ggml_mul_mat(bf_ctx.ctx, vtk, soft);    // [dk,T,H,B]
        attn = ggml_permute(bf_ctx.ctx, ctxh, 0, 2, 1, 3);  // [dk,H,T,B]
    }

    if (attn_heads_merged) {
        // merge_heads=true makes the permuted view above contiguous as
        // [d_k*n_head, q, batch], so reshape it directly without a copy.
        attn = ggml_reshape_3d(bf_ctx.ctx, attn, n_head * d_k, attn->ne[2], attn->ne[3]);
    } else {
        attn = ggml_reshape_3d(
            bf_ctx.ctx, ggml_cont(bf_ctx.ctx, attn), n_head * d_k, attn->ne[2], attn->ne[3]);
    }

    auto out_bag = TensorBag();
    out_bag.add_tensor(ggml_bf_tensor(attn, bf_ctx.buft));

    out_bag = linear_out->build_graph(session, out_bag, session_tensor_container);
    out_bag.add_tensor(ggml_bf_tensor(attn, bf_ctx.buft));

    return out_bag;
}

void
RelPositionMultiHeadAttention::set_data(Session* session) {
    session->load_weight(pos_bias_u_name);
    session->load_weight(pos_bias_v_name);

    // Stack Q/K/V weights (and biases) into the fused tensors. Standard
    // quantized rows concatenate at thirds offsets. Tensor-planar Q8 keeps
    // all three quant planes before all three scale planes instead.
    {
        ggml_bf_tensor w = session->model_tensor_container->get_tensor_by_name(qkv_weight_name);
        const size_t third = ggml_nbytes(w.tensor) / 3;
        const std::string qkv_q8_layout =
            session->gguf_loader->get_str("asr.encoder.qkv_q8_layout");
        const bool planar_q8 =
            w.tensor->type == GGML_TYPE_Q8_0 &&
            session->gguf_loader->get_str("asr.encoder.q8_layout") == "tensor_planar_v1" &&
            qkv_q8_layout != "block_q8_0";
        const char* buft_name = ggml_backend_buft_name(w.buft);
        if (planar_q8 && (buft_name == nullptr || std::string(buft_name).rfind("CUDA", 0) != 0)) {
            throw std::runtime_error(
                "fused qkv(" + name + "): tensor_planar_v1 Q8 weights require CUDA");
        }
        const size_t part_qs = (size_t)n_feat * n_feat;
        const size_t part_d = third - part_qs;
        const size_t total_qs = 3 * part_qs;
        const char* parts[3] = {".linear_q.weight", ".linear_k.weight", ".linear_v.weight"};
        for (int i = 0; i < 3; i++) {
            const std::string key = name + parts[i];
            const ggml_type disk = session->gguf_loader->get_tensor_type(key);
            if (disk != w.tensor->type) {
                throw std::runtime_error(
                    "fused qkv(" + name + "): " + key + " dtype " + std::to_string(int(disk)) +
                    " != fused dtype " + std::to_string(int(w.tensor->type)));
            }
            const char* data = session->gguf_loader->get_tensor_file_data(key, third);
            if (planar_q8) {
                // Each source tensor is [qs_part][d_part]. Assemble the fused
                // allocation as [qs_q|qs_k|qs_v][d_q|d_k|d_v].
                ggml_backend_tensor_set(w.tensor, data, i * part_qs, part_qs);
                ggml_backend_tensor_set(w.tensor, data + part_qs, total_qs + i * part_d, part_d);
            } else {
                ggml_backend_tensor_set(w.tensor, data, i * third, third);
            }
        }
        if (planar_q8) {
#ifdef NEMO_SPEECH_GGML_PATCHED
            w.tensor->flags |= GGML_TENSOR_FLAG_Q8_PLANAR;
#else
            throw std::runtime_error("tensor-planar Q8 weights require NEMO_SPEECH_GGML_PATCHED");
#endif
        }
        if (use_bias) {
            ggml_bf_tensor b = session->model_tensor_container->get_tensor_by_name(qkv_bias_name);
            const size_t bthird = ggml_nbytes(b.tensor) / 3;
            const char* bparts[3] = {".linear_q.bias", ".linear_k.bias", ".linear_v.bias"};
            for (int i = 0; i < 3; i++) {
                const std::string key = name + bparts[i];
                if (session->gguf_loader->get_tensor_type(key) != GGML_TYPE_F32) {
                    throw std::runtime_error("fused qkv(" + name + "): " + key + " is not f32");
                }
                const char* data = session->gguf_loader->get_tensor_file_data(key, bthird);
                ggml_backend_tensor_set(b.tensor, data, i * bthird, bthird);
            }
        }
    }
    linear_pos->set_data(session);
    linear_out->set_data(session);
}

}  // namespace ggml_runtime
