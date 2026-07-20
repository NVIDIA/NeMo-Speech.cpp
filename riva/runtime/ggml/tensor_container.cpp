// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <algorithm>  // std::sort (not transitively pulled in by MSVC's STL)

#include "ggml.h"
#include "runtime.h"

namespace ggml_runtime {

// Persistent tensors use the first accelerator's main buffer, falling back to
// CPU. The scheduler inserts copies for operations assigned elsewhere.
static ggml_backend_buffer_type_t
device_main_buft(const buft_list_t& buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto& cur : buft_list) {
        if (cur.second == ggml_backend_dev_buffer_type(cur.first)) {
            return cur.second;
        }
    }
    return buft_list.back().second;  // CPU main is always last
}

TensorBag::TensorBag() {
    tensors = std::vector<ggml_bf_tensor>();
}

void
TensorBag::add_tensor(ggml_bf_tensor tensor) {
    tensors.emplace_back(tensor);
}

ggml_bf_tensor
TensorBag::get_tensor(const size_t index) const {
    GGML_ASSERT(index < tensors.size());
    return tensors[index];
};

size_t
TensorBag::tensor_count() const {
    return tensors.size();
}

void
TensorBag::set_first_tensor(ggml_bf_tensor tensor) {
    if (tensors.empty()) {
        tensors.emplace_back(tensor);
    } else {
        tensors[0] = tensor;
    }
}

TensorContainer::TensorContainer(buft_list_t buft_list, ArenaSizes sizes) {
    this->buft_list = buft_list;
    this->sizes_ = std::move(sizes);
}

ggml_context*
TensorContainer::get_temp_ctx() {
    if (!temp_ctx) {
        // This arena holds tensor metadata only; backend storage is allocated separately.
        ggml_init_params params = {
            /*.mem_size   =*/sizes_.temp_ctx_bytes,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        temp_ctx.reset(ggml_init(params));
        if (!temp_ctx) {
            throw std::runtime_error(format("failed to create ggml context"));
        }
    }
    return temp_ctx.get();
}

void
TensorContainer::free_temp_ctx() {
    temp_ctx.reset();
}

TensorContainer::Measurement
TensorContainer::measure() const {
    Measurement m{};
    m.temp_ctx_bytes = temp_ctx ? ggml_used_mem(temp_ctx.get()) : 0;
    for (const auto& kv : ctx_map) {
        m.per_buft_bytes[kv.first] = ggml_used_mem(kv.second.ctx);
    }
    return m;
}
ggml_bf_tensor
TensorContainer::get_tensor_by_name(const std::string& name) {
    auto it = tensor_lookup.find(name);
    if (it == tensor_lookup.end()) {
        throw std::runtime_error(format("tensor %s not found", name.c_str()));
    }
    return it->second;
}

bool
TensorContainer::has_tensor_by_name(const std::string& name) {
    return tensor_lookup.find(name) != tensor_lookup.end();
}

ggml_bf_tensor
TensorContainer::m_create_tensor(ggml_tensor* meta, std::string& name) {
    ggml_backend_buffer_type_t buft = device_main_buft(buft_list);
    // Optional placement diagnostic. NEMO_SPEECH_LOG_PLACEMENT=1 prints
    // `name -> buft` for every tensor created here.
    static const bool log_placement = []() {
        const char* e = std::getenv("NEMO_SPEECH_LOG_PLACEMENT");
        return e && e[0] == '1';
    }();
    if (log_placement) {
        const char* buft_name = ggml_backend_buft_name(buft);
        GGMLF_LOG_INFO(
            "placement: %-48s type=%-7s -> %s\n", name.c_str(), ggml_type_name(meta->type),
            buft_name ? buft_name : "<unknown>");
    }

    ggml_bf_context bf_ctx = get_ctx_of_buffer_type(buft);
    ggml_tensor* tensor = ggml_dup_tensor(bf_ctx.ctx, meta);
    ggml_set_name(tensor, name.c_str());
    auto bf_tensor = ggml_bf_tensor(tensor, buft);
    if (!tensor_lookup.insert(std::make_pair(name, bf_tensor)).second) {
        throw std::runtime_error("duplicate tensor name declared: " + name);
    }
    return bf_tensor;
}

void
TensorContainer::cache_tensor(std::string name, ggml_bf_tensor tensor) {
    if (!tensor_lookup.insert(std::make_pair(name, tensor)).second) {
        throw std::runtime_error("duplicate tensor name cached: " + name);
    }
}

ggml_bf_tensor
TensorContainer::create_tensor_1d(std::string name, ggml_type data_type, int64_t ne0) {
    ggml_tensor* meta = ggml_new_tensor_1d(get_temp_ctx(), data_type, ne0);
    return m_create_tensor(meta, name);
}

ggml_bf_tensor
TensorContainer::create_tensor_2d(std::string name, ggml_type data_type, int64_t ne0, int64_t ne1) {
    ggml_tensor* meta = ggml_new_tensor_2d(get_temp_ctx(), data_type, ne0, ne1);
    return m_create_tensor(meta, name);
}

ggml_bf_tensor
TensorContainer::create_tensor_3d(
    std::string name, ggml_type data_type, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor* meta = ggml_new_tensor_3d(get_temp_ctx(), data_type, ne0, ne1, ne2);
    return m_create_tensor(meta, name);
}

ggml_bf_tensor
TensorContainer::create_tensor_4d(
    std::string name, ggml_type data_type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor* meta = ggml_new_tensor_4d(get_temp_ctx(), data_type, ne0, ne1, ne2, ne3);
    return m_create_tensor(meta, name);
}

ggml_bf_context
TensorContainer::get_ctx_of_buffer_type(ggml_backend_buffer_type_t buft) {
    auto it = ctx_map.find(buft);
    if (it == ctx_map.end()) {
        auto override_it = sizes_.per_buft.find(buft);
        const size_t bytes =
            override_it != sizes_.per_buft.end() ? override_it->second : sizes_.default_buft_bytes;
        ggml_init_params params = {
            /*.mem_size   =*/bytes,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context_ptr ctx_owned(ggml_init(params));
        if (!ctx_owned) {
            throw std::runtime_error(format("failed to create ggml context"));
        }

        // ctx_map stores a raw view; owned_ctxs_ keeps the ggml_context
        // alive until this TensorContainer destructs.
        ggml_bf_context bf_ctx(ctx_owned.get(), buft);
        owned_ctxs_.push_back(std::move(ctx_owned));
        ctx_map.insert(std::make_pair(buft, bf_ctx));

        return bf_ctx;
    }

    return it->second;
}

void
TensorContainer::allocate_tensors_on_backend_buffers() {
    if (sizes_.sched_managed) {
        // Materialize only the NAMED tensors (inputs + registered aux
        // outputs); graph intermediates are left for the scheduler's graph
        // allocator, which reuses memory by liveness. See ArenaSizes.
        std::map<ggml_backend_buffer_type_t, std::vector<ggml_tensor*>> by_buft;
        for (auto& kv : tensor_lookup) {
            ggml_tensor* t = kv.second.tensor;
            if (t->data != nullptr || t->view_src != nullptr) {
                continue;  // already placed, or aliases another tensor
            }
            by_buft[kv.second.buft].push_back(t);
        }
        for (auto& kv : by_buft) {
            ggml_backend_buffer_type_t buft = kv.first;
            const size_t align = ggml_backend_buft_get_alignment(buft);
            size_t total = 0;
            for (ggml_tensor* t : kv.second) {
                const size_t sz = ggml_backend_buft_get_alloc_size(buft, t);
                total += (sz + align - 1) / align * align;
            }
            if (total == 0) {
                continue;
            }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
            if (!buf) {
                const char* buft_name = ggml_backend_buft_name(buft);
                throw std::runtime_error(
                    std::string("failed to allocate named-tensor buffer for buffer type ") +
                    (buft_name ? buft_name : "<unknown>") + " (out of device memory?)");
            }
            ggml_tallocr ta = ggml_tallocr_new(buf);
            for (ggml_tensor* t : kv.second) {
                if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error(
                        std::string("failed to place named tensor ") + t->name);
                }
            }
            backend_buffers.emplace_back(buf);
        }
        return;
    }
    for (auto& p : ctx_map) {
        ggml_backend_buffer_type_t buft = p.first;
        ggml_bf_context bf_ctx = p.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(bf_ctx.ctx, buft);
        if (!buf) {
            const char* buft_name = ggml_backend_buft_name(buft);
            throw std::runtime_error(
                std::string("failed to allocate backend buffer for buffer type ") +
                (buft_name ? buft_name : "<unknown>") + " (out of device memory?)");
        }
        backend_buffers.emplace_back(buf);
    }
}

size_t
TensorContainer::total_backend_buffer_bytes() const {
    size_t total = 0;
    for (const auto& buf : backend_buffers) {
        total += ggml_backend_buffer_get_size(buf.get());
    }
    return total;
}

void
TensorContainer::dump_largest_tensors(size_t top_n) const {
    std::vector<std::pair<size_t, const ggml_tensor*>> all;
    size_t total = 0;
    for (const auto& p : ctx_map) {
        for (ggml_tensor* t = ggml_get_first_tensor(p.second.ctx); t != nullptr;
             t = ggml_get_next_tensor(p.second.ctx, t)) {
            const size_t nb = ggml_nbytes(t);
            total += nb;
            all.emplace_back(nb, t);
        }
    }
    std::sort(
        all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    fprintf(
        stderr, "[memstats]   %zu tensors, %.2f MB total; largest:\n", all.size(),
        total / 1048576.0);
    for (size_t i = 0; i < all.size() && i < top_n; ++i) {
        const ggml_tensor* t = all[i].second;
        fprintf(
            stderr, "[memstats]     %8.2f MB  %-12s %-40s [%ld,%ld,%ld,%ld]\n",
            all[i].first / 1048576.0, ggml_op_name(t->op), t->name, (long)t->ne[0], (long)t->ne[1],
            (long)t->ne[2], (long)t->ne[3]);
    }
}


}  // namespace ggml_runtime
