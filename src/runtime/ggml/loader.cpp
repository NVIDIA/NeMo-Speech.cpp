// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <algorithm>

#include "llama-mmap.h"
#include "runtime.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#include <io.h>
#endif

// llama_file/llama_mmap are llama.cpp's (vendored in llama.cpp/src/llama-mmap.{h,cpp}),
// reused as-is rather than reimplemented: they already give us mmap-backed,
// zero-copy-capable loading on every platform this runtime supports.

namespace ggml_runtime {

GGUFLoader::~GGUFLoader() = default;

GGUFLoader::GGUFLoader(const std::string& path) {
    m_path = path;
    struct ggml_context* ctx = NULL;
    struct gguf_init_params params = {
        true,
        &ctx,
    };
    m_context.reset(gguf_init_from_file(path.c_str(), params));
    if (!m_context) {
        throw std::runtime_error("Failed to load GGML file: " + path);
    }

    // Tensor metadata is owned by ctx, so capture dimensionality before freeing it.
    if (ctx != nullptr) {
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            m_tensor_n_dims[t->name] = ggml_n_dims(t);
            m_tensor_ne[t->name] = std::vector<int64_t>(t->ne, t->ne + ggml_n_dims(t));
        }
        ggml_free(ctx);
    }

    m_file = std::make_unique<llama_file>(path.c_str(), "rb");
    GGMLF_LOG_INFO("GGUF file size: %ld\n", (long)m_file->size());

    auto n_tensors = gguf_get_n_tensors(m_context.get());
    GGMLF_LOG_INFO("GGUF has %d tensors\n", n_tensors);

    const uint64_t data_offset = gguf_get_data_offset(m_context.get());
    // Pre-size the reusable read buffer; get_tensor_file_data grows it if needed.
    uint64_t max_tensor_size = 0;
    uint64_t last_tensor_offset = 0;
    for (int i = 0; i < n_tensors; i++) {
        auto tensor_name = gguf_get_tensor_name(m_context.get(), i);
        auto tensor_type = gguf_get_tensor_type(m_context.get(), i);
        const uint64_t tensor_offset = gguf_get_tensor_offset(m_context.get(), i) + data_offset;
        m_tensor_infos.emplace(
            tensor_name, std::tuple<ggml_type, uint64_t>(tensor_type, tensor_offset));
        if (last_tensor_offset != 0 && tensor_offset > last_tensor_offset) {
            max_tensor_size = std::max(max_tensor_size, tensor_offset - last_tensor_offset);
        }
        last_tensor_offset = tensor_offset;
    }
    if (m_file->size() > last_tensor_offset) {
        max_tensor_size = std::max(max_tensor_size, m_file->size() - last_tensor_offset);
    }
    const uint64_t tensor_size_mb = max_tensor_size / 1024 / 1024;
    GGMLF_LOG_INFO("Max tensor size: %zu MB\n", static_cast<size_t>(tensor_size_mb));
    m_tensor_buffer.resize((tensor_size_mb + 1) * 1024 * 1024);

    // Lazily-faulted mmap of the whole file (prefetch=0): tensor pages land in
    // the OS page cache as clean, evictable memory and, where the backend
    // supports it (see TensorContainer::allocate_tensors_on_backend_buffers),
    // are bound directly with zero copy instead of duplicated into a malloc'd
    // backend buffer. Falls back to the fread path above on unsupported
    // platforms.
    if (llama_mmap::SUPPORTED) {
        m_mapping = std::make_unique<llama_mmap>(m_file.get(), /*prefetch=*/0, /*numa=*/false);
    }
}

void
GGUFLoader::release_file_resources() {
    // m_mapping is NOT released here: any tensor bound zero-copy (see
    // TensorContainer::allocate_tensors_on_backend_buffers /
    // Session::load_weight) points directly into these pages for the
    // lifetime of the model. Closing the FILE* is safe independent of that —
    // the mapping keeps the underlying pages valid after the fd is closed.
    m_file.reset();
    std::vector<char>().swap(m_tensor_buffer);
}

bool
GGUFLoader::is_mmapped() const {
    return m_mapping != nullptr;
}

uint64_t
GGUFLoader::get_tensor_offset(const std::string& tensor_name) const {
    auto it = m_tensor_infos.find(tensor_name);
    if (it == m_tensor_infos.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    return std::get<1>(it->second);
}

void*
GGUFLoader::mapped_tensor_ptr(const std::string& tensor_name) const {
    if (!m_mapping) {
        throw std::logic_error("mapped_tensor_ptr() called without an active mmap mapping");
    }
    return static_cast<char*>(m_mapping->addr()) + get_tensor_offset(tensor_name);
}

void*
GGUFLoader::mapped_base() const {
    return m_mapping ? m_mapping->addr() : nullptr;
}

const char*
GGUFLoader::get_tensor_file_data(const std::string& tensor_name, size_t size) {
    if (!m_file) {
        m_file = std::make_unique<llama_file>(m_path.c_str(), "rb");
    }
    auto it = m_tensor_infos.find(tensor_name);
    if (it == m_tensor_infos.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    auto tensor_info = it->second;
    auto tensor_offset = std::get<1>(tensor_info);

    if (tensor_offset + size > m_file->size()) {
        throw std::runtime_error("Tensor data out of range: " + tensor_name);
    }
    if (size > m_tensor_buffer.size()) {
        m_tensor_buffer.resize(size);
    }
    m_file->seek(tensor_offset, SEEK_SET);
    m_file->read_raw(m_tensor_buffer.data(), size);
    return m_tensor_buffer.data();
}

ggml_type
GGUFLoader::get_tensor_type(const std::string& tensor_name) {
    auto it = m_tensor_infos.find(tensor_name);
    if (it == m_tensor_infos.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    auto tensor_info = it->second;
    return std::get<0>(tensor_info);
}

bool
GGUFLoader::has_tensor(const std::string& tensor_name) const {
    return m_tensor_infos.find(tensor_name) != m_tensor_infos.end();
}

std::vector<std::string>
GGUFLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(m_tensor_infos.size());
    for (const auto& kv : m_tensor_infos) {
        names.push_back(kv.first);
    }
    return names;
}

int
GGUFLoader::get_tensor_n_dims(const std::string& tensor_name) const {
    auto it = m_tensor_n_dims.find(tensor_name);
    return it == m_tensor_n_dims.end() ? 0 : it->second;
}

std::vector<int64_t>
GGUFLoader::get_tensor_ne(const std::string& tensor_name) const {
    auto it = m_tensor_ne.find(tensor_name);
    return it == m_tensor_ne.end() ? std::vector<int64_t>{} : it->second;
}

bool
GGUFLoader::has_key(const std::string& key) const {
    return gguf_find_key(m_context.get(), key.c_str()) >= 0;
}

uint32_t
GGUFLoader::get_u32(const std::string& key, uint32_t def) const {
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return def;
    return gguf_get_val_u32(m_context.get(), id);
}

int32_t
GGUFLoader::get_i32(const std::string& key, int32_t def) const {
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return def;
    return gguf_get_val_i32(m_context.get(), id);
}

float
GGUFLoader::get_f32(const std::string& key, float def) const {
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return def;
    return gguf_get_val_f32(m_context.get(), id);
}

bool
GGUFLoader::get_bool(const std::string& key, bool def) const {
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return def;
    return gguf_get_val_bool(m_context.get(), id);
}

std::string
GGUFLoader::get_str(const std::string& key, const std::string& def) const {
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return def;
    return std::string(gguf_get_val_str(m_context.get(), id));
}

std::vector<std::string>
GGUFLoader::get_str_array(const std::string& key) const {
    std::vector<std::string> out;
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0)
        return out;
    if (gguf_get_kv_type(m_context.get(), id) != GGUF_TYPE_ARRAY)
        return out;
    if (gguf_get_arr_type(m_context.get(), id) != GGUF_TYPE_STRING)
        return out;
    const int n = gguf_get_arr_n(m_context.get(), id);
    out.reserve(n);
    for (int i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(m_context.get(), id, i));
    }
    return out;
}

std::vector<int32_t>
GGUFLoader::get_i32_array(const std::string& key) const {
    std::vector<int32_t> out;
    const int id = gguf_find_key(m_context.get(), key.c_str());
    if (id < 0 || gguf_get_kv_type(m_context.get(), id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(m_context.get(), id) != GGUF_TYPE_INT32) {
        return out;
    }
    const size_t n = gguf_get_arr_n(m_context.get(), id);
    const auto* data = static_cast<const int32_t*>(gguf_get_arr_data(m_context.get(), id));
    out.assign(data, data + n);
    return out;
}


}  // namespace ggml_runtime
