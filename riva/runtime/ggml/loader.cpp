// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <algorithm>

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

struct llama_file {
#if defined(_WIN32)
    FILE* fp;
    HANDLE fp_win32;
    size_t size;

   private:
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

   public:
    llama_file(const char* fname, const char* mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(
                format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        // SEEK_* and FILE_* must remain interchangeable here.
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(
                format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void* ptr, size_t len) const {
        // Some Windows configurations reject ReadFile requests larger than 64 MiB.
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64 * 1024 * 1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(
                fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(
                    format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        };
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void* ptr, size_t len) const {
        // Match the conservative ReadFile chunk limit.
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64 * 1024 * 1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(
                fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size,
                &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(
                    format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(std::uint32_t val) const { write_raw(&val, sizeof(val)); }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    FILE* fp;
    size_t size;

    llama_file(const char* fname, const char* mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t)ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw(void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const { write_raw(&val, sizeof(val)); }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif
};


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
        }
        ggml_free(ctx);
    }

    m_file = std::make_unique<llama_file>(path.c_str(), "rb");
    GGMLF_LOG_INFO("GGUF file size: %ld\n", m_file->size);

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
    if (m_file->size > last_tensor_offset) {
        max_tensor_size = std::max(max_tensor_size, m_file->size - last_tensor_offset);
    }
    const uint64_t tensor_size_mb = max_tensor_size / 1024 / 1024;
    GGMLF_LOG_INFO("Max tensor size: %zu MB\n", static_cast<size_t>(tensor_size_mb));
    m_tensor_buffer.resize((tensor_size_mb + 1) * 1024 * 1024);
}

void
GGUFLoader::release_file_resources() {
    m_file.reset();
    std::vector<char>().swap(m_tensor_buffer);
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

    if (tensor_offset + size > m_file->size) {
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

int
GGUFLoader::get_tensor_n_dims(const std::string& tensor_name) const {
    auto it = m_tensor_n_dims.find(tensor_name);
    return it == m_tensor_n_dims.end() ? 0 : it->second;
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
