// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GGUFLoader mmap smoke test: confirms is_mmapped() is true on this
// platform and that every tensor's mapped_tensor_ptr() bytes are identical
// to what the buffered-fread path (get_tensor_file_data) reads for the same
// tensor -- the check that would fail if the mmap offset/base arithmetic in
// TensorContainer::allocate_tensors_on_backend_buffers / Session's zero-copy
// bind were wrong.
// Usage: ./test_gguf_mmap_loader <model.gguf>   (skips if no model arg)
#include <cstdio>
#include <cstring>
#include <string>

#include "runtime.h"

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stdout, "[SKIP] usage: %s <model.gguf>\n", argv[0]);
        return 0;
    }
    const std::string model_path = argv[1];

    ggml_runtime::GGUFLoader loader(model_path);

    if (!loader.is_mmapped()) {
        std::fprintf(
            stderr,
            "[FAIL] GGUFLoader did not mmap %s (llama_mmap::SUPPORTED false on this "
            "platform?)\n",
            model_path.c_str());
        return 1;
    }
    if (loader.mapped_base() == nullptr) {
        std::fprintf(stderr, "[FAIL] is_mmapped() true but mapped_base() is null\n");
        return 1;
    }

    const auto names = loader.tensor_names();
    if (names.empty()) {
        std::fprintf(stderr, "[FAIL] GGUF has no tensors: %s\n", model_path.c_str());
        return 1;
    }

    int checked = 0;
    for (const auto& name : names) {
        const ggml_type type = loader.get_tensor_type(name);
        const auto ne = loader.get_tensor_ne(name);
        if (ne.empty()) {
            continue;
        }
        size_t n_elems = 1;
        for (int64_t d : ne) n_elems *= static_cast<size_t>(d);
        const size_t nbytes = ggml_row_size(type, n_elems / static_cast<size_t>(ne[0])) *
            static_cast<size_t>(ne[0]);
        if (nbytes == 0) {
            continue;
        }

        const void* mapped = loader.mapped_tensor_ptr(name);
        const char* fread_data = loader.get_tensor_file_data(name, nbytes);

        if (std::memcmp(mapped, fread_data, nbytes) != 0) {
            std::fprintf(
                stderr,
                "[FAIL] mmap'd bytes for tensor '%s' (%zu bytes) don't match the buffered-fread "
                "path -- mmap offset/base arithmetic is wrong\n",
                name.c_str(), nbytes);
            return 1;
        }
        checked++;
    }

    if (checked == 0) {
        std::fprintf(stderr, "[FAIL] no tensor in %s had decodable extents to check\n",
            model_path.c_str());
        return 1;
    }

    std::fprintf(stdout, "[PASS] mmap'd %d/%zu tensors match buffered-read bytes\n", checked,
        names.size());
    return 0;
}
