// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GGUFLoader mmap tests. Self-contained cases (synthetic GGUF): mapped bytes
// match the buffered-fread path, a truncated file raises "out of range" rather
// than handing out a pointer past the mapping, and the mapping outlives the
// loader via mapping_handle(). If a model path is given, every tensor of that
// model is also compared against the buffered-fread path.
// Usage: ./test_gguf_mmap_loader [model.gguf]
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"

namespace fs = std::filesystem;

namespace {

int
fail(const char* what) {
    std::fprintf(stderr, "[FAIL] %s\n", what);
    return 1;
}

constexpr size_t kBytesA = 8 * 4 * sizeof(float);  // F32 [8,4]
constexpr size_t kBytesB = 16 * 2 * 2;             // F16 [16,2]

// Writes a two-tensor GGUF ("a", "b") with a known byte pattern.
bool
write_synthetic_gguf(const std::string& path) {
    gguf_context_ptr gctx(gguf_init_empty());
    ggml_init_params ip = {/*mem_size=*/1 << 20, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor* a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 4);
    ggml_tensor* b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 16, 2);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    for (ggml_tensor* t : {a, b}) {
        for (size_t i = 0; i < ggml_nbytes(t); i++) {
            static_cast<unsigned char*>(t->data)[i] = static_cast<unsigned char>(i * 7 + 1);
        }
        gguf_add_tensor(gctx.get(), t);
    }
    return gguf_write_to_file(gctx.get(), path.c_str(), false);
}

constexpr int kSkip = 77;  // matches SKIP_RETURN_CODE in CMake

int
test_synthetic(const fs::path& dir) {
    const std::string path = (dir / "synthetic.gguf").string();
    if (!write_synthetic_gguf(path)) {
        return fail("could not write synthetic GGUF");
    }

    const void* mapped_a = nullptr;
    std::shared_ptr<void> keepalive;
    {
        ggml_runtime::GGUFLoader loader(path);
        if (!loader.is_mmapped()) {
            // The loader falls back to buffered reads where mmap is unavailable.
            std::fprintf(stdout, "[SKIP] mmap unavailable on this platform/filesystem\n");
            return kSkip;
        }
        mapped_a = loader.mapped_tensor_ptr("a", kBytesA);
        const void* mapped_b = loader.mapped_tensor_ptr("b", kBytesB);
        if (std::memcmp(mapped_a, loader.get_tensor_file_data("a", kBytesA), kBytesA) != 0 ||
            std::memcmp(mapped_b, loader.get_tensor_file_data("b", kBytesB), kBytesB) != 0) {
            return fail("synthetic tensor bytes differ between mmap and fread");
        }
        keepalive = loader.mapping_handle();
    }
    // Loader destroyed: the pages must still be readable through the handle.
    if (!keepalive || static_cast<const unsigned char*>(mapped_a)[0] != 1) {
        return fail("mapping did not outlive the GGUFLoader");
    }
    keepalive.reset();

    // Cut into the last tensor: the file still parses (no_alloc) but "b" now
    // runs past EOF, so mapped_tensor_ptr() must throw rather than hand out a
    // pointer past the mapping.
    const std::string cut = (dir / "truncated.gguf").string();
    fs::copy_file(path, cut, fs::copy_options::overwrite_existing);
    fs::resize_file(cut, fs::file_size(cut) - 16);
    ggml_runtime::GGUFLoader loader(cut);
    if (!loader.is_mmapped() || loader.mapped_tensor_ptr("a", kBytesA) == nullptr) {
        return fail("truncated GGUF: untouched tensor unreadable");
    }
    try {
        loader.mapped_tensor_ptr("b", kBytesB);
    }
    catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("out of range") != std::string::npos) {
            return 0;
        }
    }
    return fail("truncated tensor did not raise 'Tensor data out of range'");
}

int
test_model(const std::string& model_path) {
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

    struct gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context_ptr ctx(gguf_init_from_file(model_path.c_str(), params));
    const int64_t n_tensors = ctx ? gguf_get_n_tensors(ctx.get()) : 0;
    if (n_tensors == 0) {
        std::fprintf(stderr, "[FAIL] GGUF has no tensors: %s\n", model_path.c_str());
        return 1;
    }

    int checked = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const std::string name = gguf_get_tensor_name(ctx.get(), i);
        const ggml_type type = loader.get_tensor_type(name);
        const auto ne = loader.get_tensor_ne(name);
        if (ne.empty()) {
            continue;
        }
        size_t n_elems = 1;
        for (int64_t d : ne) n_elems *= static_cast<size_t>(d);
        const size_t n_rows = n_elems / static_cast<size_t>(ne[0]);
        const size_t nbytes = ggml_row_size(type, static_cast<size_t>(ne[0])) * n_rows;
        if (nbytes == 0) {
            continue;
        }

        const void* mapped = loader.mapped_tensor_ptr(name, nbytes);
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
        std::fprintf(
            stderr, "[FAIL] no tensor in %s had decodable extents to check\n", model_path.c_str());
        return 1;
    }

    std::fprintf(
        stdout, "[PASS] mmap'd %d/%lld tensors match buffered-read bytes\n", checked,
        (long long)n_tensors);
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    const fs::path dir = fs::temp_directory_path() / "nemo_speech_gguf_mmap_test";
    fs::create_directories(dir);
    const int rc = test_synthetic(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (rc != 0)
        return rc;
    std::fprintf(stdout, "[PASS] synthetic GGUF: bytes, range check, mapping lifetime\n");
    if (argc >= 2)
        return test_model(argv[1]);
    return 0;
}
