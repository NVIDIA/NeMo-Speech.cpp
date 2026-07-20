// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#if defined(GGML_USE_CUDA) && defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GGML_EXAMPLES_NVTX_ENABLED 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GGML_EXAMPLES_NVTX_ENABLED 1
#endif
#endif

#ifndef GGML_EXAMPLES_NVTX_ENABLED
#define GGML_EXAMPLES_NVTX_ENABLED 0
#endif

namespace ggml_nvtx {

inline uint32_t
color_for_name(const char* name) {
    static constexpr uint32_t palette[] = {
        0xff4e79a7, 0xfff28e2b, 0xffe15759, 0xff76b7b2, 0xff59a14f, 0xffedc949,
        0xffaf7aa1, 0xffff9da7, 0xff9c755f, 0xffbab0ab, 0xff499894, 0xffd37295,
    };

    uint32_t hash = 2166136261u;
    for (const char* p = name; p && *p; ++p) {
        hash ^= (uint8_t)*p;
        hash *= 16777619u;
    }
    return palette[hash % (sizeof(palette) / sizeof(palette[0]))];
}

#if GGML_EXAMPLES_NVTX_ENABLED

inline void
push(const char* name) {
    nvtxEventAttributes_t event = {};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = color_for_name(name);
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    nvtxRangePushEx(&event);
}

inline void
pop() {
    nvtxRangePop();
}

inline void
mark(const char* name) {
    nvtxEventAttributes_t event = {};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = color_for_name(name);
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    nvtxMarkEx(&event);
}

#else

inline void
push(const char* name) {
    (void)name;
}

inline void
pop() {}

inline void
mark(const char* name) {
    (void)name;
}

#endif

class range {
   public:
    explicit range(const char* name) : active(name && name[0]) {
        if (active) {
            push(name);
        }
    }

    ~range() {
        if (active) {
            pop();
        }
    }

    range(const range&) = delete;
    range& operator=(const range&) = delete;

   private:
    bool active;
};

}  // namespace ggml_nvtx
