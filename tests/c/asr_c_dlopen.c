// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/* Dynamic-load smoke test: load the C ABI library and resolve the nemo_speech_asr_*
 * entry points at runtime. Proves a loader can use the library with no link-time
 * dependency and that the public symbols are reachable. (That ONLY nemo_speech_asr_* are
 * exported is asserted separately by the build's nm check.)
 *
 *   asr_c_dlopen [path/to/nemo_speech_asr_c.{dll,so}]
 */
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#define DL_DEFAULT "nemo_speech_asr_c.dll"
#else
#include <dlfcn.h>
#define DL_DEFAULT "libnemo_speech_asr_c.so.1"
#endif

typedef const char* (*ver_fn)(void);

int
main(int argc, char** argv) {
    const char* lib = (argc > 1) ? argv[1] : DL_DEFAULT;
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(lib);
    if (!h) {
        fprintf(stderr, "LoadLibrary(%s) failed: error %lu\n", lib, (unsigned long)GetLastError());
        return 1;
    }
    ver_fn ver = (ver_fn)(void*)GetProcAddress(h, "nemo_speech_asr_version");
    void* create = (void*)GetProcAddress(h, "nemo_speech_asr_create");
    void* next = (void*)GetProcAddress(h, "nemo_speech_asr_stream_next");
    if (!ver || !create || !next) {
        fprintf(stderr, "GetProcAddress failed for a nemo_speech_asr_* entry point\n");
        FreeLibrary(h);
        return 1;
    }
    printf("dlopen nemo_speech_asr_version: %s\n", ver());
    FreeLibrary(h);
#else
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", lib, dlerror());
        return 1;
    }
    ver_fn ver = (ver_fn)dlsym(h, "nemo_speech_asr_version");
    void* create = dlsym(h, "nemo_speech_asr_create");
    void* next = dlsym(h, "nemo_speech_asr_stream_next");
    if (!ver || !create || !next) {
        fprintf(stderr, "dlsym failed for a nemo_speech_asr_* entry point\n");
        dlclose(h);
        return 1;
    }
    printf("dlopen nemo_speech_asr_version: %s\n", ver());
    dlclose(h);
#endif
    printf("[dlopen OK]\n");
    return 0;
}
