// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Cross-platform popen/pclose. C++ has no standard subprocess API, and the only
// difference between platforms is the spelling (`_popen`/`_pclose` on MSVC) and
// pclose's return encoding - so the single platform branch lives here and call
// sites stay platform-agnostic. Uses an anonymous pipe (no temp file).
#include <cstdio>

#if !defined(_WIN32)
#include <sys/wait.h>  // WIFEXITED / WEXITSTATUS
#endif

namespace nemo_speech {

// Spawn `cmd` connected to a pipe (mode "r" to read its stdout, "w" to write its
// stdin). Returns nullptr on failure.
inline FILE*
open_pipe(const char* cmd, const char* mode) {
#if defined(_WIN32)
    return _popen(cmd, mode);
#else
    return popen(cmd, mode);
#endif
}

// Close a pipe from open_pipe and return the child's exit code (0 = success,
// nonzero = failure, -1 = error). Hides POSIX wait-status encoding so callers
// can just check `!= 0`.
inline int
close_pipe(FILE* pipe) {
#if defined(_WIN32)
    return _pclose(pipe);  // already the child's exit code
#else
    const int status = pclose(pipe);
    if (status == -1) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace nemo_speech
