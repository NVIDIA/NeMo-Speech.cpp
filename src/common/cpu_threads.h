// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Default CPU thread count of the compute backends.
#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace nemo_speech::common {

// CPU limit in a cgroup v2 cpu.max file ("<quota> <period>" in microseconds, or "max <period>"),
// rounded up to whole CPUs. 0 = no limit, also when the file is missing or unreadable.
inline int
cgroup_cpu_limit(const char* cpu_max_path) {
    long long quota = 0;
    long long period = 0;
    std::FILE* f = std::fopen(cpu_max_path, "r");
    if (!f)
        return 0;
    const bool parsed = std::fscanf(f, "%lld %lld", &quota, &period) == 2;
    std::fclose(f);
    if (!parsed || quota <= 0 || period <= 0)
        return 0;
    const long long cpus = quota / period + (quota % period != 0 ? 1 : 0);
    return static_cast<int>(std::min<long long>(cpus, INT_MAX));
}

// min(8, CPUs this process may use). More busy-waiting CPU workers than usable CPUs slows
// the graphs down, so on Linux the count follows the CPU affinity mask and the cgroup v2 CPU
// quota as seen at /sys/fs/cgroup/cpu.max (a container limited to 4 CPUs gets 4 threads,
// whatever the host has).
inline int
default_cpu_threads() {
    int cpus = static_cast<int>(std::thread::hardware_concurrency());
#if defined(__linux__)
    // The calling thread's mask, which the backend's workers inherit.
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        cpus = CPU_COUNT(&set);
    if (const int limit = cgroup_cpu_limit("/sys/fs/cgroup/cpu.max"); limit > 0)
        cpus = std::min(cpus, limit);
#endif
    return cpus > 0 ? std::min(8, cpus) : 8;
}

}  // namespace nemo_speech::common
