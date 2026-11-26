// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

#include "cpu_threads.h"

namespace {

void
require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

int
limit_of(const std::string& path, const char* cpu_max) {
    std::ofstream(path) << cpu_max;
    return nemo_speech::common::cgroup_cpu_limit(path.c_str());
}

}  // namespace

int
main() {
    using nemo_speech::common::cgroup_cpu_limit;
    using nemo_speech::common::default_cpu_threads;

    // A directory of this run only: create_directory fails on a name that exists, so runs do not
    // share the fixture and nothing placed in the temporary directory beforehand is followed.
    std::filesystem::path dir;
    std::random_device random;
    do {
        dir = std::filesystem::temp_directory_path() /
              ("nemo_speech_test_cpu_threads_" + std::to_string(random()));
    } while (!std::filesystem::create_directory(dir));
    const std::string path = (dir / "cpu.max").string();
    require(limit_of(path, "max 100000\n") == 0, "max is no limit");
    require(limit_of(path, "200000 100000\n") == 2, "two CPUs");
    require(limit_of(path, "250000 100000\n") == 3, "a fractional quota rounds up");
    require(limit_of(path, "0 100000\n") == 0, "a zero quota is no limit");
    require(limit_of(path, "9223372036854775807 1\n") == INT_MAX, "a huge quota does not wrap");
    require(limit_of(path, "") == 0, "an empty file is no limit");
    std::filesystem::remove_all(dir);
    require(cgroup_cpu_limit(path.c_str()) == 0, "a missing file is no limit");

    const int threads = default_cpu_threads();
    std::cout << "default_cpu_threads: " << threads << '\n';
    require(threads >= 1 && threads <= 8, "default is within [1, 8]");
#if defined(__linux__)
    // The mask does not fit a cpu_set_t on hosts with more than 1024 CPUs; the library falls
    // back to the hardware thread count there.
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        require(threads <= CPU_COUNT(&allowed), "default does not exceed the affinity mask");
        // Pin the process to one of its CPUs: the default follows the mask, not the host.
        int cpu = 0;
        while (!CPU_ISSET(cpu, &allowed)) ++cpu;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        require(sched_setaffinity(0, sizeof(one), &one) == 0, "sched_setaffinity");
        require(default_cpu_threads() == 1, "one usable CPU gives one thread");
    }
#endif
    std::cout << "PASS\n";
    return 0;
}
