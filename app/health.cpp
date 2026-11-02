// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "cli_util.h"
#include "commands.h"

namespace {

struct Endpoint {
    std::string origin;
    std::string path;
};

Endpoint
parse_endpoint(const std::string& url) {
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0)
        throw std::invalid_argument("--url must begin with http:// or https://");
    const auto scheme = url.find("://");
    const auto slash = url.find('/', scheme + 3);
    return {
        slash == std::string::npos ? url : url.substr(0, slash),
        slash == std::string::npos ? "/ready" : url.substr(slash)};
}

}  // namespace

void
print_health_help(const char* program) {
    std::printf(
        "Usage: %s health [options]\n\n"
        "Check the readiness endpoint of a running NeMo-Speech.cpp server.\n\n"
        "  --url URL       Endpoint (default: http://127.0.0.1:8080/ready)\n"
        "  --timeout SEC   Per-request timeout (default: 2)\n"
        "  --wait SEC      Retry until ready for this long (default: 0)\n"
        "  --quiet         Suppress a successful response body\n",
        program);
}

int
command_health(int argc, char** argv) {
    try {
        std::string url = "http://127.0.0.1:8080/ready";
        int timeout = 2;
        int wait = 0;
        bool quiet = cli_quiet();
        auto value = [&](int& index, const std::string& option) {
            if (++index >= argc)
                throw std::invalid_argument(option + " requires a value");
            return std::string(argv[index]);
        };
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (is_help_argument(arg)) {
                print_health_help("nemo-speech");
                return 0;
            }
            if (arg == "--url")
                url = value(i, arg);
            else if (arg == "--timeout")
                timeout = parse_int(value(i, arg), arg, 1, 300);
            else if (arg == "--wait")
                wait = parse_int(value(i, arg), arg, 0, 3600);
            else if (arg == "--quiet")
                quiet = true;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        const auto endpoint = parse_endpoint(url);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(wait);
        std::string last_error;
        do {
            const auto now = std::chrono::steady_clock::now();
            if (!last_error.empty() && wait > 0 && now >= deadline)
                break;
            auto request_budget = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::seconds(timeout));
            if (wait > 0) {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
                request_budget =
                    std::min(request_budget, std::max(remaining, std::chrono::microseconds(1)));
            }
            httplib::Client client(endpoint.origin);
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
            client.enable_server_certificate_verification(true);
#endif
            client.set_connection_timeout(request_budget);
            client.set_read_timeout(request_budget);
            const auto response = client.Get(endpoint.path);
            if (response && response->status == 200) {
                if (cli_json()) {
                    std::printf(
                        "{\"ready\":true,\"url\":\"%s\",\"response_body\":\"%s\"}\n",
                        json_escape(url).c_str(), json_escape(response->body).c_str());
                } else if (!quiet) {
                    std::printf("%s\n", response->body.c_str());
                }
                return 0;
            }
            last_error = response ? "HTTP " + std::to_string(response->status)
                                  : httplib::to_string(response.error());
            const auto retry_time = std::chrono::steady_clock::now();
            if (retry_time >= deadline)
                break;
            std::this_thread::sleep_for(
                std::min(
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::milliseconds(250)),
                    deadline - retry_time));
        } while (true);
        return print_cli_error(
            "health", "not ready (" + last_error + ")", kCliExitRuntime, "not_ready");
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error("health", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("health", error);
    }
}
