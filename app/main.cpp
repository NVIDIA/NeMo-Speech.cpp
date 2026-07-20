// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cli_util.h"
#include "commands.h"

namespace {

void
on_fatal_signal(int signal_number) {
    const char* name = signal_number == SIGSEGV  ? "SIGSEGV"
                       : signal_number == SIGFPE ? "SIGFPE"
                                                 : "fatal signal";
    std::fprintf(stderr, "\n[nemo-speech] FATAL: caught %s; re-raising.\n", name);
    std::fflush(stderr);
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

std::string
unavailable_command(const std::string& command) {
    (void)command;
#if !defined(NEMO_SPEECH_CLI_ASR)
    if (command == "transcribe" || command == "bench")
        return "this build does not include ASR; rebuild with -DNEMO_SPEECH_BUILD_ASR=ON";
#endif
#if !defined(NEMO_SPEECH_CLI_DIAR)
    if (command == "diarize")
        return "this build does not include diarization; rebuild with "
               "-DNEMO_SPEECH_BUILD_DIAR=ON";
#endif
#if !defined(NEMO_SPEECH_CLI_NMT)
    if (command == "translate")
        return "this build does not include NMT; rebuild with -DNEMO_SPEECH_BUILD_NMT=ON";
#endif
#if !defined(NEMO_SPEECH_CLI_TTS)
    if (command == "synthesize")
        return "this build does not include TTS; rebuild with -DNEMO_SPEECH_BUILD_TTS=ON";
#endif
#if !defined(NEMO_SPEECH_CLI_HTTP)
    if (command == "serve" || command == "health")
        return "this build does not include HTTP serving; rebuild with "
               "-DNEMO_SPEECH_BUILD_HTTP=ON";
#endif
    return {};
}

void
print_help(const char* program) {
    std::printf(
        "NeMo-Speech.cpp %s\n\n"
        "Usage: %s <command> [options]\n\n"
        "Commands:\n"
#if defined(NEMO_SPEECH_CLI_ASR)
        "  transcribe   Transcribe an audio file or directory\n"
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        "  diarize      Identify speaker segments in an audio file\n"
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        "  translate    Translate text\n"
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        "  synthesize   Synthesize speech to a WAV file\n"
#endif
#if defined(NEMO_SPEECH_CLI_ASR)
        "  bench        Benchmark an end-to-end ASR workload\n"
#endif
        "  model        Inspect local GGUF metadata\n"
        "  doctor       Inspect runtime and device availability\n"
#if defined(NEMO_SPEECH_CLI_HTTP)
        "  health       Check a running local HTTP server\n"
        "  serve        Start the local API and playground\n"
#endif
        "  help         Show help for a command\n\n"
        "Global options:\n"
        "  -h, --help       Show help\n"
        "  --version        Show version\n"
        "  --json           Emit machine-readable results and errors\n"
        "  --quiet          Suppress non-result progress messages\n"
        "  --verbose        Emit additional diagnostics on stderr\n",
        NEMO_SPEECH_VERSION_STR, program);
}

}  // namespace

int
main(int argc, char** argv) {
    // Emit a useful last-resort diagnostic around model execution. Recovery is
    // unsafe after these signals, so restore the default handler and re-raise.
    std::signal(SIGSEGV, on_fatal_signal);
    std::signal(SIGFPE, on_fatal_signal);

    bool json = false;
    bool quiet = false;
    bool verbose = false;
    std::vector<char*> filtered;
    filtered.reserve(static_cast<size_t>(argc));
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json")
            json = true;
        else if (arg == "--quiet")
            quiet = true;
        else if (arg == "--verbose")
            verbose = true;
        else
            filtered.push_back(argv[i]);
    }
    configure_cli_output(json, quiet, verbose);
    if (quiet && verbose)
        return print_cli_error(
            "", "--quiet and --verbose cannot be used together", 2, "invalid_argument");
    argc = static_cast<int>(filtered.size());
    argv = filtered.data();

    if (argc == 1 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_help(argv[0]);
        return 0;
    }
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "version") == 0) {
        std::printf("nemo-speech %s\n", NEMO_SPEECH_VERSION_STR);
        return 0;
    }
    if (std::strcmp(argv[1], "help") == 0) {
        if (argc == 2) {
            print_help(argv[0]);
            return 0;
        }
#if defined(NEMO_SPEECH_CLI_ASR)
        if (std::strcmp(argv[2], "transcribe") == 0) {
            print_transcribe_help(argv[0]);
            return 0;
        }
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        if (std::strcmp(argv[2], "diarize") == 0) {
            print_diarize_help(argv[0]);
            return 0;
        }
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        if (std::strcmp(argv[2], "translate") == 0) {
            print_translate_help(argv[0]);
            return 0;
        }
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        if (std::strcmp(argv[2], "synthesize") == 0) {
            print_synthesize_help(argv[0]);
            return 0;
        }
#endif
#if defined(NEMO_SPEECH_CLI_ASR)
        if (std::strcmp(argv[2], "bench") == 0) {
            print_bench_help(argv[0]);
            return 0;
        }
#endif
        if (std::strcmp(argv[2], "model") == 0)
            print_model_help(argv[0]);
        else if (std::strcmp(argv[2], "doctor") == 0)
            print_doctor_help(argv[0]);
#if defined(NEMO_SPEECH_CLI_HTTP)
        else if (std::strcmp(argv[2], "health") == 0)
            print_health_help(argv[0]);
        else if (std::strcmp(argv[2], "serve") == 0)
            print_serve_help(argv[0]);
#endif
        else {
            const std::string unavailable = unavailable_command(argv[2]);
            if (!unavailable.empty())
                return print_cli_error(
                    argv[2], unavailable, kCliExitUnsupportedFeature, "unsupported_feature");
            return print_cli_error(
                "", "unknown command: " + std::string(argv[2]), kCliExitInvalidArgument,
                "invalid_argument");
        }
        return 0;
    }
#if defined(NEMO_SPEECH_CLI_ASR)
    if (std::strcmp(argv[1], "transcribe") == 0)
        return command_transcribe(argc - 2, argv + 2);
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    if (std::strcmp(argv[1], "diarize") == 0)
        return command_diarize(argc - 2, argv + 2);
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    if (std::strcmp(argv[1], "translate") == 0)
        return command_translate(argc - 2, argv + 2);
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    if (std::strcmp(argv[1], "synthesize") == 0)
        return command_synthesize(argc - 2, argv + 2);
#endif
#if defined(NEMO_SPEECH_CLI_ASR)
    if (std::strcmp(argv[1], "bench") == 0)
        return command_bench(argc - 2, argv + 2);
#endif
    if (std::strcmp(argv[1], "model") == 0)
        return command_model(argc - 2, argv + 2);
    if (std::strcmp(argv[1], "doctor") == 0)
        return command_doctor(argc - 2, argv + 2);
#if defined(NEMO_SPEECH_CLI_HTTP)
    if (std::strcmp(argv[1], "health") == 0)
        return command_health(argc - 2, argv + 2);
    if (std::strcmp(argv[1], "serve") == 0)
        return command_serve(argc - 2, argv + 2);
#endif
    const std::string unavailable = unavailable_command(argv[1]);
    if (!unavailable.empty())
        return print_cli_error(
            argv[1], unavailable, kCliExitUnsupportedFeature, "unsupported_feature");
    return print_cli_error(
        "", "unknown command: " + std::string(argv[1]) + " (run 'nemo-speech --help')",
        kCliExitInvalidArgument, "invalid_argument");
}
