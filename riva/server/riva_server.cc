// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo-speech Riva gRPC server.
// Hosts ASR, TTS, or both on one port depending on configured model paths.
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "grpc_asr.h"
#include "grpc_tts.h"
#include "parameter_parser.h"
#include "recognizer.h"
#include "tts/magpietts/config.h"
#include "tts/synthesizer.h"
#if defined(NEMO_SPEECH_BUILD_NMT)
#include "grpc_nmt.h"
#include "speech_translator.h"
#include "translator.h"
#endif

namespace {

enum class ServiceEnablement {
    Auto = -1,
    Off = 0,
    On = 1,
};

ServiceEnablement
parse_enablement(const std::string& key, const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "auto") {
        return ServiceEnablement::Auto;
    }
    return nemo_speech::common::detail::to_bool(key, value) ? ServiceEnablement::On
                                                            : ServiceEnablement::Off;
}

struct AsrServerConfig {
    ServiceEnablement enabled = ServiceEnablement::Auto;
    nemo_speech::asr::RecognizerConfig recognizer;

    void Register(nemo_speech::common::ParameterParser& parser) {
        parser.Register(
            "enabled",
            [this](const std::string& value) { enabled = parse_enablement("asr.enabled", value); },
            "Enable ASR service: auto, true, or false");
        recognizer.Register(parser);
    }
};

struct TtsServerConfig {
    ServiceEnablement enabled = ServiceEnablement::Auto;
    nemo_speech::tts::MagpieTtsServerConfig server;

    void Register(nemo_speech::common::ParameterParser& parser) {
        parser.Register(
            "enabled",
            [this](const std::string& value) { enabled = parse_enablement("tts.enabled", value); },
            "Enable TTS service: auto, true, or false");
        server.Register(parser);
    }
};

#if defined(NEMO_SPEECH_BUILD_NMT)
struct NmtServerConfig {
    ServiceEnablement enabled = ServiceEnablement::Auto;
    nemo_speech::nmt::TranslatorConfig translator;

    void Register(nemo_speech::common::ParameterParser& parser) {
        parser.Register(
            "enabled",
            [this](const std::string& value) { enabled = parse_enablement("nmt.enabled", value); },
            "Enable NMT service: auto, true, or false");
        translator.Register(parser);
    }
};
#endif

struct RivaServerConfig {
    AsrServerConfig asr;
    TtsServerConfig tts;
#if defined(NEMO_SPEECH_BUILD_NMT)
    NmtServerConfig nmt;
#endif

    void Register(nemo_speech::common::ParameterParser& parser) {
        parser.Register("asr", asr);
        parser.Register("tts", tts);
#if defined(NEMO_SPEECH_BUILD_NMT)
        parser.Register("nmt", nmt);
#endif
    }
};

void
configure_cuda_graph_cache_defaults() {
    if (std::getenv("GGML_CUDA_GRAPH_EVICT_AFTER_MS") != nullptr) {
        return;
    }
    // putenv (not setenv) so there is no _WIN32 branch; the string must outlive
    // the call, hence static.
    static char kv[] = "GGML_CUDA_GRAPH_EVICT_AFTER_MS=0";
    putenv(kv);
    std::cerr << "[riva_server] defaulting GGML_CUDA_GRAPH_EVICT_AFTER_MS=0"
              << " to keep CUDA graphs resident across TTS requests\n";
}

bool
asr_configured(const AsrServerConfig& cfg) {
    return !cfg.recognizer.model.path.empty();
}

bool
tts_configured(const TtsServerConfig& cfg) {
    return !cfg.server.runtime.magpie_model.empty() && !cfg.server.runtime.codec_model.empty() &&
           !cfg.server.tokenizer_model_dir.empty();
}

#if defined(NEMO_SPEECH_BUILD_NMT)
bool
nmt_configured(const NmtServerConfig& cfg) {
    return !cfg.translator.model.path.empty();
}
#endif

bool
enabled(ServiceEnablement setting, bool configured) {
    switch (setting) {
        case ServiceEnablement::Auto:
            return configured;
        case ServiceEnablement::Off:
            return false;
        case ServiceEnablement::On:
            return true;
    }
    return false;
}

void
print_usage(const char* prog) {
    RivaServerConfig cfg;
    nemo_speech::common::ParameterParser parser;
    cfg.Register(parser);
    std::cerr << "Usage: " << prog << " [--asr.model.path ASR.gguf] "
              << "[--tts.magpie-model TTS.gguf --tts.codec-model CODEC.gguf "
              << "--tts.tokenizer-model-dir DIR] [--bind HOST:PORT] [--config FILE.yaml]\n"
              << "                  [--<dotted.key>=VALUE ...] [legacy flags ...]\n\n"
              << "  --bind HOST:PORT   listen address (default 0.0.0.0:50051)\n"
              << "  --config FILE      YAML config file (applied before env and CLI)\n\n"
              << "Services are enabled automatically when their required model config is present.\n"
              << "Use --asr.enabled true|false or --tts.enabled true|false to override.\n\n"
              << "Config keys (precedence: defaults < --config YAML < env < CLI):\n"
              << parser.Help() << "  Env: NEMO_SPEECH_<KEY> (dotted key uppercased, '.'->'_').\n";
}

std::unique_ptr<grpc::Server> g_server;

// Set from the signal handler. The handler is restricted to async-signal-safe
// operations (writing an atomic, no allocations / no mutex acquisitions),
// because calling grpc::Server::Shutdown() from a signal context interrupts
// whatever the main thread was doing inside gRPC and re-enters its internal
// mutexes. A dedicated watcher thread polls this flag and calls Shutdown from
// normal thread context.
std::atomic<bool> g_shutdown_requested{false};

void
on_signal(int) {
    g_shutdown_requested.store(true, std::memory_order_relaxed);
}

// Last-resort crash diagnostic for the unrecoverable fault class - a genuine
// segfault, or a GPU/ggml fault that surfaces as one. These can't be caught or
// recovered (process state is corrupt), so we don't try: we log a clear line and
// re-raise the default handler, letting the exit code / core dump reflect the
// fault so a supervisor (systemd/k8s) restarts the process. Handles the two
// fault signals that exist on every platform (SIGSEGV, SIGFPE) - fully portable,
// no per-OS backtrace code (use a core dump / debugger for the stack).
void
on_fatal_signal(int sig) {
    const char* name = (sig == SIGSEGV) ? "SIGSEGV" : (sig == SIGFPE) ? "SIGFPE" : "fatal signal";
    std::fprintf(stderr, "\n[riva_server] FATAL: caught %s; re-raising.\n", name);
    std::fflush(stderr);
    signal(sig, SIG_DFL);
    raise(sig);
}

}  // namespace

int
main(int argc, char** argv) {
    // Crash diagnostics for the unrecoverable fault class (segfault / bus / FPE,
    // including GPU faults that surface as one). Installed first so model load
    // and warmup are covered too. SIGABRT is left to ggml's handler.
    signal(SIGSEGV, on_fatal_signal);
    signal(SIGFPE, on_fatal_signal);

    std::string bind_addr = "0.0.0.0:50051";
    std::string config_file;
    RivaServerConfig cfg;
    nemo_speech::common::ParameterParser parser;
    cfg.Register(parser);

    // Pre-scan for --config so the YAML file is applied before env and CLI
    // (precedence: defaults < file < env < CLI).
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_file = argv[i + 1];
        } else if (arg.rfind("--config=", 0) == 0) {
            config_file = arg.substr(9);
        }
    }

    try {
        if (!config_file.empty()) {
            parser.ApplyYaml(config_file);
        }
        parser.ApplyEnv();

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--bind" || arg == "-b") {
                if (i + 1 >= argc) {
                    std::cerr << "--bind needs a value\n";
                    return 1;
                }
                bind_addr = argv[++i];
                continue;
            }
            if (arg.rfind("--bind=", 0) == 0) {
                bind_addr = arg.substr(7);
                continue;
            }
            if (arg == "--config" || arg == "-c") {
                if (i + 1 >= argc) {
                    std::cerr << arg << " requires a value\n";
                    return 1;
                }
                ++i;
                continue;
            }
            if (arg.rfind("--config=", 0) == 0) {
                continue;
            }
            bool consumed_next = false;
            if (parser.ParseCliArg(arg, (i + 1 < argc) ? argv[i + 1] : nullptr, &consumed_next)) {
                if (consumed_next) {
                    ++i;
                }
                continue;
            }
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    const bool asr_has_config = asr_configured(cfg.asr);
    const bool tts_has_config = tts_configured(cfg.tts);
    const bool enable_asr = enabled(cfg.asr.enabled, asr_has_config);
    const bool enable_tts = enabled(cfg.tts.enabled, tts_has_config);
#if defined(NEMO_SPEECH_BUILD_NMT)
    const bool nmt_has_config = nmt_configured(cfg.nmt);
    const bool enable_nmt = enabled(cfg.nmt.enabled, nmt_has_config);
#else
    const bool enable_nmt = false;
#endif

    if (!enable_asr && !enable_tts && !enable_nmt) {
        std::cerr << "No service enabled. Configure ASR model path, TTS model paths, NMT model "
                     "path, or set --asr.enabled/--tts.enabled/--nmt.enabled explicitly.\n";
        print_usage(argv[0]);
        return 1;
    }
    if (enable_asr && !asr_has_config) {
        std::cerr << "ASR is enabled but asr.model.path is missing\n";
        return 1;
    }
    if (enable_tts && !tts_has_config) {
        std::cerr << "TTS is enabled but --tts.magpie-model, --tts.codec-model, and "
                     "--tts.tokenizer-model-dir are required\n";
        return 1;
    }
#if defined(NEMO_SPEECH_BUILD_NMT)
    if (enable_nmt && !nmt_has_config) {
        std::cerr << "NMT is enabled but nmt.model.path is missing\n";
        return 1;
    }
#endif
    if (enable_asr && (cfg.asr.recognizer.streaming.chunk_size <= 0.0f ||
                       cfg.asr.recognizer.streaming.ctc_left_padding < 0.0f ||
                       cfg.asr.recognizer.streaming.ctc_right_padding < 0.0f)) {
        std::cerr << "Invalid streaming geometry: chunk_size must be > 0, paddings >= 0.\n";
        return 1;
    }

    // The skinny-q8 ggml kernel (see ggml-patches/0005) is an ASR-encoder
    // optimization; its in-place repack is unsafe for the NMT decoder and its
    // padded single-token decode is slower than the stock path. The flag is read
    // once, on the first skinny repack, so set the right env process-wide before
    // any service warms up:
    //   * NMT alone: disable skinny (GGML_SKINNY_Q8=0). TTS is fp16 and does not
    //     use skinny.
    //   * NMT + ASR: keep skinny for the ASR encoder, force the safe non-in-place
    //     repack (GGML_SKINNY_Q8_INPLACE=0); NMT then runs skinny.
    if (enable_nmt) {
        if (enable_asr) {
            static char kv[] = "GGML_SKINNY_Q8_INPLACE=0";
            putenv(kv);
            std::cerr << "[riva_server] NMT+ASR: forcing GGML_SKINNY_Q8_INPLACE=0 "
                         "(keep ASR skinny-q8, non-in-place)\n";
        } else {
            static char kv[] = "GGML_SKINNY_Q8=0";
            putenv(kv);
            std::cerr << "[riva_server] NMT without ASR: forcing GGML_SKINNY_Q8=0 "
                         "(disable skinny-q8; faster decode)\n";
        }
    }

    // Core capabilities outlive the protocol adapters.
    std::shared_ptr<nemo_speech::asr::Recognizer> recognizer;
    std::shared_ptr<nemo_speech::tts::Synthesizer> synthesizer;
#if defined(NEMO_SPEECH_BUILD_NMT)
    std::shared_ptr<nemo_speech::nmt::Translator> translator;
    std::shared_ptr<nemo_speech::speech::SpeechTranslator> speech_translator;
#endif

    std::unique_ptr<nemo_speech::GrpcAsrService> asr_service;
    std::unique_ptr<nemo_speech::GrpcTtsService> tts_service;
#if defined(NEMO_SPEECH_BUILD_NMT)
    std::unique_ptr<nemo_speech::GrpcNmtService> nmt_service;
#endif

    if (enable_asr) {
        std::cerr << "[riva_server] loading ASR model: " << cfg.asr.recognizer.model.path << "\n";
        try {
            recognizer =
                std::make_shared<nemo_speech::asr::Recognizer>(std::move(cfg.asr.recognizer));
            // Force lazy graph/session setup before attaching any transport.
            std::cerr << "[riva_server] warming up ASR\n";
            recognizer->warmup();
        }
        catch (const std::exception& e) {
            std::cerr << "[riva_server] failed to initialize ASR: " << e.what() << "\n";
            return 3;
        }
        std::cerr << "[riva_server] ASR warmup complete\n";
    }

    if (enable_tts) {
        configure_cuda_graph_cache_defaults();
        std::cerr << "[riva_server] loading MagpieTTS model: "
                  << cfg.tts.server.runtime.magpie_model << "\n";
        std::cerr << "[riva_server] loading NanoCodec model: " << cfg.tts.server.runtime.codec_model
                  << "\n";
        try {
            nemo_speech::tts::SynthesizerConfig synthesizer_config;
            synthesizer_config.runtime = std::move(cfg.tts.server.runtime);
            synthesizer_config.tokenizer_model_dir = std::move(cfg.tts.server.tokenizer_model_dir);
            synthesizer_config.text_normalizer_model_dir = std::move(cfg.tts.server.tn_model_dir);
            synthesizer_config.tokenizer = cfg.tts.server.tokenizer_config;
            synthesizer_config.default_language_code =
                std::move(cfg.tts.server.default_language_code);
            synthesizer_config.default_voice_name = std::move(cfg.tts.server.default_voice_name);
            synthesizer =
                std::make_shared<nemo_speech::tts::Synthesizer>(std::move(synthesizer_config));
            if (cfg.tts.server.warmup) {
                std::cerr << "[riva_server] warming up TTS\n";
                synthesizer->warmup(cfg.tts.server.warmup_text, cfg.tts.server.warmup_steps);
                std::cerr << "[riva_server] TTS warmup complete\n";
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[riva_server] failed to initialize TTS: " << e.what() << "\n";
            return 1;
        }
    }

#if defined(NEMO_SPEECH_BUILD_NMT)
    if (enable_nmt) {
        std::cerr << "[riva_server] loading NMT model: " << cfg.nmt.translator.model.path << "\n";
        try {
            translator =
                std::make_shared<nemo_speech::nmt::Translator>(std::move(cfg.nmt.translator));
            if (recognizer) {
                speech_translator = std::make_shared<nemo_speech::speech::SpeechTranslator>(
                    recognizer, translator, synthesizer);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[riva_server] failed to initialize NMT: " << e.what() << "\n";
            return 1;
        }
        const char* cascade = !enable_asr  ? "text-only (no ASR for speech cascades)"
                              : enable_tts ? "speech-to-text + speech-to-speech cascades ready"
                                           : "speech-to-text cascade ready (no TTS)";
        std::cerr << "[riva_server] NMT loaded; " << cascade << "\n";
    }
#endif

    // Construct protocol adapters from the initialized core capabilities.
    if (recognizer)
        asr_service = std::make_unique<nemo_speech::GrpcAsrService>(recognizer);
    if (synthesizer) {
        tts_service =
            std::make_unique<nemo_speech::GrpcTtsService>(synthesizer, cfg.tts.server.benchmark);
    }
#if defined(NEMO_SPEECH_BUILD_NMT)
    if (translator) {
        nmt_service = std::make_unique<nemo_speech::GrpcNmtService>(translator, speech_translator);
    }
#endif

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    int bound_port = 0;
    builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials(), &bound_port);
    if (asr_service) {
        builder.RegisterService(asr_service.get());
    }
    if (tts_service) {
        builder.RegisterService(tts_service.get());
    }
#if defined(NEMO_SPEECH_BUILD_NMT)
    if (nmt_service) {
        builder.RegisterService(nmt_service.get());
    }
#endif
    // Preserve the former TTS server's large receive/send limits in the merged
    // binary. This also keeps ASR offline Recognize requests at least as capable
    // as the previous 64 MB receive cap.
    builder.SetMaxReceiveMessageSize(INT32_MAX);
    builder.SetMaxSendMessageSize(INT32_MAX);

    g_server = builder.BuildAndStart();
    if (!g_server || bound_port == 0) {
        std::cerr << "[riva_server] failed to start on " << bind_addr << "\n";
        return 2;
    }
    std::string services;
    auto add_service = [&services](const char* name) {
        if (!services.empty())
            services += "+";
        services += name;
    };
    if (asr_service)
        add_service("ASR");
    if (tts_service)
        add_service("TTS");
#if defined(NEMO_SPEECH_BUILD_NMT)
    if (nmt_service)
        add_service("NMT");
#endif
    std::cerr << "[riva_server] listening on " << bind_addr << " (" << services << ")\n";

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // Watcher: poll the signal-handler-set flag; when set, call Shutdown
    // from a normal thread context. Wait() returns once Shutdown completes;
    // we then signal the watcher to exit in case Wait returned for another
    // reason.
    std::thread shutdown_watcher([&]() {
        while (!g_shutdown_requested.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (g_server) {
            // Deadline: an active client must not be able to hold off SIGTERM
            // forever; in-flight RPCs past the grace period are cancelled.
            g_server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(10));
        }
    });

    g_server->Wait();
    g_shutdown_requested.store(true, std::memory_order_relaxed);
    shutdown_watcher.join();
    std::cerr << "[riva_server] shutdown complete\n";
    return 0;
}
