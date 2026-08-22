// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#include "http_server.h"
#include "json.h"
#include "model_utils.h"
#include "parameter_parser.h"
#if defined(NEMO_SPEECH_CLI_TTS)
#include "config.h"
#endif

namespace {

std::atomic<bool> shutdown_requested{false};

template <typename Callback>
class ScopeExit {
   public:
    explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
    ~ScopeExit() { callback_(); }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

   private:
    Callback callback_;
};

void
request_shutdown(int) {
    shutdown_requested.store(true, std::memory_order_relaxed);
}

[[maybe_unused]] std::string
optional_model(
    const std::string& reference, const std::string& role, const std::string& description,
    bool required, bool directory = false) {
    if (reference.empty()) {
        if (!required)
            return {};
        if (role.empty())
            throw MissingModelError(description + " path is required");
    }
    return (role.empty() ? (directory ? require_model_directory(reference, description)
                                      : require_model_file(reference, description))
                         : (directory ? resolve_model_directory(reference, role, description)
                                      : resolve_model_file(reference, role, description)))
        .string();
}

#if defined(NEMO_SPEECH_CLI_ASR) || defined(NEMO_SPEECH_CLI_NMT) || defined(NEMO_SPEECH_CLI_TTS)
int
parse_enablement(const std::string& key, const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "auto")
        return -1;
    return nemo_speech::common::detail::to_bool(key, value) ? 1 : 0;
}
#endif

void
validate_compiled_feature_options(int argc, char** argv) {
    std::vector<std::string> issues;
    [[maybe_unused]] auto add = [&](const std::string& option, const std::string& feature) {
        issues.push_back(option + " requires a build with " + feature);
    };
    for (int i = 0; i < argc; ++i) {
        const std::string option = argv[i];
#if !defined(NEMO_SPEECH_CLI_ASR)
        if (option == "--asr-model" || option == "--vad-model" || option == "--pnc-model" ||
            option == "--itn-model-dir" || option.rfind("--asr.", 0) == 0)
            add(option, "NEMO_SPEECH_BUILD_ASR=ON");
#endif
#if !defined(NEMO_SPEECH_CLI_DIAR)
        if (option == "--diar-model")
            add(option, "NEMO_SPEECH_BUILD_DIAR=ON");
#endif
#if !defined(NEMO_SPEECH_CLI_NMT)
        if (option == "--nmt-model" || option.rfind("--nmt.", 0) == 0)
            add(option, "NEMO_SPEECH_BUILD_NMT=ON");
#endif
#if !defined(NEMO_SPEECH_CLI_TTS)
        if (option == "--tts-model" || option == "--codec-model" || option == "--tokenizer-dir" ||
            option == "--tn-model-dir" || option.rfind("--tts.", 0) == 0)
            add(option, "NEMO_SPEECH_BUILD_TTS=ON");
#endif
    }
    if (!issues.empty()) {
        std::ostringstream message;
        message << "configuration uses unavailable features:";
        for (const auto& issue : issues) message << "\n  - " << issue;
        throw std::invalid_argument(message.str());
    }
}

int
run_server(int argc, char** argv) {
    validate_compiled_feature_options(argc, argv);
    nemo_speech::http::ServerConfig server_config;
    std::string config_file;
    std::string device_name = "auto";
    int gpu = default_gpu_index();
    bool device_set = false;
    bool no_warmup = false;
    bool open_browser = false;
#if defined(NEMO_SPEECH_CLI_ASR)
    nemo_speech::asr::RecognizerConfig asr_config;
    asr_config.backend.gpu = default_gpu_index();
    asr_config.batching.enabled = true;
    std::string asr_model, vad_model, pnc_model, itn_model;
    int asr_enabled = -1;
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    std::string diar_model;
    nemo_speech::asr::DiarConfig diar_config;
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    nemo_speech::nmt::TranslatorConfig nmt_config;
    nmt_config.backend.gpu = default_gpu_index();
    std::string nmt_model;
    int nmt_enabled = -1;
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    nemo_speech::tts::MagpieTtsServerConfig tts_config;
    std::string tts_model, codec_model, tokenizer_model, tn_model;
    int tts_enabled = -1;
#endif
    nemo_speech::common::ParameterParser parser;
    parser.Register(
        "http.enabled",
        [&](const std::string& value) {
            if (!nemo_speech::common::detail::to_bool("http.enabled", value))
                throw std::invalid_argument("http.enabled=false is not valid for the HTTP server");
        },
        "Enable the HTTP listener", {}, true);
    parser.Register("http.host", &server_config.address, "HTTP bind address");
    parser.Register(
        "http.port",
        [&](const std::string& value) {
            server_config.port = parse_int(value, "http.port", 1, 65535);
        },
        "HTTP port");
    parser.Register(
        "http.threads",
        [&](const std::string& value) {
            server_config.threads = parse_int(value, "http.threads", 1, 1024);
        },
        "Bounded HTTP worker count");
    parser.Register(
        "http.max-upload-mb",
        [&](const std::string& value) {
            server_config.max_upload_bytes =
                static_cast<size_t>(parse_int(value, "http.max-upload-mb", 1, 16384)) * 1024 * 1024;
        },
        "Maximum HTTP request body in MiB");
    parser.Register(
        "http.read-timeout",
        [&](const std::string& value) {
            server_config.read_timeout_seconds = parse_int(value, "http.read-timeout", 1, 3600);
        },
        "HTTP socket read timeout in seconds");
    parser.Register(
        "http.write-timeout",
        [&](const std::string& value) {
            server_config.write_timeout_seconds = parse_int(value, "http.write-timeout", 1, 3600);
        },
        "HTTP socket write timeout in seconds");
    parser.Register("http.access-log", &server_config.access_log, "Log completed HTTP requests");
    parser.Register(
        "http.log-format",
        [&](const std::string& value) {
            if (value == "text")
                server_config.json_logs = false;
            else if (value == "json")
                server_config.json_logs = true;
            else
                throw std::invalid_argument("http.log-format must be text or json");
        },
        "HTTP access-log format: text or json");
    parser.Register("http.api-key", &server_config.api_key, "Bearer token required by API routes");
    parser.Register("http.cors-origin", &server_config.cors_origin, "Allowed browser origin");
    parser.Register("http.tls-cert", &server_config.tls_certificate, "TLS certificate path");
    parser.Register("http.tls-key", &server_config.tls_private_key, "TLS private-key path");
    parser.Register(
        "http.playground", &server_config.enable_playground,
        "Serve the embedded browser playground");
#if defined(NEMO_SPEECH_CLI_ASR)
    parser.Register("asr", asr_config);
    parser.Register(
        "asr.enabled",
        [&](const std::string& value) { asr_enabled = parse_enablement("asr.enabled", value); },
        "Enable ASR: auto, true, or false");
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    parser.Register("diar", diar_config);
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    parser.Register("nmt", nmt_config);
    parser.Register(
        "nmt.enabled",
        [&](const std::string& value) { nmt_enabled = parse_enablement("nmt.enabled", value); },
        "Enable NMT: auto, true, or false");
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    parser.Register("tts", tts_config);
    parser.Register(
        "tts.enabled",
        [&](const std::string& value) { tts_enabled = parse_enablement("tts.enabled", value); },
        "Enable TTS: auto, true, or false");
#endif
    auto value = [&](int& index, const std::string& option) {
        if (++index >= argc)
            throw std::invalid_argument(option + " requires a value");
        return std::string(argv[index]);
    };
    for (int i = 0; i < argc; ++i)
        if (std::string(argv[i]) == "--config")
            config_file = value(i, "--config");
    if (!config_file.empty())
        parser.ApplyYaml(config_file);
    parser.ApplyEnv("NEMO_SPEECH");
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config")
            ++i;
        else if (arg == "--host")
            server_config.address = value(i, arg);
        else if (arg == "--port")
            server_config.port = parse_int(value(i, arg), arg, 1, 65535);
        else if (arg == "--threads") {
            server_config.threads = parse_int(value(i, arg), arg, 1, 1024);
        } else if (arg == "--max-upload-mb")
            server_config.max_upload_bytes =
                static_cast<size_t>(parse_int(value(i, arg), arg, 1, 16384)) * 1024 * 1024;
        else if (arg == "--read-timeout")
            server_config.read_timeout_seconds = parse_int(value(i, arg), arg, 1, 3600);
        else if (arg == "--write-timeout")
            server_config.write_timeout_seconds = parse_int(value(i, arg), arg, 1, 3600);
        else if (arg == "--access-log")
            server_config.access_log = true;
        else if (arg == "--log-format") {
            const std::string format = value(i, arg);
            if (format == "text")
                server_config.json_logs = false;
            else if (format == "json")
                server_config.json_logs = true;
            else
                throw std::invalid_argument("--log-format must be text or json");
        } else if (arg == "--tls-cert")
            server_config.tls_certificate = value(i, arg);
        else if (arg == "--tls-key")
            server_config.tls_private_key = value(i, arg);
        else if (arg == "--api-key")
            server_config.api_key = value(i, arg);
        else if (arg == "--cors-origin")
            server_config.cors_origin = value(i, arg);
        else if (arg == "--no-ui")
            server_config.enable_playground = false;
        else if (arg == "--open")
            open_browser = true;
        else if (arg == "--gpu") {
            gpu = parse_int(value(i, arg), arg, -1, 1024);
            device_name = gpu < 0 ? "cpu" : "gpu:" + std::to_string(gpu);
            device_set = true;
        } else if (arg == "--device" || arg == "--backend") {
            device_name = value(i, arg);
            gpu = parse_device(device_name, arg);
            device_set = true;
        } else if (arg == "--no-warmup")
            no_warmup = true;
#if defined(NEMO_SPEECH_CLI_ASR)
        else if (arg == "--asr-model")
            asr_model = value(i, arg);
        else if (arg == "--vad-model")
            vad_model = value(i, arg);
        else if (arg == "--pnc-model")
            pnc_model = value(i, arg);
        else if (arg == "--itn-model-dir")
            itn_model = value(i, arg);
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        else if (arg == "--diar-model")
            diar_model = value(i, arg);
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        else if (arg == "--nmt-model")
            nmt_model = value(i, arg);
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        else if (arg == "--tts-model")
            tts_model = value(i, arg);
        else if (arg == "--codec-model")
            codec_model = value(i, arg);
        else if (arg == "--tokenizer-dir")
            tokenizer_model = value(i, arg);
        else if (arg == "--tn-model-dir")
            tn_model = value(i, arg);
#endif
        else if (!arg.empty() && arg.front() == '-') {
            bool consumed = false;
            if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                throw std::invalid_argument("unknown option: " + arg);
            if (consumed)
                ++i;
        } else {
            throw std::invalid_argument("unexpected argument: " + arg);
        }
    }

    if (cli_json())
        server_config.json_logs = true;
    if (open_browser && !server_config.enable_playground)
        throw std::invalid_argument("--open requires the HTTP playground");
    if (server_config.tls_certificate.empty() != server_config.tls_private_key.empty())
        throw std::invalid_argument("--tls-cert and --tls-key must be provided together");
    if (server_config.cors_origin.find_first_of("\r\n") != std::string::npos)
        throw std::invalid_argument("--cors-origin must not contain line breaks");

    std::vector<std::string> validation_errors;
    bool missing_model = false;
    [[maybe_unused]] auto resolve = [&](const std::string& label, const auto& resolver) {
        try {
            return resolver();
        }
        catch (const MissingModelError& error) {
            missing_model = true;
            validation_errors.push_back(label + ": " + error.what());
            return std::string();
        }
        catch (const std::exception& error) {
            validation_errors.push_back(label + ": " + error.what());
            return std::string();
        }
    };
#if defined(NEMO_SPEECH_CLI_ASR)
    const bool asr_requested = asr_enabled > 0 || !asr_model.empty() ||
                               !asr_config.model.path.empty() || !vad_model.empty() ||
                               !pnc_model.empty() || !itn_model.empty();
    const auto asr_path = asr_enabled == 0 ? std::string() : resolve("ASR model", [&] {
        return optional_model(
            asr_model.empty() ? asr_config.model.path : asr_model, "asr", "ASR model",
            asr_requested);
    });
    if (!asr_path.empty()) {
        if (device_set)
            asr_config.backend.gpu = gpu;
        asr_config.model.path = asr_path;
    }
    if (asr_enabled != 0 && (asr_requested || !asr_path.empty())) {
        asr_config.vad.model_path = resolve("VAD model", [&] {
            return optional_model(
                vad_model.empty() ? asr_config.vad.model_path : vad_model, "", "VAD model", false);
        });
#if defined(NEMO_SPEECH_CLI_DIAR)
        if (asr_config.diar.model_path.empty() && !diar_config.model_path.empty())
            asr_config.diar = diar_config;
        asr_config.diar.model_path = resolve("diarization model", [&] {
            return optional_model(
                diar_model.empty() ? asr_config.diar.model_path : diar_model, "diarization",
                "diarization model", false);
        });
#endif
        asr_config.postproc.pnc_model_path = resolve("punctuation model", [&] {
            return optional_model(
                pnc_model.empty() ? asr_config.postproc.pnc_model_path : pnc_model, "",
                "punctuation and capitalization model", false);
        });
        asr_config.postproc.itn_model_dir = resolve("ITN grammar", [&] {
            return optional_model(
                itn_model.empty() ? asr_config.postproc.itn_model_dir : itn_model, "", "ITN model",
                false, true);
        });
    }
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    std::string standalone_diar;
#if defined(NEMO_SPEECH_CLI_ASR)
    if (asr_path.empty())
#endif
        standalone_diar = resolve("diarization model", [&] {
            return optional_model(
                diar_model.empty() ? diar_config.model_path : diar_model, "diarization",
                "diarization model", false);
        });
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    const bool nmt_requested =
        nmt_enabled > 0 || !nmt_model.empty() || !nmt_config.model.path.empty();
    const auto nmt_path = nmt_enabled == 0 ? std::string() : resolve("NMT model", [&] {
        return optional_model(
            nmt_model.empty() ? nmt_config.model.path : nmt_model, "", "translation model",
            nmt_requested);
    });
    if (!nmt_path.empty()) {
        if (device_set)
            nmt_config.backend.gpu = gpu;
        nmt_config.model.path = nmt_path;
    }
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    const bool tts_requested = tts_enabled > 0 || !tts_model.empty() ||
                               !tts_config.runtime.magpie_model.empty() || !codec_model.empty() ||
                               !tokenizer_model.empty() || !tn_model.empty();
    const auto magpie_path = tts_enabled == 0 ? std::string() : resolve("TTS model", [&] {
        return optional_model(
            tts_model.empty() ? tts_config.runtime.magpie_model : tts_model, "tts",
            "MagpieTTS model", tts_requested);
    });
    if (tts_enabled != 0 && (tts_requested || !magpie_path.empty())) {
        tts_config.runtime.codec_model = resolve("TTS codec model", [&] {
            return optional_model(
                codec_model.empty() ? tts_config.runtime.codec_model : codec_model, "codec",
                "NanoCodec model", true);
        });
        tts_config.tokenizer_model_dir = resolve("TTS tokenizer", [&] {
            return optional_model(
                tokenizer_model.empty() ? tts_config.tokenizer_model_dir : tokenizer_model,
                "tokenizer", "tokenizer model", true, true);
        });
        tts_config.tn_model_dir = resolve("TTS normalization grammar", [&] {
            return optional_model(
                tn_model.empty() ? tts_config.tn_model_dir : tn_model, "",
                "text normalization model", false, true);
        });
    }
#endif
    if (!server_config.tls_certificate.empty()) {
        if (!std::filesystem::is_regular_file(server_config.tls_certificate))
            validation_errors.push_back(
                "TLS certificate: file does not exist: " + server_config.tls_certificate);
        if (!std::filesystem::is_regular_file(server_config.tls_private_key))
            validation_errors.push_back(
                "TLS private key: file does not exist: " + server_config.tls_private_key);
    }
    if (!validation_errors.empty()) {
        std::ostringstream message;
        message << "configuration validation failed:";
        for (const auto& error : validation_errors) message << "\n  - " << error;
        if (missing_model)
            throw MissingModelError(message.str());
        throw std::invalid_argument(message.str());
    }

    nemo_speech::EngineRegistryConfig registry_config;
#if defined(NEMO_SPEECH_CLI_ASR)
    asr_config.log_status = !cli_quiet() && !cli_json();
    registry_config.asr = !asr_path.empty();
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    registry_config.nmt = !nmt_path.empty();
#endif
    nemo_speech::EngineRegistry engines(registry_config);
    engines.set_device_label(device_set ? device_name : "auto");
#if defined(NEMO_SPEECH_CLI_ASR)
    if (!asr_path.empty())
        engines.load_asr(std::move(asr_config));
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    if (!standalone_diar.empty())
        engines.load_diarization(gpu, standalone_diar, diar_config.resolved_geometry());
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    if (!nmt_path.empty()) {
        nmt_config.verbose = cli_verbose();
        engines.load_nmt(std::move(nmt_config));
    }
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    if (!magpie_path.empty()) {
        bool tts_cuda = false;
#if defined(NEMO_SPEECH_CLI_CUDA)
        tts_cuda = device_name == "auto" || device_name == "cuda" ||
                   device_name.rfind("cuda:", 0) == 0 || device_name == "gpu" ||
                   device_name.rfind("gpu:", 0) == 0;
#endif
        if (device_set) {
            if (!tts_cuda) {
                tts_config.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                tts_config.runtime.sampling_backend =
                    nemo_speech::tts::MagpieBackendPreference::Cpu;
                tts_config.runtime.magpie_cpu = gpu < 0;
                tts_config.runtime.codec_cpu = gpu < 0;
            } else {
                tts_config.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cuda;
                tts_config.runtime.sampling_backend =
                    nemo_speech::tts::MagpieBackendPreference::Cuda;
            }
        }
        tts_config.runtime.magpie_model = magpie_path;
        tts_config.runtime.verbose = cli_verbose();
        nemo_speech::tts::SynthesizerConfig config;
        config.runtime = tts_config.runtime;
        config.tokenizer_model_dir = tts_config.tokenizer_model_dir;
        config.text_normalizer_model_dir = tts_config.tn_model_dir;
        config.tokenizer = tts_config.tokenizer_config;
        config.default_language_code = tts_config.default_language_code;
        config.default_voice_name = tts_config.default_voice_name;
        engines.load_tts(std::move(config));
    }
#endif
    if (!engines.ready())
        throw std::runtime_error(
            "no models were loaded; pass --asr-model, --diar-model, --tts-model, --nmt-model, "
            "or --config");
    if (!no_warmup) {
        nemo_speech::WarmupOptions warmup;
#if defined(NEMO_SPEECH_CLI_TTS)
        warmup.tts = tts_config.warmup;
        warmup.tts_text = tts_config.warmup_text;
        warmup.tts_steps = tts_config.warmup_steps;
#endif
        engines.warmup(warmup);
    }

    auto http_server = std::make_unique<nemo_speech::http::Server>(engines, server_config);
    shutdown_requested.store(false, std::memory_order_relaxed);
    const auto old_int = std::signal(SIGINT, request_shutdown);
    const auto old_term = std::signal(SIGTERM, request_shutdown);
    ScopeExit restore_signal_handlers([&] {
        std::signal(SIGINT, old_int);
        std::signal(SIGTERM, old_term);
    });
    std::thread shutdown_watcher([&] {
        while (!shutdown_requested.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (http_server)
            http_server->stop();
    });
    ScopeExit join_shutdown_watcher([&] {
        shutdown_requested.store(true, std::memory_order_relaxed);
        if (shutdown_watcher.joinable())
            shutdown_watcher.join();
    });
    bool ok = true;
    {
        if (server_config.address != "127.0.0.1" && server_config.address != "localhost" &&
            server_config.address != "::1") {
            std::fprintf(
                stderr,
                "warning: HTTP is listening beyond loopback; configure --api-key and TLS "
                "before exposing it to an untrusted network\n");
        }
        std::string browser_host = server_config.address;
        if (browser_host == "0.0.0.0" || browser_host == "::" || browser_host == "*")
            browser_host = "127.0.0.1";
        if (!browser_host.empty() && browser_host.find(':') != std::string::npos &&
            browser_host.front() != '[')
            browser_host = "[" + browser_host + "]";
        const std::string playground_url =
            std::string(server_config.tls_certificate.empty() ? "http://" : "https://") +
            browser_host + ":" + std::to_string(server_config.port) + "/";
        if (!http_server->bind()) {
            std::fprintf(
                stderr, "error: could not bind HTTP to %s:%d\n", server_config.address.c_str(),
                server_config.port);
            ok = false;
        } else {
            if (cli_json()) {
                nemo_speech::json::Value event(nemo_speech::json::Value::Object{});
                event["event"] = "listener.ready";
                event["transport"] = "http";
                event["url"] = playground_url;
                nemo_speech::json::Value::Array capabilities;
                for (const auto& capability : engines.capabilities())
                    capabilities.emplace_back(capability);
                event["capabilities"] = std::move(capabilities);
                std::printf("%s\n", event.dump().c_str());
            } else if (!cli_quiet()) {
                std::printf(
                    "HTTP API listening on %s://%s:%d\n",
                    server_config.tls_certificate.empty() ? "http" : "https",
                    server_config.address.c_str(), server_config.port);
                if (server_config.enable_playground)
                    std::printf("Playground: %s\n", playground_url.c_str());
            }
            if (open_browser && !open_url_in_browser(playground_url))
                std::fprintf(stderr, "warning: could not open the default browser\n");
            ok = http_server->listen_after_bind();
        }
    }
    return ok ? 0 : 1;
}
}  // namespace

void
print_serve_help(const char* program) {
    std::printf(
        "Usage: %s serve [options]\n\n"
        "Start the local speech HTTP API and browser playground. The transcription\n"
        "and speech routes expose OpenAI-compatible subsets. Models are provided as\n"
        "local paths, indexed names, or through a YAML configuration.\n\n"
        "Server:\n"
        "  --host ADDRESS          Bind address (default: 127.0.0.1)\n"
        "  --port N                HTTP port (default: 8080)\n"
        "  --threads N             Bounded HTTP worker count (default: 4)\n"
        "  --max-upload-mb N       Maximum request body (default: 512)\n"
        "  --read-timeout SEC      Socket read timeout (default: 30)\n"
        "  --write-timeout SEC     Socket write timeout (default: 30)\n"
        "  --access-log            Log completed HTTP requests\n"
        "  --log-format text|json  Access-log format (--json also selects JSON)\n"
        "  --api-key KEY           Require Authorization: Bearer KEY\n"
        "  --cors-origin ORIGIN    Allow one cross-origin browser origin\n"
        "  --tls-cert FILE         TLS certificate (requires --tls-key)\n"
        "  --tls-key FILE          TLS private key\n"
        "  --no-ui                 Disable the browser playground\n"
        "  --open                  Open the playground in the default browser\n"
        "  --http.* VALUE          Override any HTTP listener setting\n"
        "                          Realtime ASR: WebSocket /v1/realtime\n\n"
        "Models:\n"
#if defined(NEMO_SPEECH_CLI_ASR)
        "  --asr-model MODEL       ASR GGUF path or indexed model\n"
        "  --vad-model MODEL       Optional VAD model\n"
        "  --pnc-model MODEL       Optional punctuation model\n"
        "  --itn-model-dir MODEL   Optional ITN grammar directory\n"
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        "  --diar-model MODEL      Optional Sortformer path or indexed model\n"
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        "  --tts-model MODEL       Optional MagpieTTS path or indexed model\n"
        "  --codec-model MODEL     NanoCodec path or indexed model\n"
        "  --tokenizer-dir MODEL   TTS tokenizer directory or indexed model\n"
        "  --tn-model-dir MODEL    Optional TTS text-normalization assets\n"
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --nmt-model MODEL       Optional translation model\n"
#endif
        "  --device, --backend DEVICE\n"
        "                          auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  --skinny-q8 auto|on|off CUDA control: auto uses the safe Pascal fallback; off\n"
        "                          replaces GGML_SKINNY_Q8=0; on requires SM 8.0+\n"
        "  --suppress-cuda-graph-log Suppress only the repeated CUDA-graph architecture log\n"
        "  --config FILE           Apply YAML configuration\n"
#if defined(NEMO_SPEECH_CLI_ASR)
        "  --asr.* VALUE           Override ASR engine configuration\n"
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        "  --tts.* VALUE           Override TTS engine configuration\n"
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --nmt.* VALUE           Override NMT engine configuration\n"
#endif
        "  --no-warmup             Skip engine warmup\n",
        program);
}

int
command_serve(int argc, char** argv) {
    if (argc > 0 && is_help_argument(argv[0])) {
        print_serve_help("nemo-speech");
        return 0;
    }
    try {
        return run_server(argc, argv);
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error("serve", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("serve", error);
    }
}
