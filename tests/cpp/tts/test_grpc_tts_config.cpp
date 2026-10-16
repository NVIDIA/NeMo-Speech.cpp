// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpc_tts.h"
#include "tts/tokenizer/tokenizer.h"

namespace fs = std::filesystem;
namespace tts = nemo_speech::tts;

namespace {

fs::path
source_path(const char* relative) {
#ifdef NEMO_SPEECH_SOURCE_DIR
    return fs::path(NEMO_SPEECH_SOURCE_DIR) / relative;
#else
    return fs::path(relative);
#endif
}

std::string
env_or_default(const char* name, fs::path fallback) {
    const char* value = std::getenv(name);
    if (value && *value) {
        return value;
    }
    return fallback.string();
}

std::vector<std::string>
split_csv(const std::string& csv) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        const size_t end = comma == std::string::npos ? csv.size() : comma;
        out.push_back(csv.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

bool
contains(const std::vector<std::string>& values, const std::string& wanted) {
    for (const auto& value : values) {
        if (value == wanted) {
            return true;
        }
    }
    return false;
}

bool
expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
    }
    return cond;
}

bool
has_param(const google::protobuf::Map<std::string, std::string>& params, const char* key) {
    return params.find(key) != params.end();
}

std::string
param_or_empty(const google::protobuf::Map<std::string, std::string>& params, const char* key) {
    const auto it = params.find(key);
    return it == params.end() ? std::string{} : it->second;
}

}  // namespace

int
main() {
    const std::string magpie_model = env_or_default(
        "NEMO_SPEECH_TEST_TTS_MAGPIE_MODEL",
        source_path("models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.f16.gguf"));
    const std::string codec_model = env_or_default(
        "NEMO_SPEECH_TEST_TTS_CODEC_MODEL",
        source_path(
            "models/nemo_nano_codec_22khz_1.89kbps_21.5fps/"
            "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"));
    const std::string tokenizer_dir = env_or_default(
        "NEMO_SPEECH_TEST_TTS_TOKENIZER_DIR",
        source_path("models/magpie_tts_multilingual_357m/extracted"));

    if (!fs::exists(magpie_model) || !fs::exists(codec_model) || !fs::is_directory(tokenizer_dir)) {
        std::cerr << "SKIP: MagpieTTS config test needs local GGUF/tokenizer fixtures\n";
        return 0;
    }

    tts::MagpieRuntimeConfig runtime_config;
    runtime_config.magpie_model = magpie_model;
    runtime_config.codec_model = codec_model;
    runtime_config.codec_cpu = true;
    runtime_config.threads = 1;
    runtime_config.codec_threads = 1;

    tts::SynthesizerConfig synthesizer_config;
    synthesizer_config.runtime = std::move(runtime_config);
    synthesizer_config.tokenizer_model_dir = tokenizer_dir;
    synthesizer_config.default_language_code = "en-US";
    auto synthesizer = std::make_shared<tts::Synthesizer>(std::move(synthesizer_config));
    std::vector<std::string> configured_languages = synthesizer->supported_language_codes();
    if (configured_languages.empty()) {
        configured_languages.push_back(synthesizer->default_language_code());
    }
    nemo_speech::GrpcTtsService service(std::move(synthesizer));

    nr_tts::RivaSynthesisConfigRequest req;
    nr_tts::RivaSynthesisConfigResponse resp;
    grpc::ServerContext ctx;
    const grpc::Status status = service.GetRivaSynthesisConfig(&ctx, &req, &resp);

    bool ok = true;
    ok &= expect(status.ok(), "GetRivaSynthesisConfig succeeds");
    ok &= expect(
        resp.model_config_size() == static_cast<int>(configured_languages.size()),
        "one TTS model config is returned per supported language");
    if (!ok) {
        return 1;
    }

    std::vector<std::string> advertised_languages;
    advertised_languages.reserve(resp.model_config_size());
    for (const auto& config : resp.model_config()) {
        ok &= expect(config.model_name() == "magpietts", "model name is magpietts");
        const auto& params = config.parameters();
        ok &= expect(has_param(params, "language_code"), "language_code parameter is present");
        ok &= expect(has_param(params, "subvoices"), "subvoices parameter is present");
        ok &= expect(has_param(params, "voices"), "voices parameter is present");
        ok &= expect(
            has_param(params, "voices_by_language"), "voices_by_language parameter is present");
        const std::string language = param_or_empty(params, "language_code");
        ok &= expect(language.find(',') == std::string::npos, "language_code is not CSV");
        advertised_languages.push_back(language);
    }

    for (const auto& language : configured_languages) {
        ok &= expect(contains(advertised_languages, language), "supported language is advertised");
    }

    const auto& first_params = resp.model_config(0).parameters();
    const std::vector<std::string> subvoices = split_csv(param_or_empty(first_params, "subvoices"));
    ok &= expect(
        param_or_empty(first_params, "voices") == param_or_empty(first_params, "subvoices"),
        "legacy voices matches subvoices");
    ok &= expect(contains(subvoices, "John"), "John subvoice is advertised");
    ok &= expect(contains(subvoices, "Sofia"), "Sofia subvoice is advertised");
    ok &= expect(contains(subvoices, "Aria"), "Aria subvoice is advertised");
    ok &= expect(contains(subvoices, "Jason"), "Jason subvoice is advertised");
    ok &= expect(contains(subvoices, "Leo"), "Leo subvoice is advertised");

    const std::string voices_by_language = param_or_empty(first_params, "voices_by_language");
    const std::string model_prefix = param_or_empty(first_params, "voice_name") + ".";
    for (const auto& language : configured_languages) {
        const std::string marker = "\"" + language + "\":{\"voices\":[";
        const size_t start = voices_by_language.find(marker);
        ok &= expect(start != std::string::npos, "language has voices_by_language entry");
        if (start == std::string::npos) {
            continue;
        }
        const size_t end = voices_by_language.find(']', start);
        ok &= expect(end != std::string::npos, "language voice list is closed");
        const std::string entry = voices_by_language.substr(start, end - start);
        for (const auto& subvoice : subvoices) {
            const std::string dotted = "\"" + model_prefix + subvoice + "\"";
            ok &= expect(
                entry.find(dotted) != std::string::npos, "language entry contains dotted subvoice");
        }
    }

    return ok ? 0 : 1;
}
