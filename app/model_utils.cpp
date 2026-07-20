// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "model_utils.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>

#include "cli_util.h"
#include "ggml.h"
#include "gguf.h"
#include "json.h"

namespace {
namespace fs = std::filesystem;
using nemo_speech::json::Value;

fs::path
require_path(const std::string& value, const std::string& description, bool directory) {
    if (value.empty())
        throw MissingModelError(description + " path is required");
    const fs::path path(value);
    std::error_code error;
    const bool valid = directory ? fs::is_directory(path, error) : fs::is_regular_file(path, error);
    if (!valid || error)
        throw MissingModelError(
            description + (directory ? " directory does not exist: " : " file does not exist: ") +
            value);
    return fs::absolute(path);
}

}  // namespace

fs::path
require_model_file(const std::string& path, const std::string& description) {
    return require_path(path, description, false);
}

fs::path
require_model_directory(const std::string& path, const std::string& description) {
    return require_path(path, description, true);
}

std::string
inspect_gguf_json(const fs::path& path) {
    gguf_init_params params{true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> context(
        gguf_init_from_file(path.string().c_str(), params), &gguf_free);
    if (!context)
        throw std::runtime_error(path.string() + " is not a readable GGUF file");
    Value result(Value::Object{});
    result["path"] = fs::absolute(path).string();
    result["size"] = static_cast<double>(fs::file_size(path));
    const int64_t architecture_key = gguf_find_key(context.get(), "general.architecture");
    const int64_t name_key = gguf_find_key(context.get(), "general.name");
    result["architecture"] =
        architecture_key >= 0 ? gguf_get_val_str(context.get(), architecture_key) : "unknown";
    result["name"] =
        name_key >= 0 ? gguf_get_val_str(context.get(), name_key) : path.stem().string();
    result["tensor_count"] = static_cast<double>(gguf_get_n_tensors(context.get()));
    const int64_t file_type_key = gguf_find_key(context.get(), "general.file_type");
    if (file_type_key >= 0 && gguf_get_kv_type(context.get(), file_type_key) == GGUF_TYPE_UINT32)
        result["file_type"] = static_cast<double>(gguf_get_val_u32(context.get(), file_type_key));
    Value::Object types;
    for (int64_t i = 0; i < gguf_get_n_tensors(context.get()); ++i) {
        const std::string type = ggml_type_name(gguf_get_tensor_type(context.get(), i));
        auto found = types.find(type);
        types[type] = found == types.end() ? 1 : static_cast<int>(found->second.number()) + 1;
    }
    result["tensor_types"] = std::move(types);
    const std::string architecture = result.at("architecture").string();
    std::string normalized_name = result.at("name").string();
    std::transform(
        normalized_name.begin(), normalized_name.end(), normalized_name.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::string role;
    if (architecture == "asr" || architecture == "parakeet-ctc" || architecture == "nemo")
        role = "asr";
    else if (architecture == "sortformer")
        role = "diarization";
    else if (architecture == "vad")
        role = "vad";
    else if (architecture == "pnc")
        role = "pnc";
    else if (architecture == "magpietts")
        role = "tts";
    else if (architecture == "nemo-nano-codec")
        role = "codec";
    else if (
        architecture == "llama" && (normalized_name.find("riva-translate") != std::string::npos ||
                                    normalized_name.find("riva translate") != std::string::npos))
        role = "nmt";
    if (!role.empty())
        result["role"] = role;

    Value::Array languages;
    const int64_t prompts = gguf_find_key(context.get(), "asr.rnnt.prompt_dictionary");
    if (prompts >= 0 && gguf_get_kv_type(context.get(), prompts) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(context.get(), prompts) == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < gguf_get_arr_n(context.get(), prompts); ++i) {
            std::string entry = gguf_get_arr_str(context.get(), prompts, i);
            const auto colon = entry.find(':');
            if (colon != std::string::npos)
                entry.resize(colon);
            if (!entry.empty())
                languages.emplace_back(std::move(entry));
        }
    }
    result["languages"] = std::move(languages);
    Value::Array companions;
    if (architecture == "magpietts") {
        companions.emplace_back("codec");
        companions.emplace_back("tokenizer");
    }
    result["required_companion_roles"] = std::move(companions);
    Value::Array errors;
    Value::Array warnings;
    bool compiled = true;
    if (role.empty()) {
        compiled = false;
        errors.emplace_back(
            "the GGUF architecture is not recognized by this NeMo-Speech.cpp runtime");
    }
#if !defined(NEMO_SPEECH_CLI_ASR)
    if (role == "asr" || role == "vad" || role == "pnc") {
        compiled = false;
        errors.emplace_back("this build does not include ASR support");
    }
#endif
#if !defined(NEMO_SPEECH_CLI_DIAR)
    if (role == "diarization") {
        compiled = false;
        errors.emplace_back("this build does not include diarization support");
    }
#endif
#if !defined(NEMO_SPEECH_CLI_NMT)
    if (role == "nmt") {
        compiled = false;
        errors.emplace_back("this build does not include NMT support");
    }
#endif
#if !defined(NEMO_SPEECH_CLI_TTS)
    if (role == "tts" || role == "codec") {
        compiled = false;
        errors.emplace_back("this build does not include TTS support");
    }
#endif
    if (architecture == "llama" && role.empty())
        warnings.emplace_back(
            "generic llama.cpp models are not validated as Riva-Translate models");
    if (role == "tts")
        warnings.emplace_back("text synthesis also requires codec and tokenizer companions");
    result["runtime_compatible"] = compiled;
    result["errors"] = std::move(errors);
    result["warnings"] = std::move(warnings);
    return result.dump(2) + "\n";
}
