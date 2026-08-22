// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <stdexcept>
#include <string>

#include "cli_util.h"
#include "commands.h"
#include "ggml-backend.h"
#include "json.h"
#include "model_store.h"

namespace {
using nemo_speech::json::Value;

const char*
device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "integrated-gpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META:
            return "meta";
    }
    return "unknown";
}

Value
build_features() {
    Value result(Value::Object{});
#if defined(NEMO_SPEECH_CLI_ASR)
    result["asr"] = true;
    result["integrated_vad"] = true;
    result["punctuation"] = true;
#else
    result["asr"] = false;
    result["integrated_vad"] = false;
    result["punctuation"] = false;
#endif
    result["model_pull"] = true;
#if defined(NEMO_SPEECH_CLI_DIAR)
    result["diarization"] = true;
#else
    result["diarization"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    result["tts"] = true;
#else
    result["tts"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    result["translation"] = true;
#else
    result["translation"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_HTTP)
    result["http"] = true;
#else
    result["http"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_HTTP) && defined(NEMO_SPEECH_CLI_ASR)
    result["realtime_websocket"] = true;
#else
    result["realtime_websocket"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_ASR) && defined(NEMO_SPEECH_CLI_NMT)
    result["speech_translation"] = true;
#else
    result["speech_translation"] = false;
#endif
#if defined(NEMO_SPEECH_GRPC_SERVER_BUILT)
    result["grpc"] = true;
#else
    result["grpc"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_FLASHLIGHT)
    result["flashlight"] = true;
#else
    result["flashlight"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_NORM)
    result["text_normalization"] = true;
#else
    result["text_normalization"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_CUDA)
    result["backend_cuda"] = true;
#else
    result["backend_cuda"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_METAL)
    result["backend_metal"] = true;
#else
    result["backend_metal"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_VULKAN)
    result["backend_vulkan"] = true;
#else
    result["backend_vulkan"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_TTS_JA)
    result["tts_japanese"] = true;
#else
    result["tts_japanese"] = false;
#endif
#if defined(NEMO_SPEECH_CLI_TTS_ZH)
    result["tts_mandarin"] = true;
#else
    result["tts_mandarin"] = false;
#endif
    return result;
}

Value
backend_devices() {
    ggml_backend_load_all();
    Value::Array devices;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        const auto device = ggml_backend_dev_get(index);
        ggml_backend_dev_props properties{};
        ggml_backend_dev_get_props(device, &properties);
        Value item(Value::Object{});
        item["index"] = static_cast<double>(index);
        item["name"] = properties.name ? properties.name : "unknown";
        item["description"] = properties.description ? properties.description : "";
        item["type"] = device_type_name(properties.type);
        item["memory_free"] = static_cast<double>(properties.memory_free);
        item["memory_total"] = static_cast<double>(properties.memory_total);
        item["async"] = properties.caps.async;
        item["events"] = properties.caps.events;
        devices.emplace_back(std::move(item));
    }
    return Value(std::move(devices));
}

}  // namespace

void
print_doctor_help(const char* program) {
    std::printf(
        "Usage: %s doctor [--json]\n\n"
        "Inspect compiled capabilities, compute devices, and driver availability.\n",
        program);
}

int
command_doctor(int argc, char** argv) {
    try {
        bool json = cli_json();
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (is_help_argument(arg)) {
                print_doctor_help("nemo-speech");
                return 0;
            }
            if (arg == "--json")
                json = true;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        Value result(Value::Object{});
        result["version"] = NEMO_SPEECH_VERSION_STR;
        result["features"] = build_features();
        result["devices"] = backend_devices();
        bool accelerator_available = false;
        for (const auto& device : result.at("devices").array())
            accelerator_available = accelerator_available || device.at("type").string() == "gpu" ||
                                    device.at("type").string() == "integrated-gpu";
        const auto& features = result.at("features");
        const bool accelerator_compiled = features.at("backend_cuda").boolean() ||
                                          features.at("backend_metal").boolean() ||
                                          features.at("backend_vulkan").boolean();
        result["accelerator_compiled"] = accelerator_compiled;
        result["accelerator_available"] = accelerator_available;
        result["driver_runtime_compatible"] = !accelerator_compiled || accelerator_available;
        const auto downloader = model_downloader_executable();
        Value model_download(Value::Object{});
        model_download["available"] = !downloader.empty();
        model_download["executable"] = downloader.u8string();
        result["model_download"] = std::move(model_download);
        Value::Array runtime_warnings;
        if (accelerator_compiled && !accelerator_available)
            runtime_warnings.emplace_back(
                "this build includes a GPU backend, but no compatible accelerator/driver was "
                "discovered; use --device cpu or repair the driver/runtime installation");
        if (downloader.empty())
            runtime_warnings.emplace_back(
                "automatic model downloads require curl on PATH; local and cached models still "
                "work");
        result["runtime_warnings"] = std::move(runtime_warnings);
        if (json) {
            std::printf("%s\n", result.dump(2).c_str());
        } else {
            std::printf("NeMo-Speech.cpp %s\n", NEMO_SPEECH_VERSION_STR);
            std::printf("Features:");
            for (const auto& feature : result.at("features").object())
                if (feature.second.boolean())
                    std::printf(" %s", feature.first.c_str());
            std::printf("\nDevices:\n");
            for (const auto& device : result.at("devices").array()) {
                std::printf(
                    "  [%d] %-14s %s", static_cast<int>(device.at("index").number()),
                    device.at("type").string().c_str(), device.at("description").string().c_str());
                const auto total = static_cast<uint64_t>(device.at("memory_total").number());
                if (total > 0)
                    std::printf(" (%.1f GiB)", total / 1073741824.0);
                std::printf("\n");
            }
            if (downloader.empty())
                std::printf("Model downloads: unavailable (curl not found on PATH)\n");
            else
                std::printf("Model downloads: %s\n", downloader.u8string().c_str());
            for (const auto& warning : result.at("runtime_warnings").array())
                std::printf("Runtime warning: %s\n", warning.string().c_str());
        }
        return result.at("driver_runtime_compatible").boolean() ? 0 : 1;
    }
    catch (const std::exception& error) {
        return print_cli_exception("doctor", error);
    }
}
