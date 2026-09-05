// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "voicechat.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "llm/llama_common.h"
#include "runtime.h"

namespace nemo_speech::s2s {
namespace {

void
set_environment_default(const char* name, const char* value) {
    if (std::getenv(name))
        return;
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0)
        throw std::runtime_error(std::string("could not set process default ") + name);
#else
    if (setenv(name, value, 0) != 0)
        throw std::runtime_error(std::string("could not set process default ") + name);
#endif
}

}  // namespace

VoiceChatConfig
voicechat_config_from_model_dir(const std::string& model_dir, int gpu, int max_streams) {
    if (model_dir.empty())
        throw std::invalid_argument("S2S model directory must not be empty");
    if (max_streams < 1)
        throw std::invalid_argument("S2S max_streams must be positive");

    VoiceChatConfig config;
    config.gpu = gpu;
    config.pipeline.perception_gguf = model_dir + "/perception.gguf";
    config.pipeline.llm_gguf = model_dir + "/nano-v2-vllm/nano-v2-llm.gguf";
    config.pipeline.llm_aux_gguf = model_dir + "/llm_aux.gguf";
    config.pipeline.tts_backbone_gguf = model_dir + "/eartts_vllm/eartts_gemma3.gguf";
    config.pipeline.tts_side_gguf = model_dir + "/eartts_vllm/eartts_side.gguf";
    config.pipeline.tts_prompt_gguf = model_dir + "/eartts_vllm/tts_prompt.gguf";
    config.pipeline.codec_gguf = model_dir + "/codec.gguf";
    config.pipeline.rnnt_vocab_json = model_dir + "/rnnt_tokenizer/vocab.json";
    config.pipeline.max_streams = max_streams;
    return config;
}

struct VoiceChat::Impl {
    explicit Impl(VoiceChatConfig value) : voicechat_config(std::move(value)) {
        configure_llama_logging(voicechat_config.verbose);
        set_environment_default("GGML_CUDA_GRAPH_EVICT_AFTER_MS", "0");
        ggml_runtime::Params params;
        params.use_gpu = voicechat_config.gpu >= 0;
        params.gpu_device_idx = voicechat_config.gpu >= 0 ? voicechat_config.gpu : 0;
        params.pe_bin_path = const_cast<char*>("");
        backend = std::make_unique<ggml_runtime::BackendManager>(params);
        pipeline = std::make_unique<S2SPipeline>(*backend, voicechat_config.pipeline);
    }

    VoiceChatConfig voicechat_config;
    std::unique_ptr<ggml_runtime::BackendManager> backend;
    std::unique_ptr<S2SPipeline> pipeline;
    std::atomic<int64_t> next_stream_id{1};
    std::once_flag warmup_once;
};

VoiceChat::VoiceChat(VoiceChatConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

VoiceChat::~VoiceChat() = default;

std::unique_ptr<S2SStream>
VoiceChat::create_stream() {
    return impl_->pipeline->create_stream(impl_->next_stream_id.fetch_add(1));
}

void
VoiceChat::prefill_system_prompt(S2SStream& stream, const std::string& prompt) {
    impl_->pipeline->prefill_system_prompt(stream, prompt);
}

S2SChunkResult
VoiceChat::process_chunk(
    S2SStream& stream, const float* audio, int samples, const std::string& function_response) {
    return impl_->pipeline->process_chunk(stream, audio, samples, function_response);
}

S2SChunkResult
VoiceChat::finish_stream(S2SStream& stream) {
    return impl_->pipeline->finish_stream(stream);
}

void
VoiceChat::end_stream(S2SStream& stream) {
    impl_->pipeline->end_stream(stream);
}

void
VoiceChat::warmup() {
    std::call_once(impl_->warmup_once, [this] { impl_->pipeline->warmup(); });
}

const S2SPipelineConfig&
VoiceChat::config() const {
    return impl_->pipeline->config();
}

int
VoiceChat::output_sample_rate() const {
    return impl_->pipeline->output_sample_rate();
}

int
VoiceChat::samples_per_step() const {
    return config().steps_per_call * config().samples_per_chunk;
}

}  // namespace nemo_speech::s2s
