// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "tts/tokenizer/tokenizer.h"

namespace nemo_speech::tts {

enum magpie_longform_mode {
    MAGPIE_LONGFORM_AUTO,
    MAGPIE_LONGFORM_OFF,
    MAGPIE_LONGFORM_ON,
};

struct magpie_stream_params {
    std::string magpie_model;
    std::string codec_model;

    std::string wav_out;
    std::string raw_out;
    std::string audio_cmd;
    std::string codes_out;

    std::vector<int32_t> tokens;
    std::vector<std::vector<int32_t>> token_chunks;
    std::string tokens_file;
    std::vector<int32_t> warmup_tokens;
    std::vector<std::vector<int32_t>> warmup_token_chunks;
    std::string warmup_tokens_file;
    std::string text;
    std::string text_file;
    std::string warmup_text;
    std::string warmup_text_file;
    std::string tokenizer_model_dir;
    std::string tn_model_dir;
    MagpieTokenizerConfig tokenizer_config;
    std::string language_code = "en-US";

    int speaker = 0;
    int threads = 4;
    int codec_threads = 0;
    int seed = -1;
    int steps = -1;
    int top_k = -1;
    int chunk_frames = 4;
    int codec_queue_depth = 4;
    int codec_history_frames = -1;
    int codec_future_frames = 1;
    int window_ms = 0;
    float temperature = NAN;
    float cfg_scale = NAN;
    bool use_cfg = true;
    bool use_local_transformer = true;
    bool lt_fp32 = false;
    bool use_kv_cache = true;
    bool use_stateful_codec = true;
    bool codec_cpu = false;
    bool flush_partial_chunk = true;
    bool benchmark = false;
    bool verbose = false;
    bool show_help = false;
    magpietts_backend_preference lt_backend = MAGPIETTS_BACKEND_AUTO;
    magpietts_backend_preference sampling_backend = MAGPIETTS_BACKEND_AUTO;
    magpietts_uma_mode uma_mode = MAGPIETTS_UMA_AUTO;
    magpie_longform_mode longform_mode = MAGPIE_LONGFORM_AUTO;
};

struct stream_latency_metrics {
    int64_t start_us = 0;
    int64_t first_event_us = 0;
    int64_t last_event_us = 0;
    int events = 0;
    double first_event_ms = 0.0;
    double inter_event_sum_ms = 0.0;
    double inter_event_min_ms = std::numeric_limits<double>::max();
    double inter_event_max_ms = 0.0;
    double elapsed_s = 0.0;
    std::vector<double> inter_event_ms;

    void begin(int64_t now_us);
    double record_event(int64_t now_us, bool& first_event);
    void finish(int64_t now_us);
    double inter_event_avg_ms() const;
    double inter_event_min_value_ms() const;
    double inter_event_percentile_ms(double percentile) const;
    double inter_event_p95_ms() const;
    double inter_event_p99_ms() const;
};

struct stream_run_metrics {
    int64_t start_us = 0;
    int chunks = 0;
    int e2e_chunks = 0;
    int generated_frames = 0;
    uint64_t samples_written = 0;
    double tokenizer_ms = 0.0;
    double encoder_ms = 0.0;
    double ttfa_ms = 0.0;
    double e2e_elapsed_s = 0.0;
    double audio_s = 0.0;
    double e2e_rtf = 0.0;
    double e2e_rtfx = 0.0;
    double decoder_audio_s = 0.0;
    double decoder_rtfx = 0.0;
    double codec_audio_s = 0.0;
    double codec_elapsed_s = 0.0;
    double codec_rtfx = 0.0;
    stream_latency_metrics decoder;
    stream_latency_metrics codec;
    stream_latency_metrics e2e;

    void begin();
    double record_decoder_frame(int64_t now_us, bool& first_frame);
    double record_codec_chunk(int64_t now_us, bool& first_chunk);
    double record_audio_write(int64_t now_us, bool& first_write);
    void add_codec_work(double elapsed_s, double audio_s);
    void finish(uint64_t samples_written, int sample_rate, double codec_fps);
    double inter_chunk_avg_ms() const;
    double inter_chunk_min_value_ms() const;
};

using magpie_pcm_callback = std::function<bool(const std::vector<uint8_t>&)>;

class MagpieStreamingRuntime {
   public:
    MagpieStreamingRuntime();
    ~MagpieStreamingRuntime();

    MagpieStreamingRuntime(const MagpieStreamingRuntime&) = delete;
    MagpieStreamingRuntime& operator=(const MagpieStreamingRuntime&) = delete;

    bool load(
        const std::string& magpie_model, const std::string& codec_model,
        magpietts_uma_mode uma_mode, bool magpie_cpu, bool codec_cpu, bool verbose = false);
    int sampleRate() const;
    int speakerCount() const;
    std::vector<std::string> speakerNames() const;
    const std::string& tokenizerProfile() const;
    int textVocabSize() const;
    bool synthesize(
        magpie_stream_params& params, const std::vector<int32_t>& tokens,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics);
    bool synthesize(
        magpie_stream_params& params, const std::vector<int32_t>& tokens,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics, const char* run_label,
        bool write_codes);
    bool synthesize(
        magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics, const char* run_label,
        bool write_codes);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

using magpie_stream_runtime = MagpieStreamingRuntime;

bool magpie_stream_runtime_init(
    magpie_stream_runtime*& runtime, const std::string& magpie_model,
    const std::string& codec_model, magpietts_uma_mode uma_mode, bool magpie_cpu, bool codec_cpu);
void magpie_stream_runtime_free(magpie_stream_runtime* runtime);
int magpie_stream_runtime_sample_rate(const magpie_stream_runtime* runtime);
int magpie_stream_runtime_speaker_count(const magpie_stream_runtime* runtime);
std::vector<std::string> magpie_stream_runtime_speaker_names(const magpie_stream_runtime* runtime);
bool magpie_stream_runtime_synthesize(
    magpie_stream_runtime* runtime, magpie_stream_params& params,
    const std::vector<int32_t>& tokens, const magpie_pcm_callback& pcm_callback,
    stream_run_metrics& metrics);

const char* magpie_longform_mode_name(magpie_longform_mode mode);
bool parse_magpie_longform_mode(const std::string& value, magpie_longform_mode& mode);

}  // namespace nemo_speech::tts
