// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Transport-independent request and result values shared by the ASR interfaces.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nemo_speech::asr {

struct AsrRequestOptions {
    // RecognitionConfig.language_code. Recognizer stores the requested
    // prompt language here so final postprocessing can select the same ITN
    // grammar (or the model-detected language when this is "auto").
    std::string language_code;
    // Maximum ranked hypotheses; greedy decoders always return one.
    int max_alternatives = 1;
    bool enable_word_time_offsets = false;
    // Skips ITN but does not disable punctuation and capitalization.
    bool verbatim_transcripts = false;
    bool enable_automatic_punctuation = false;
    bool profanity_filter = false;

    struct Boost {
        std::vector<std::string> phrases;
        float boost = 0.0f;
    };
    std::vector<Boost> speech_contexts;

    // Per-request EOU silence threshold in milliseconds; <= 0 keeps the default.
    float stop_history_eou_ms = -1.0f;

    // Tags words in the top alternative when a diarizer model is available.
    bool enable_speaker_diarization = false;
    int max_speaker_count = 8;

    // Diarization requires words even when offsets were not explicitly requested.
    bool needs_word_timings() const {
        return enable_word_time_offsets || enable_speaker_diarization;
    }
};

struct Word {  // proto: WordInfo
    std::string word;
    int32_t start_time = 0;  // ms from audio start (proto WordInfo.start_time)
    int32_t end_time = 0;    // ms from audio start (proto WordInfo.end_time)
    float confidence = 0.0f;
    std::string language_code;  // empty unless language detection tagged it
    // proto WordInfo.speaker_tag: 1-based speaker id, set only when
    // diarization was requested (0 = untagged).
    int32_t speaker_tag = 0;
};

struct Alternative {         // proto: SpeechRecognitionAlternative
    std::string transcript;  // post-processed (PnC/ITN/profanity already applied)
    float confidence = 0.0f;
    std::vector<Word> words;                  // populated only when requested
    std::vector<std::string> language_codes;  // detected languages, if any
};

struct Result {  // proto: Streaming/SpeechRecognitionResult
    // Ranked best-first and capped by max_alternatives.
    std::vector<Alternative> alternatives;
    bool is_final = true;
    float stability = 0.0f;
    int32_t channel_tag = 1;       // mono channel
    float audio_processed = 0.0f;  // seconds (proto audio_processed)
};

}  // namespace nemo_speech::asr
