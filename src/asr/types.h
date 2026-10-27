// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Transport-independent request and result values shared by the ASR interfaces.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nemo_speech::asr {

// Speech-LLM (SALM) request options; other backends ignore them.
struct SpeechLmOptions {
    std::string task = "transcribe";  // "transcribe" or "translate" (speech translation)
    std::string target_language;      // translate: the target language code; empty = English
    // Client instructions: replace the served instruction text (" <|audio|>" is appended when
    // absent; no language hint is added). Empty = the served prompt.
    std::string instructions;
    float temperature = 0.0f;  // 0 = greedy
    bool has_seed = false;
    uint64_t seed = 0;
    // The response text as it is generated: pieces that add up to SpeechLmDetails::text (none for
    // an empty transcript). Called from a decoder thread; it must return quickly and not throw.
    std::function<void(std::string_view)> on_text;
};

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

    SpeechLmOptions speechlm;

    // True once the client is gone (disconnected, or cancelled the call). Backends that queue or
    // decode for a while poll it from the request's thread and drop the request when it turns true
    // (asr::Cancelled). Empty: never cancelled. Must be thread-safe and cheap.
    std::function<bool()> cancelled;

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
    // Not in the proto: the word in running text and subtitles when that differs from `word`, and
    // whether it follows the previous word without a space (SpeechLmWord).
    std::string display;
    bool joins_previous = false;
};

struct Alternative {         // proto: SpeechRecognitionAlternative
    std::string transcript;  // post-processed (PnC/ITN/profanity already applied)
    float confidence = 0.0f;
    std::vector<Word> words;                  // populated only when requested
    std::vector<std::string> language_codes;  // detected languages, if any
};

// Speech-LLM result details beyond the Riva fields: word times, speaker slots, diarization,
// alignment status and usage. Times are seconds from the audio start.
struct SpeechLmWord {
    std::string word;
    // Running text when it differs from `word` (a Chinese, Japanese or Thai character keeps the
    // punctuation around it) and whether it follows the previous word without a space.
    std::string display;
    bool joins_previous = false;
    int word_index = 0;
    double start = 0.0, end = 0.0;
    int speaker = -1;       // inline <spk:N> tag, 0-based ("spk:N" on the wire); -1 = none
    int speaker_slot = -1;  // diarizer column; -1 = none
    float speaker_confidence = -1.0f;  // mean slot activity over the word; < 0 = none
};

struct SpeechLmSegment {
    int speaker = 0;  // diarizer column
    double start = 0.0, end = 0.0;
};

struct SpeechLmDetails {
    std::string text;      // cleaned transcript, inline speaker tags kept
    std::string language;  // as requested, or "und"
    std::string task = "transcribe";
    std::string transcript_origin = "llm";  // "llm": the decoder wrote the transcript
    double duration_seconds = 0.0;
    // Word timestamps (requested with enable_word_time_offsets).
    bool timestamps_requested = false;
    std::string alignment_status;       // "aligned" | "unavailable"
    std::string alignment_model;        // empty = null
    std::string failure_reason;         // empty = null
    double frame_stride_seconds = 0.0;  // 0 = null
    std::vector<SpeechLmWord> words;    // single-stream timing, transcript order
    // Words of speaker streams without a legal alignment: in the transcript, not in `words`.
    int unaligned_words = 0;
    std::map<std::string, std::vector<SpeechLmWord>>
        speaker_words;                         // "spk:N" / "unknown": overlap re-timed
    std::vector<SpeechLmSegment> diarization;  // post-processed speaker turns
    double diarization_frame_seconds = 0.0;    // 0 = null
    std::vector<std::string> warnings;
    // Usage, summed over every decode of the request.
    int prompt_tokens = 0, audio_tokens = 0, completion_tokens = 0;
};

struct Result {  // proto: Streaming/SpeechRecognitionResult
    // Ranked best-first and capped by max_alternatives.
    std::vector<Alternative> alternatives;
    bool is_final = true;
    float stability = 0.0f;
    int32_t channel_tag = 1;       // mono channel
    float audio_processed = 0.0f;  // seconds (proto audio_processed)
    // Streaming only: sentence punctuation emitted after the previous final's
    // endpoint. Append it to that final; it is not part of this result's text.
    std::string late_punctuation;
    // Speech-LLM details (null for other backends).
    std::shared_ptr<const SpeechLmDetails> speechlm;
};

}  // namespace nemo_speech::asr
