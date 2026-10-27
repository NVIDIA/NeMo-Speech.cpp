// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Request rules of the SALM transcription API: the decoder's instruction and language hint, the
// output budget, the response text and usage.
//
// Decoding guards: the loop and script guards (salm_guards.h) and the empty-transcript fallback
// prompt (kFallbackPrompt).
//
// Pure functions, no llama.cpp context. Invalid requests throw PolicyError with the exact message
// clients see (HTTP 400) and, where there is one, the name of the offending parameter.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nemo_speech::speechlm::policy {

// ---- served defaults
// --------------------------------------------------------------------------------------------
inline constexpr std::string_view kAudioLocator = "<|audio|>";
// Second decode of a request whose first transcript is empty (no word timestamps requested).
inline constexpr std::string_view kFallbackPrompt =
    "Transcribe the following. Write all numbers in words. Ignore any music or background sounds: "
    "<|audio|>";
inline constexpr std::string_view kTranslationPrompt =
    "Translate the audio into English. <|audio|>";
// Decoder chat template (thinking disabled, empty system turn, no BOS).
inline constexpr std::string_view kChatPrefix =
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
inline constexpr std::string_view kChatSuffix =
    "<|im_end|>\n<|im_start|>assistant\n<think></think>";
inline constexpr int kMaxOutputTokens = 21664;       // largest output budget
inline constexpr double kMaxDurationSeconds = 3600;  // longest accepted audio
inline constexpr double kGreedyTemperature = 1e-5;   // temperatures below this decode greedily

// An invalid request (HTTP 400). `param` names the offending public field (chat route), or is
// empty.
class PolicyError : public std::invalid_argument {
   public:
    explicit PolicyError(const std::string& message, std::string param = {})
        : std::invalid_argument(message), param_(std::move(param)) {}
    const std::string& param() const { return param_; }

   private:
    std::string param_;
};

// ---- Python str semantics for whitespace and stripping
// --------------------------------------------------
bool py_isspace(char32_t c);  // str.isspace(): the 29 Unicode whitespace code points
std::string py_strip(std::string_view text);   // str.strip()
std::string py_rstrip(std::string_view text);  // str.rstrip()

// ---- language
// ---------------------------------------------------------------------------------------------------
// Client language tag -> the name the hint uses ("en", "en-US", "de_DE" ...), nullopt when
// unusable: absent or empty, more than two subtags, empty primary subtag, or a primary subtag
// outside the 43-entry map (und, auto, eng ...).
std::optional<std::string> language_name(std::optional<std::string_view> code);
// The supported languages, in order.
const std::vector<std::pair<std::string_view, std::string_view>>& language_names();

enum class Task { kTranscribe, kTranslate };

// ---- instruction
// ------------------------------------------------------------------------------------------------
struct PromptConfig {
    std::string transcription_prompt;  // the served transcription instruction
    std::string translation_prompt{kTranslationPrompt};
    // Deployment override of the transcription instruction. Stripped; blank means "use
    // transcription_prompt". It gets the language hint like the default prompt.
    std::string transcription_override;
    bool language_hint = true;
};

struct PromptRequest {
    Task task = Task::kTranscribe;
    std::optional<std::string>
        instruction;  // the client's prompt (instructions / text part), as received
    std::optional<std::string> language;  // the client's language tag, as received
    // Translation target, as received; absent or empty means English. A target outside
    // language_names() is a PolicyError (param target_language).
    std::optional<std::string> target_language;
};

struct PromptPlan {
    std::string user_turn;  // the validated user turn; always ends with " <|audio|>"
    std::string language;  // effective language: "en" for translation, else the client tag or "und"
    bool from_client = false;  // the client's instruction replaced the served one
    bool hinted = false;       // the language-hint sentence was inserted
};

// "<|audio|> locator must appear exactly once" / "<|audio|> locator must be final and be preceded
// by exactly one space" (PolicyError); returns the prompt unchanged when valid.
std::string validate_prompt(std::string prompt);
// Insert "The speech is in {Name}." before the locator (see language_name). Unchanged when
// disabled, the language is unusable, the prompt does not end with the locator, or the sentence is
// already present.
std::string language_hint(
    std::string_view prompt, std::optional<std::string_view> language, bool enabled = true);
// Instruction selection: client instruction (stripped) > translation prompt > override > default
// prompt; append " <|audio|>" when the locator is missing; hint only a served transcription
// instruction; validate.
PromptPlan plan_prompt(const PromptConfig& config, const PromptRequest& request);

// ---- budget
// -----------------------------------------------------------------------------------------------------
struct BudgetLimits {
    int64_t max_model_len = 0;  // the decoder's longest sequence (asr.salm.max_context)
    int64_t max_output_tokens = kMaxOutputTokens;
};
// max_tokens of every decode: max(24, ceil(6 s - 1e-9) + 24). PolicyError (400) when it exceeds
// the configured maximum, when the estimated audio rows (max(1, ceil(12.5 s - 1e-9))) + 128 +
// output exceed the context, or on invalid limits or duration.
int64_t output_budget(double seconds, const BudgetLimits& limits);
// "audio is {seconds:.1f} s, the limit is {limit:.0f} s" when seconds > limit + 1e-6, else nullopt
// (limit <= 0: none).
std::optional<std::string> duration_limit_error(double seconds, double limit = kMaxDurationSeconds);
// Near-digital silence: no samples, or peak below 1e-3 (-60 dBFS) and RMS below 3e-5 (-90 dBFS).
// Such a request gets an empty transcript without running the encoder or the decoder.
inline constexpr double kSilencePeak = 1e-3, kSilenceRms = 3e-5;
bool near_silent(const float* samples, size_t n);

// ---- text
// --------------------------------------------------------------------------------------------------------
// Lossy UTF-8 decode (maximal-subpart replacement, as Python bytes.decode(errors="replace")).
std::string utf8_lossy(std::string_view bytes);
// Generated bytes (token pieces concatenated) as text: invalid sequences become U+FFFD as in
// utf8_lossy, but a valid incomplete last sequence is dropped (more bytes may complete it).
std::string utf8_finish(std::string_view bytes);
// Length of that incomplete last sequence. The bytes before it are followed by more text, so
// utf8_lossy() converts them as they convert within the whole: a growing text converts piece by
// piece, with utf8_finish() for the last piece.
size_t utf8_pending(std::string_view bytes);

// Response `text`: the transcript with surrounding whitespace removed (str.strip()).
std::string response_text(std::string_view transcript);
// The transcript without its <spk:N> tags (ASCII digits, any case) and the spaces after them, for
// transports that carry speakers on the words.
std::string strip_speaker_tags(std::string_view transcript);
// response_text() of a transcript that arrives in pieces of valid UTF-8: push() returns what can be
// shown so far. Leading whitespace is dropped and trailing whitespace held back until more text
// follows, so the returned pieces add up to response_text() of the whole.
class ResponseTextStream {
   public:
    std::string push(std::string_view piece);

   private:
    bool started_ = false;
    std::string held_;  // whitespace after the last shown character
};

// ---- usage
// ------------------------------------------------------------------------------------------------------
struct Usage {
    int64_t prompt_tokens = 0;  // text tokens + audio rows
    int64_t completion_tokens =
        0;  // generated tokens, the end-of-turn token included when it ended the decode
    std::optional<int64_t> audio_tokens;
};
Usage decode_usage(
    int64_t text_prompt_tokens, int64_t audio_rows, int64_t generated_text_tokens,
    bool ended_by_end_of_turn);
// Sum a further decode of the same upload: prompt and completion tokens add, audio_tokens keeps the
// first value.
void add_usage(std::optional<Usage>& total, const std::optional<Usage>& decode);

}  // namespace nemo_speech::speechlm::policy
