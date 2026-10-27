// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "salm_policy.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>

namespace nemo_speech::speechlm::policy {

namespace {

// Python 3.12 / Unicode 15.0 str.isspace().
constexpr char32_t kSpaces[] = {0x9,    0xa,    0xb,    0xc,    0xd,    0x1c,   0x1d,   0x1e,
                                0x1f,   0x20,   0x85,   0xa0,   0x1680, 0x2000, 0x2001, 0x2002,
                                0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
                                0x2028, 0x2029, 0x202f, 0x205f, 0x3000};

constexpr char32_t kReplacement = 0xFFFD;

// Decode one code point at s[i] (maximal-subpart rule). Returns the number of bytes consumed (>=
// 1); `cp` is kReplacement for an invalid or truncated sequence.
size_t
decode_one(std::string_view s, size_t i, char32_t& cp) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        cp = b0;
        return 1;
    }
    size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        len = 2;
    } else if (b0 == 0xE0) {
        len = 3, lo = 0xA0;
    } else if ((b0 >= 0xE1 && b0 <= 0xEC) || b0 == 0xEE || b0 == 0xEF) {
        len = 3;
    } else if (b0 == 0xED) {
        len = 3, hi = 0x9F;
    } else if (b0 == 0xF0) {
        len = 4, lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
        len = 4;
    } else if (b0 == 0xF4) {
        len = 4, hi = 0x8F;
    } else {
        cp = kReplacement;
        return 1;
    }
    char32_t value = len == 2 ? (b0 & 0x1F) : len == 3 ? (b0 & 0x0F) : (b0 & 0x07);
    for (size_t k = 1; k < len; ++k) {
        if (i + k >= s.size()) {
            cp = kReplacement;
            return k;
        }
        const auto b = static_cast<unsigned char>(s[i + k]);
        const unsigned char l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
        if (b < l || b > h) {
            cp = kReplacement;
            return k;
        }
        value = (value << 6) | (b & 0x3F);
    }
    cp = value;
    return len;
}

void
append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Byte range [begin, end) of `text` without leading (lead) / trailing (trail) Python whitespace.
std::string_view
strip_view(std::string_view text, bool lead, bool trail) {
    size_t begin = 0, end = text.size();
    if (lead) {
        while (begin < end) {
            char32_t cp;
            const size_t n = decode_one(text, begin, cp);
            if (!py_isspace(cp))
                break;
            begin += n;
        }
    }
    if (trail) {
        while (end > begin) {
            size_t start = end - 1;  // back up to the first byte of the last code point
            while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80 &&
                   end - start < 4)
                --start;
            char32_t cp;
            const size_t n = decode_one(text, start, cp);
            if (start + n != end || !py_isspace(cp))
                break;
            end = start;
        }
    }
    return text.substr(begin, end - begin);
}

int64_t
py_ceil(double x) {
    return static_cast<int64_t>(std::ceil(x));
}

// Lead byte `c` of a multi-byte UTF-8 sequence: the sequence length (0 when `c` cannot start
// one) and the allowed range of the first continuation byte.
size_t
utf8_lead(unsigned char c, unsigned char& lo, unsigned char& hi) {
    lo = 0x80;
    hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF)
        return 2;
    if (c >= 0xE0 && c <= 0xEF) {
        if (c == 0xE0)
            lo = 0xA0;
        if (c == 0xED)
            hi = 0x9F;
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (c == 0xF0)
            lo = 0x90;
        if (c == 0xF4)
            hi = 0x8F;
        return 4;
    }
    return 0;
}

}  // namespace

// ---- Python string semantics
// -------------------------------------------------------------------------------------
bool
py_isspace(char32_t c) {
    return std::find(std::begin(kSpaces), std::end(kSpaces), c) != std::end(kSpaces);
}
std::string
py_strip(std::string_view text) {
    return std::string(strip_view(text, true, true));
}
std::string
py_rstrip(std::string_view text) {
    return std::string(strip_view(text, false, true));
}

// ---- language
// ----------------------------------------------------------------------------------------------------
const std::vector<std::pair<std::string_view, std::string_view>>&
language_names() {
    // BCP-47 primary subtag -> hint name.
    static const std::vector<std::pair<std::string_view, std::string_view>> kNames = {
        {"en", "English"},   {"ar", "Arabic"},     {"zh", "Chinese"},    {"fr", "French"},
        {"de", "German"},    {"hi", "Hindi"},      {"it", "Italian"},    {"ja", "Japanese"},
        {"es", "Spanish"},   {"ko", "Korean"},     {"pt", "Portuguese"}, {"ru", "Russian"},
        {"th", "Thai"},      {"hy", "Armenian"},   {"nn", "Norwegian"},  {"nb", "Norwegian"},
        {"no", "Norwegian"}, {"bg", "Bulgarian"},  {"hr", "Croatian"},   {"cs", "Czech"},
        {"da", "Danish"},    {"nl", "Dutch"},      {"et", "Estonian"},   {"fi", "Finnish"},
        {"el", "Greek"},     {"he", "Hebrew"},     {"hu", "Hungarian"},  {"id", "Indonesian"},
        {"lv", "Latvian"},   {"lt", "Lithuanian"}, {"mt", "Maltese"},    {"ta", "Tamil"},
        {"bn", "Bengali"},   {"pl", "Polish"},     {"ro", "Romanian"},   {"sk", "Slovak"},
        {"sl", "Slovenian"}, {"sv", "Swedish"},    {"tr", "Turkish"},    {"tl", "Filipino"},
        {"fil", "Filipino"}, {"uk", "Ukrainian"},  {"vi", "Vietnamese"},
    };
    return kNames;
}

std::optional<std::string>
language_name(std::optional<std::string_view> code) {
    if (!code || code->empty())
        return std::nullopt;
    // str(code).strip().lower().replace("_", "-").split("-"); the only non-ASCII code point lower()
    // maps into ASCII is U+212A KELVIN SIGN -> "k", so every other non-ASCII character keeps a
    // subtag out of the (ASCII) map.
    const std::string_view stripped = strip_view(*code, true, true);
    std::vector<std::string> parts(1);
    for (size_t i = 0; i < stripped.size();) {
        char32_t cp;
        const size_t n = decode_one(stripped, i, cp);
        if (cp == '-' || cp == '_') {
            parts.emplace_back();
        } else if (cp == 0x212A) {
            parts.back() += 'k';
        } else if (cp < 0x80) {
            parts.back() += static_cast<char>(std::tolower(static_cast<int>(cp)));
        } else {
            parts.back().append(stripped.substr(i, n)).push_back('\x01');  // never matches a key
        }
        i += n;
    }
    if (parts.size() > 2 || parts[0].empty())
        return std::nullopt;
    for (const auto& [key, name] : language_names())
        if (parts[0] == key)
            return std::string(name);
    return std::nullopt;
}

namespace {
// The effective language: for translation the target (as received; "en" when absent or empty),
// else the client's tag as received, "und" when absent or empty.
std::string
effective_language(
    Task task, std::optional<std::string_view> client_language,
    std::optional<std::string_view> target_language) {
    if (task == Task::kTranslate)
        return target_language && !target_language->empty() ? std::string(*target_language) : "en";
    return client_language && !client_language->empty() ? std::string(*client_language) : "und";
}
}  // namespace

// ---- instruction
// -------------------------------------------------------------------------------------------------
std::string
validate_prompt(std::string prompt) {
    size_t count = 0;
    for (size_t at = prompt.find(kAudioLocator); at != std::string::npos;
         at = prompt.find(kAudioLocator, at + kAudioLocator.size()))
        ++count;
    if (count != 1)
        throw PolicyError("<|audio|> locator must appear exactly once");
    const bool final =
        prompt.size() >= kAudioLocator.size() &&
        std::string_view(prompt).substr(prompt.size() - kAudioLocator.size()) == kAudioLocator;
    const std::string_view prefix =
        std::string_view(prompt).substr(0, prompt.size() - kAudioLocator.size());
    const bool one_space = !prefix.empty() && prefix.back() == ' ' &&
                           !(prefix.size() >= 2 && prefix[prefix.size() - 2] == ' ');
    if (!final || !one_space || strip_view(prefix, true, true).empty())
        throw PolicyError("<|audio|> locator must be final and be preceded by exactly one space");
    return prompt;
}

std::string
language_hint(std::string_view prompt, std::optional<std::string_view> language, bool enabled) {
    if (!enabled)
        return std::string(prompt);
    const auto name = language_name(language);
    const bool final = prompt.size() >= kAudioLocator.size() &&
                       prompt.substr(prompt.size() - kAudioLocator.size()) == kAudioLocator;
    if (!name || !final)
        return std::string(prompt);
    const std::string sentence = "The speech is in " + *name + ".";
    if (prompt.find(sentence) != std::string_view::npos)
        return std::string(prompt);
    const std::string_view prefix =
        strip_view(prompt.substr(0, prompt.size() - kAudioLocator.size()), false, true);
    return std::string(prefix) + " " + sentence + " " + std::string(kAudioLocator);
}

PromptPlan
plan_prompt(const PromptConfig& config, const PromptRequest& request) {
    const bool translate = request.task == Task::kTranslate;
    PromptPlan plan;
    const std::string custom = request.instruction ? py_strip(*request.instruction) : std::string();
    plan.language = effective_language(request.task, request.language, request.target_language);
    std::string prompt;
    if (!custom.empty()) {
        prompt = custom;
    } else if (translate) {
        // Any language the model knows; English keeps the served translation prompt.
        const auto target = language_name(request.target_language);
        if (request.target_language && !request.target_language->empty() && !target)
            throw PolicyError(
                "target_language " + *request.target_language + " is not supported",
                "target_language");
        prompt = !target || *target == "English"
                     ? config.translation_prompt
                     : "Translate the audio into " + *target + ". " + std::string(kAudioLocator);
    } else {
        const std::string override_prompt = py_strip(config.transcription_override);
        prompt = override_prompt.empty() ? config.transcription_prompt : override_prompt;
    }
    if (prompt.find(kAudioLocator) == std::string::npos)
        prompt += " " + std::string(kAudioLocator);
    if (custom.empty() && !translate) {
        std::string hinted = language_hint(prompt, plan.language, config.language_hint);
        plan.hinted = hinted != prompt;
        prompt = std::move(hinted);
    }
    plan.user_turn = validate_prompt(std::move(prompt));
    plan.from_client = !custom.empty();
    return plan;
}

// ---- budget
// ------------------------------------------------------------------------------------------------------
namespace {
void
check_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0)
        throw PolicyError("audio duration must be finite and non-negative");
}

// Audio rows assumed by the budget: max(1, ceil(12.5 s - 1e-9)).
int64_t
audio_prompt_tokens(double seconds) {
    check_duration(seconds);
    return std::max<int64_t>(1, py_ceil(seconds * 12.5 - 1e-9));
}

// Output tokens the budget allows: max(24, ceil(6 s - 1e-9) + 24).
int64_t
required_output_tokens(double seconds) {
    check_duration(seconds);
    return std::max<int64_t>(24, py_ceil(seconds * 6.0 - 1e-9) + 24);
}
}  // namespace

int64_t
output_budget(double seconds, const BudgetLimits& limits) {
    if (limits.max_model_len < 1 || limits.max_output_tokens < 1)
        throw PolicyError("model and output limits must be positive");
    const int64_t prompt = audio_prompt_tokens(seconds) + 128;
    const int64_t available = limits.max_model_len - prompt;
    const int64_t required = required_output_tokens(seconds);
    if (required > limits.max_output_tokens)
        throw PolicyError(
            "duration-derived output budget " + std::to_string(required) +
            " exceeds configured maximum " + std::to_string(limits.max_output_tokens));
    if (required > available)
        throw PolicyError(
            "audio and output need " + std::to_string(prompt + required) +
            " tokens, max_model_len is " + std::to_string(limits.max_model_len));
    return required;
}

std::optional<std::string>
duration_limit_error(double seconds, double limit) {
    // `limit <= 0 or duration <= limit + 1e-6` is None; NaN compares false both ways, as in Python.
    if (limit <= 0 || seconds <= limit + 1e-6)
        return std::nullopt;
    char buffer[128];
    std::snprintf(buffer, sizeof buffer, "audio is %.1f s, the limit is %.0f s", seconds, limit);
    return std::string(buffer);
}

bool
near_silent(const float* samples, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = samples[i];
        if (std::fabs(x) >= kSilencePeak)
            return false;
        sum += x * x;
    }
    return n == 0 || std::sqrt(sum / static_cast<double>(n)) < kSilenceRms;
}

// ---- text
// --------------------------------------------------------------------------------------------------------
std::string
utf8_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size();) {
        char32_t cp;
        i += decode_one(bytes, i, cp);
        append_utf8(out, cp);
    }
    return out;
}

std::string
utf8_finish(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    const auto* b = reinterpret_cast<const unsigned char*>(in.data());
    const size_t n = in.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = b[i];
        if (c < 0x80) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }
        unsigned char lo, hi;  // allowed range of the first continuation byte
        const size_t len = utf8_lead(c, lo, hi);
        if (len == 0) {
            out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }
        size_t k = 1;
        while (k < len && i + k < n) {
            const unsigned char d = b[i + k];
            if (k == 1 ? (d < lo || d > hi) : (d < 0x80 || d > 0xBF))
                break;
            ++k;
        }
        if (k == len) {
            out.append(in, i, len);
            i += len;
            continue;
        }
        if (i + k == n)
            break;  // incomplete at the end: not emitted yet
        out += "\xEF\xBF\xBD";
        i += k;
    }
    return out;
}

size_t
utf8_pending(std::string_view in) {
    const auto* b = reinterpret_cast<const unsigned char*>(in.data());
    const size_t n = in.size();
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = b[n - back];
        if (c >= 0x80 && c <= 0xBF)
            continue;  // a continuation byte: its lead is further back
        unsigned char lo, hi;
        if (utf8_lead(c, lo, hi) <= back)
            return 0;  // ASCII, an invalid byte, or a sequence that is already complete or invalid
        // the bytes after the lead are continuation bytes; only the first has a narrower range
        if (back > 1 && (b[n - back + 1] < lo || b[n - back + 1] > hi))
            return 0;
        return back;
    }
    return 0;
}

std::string
response_text(std::string_view transcript) {
    return py_strip(transcript);
}

std::string
strip_speaker_tags(std::string_view transcript) {
    // The end of a <spk:N> tag at `at`, 0 when there is none.
    const auto tag_end = [&](size_t at) -> size_t {
        constexpr std::string_view kOpen = "<spk:";
        if (transcript.size() - at < kOpen.size() + 2)
            return 0;
        for (size_t k = 0; k < kOpen.size(); ++k)
            if (std::tolower(static_cast<unsigned char>(transcript[at + k])) != kOpen[k])
                return 0;
        size_t end = at + kOpen.size();
        while (end < transcript.size() && transcript[end] >= '0' && transcript[end] <= '9') ++end;
        return end > at + kOpen.size() && end < transcript.size() && transcript[end] == '>'
                   ? end + 1
                   : 0;
    };
    std::string out;
    for (size_t i = 0; i < transcript.size();) {
        if (const size_t end = tag_end(i)) {
            i = end;
            while (i < transcript.size() && transcript[i] == ' ') ++i;
            continue;
        }
        out += transcript[i++];
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string
ResponseTextStream::push(std::string_view piece) {
    std::string out;
    for (size_t i = 0; i < piece.size();) {
        char32_t cp;
        const size_t n = decode_one(piece, i, cp);
        const std::string_view bytes = piece.substr(i, n);
        i += n;
        if (!py_isspace(cp)) {
            out += held_;
            held_.clear();
            out += bytes;
            started_ = true;
        } else if (started_) {
            held_ += bytes;
        }
    }
    return out;
}

// ---- usage
// -------------------------------------------------------------------------------------------------------
Usage
decode_usage(
    int64_t text_prompt_tokens, int64_t audio_rows, int64_t generated_text_tokens,
    bool ended_by_end_of_turn) {
    Usage usage;
    usage.prompt_tokens = text_prompt_tokens + audio_rows;
    usage.completion_tokens = generated_text_tokens + (ended_by_end_of_turn ? 1 : 0);
    usage.audio_tokens = audio_rows;
    return usage;
}

void
add_usage(std::optional<Usage>& total, const std::optional<Usage>& decode) {
    if (!decode)
        return;
    if (!total) {
        total = decode;
        return;
    }
    total->prompt_tokens += decode->prompt_tokens;
    total->completion_tokens += decode->completion_tokens;
    if (!total->audio_tokens)
        total->audio_tokens = decode->audio_tokens;
}

}  // namespace nemo_speech::speechlm::policy
