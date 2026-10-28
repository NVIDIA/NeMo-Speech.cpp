// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Speech-LLM chat API (/v1/chat/completions): parsing, validation, admission and rendering. Error
// messages, field order and number formatting are exact because clients match them.
#include "chat_api.h"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "engine_registry.h"
#if defined(NEMO_SPEECH_REGISTRY_ASR)
#include "audio_file.h"
#include "audio_ingest.h"
#include "base64.h"
#include "errors.h"
#include "subtitles.h"
#endif

namespace nemo_speech::http::chat {

#if defined(NEMO_SPEECH_REGISTRY_ASR)
namespace {

// ------------------------------------------------------------------------------------------------
// Errors. Thrown inside the request path and rendered once as the OpenAI envelope.

struct ApiError {
    int status;
    std::string code;
    std::string message;
    std::string param;  // empty = null
};

ApiError
bad_request(std::string message, std::string param = {}) {
    return {400, "invalid_request", std::move(message), std::move(param)};
}

ApiError
engine_failure() {
    return {503, "engine_error", "backend request failed", {}};
}

// ------------------------------------------------------------------------------------------------
// JSON input. The parser keeps document order, repeated keys and number lexemes so requests can be
// checked strictly (unknown fields, repeated keys, integer seeds); clients match substrings of its
// errors such as "unknown field".

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    std::string text;  // string value, or the number's lexeme
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;  // document order, repeated keys kept
    std::vector<size_t> key_ends;                       // offset just past each member key
    size_t end = 0;                                     // offset just past the value

    bool is(Kind k) const { return kind == k; }
    bool is_string(std::string_view value) const { return kind == Kind::String && text == value; }

    // Object view with sorted keys; the last repeated key wins.
    std::map<std::string, const Json*> map() const {
        std::map<std::string, const Json*> out;
        for (const auto& [key, value] : members) out[key] = &value;
        return out;
    }
};

struct JsonError {
    std::string message;
};

// "at line L column C": `at_char` names the offending byte (1-based column), otherwise the position
// just past the consumed input.
std::string
json_position(std::string_view input, size_t offset, bool at_char) {
    offset = std::min(offset, input.size());
    size_t line = 1, line_start = 0;
    for (size_t i = 0; i < offset; ++i)
        if (input[i] == '\n') {
            ++line;
            line_start = i + 1;
        }
    return " at line " + std::to_string(line) + " column " +
           std::to_string(offset - line_start + (at_char ? 1 : 0));
}

class JsonParser {
   public:
    explicit JsonParser(std::string_view input) : in_(input) {}

    Json document() {
        Json value = parse_value(0);
        whitespace();
        if (pos_ < in_.size())
            fail_at("trailing characters");
        return value;
    }

   private:
    static constexpr size_t kMaxDepth = 128;  // nesting limit

    [[noreturn]] void fail_at(const char* what) {
        throw JsonError{what + json_position(in_, pos_, true)};
    }
    [[noreturn]] void fail_eof(const char* what) {
        throw JsonError{what + json_position(in_, in_.size(), false)};
    }

    void whitespace() {
        while (pos_ < in_.size() &&
               (in_[pos_] == ' ' || in_[pos_] == '\t' || in_[pos_] == '\n' || in_[pos_] == '\r'))
            ++pos_;
    }

    Json parse_value(size_t depth) {
        whitespace();
        if (pos_ >= in_.size())
            fail_eof("EOF while parsing a value");
        Json value;
        switch (in_[pos_]) {
            case 'n':
                literal("null");
                break;
            case 't':
                literal("true");
                value.kind = Json::Kind::Bool;
                value.boolean = true;
                break;
            case 'f':
                literal("false");
                value.kind = Json::Kind::Bool;
                break;
            case '"':
                value.kind = Json::Kind::String;
                value.text = parse_string();
                break;
            case '[':
                if (depth + 1 > kMaxDepth)
                    fail_at("recursion limit exceeded");
                value = parse_array(depth + 1);
                break;
            case '{':
                if (depth + 1 > kMaxDepth)
                    fail_at("recursion limit exceeded");
                value = parse_object(depth + 1);
                break;
            default:
                if (in_[pos_] == '-' || (in_[pos_] >= '0' && in_[pos_] <= '9')) {
                    value.kind = Json::Kind::Number;
                    value.text = parse_number();
                } else {
                    fail_at("expected value");
                }
        }
        value.end = pos_;
        return value;
    }

    void literal(std::string_view word) {
        for (char c : word) {
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing a value");
            if (in_[pos_] != c)
                fail_at("expected ident");
            ++pos_;
        }
    }

    std::string parse_number() {
        const size_t begin = pos_;
        auto digits = [&] {
            const size_t start = pos_;
            while (pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9') ++pos_;
            return pos_ > start;
        };
        if (in_[pos_] == '-')
            ++pos_;
        if (pos_ >= in_.size())
            fail_eof("EOF while parsing a value");
        if (in_[pos_] == '0') {
            ++pos_;
            if (pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9')
                fail_at("invalid number");
        } else if (!digits()) {
            fail_at("invalid number");
        }
        bool fractional = false;
        if (pos_ < in_.size() && in_[pos_] == '.') {
            ++pos_;
            fractional = true;
            if (!digits())
                pos_ >= in_.size() ? fail_eof("EOF while parsing a value")
                                   : fail_at("invalid number");
        }
        if (pos_ < in_.size() && (in_[pos_] == 'e' || in_[pos_] == 'E')) {
            ++pos_;
            fractional = true;
            if (pos_ < in_.size() && (in_[pos_] == '+' || in_[pos_] == '-'))
                ++pos_;
            if (!digits())
                pos_ >= in_.size() ? fail_eof("EOF while parsing a value")
                                   : fail_at("invalid number");
        }
        std::string lexeme(in_.substr(begin, pos_ - begin));
        if (fractional && !std::isfinite(std::strtod(lexeme.c_str(), nullptr)))
            throw JsonError{"number out of range" + json_position(in_, pos_, false)};
        return lexeme;
    }

    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing a string");
            const char c = in_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9')
                value |= c - '0';
            else if (c >= 'a' && c <= 'f')
                value |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                value |= c - 'A' + 10;
            else
                fail_at("invalid escape");
        }
        return value;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    // One UTF-8 sequence starting at pos_ (a lead byte >= 0x80), strictly validated.
    void utf8_sequence(std::string& out) {
        const auto byte = [&](size_t i) { return static_cast<unsigned char>(in_[i]); };
        const unsigned char lead = byte(pos_);
        size_t length = 0;
        uint32_t cp = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
            cp = lead & 0x1F;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            cp = lead & 0x0F;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            cp = lead & 0x07;
        } else {
            fail_at("invalid unicode code point");
        }
        if (pos_ + length > in_.size())
            fail_eof("EOF while parsing a string");
        for (size_t i = 1; i < length; ++i) {
            if ((byte(pos_ + i) & 0xC0) != 0x80)
                fail_at("invalid unicode code point");
            cp = (cp << 6) | (byte(pos_ + i) & 0x3F);
        }
        if ((length == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
            (length == 4 && (cp < 0x10000 || cp > 0x10FFFF)))
            fail_at("invalid unicode code point");
        out.append(in_.substr(pos_, length));
        pos_ += length;
    }

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        for (;;) {
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing a string");
            const auto c = static_cast<unsigned char>(in_[pos_]);
            if (c == '"') {
                ++pos_;
                return out;
            }
            if (c < 0x20)
                fail_at("control character (\\u0000-\\u001F) found while parsing a string");
            if (c >= 0x80) {
                utf8_sequence(out);
                continue;
            }
            if (c != '\\') {
                // a run of plain ASCII (a base64 upload is one run): one append
                size_t end = pos_ + 1;
                while (end < in_.size()) {
                    const auto next = static_cast<unsigned char>(in_[end]);
                    if (next == '"' || next == '\\' || next < 0x20 || next >= 0x80)
                        break;
                    ++end;
                }
                out.append(in_.data() + pos_, end - pos_);
                pos_ = end;
                continue;
            }
            ++pos_;
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing a string");
            const char escape = in_[pos_++];
            switch (escape) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xDC00 && cp <= 0xDFFF)
                        fail_at("lone leading surrogate in hex escape");
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 1 >= in_.size() || in_[pos_] != '\\' || in_[pos_ + 1] != 'u')
                            fail_at("unexpected end of hex escape");
                        pos_ += 2;
                        const uint32_t low = hex4();
                        if (low < 0xDC00 || low > 0xDFFF)
                            fail_at("lone leading surrogate in hex escape");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    --pos_;
                    fail_at("invalid escape");
            }
        }
    }

    Json parse_array(size_t depth) {
        Json value;
        value.kind = Json::Kind::Array;
        ++pos_;
        whitespace();
        if (pos_ < in_.size() && in_[pos_] == ']') {
            ++pos_;
            return value;
        }
        for (;;) {
            value.items.push_back(parse_value(depth));
            whitespace();
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing a list");
            if (in_[pos_] == ']') {
                ++pos_;
                return value;
            }
            if (in_[pos_] != ',')
                fail_at("expected `,` or `]`");
            ++pos_;
            whitespace();
            if (pos_ < in_.size() && in_[pos_] == ']')
                fail_at("trailing comma");
        }
    }

    Json parse_object(size_t depth) {
        Json value;
        value.kind = Json::Kind::Object;
        ++pos_;
        whitespace();
        if (pos_ < in_.size() && in_[pos_] == '}') {
            ++pos_;
            return value;
        }
        for (;;) {
            whitespace();
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing an object");
            if (in_[pos_] != '"')
                fail_at("key must be a string");
            std::string key = parse_string();
            const size_t key_end = pos_;
            whitespace();
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing an object");
            if (in_[pos_] != ':')
                fail_at("expected `:`");
            ++pos_;
            Json member = parse_value(depth);
            value.members.emplace_back(std::move(key), std::move(member));
            value.key_ends.push_back(key_end);
            whitespace();
            if (pos_ >= in_.size())
                fail_eof("EOF while parsing an object");
            if (in_[pos_] == '}') {
                ++pos_;
                return value;
            }
            if (in_[pos_] != ',')
                fail_at("expected `,` or `}`");
            ++pos_;
            whitespace();
            if (pos_ < in_.size() && in_[pos_] == '}')
                fail_at("trailing comma");
        }
    }

    std::string_view in_;
    size_t pos_ = 0;
};

bool
is_integer_lexeme(const std::string& lexeme) {
    return lexeme.find_first_of(".eE") == std::string::npos;
}

// Error wording for a value of the wrong type.
std::string
unexpected(const Json& value) {
    switch (value.kind) {
        case Json::Kind::Null:
            return "null";
        case Json::Kind::Bool:
            return std::string("boolean `") + (value.boolean ? "true" : "false") + "`";
        case Json::Kind::Number:
            return (is_integer_lexeme(value.text) ? "integer `" : "floating point `") + value.text +
                   "`";
        case Json::Kind::String: {
            std::string quoted = "string \"";
            for (char c : value.text) {
                if (c == '"' || c == '\\')
                    quoted += '\\';
                quoted += c;
            }
            return quoted + "\"";
        }
        case Json::Kind::Array:
            return "sequence";
        case Json::Kind::Object:
            return "map";
    }
    return "value";
}

// The `expected ...` list of an unknown-field error.
std::string
expected_fields(const std::vector<const char*>& names) {
    std::string out;
    if (names.size() == 1)
        return std::string("expected `") + names[0] + "`";
    if (names.size() == 2)
        return std::string("expected `") + names[0] + "` or `" + names[1] + "`";
    out = "expected one of ";
    for (size_t i = 0; i < names.size(); ++i)
        out += (i ? ", `" : "`") + std::string(names[i]) + "`";
    return out;
}

enum class FieldType { String, Bool, I64, Number, Value };

struct FieldSpec {
    const char* name;
    FieldType type;
};

const std::vector<FieldSpec> kChatFields = {
    {"model", FieldType::String},
    {"messages", FieldType::Value},
    {"stream", FieldType::Bool},
    {"stream_options", FieldType::Value},
    {"temperature", FieldType::Number},
    {"seed", FieldType::I64},
    {"user", FieldType::Value},
    {"metadata", FieldType::Value},
    {"modalities", FieldType::Value},
    {"n", FieldType::Value},
    {"tools", FieldType::Value},
    {"tool_choice", FieldType::Value},
    {"response_format", FieldType::Value},
    {"max_tokens", FieldType::Value},
    {"max_completion_tokens", FieldType::Value},
    {"audio", FieldType::Value},
    {"language", FieldType::String},
    {"timestamp_granularities", FieldType::Value},
    {"word_timestamps", FieldType::Bool},
    {"subtitle_format", FieldType::String}};

const std::vector<FieldSpec> kStreamOptionFields = {{"include_usage", FieldType::Bool}};

using Fields = std::map<std::string, const Json*>;

void
check_type(std::string_view body, const Json& value, FieldType type) {
    const auto wrong = [&](const char* expected) {
        throw JsonError{
            "invalid type: " + unexpected(value) + ", expected " + expected +
            json_position(body, value.end, false)};
    };
    switch (type) {
        case FieldType::String:
            if (!value.is(Json::Kind::String))
                wrong("a string");
            break;
        case FieldType::Bool:
            if (!value.is(Json::Kind::Bool))
                wrong("a boolean");
            break;
        case FieldType::Number:
            if (!value.is(Json::Kind::Number))
                wrong("a JSON number");
            break;
        case FieldType::I64: {
            if (!value.is(Json::Kind::Number) || !is_integer_lexeme(value.text))
                wrong("i64");
            int64_t parsed = 0;
            const auto* first = value.text.data();
            const auto* last = first + value.text.size();
            const auto [ptr, ec] = std::from_chars(first, last, parsed);
            if (ec != std::errc() || ptr != last) {
                if (value.text[0] == '-')  // below i64: reported as a float
                    throw JsonError{
                        "invalid type: floating point `" + value.text + "`, expected i64" +
                        json_position(body, value.end, false)};
                throw JsonError{
                    "invalid value: integer `" + value.text + "`, expected i64" +
                    json_position(body, value.end, false)};
            }
            break;
        }
        case FieldType::Value:
            break;
    }
}

// A strict object whose fields are all optional: members in document order; unknown and repeated
// keys and wrong types are errors; null is absent. Returns the present fields.
Fields
walk_struct(
    std::string_view body, const Json& object, const char* name,
    const std::vector<FieldSpec>& specs) {
    if (!object.is(Json::Kind::Object))
        throw JsonError{
            "invalid type: " + unexpected(object) + ", expected struct " + name +
            json_position(body, object.end, false)};
    std::vector<const char*> names;
    for (const auto& spec : specs) names.push_back(spec.name);
    Fields present;
    std::set<std::string> seen;
    for (size_t i = 0; i < object.members.size(); ++i) {
        const auto& [key, value] = object.members[i];
        const auto spec = std::find_if(
            specs.begin(), specs.end(), [&](const FieldSpec& s) { return key == s.name; });
        if (spec == specs.end())
            throw JsonError{
                "unknown field `" + key + "`, " + expected_fields(names) +
                json_position(body, object.key_ends[i], false)};
        if (!seen.insert(key).second)
            throw JsonError{
                "duplicate field `" + key + "`" + json_position(body, object.key_ends[i], false)};
        if (value.is(Json::Kind::Null))
            continue;  // null is absent
        check_type(body, value, spec->type);
        present[key] = &value;
    }
    return present;
}

// An object whose fields are all required strings (input_audio, a text part): keys checked in
// sorted order, errors without a position. Returns an error message or the field values (views of
// the document, which must outlive them).
std::optional<std::string>
string_struct(
    const Json& value, const char* name, const std::vector<const char*>& fields,
    std::map<std::string, std::string_view>& out) {
    if (!value.is(Json::Kind::Object))
        return "invalid type: " + unexpected(value) + ", expected struct " + name;
    for (const auto& [key, member] : value.map()) {
        if (std::find_if(fields.begin(), fields.end(), [&](const char* f) { return key == f; }) ==
            fields.end())
            return "unknown field `" + key + "`, " + expected_fields(fields);
        if (!member->is(Json::Kind::String))
            return "invalid type: " + unexpected(*member) + ", expected a string";
        out[key] = member->text;
    }
    for (const char* field : fields)
        if (!out.count(field))
            return std::string("missing field `") + field + "`";
    return std::nullopt;
}

// ------------------------------------------------------------------------------------------------
// JSON output: fixed field order, string escaping and float formatting.

// Moves `digits` x 10^exponent (first digit before the point) one unit in the last digit up or
// down, keeping the digit count.
void
step_last_digit(std::string& digits, int& exponent, bool up) {
    size_t i = digits.size();
    while (i-- > 0) {
        if (up ? digits[i] != '9' : digits[i] != '0') {
            digits[i] = static_cast<char>(digits[i] + (up ? 1 : -1));
            break;
        }
        digits[i] = up ? '0' : '9';
    }
    if (up && digits[0] == '0') {  // 9.99 -> 1.00 x 10
        digits[0] = '1';
        ++exponent;
    } else if (!up && digits[0] == '0') {  // 1.00 -> 9.99 / 10
        digits.assign(digits.size(), '9');
        --exponent;
    }
}

// The fewest significant digits that read back as `magnitude` (finite, positive), the closest of
// them if several do, and the decimal exponent of the first: the digits std::to_chars chooses.
// Floating-point std::to_chars needs libstdc++ from GCC 11; the Linux release image builds with
// GCC 9. snprintf and strtod round correctly on every supported C library.
void
shortest_digits(double magnitude, std::string& digits, int& exponent) {
    char text[40];
    for (int precision = 0; precision < 17; ++precision) {
        std::snprintf(text, sizeof(text), "%.*e", precision, magnitude);
        const char* e = std::strchr(text, 'e');
        digits.assign(1, text[0]);
        if (precision > 0)
            digits.append(text + 2, static_cast<size_t>(e - text - 2));
        exponent = std::atoi(e + 1);
        const double nearest = std::strtod(text, nullptr);
        if (nearest == magnitude)
            break;
        // The values that read back as `magnitude` extend further to one side at powers of two,
        // so the next digit string on the far side can read back when the nearest does not.
        std::string candidate = digits;
        int candidate_exponent = exponent;
        step_last_digit(candidate, candidate_exponent, nearest < magnitude);
        std::string candidate_text(1, candidate[0]);
        if (precision > 0)
            candidate_text += text[1] + candidate.substr(1);  // text[1]: the locale's radix
        candidate_text += "e" + std::to_string(candidate_exponent);
        if (std::strtod(candidate_text.c_str(), nullptr) == magnitude) {
            digits = candidate;
            exponent = candidate_exponent;
            break;
        }
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
}

// Shortest round-trip digits, decimal notation with a fractional part in [1e-5, 1e16), exponent
// outside
// ("1e16", "1.5e-7").
std::string
format_double(double value) {
    if (!std::isfinite(value))
        return "null";
    if (value == 0.0)
        return std::signbit(value) ? "-0.0" : "0.0";
    const double magnitude = std::fabs(value);
    std::string digits;
    int exponent = 0;
    shortest_digits(magnitude, digits, exponent);
    std::string text = value < 0 ? "-" : "";
    if (magnitude >= 1e16 || magnitude < 1e-5) {
        text += digits[0];
        if (digits.size() > 1)
            text += "." + digits.substr(1);
        return text + "e" + std::to_string(exponent);
    }
    if (exponent < 0)
        return text + "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
    const size_t integer_digits = static_cast<size_t>(exponent) + 1;
    if (digits.size() <= integer_digits)
        return text + digits + std::string(integer_digits - digits.size(), '0') + ".0";
    return text + digits.substr(0, integer_digits) + "." + digits.substr(integer_digits);
}

class JsonWriter {
   public:
    JsonWriter& begin_object() { return open('{'); }
    JsonWriter& end_object() { return close('}'); }
    JsonWriter& begin_array() { return open('['); }
    JsonWriter& end_array() { return close(']'); }

    JsonWriter& key(std::string_view name) {
        separate();
        escape(name);
        out_ += ':';
        after_key_ = true;
        return *this;
    }
    JsonWriter& string(std::string_view value) {
        separate();
        escape(value);
        return *this;
    }
    JsonWriter& number(double value) {
        separate();
        out_ += format_double(value);
        return *this;
    }
    JsonWriter& integer(int64_t value) {
        separate();
        out_ += std::to_string(value);
        return *this;
    }
    JsonWriter& boolean(bool value) {
        separate();
        out_ += value ? "true" : "false";
        return *this;
    }
    JsonWriter& null() {
        separate();
        out_ += "null";
        return *this;
    }
    std::string take() { return std::move(out_); }

   private:
    JsonWriter& open(char c) {
        separate();
        out_ += c;
        first_.push_back(true);
        return *this;
    }
    JsonWriter& close(char c) {
        out_ += c;
        first_.pop_back();
        return *this;
    }
    void separate() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (!first_.empty()) {
            if (!first_.back())
                out_ += ',';
            first_.back() = false;
        }
    }
    // Invalid UTF-8 becomes U+FFFD so the body stays valid JSON.
    void escape(std::string_view value) {
        static const char* hex = "0123456789abcdef";
        out_ += '"';
        for (size_t i = 0; i < value.size();) {
            const auto c = static_cast<unsigned char>(value[i]);
            if (c < 0x80) {
                switch (c) {
                    case '"':
                        out_ += "\\\"";
                        break;
                    case '\\':
                        out_ += "\\\\";
                        break;
                    case '\b':
                        out_ += "\\b";
                        break;
                    case '\f':
                        out_ += "\\f";
                        break;
                    case '\n':
                        out_ += "\\n";
                        break;
                    case '\r':
                        out_ += "\\r";
                        break;
                    case '\t':
                        out_ += "\\t";
                        break;
                    default:
                        if (c < 0x20) {
                            out_ += "\\u00";
                            out_ += hex[c >> 4];
                            out_ += hex[c & 15];
                        } else {
                            out_ += static_cast<char>(c);
                        }
                }
                ++i;
                continue;
            }
            const size_t length = utf8_length(value, i);
            if (length == 0) {
                out_ += "\xEF\xBF\xBD";
                ++i;
            } else {
                out_.append(value.substr(i, length));
                i += length;
            }
        }
        out_ += '"';
    }
    static size_t utf8_length(std::string_view s, size_t i) {
        const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
        const unsigned char lead = b(i);
        size_t length = lead >= 0xC2 && lead <= 0xDF   ? 2
                        : lead >= 0xE0 && lead <= 0xEF ? 3
                        : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                       : 0;
        if (length == 0 || i + length > s.size())
            return 0;
        uint32_t cp = lead & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
        for (size_t k = 1; k < length; ++k) {
            if ((b(i + k) & 0xC0) != 0x80)
                return 0;
            cp = (cp << 6) | (b(i + k) & 0x3F);
        }
        if ((length == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
            (length == 4 && (cp < 0x10000 || cp > 0x10FFFF)))
            return 0;
        return length;
    }

    std::string out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

std::string
error_body(const ApiError& error) {
    JsonWriter json;
    json.begin_object().key("error").begin_object();
    json.key("message").string(error.message);
    json.key("type").string(
        error.status == 429   ? "rate_limit_error"
        : error.status >= 500 ? "server_error"
                              : "invalid_request_error");
    json.key("param");
    if (error.param.empty())
        json.null();
    else
        json.string(error.param);
    json.key("code").string(error.code);
    json.end_object().end_object();
    return json.take();
}

// ------------------------------------------------------------------------------------------------
// Text helpers.

// Decode one UTF-8 code point at `i` (advancing it); malformed bytes decode as U+FFFD.
uint32_t
next_code_point(std::string_view s, size_t& i) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char lead = b(i);
    size_t length = lead < 0x80                    ? 1
                    : lead >= 0xC2 && lead <= 0xDF ? 2
                    : lead >= 0xE0 && lead <= 0xEF ? 3
                    : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                   : 0;
    if (length == 0 || i + length > s.size()) {
        ++i;
        return 0xFFFD;
    }
    if (length == 1) {
        ++i;
        return lead;
    }
    uint32_t cp = lead & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
    for (size_t k = 1; k < length; ++k) {
        if ((b(i + k) & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (b(i + k) & 0x3F);
    }
    i += length;
    return cp;
}

bool
is_white_space(uint32_t cp) {  // Unicode White_Space
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

std::string_view
trim(std::string_view s) {
    size_t begin = 0;
    while (begin < s.size()) {
        size_t next = begin;
        if (!is_white_space(next_code_point(s, next)))
            break;
        begin = next;
    }
    size_t end = begin;
    for (size_t i = begin; i < s.size();) {
        if (!is_white_space(next_code_point(s, i)))
            end = i;
    }
    return s.substr(begin, end - begin);
}

// ------------------------------------------------------------------------------------------------
// Subtitle cues.

// A word as subtitles show it: its running text (SpeechLmWord::display) when set.
std::string_view
shown(const asr::SpeechLmWord& word) {
    return word.display.empty() ? std::string_view(word.word) : std::string_view(word.display);
}

// The cues of the aligned words (subtitles.h), which come in the order of their starts.
std::vector<subtitle::WordCue>
build_cues(const std::vector<asr::SpeechLmWord>& words) {
    std::vector<subtitle::TimedWord> timed;
    timed.reserve(words.size());
    for (const auto& word : words)
        timed.push_back({std::string(shown(word)), word.start, word.end, word.joins_previous});
    return subtitle::word_cues(timed);
}

// ------------------------------------------------------------------------------------------------
// The validated request.

struct ValidatedRequest {
    std::string model;
    std::optional<std::string> language;  // as forwarded
    bool word_timestamps = false;  // word_timestamps, or timestamp_granularities named word/segment
    std::optional<std::string> subtitle_format;  // srt or vtt: the cues also as subtitle text
    std::optional<int64_t> seed;
    std::optional<double> temperature;
    std::optional<std::string> prompt;  // the instruction as given
    bool stream = false;                // server-sent events
    bool include_usage = false;         // stream_options.include_usage: a final usage chunk
    std::string audio;                  // decoded upload bytes
    std::string format;                 // input_audio.format
};

struct InputAudioPart {
    std::string_view data;  // base64, in the request document
    std::string format;
};

// Exactly one user message whose content holds one input_audio part and at most one text part.
std::pair<InputAudioPart, std::optional<std::string>>
message_parts(const Json& list) {
    const auto refuse = [](std::string message) {
        return bad_request(std::move(message), "messages");
    };
    const auto shape = [&] {
        return refuse(
            "messages must be a list with exactly one user message whose content is a list of "
            "input_audio and text parts");
    };
    if (!list.is(Json::Kind::Array) || list.items.size() != 1 ||
        !list.items[0].is(Json::Kind::Object))
        throw shape();
    auto message = list.items[0].map();
    {
        const auto it = message.find("role");
        if (it == message.end() || !it->second->is_string("user"))
            throw refuse("the message role must be user");
        message.erase(it);
    }
    const auto content = message.find("content");
    if (content == message.end() || !content->second->is(Json::Kind::Array))
        throw shape();
    const Json& parts = *content->second;
    message.erase(content);
    message.erase("name");
    if (!message.empty())
        throw refuse("unsupported message field: " + message.begin()->first);

    std::optional<InputAudioPart> audio;
    std::optional<std::string> text;
    for (const Json& part : parts.items) {
        if (!part.is(Json::Kind::Object))
            throw shape();
        auto fields = part.map();
        const auto kind = fields.find("type");
        const Json* type = kind == fields.end() ? nullptr : kind->second;
        if (kind != fields.end())
            fields.erase(kind);
        if (!type || !type->is(Json::Kind::String))
            throw refuse("content parts need a type");
        if (type->text == "input_audio") {
            if (audio)
                throw refuse("the message must carry exactly one input_audio part");
            // {input_audio: {data, format}}; unknown fields are refused at both levels.
            std::optional<std::string> error;
            const Json* inner = nullptr;
            for (const auto& [key, value] : fields) {
                if (key != "input_audio") {
                    error = "unknown field `" + key + "`, expected `input_audio`";
                    break;
                }
                inner = value;
                std::map<std::string, std::string_view> values;
                if ((error = string_struct(*value, "InputAudio", {"data", "format"}, values)))
                    break;
                audio = InputAudioPart{values["data"], std::string(values["format"])};
            }
            if (!error && !inner)
                error = "missing field `input_audio`";
            if (error)
                throw refuse("invalid input_audio part: " + *error);
        } else if (type->text == "text") {
            if (text)
                throw refuse("the message must carry at most one text part");
            Json object;
            object.kind = Json::Kind::Object;
            for (const auto& [key, value] : fields) object.members.emplace_back(key, *value);
            std::map<std::string, std::string_view> values;
            if (auto error = string_struct(object, "InputTextPart", {"text"}, values))
                throw refuse("invalid text part: " + *error);
            text = std::string(values["text"]);
        } else {
            throw refuse(
                "unsupported content part type: " + type->text + " (supported: input_audio, text)");
        }
    }
    if (!audio)
        throw refuse("the message must carry exactly one input_audio part");
    // An empty or whitespace-only text part is no instruction.
    if (text && trim(*text).empty())
        text.reset();
    return {std::move(*audio), std::move(text)};
}

std::optional<std::string>
field_string(const Fields& fields, const char* name) {
    const auto it = fields.find(name);
    return it == fields.end() ? std::nullopt : std::optional<std::string>(it->second->text);
}

// Admission of the sampling, audio, language and timestamp fields. Every refusal names its field.
ValidatedRequest
admit(
    std::string model, const Json* language, const Json* granularities, const Json* word_timestamps,
    const Json* subtitle_format, const Json* temperature, const Json* seed,
    InputAudioPart audio_part, std::optional<std::string> prompt) {
    ValidatedRequest request;
    request.model = std::move(model);
    if (seed) {
        request.seed = std::stoll(seed->text);
        if (*request.seed < 0)
            throw bad_request("seed must be a non-negative integer", "seed");
    }
    if (temperature) {
        const double value = std::strtod(temperature->text.c_str(), nullptr);
        if (!std::isfinite(value) || value < 0.0 || value > 1.0)
            throw bad_request("temperature must be a finite number between 0 and 1", "temperature");
        request.temperature = value;
    }
    // timestamp_granularities as in /v1/audio/transcriptions: any of "word" and "segment" asks for
    // the aligned words ("segments" stays empty).
    if (granularities) {
        const auto refuse = [] {
            return bad_request(
                "timestamp_granularities must be a list of \"word\" and \"segment\"",
                "timestamp_granularities");
        };
        if (!granularities->is(Json::Kind::Array))
            throw refuse();
        for (const Json& item : granularities->items)
            if (!item.is_string("word") && !item.is_string("segment"))
                throw refuse();
        request.word_timestamps = !granularities->items.empty();
    }
    // word_timestamps asks for the same as timestamp_granularities; a request names one of them.
    if (word_timestamps) {
        if (granularities)
            throw bad_request(
                "use timestamp_granularities or word_timestamps", "timestamp_granularities");
        request.word_timestamps = word_timestamps->boolean;
    }
    // subtitle_format: the subtitle cues also as SRT or WebVTT text (transcription.subtitles).
    if (subtitle_format) {
        if (subtitle_format->text != "srt" && subtitle_format->text != "vtt")
            throw bad_request("subtitle_format must be srt or vtt", "subtitle_format");
        if (!request.word_timestamps)
            throw bad_request(
                subtitle_format->text + " requires word_timestamps=true", "word_timestamps");
        request.subtitle_format = subtitle_format->text;
    }
    // Audio errors name no param.
    if (audio_part.data.empty())
        throw bad_request("input_audio.data must not be empty");
    if (audio_part.format.empty() || audio_part.format.size() > 16 ||
        !std::all_of(audio_part.format.begin(), audio_part.format.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        }))
        throw bad_request("input_audio.format must be a lowercase container name such as wav");
    auto decoded = decode_base64(audio_part.data);
    if (!decoded)
        throw bad_request("input_audio.data is not valid base64");
    if (decoded->empty())
        throw bad_request("input_audio.data must not be empty");
    request.audio = std::move(*decoded);
    request.format = std::move(audio_part.format);
    // Timestamps in any language; without one the model picks it, as without timestamps.
    if (language)
        request.language = language->text;
    request.prompt = std::move(prompt);
    return request;
}

ValidatedRequest
validate_chat(std::string_view body, const Json& root) {
    const auto fields = walk_struct(body, root, "ChatCompletionsRequest", kChatFields);
    const auto present = [&](const char* name) -> const Json* {
        const auto it = fields.find(name);
        return it == fields.end() ? nullptr : it->second;
    };
    const Json* stream = present("stream");
    const bool streamed = stream && stream->boolean;
    bool include_usage = false;
    if (const Json* options = present("stream_options")) {
        if (!streamed)
            throw bad_request("stream_options requires stream", "stream_options");
        const auto values = walk_struct(body, *options, "StreamOptions", kStreamOptionFields);
        const auto usage = values.find("include_usage");
        include_usage = usage != values.end() && usage->second->boolean;
    }
    if (const Json* n = present("n");
        n && (!n->is(Json::Kind::Number) || std::strtod(n->text.c_str(), nullptr) != 1.0))
        throw bad_request("n must be 1: one transcript per upload", "n");
    if (present("tools"))
        throw bad_request("tools are not supported", "tools");
    if (const Json* choice = present("tool_choice"); choice && !choice->is_string("none"))
        throw bad_request("tools are not supported", "tool_choice");
    if (const Json* format = present("response_format")) {
        const auto map = format->map();
        const auto type = map.find("type");
        if (!format->is(Json::Kind::Object) || type == map.end() ||
            !type->second->is_string("text"))
            throw bad_request(
                "response_format must be {\"type\": \"text\"}: the transcript is plain text and "
                "the structured payload rides under transcription",
                "response_format");
    }
    for (const char* name : {"max_tokens", "max_completion_tokens"})
        if (present(name))
            throw bad_request(
                std::string(name) +
                    " is not supported: the decoder budget follows the audio duration",
                name);
    if (present("audio"))
        throw bad_request("audio output is not supported", "audio");
    if (const Json* modalities = present("modalities");
        modalities && !(modalities->is(Json::Kind::Array) && modalities->items.size() == 1 &&
                        modalities->items[0].is_string("text")))
        throw bad_request("modalities must be [\"text\"]", "modalities");
    const auto model = field_string(fields, "model");
    if (!model || model->empty())
        throw bad_request("model is required", "model");
    const Json* messages = present("messages");
    if (!messages)
        throw bad_request("messages is required", "messages");
    auto [audio, text] = message_parts(*messages);
    auto request = admit(
        *model, present("language"), present("timestamp_granularities"), present("word_timestamps"),
        present("subtitle_format"), present("temperature"), present("seed"), std::move(audio),
        std::move(text));
    // subtitles are rendered from the whole transcript
    if (streamed && request.subtitle_format)
        throw bad_request("subtitle_format does not support stream=true", "subtitle_format");
    request.stream = streamed;
    request.include_usage = include_usage;
    return request;
}

// ------------------------------------------------------------------------------------------------
// Audio. PCM16 and G.711 WAV headers are checked before admission (400s with the messages below);
// decoding failures of any upload are reported after admission.

constexpr size_t kMinSamples = 320;  // two 10 ms mel hops at 16 kHz

struct DecodedAudio {
    std::vector<float> samples;
    int sample_rate = 0;
};

// Decode an upload (WAV, FLAC or MP3) of at most `max_seconds`; decoding stops at the limit. A
// PCM16 or G.711 WAV that passed frontend_wav_checks keeps its sample-count rule (max_samples + 1
// frames pass). An ApiError when it cannot be decoded (422) or is longer (400).
std::variant<DecodedAudio, ApiError>
decode_samples(const std::string& bytes, int max_seconds, bool wav_checked) {
    audio::IngestOptions limits;
    if (wav_checked) {
        limits.max_samples = static_cast<size_t>(max_seconds) * 16000;
        limits.max_duration_seconds = 0;
    } else {
        limits.max_samples = static_cast<size_t>(-1);
        limits.max_duration_seconds = max_seconds;
    }
    try {
        auto file = audio::decode_any(bytes.data(), bytes.size(), "audio upload", limits);
        return DecodedAudio{std::move(file.samples), 16000};
    }
    catch (const audio::AudioLimitError& e) {
        return bad_request(e.what());
    }
    catch (const std::exception&) {
        return ApiError{422, "invalid_request", "invalid or unsupported audio file", {}};
    }
}

uint32_t
le32(const std::string& s, size_t at) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | static_cast<unsigned char>(s[at + i]);
    return v;
}

uint16_t
le16(const std::string& s, size_t at) {
    return static_cast<uint16_t>(
        static_cast<unsigned char>(s[at]) | (static_cast<unsigned char>(s[at + 1]) << 8));
}

// RIFF checks for PCM16 and G.711 WAVs: throws the 400s; returns false for other encodings, which
// are left to the decoder. A streamed WAV's 0xFFFFFFFF RIFF and data sizes mean "to the end of the
// file" (whole frames), as in audio_ingest.
bool
frontend_wav_checks(const std::string& bytes, size_t max_samples) {
    const auto malformed = [](const char* what) {
        return bad_request(std::string("malformed WAV: ") + what);
    };
    constexpr uint32_t kUnknownSize = 0xFFFFFFFFu;
    if (le32(bytes, 4) != kUnknownSize && static_cast<uint64_t>(le32(bytes, 4)) + 8 > bytes.size())
        throw malformed("truncated RIFF payload");
    bool open_ended = false;
    size_t pos = 12;
    std::optional<std::array<uint32_t, 4>> fmt;  // format, channels, rate, bits
    std::optional<std::pair<size_t, size_t>> data;
    while (pos + 8 <= bytes.size()) {
        const std::string id = bytes.substr(pos, 4);
        size_t length = le32(bytes, pos + 4);
        const size_t start = pos + 8;
        if (id == "data" && !data && length == kUnknownSize) {
            data = std::make_pair(start, bytes.size() - start);
            open_ended = true;
            break;
        }
        if (length > bytes.size() - start)
            throw malformed("truncated chunk");
        if (id == "fmt " && !fmt) {
            if (length < 16)
                throw malformed("short fmt chunk");
            fmt = std::array<uint32_t, 4>{
                le16(bytes, start), le16(bytes, start + 2), le32(bytes, start + 4),
                le16(bytes, start + 14)};
        } else if (id == "data" && !data) {
            data = std::make_pair(start, length);
        }
        pos = start + length + (length & 1);
    }
    if (!fmt)
        throw malformed("missing fmt chunk");
    if (!data)
        throw malformed("missing data chunk");
    const auto [format, channels, rate, bits] = *fmt;
    if (channels == 0 || rate == 0)
        throw bad_request("invalid audio: zero channels or sample rate");
    size_t sample_bytes = 0;
    if (format == 1 && bits == 16)
        sample_bytes = 2;
    else if ((format == 6 || format == 7) && bits == 8)
        sample_bytes = 1;
    else
        return false;  // 24/32-bit PCM, float, EXTENSIBLE, ADPCM: left to the decoder
    const size_t frame = sample_bytes * channels;
    if (open_ended)
        data->second -= data->second % frame;
    if (data->second % frame != 0)
        throw malformed("partial audio frame");
    const size_t frames = data->second / frame;
    if (frames > static_cast<size_t>(static_cast<double>(max_samples) * rate / 16000.0) + 1)
        throw bad_request("audio exceeds configured limit");
    // A canonical 16 kHz mono PCM16 file with the 44-byte header passes as it is; anything else is
    // decoded (and resampled) first and bounded by its 16 kHz sample count.
    const bool canonical = data->first == 44 && bytes.size() == 44 + data->second && format == 1 &&
                           channels == 1 && rate == 16000 && bits == 16;
    const size_t samples =
        canonical || rate == 16000
            ? frames
            : static_cast<size_t>(std::llround(static_cast<double>(frames) * 16000.0 / rate));
    if (!canonical && samples > max_samples)
        throw bad_request("audio exceeds configured limit");
    if (samples < kMinSamples)
        throw bad_request("audio is shorter than one feature frame (320 samples at 16 kHz)");
    return true;
}

// ------------------------------------------------------------------------------------------------
// Waiting on the client.

// How often a waiting request checks its client (ClientGone).
constexpr std::chrono::milliseconds kClientPoll{200};

// The status of a request whose client disconnected (nobody reads it).
ApiError
client_gone() {
    return {499, "client_closed_request", "client closed the request", {}};
}

// Waits on `cv` until `ready` holds. With `gone`, the client is checked every kClientPoll (outside
// the lock); false once it has disconnected.
template <class Ready>
bool
wait_for_client(
    std::unique_lock<std::mutex>& lock, std::condition_variable& cv, Ready ready,
    const ClientGone& gone) {
    if (!gone) {
        cv.wait(lock, ready);
        return true;
    }
    while (!cv.wait_for(lock, kClientPoll, ready)) {
        lock.unlock();
        const bool left = gone();
        lock.lock();
        if (left)
            return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Rendering.

constexpr const char* kCanonicalWordVersion = "nemo-ctc-sot-v1";

void
write_word(JsonWriter& json, const asr::SpeechLmWord& word) {
    json.begin_object();
    json.key("word").string(word.word);
    json.key("word_index").integer(word.word_index);
    json.key("start").number(word.start);
    json.key("end").number(word.end);
    if (word.speaker >= 0)
        json.key("speaker").string("spk:" + std::to_string(word.speaker));
    if (word.speaker_slot >= 0)
        json.key("speaker_slot").integer(word.speaker_slot);
    if (word.speaker_confidence >= 0.0f)
        json.key("speaker_confidence").number(word.speaker_confidence);
    json.end_object();
}

// The `transcription` object of a completion: the verbose_json fields of /v1/audio/transcriptions
// (task, language, duration, text, words, segments) plus the alignment and speaker details.
void
write_payload(
    JsonWriter& json, const ValidatedRequest& request, const asr::SpeechLmDetails& details,
    const std::vector<subtitle::WordCue>& cues, const std::string* subtitles = nullptr) {
    const bool aligned = details.alignment_status == "aligned";
    json.begin_object();
    json.key("schema_version").string("1");
    json.key("task").string("transcribe");
    json.key("transcript_origin")
        .string(details.transcript_origin.empty() ? "llm" : details.transcript_origin);
    json.key("language").string(details.language);
    json.key("duration").number(details.duration_seconds);
    json.key("text").string(details.text);
    json.key("segments").begin_array().end_array();
    if (request.word_timestamps) {
        json.key("words").begin_array();
        if (aligned)
            for (const auto& word : details.words) write_word(json, word);
        json.end_array();
        json.key("subtitle_cues").begin_array();
        for (const auto& cue : cues) {
            json.begin_object();
            json.key("first_word").integer(static_cast<int64_t>(cue.first_word));
            json.key("last_word").integer(static_cast<int64_t>(cue.last_word));
            json.key("start").number(cue.start);
            json.key("end").number(cue.end);
            json.key("text").string(cue.text);
            json.end_object();
        }
        json.end_array();
        json.key("word_timestamp_alignment").begin_object();
        json.key("status").string(aligned ? "aligned" : "unavailable");
        if (!details.transcript_origin.empty() && details.transcript_origin != "llm")
            json.key("transcript_origin").string(details.transcript_origin);
        json.key("model");
        details.alignment_model.empty() ? json.null() : json.string(details.alignment_model);
        json.key("canonical_word_version").string(kCanonicalWordVersion);
        if (details.frame_stride_seconds > 0.0)
            json.key("frame_stride_seconds").number(details.frame_stride_seconds);
        json.key("failure_reason");
        details.failure_reason.empty() ? json.null() : json.string(details.failure_reason);
        json.end_object();
        if (aligned) {  // the diarization products come with an alignment
            json.key("diarization_timestamps").begin_array();
            for (const auto& segment : details.diarization) {
                json.begin_object();
                json.key("speaker").integer(segment.speaker);
                json.key("start").number(segment.start);
                json.key("end").number(segment.end);
                json.end_object();
            }
            json.end_array();
            if (details.diarization_frame_seconds > 0.0)
                json.key("diarization_frame_seconds").number(details.diarization_frame_seconds);
            json.key("speaker_word_timestamps").begin_object();
            for (const auto& [label, words] :
                 details.speaker_words) {  // std::map: labels in byte order
                json.key(label).begin_array();
                for (const auto& word : words) write_word(json, word);
                json.end_array();
            }
            json.end_object();
        }
        if (subtitles && request.subtitle_format) {
            json.key("subtitles").begin_object();
            json.key("format").string(*request.subtitle_format);
            json.key("text").string(*subtitles);
            json.end_object();
        }
    }
    json.end_object();
}

// The recognizer's failure as an API error: refusals keep their message, a full backend is a 429
// and anything else is sanitised (details in the log).
ApiError
backend_error(const std::exception_ptr& failure) {
    try {
        std::rethrow_exception(failure);
    }
    catch (const std::invalid_argument& error) {  // the backend refused the request
        const auto message = std::string(trim(error.what()));
        return bad_request(
            !message.empty() && message.size() <= 512 ? message
                                                      : "ASR backend rejected the request");
    }
    catch (const asr::Cancelled&) {  // the client went away
        return client_gone();
    }
    catch (const asr::ResourceExhausted& error) {
        const auto message = std::string(trim(error.what()));
        return ApiError{
            429,
            "overloaded",
            !message.empty() && message.size() <= 512 ? message
                                                      : "ASR backend is at capacity, retry later",
            {}};
    }
    catch (const std::exception& error) {
        std::cerr << "[chat] backend failure: " << error.what() << '\n';
        return engine_failure();
    }
    catch (...) {
        std::cerr << "[chat] backend failure\n";
        return engine_failure();
    }
}

// A completion's speech-LLM facts (from backends without the payload: the plain transcript and the
// request's facts) and the subtitle cues of an aligned transcript.
struct Completion {
    asr::SpeechLmDetails details;
    std::vector<subtitle::WordCue> cues;
};

Completion
completion(const ValidatedRequest& request, const asr::Result& result, double seconds) {
    Completion out;
    auto& details = out.details;
    if (result.speechlm) {
        details = *result.speechlm;
    } else {
        if (!result.alternatives.empty()) {
            const auto& alternative = result.alternatives.front();
            details.text = alternative.transcript;
            for (size_t i = 0; i < alternative.words.size(); ++i) {
                const auto& word = alternative.words[i];
                asr::SpeechLmWord word_out;
                word_out.word = word.word;
                word_out.display = word.display;
                word_out.joins_previous = word.joins_previous;
                word_out.word_index = static_cast<int>(i);
                word_out.start = word.start_time / 1000.0;
                word_out.end = word.end_time / 1000.0;
                word_out.speaker = word.speaker_tag > 0 ? word.speaker_tag - 1 : -1;
                details.words.push_back(std::move(word_out));
            }
        }
        details.duration_seconds = seconds;
        details.timestamps_requested = request.word_timestamps;
        if (request.word_timestamps) {
            details.alignment_status = details.words.empty() ? "unavailable" : "aligned";
            if (details.words.empty())
                details.failure_reason =
                    details.text.empty() ? "empty_transcript" : "ctc_alignment_failed";
        }
    }
    if (details.language.empty())
        details.language = request.language.value_or("und");
    if (details.duration_seconds <= 0.0)
        details.duration_seconds = seconds;
    // A transcript the cue builder cannot use is a backend failure.
    try {
        if (request.word_timestamps && details.alignment_status == "aligned")
            out.cues = build_cues(details.words);
    }
    catch (const std::exception& error) {
        std::cerr << "[chat] response render failed: " << error.what() << '\n';
        throw engine_failure();
    }
    return out;
}

void
write_usage(JsonWriter& json, const asr::SpeechLmDetails& details) {
    json.key("usage").begin_object();
    json.key("prompt_tokens").integer(details.prompt_tokens);
    json.key("completion_tokens").integer(details.completion_tokens);
    json.key("total_tokens").integer(int64_t{details.prompt_tokens} + details.completion_tokens);
    if (details.audio_tokens > 0) {
        json.key("prompt_tokens_details").begin_object();
        json.key("audio_tokens").integer(details.audio_tokens);
        json.end_object();
    }
    json.end_object();
}

// What a streamed completion's decode reports to its writer.
struct StreamChannel {
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;  // reported, not yet written
    bool finished = false;
};

// Wakes the writers of streamed completions. Their text arrives on the engine thread between
// decode steps, where appending it and queueing the channel here is all that happens: the wakeups
// of every streaming writer run on this thread instead of the decode loop.
class StreamWaker {
   public:
    StreamWaker() : thread_([this] { run(); }) {}
    ~StreamWaker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }
    // `piece` of a channel's text: appended, and its writer woken unless a wakeup is pending.
    void report(const std::shared_ptr<StreamChannel>& channel, std::string_view piece) {
        bool first;
        {
            std::lock_guard<std::mutex> lock(channel->mutex);
            first = channel->text.empty();
            channel->text.append(piece);
        }
        if (!first)
            return;
        bool idle;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            idle = pending_.empty();
            pending_.push_back(channel);
        }
        if (idle)
            cv_.notify_one();
    }

   private:
    void run() {
        std::vector<std::shared_ptr<StreamChannel>> wake;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !pending_.empty(); });
                if (pending_.empty())
                    return;
                wake.swap(pending_);
            }
            for (const auto& channel : wake) channel->cv.notify_one();
            wake.clear();
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::shared_ptr<StreamChannel>> pending_;
    bool stop_ = false;
    std::thread thread_;  // last: starts after the members it reads
};

// A streamed completion's decode: the recognizer runs on its own thread and reports the text as it
// is generated. Destroying the run cancels the decode if it is still running (no one will write
// its text) and waits for it (it reads `samples`).
struct StreamRun {
    std::shared_ptr<StreamChannel> channel = std::make_shared<StreamChannel>();
    asr::Result result;  // with `failure`, set under the channel's mutex before `finished`
    std::exception_ptr failure;
    std::vector<float> samples;
    std::atomic<bool> cancel{false};  // the decode's AsrRequestOptions::cancelled
    std::thread worker;

    ~StreamRun() {
        cancel = true;
        if (worker.joinable())
            worker.join();
    }
    // Waits for text or the end of the decode; `piece` is the text not written yet. False once
    // the client has disconnected (`gone`, checked while waiting).
    bool next(std::string& piece, bool& done, const ClientGone& gone) {
        StreamChannel& c = *channel;
        std::unique_lock<std::mutex> lock(c.mutex);
        if (!wait_for_client(lock, c.cv, [&c] { return c.finished || !c.text.empty(); }, gone))
            return false;
        done = c.finished;
        piece = std::exchange(c.text, {});
        return true;
    }
};

// The events of a streamed completion: the assistant role, the text as it is generated, the stop
// chunk with the transcription payload, the usage chunk (include_usage) and [DONE]. A failure
// after the first event is an error event. False when the client went away (a failed write, or
// `gone` while waiting for text).
bool
write_stream(
    StreamRun& run, const ValidatedRequest& request, double seconds, const std::string& id,
    uint64_t created, const StreamWriter& write, const ClientGone& gone) {
    const auto event = [&](const std::function<void(JsonWriter&)>& fill) {
        JsonWriter json;
        json.begin_object();
        json.key("id").string("chatcmpl-" + id);
        json.key("object").string("chat.completion.chunk");
        json.key("created").integer(static_cast<int64_t>(created));
        json.key("model").string(request.model);
        fill(json);
        json.end_object();
        return write("data: " + json.take() + "\n\n");
    };
    const auto choice = [](JsonWriter& json, bool role, std::string_view content, bool stop) {
        json.key("choices").begin_array().begin_object();
        json.key("index").integer(0);
        json.key("delta").begin_object();
        if (role)
            json.key("role").string("assistant");
        if (role || !content.empty())
            json.key("content").string(content);
        json.end_object();
        json.key("logprobs").null();
        json.key("finish_reason");
        stop ? json.string("stop") : json.null();
        json.end_object().end_array();
    };
    const auto fail = [&](const ApiError& error) {
        return write("data: " + error_body(error) + "\n\n") && write("data: [DONE]\n\n");
    };
    if (!event([&](JsonWriter& json) { choice(json, true, "", false); }))
        return false;
    std::string sent;
    for (bool done = false; !done;) {
        std::string piece;
        if (!run.next(piece, done, gone)) {
            run.cancel = true;
            return false;
        }
        if (piece.empty())
            continue;
        if (!event([&](JsonWriter& json) { choice(json, false, piece, false); }))
            return false;
        sent += piece;
    }
    if (run.failure)
        return fail(backend_error(run.failure));
    Completion out;
    try {
        out = completion(request, run.result, seconds);
    }
    catch (const ApiError& error) {
        return fail(error);
    }
    // Backends that do not report text while decoding: the content arrives now.
    const std::string& content = out.details.text;
    if (content.size() > sent.size() && content.compare(0, sent.size(), sent) == 0) {
        const std::string_view rest = std::string_view(content).substr(sent.size());
        if (!event([&](JsonWriter& json) { choice(json, false, rest, false); }))
            return false;
    } else if (content != sent) {
        std::cerr << "[chat] streamed text differs from the final transcript\n";
    }
    if (!event([&](JsonWriter& json) {
            choice(json, false, "", true);
            json.key("transcription");
            write_payload(json, request, out.details, out.cues);
        }))
        return false;
    if (request.include_usage && !event([&](JsonWriter& json) {
            json.key("choices").begin_array().end_array();
            write_usage(json, out.details);
        }))
        return false;
    return write("data: [DONE]\n\n");
}

std::string
request_id_default() {
    static std::atomic<uint64_t> counter{1};
#if defined(_WIN32)
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    return "asr_" + std::to_string(pid) + "_" + std::to_string(millis) + "_" +
           std::to_string(counter.fetch_add(1));
}

uint64_t
unix_seconds_default() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// The routes.

struct Api::State {
    Limits limits;
    Backend backend;
    // Shared with the streamed decodes, which may outlive the routes.
    std::shared_ptr<StreamWaker> waker = std::make_shared<StreamWaker>();

    State(Limits l, Backend b) : limits(std::move(l)), backend(std::move(b)) {}

    HttpResponse json(int status, std::string body) const {
        HttpResponse response;
        response.status = status;
        response.headers = {{"Content-Type", "application/json"}};
        response.body = std::move(body);
        return response;
    }

    HttpResponse error(const ApiError& error) const {
        auto response = json(error.status, error_body(error));
        response.headers.emplace_back("Cache-Control", "no-store");
        return response;
    }
};

Api::Api(Limits limits, Backend backend) {
    if (limits.max_request_bytes < 1 || limits.max_audio_seconds < 1)
        throw std::invalid_argument("http.chat limits must be positive");
    if (!backend.request_id)
        backend.request_id = request_id_default;
    if (!backend.unix_seconds)
        backend.unix_seconds = unix_seconds_default;
    state_ = std::make_unique<State>(std::move(limits), std::move(backend));
}

Api::~Api() = default;

HttpResponse
Api::post(const HeaderLookup& headers, const BodyReader& read_body, const ClientGone& gone) {
    State& s = *state_;
    const size_t body_limit = static_cast<size_t>(s.limits.max_request_bytes);
    const ApiError too_large{
        413,
        "payload_too_large",
        "request body exceeds the " + std::to_string(body_limit) + "-byte upload limit",
        {}};
    try {
        std::string value;
        std::optional<uint64_t> declared;
        if (headers("Content-Length", value)) {
            const auto text = std::string(trim(value));
            uint64_t length = 0;
            const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), length);
            if (ec == std::errc() && ptr == text.data() + text.size() && !text.empty())
                declared = length;
        }
        if (declared && *declared > body_limit)
            throw too_large;
        std::string content_type;
        headers("Content-Type", content_type);
        std::string media(trim(content_type.substr(0, content_type.find(';'))));
        std::transform(media.begin(), media.end(), media.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        if (media != "application/json")
            throw ApiError{
                415, "unsupported_media_type", "Content-Type must be application/json", {}};

        std::string body;
        if (declared)
            body.reserve(*declared);
        bool over = false;
        const bool read = read_body([&](const char* data, size_t size) {
            if (body.size() + size > body_limit) {
                over = true;
                return false;
            }
            body.append(data, size);
            return true;
        });
        if (over)
            throw too_large;
        if (!read)
            throw bad_request("invalid request body: error reading a body from connection");

        ValidatedRequest request;
        try {
            const Json root = JsonParser(body).document();
            request = validate_chat(body, root);
        }
        catch (const JsonError& error) {
            throw bad_request("invalid request body: " + error.message);
        }
        std::string().swap(body);

        // Audio: WAV header errors are 400s now; decoding errors wait until the request has passed
        // the model check.
        const auto max_samples = static_cast<size_t>(s.limits.max_audio_seconds) * 16000;
        const bool riff = request.audio.size() >= 12 && request.audio.compare(0, 4, "RIFF") == 0 &&
                          request.audio.compare(8, 4, "WAVE") == 0;
        const bool frontend_decoded = riff && frontend_wav_checks(request.audio, max_samples);
        std::optional<ApiError> deferred;
        auto decoded = decode_samples(request.audio, s.limits.max_audio_seconds, frontend_decoded);
        std::optional<DecodedAudio> audio;
        if (auto* error = std::get_if<ApiError>(&decoded))
            deferred = std::move(*error);
        else if (auto& pcm = std::get<DecodedAudio>(decoded); pcm.samples.empty())
            deferred = ApiError{422, "invalid_request", "audio must be non-empty", {}};
        else
            audio = std::move(pcm);
        if (std::string header; headers("Authorization", header) &&
                                !std::all_of(header.begin(), header.end(), [](unsigned char c) {
                                    return (c >= 32 && c < 127) || c == '\t';
                                }))
            throw bad_request("request header is not UTF-8");
        std::string().swap(request.audio);

        // Text fields: each non-empty and at most 256 KiB (language checked first).
        const std::pair<const std::optional<std::string>*, const char*> texts[] = {
            {&request.language, "language"}, {&request.prompt, "messages"}};
        for (const auto& [text, param] : texts) {
            if (!*text)
                continue;
            const std::string what =
                param == std::string("language") ? "language" : "the text part";
            if ((*text)->empty())
                throw bad_request(what + " must not be empty", param);
            if ((*text)->size() > 262144)
                throw bad_request(what + " exceeds the configured limit of 262144 bytes", param);
        }
        const bool served =
            (!s.limits.model_name.empty() && request.model == s.limits.model_name) ||
            (s.backend.model_alias && request.model == s.backend.model_alias());
        if (!served || !s.backend.recognize)
            throw ApiError{404, "unknown_model", "requested model is not available", {}};
        if (deferred)
            throw *deferred;
        if (!(s.backend.ready && s.backend.ready()))
            throw engine_failure();

        asr::AsrRequestOptions options;
        options.language_code = request.language.value_or("");
        options.enable_word_time_offsets = request.word_timestamps;
        options.verbatim_transcripts = true;
        options.speechlm.task = "transcribe";
        if (request.prompt) {
            // The served instruction: trimmed, with the audio locator appended when absent; a
            // locator anywhere else is refused.
            const std::string custom(trim(*request.prompt));
            // The chat template's turn markers would forge or close a turn; the recognizer refuses
            // every other special token of the tokenizer.
            for (const char* token : {"<|im_start|>", "<|im_end|>", "<think>", "</think>"})
                if (custom.find(token) != std::string::npos)
                    throw bad_request(
                        "the final messages text must not contain chat template control tokens",
                        "messages");
            if (!custom.empty()) {
                const std::string locator = "<|audio|>";
                std::string served =
                    custom.find(locator) == std::string::npos ? custom + " " + locator : custom;
                size_t count = 0;
                for (size_t at = served.find(locator); at != std::string::npos;
                     at = served.find(locator, at + 1))
                    ++count;
                if (count != 1)
                    throw bad_request("<|audio|> locator must appear exactly once");
                const std::string prefix = served.substr(0, served.size() - locator.size());
                if (served.compare(served.size() - locator.size(), locator.size(), locator) != 0 ||
                    prefix.empty() || prefix.back() != ' ' ||
                    (prefix.size() >= 2 && prefix[prefix.size() - 2] == ' ') ||
                    trim(prefix).empty())
                    throw bad_request(
                        "<|audio|> locator must be final and be preceded by exactly one space");
                options.speechlm.instructions = custom;
            }
        }
        // Other models decode greedily from a fixed task; they would ignore both. A seed stays
        // accepted: a greedy decode gives the same transcript for every seed.
        const bool speech_llm =
            s.backend.supports_instructions && s.backend.supports_instructions();
        if (!speech_llm && !options.speechlm.instructions.empty())
            throw bad_request("an instruction needs a speech-LLM model", "messages");
        if (!speech_llm && request.temperature && *request.temperature > 0)
            throw bad_request("a temperature above 0 needs a speech-LLM model", "temperature");
        // Without a timestamp model, timestamp requests are refused with 422.
        if (request.word_timestamps &&
            !(s.backend.supports_timestamps && s.backend.supports_timestamps()))
            throw ApiError{422, "timestamp_unavailable", "word timestamps are unavailable", {}};
        if (request.temperature)
            options.speechlm.temperature = static_cast<float>(*request.temperature);
        if (request.seed) {
            options.speechlm.has_seed = true;
            options.speechlm.seed = static_cast<uint64_t>(*request.seed);
        }

        const double seconds = static_cast<double>(audio->samples.size()) / audio->sample_rate;
        if (request.stream) {
            auto run = std::make_shared<StreamRun>();
            StreamRun* r = run.get();
            r->samples = std::move(audio->samples);
            // The writer checks the client; the decode follows the run's flag.
            options.cancelled = [r] { return r->cancel.load(std::memory_order_relaxed); };
            options.speechlm.on_text = [channel = r->channel,
                                        waker = s.waker](std::string_view piece) {
                waker->report(channel, piece);
            };
            r->worker = std::thread([r, rate = audio->sample_rate, options = std::move(options),
                                     recognize = s.backend.recognize] {
                asr::Result result;
                std::exception_ptr failure;
                try {
                    result = recognize(r->samples.data(), r->samples.size(), rate, options);
                }
                catch (...) {
                    failure = std::current_exception();
                }
                StreamChannel& c = *r->channel;
                {
                    std::lock_guard<std::mutex> lock(c.mutex);
                    r->result = std::move(result);
                    r->failure = failure;
                    c.finished = true;
                }
                c.cv.notify_one();
            });
            // Refusals and failures before any text keep their status: the event stream starts
            // with the first text or the end of the decode (text reported before a failure is
            // streamed, then the error event).
            {
                StreamChannel& c = *r->channel;
                std::unique_lock<std::mutex> lock(c.mutex);
                if (!wait_for_client(
                        lock, c.cv, [&c] { return c.finished || !c.text.empty(); }, gone)) {
                    r->cancel = true;
                    throw client_gone();
                }
                if (c.finished && r->failure && c.text.empty())
                    throw backend_error(r->failure);
            }
            auto response = s.json(200, {});
            response.headers.front().second = "text/event-stream";
            response.headers.emplace_back("Cache-Control", "no-cache");
            response.stream = [run, request = std::move(request), seconds,
                               id = s.backend.request_id(), created = s.backend.unix_seconds(),
                               gone](const StreamWriter& write) {
                return write_stream(*run, request, seconds, id, created, write, gone);
            };
            return response;
        }

        options.cancelled = gone;
        asr::Result result;
        try {
            result = s.backend.recognize(
                audio->samples.data(), audio->samples.size(), audio->sample_rate, options);
        }
        catch (...) {
            throw backend_error(std::current_exception());
        }
        const Completion out = completion(request, result, seconds);
        const asr::SpeechLmDetails* details = &out.details;
        // Subtitles need the alignment: without it the request fails (the transcript is not sent).
        std::optional<std::string> subtitles;
        if (request.subtitle_format) {
            if (details->alignment_status != "aligned")
                throw ApiError{
                    422, "timestamp_unavailable",
                    "word timestamps are unavailable (" +
                        (details->failure_reason.empty() ? std::string("timestamp_unavailable")
                                                         : details->failure_reason) +
                        "), so no subtitles can be rendered",
                    "subtitle_format"};
            subtitles = subtitle::render_word_cues(out.cues, *request.subtitle_format);
        }

        const std::string id = s.backend.request_id();
        const uint64_t created = s.backend.unix_seconds();
        JsonWriter json;
        json.begin_object();
        json.key("id").string("chatcmpl-" + id);
        json.key("object").string("chat.completion");
        json.key("created").integer(static_cast<int64_t>(created));
        json.key("model").string(request.model);
        json.key("choices").begin_array().begin_object();
        json.key("index").integer(0);
        json.key("message").begin_object();
        json.key("role").string("assistant");
        json.key("content").string(details->text);
        json.key("refusal").null();
        json.end_object();
        json.key("finish_reason").string("stop");
        json.key("logprobs").null();
        json.end_object().end_array();
        if (details->prompt_tokens > 0 ||
            details->completion_tokens > 0)  // absent when no decode ran
            write_usage(json, *details);
        json.key("transcription");
        write_payload(json, request, *details, out.cues, subtitles ? &*subtitles : nullptr);
        json.end_object();
        return s.json(200, json.take());
    }
    catch (const ApiError& error) {
        return s.error(error);
    }
    catch (const std::exception& error) {
        std::cerr << "[chat] request failed: " << error.what() << '\n';
        return s.error({500, "server_error", "backend request failed", {}});
    }
}

#endif  // NEMO_SPEECH_REGISTRY_ASR

void
install(httplib::Server& server, EngineRegistry& engines, const ApiConfig& config) {
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    Backend backend;
    backend.recognize = [&engines](
                            const float* samples, size_t n, int rate,
                            const asr::AsrRequestOptions& options) {
        return engines.asr()->recognize(samples, n, options, options.language_code, rate);
    };
    backend.ready = [&engines] {
        try {
            (void)engines.asr();
            return engines.ready();
        }
        catch (const std::exception&) {
            return false;
        }
    };
    backend.supports_timestamps = [&engines] {
        try {
            return engines.asr()->supports_word_timestamps();
        }
        catch (const std::exception&) {
            return false;
        }
    };
    backend.supports_instructions = [&engines] {
        try {
            return engines.asr()->supports_instructions();
        }
        catch (const std::exception&) {
            return false;
        }
    };
    backend.model_alias = [&engines] {
        try {
            return engines.asr()->model_name();
        }
        catch (const std::exception&) {
            return std::string();
        }
    };
    auto api = std::make_shared<Api>(config.limits, std::move(backend));

    const auto send = [](const HttpResponse& from, httplib::Response& to) {
        to.status = from.status;
        std::string content_type;
        for (const auto& [name, value] : from.headers) {
            if (name == "Content-Type")
                content_type = value;
            else
                to.set_header(name, value);
        }
        if (!from.stream) {
            to.set_content(from.body, content_type);
            return;
        }
        to.set_chunked_content_provider(
            content_type, [stream = from.stream](size_t, httplib::DataSink& sink) {
                const bool complete = stream([&sink](std::string_view data) {
                    return sink.write(data.data(), data.size());
                });
                if (complete)
                    sink.done();
                return complete;
            });
    };
    server.Post(
        "/v1/chat/completions", [api, send](
                                    const httplib::Request& request, httplib::Response& response,
                                    const httplib::ContentReader& reader) {
            const HeaderLookup headers = [&](const std::string& name, std::string& value) {
                if (!request.has_header(name))
                    return false;
                value = request.get_header_value(name);
                return true;
            };
            const BodyReader body = [&](const std::function<bool(const char*, size_t)>& sink) {
                return reader([&](const char* data, size_t size) { return sink(data, size); });
            };
            // The request (and its socket) outlives the response's stream.
            const ClientGone gone = [&request] { return request.is_connection_closed(); };
            send(api->post(headers, body, gone), response);
        });
    // Other methods: 405 with an empty body and Allow.
    const auto not_allowed = [](const httplib::Request&, httplib::Response& response) {
        response.status = 405;
        response.set_header("Allow", "POST");
    };
    server.Get("/v1/chat/completions", not_allowed);
    server.Put("/v1/chat/completions", not_allowed);
    server.Patch("/v1/chat/completions", not_allowed);
    server.Delete("/v1/chat/completions", not_allowed);
#else
    (void)server;
    (void)engines;
    (void)config;
#endif
}

}  // namespace nemo_speech::http::chat
