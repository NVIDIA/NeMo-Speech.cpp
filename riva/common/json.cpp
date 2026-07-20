// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "json.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nemo_speech::json {
namespace {

constexpr size_t kMaxNestingDepth = 128;

[[noreturn]] void
type_error(const char* expected) {
    throw std::runtime_error(std::string("JSON value is not ") + expected);
}

void
append_utf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

class Parser {
   public:
    explicit Parser(const std::string& input) : input_(input) {}

    Value parse() {
        Value value = parse_value(0);
        whitespace();
        if (position_ != input_.size())
            fail("unexpected trailing input");
        return value;
    }

   private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(
            "invalid JSON at byte " + std::to_string(position_) + ": " + message);
    }

    void whitespace() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\n' ||
                input_[position_] == '\r' || input_[position_] == '\t'))
            ++position_;
    }

    bool take(char c) {
        whitespace();
        if (position_ < input_.size() && input_[position_] == c) {
            ++position_;
            return true;
        }
        return false;
    }

    Value parse_value(size_t depth) {
        whitespace();
        if (position_ >= input_.size())
            fail("expected a value");
        switch (input_[position_]) {
            case 'n':
                literal("null");
                return nullptr;
            case 't':
                literal("true");
                return true;
            case 'f':
                literal("false");
                return false;
            case '"':
                return parse_string();
            case '[':
                check_container_depth(depth);
                return parse_array(depth);
            case '{':
                check_container_depth(depth);
                return parse_object(depth);
            default:
                return parse_number();
        }
    }

    void check_container_depth(size_t depth) const {
        if (depth >= kMaxNestingDepth)
            fail("maximum nesting depth exceeded");
    }

    void literal(const char* value) {
        const std::string expected(value);
        if (input_.compare(position_, expected.size(), expected) != 0)
            fail("unexpected token");
        position_ += expected.size();
    }

    uint32_t hex4() {
        if (position_ + 4 > input_.size())
            fail("incomplete Unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9')
                value |= c - '0';
            else if (c >= 'a' && c <= 'f')
                value |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                value |= c - 'A' + 10;
            else
                fail("invalid Unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        if (!take('"'))
            fail("expected a string");
        std::string output;
        while (position_ < input_.size()) {
            const unsigned char c = input_[position_++];
            if (c == '"')
                return output;
            if (c < 0x20)
                fail("control character in string");
            if (c != '\\') {
                output.push_back(static_cast<char>(c));
                continue;
            }
            if (position_ >= input_.size())
                fail("incomplete escape");
            switch (input_[position_++]) {
                case '"':
                    output.push_back('"');
                    break;
                case '\\':
                    output.push_back('\\');
                    break;
                case '/':
                    output.push_back('/');
                    break;
                case 'b':
                    output.push_back('\b');
                    break;
                case 'f':
                    output.push_back('\f');
                    break;
                case 'n':
                    output.push_back('\n');
                    break;
                case 'r':
                    output.push_back('\r');
                    break;
                case 't':
                    output.push_back('\t');
                    break;
                case 'u': {
                    uint32_t codepoint = hex4();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                            input_[position_ + 1] != 'u')
                            fail("high surrogate without low surrogate");
                        position_ += 2;
                        const uint32_t low = hex4();
                        if (low < 0xdc00 || low > 0xdfff)
                            fail("invalid low surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unexpected low surrogate");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default:
                    fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    Value parse_number() {
        whitespace();
        const size_t begin = position_;
        if (position_ < input_.size() && input_[position_] == '-')
            ++position_;
        if (position_ >= input_.size())
            fail("invalid number");
        if (input_[position_] == '0') {
            ++position_;
        } else {
            if (input_[position_] < '1' || input_[position_] > '9')
                fail("invalid number");
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9')
                ++position_;
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const size_t digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9')
                ++position_;
            if (digits == position_)
                fail("invalid fractional number");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-'))
                ++position_;
            const size_t digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9')
                ++position_;
            if (digits == position_)
                fail("invalid exponent");
        }
        try {
            const double value = std::stod(input_.substr(begin, position_ - begin));
            if (!std::isfinite(value))
                fail("number is out of range");
            return value;
        }
        catch (const std::exception&) {
            fail("invalid number");
        }
    }

    Value parse_array(size_t depth) {
        take('[');
        Value::Array output;
        if (take(']'))
            return output;
        do {
            output.push_back(parse_value(depth + 1));
        } while (take(','));
        if (!take(']'))
            fail("expected ']' in array");
        return output;
    }

    Value parse_object(size_t depth) {
        take('{');
        Value::Object output;
        if (take('}'))
            return output;
        do {
            whitespace();
            if (position_ >= input_.size() || input_[position_] != '"')
                fail("expected object key");
            std::string key = parse_string();
            if (!take(':'))
                fail("expected ':' after object key");
            if (!output.emplace(std::move(key), parse_value(depth + 1)).second)
                fail("duplicate object key");
        } while (take(','));
        if (!take('}'))
            fail("expected '}' in object");
        return output;
    }

    const std::string& input_;
    size_t position_ = 0;
};

std::string
escape(const std::string& input) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char c : input) {
        switch (c) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (c < 0x20)
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(c) << std::dec;
                else
                    output << static_cast<char>(c);
        }
    }
    output << '"';
    return output.str();
}

void
dump_value(const Value& value, std::ostringstream& output, int indent, size_t depth) {
    if ((value.is_array() || value.is_object()) && depth >= kMaxNestingDepth)
        throw std::runtime_error("JSON maximum nesting depth exceeded");
    auto newline = [&](size_t extra = 0) {
        if (indent >= 0)
            output << '\n' << std::string((depth + extra) * static_cast<size_t>(indent), ' ');
    };
    if (value.is_null())
        output << "null";
    else if (value.is_bool())
        output << (value.boolean() ? "true" : "false");
    else if (value.is_number())
        output << std::setprecision(17) << value.number();
    else if (value.is_string())
        output << escape(value.string());
    else if (value.is_array()) {
        output << '[';
        const auto& values = value.array();
        for (size_t i = 0; i < values.size(); ++i) {
            if (i)
                output << ',';
            newline(1);
            dump_value(values[i], output, indent, depth + 1);
        }
        if (!values.empty())
            newline();
        output << ']';
    } else {
        output << '{';
        const auto& values = value.object();
        size_t i = 0;
        for (const auto& [key, child] : values) {
            if (i++)
                output << ',';
            newline(1);
            output << escape(key) << (indent >= 0 ? ": " : ":");
            dump_value(child, output, indent, depth + 1);
        }
        if (!values.empty())
            newline();
        output << '}';
    }
}

}  // namespace

bool
Value::is_null() const {
    return std::holds_alternative<std::nullptr_t>(data_);
}
bool
Value::is_bool() const {
    return std::holds_alternative<bool>(data_);
}
bool
Value::is_number() const {
    return std::holds_alternative<double>(data_);
}
bool
Value::is_string() const {
    return std::holds_alternative<std::string>(data_);
}
bool
Value::is_array() const {
    return std::holds_alternative<Array>(data_);
}
bool
Value::is_object() const {
    return std::holds_alternative<Object>(data_);
}
bool
Value::boolean() const {
    if (!is_bool())
        type_error("a boolean");
    return std::get<bool>(data_);
}
double
Value::number() const {
    if (!is_number())
        type_error("a number");
    return std::get<double>(data_);
}
const std::string&
Value::string() const {
    if (!is_string())
        type_error("a string");
    return std::get<std::string>(data_);
}
const Value::Array&
Value::array() const {
    if (!is_array())
        type_error("an array");
    return std::get<Array>(data_);
}
Value::Array&
Value::array() {
    if (!is_array())
        type_error("an array");
    return std::get<Array>(data_);
}
const Value::Object&
Value::object() const {
    if (!is_object())
        type_error("an object");
    return std::get<Object>(data_);
}
Value::Object&
Value::object() {
    if (!is_object())
        type_error("an object");
    return std::get<Object>(data_);
}

const Value*
Value::find(const std::string& key) const {
    if (!is_object())
        return nullptr;
    const auto found = object().find(key);
    return found == object().end() ? nullptr : &found->second;
}
Value*
Value::find(const std::string& key) {
    if (!is_object())
        return nullptr;
    const auto found = object().find(key);
    return found == object().end() ? nullptr : &found->second;
}
const Value&
Value::at(const std::string& key) const {
    const auto* value = find(key);
    if (!value)
        throw std::runtime_error("JSON object is missing key '" + key + "'");
    return *value;
}
Value&
Value::operator[](const std::string& key) {
    if (is_null())
        data_ = Object{};
    if (!is_object())
        type_error("an object");
    return object()[key];
}
std::string
Value::string_or(const std::string& key, const std::string& fallback) const {
    const auto* value = find(key);
    return value && value->is_string() ? value->string() : fallback;
}
double
Value::number_or(const std::string& key, double fallback) const {
    const auto* value = find(key);
    return value && value->is_number() ? value->number() : fallback;
}
bool
Value::bool_or(const std::string& key, bool fallback) const {
    const auto* value = find(key);
    return value && value->is_bool() ? value->boolean() : fallback;
}

Value
Value::parse(const std::string& input) {
    return Parser(input).parse();
}
std::string
Value::dump(int indent) const {
    std::ostringstream output;
    dump_value(*this, output, indent, 0);
    return output.str();
}

}  // namespace nemo_speech::json
