// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "parameter_parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace nemo_speech::common {

namespace detail {
namespace {
[[noreturn]] void
bad_value(const std::string& key, const std::string& v, const char* type) {
    throw std::invalid_argument("config " + key + ": invalid " + type + " value '" + v + "'");
}
}  // namespace

int
to_int(const std::string& key, const std::string& v) {
    try {
        size_t p = 0;
        const int x = std::stoi(v, &p);
        if (p != v.size())
            bad_value(key, v, "int");
        return x;
    }
    catch (const std::exception&) {
        bad_value(key, v, "int");
    }
}

double
to_double(const std::string& key, const std::string& v) {
    try {
        size_t p = 0;
        const double x = std::stod(v, &p);
        if (p != v.size() || !std::isfinite(x))
            bad_value(key, v, "number");
        return x;
    }
    catch (const std::exception&) {
        bad_value(key, v, "number");
    }
}

float
to_float(const std::string& key, const std::string& v) {
    return static_cast<float>(to_double(key, v));
}

bool
to_bool(const std::string& key, const std::string& v) {
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "1" || s == "true" || s == "yes" || s == "on")
        return true;
    if (s == "0" || s == "false" || s == "no" || s == "off")
        return false;
    bad_value(key, v, "bool");
}
}  // namespace detail

namespace {
std::string
trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

std::string
env_name(const std::string& env_prefix, const std::string& dotted_key) {
    std::string e = env_prefix + "_" + dotted_key;
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) {
        return (c == '.' || c == '-') ? '_' : static_cast<char>(std::toupper(c));
    });
    return e;
}
}  // namespace

void
ParameterParser::Add(
    std::string key, std::vector<std::string> aliases, bool is_bool,
    std::function<void(const std::string&)> set, std::string doc) {
    entries_.push_back(
        {std::move(key), std::move(aliases), is_bool, std::move(set), std::move(doc)});
}

const ParameterParser::Entry*
ParameterParser::ByKey(const std::string& k) const {
    for (const auto& e : entries_)
        if (e.key == k)
            return &e;
    return nullptr;
}

const ParameterParser::Entry*
ParameterParser::ByAlias(const std::string& a) const {
    for (const auto& e : entries_)
        for (const auto& al : e.aliases)
            if (al == a)
                return &e;
    return nullptr;
}

bool
ParameterParser::ParseCliArg(const std::string& arg, const char* next, bool* consumed_next) const {
    *consumed_next = false;
    std::string key = arg, val;
    bool has_eq = false;
    if (const auto pos = arg.find('='); pos != std::string::npos) {
        key = arg.substr(0, pos);
        val = arg.substr(pos + 1);
        has_eq = true;
    }
    const Entry* e = ByAlias(key);
    if (!e) {
        // Dotted form: strip leading dashes ("--asr.x.y" -> "asr.x.y").
        std::string k = key;
        while (!k.empty() && k.front() == '-') k.erase(k.begin());
        e = ByKey(k);
    }
    if (!e)
        return false;
    if (e->is_bool) {
        e->set(has_eq ? val : "true");  // bare flag = true
        return true;
    }
    if (!has_eq) {
        if (!next)
            throw std::invalid_argument("config " + e->key + ": missing value");
        val = next;
        *consumed_next = true;
    }
    e->set(val);
    return true;
}

void
ParameterParser::ApplyEnv(const std::string& env_prefix) const {
    for (const auto& e : entries_) {
        if (const char* v = std::getenv(env_name(env_prefix, e.key).c_str()))
            e.set(v);
    }
}

void
ParameterParser::ApplyYaml(const std::string& path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::invalid_argument("config: cannot open file '" + path + "'");

    // Minimal nested-config parser for the exact subset our config files use:
    // indentation (spaces) defines nested maps, `key: value` sets a scalar leaf,
    // `key:` opens a section, `#` starts a comment, and a value may be quoted.
    // The dotted keys it produces (e.g. asr.vad.masker.onset) mirror the CLI
    // flags one-to-one. Anything outside the subset - tabs, sequences, flow
    // `[`/`{`, anchors, multi-line scalars - is rejected. Leaf keys must be
    // registered (unknown key = error); a key with no value (a commented-out
    // default, or a section whose keys are all commented out) is skipped, so
    // documentation configs that comment out whole blocks load cleanly.
    struct Frame {
        int indent;
        std::string key;
    };
    std::vector<Frame> stack;  // open sections; dotted prefix = join(keys, '.')
    std::string line;
    int lineno = 0;
    auto fail = [&](const std::string& msg) {
        throw std::invalid_argument(
            "config '" + path + "' line " + std::to_string(lineno) + ": " + msg);
    };

    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();  // tolerate CRLF

        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') ++indent;
        if (indent < line.size() && line[indent] == '\t')
            fail("tab in indentation (use spaces)");

        const std::string content = line.substr(indent);
        if (content.empty() || content[0] == '#')
            continue;  // blank line or comment
        if (content == "---" || content == "...")
            continue;  // YAML document markers, no data
        if (content[0] == '-' && (content.size() == 1 || content[1] == ' '))
            fail("sequences are not supported in config");

        const size_t colon = content.find(':');
        if (colon == std::string::npos)
            fail("expected 'key: value', got '" + content + "'");
        const std::string key = trim(content.substr(0, colon));
        if (key.empty())
            fail("empty key");

        // This line nests under whatever sections remain once we drop any at the
        // same indent or deeper (siblings + closed children).
        while (!stack.empty() && stack.back().indent >= static_cast<int>(indent)) stack.pop_back();

        // Value after the colon: skip leading space, honor a quoted string, drop
        // a trailing ` #` inline comment. Empty (or comment-only) => no value.
        std::string rest = content.substr(colon + 1);
        const size_t vb = rest.find_first_not_of(" \t");
        std::string val;
        if (vb != std::string::npos && rest[vb] != '#') {
            rest = rest.substr(vb);
            if (rest[0] == '"' || rest[0] == '\'') {
                const char q = rest[0];
                const size_t close = rest.find(q, 1);
                if (close == std::string::npos)
                    fail("unterminated quoted value");
                val = rest.substr(1, close - 1);
            } else {
                const size_t c = rest.find(" #");  // inline comment
                val = trim(c == std::string::npos ? rest : rest.substr(0, c));
            }
        }

        if (val.empty()) {
            // A section (its leaves follow at deeper indent) or a commented-out
            // leaf - pushed, then popped unused if nothing nests under it.
            stack.push_back({static_cast<int>(indent), key});
            continue;
        }

        std::string dotted;
        for (const auto& f : stack) {
            dotted += f.key;
            dotted += '.';
        }
        dotted += key;
        const Entry* e = ByKey(dotted);
        if (!e)
            fail("unknown key '" + dotted + "'");
        e->set(val);
    }
}

std::string
ParameterParser::Help() const {
    std::ostringstream os;
    for (const auto& e : entries_) {
        os << "  --" << e.key;
        for (const auto& a : e.aliases) os << " | " << a;
        os << "\n      " << e.doc << "\n";
    }
    return os.str();
}

}  // namespace nemo_speech::common
