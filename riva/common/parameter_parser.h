// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Generic, domain-agnostic configuration parser. Any pipeline (ASR, TTS, S2S,
// NMT) describes its config by giving each config struct a method
//
//   void Register(common::ParameterParser& p);
//
// that registers its scalar fields and its nested child configs - all with the
// SAME Register() call. The parser tracks the dotted-key prefix as it recurses,
// so leaves never repeat it and children never concatenate it; dotted keys
// (e.g. "asr.vad.masker.onset") fall out of the nesting automatically. The
// parser has no knowledge of any pipeline. One registered table is then driven
// by CLI flags, env vars, and a YAML file:
//
//   MyConfig cfg;                         // defaults
//   common::ParameterParser p;
//   p.Register("asr", cfg);               // whole tree under "asr."
//   p.ApplyYaml(path);                    // file overrides defaults
//   p.ApplyEnv();                         // env  overrides file
//   for (argv) p.ParseCliArg(arg, ...);   // CLI  overrides env
//
// Precedence is the caller's apply order (defaults < file < env < CLI above).
#pragma once

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nemo_speech::common {

// String -> value parsers (throw std::invalid_argument for invalid user input).
namespace detail {
int to_int(const std::string& key, const std::string& v);
double to_double(const std::string& key, const std::string& v);
float to_float(const std::string& key, const std::string& v);
bool to_bool(const std::string& key, const std::string& v);
}  // namespace detail

class ParameterParser {
   public:
    // A scalar leaf field, by pointer. T in {bool, int, float, double,
    // std::string}. The key is <current prefix> + name; `aliases` are full
    // legacy flag strings (e.g. "--vad-onset"). bool fields get bare-flag CLI
    // semantics ("--flag" = true, "--flag=false" to negate).
    template <class T>
    void Register(
        const std::string& name, T* field, const std::string& doc,
        std::vector<std::string> aliases = {}) {
        const std::string key = prefix_ + name;
        Add(
            key, std::move(aliases), std::is_same_v<std::remove_cv_t<T>, bool>,
            [field, key](const std::string& v) { Assign(field, key, v); }, doc);
    }

    // A leaf with a custom setter (enum parsing, one flag driving several
    // fields, side effects). `is_bool` gives bare-flag CLI semantics.
    void Register(
        const std::string& name, std::function<void(const std::string&)> set,
        const std::string& doc, std::vector<std::string> aliases = {}, bool is_bool = false) {
        Add(prefix_ + name, std::move(aliases), is_bool, std::move(set), doc);
    }

    // A nested child config: anything exposing Register(ParameterParser&).
    // Recurses under "<name>." so the child's keys nest - same verb as a leaf.
    // (The trailing template arg SFINAE-restricts this to config structs, so a
    // field-pointer leaf can never accidentally match here.)
    template <
        class C, class = decltype(std::declval<C&>().Register(std::declval<ParameterParser&>()))>
    void Register(const std::string& name, C& child) {
        const std::size_t base = prefix_.size();
        prefix_ += name;
        prefix_ += '.';
        child.Register(*this);
        prefix_.resize(base);
    }

    // Apply one CLI token: a dotted form ("--asr.x.y=v" / "--asr.x.y v") or a
    // registered alias ("--vad-onset 0.5", "--endpointing"). Pulls a value from
    // `next` (argv[i+1]) when the token has no '=' and the field needs one,
    // setting *consumed_next. Returns false if `arg` matches no key/alias.
    bool ParseCliArg(const std::string& arg, const char* next, bool* consumed_next) const;

    // Apply <env_prefix>_<KEY> environment variables (dotted key uppercased,
    // '.' and '-' -> '_') for any registered field present in the environment.
    void ApplyEnv(const std::string& env_prefix = "NEMO_SPEECH") const;

    // Load a YAML file whose nested maps mirror the dotted keys; set each scalar
    // leaf. Throws on a parse error, an unknown key, or a bad value.
    void ApplyYaml(const std::string& path) const;

    // Human-readable list of keys + aliases for --help.
    std::string Help() const;

   private:
    struct Entry {
        std::string key;
        std::vector<std::string> aliases;
        bool is_bool;  // bare-flag CLI semantics
        std::function<void(const std::string&)> set;
        std::string doc;
    };
    void Add(
        std::string key, std::vector<std::string> aliases, bool is_bool,
        std::function<void(const std::string&)> set, std::string doc);
    const Entry* ByKey(const std::string& k) const;
    const Entry* ByAlias(const std::string& a) const;

    // Typed assignment used by the leaf Register (kept thin so the parsers live
    // in the .cpp). Defined for the supported scalar types.
    template <class T>
    static void Assign(T* field, const std::string& key, const std::string& v) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, bool>)
            *field = detail::to_bool(key, v);
        else if constexpr (std::is_same_v<std::remove_cv_t<T>, int>)
            *field = detail::to_int(key, v);
        else if constexpr (std::is_same_v<std::remove_cv_t<T>, float>)
            *field = detail::to_float(key, v);
        else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>)
            *field = detail::to_double(key, v);
        else
            *field = v;  // std::string
    }

    std::string prefix_;  // current dotted prefix during the recursive Register walk
    std::vector<Entry> entries_;
};

}  // namespace nemo_speech::common
