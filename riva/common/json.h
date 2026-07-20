// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace nemo_speech::json {

class Value {
   public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool value) : data_(value) {}
    Value(double value) : data_(value) {}
    Value(int value) : data_(static_cast<double>(value)) {}
    Value(std::string value) : data_(std::move(value)) {}
    Value(const char* value) : data_(std::string(value)) {}
    Value(Array value) : data_(std::move(value)) {}
    Value(Object value) : data_(std::move(value)) {}

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    bool boolean() const;
    double number() const;
    const std::string& string() const;
    const Array& array() const;
    Array& array();
    const Object& object() const;
    Object& object();

    const Value* find(const std::string& key) const;
    Value* find(const std::string& key);
    const Value& at(const std::string& key) const;
    Value& operator[](const std::string& key);
    std::string string_or(const std::string& key, const std::string& fallback = "") const;
    double number_or(const std::string& key, double fallback = 0.0) const;
    bool bool_or(const std::string& key, bool fallback = false) const;

    static Value parse(const std::string& input);
    std::string dump(int indent = -1) const;

   private:
    using Data = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    Data data_ = nullptr;
};

}  // namespace nemo_speech::json
