// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int nemotron_text_create(
    const char* config_json, const char* tokenizer_path, void** handle, char** error);
int nemotron_text_prepare(
    void* handle, const uint8_t* text, size_t text_len, uint8_t apply_normalize, char** output_json,
    char** error);
void nemotron_text_free(char* value);
void nemotron_text_destroy(void* handle);
#ifdef __cplusplus
}
#endif
