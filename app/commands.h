// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

int command_transcribe(int argc, char** argv);
int command_diarize(int argc, char** argv);
int command_translate(int argc, char** argv);
int command_synthesize(int argc, char** argv);
int command_bench(int argc, char** argv);
int command_model(int argc, char** argv);
int command_doctor(int argc, char** argv);
int command_health(int argc, char** argv);
int command_serve(int argc, char** argv);

void print_transcribe_help(const char* program);
void print_diarize_help(const char* program);
void print_translate_help(const char* program);
void print_synthesize_help(const char* program);
void print_bench_help(const char* program);
void print_model_help(const char* program);
void print_doctor_help(const char* program);
void print_health_help(const char* program);
void print_serve_help(const char* program);
