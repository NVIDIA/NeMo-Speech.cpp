/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "processor_abi.h"

namespace {

NemotronSpeechByteSpanV1
Span(const std::string& value)
{
  return {reinterpret_cast<const uint8_t*>(value.data()), value.size()};
}

NemotronSpeechRecordViewV1
Utf8(const std::string& name, const std::string& value)
{
  NemotronSpeechRecordViewV1 record{};
  record.struct_size = sizeof(record);
  record.name = Span(name);
  record.kind = NEMOTRON_SPEECH_RECORD_UTF8_V1;
  record.element_type = NEMOTRON_SPEECH_ELEMENT_U8_V1;
  record.data = Span(value);
  return record;
}

NemotronSpeechRecordViewV1
Boolean(const std::string& name, const std::string& value)
{
  NemotronSpeechRecordViewV1 record{};
  record.struct_size = sizeof(record);
  record.name = Span(name);
  record.kind = NEMOTRON_SPEECH_RECORD_BOOLEAN_V1;
  record.element_type = NEMOTRON_SPEECH_ELEMENT_U8_V1;
  record.data = Span(value);
  return record;
}

TEST(TextPreprocessorC, NegotiatesAbiAndProcessesText)
{
  EXPECT_EQ(nemotron_speech_processor_get_api(999), nullptr);
  const auto* api =
      nemotron_speech_processor_get_api(NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION);
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->abi_version, NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION);

  NemotronSpeechProcessorCapabilitiesV1 capabilities{};
  capabilities.struct_size = sizeof(capabilities);
  auto status = api->capabilities(&capabilities);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);
  EXPECT_EQ(capabilities.thread_safe, 0);
  EXPECT_EQ(capabilities.streaming_input, 1);
  EXPECT_EQ(capabilities.stateful, 0);

  const std::string language_name = "language_code";
  const std::string language = "en-US";
  const auto config_record = Utf8(language_name, language);
  NemotronSpeechEnvelopeViewV1 config{};
  config.struct_size = sizeof(config);
  config.schema_version = 1;
  config.records = &config_record;
  config.record_count = 1;
  config.is_final = 1;

  NemotronSpeechProcessorHandleV1 handle = nullptr;
  status = api->create(&config, &handle);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);
  ASSERT_NE(handle, nullptr);

  const std::string text_name = "text";
  const std::string text = "Hello world";
  const auto text_record = Utf8(text_name, text);
  const auto language_record = Utf8(language_name, language);
  const NemotronSpeechRecordViewV1 request_records[] = {text_record, language_record};
  const std::string request_id = "request-1";
  NemotronSpeechEnvelopeViewV1 request{};
  request.struct_size = sizeof(request);
  request.schema_version = 1;
  request.request_id = Span(request_id);
  request.records = request_records;
  request.record_count = 2;
  request.is_final = 1;

  NemotronSpeechEnvelopeOutputV1 output{};
  output.struct_size = sizeof(output);
  status = api->process(handle, &request, &output);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);
  ASSERT_EQ(output.record_count, 1);
  EXPECT_EQ(
      std::string(
          reinterpret_cast<const char*>(output.records[0].data.data), output.records[0].data.len),
      "Hello world.");
  EXPECT_EQ(output.records[0].is_final, 1);

  api->release_output(handle, &output);
  api->destroy(handle);
}

TEST(TextPreprocessorC, ContinuousRequestReturnsOneUnpunctuatedCanonicalSegment)
{
  const auto* api =
      nemotron_speech_processor_get_api(NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION);
  ASSERT_NE(api, nullptr);

  const std::string language_name = "language_code";
  const std::string language = "en-US";
  const auto config_record = Utf8(language_name, language);
  NemotronSpeechEnvelopeViewV1 config{};
  config.struct_size = sizeof(config);
  config.schema_version = 1;
  config.records = &config_record;
  config.record_count = 1;

  NemotronSpeechProcessorHandleV1 handle = nullptr;
  auto status = api->create(&config, &handle);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);

  const std::string text_name = "text";
  const std::string text = " \tFirst unit.  Second unit \n";
  const std::string continuous_name = "continuous_streaming";
  const std::string continuous_value(1, '\1');
  const NemotronSpeechRecordViewV1 request_records[] = {
      Utf8(text_name, text), Utf8(language_name, language),
      Boolean(continuous_name, continuous_value)};
  NemotronSpeechEnvelopeViewV1 request{};
  request.struct_size = sizeof(request);
  request.schema_version = 1;
  request.records = request_records;
  request.record_count = 3;

  NemotronSpeechEnvelopeOutputV1 output{};
  output.struct_size = sizeof(output);
  status = api->process(handle, &request, &output);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);
  ASSERT_EQ(output.record_count, 1);
  EXPECT_EQ(
      std::string(
          reinterpret_cast<const char*>(output.records[0].data.data), output.records[0].data.len),
      " First unit. Second unit ");
  EXPECT_EQ(output.records[0].sequence, 0);
  EXPECT_EQ(output.records[0].is_final, 1);

  api->release_output(handle, &output);
  api->destroy(handle);
}

TEST(TextPreprocessorC, DirectGrammarConfigurationLoadsFarPair)
{
  const auto* api =
      nemotron_speech_processor_get_api(NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION);
  ASSERT_NE(api, nullptr);

  const std::string language_name = "language_code";
  const std::string language = "en-US";
  const std::string path_name = "normalizer_path";
  const std::string path = "testdata/en-US";
  const std::string direct_name = "direct_grammar";
  const std::string true_value(1, '\1');
  const NemotronSpeechRecordViewV1 config_records[] = {
      Utf8(language_name, language), Utf8(path_name, path), Boolean(direct_name, true_value)};
  NemotronSpeechEnvelopeViewV1 config{};
  config.struct_size = sizeof(config);
  config.schema_version = 1;
  config.records = config_records;
  config.record_count = 3;

  NemotronSpeechProcessorHandleV1 handle = nullptr;
  auto status = api->create(&config, &handle);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);

  const std::string text_name = "text";
  const std::string text = "I spent $2 today";
  const std::string continuous_name = "continuous_streaming";
  const NemotronSpeechRecordViewV1 request_records[] = {
      Utf8(text_name, text), Utf8(language_name, language),
      Boolean(continuous_name, true_value)};
  NemotronSpeechEnvelopeViewV1 request{};
  request.struct_size = sizeof(request);
  request.schema_version = 1;
  request.records = request_records;
  request.record_count = 3;

  NemotronSpeechEnvelopeOutputV1 output{};
  output.struct_size = sizeof(output);
  status = api->process(handle, &request, &output);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);
  ASSERT_EQ(output.record_count, 1);
  EXPECT_EQ(
      std::string(
          reinterpret_cast<const char*>(output.records[0].data.data), output.records[0].data.len),
      "I spent two dollars today");

  api->release_output(handle, &output);
  api->destroy(handle);
}

TEST(TextPreprocessorC, ConvertsExceptionsToStatus)
{
  const auto* api =
      nemotron_speech_processor_get_api(NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION);
  const std::string language_name = "language_code";
  const std::string language = "en-US";
  const auto config_record = Utf8(language_name, language);
  NemotronSpeechEnvelopeViewV1 config{};
  config.struct_size = sizeof(config);
  config.schema_version = 1;
  config.records = &config_record;
  config.record_count = 1;

  NemotronSpeechProcessorHandleV1 handle = nullptr;
  auto status = api->create(&config, &handle);
  ASSERT_EQ(status.code, 0);
  api->release_status(&status);

  const std::string text_name = "text";
  const std::string empty;
  const auto text_record = Utf8(text_name, empty);
  const auto language_record = Utf8(language_name, language);
  const NemotronSpeechRecordViewV1 records[] = {text_record, language_record};
  NemotronSpeechEnvelopeViewV1 request{};
  request.struct_size = sizeof(request);
  request.schema_version = 1;
  request.records = records;
  request.record_count = 2;
  NemotronSpeechEnvelopeOutputV1 output{};
  output.struct_size = sizeof(output);
  status = api->process(handle, &request, &output);
  EXPECT_NE(status.code, 0);
  EXPECT_GT(status.message.len, 0);
  api->release_status(&status);
  api->destroy(handle);
}

}  // namespace
