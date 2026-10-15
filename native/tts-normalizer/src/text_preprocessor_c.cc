/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "processor_abi.h"
#include "text_preprocessor.h"

namespace {

using nemotron_speech::tts::preprocessing::LanguageConfig;
using nemotron_speech::tts::preprocessing::PreprocessOptions;
using nemotron_speech::tts::preprocessing::PreprocessorErrorCode;
using nemotron_speech::tts::preprocessing::TextPreprocessor;
using nemotron_speech::tts::preprocessing::TextPreprocessorConfig;
using nemotron_speech::tts::preprocessing::TextPreprocessorException;

constexpr uint32_t kSchemaVersion = 1;
constexpr char kProcessorId[] = "nemotron_speech.tts.text_preprocessor";

struct ProcessorState {
  explicit ProcessorState(TextPreprocessorConfig config) : processor(std::move(config)) {}
  TextPreprocessor processor;
};

struct OutputStorage {
  std::string request_id;
  std::string session_id;
  std::vector<std::string> names;
  std::vector<std::string> data;
  std::vector<NemotronSpeechRecordViewV1> views;
};

NemotronSpeechByteSpanV1
Span(const std::string& value)
{
  return {reinterpret_cast<const uint8_t*>(value.data()), value.size()};
}

NemotronSpeechByteSpanV1
Span(const char* value)
{
  return {reinterpret_cast<const uint8_t*>(value), std::strlen(value)};
}

NemotronSpeechStatusV1
Status(int32_t code, const std::string& message = "") noexcept
{
  NemotronSpeechStatusV1 status{
      sizeof(NemotronSpeechStatusV1), code, {nullptr, 0}, nullptr};
  try {
    if (!message.empty()) {
      auto owned = std::make_unique<std::string>(message);
      status.message = Span(*owned);
      status.context = owned.release();
    }
  }
  catch (...) {
    status.message = Span("native processor error (message unavailable)");
  }
  return status;
}

void
ReleaseStatus(NemotronSpeechStatusV1* status)
{
  if (status == nullptr) {
    return;
  }
  delete static_cast<std::string*>(status->context);
  *status = {sizeof(NemotronSpeechStatusV1), 0, {nullptr, 0}, nullptr};
}

std::string
CopySpan(NemotronSpeechByteSpanV1 value, const char* field)
{
  if (value.len != 0 && value.data == nullptr) {
    throw std::invalid_argument(std::string(field) + " pointer is null");
  }
  if (value.len == 0) {
    return "";
  }
  return {reinterpret_cast<const char*>(value.data), value.len};
}

const NemotronSpeechRecordViewV1*
FindRecord(const NemotronSpeechEnvelopeViewV1& envelope, const std::string& name)
{
  if (envelope.record_count != 0 && envelope.records == nullptr) {
    throw std::invalid_argument("record array is null");
  }
  for (size_t index = 0; index < envelope.record_count; ++index) {
    const auto& record = envelope.records[index];
    if (record.struct_size < sizeof(NemotronSpeechRecordViewV1)) {
      throw std::invalid_argument("record structure is too small");
    }
    if (CopySpan(record.name, "record name") == name) {
      return &record;
    }
  }
  return nullptr;
}

std::string
Utf8Record(
    const NemotronSpeechEnvelopeViewV1& envelope, const std::string& name,
    bool required = true)
{
  const auto* record = FindRecord(envelope, name);
  if (record == nullptr) {
    if (required) {
      throw std::invalid_argument("missing UTF-8 record: " + name);
    }
    return "";
  }
  if (record->kind != NEMOTRON_SPEECH_RECORD_UTF8_V1) {
    throw std::invalid_argument("record is not UTF-8: " + name);
  }
  return CopySpan(record->data, name.c_str());
}

bool
BooleanRecord(
    const NemotronSpeechEnvelopeViewV1& envelope, const std::string& name,
    bool default_value)
{
  const auto* record = FindRecord(envelope, name);
  if (record == nullptr) {
    return default_value;
  }
  if (record->kind != NEMOTRON_SPEECH_RECORD_BOOLEAN_V1 || record->data.len != 1 ||
      record->data.data == nullptr) {
    throw std::invalid_argument("record is not a boolean: " + name);
  }
  return record->data.data[0] != 0;
}

size_t
IntegerRecord(
    const NemotronSpeechEnvelopeViewV1& envelope, const std::string& name,
    size_t default_value)
{
  const auto* record = FindRecord(envelope, name);
  if (record == nullptr) {
    return default_value;
  }
  if (record->kind != NEMOTRON_SPEECH_RECORD_INTEGER_V1 ||
      record->data.len != sizeof(int64_t) ||
      record->data.data == nullptr) {
    throw std::invalid_argument("record is not an integer: " + name);
  }
  int64_t value = 0;
  std::memcpy(&value, record->data.data, sizeof(value));
  if (value <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return static_cast<size_t>(value);
}

NemotronSpeechStatusV1
Capabilities(NemotronSpeechProcessorCapabilitiesV1* output)
{
  if (output == nullptr ||
      output->struct_size < sizeof(NemotronSpeechProcessorCapabilitiesV1)) {
    return Status(1, "capability output structure is missing or too small");
  }
  output->processor_id = Span(kProcessorId);
  output->batch_input = 0;
  output->streaming_input = 1;
  output->thread_safe = 0;
  output->stateful = 0;
  return Status(0);
}

NemotronSpeechStatusV1
Create(
    const NemotronSpeechEnvelopeViewV1* config,
    NemotronSpeechProcessorHandleV1* output)
{
  if (config == nullptr ||
      config->struct_size < sizeof(NemotronSpeechEnvelopeViewV1) || output == nullptr) {
    return Status(1, "processor configuration is missing or too small");
  }
  *output = nullptr;
  try {
    if (config->schema_version != kSchemaVersion) {
      return Status(1, "unsupported configuration schema version");
    }
    const auto language_code = Utf8Record(*config, "language_code");
    LanguageConfig language;
    language.normalizer_path = Utf8Record(*config, "normalizer_path", false);
    const auto normalizer_config = Utf8Record(*config, "normalizer_config", false);
    if (!normalizer_config.empty()) {
      language.normalizer_config = normalizer_config;
    }
    language.abbreviations_path = Utf8Record(*config, "abbreviations_path", false);
    language.append_punctuation = BooleanRecord(*config, "append_punctuation", true);
    language.direct_grammar = BooleanRecord(*config, "direct_grammar", false);

    TextPreprocessorConfig processor_config;
    processor_config.languages.emplace(language_code, std::move(language));
    processor_config.max_input_bytes =
        IntegerRecord(*config, "max_input_bytes", processor_config.max_input_bytes);
    processor_config.max_normalized_bytes =
        IntegerRecord(*config, "max_normalized_bytes", processor_config.max_normalized_bytes);
    processor_config.max_segments =
        IntegerRecord(*config, "max_segments", processor_config.max_segments);
    *output = new ProcessorState(std::move(processor_config));
    return Status(0);
  }
  catch (const TextPreprocessorException& error) {
    return Status(100 + static_cast<int32_t>(error.code()), error.what());
  }
  catch (const std::exception& error) {
    return Status(1, error.what());
  }
  catch (...) {
    return Status(1, "unknown exception while creating text processor");
  }
}

void
Destroy(NemotronSpeechProcessorHandleV1 handle)
{
  delete static_cast<ProcessorState*>(handle);
}

NemotronSpeechStatusV1
Process(
    NemotronSpeechProcessorHandleV1 handle, const NemotronSpeechEnvelopeViewV1* request,
    NemotronSpeechEnvelopeOutputV1* output)
{
  if (handle == nullptr || request == nullptr || output == nullptr ||
      request->struct_size < sizeof(NemotronSpeechEnvelopeViewV1) ||
      output->struct_size < sizeof(NemotronSpeechEnvelopeOutputV1)) {
    return Status(1, "processor handle, request, or output is invalid");
  }
  if (output->context != nullptr) {
    return Status(1, "output must be released before reuse");
  }
  try {
    if (request->schema_version != kSchemaVersion) {
      return Status(1, "unsupported request schema version");
    }
    const auto text = Utf8Record(*request, "text");
    const auto language_code = Utf8Record(*request, "language_code");
    PreprocessOptions options;
    options.continuous_streaming = BooleanRecord(*request, "continuous_streaming", false);
    const auto segments = static_cast<ProcessorState*>(handle)->processor.Preprocess(
        text, language_code, options);

    auto storage = std::make_unique<OutputStorage>();
    storage->request_id = CopySpan(request->request_id, "request ID");
    storage->session_id = CopySpan(request->session_id, "session ID");
    storage->names.reserve(segments.size());
    storage->data.reserve(segments.size());
    storage->views.reserve(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
      storage->names.push_back("normalized_text." + std::to_string(index));
      storage->data.push_back(segments[index].text);
    }
    for (size_t index = 0; index < segments.size(); ++index) {
      NemotronSpeechRecordViewV1 view{};
      view.struct_size = sizeof(NemotronSpeechRecordViewV1);
      view.name = Span(storage->names[index]);
      view.kind = NEMOTRON_SPEECH_RECORD_UTF8_V1;
      view.element_type = NEMOTRON_SPEECH_ELEMENT_U8_V1;
      view.data = Span(storage->data[index]);
      view.sequence = static_cast<uint64_t>(segments[index].sentence_num);
      view.has_sequence = 1;
      view.is_final = segments[index].is_last_sentence ? 1 : 0;
      storage->views.push_back(view);
    }

    output->schema_version = kSchemaVersion;
    output->request_id = Span(storage->request_id);
    output->session_id = Span(storage->session_id);
    output->records = storage->views.data();
    output->record_count = storage->views.size();
    output->sequence = request->sequence;
    output->has_sequence = request->has_sequence;
    output->is_final = request->is_final;
    output->context = storage.release();
    return Status(0);
  }
  catch (const TextPreprocessorException& error) {
    return Status(100 + static_cast<int32_t>(error.code()), error.what());
  }
  catch (const std::exception& error) {
    return Status(1, error.what());
  }
  catch (...) {
    return Status(1, "unknown exception while processing text");
  }
}

void
ReleaseOutput(NemotronSpeechProcessorHandleV1, NemotronSpeechEnvelopeOutputV1* output)
{
  if (output == nullptr) {
    return;
  }
  delete static_cast<OutputStorage*>(output->context);
  *output = {};
  output->struct_size = sizeof(NemotronSpeechEnvelopeOutputV1);
}

const NemotronSpeechProcessorApiV1 kApi = {
    sizeof(NemotronSpeechProcessorApiV1),
    NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION,
    Capabilities,
    Create,
    Destroy,
    Process,
    ReleaseOutput,
    ReleaseStatus};

}  // namespace

extern "C" const NemotronSpeechProcessorApiV1*
nemotron_speech_processor_get_api(uint32_t requested_version)
{
  return requested_version == NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION ? &kApi : nullptr;
}
