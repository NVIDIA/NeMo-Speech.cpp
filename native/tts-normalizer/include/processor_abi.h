/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEMOTRON_SPEECH_PROCESSOR_ABI_VERSION 1u

enum NemotronSpeechRecordKindV1 {
  NEMOTRON_SPEECH_RECORD_UTF8_V1 = 1,
  NEMOTRON_SPEECH_RECORD_BYTES_V1 = 2,
  NEMOTRON_SPEECH_RECORD_AUDIO_V1 = 3,
  NEMOTRON_SPEECH_RECORD_TOKEN_IDS_V1 = 4,
  NEMOTRON_SPEECH_RECORD_TENSOR_V1 = 5,
  NEMOTRON_SPEECH_RECORD_INTEGER_V1 = 6,
  NEMOTRON_SPEECH_RECORD_FLOAT_V1 = 7,
  NEMOTRON_SPEECH_RECORD_BOOLEAN_V1 = 8,
};

enum NemotronSpeechElementTypeV1 {
  NEMOTRON_SPEECH_ELEMENT_NONE_V1 = 0,
  NEMOTRON_SPEECH_ELEMENT_U8_V1 = 1,
  NEMOTRON_SPEECH_ELEMENT_I32_V1 = 2,
  NEMOTRON_SPEECH_ELEMENT_I64_V1 = 3,
  NEMOTRON_SPEECH_ELEMENT_F32_V1 = 4,
};

enum NemotronSpeechAudioEncodingV1 {
  NEMOTRON_SPEECH_AUDIO_NONE_V1 = 0,
  NEMOTRON_SPEECH_AUDIO_PCM_S16_LE_V1 = 1,
  NEMOTRON_SPEECH_AUDIO_PCM_F32_LE_V1 = 2,
};

typedef struct NemotronSpeechByteSpanV1 {
  const uint8_t* data;
  size_t len;
} NemotronSpeechByteSpanV1;

typedef struct NemotronSpeechStatusV1 {
  size_t struct_size;
  int32_t code;
  NemotronSpeechByteSpanV1 message;
  void* context;
} NemotronSpeechStatusV1;

typedef struct NemotronSpeechRecordViewV1 {
  size_t struct_size;
  NemotronSpeechByteSpanV1 name;
  uint32_t kind;
  uint32_t element_type;
  NemotronSpeechByteSpanV1 data;
  const size_t* shape;
  size_t rank;
  uint32_t sample_rate_hz;
  uint16_t channels;
  uint16_t audio_encoding;
  uint64_t sequence;
  uint8_t has_sequence;
  uint8_t is_final;
  uint8_t reserved[6];
} NemotronSpeechRecordViewV1;

typedef struct NemotronSpeechEnvelopeViewV1 {
  size_t struct_size;
  uint32_t schema_version;
  NemotronSpeechByteSpanV1 request_id;
  NemotronSpeechByteSpanV1 session_id;
  const NemotronSpeechRecordViewV1* records;
  size_t record_count;
  uint64_t sequence;
  uint8_t has_sequence;
  uint8_t is_final;
  uint8_t reserved[6];
} NemotronSpeechEnvelopeViewV1;

typedef struct NemotronSpeechEnvelopeOutputV1 {
  size_t struct_size;
  uint32_t schema_version;
  NemotronSpeechByteSpanV1 request_id;
  NemotronSpeechByteSpanV1 session_id;
  const NemotronSpeechRecordViewV1* records;
  size_t record_count;
  uint64_t sequence;
  uint8_t has_sequence;
  uint8_t is_final;
  uint8_t reserved[6];
  void* context;
} NemotronSpeechEnvelopeOutputV1;

typedef struct NemotronSpeechProcessorCapabilitiesV1 {
  size_t struct_size;
  NemotronSpeechByteSpanV1 processor_id;
  uint8_t batch_input;
  uint8_t streaming_input;
  uint8_t thread_safe;
  uint8_t stateful;
  uint8_t reserved[4];
} NemotronSpeechProcessorCapabilitiesV1;

typedef void* NemotronSpeechProcessorHandleV1;

typedef struct NemotronSpeechProcessorApiV1 {
  size_t struct_size;
  uint32_t abi_version;
  NemotronSpeechStatusV1 (*capabilities)(NemotronSpeechProcessorCapabilitiesV1* output);
  NemotronSpeechStatusV1 (*create)(const NemotronSpeechEnvelopeViewV1* config,
                                  NemotronSpeechProcessorHandleV1* output);
  void (*destroy)(NemotronSpeechProcessorHandleV1 handle);
  NemotronSpeechStatusV1 (*process)(
      NemotronSpeechProcessorHandleV1 handle, const NemotronSpeechEnvelopeViewV1* request,
      NemotronSpeechEnvelopeOutputV1* output);
  void (*release_output)(NemotronSpeechProcessorHandleV1 handle,
                         NemotronSpeechEnvelopeOutputV1* output);
  void (*release_status)(NemotronSpeechStatusV1* status);
} NemotronSpeechProcessorApiV1;

const NemotronSpeechProcessorApiV1* nemotron_speech_processor_get_api(
    uint32_t requested_version);

#ifdef __cplusplus
}
#endif
