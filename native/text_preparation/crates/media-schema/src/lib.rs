// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_VERSION: u32 = 1;
pub const STREAM_ERROR_CODE_RECORD: &str = "stream_error_code";
pub const STREAM_ERROR_MESSAGE_RECORD: &str = "stream_error_message";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordEnvelope {
    pub schema_version: u32,
    pub request_id: String,
    pub session_id: Option<String>,
    pub sequence: Option<u64>,
    pub is_final: bool,
    pub records: Vec<NamedRecord>,
}

impl RecordEnvelope {
    pub fn new(request_id: impl Into<String>, records: Vec<NamedRecord>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            request_id: request_id.into(),
            session_id: None,
            sequence: None,
            is_final: true,
            records,
        }
    }

    pub fn record(&self, name: &str) -> Option<&MediaRecord> {
        self.records
            .iter()
            .find(|record| record.name == name)
            .map(|record| &record.value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedRecord {
    pub name: String,
    pub value: MediaRecord,
}

impl NamedRecord {
    pub fn new(name: impl Into<String>, value: MediaRecord) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum MediaRecord {
    Utf8(String),
    Bytes(#[serde(with = "serde_bytes")] Vec<u8>),
    Audio(AudioBuffer),
    TokenIds(Vec<u32>),
    Tensor(Tensor),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioEncoding {
    PcmS16Le,
}

impl AudioEncoding {
    fn bytes_per_sample(self) -> usize {
        match self {
            Self::PcmS16Le => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub encoding: AudioEncoding,
    pub sample_rate_hz: u32,
    pub channels: u16,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    U8,
    I32,
    I64,
    F32,
}

impl ElementType {
    fn size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub element_type: ElementType,
    pub shape: Vec<usize>,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationLimits {
    pub max_request_id_bytes: usize,
    pub max_session_id_bytes: usize,
    pub max_records: usize,
    pub max_record_name_bytes: usize,
    pub max_text_bytes: usize,
    pub max_binary_bytes: usize,
    pub max_audio_bytes: usize,
    pub max_tokens: usize,
    pub max_tensor_rank: usize,
    pub max_tensor_bytes: usize,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_request_id_bytes: 256,
            max_session_id_bytes: 256,
            max_records: 256,
            max_record_name_bytes: 128,
            max_text_bytes: 256 * 1024,
            max_binary_bytes: 16 * 1024 * 1024,
            max_audio_bytes: 64 * 1024 * 1024,
            max_tokens: 1_048_576,
            max_tensor_rank: 8,
            max_tensor_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ValidationError {
    #[error("unsupported schema version {actual}; expected {expected}")]
    UnsupportedSchema { actual: u32, expected: u32 },
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} exceeds the configured limit of {limit}")]
    LimitExceeded { field: &'static str, limit: usize },
    #[error("record name is duplicated: {0}")]
    DuplicateRecord(String),
    #[error("audio configuration is invalid: {0}")]
    InvalidAudio(&'static str),
    #[error("tensor shape or byte length is invalid")]
    InvalidTensor,
    #[error("floating-point record is not finite")]
    NonFiniteFloat,
}

pub fn validate_envelope(
    envelope: &RecordEnvelope,
    limits: &ValidationLimits,
) -> Result<(), ValidationError> {
    if envelope.schema_version != SCHEMA_VERSION {
        return Err(ValidationError::UnsupportedSchema {
            actual: envelope.schema_version,
            expected: SCHEMA_VERSION,
        });
    }
    validate_identifier(
        "request_id",
        &envelope.request_id,
        limits.max_request_id_bytes,
    )?;
    if let Some(session_id) = &envelope.session_id {
        validate_identifier("session_id", session_id, limits.max_session_id_bytes)?;
    }
    if envelope.records.len() > limits.max_records {
        return Err(ValidationError::LimitExceeded {
            field: "records",
            limit: limits.max_records,
        });
    }

    let mut names = HashSet::with_capacity(envelope.records.len());
    for record in &envelope.records {
        validate_identifier("record name", &record.name, limits.max_record_name_bytes)?;
        if !names.insert(&record.name) {
            return Err(ValidationError::DuplicateRecord(record.name.clone()));
        }
        validate_record(&record.value, limits)?;
    }
    Ok(())
}

pub fn validate_record(
    record: &MediaRecord,
    limits: &ValidationLimits,
) -> Result<(), ValidationError> {
    match record {
        MediaRecord::Utf8(text) => {
            if text.is_empty() {
                return Err(ValidationError::Empty { field: "text" });
            }
            check_limit("text", text.len(), limits.max_text_bytes)
        }
        MediaRecord::Bytes(data) => {
            check_limit("binary record", data.len(), limits.max_binary_bytes)
        }
        MediaRecord::Audio(audio) => validate_audio(audio, limits),
        MediaRecord::TokenIds(tokens) => check_limit("token IDs", tokens.len(), limits.max_tokens),
        MediaRecord::Tensor(tensor) => validate_tensor(tensor, limits),
        MediaRecord::Float(value) if !value.is_finite() => Err(ValidationError::NonFiniteFloat),
        MediaRecord::Integer(_) | MediaRecord::Float(_) | MediaRecord::Boolean(_) => Ok(()),
    }
}

fn validate_identifier(
    field: &'static str,
    value: &str,
    limit: usize,
) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    check_limit(field, value.len(), limit)
}

fn check_limit(field: &'static str, actual: usize, limit: usize) -> Result<(), ValidationError> {
    if actual > limit {
        Err(ValidationError::LimitExceeded { field, limit })
    } else {
        Ok(())
    }
}

fn validate_audio(audio: &AudioBuffer, limits: &ValidationLimits) -> Result<(), ValidationError> {
    check_limit("audio", audio.data.len(), limits.max_audio_bytes)?;
    if audio.sample_rate_hz == 0 {
        return Err(ValidationError::InvalidAudio(
            "sample rate must be positive",
        ));
    }
    if audio.channels == 0 || audio.channels > 32 {
        return Err(ValidationError::InvalidAudio("channel count must be 1-32"));
    }
    let frame_bytes = audio.encoding.bytes_per_sample() * usize::from(audio.channels);
    if !audio.data.len().is_multiple_of(frame_bytes) {
        return Err(ValidationError::InvalidAudio(
            "payload must contain complete sample frames",
        ));
    }
    Ok(())
}

fn validate_tensor(tensor: &Tensor, limits: &ValidationLimits) -> Result<(), ValidationError> {
    check_limit("tensor rank", tensor.shape.len(), limits.max_tensor_rank)?;
    check_limit("tensor", tensor.data.len(), limits.max_tensor_bytes)?;
    let elements = tensor
        .shape
        .iter()
        .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
        .ok_or(ValidationError::InvalidTensor)?;
    let expected = elements
        .checked_mul(tensor.element_type.size_bytes())
        .ok_or(ValidationError::InvalidTensor)?;
    if expected != tensor.data.len() {
        return Err(ValidationError::InvalidTensor);
    }
    if tensor.element_type == ElementType::F32 {
        for sample in tensor.data.chunks_exact(4) {
            let value = f32::from_le_bytes(sample.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(ValidationError::NonFiniteFloat);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_record_names() {
        let envelope = RecordEnvelope::new(
            "request-1",
            vec![
                NamedRecord::new("text", MediaRecord::Utf8("one".into())),
                NamedRecord::new("text", MediaRecord::Utf8("two".into())),
            ],
        );
        assert_eq!(
            validate_envelope(&envelope, &ValidationLimits::default()),
            Err(ValidationError::DuplicateRecord("text".into()))
        );
    }

    #[test]
    fn rejects_malformed_audio_frames() {
        let audio = MediaRecord::Audio(AudioBuffer {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate_hz: 16_000,
            channels: 1,
            data: vec![0],
        });
        assert!(matches!(
            validate_record(&audio, &ValidationLimits::default()),
            Err(ValidationError::InvalidAudio(_))
        ));
    }

    #[test]
    fn rejects_tensor_size_mismatch() {
        let tensor = MediaRecord::Tensor(Tensor {
            element_type: ElementType::I64,
            shape: vec![2, 2],
            data: vec![0; 24],
        });
        assert_eq!(
            validate_record(&tensor, &ValidationLimits::default()),
            Err(ValidationError::InvalidTensor)
        );
    }
}
