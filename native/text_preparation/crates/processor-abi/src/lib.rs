// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

pub mod raw;

use std::{
    mem::size_of,
    path::Path,
    slice,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use libloading::Library;
use media_schema::{
    AudioBuffer, AudioEncoding, ElementType, MediaRecord, NamedRecord, RecordEnvelope, Tensor,
};
use thiserror::Error;

use raw::{
    ByteSpanV1, EnvelopeOutputV1, EnvelopeViewV1, ProcessorApiV1, ProcessorCapabilitiesV1,
    ProcessorHandleV1, RecordViewV1, StatusV1,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessorCapabilities {
    pub processor_id: String,
    pub batch_input: bool,
    pub streaming_input: bool,
    pub thread_safe: bool,
    pub stateful: bool,
}

#[derive(Debug, Error)]
pub enum ProcessorError {
    #[error("unable to load native processor: {0}")]
    Load(String),
    #[error("processor ABI version or layout is incompatible")]
    IncompatibleAbi,
    #[error("native processor returned status {code}: {message}")]
    Native { code: i32, message: String },
    #[error("native processor returned malformed output: {0}")]
    MalformedOutput(String),
    #[error("native processor received malformed input: {0}")]
    MalformedInput(String),
}

struct NativeHandle(ProcessorHandleV1);

unsafe impl Send for NativeHandle {}

struct ConcurrentNativeHandle(NativeHandle);

unsafe impl Sync for ConcurrentNativeHandle {}

enum NativeHandleAccess {
    Serialized(Mutex<NativeHandle>),
    Concurrent(ConcurrentNativeHandle),
}

pub struct NativeProcessor {
    _library: Library,
    api: &'static ProcessorApiV1,
    handle: NativeHandleAccess,
    capabilities: ProcessorCapabilities,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessorTimings {
    pub queue_wait: Duration,
    pub active: Duration,
}

pub struct NativeProcessorPool {
    processors: Vec<Arc<NativeProcessor>>,
    available: Mutex<Vec<usize>>,
    ready: Condvar,
}

impl NativeProcessorPool {
    pub unsafe fn load(
        path: &Path,
        config: &RecordEnvelope,
        instances: usize,
    ) -> Result<Arc<Self>, ProcessorError> {
        if instances == 0 {
            return Err(ProcessorError::Load(
                "processor pool must contain at least one instance".into(),
            ));
        }
        let first = unsafe { NativeProcessor::load(path, config) }?;
        let actual_instances = if first.capabilities().thread_safe {
            1
        } else {
            instances
        };
        let mut processors = Vec::with_capacity(actual_instances);
        processors.push(first);
        for _ in 1..actual_instances {
            processors.push(unsafe { NativeProcessor::load(path, config) }?);
        }
        let available = (0..actual_instances).rev().collect();
        Ok(Arc::new(Self {
            processors,
            available: Mutex::new(available),
            ready: Condvar::new(),
        }))
    }

    pub fn capabilities(&self) -> &ProcessorCapabilities {
        self.processors[0].capabilities()
    }

    pub fn size(&self) -> usize {
        self.processors.len()
    }

    pub fn process(&self, request: &RecordEnvelope) -> Result<RecordEnvelope, ProcessorError> {
        self.process_with_timings(request).map(|(output, _)| output)
    }

    pub fn process_with_timings(
        &self,
        request: &RecordEnvelope,
    ) -> Result<(RecordEnvelope, ProcessorTimings), ProcessorError> {
        if self.capabilities().thread_safe {
            let active_started = Instant::now();
            let output = self.processors[0].process(request)?;
            return Ok((
                output,
                ProcessorTimings {
                    queue_wait: Duration::ZERO,
                    active: active_started.elapsed(),
                },
            ));
        }
        let queue_started = Instant::now();
        let index = {
            let mut available = self.available.lock().expect("processor pool lock poisoned");
            while available.is_empty() {
                available = self
                    .ready
                    .wait(available)
                    .expect("processor pool lock poisoned");
            }
            available.pop().expect("availability checked")
        };
        let queue_wait = queue_started.elapsed();
        let _checkout = ProcessorCheckout { pool: self, index };
        let active_started = Instant::now();
        let output = self.processors[index].process(request)?;
        Ok((
            output,
            ProcessorTimings {
                queue_wait,
                active: active_started.elapsed(),
            },
        ))
    }
}

struct ProcessorCheckout<'a> {
    pool: &'a NativeProcessorPool,
    index: usize,
}

impl Drop for ProcessorCheckout<'_> {
    fn drop(&mut self) {
        self.pool
            .available
            .lock()
            .expect("processor pool lock poisoned")
            .push(self.index);
        self.pool.ready.notify_one();
    }
}

impl NativeProcessor {
    pub unsafe fn load(path: &Path, config: &RecordEnvelope) -> Result<Arc<Self>, ProcessorError> {
        let library = unsafe { Library::new(path) }
            .map_err(|error| ProcessorError::Load(error.to_string()))?;
        let get_api: libloading::Symbol<'_, raw::GetProcessorApi> = unsafe {
            library
                .get(raw::API_SYMBOL)
                .map_err(|error| ProcessorError::Load(error.to_string()))?
        };
        let api_ptr = unsafe { get_api(raw::PROCESSOR_ABI_VERSION) };
        if api_ptr.is_null() {
            return Err(ProcessorError::IncompatibleAbi);
        }
        let api = unsafe { &*api_ptr };
        if api.abi_version != raw::PROCESSOR_ABI_VERSION
            || api.struct_size < size_of::<ProcessorApiV1>()
        {
            return Err(ProcessorError::IncompatibleAbi);
        }

        let capabilities = read_capabilities(api)?;
        let borrowed = BorrowedEnvelope::new(config)?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { (api.create)(&borrowed.view, &mut handle) };
        check_status(api, status)?;
        if handle.is_null() {
            return Err(ProcessorError::MalformedOutput(
                "create returned a null handle".into(),
            ));
        }
        let api = unsafe { &*(api as *const ProcessorApiV1) };
        let handle = if capabilities.thread_safe {
            NativeHandleAccess::Concurrent(ConcurrentNativeHandle(NativeHandle(handle)))
        } else {
            NativeHandleAccess::Serialized(Mutex::new(NativeHandle(handle)))
        };
        Ok(Arc::new(Self {
            _library: library,
            api,
            handle,
            capabilities,
        }))
    }

    pub fn capabilities(&self) -> &ProcessorCapabilities {
        &self.capabilities
    }

    pub fn process(&self, request: &RecordEnvelope) -> Result<RecordEnvelope, ProcessorError> {
        let borrowed = BorrowedEnvelope::new(request)?;
        match &self.handle {
            NativeHandleAccess::Serialized(handle) => {
                let handle = handle.lock().expect("native processor lock poisoned");
                self.process_handle(handle.0, &borrowed)
            }
            NativeHandleAccess::Concurrent(handle) => self.process_handle((handle.0).0, &borrowed),
        }
    }

    fn process_handle(
        &self,
        handle: ProcessorHandleV1,
        borrowed: &BorrowedEnvelope,
    ) -> Result<RecordEnvelope, ProcessorError> {
        let mut output = raw::empty_output();
        let status = unsafe { (self.api.process)(handle, &borrowed.view, &mut output) };
        check_status(self.api, status)?;
        let converted = copy_output(&output);
        unsafe { (self.api.release_output)(handle, &mut output) };
        converted
    }
}

impl Drop for NativeProcessor {
    fn drop(&mut self) {
        match &mut self.handle {
            NativeHandleAccess::Serialized(handle) => {
                if let Ok(handle) = handle.get_mut() {
                    destroy_handle(self.api, handle);
                }
            }
            NativeHandleAccess::Concurrent(handle) => destroy_handle(self.api, &mut handle.0),
        }
    }
}

fn destroy_handle(api: &ProcessorApiV1, handle: &mut NativeHandle) {
    if !handle.0.is_null() {
        unsafe { (api.destroy)(handle.0) };
        handle.0 = std::ptr::null_mut();
    }
}

struct BorrowedRecord {
    name: Vec<u8>,
    data: Vec<u8>,
    shape: Vec<usize>,
    view: RecordViewV1,
}

impl BorrowedRecord {
    fn new(record: &NamedRecord) -> Result<Self, ProcessorError> {
        let name = record.name.as_bytes().to_vec();
        let (kind, element_type, data, shape, sample_rate_hz, channels, audio_encoding) =
            encode_record(&record.value);
        let mut value = Self {
            name,
            data,
            shape,
            view: RecordViewV1 {
                struct_size: size_of::<RecordViewV1>(),
                name: ByteSpanV1 {
                    data: std::ptr::null(),
                    len: 0,
                },
                kind,
                element_type,
                data: ByteSpanV1 {
                    data: std::ptr::null(),
                    len: 0,
                },
                shape: std::ptr::null(),
                rank: 0,
                sample_rate_hz,
                channels,
                audio_encoding,
                sequence: 0,
                has_sequence: 0,
                is_final: 0,
                reserved: [0; 6],
            },
        };
        value.view.name = span(&value.name);
        value.view.data = span(&value.data);
        value.view.shape = value.shape.as_ptr();
        value.view.rank = value.shape.len();
        Ok(value)
    }
}

struct BorrowedEnvelope {
    request_id: Vec<u8>,
    session_id: Vec<u8>,
    records: Vec<BorrowedRecord>,
    views: Vec<RecordViewV1>,
    view: EnvelopeViewV1,
}

impl BorrowedEnvelope {
    fn new(envelope: &RecordEnvelope) -> Result<Self, ProcessorError> {
        let request_id = envelope.request_id.as_bytes().to_vec();
        let session_id = envelope
            .session_id
            .as_deref()
            .unwrap_or_default()
            .as_bytes()
            .to_vec();
        let records = envelope
            .records
            .iter()
            .map(BorrowedRecord::new)
            .collect::<Result<Vec<_>, _>>()?;
        let views = records.iter().map(|record| record.view).collect::<Vec<_>>();
        let mut value = Self {
            request_id,
            session_id,
            records,
            views,
            view: EnvelopeViewV1 {
                struct_size: size_of::<EnvelopeViewV1>(),
                schema_version: envelope.schema_version,
                request_id: ByteSpanV1 {
                    data: std::ptr::null(),
                    len: 0,
                },
                session_id: ByteSpanV1 {
                    data: std::ptr::null(),
                    len: 0,
                },
                records: std::ptr::null(),
                record_count: 0,
                sequence: envelope.sequence.unwrap_or_default(),
                has_sequence: u8::from(envelope.sequence.is_some()),
                is_final: u8::from(envelope.is_final),
                reserved: [0; 6],
            },
        };
        value.view.request_id = span(&value.request_id);
        value.view.session_id = span(&value.session_id);
        value.view.records = value.views.as_ptr();
        value.view.record_count = value.views.len();
        debug_assert_eq!(value.records.len(), value.views.len());
        Ok(value)
    }
}

fn encode_record(record: &MediaRecord) -> (u32, u32, Vec<u8>, Vec<usize>, u32, u16, u16) {
    match record {
        MediaRecord::Utf8(value) => (
            raw::RECORD_UTF8,
            raw::ELEMENT_U8,
            value.as_bytes().to_vec(),
            Vec::new(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Bytes(value) => (
            raw::RECORD_BYTES,
            raw::ELEMENT_U8,
            value.clone(),
            Vec::new(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Audio(value) => (
            raw::RECORD_AUDIO,
            raw::ELEMENT_U8,
            value.data.clone(),
            Vec::new(),
            value.sample_rate_hz,
            value.channels,
            match value.encoding {
                AudioEncoding::PcmS16Le => raw::AUDIO_PCM_S16_LE as u16,
            },
        ),
        MediaRecord::TokenIds(value) => (
            raw::RECORD_TOKEN_IDS,
            raw::ELEMENT_I32,
            value.iter().flat_map(|item| item.to_le_bytes()).collect(),
            vec![value.len()],
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Tensor(value) => (
            raw::RECORD_TENSOR,
            match value.element_type {
                ElementType::U8 => raw::ELEMENT_U8,
                ElementType::I32 => raw::ELEMENT_I32,
                ElementType::I64 => raw::ELEMENT_I64,
                ElementType::F32 => raw::ELEMENT_F32,
            },
            value.data.clone(),
            value.shape.clone(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Integer(value) => (
            raw::RECORD_INTEGER,
            raw::ELEMENT_I64,
            value.to_le_bytes().to_vec(),
            Vec::new(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Float(value) => (
            raw::RECORD_FLOAT,
            raw::ELEMENT_I64,
            value.to_le_bytes().to_vec(),
            Vec::new(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
        MediaRecord::Boolean(value) => (
            raw::RECORD_BOOLEAN,
            raw::ELEMENT_U8,
            vec![u8::from(*value)],
            Vec::new(),
            0,
            0,
            raw::AUDIO_NONE as u16,
        ),
    }
}

fn read_capabilities(api: &ProcessorApiV1) -> Result<ProcessorCapabilities, ProcessorError> {
    let mut raw_capabilities = ProcessorCapabilitiesV1 {
        struct_size: size_of::<ProcessorCapabilitiesV1>(),
        processor_id: ByteSpanV1 {
            data: std::ptr::null(),
            len: 0,
        },
        batch_input: 0,
        streaming_input: 0,
        thread_safe: 0,
        stateful: 0,
        reserved: [0; 4],
    };
    let status = unsafe { (api.capabilities)(&mut raw_capabilities) };
    check_status(api, status)?;
    Ok(ProcessorCapabilities {
        processor_id: copy_utf8(raw_capabilities.processor_id, "processor ID")?,
        batch_input: raw_capabilities.batch_input != 0,
        streaming_input: raw_capabilities.streaming_input != 0,
        thread_safe: raw_capabilities.thread_safe != 0,
        stateful: raw_capabilities.stateful != 0,
    })
}

fn check_status(api: &ProcessorApiV1, mut status: StatusV1) -> Result<(), ProcessorError> {
    let result = if status.code == 0 {
        Ok(())
    } else {
        Err(ProcessorError::Native {
            code: status.code,
            message: copy_utf8(status.message, "status message")
                .unwrap_or_else(|_| "invalid native error message".into()),
        })
    };
    unsafe { (api.release_status)(&mut status) };
    result
}

fn copy_output(output: &EnvelopeOutputV1) -> Result<RecordEnvelope, ProcessorError> {
    if output.struct_size < size_of::<EnvelopeOutputV1>() {
        return Err(ProcessorError::MalformedOutput(
            "output structure is too small".into(),
        ));
    }
    let request_id = copy_utf8(output.request_id, "request ID")?;
    let session_id = if output.session_id.len == 0 {
        None
    } else {
        Some(copy_utf8(output.session_id, "session ID")?)
    };
    if output.record_count > 0 && output.records.is_null() {
        return Err(ProcessorError::MalformedOutput(
            "record array is null".into(),
        ));
    }
    let views = if output.record_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(output.records, output.record_count) }
    };
    let records = views
        .iter()
        .map(copy_record)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RecordEnvelope {
        schema_version: output.schema_version,
        request_id,
        session_id,
        sequence: (output.has_sequence != 0).then_some(output.sequence),
        is_final: output.is_final != 0,
        records,
    })
}

fn copy_record(view: &RecordViewV1) -> Result<NamedRecord, ProcessorError> {
    if view.struct_size < size_of::<RecordViewV1>() {
        return Err(ProcessorError::MalformedOutput(
            "record structure is too small".into(),
        ));
    }
    let name = copy_utf8(view.name, "record name")?;
    let data = copy_bytes(view.data, "record data")?;
    let value = match view.kind {
        raw::RECORD_UTF8 => MediaRecord::Utf8(
            String::from_utf8(data)
                .map_err(|_| ProcessorError::MalformedOutput("invalid UTF-8 record".into()))?,
        ),
        raw::RECORD_BYTES => MediaRecord::Bytes(data),
        raw::RECORD_AUDIO => MediaRecord::Audio(AudioBuffer {
            encoding: match u32::from(view.audio_encoding) {
                raw::AUDIO_PCM_S16_LE => AudioEncoding::PcmS16Le,
                _ => {
                    return Err(ProcessorError::MalformedOutput(
                        "unknown audio encoding".into(),
                    ))
                }
            },
            sample_rate_hz: view.sample_rate_hz,
            channels: view.channels,
            data,
        }),
        raw::RECORD_TOKEN_IDS => {
            if data.len() % 4 != 0 {
                return Err(ProcessorError::MalformedOutput(
                    "token byte length is invalid".into(),
                ));
            }
            MediaRecord::TokenIds(
                data.chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                    .collect(),
            )
        }
        raw::RECORD_TENSOR => {
            if view.rank > 0 && view.shape.is_null() {
                return Err(ProcessorError::MalformedOutput(
                    "tensor shape is null".into(),
                ));
            }
            let shape = if view.rank == 0 {
                Vec::new()
            } else {
                unsafe { slice::from_raw_parts(view.shape, view.rank) }.to_vec()
            };
            MediaRecord::Tensor(Tensor {
                element_type: match view.element_type {
                    raw::ELEMENT_U8 => ElementType::U8,
                    raw::ELEMENT_I32 => ElementType::I32,
                    raw::ELEMENT_I64 => ElementType::I64,
                    raw::ELEMENT_F32 => ElementType::F32,
                    _ => {
                        return Err(ProcessorError::MalformedOutput(
                            "unknown tensor element type".into(),
                        ))
                    }
                },
                shape,
                data,
            })
        }
        raw::RECORD_INTEGER if data.len() == 8 => MediaRecord::Integer(i64::from_le_bytes(
            data.as_slice().try_into().expect("eight-byte value"),
        )),
        raw::RECORD_FLOAT if data.len() == 8 => MediaRecord::Float(f64::from_le_bytes(
            data.as_slice().try_into().expect("eight-byte value"),
        )),
        raw::RECORD_BOOLEAN if data.len() == 1 => MediaRecord::Boolean(data[0] != 0),
        _ => {
            return Err(ProcessorError::MalformedOutput(format!(
                "unknown or malformed record kind {}",
                view.kind
            )))
        }
    };
    Ok(NamedRecord { name, value })
}

fn span(bytes: &[u8]) -> ByteSpanV1 {
    ByteSpanV1 {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

fn copy_utf8(value: ByteSpanV1, field: &str) -> Result<String, ProcessorError> {
    String::from_utf8(copy_bytes(value, field)?)
        .map_err(|_| ProcessorError::MalformedOutput(format!("{field} is not valid UTF-8")))
}

fn copy_bytes(value: ByteSpanV1, field: &str) -> Result<Vec<u8>, ProcessorError> {
    if value.len == 0 {
        return Ok(Vec::new());
    }
    if value.data.is_null() {
        return Err(ProcessorError::MalformedOutput(format!(
            "{field} pointer is null"
        )));
    }
    Ok(unsafe { slice::from_raw_parts(value.data, value.len) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_abi_arrays_accept_null_pointers() {
        let mut output = raw::empty_output();
        output.schema_version = media_schema::SCHEMA_VERSION;
        let copied = copy_output(&output).unwrap();
        assert!(copied.records.is_empty());

        let name = b"tensor";
        let tensor = RecordViewV1 {
            struct_size: size_of::<RecordViewV1>(),
            name: ByteSpanV1 {
                data: name.as_ptr(),
                len: name.len(),
            },
            kind: raw::RECORD_TENSOR,
            element_type: raw::ELEMENT_U8,
            data: ByteSpanV1 {
                data: std::ptr::null(),
                len: 0,
            },
            shape: std::ptr::null(),
            rank: 0,
            sample_rate_hz: 0,
            channels: 0,
            audio_encoding: raw::AUDIO_NONE as u16,
            sequence: 0,
            has_sequence: 0,
            is_final: 0,
            reserved: [0; 6],
        };
        let copied = copy_record(&tensor).unwrap();
        assert!(matches!(
            copied.value,
            MediaRecord::Tensor(Tensor { shape, data, .. }) if shape.is_empty() && data.is_empty()
        ));
    }
}
