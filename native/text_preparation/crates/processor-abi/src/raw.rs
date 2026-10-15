// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

use std::{
    ffi::{c_char, c_void},
    mem::size_of,
};

pub const PROCESSOR_ABI_VERSION: u32 = 1;

pub const STATUS_OK: i32 = 0;
pub const STATUS_INTERNAL: i32 = 1;
pub const STATUS_INVALID_ARGUMENT: i32 = 2;

pub const RECORD_UTF8: u32 = 1;
pub const RECORD_BYTES: u32 = 2;
pub const RECORD_AUDIO: u32 = 3;
pub const RECORD_TOKEN_IDS: u32 = 4;
pub const RECORD_TENSOR: u32 = 5;
pub const RECORD_INTEGER: u32 = 6;
pub const RECORD_FLOAT: u32 = 7;
pub const RECORD_BOOLEAN: u32 = 8;

pub const ELEMENT_NONE: u32 = 0;
pub const ELEMENT_U8: u32 = 1;
pub const ELEMENT_I32: u32 = 2;
pub const ELEMENT_I64: u32 = 3;
pub const ELEMENT_F32: u32 = 4;

pub const AUDIO_NONE: u32 = 0;
pub const AUDIO_PCM_S16_LE: u32 = 1;
pub const AUDIO_PCM_F32_LE: u32 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSpanV1 {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StatusV1 {
    pub struct_size: usize,
    pub code: i32,
    pub message: ByteSpanV1,
    pub context: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RecordViewV1 {
    pub struct_size: usize,
    pub name: ByteSpanV1,
    pub kind: u32,
    pub element_type: u32,
    pub data: ByteSpanV1,
    pub shape: *const usize,
    pub rank: usize,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub audio_encoding: u16,
    pub sequence: u64,
    pub has_sequence: u8,
    pub is_final: u8,
    pub reserved: [u8; 6],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeViewV1 {
    pub struct_size: usize,
    pub schema_version: u32,
    pub request_id: ByteSpanV1,
    pub session_id: ByteSpanV1,
    pub records: *const RecordViewV1,
    pub record_count: usize,
    pub sequence: u64,
    pub has_sequence: u8,
    pub is_final: u8,
    pub reserved: [u8; 6],
}

#[repr(C)]
#[derive(Debug)]
pub struct EnvelopeOutputV1 {
    pub struct_size: usize,
    pub schema_version: u32,
    pub request_id: ByteSpanV1,
    pub session_id: ByteSpanV1,
    pub records: *const RecordViewV1,
    pub record_count: usize,
    pub sequence: u64,
    pub has_sequence: u8,
    pub is_final: u8,
    pub reserved: [u8; 6],
    pub context: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProcessorCapabilitiesV1 {
    pub struct_size: usize,
    pub processor_id: ByteSpanV1,
    pub batch_input: u8,
    pub streaming_input: u8,
    pub thread_safe: u8,
    pub stateful: u8,
    pub reserved: [u8; 4],
}

pub type ProcessorHandleV1 = *mut c_void;

#[repr(C)]
pub struct ProcessorApiV1 {
    pub struct_size: usize,
    pub abi_version: u32,
    pub capabilities: unsafe extern "C" fn(*mut ProcessorCapabilitiesV1) -> StatusV1,
    pub create: unsafe extern "C" fn(*const EnvelopeViewV1, *mut ProcessorHandleV1) -> StatusV1,
    pub destroy: unsafe extern "C" fn(ProcessorHandleV1),
    pub process: unsafe extern "C" fn(
        ProcessorHandleV1,
        *const EnvelopeViewV1,
        *mut EnvelopeOutputV1,
    ) -> StatusV1,
    pub release_output: unsafe extern "C" fn(ProcessorHandleV1, *mut EnvelopeOutputV1),
    pub release_status: unsafe extern "C" fn(*mut StatusV1),
}

pub type GetProcessorApi = unsafe extern "C" fn(u32) -> *const ProcessorApiV1;

pub const API_SYMBOL: &[u8] = b"nemotron_speech_processor_get_api\0";

pub fn empty_output() -> EnvelopeOutputV1 {
    EnvelopeOutputV1 {
        struct_size: size_of::<EnvelopeOutputV1>(),
        schema_version: 0,
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
        sequence: 0,
        has_sequence: 0,
        is_final: 0,
        reserved: [0; 6],
        context: std::ptr::null_mut(),
    }
}

#[allow(dead_code)]
const _: *const c_char = std::ptr::null();
