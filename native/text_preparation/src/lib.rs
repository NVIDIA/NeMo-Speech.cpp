// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use media_schema::{MediaRecord, NamedRecord, RecordEnvelope};
use processor_abi::{NativeProcessorPool, ProcessorError};
use regex::Regex;
use serde::Deserialize;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    ptr, slice,
    sync::Arc,
};
use tokenizers::Tokenizer;

type Outcome<T> = Result<T, (i32, String)>;
fn invalid(message: impl Into<String>) -> (i32, String) {
    (1, message.into())
}
fn internal(message: impl Into<String>) -> (i32, String) {
    (2, message.into())
}
fn yes() -> bool {
    true
}
fn language() -> String {
    "en-US".into()
}
fn config_name() -> String {
    "sparrowhawk_configuration.ascii_proto".into()
}
fn pool() -> usize {
    2
}
fn input_limit() -> usize {
    65536
}
fn output_limit() -> usize {
    262144
}
fn segments_limit() -> usize {
    256
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "yes")]
    enabled: bool,
    native_library_root: PathBuf,
    native_processor: PathBuf,
    normalizer_root: PathBuf,
    #[serde(default)]
    normalizer_path: Option<PathBuf>,
    #[serde(default = "language")]
    language_code: String,
    #[serde(default = "config_name")]
    normalizer_config: String,
    #[serde(default)]
    direct_grammar: bool,
    #[serde(default)]
    abbreviations_path: Option<PathBuf>,
    #[serde(default = "yes")]
    append_punctuation: bool,
    #[serde(default = "pool")]
    pool_size: usize,
    #[serde(default)]
    bounded_preparation: bool,
    #[serde(default = "input_limit")]
    max_input_bytes: usize,
    #[serde(default = "output_limit")]
    max_normalized_bytes: usize,
    #[serde(default = "segments_limit")]
    max_segments: usize,
}

struct Preparation {
    tokenizer: Tokenizer,
    normalizer: Option<Arc<NativeProcessorPool>>,
    enabled: bool,
    max_input: usize,
    max_output: usize,
    max_segments: usize,
    punctuation: Regex,
}

fn canonical(path: &Path, directory: bool) -> Outcome<PathBuf> {
    let path = path
        .canonicalize()
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    if if directory {
        !path.is_dir()
    } else {
        !path.is_file()
    } {
        return Err(invalid("normalizer path has the wrong file type"));
    }
    Ok(path)
}
fn contained(root: &Path, path: &Path, directory: bool) -> Outcome<PathBuf> {
    let path = canonical(&root.join(path), directory)?;
    if !path.starts_with(root) {
        return Err(invalid("normalizer path escapes its root"));
    }
    Ok(path)
}
fn utf8_path(path: PathBuf) -> Outcome<String> {
    path.into_os_string()
        .into_string()
        .map_err(|_| invalid("normalizer path must be UTF-8"))
}
fn processor_error(error: ProcessorError) -> (i32, String) {
    let code = if matches!(
        error,
        ProcessorError::Native {
            code: 100..=102,
            ..
        }
    ) {
        1
    } else {
        2
    };
    (code, error.to_string())
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

impl Preparation {
    fn new(config: Option<Config>, tokenizer_path: &str) -> Outcome<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| invalid(format!("tokenizer failed: {e}")))?;
        let mut result = Self {
            tokenizer,
            normalizer: None,
            enabled: false,
            max_input: input_limit(),
            max_output: output_limit(),
            max_segments: segments_limit(),
            punctuation: Regex::new(r" +([,.?!।])").map_err(|e| internal(e.to_string()))?,
        };
        if let Some(c) = config {
            let _ = c.bounded_preparation;
            if c.language_code != "en-US" {
                return Err(invalid("normalization language_code must be en-US"));
            }
            for n in [
                c.max_input_bytes,
                c.max_normalized_bytes,
                c.max_segments,
                c.pool_size,
            ] {
                if n == 0 || n > i64::MAX as usize {
                    return Err(invalid(
                        "normalization limits must fit positive ABI integers",
                    ));
                }
            }
            let library_root = canonical(&c.native_library_root, true)?;
            let asset_root = canonical(&c.normalizer_root, true)?;
            if library_root.starts_with(&asset_root) || asset_root.starts_with(&library_root) {
                return Err(invalid(
                    "normalizer libraries and assets must have separate roots",
                ));
            }
            let library = canonical(&c.native_processor, false)?;
            if !library.starts_with(&library_root) {
                return Err(invalid("native_processor escapes native_library_root"));
            }
            let language_path = contained(
                &asset_root,
                c.normalizer_path.as_deref().unwrap_or(Path::new("en-US")),
                true,
            )?;
            let name = Path::new(&c.normalizer_config);
            if name.components().count() != 1
                || name.file_name().and_then(|n| n.to_str()) != Some(c.normalizer_config.as_str())
            {
                return Err(invalid(
                    "normalizer_config must name a file in normalizer_path",
                ));
            }
            let required: Vec<&str> = if c.direct_grammar {
                vec!["tokenize_and_classify.far", "verbalize.far"]
            } else {
                vec![&c.normalizer_config]
            };
            for name in required {
                contained(&language_path, Path::new(name), false)?;
            }
            if c.direct_grammar {
                for name in ["pre_process.far", "post_process.far", "speech_class.far"] {
                    let p = language_path.join(name);
                    if p.exists() || p.is_symlink() {
                        contained(&language_path, Path::new(name), false)?;
                    }
                }
            }
            let mut records = vec![
                NamedRecord::new("language_code", MediaRecord::Utf8("en-US".into())),
                NamedRecord::new(
                    "normalizer_path",
                    MediaRecord::Utf8(utf8_path(language_path.clone())?),
                ),
                NamedRecord::new("normalizer_config", MediaRecord::Utf8(c.normalizer_config)),
                NamedRecord::new("direct_grammar", MediaRecord::Boolean(c.direct_grammar)),
                NamedRecord::new(
                    "append_punctuation",
                    MediaRecord::Boolean(c.append_punctuation),
                ),
            ];
            for (key, value) in [
                ("max_input_bytes", c.max_input_bytes),
                ("max_normalized_bytes", c.max_normalized_bytes),
                ("max_segments", c.max_segments),
            ] {
                records.push(NamedRecord::new(key, MediaRecord::Integer(value as i64)));
            }
            if let Some(path) = c.abbreviations_path {
                let path = canonical(&language_path.join(path), false)?;
                if !path.starts_with(&asset_root) {
                    return Err(invalid("abbreviations_path escapes normalizer_root"));
                }
                records.push(NamedRecord::new(
                    "abbreviations_path",
                    MediaRecord::Utf8(utf8_path(path)?),
                ));
            }
            let normalizer = unsafe {
                NativeProcessorPool::load(
                    &library,
                    &RecordEnvelope::new("nemotron-cpp-normalization", records),
                    c.pool_size,
                )
            }
            .map_err(processor_error)?;
            if normalizer.capabilities().processor_id != "nemotron_speech.tts.text_preprocessor"
                || normalizer.capabilities().stateful
            {
                return Err(invalid(
                    "native processor is not the stateless TTS normalizer",
                ));
            }
            result.normalizer = Some(normalizer);
            result.enabled = c.enabled;
            result.max_input = c.max_input_bytes;
            result.max_output = c.max_normalized_bytes;
            result.max_segments = c.max_segments;
        }
        Ok(result)
    }
    fn prepare(&self, input: &str, apply: bool) -> Outcome<String> {
        let text = input.trim_matches(whitespace);
        if text.is_empty() || text.len() > self.max_input {
            return Err(invalid("text is empty or exceeds max_input_bytes"));
        }
        let normalized = if self.enabled && apply {
            let request = RecordEnvelope::new(
                "nemotron-cpp-normalization",
                vec![
                    NamedRecord::new("text", MediaRecord::Utf8(text.into())),
                    NamedRecord::new("language_code", MediaRecord::Utf8("en-US".into())),
                ],
            );
            let output = self
                .normalizer
                .as_ref()
                .ok_or_else(|| internal("normalizer unavailable"))?
                .process(&request)
                .map_err(processor_error)?;
            if output.schema_version != request.schema_version
                || output.request_id != request.request_id
                || output.session_id != request.session_id
                || output.sequence != request.sequence
                || output.is_final != request.is_final
            {
                return Err(internal("native normalizer changed envelope identity"));
            }
            if output.records.is_empty() || output.records.len() > self.max_segments {
                return Err(internal("native normalizer returned invalid segment count"));
            }
            let mut bytes = output.records.len() - 1;
            let mut segments = Vec::with_capacity(output.records.len());
            for (index, record) in output.records.into_iter().enumerate() {
                if record.name != format!("normalized_text.{index}") {
                    return Err(internal(
                        "native normalizer returned invalid segment order/type",
                    ));
                }
                let MediaRecord::Utf8(segment) = record.value else {
                    return Err(internal("native normalizer text is not UTF-8"));
                };
                bytes = bytes
                    .checked_add(segment.len())
                    .ok_or_else(|| internal("normalizer length overflow"))?;
                if segment.trim_matches(whitespace).is_empty() || bytes > self.max_output {
                    return Err(internal(
                        "native normalizer returned empty or oversized text",
                    ));
                }
                segments.push(segment);
            }
            self.punctuation
                .replace_all(&segments.join(" "), "$1")
                .into_owned()
        } else {
            text.to_owned()
        };
        let encoding = self
            .tokenizer
            .encode(normalized.as_str(), false)
            .map_err(|e| internal(format!("tokenizer failed: {e}")))?;
        let mut tokens = encoding.get_ids().to_vec();
        tokens.push(131073);
        if tokens.len() > 512 {
            return Err(invalid(
                "native TTS supports at most 512 text tokens including EOS",
            ));
        }
        if tokens.iter().any(|t| *t >= 131075) {
            return Err(invalid("tokenizer returned an out-of-vocabulary token"));
        }
        serde_json::to_string(&serde_json::json!({"text":normalized,"tokens":tokens}))
            .map_err(|e| internal(e.to_string()))
    }
}

unsafe fn cstring<'a>(value: *const c_char) -> Outcome<&'a str> {
    if value.is_null() {
        return Err(invalid("null string argument"));
    }
    CStr::from_ptr(value)
        .to_str()
        .map_err(|_| invalid("argument must be UTF-8"))
}
unsafe fn execute(error: *mut *mut c_char, f: impl FnOnce() -> Outcome<()>) -> i32 {
    if error.is_null() {
        return 1;
    }
    *error = ptr::null_mut();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        result => {
            let (code, message) = match result {
                Ok(Err(e)) => e,
                _ => internal("native text preparation panicked"),
            };
            *error = CString::new(message.replace('\0', "\\0"))
                .unwrap()
                .into_raw();
            code
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn nemotron_text_create(
    config_json: *const c_char,
    tokenizer_path: *const c_char,
    handle: *mut *mut c_void,
    error: *mut *mut c_char,
) -> i32 {
    execute(error, || {
        if handle.is_null() {
            return Err(invalid("null handle output"));
        }
        *handle = ptr::null_mut();
        let config: Option<Config> = if config_json.is_null() {
            None
        } else {
            serde_json::from_str(cstring(config_json)?)
                .map_err(|e| invalid(format!("normalization config: {e}")))?
        };
        *handle = Box::into_raw(Box::new(Preparation::new(
            config,
            cstring(tokenizer_path)?,
        )?))
        .cast();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nemotron_text_prepare(
    handle: *mut c_void,
    text: *const u8,
    text_len: usize,
    apply_normalize: u8,
    output: *mut *mut c_char,
    error: *mut *mut c_char,
) -> i32 {
    execute(error, || {
        if output.is_null() {
            return Err(invalid("null output pointer"));
        }
        *output = ptr::null_mut();
        if handle.is_null() || text.is_null() || text_len > 1_000_000 || apply_normalize > 1 {
            return Err(invalid("invalid preparation arguments"));
        }
        let text = std::str::from_utf8(slice::from_raw_parts(text, text_len))
            .map_err(|_| invalid("text must be UTF-8"))?;
        let payload = (&*handle.cast::<Preparation>()).prepare(text, apply_normalize != 0)?;
        *output = CString::new(payload)
            .map_err(|_| internal("invalid JSON encoding"))?
            .into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nemotron_text_free(value: *mut c_char) {
    if !value.is_null() {
        drop(CString::from_raw(value));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nemotron_text_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(handle.cast::<Preparation>()))
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trimming_matches_python_controls() {
        for value in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}', '\u{a0}'] {
            assert_eq!(
                format!("{value}text{value}").trim_matches(whitespace),
                "text"
            );
        }
        assert_eq!(
            "\u{200b}text\u{200b}".trim_matches(whitespace),
            "\u{200b}text\u{200b}"
        );
    }
    #[test]
    fn invalid_ffi_arguments_report_errors() {
        unsafe {
            let mut error = ptr::null_mut();
            let mut output = ptr::null_mut();
            assert_eq!(
                nemotron_text_prepare(
                    ptr::null_mut(),
                    b"x".as_ptr(),
                    1,
                    1,
                    &mut output,
                    &mut error
                ),
                1
            );
            assert!(output.is_null());
            assert!(!error.is_null());
            nemotron_text_free(error);
        }
    }
}
