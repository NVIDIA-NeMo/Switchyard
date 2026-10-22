// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C bridge to Switchyard's request and response codecs.

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Deserialize, Serialize};
use switchyard_protocol::{LlmRequest, ModelId, WireFormat};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Returns the compatibility generation for C signatures and JSON payloads.
#[unsafe(no_mangle)]
pub extern "C" fn sy_abi_version() -> u32 {
    1
}

#[derive(Deserialize)]
struct DecodeInput {
    format: WireFormat,
    body: serde_json::Value,
}

#[derive(Deserialize)]
struct EncodeInput {
    format: WireFormat,
    request: LlmRequest,
    model: ModelId,
}

fn error_string(message: String) -> *mut c_char {
    match CString::new(message) {
        Ok(message) => message.into_raw(),
        Err(_) => c"Switchyard error contained a NUL byte"
            .to_owned()
            .into_raw(),
    }
}

// The input must be NUL-terminated; parsed values do not borrow it.
unsafe fn parse_json_input<T: serde::de::DeserializeOwned>(input: *const c_char) -> Result<T> {
    if input.is_null() {
        return Err("missing JSON input".into());
    }
    let input = unsafe { CStr::from_ptr(input) };
    Ok(serde_json::from_slice(input.to_bytes())?)
}

fn encode_json(output: &impl Serialize) -> Result<CString> {
    Ok(CString::new(serde_json::to_vec(output)?)?)
}

/// Decodes provider JSON into a normalized request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse_json_input(input_json) }?;
        *out = encode_json(&switchyard_translation::decode_request(
            input.format,
            &input.body,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Applies the selected model and encodes the request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_encode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let mut input: EncodeInput = unsafe { parse_json_input(input_json) }?;
        switchyard_translation::prepare_request_for_target(&mut input.request, &input.model, None);
        *out = encode_json(&switchyard_translation::encode_request(
            &input.request,
            input.format,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Decodes a buffered provider response.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_response(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing response output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse_json_input(input_json) }?;
        *out = encode_json(&switchyard_translation::decode_aggregated_response(
            &input.body,
            input.format,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Frees a library-owned error or JSON string.
/// # Safety
/// Pass a returned string once, with no remaining readers, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_string_free(string: *mut c_char) {
    if !string.is_null() {
        drop(unsafe { CString::from_raw(string) });
    }
}

#[cfg(test)]
mod tests;
