// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the C bridge.

use super::*;
use serde_json::{Value, json};

// ABI generations are independent of package releases.
#[test]
fn abi_version_identifies_v1_contract() {
    assert_eq!(sy_abi_version(), 1);
}

fn check(error: *mut c_char) {
    if !error.is_null() {
        panic!("{}", unsafe { CString::from_raw(error) }.to_string_lossy());
    }
}

fn translated(
    call: unsafe extern "C" fn(*const c_char, *mut *mut c_char) -> *mut c_char,
    input: Value,
) -> Value {
    let input = CString::new(input.to_string()).unwrap();
    let mut output = std::ptr::null_mut();
    check(unsafe { call(input.as_ptr(), &mut output) });
    assert!(!output.is_null());
    let result = serde_json::from_slice(unsafe { CStr::from_ptr(output) }.to_bytes());
    unsafe { sy_string_free(output) };
    result.unwrap()
}

// Translation exports return owned JSON strings that the caller releases through the ABI.
#[test]
fn translation_exports_return_owned_json() {
    let request = translated(
        sy_decode_request,
        json!({
            "format": "openai_chat",
            "body": {"model": "route", "messages": [{"role": "user", "content": "hello"}]},
        }),
    );
    assert_eq!(request["messages"][0]["content"][0]["text"], "hello");
    let encoded = translated(
        sy_encode_request,
        json!({
            "format": "openai_chat", "request": request, "model": "selected",
        }),
    );
    assert_eq!(encoded["model"], "selected");
    let response = translated(
        sy_decode_response,
        json!({
            "format": "openai_chat",
            "body": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]},
        }),
    );
    assert_eq!(response["outputs"][0]["content"][0]["text"], "hello");
}
