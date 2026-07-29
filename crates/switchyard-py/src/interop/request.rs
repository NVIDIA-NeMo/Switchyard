// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private adapter for Rust-owned chat request values.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_components::{ChatRequest, ChatRequestType};

use crate::py_serde::{value_from_python, value_to_python};

/// Converts a public Python request into the native request value.
pub(crate) fn request_from_python(value: &Bound<'_, PyAny>) -> PyResult<ChatRequest> {
    let request_type = request_type_from_python(&value.getattr("request_type")?)?;
    let body = value_from_python(&value.getattr("_body")?)?;
    Ok(match request_type {
        ChatRequestType::OpenAiChat => ChatRequest::openai_chat(body),
        ChatRequestType::OpenAiResponses => ChatRequest::openai_responses(body),
        ChatRequestType::Anthropic => ChatRequest::anthropic(body),
    })
}

/// Converts a native request into the public Python request value.
pub(crate) fn request_to_python(py: Python<'_>, request: ChatRequest) -> PyResult<Py<PyAny>> {
    let request_type = request.request_type();
    let body = value_to_python(py, request.body())?;
    let factory = match request_type {
        ChatRequestType::OpenAiChat => "openai_chat",
        ChatRequestType::OpenAiResponses => "openai_responses",
        ChatRequestType::Anthropic => "anthropic",
    };
    py.import("switchyard_rust.core")?
        .getattr("ChatRequest")?
        .call_method1(factory, (body,))
        .map(Bound::unbind)
}

pub(crate) fn request_type_from_python(value: &Bound<'_, PyAny>) -> PyResult<ChatRequestType> {
    let raw = if let Ok(value_attr) = value.getattr("value") {
        value_attr.extract::<String>()?
    } else {
        value.extract::<String>()?
    };
    match raw.as_str() {
        "openai_chat" => Ok(ChatRequestType::OpenAiChat),
        "openai_responses" => Ok(ChatRequestType::OpenAiResponses),
        "anthropic" | "anthropic_messages" => Ok(ChatRequestType::Anthropic),
        _ => Err(PyValueError::new_err(format!(
            "Unknown request type: {raw:?}"
        ))),
    }
}

pub(crate) fn request_type_variant_name(request_type: ChatRequestType) -> &'static str {
    match request_type {
        ChatRequestType::OpenAiChat => "OPENAI_CHAT",
        ChatRequestType::OpenAiResponses => "OPENAI_RESPONSES",
        ChatRequestType::Anthropic => "ANTHROPIC",
    }
}

pub(crate) fn request_type_object(
    py: Python<'_>,
    request_type: ChatRequestType,
) -> PyResult<Py<PyAny>> {
    py.import("switchyard_rust.core")?
        .getattr("ChatRequestType")?
        .getattr(request_type_variant_name(request_type))
        .map(Bound::unbind)
}
