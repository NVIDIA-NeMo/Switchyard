// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Rust-owned chat request values.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use switchyard_core::{ChatRequest, ChatRequestType};

use crate::errors::py_core_error;
use crate::py_serde::{value_from_python, value_to_python};

#[pyclass(name = "ChatRequestType", frozen, eq, skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PyChatRequestType {
    inner: ChatRequestType,
}

impl PyChatRequestType {
    const fn new(inner: ChatRequestType) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChatRequestType {
    #[classattr]
    const OPENAI_CHAT: Self = Self::new(ChatRequestType::OpenAiChat);

    #[classattr]
    const OPENAI_RESPONSES: Self = Self::new(ChatRequestType::OpenAiResponses);

    #[classattr]
    const ANTHROPIC: Self = Self::new(ChatRequestType::Anthropic);

    #[classattr]
    const GEMINI: Self = Self::new(ChatRequestType::Gemini);

    #[getter]
    fn value(&self) -> &'static str {
        request_type_name(self.inner)
    }

    fn __repr__(&self) -> String {
        format!("ChatRequestType.{}", request_type_variant_name(self.inner))
    }

    fn __str__(&self) -> &'static str {
        request_type_name(self.inner)
    }

    fn __hash__(&self) -> isize {
        match self.inner {
            ChatRequestType::OpenAiChat => 1,
            ChatRequestType::OpenAiResponses => 2,
            ChatRequestType::Anthropic => 3,
            ChatRequestType::Gemini => 4,
        }
    }
}

#[pyclass(name = "ChatRequest")]
#[derive(Debug, PartialEq)]
pub(crate) struct PyChatRequest {
    inner: ChatRequest,
}

impl PyChatRequest {
    pub(crate) fn from_core(inner: ChatRequest) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> ChatRequest {
        self.inner.clone()
    }
}

#[pymethods]
impl PyChatRequest {
    #[classmethod]
    fn openai_chat(_cls: &Bound<'_, PyType>, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: ChatRequest::openai_chat(value_from_python(body)?),
        })
    }

    #[classmethod]
    fn openai_responses(_cls: &Bound<'_, PyType>, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: ChatRequest::openai_responses(value_from_python(body)?),
        })
    }

    #[classmethod]
    fn anthropic(_cls: &Bound<'_, PyType>, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: ChatRequest::anthropic(value_from_python(body)?),
        })
    }

    #[classmethod]
    fn gemini(_cls: &Bound<'_, PyType>, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: ChatRequest::gemini(value_from_python(body)?),
        })
    }

    /// Validates the request at the inbound trust boundary.
    ///
    /// Kept separate from construction so internal re-wraps (e.g. the
    /// translation engine rebuilding a request after format conversion) are
    /// not subject to inbound-only checks. Endpoints call this explicitly on
    /// the client-supplied request; raises ``SwitchyardInvalidRequestError``
    /// for a present-but-empty ``messages`` array.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(py_core_error)
    }

    #[getter]
    fn request_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        request_type_object(py, self.inner.request_type())
    }

    #[getter]
    fn body(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_body(py)
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model().map(str::to_owned)
    }

    fn set_model(&mut self, model: &str) {
        self.inner.set_model(model);
    }

    fn replace_body(&mut self, body: &Bound<'_, PyAny>) -> PyResult<()> {
        let body = value_from_python(body)?;
        self.inner = match self.inner.request_type() {
            ChatRequestType::OpenAiChat => ChatRequest::openai_chat(body),
            ChatRequestType::OpenAiResponses => ChatRequest::openai_responses(body),
            ChatRequestType::Anthropic => ChatRequest::anthropic(body),
            ChatRequestType::Gemini => ChatRequest::gemini(body),
        };
        Ok(())
    }

    fn to_body(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, self.inner.body())
    }

    fn __repr__(&self) -> String {
        match self.inner.model() {
            Some(model) => format!(
                "ChatRequest(request_type='{}', model='{}')",
                request_type_name(self.inner.request_type()),
                model
            ),
            None => format!(
                "ChatRequest(request_type='{}')",
                request_type_name(self.inner.request_type())
            ),
        }
    }
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
        "gemini" | "gemini_generate_content" => Ok(ChatRequestType::Gemini),
        _ => Err(PyValueError::new_err(format!(
            "Unknown request type: {raw:?}"
        ))),
    }
}

pub(crate) fn request_type_name(request_type: ChatRequestType) -> &'static str {
    match request_type {
        ChatRequestType::OpenAiChat => "openai_chat",
        ChatRequestType::OpenAiResponses => "openai_responses",
        ChatRequestType::Anthropic => "anthropic",
        ChatRequestType::Gemini => "gemini",
    }
}

pub(crate) fn request_type_variant_name(request_type: ChatRequestType) -> &'static str {
    match request_type {
        ChatRequestType::OpenAiChat => "OPENAI_CHAT",
        ChatRequestType::OpenAiResponses => "OPENAI_RESPONSES",
        ChatRequestType::Anthropic => "ANTHROPIC",
        ChatRequestType::Gemini => "GEMINI",
    }
}

pub(crate) fn request_type_object(
    py: Python<'_>,
    request_type: ChatRequestType,
) -> PyResult<Py<PyAny>> {
    py.get_type::<PyChatRequestType>()
        .getattr(request_type_variant_name(request_type))
        .map(Bound::unbind)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyChatRequestType>()?;
    module.add_class::<PyChatRequest>()?;
    Ok(())
}
