// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed Python wrappers for the Rust-owned Switchyard protocol.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Serialize;
use serde_json::{Map, Value};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, Decision, FileSource, FormatId, ImageSource, InstructionBlock,
    LlmRequest, LlmResponseChunk, MediaSource, Message, OutputParams, PreservationMetadata,
    ProviderExtensions, ReasoningParams, ResponseOutput, Role, SamplingParams, StopReason,
    ToolCall, ToolChoice, ToolDefinition, ToolResult, Usage, WireFormat,
};

use crate::py_serde::{value_from_python, value_to_python};

fn serialized_to_python<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let value = serde_json::to_value(value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    value_to_python(py, &value)
}

fn optional_json(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Value>> {
    value.map(value_from_python).transpose()
}

fn json_or_null(py: Python<'_>, value: Option<&Value>) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) => value_to_python(py, value),
        None => Ok(py.None()),
    }
}

/// A routing decision exposed through the protocol's neutral interface.
#[pyclass(name = "Decision", module = "switchyard.libsy.protocol", frozen)]
pub(crate) struct PyDecision {
    inner: Arc<dyn Decision>,
}

impl PyDecision {
    pub(crate) fn new(inner: Arc<dyn Decision>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDecision {
    #[getter]
    fn selected_model(&self) -> String {
        self.inner.selected_model().to_string()
    }

    #[getter]
    fn reasoning(&self) -> Option<String> {
        self.inner.reasoning().map(str::to_owned)
    }

    fn __repr__(&self) -> String {
        format!(
            "Decision(selected_model={:?}, reasoning={:?})",
            self.inner.selected_model(),
            self.inner.reasoning()
        )
    }
}

#[pyclass(
    name = "WireFormat",
    module = "switchyard.libsy.protocol",
    eq,
    frozen,
    from_py_object
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PyWireFormat {
    #[pyo3(name = "OPENAI_CHAT")]
    OpenAiChat,
    #[pyo3(name = "ANTHROPIC_MESSAGES")]
    AnthropicMessages,
    #[pyo3(name = "OPENAI_RESPONSES")]
    OpenAiResponses,
}

impl From<PyWireFormat> for WireFormat {
    fn from(value: PyWireFormat) -> Self {
        match value {
            PyWireFormat::OpenAiChat => Self::OpenAiChat,
            PyWireFormat::AnthropicMessages => Self::AnthropicMessages,
            PyWireFormat::OpenAiResponses => Self::OpenAiResponses,
        }
    }
}

impl From<WireFormat> for PyWireFormat {
    fn from(value: WireFormat) -> Self {
        match value {
            WireFormat::OpenAiChat => Self::OpenAiChat,
            WireFormat::AnthropicMessages => Self::AnthropicMessages,
            WireFormat::OpenAiResponses => Self::OpenAiResponses,
        }
    }
}

#[pymethods]
impl PyWireFormat {
    #[getter]
    fn value(&self) -> &'static str {
        WireFormat::from(*self).as_str()
    }

    fn __str__(&self) -> &'static str {
        self.value()
    }
}

#[pyclass(
    name = "FormatId",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyFormatId {
    inner: FormatId,
}

impl PyFormatId {
    fn from_core(inner: FormatId) -> Self {
        Self { inner }
    }

    fn clone_core(&self) -> FormatId {
        self.inner.clone()
    }
}

#[pymethods]
impl PyFormatId {
    #[new]
    fn new(value: String) -> Self {
        Self::from_core(FormatId::new(value))
    }

    #[classmethod]
    fn known(_cls: &Bound<'_, PyType>, value: PyWireFormat) -> Self {
        Self::from_core(FormatId::known(value.into()))
    }

    #[getter]
    fn value(&self) -> &str {
        self.inner.as_str()
    }

    fn __str__(&self) -> &str {
        self.inner.as_str()
    }

    fn __repr__(&self) -> String {
        format!("FormatId({:?})", self.inner.as_str())
    }
}

#[pyclass(
    name = "Role",
    module = "switchyard.libsy.protocol",
    eq,
    frozen,
    from_py_object
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PyRole {
    #[pyo3(name = "SYSTEM")]
    System,
    #[pyo3(name = "DEVELOPER")]
    Developer,
    #[pyo3(name = "USER")]
    User,
    #[pyo3(name = "ASSISTANT")]
    Assistant,
    #[pyo3(name = "TOOL")]
    Tool,
}

impl From<PyRole> for Role {
    fn from(value: PyRole) -> Self {
        match value {
            PyRole::System => Self::System,
            PyRole::Developer => Self::Developer,
            PyRole::User => Self::User,
            PyRole::Assistant => Self::Assistant,
            PyRole::Tool => Self::Tool,
        }
    }
}

impl From<Role> for PyRole {
    fn from(value: Role) -> Self {
        match value {
            Role::System => Self::System,
            Role::Developer => Self::Developer,
            Role::User => Self::User,
            Role::Assistant => Self::Assistant,
            Role::Tool => Self::Tool,
        }
    }
}

#[pymethods]
impl PyRole {
    #[getter]
    fn value(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[pyclass(
    name = "StopReason",
    module = "switchyard.libsy.protocol",
    eq,
    frozen,
    from_py_object
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PyStopReason {
    #[pyo3(name = "END_TURN")]
    EndTurn,
    #[pyo3(name = "MAX_TOKENS")]
    MaxTokens,
    #[pyo3(name = "TOOL_USE")]
    ToolUse,
    #[pyo3(name = "CONTENT_FILTER")]
    ContentFilter,
    #[pyo3(name = "ERROR")]
    Error,
    #[pyo3(name = "UNKNOWN")]
    Unknown,
}

impl From<PyStopReason> for StopReason {
    fn from(value: PyStopReason) -> Self {
        match value {
            PyStopReason::EndTurn => Self::EndTurn,
            PyStopReason::MaxTokens => Self::MaxTokens,
            PyStopReason::ToolUse => Self::ToolUse,
            PyStopReason::ContentFilter => Self::ContentFilter,
            PyStopReason::Error => Self::Error,
            PyStopReason::Unknown => Self::Unknown,
        }
    }
}

impl From<StopReason> for PyStopReason {
    fn from(value: StopReason) -> Self {
        match value {
            StopReason::EndTurn => Self::EndTurn,
            StopReason::MaxTokens => Self::MaxTokens,
            StopReason::ToolUse => Self::ToolUse,
            StopReason::ContentFilter => Self::ContentFilter,
            StopReason::Error => Self::Error,
            StopReason::Unknown => Self::Unknown,
        }
    }
}

#[pymethods]
impl PyStopReason {
    #[getter]
    fn value(&self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::ToolUse => "tool_use",
            Self::ContentFilter => "content_filter",
            Self::Error => "error",
            Self::Unknown => "unknown",
        }
    }
}

#[pyclass(name = "ImageSource", module = "switchyard.libsy.protocol", frozen)]
pub(crate) enum PyImageSource {
    Url {
        url: String,
        detail: Option<String>,
    },
    Base64 {
        media_type: Option<String>,
        data: String,
    },
    Raw {
        value: Py<PyAny>,
    },
}

impl PyImageSource {
    fn from_core(py: Python<'_>, source: ImageSource) -> PyResult<Self> {
        match source {
            ImageSource::Url { url, detail } => Ok(Self::Url { url, detail }),
            ImageSource::Base64 { media_type, data } => Ok(Self::Base64 { media_type, data }),
            ImageSource::Raw(value) => Ok(Self::Raw {
                value: value_to_python(py, &value)?,
            }),
        }
    }

    fn clone_core(&self, py: Python<'_>) -> PyResult<ImageSource> {
        match self {
            Self::Url { url, detail } => Ok(ImageSource::Url {
                url: url.clone(),
                detail: detail.clone(),
            }),
            Self::Base64 { media_type, data } => Ok(ImageSource::Base64 {
                media_type: media_type.clone(),
                data: data.clone(),
            }),
            Self::Raw { value } => Ok(ImageSource::Raw(value_from_python(value.bind(py))?)),
        }
    }
}

#[pyclass(name = "FileSource", module = "switchyard.libsy.protocol", frozen)]
pub(crate) enum PyFileSource {
    FileId {
        id: String,
    },
    FileData {
        data: String,
        filename: Option<String>,
    },
    Raw {
        value: Py<PyAny>,
    },
}

impl PyFileSource {
    fn from_core(py: Python<'_>, source: FileSource) -> PyResult<Self> {
        match source {
            FileSource::FileId(id) => Ok(Self::FileId { id }),
            FileSource::FileData { data, filename } => Ok(Self::FileData { data, filename }),
            FileSource::Raw(value) => Ok(Self::Raw {
                value: value_to_python(py, &value)?,
            }),
        }
    }

    fn clone_core(&self, py: Python<'_>) -> PyResult<FileSource> {
        match self {
            Self::FileId { id } => Ok(FileSource::FileId(id.clone())),
            Self::FileData { data, filename } => Ok(FileSource::FileData {
                data: data.clone(),
                filename: filename.clone(),
            }),
            Self::Raw { value } => Ok(FileSource::Raw(value_from_python(value.bind(py))?)),
        }
    }
}

#[pyclass(name = "MediaSource", module = "switchyard.libsy.protocol", frozen)]
pub(crate) enum PyMediaSource {
    Url {
        url: String,
        media_type: Option<String>,
    },
    Base64 {
        media_type: Option<String>,
        data: String,
    },
    Raw {
        value: Py<PyAny>,
    },
}

impl PyMediaSource {
    fn from_core(py: Python<'_>, source: MediaSource) -> PyResult<Self> {
        match source {
            MediaSource::Url { url, media_type } => Ok(Self::Url { url, media_type }),
            MediaSource::Base64 { media_type, data } => Ok(Self::Base64 { media_type, data }),
            MediaSource::Raw(value) => Ok(Self::Raw {
                value: value_to_python(py, &value)?,
            }),
        }
    }

    fn clone_core(&self, py: Python<'_>) -> PyResult<MediaSource> {
        match self {
            Self::Url { url, media_type } => Ok(MediaSource::Url {
                url: url.clone(),
                media_type: media_type.clone(),
            }),
            Self::Base64 { media_type, data } => Ok(MediaSource::Base64 {
                media_type: media_type.clone(),
                data: data.clone(),
            }),
            Self::Raw { value } => Ok(MediaSource::Raw(value_from_python(value.bind(py))?)),
        }
    }
}

#[pyclass(
    name = "ToolCall",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyToolCall {
    inner: ToolCall,
}

impl PyToolCall {
    fn from_core(inner: ToolCall) -> Self {
        Self { inner }
    }

    fn clone_core(&self) -> ToolCall {
        self.inner.clone()
    }
}

#[pymethods]
impl PyToolCall {
    #[new]
    fn new(id: String, name: String, arguments: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_core(ToolCall {
            id,
            name,
            arguments: value_from_python(arguments)?,
        }))
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &self.inner.arguments)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ToolResult",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyToolResult {
    inner: ToolResult,
}

impl PyToolResult {
    fn from_core(inner: ToolResult) -> Self {
        Self { inner }
    }

    fn clone_core(&self) -> ToolResult {
        self.inner.clone()
    }
}

#[pymethods]
impl PyToolResult {
    #[new]
    #[pyo3(signature = (tool_call_id, content, *, is_error=None))]
    fn new(
        py: Python<'_>,
        tool_call_id: String,
        content: Vec<Py<PyContentBlock>>,
        is_error: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(ToolResult {
            tool_call_id,
            content: content_blocks_to_core(py, content)?,
            is_error,
        }))
    }

    #[getter]
    fn tool_call_id(&self) -> &str {
        &self.inner.tool_call_id
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        content_blocks_from_core(py, self.inner.content.clone())
    }

    #[getter]
    fn is_error(&self) -> Option<bool> {
        self.inner.is_error
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(name = "ContentBlock", module = "switchyard.libsy.protocol", frozen)]
pub(crate) enum PyContentBlock {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    Image {
        source: Py<PyImageSource>,
    },
    Audio {
        source: Py<PyMediaSource>,
    },
    Video {
        source: Py<PyMediaSource>,
    },
    File {
        source: Py<PyFileSource>,
    },
    #[pyo3(name = "ToolCallBlock")]
    ToolCall {
        tool_call: Py<PyToolCall>,
    },
    #[pyo3(name = "ToolResultBlock")]
    ToolResult {
        tool_result: Py<PyToolResult>,
    },
    Refusal {
        text: String,
    },
    Unknown {
        provider: Py<PyFormatId>,
        raw: Py<PyAny>,
    },
}

impl PyContentBlock {
    fn from_core(py: Python<'_>, block: ContentBlock) -> PyResult<Self> {
        match block {
            ContentBlock::Text { text } => Ok(Self::Text { text }),
            ContentBlock::Reasoning { text, signature } => Ok(Self::Reasoning { text, signature }),
            ContentBlock::Image { source } => Ok(Self::Image {
                source: PyImageSource::from_core(py, source)?
                    .into_pyobject(py)?
                    .unbind(),
            }),
            ContentBlock::Audio { source } => Ok(Self::Audio {
                source: PyMediaSource::from_core(py, source)?
                    .into_pyobject(py)?
                    .unbind(),
            }),
            ContentBlock::Video { source } => Ok(Self::Video {
                source: PyMediaSource::from_core(py, source)?
                    .into_pyobject(py)?
                    .unbind(),
            }),
            ContentBlock::File { source } => Ok(Self::File {
                source: PyFileSource::from_core(py, source)?
                    .into_pyobject(py)?
                    .unbind(),
            }),
            ContentBlock::ToolCall(tool_call) => Ok(Self::ToolCall {
                tool_call: Py::new(py, PyToolCall::from_core(tool_call))?,
            }),
            ContentBlock::ToolResult(tool_result) => Ok(Self::ToolResult {
                tool_result: Py::new(py, PyToolResult::from_core(tool_result))?,
            }),
            ContentBlock::Refusal { text } => Ok(Self::Refusal { text }),
            ContentBlock::Unknown { provider, raw } => Ok(Self::Unknown {
                provider: Py::new(py, PyFormatId::from_core(provider))?,
                raw: value_to_python(py, &raw)?,
            }),
        }
    }

    fn clone_core(&self, py: Python<'_>) -> PyResult<ContentBlock> {
        match self {
            Self::Text { text } => Ok(ContentBlock::Text { text: text.clone() }),
            Self::Reasoning { text, signature } => Ok(ContentBlock::Reasoning {
                text: text.clone(),
                signature: signature.clone(),
            }),
            Self::Image { source } => Ok(ContentBlock::Image {
                source: source.borrow(py).clone_core(py)?,
            }),
            Self::Audio { source } => Ok(ContentBlock::Audio {
                source: source.borrow(py).clone_core(py)?,
            }),
            Self::Video { source } => Ok(ContentBlock::Video {
                source: source.borrow(py).clone_core(py)?,
            }),
            Self::File { source } => Ok(ContentBlock::File {
                source: source.borrow(py).clone_core(py)?,
            }),
            Self::ToolCall { tool_call } => {
                Ok(ContentBlock::ToolCall(tool_call.borrow(py).clone_core()))
            }
            Self::ToolResult { tool_result } => Ok(ContentBlock::ToolResult(
                tool_result.borrow(py).clone_core(),
            )),
            Self::Refusal { text } => Ok(ContentBlock::Refusal { text: text.clone() }),
            Self::Unknown { provider, raw } => Ok(ContentBlock::Unknown {
                provider: provider.borrow(py).clone_core(),
                raw: value_from_python(raw.bind(py))?,
            }),
        }
    }
}

fn content_blocks_to_core(
    py: Python<'_>,
    content: Vec<Py<PyContentBlock>>,
) -> PyResult<Vec<ContentBlock>> {
    content
        .into_iter()
        .map(|block| block.borrow(py).clone_core(py))
        .collect()
}

fn content_blocks_from_core(
    py: Python<'_>,
    content: Vec<ContentBlock>,
) -> PyResult<Vec<Py<PyAny>>> {
    content
        .into_iter()
        .map(|block| {
            PyContentBlock::from_core(py, block)?
                .into_pyobject(py)
                .map(|value| value.unbind().into_any())
        })
        .collect()
}

#[pyclass(
    name = "InstructionBlock",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyInstructionBlock {
    inner: InstructionBlock,
}

impl PyInstructionBlock {
    fn from_core(inner: InstructionBlock) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInstructionBlock {
    #[new]
    fn new(py: Python<'_>, role: PyRole, content: Vec<Py<PyContentBlock>>) -> PyResult<Self> {
        Ok(Self::from_core(InstructionBlock {
            role: role.into(),
            content: content_blocks_to_core(py, content)?,
        }))
    }

    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        content_blocks_from_core(py, self.inner.content.clone())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "Message",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMessage {
    inner: Message,
}

impl PyMessage {
    fn from_core(inner: Message) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(py: Python<'_>, role: PyRole, content: Vec<Py<PyContentBlock>>) -> PyResult<Self> {
        Ok(Self::from_core(Message {
            role: role.into(),
            content: content_blocks_to_core(py, content)?,
        }))
    }

    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        content_blocks_from_core(py, self.inner.content.clone())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ToolDefinition",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyToolDefinition {
    inner: ToolDefinition,
}

impl PyToolDefinition {
    fn from_core(inner: ToolDefinition) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyToolDefinition {
    #[new]
    #[pyo3(signature = (name, parameters, *, description=None, strict=None))]
    fn new(
        name: String,
        parameters: &Bound<'_, PyAny>,
        description: Option<String>,
        strict: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(ToolDefinition {
            name,
            description,
            parameters: value_from_python(parameters)?,
            strict,
        }))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &self.inner.parameters)
    }

    #[getter]
    fn strict(&self) -> Option<bool> {
        self.inner.strict
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ToolChoice",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyToolChoice {
    inner: ToolChoice,
}

impl PyToolChoice {
    fn from_core(inner: ToolChoice) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyToolChoice {
    #[classmethod]
    fn auto(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_core(ToolChoice::Auto)
    }

    #[classmethod]
    fn required(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_core(ToolChoice::Required)
    }

    #[classmethod]
    fn none(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_core(ToolChoice::None)
    }

    #[classmethod]
    fn tool(_cls: &Bound<'_, PyType>, name: String) -> Self {
        Self::from_core(ToolChoice::Tool { name })
    }

    #[classmethod]
    fn raw(_cls: &Bound<'_, PyType>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_core(ToolChoice::Raw(value_from_python(value)?)))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ToolChoice::Auto => "auto",
            ToolChoice::Required => "required",
            ToolChoice::None => "none",
            ToolChoice::Tool { .. } => "tool",
            ToolChoice::Raw(_) => "raw",
        }
    }

    #[getter]
    fn name(&self) -> Option<String> {
        match &self.inner {
            ToolChoice::Tool { name } => Some(name.clone()),
            _ => None,
        }
    }

    #[getter]
    fn raw_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ToolChoice::Raw(value) => value_to_python(py, value),
            _ => Ok(py.None()),
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "SamplingParams",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PySamplingParams {
    inner: SamplingParams,
}

impl PySamplingParams {
    fn from_core(inner: SamplingParams) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySamplingParams {
    #[new]
    #[pyo3(signature = (*, temperature=None, top_p=None, top_k=None))]
    fn new(temperature: Option<f64>, top_p: Option<f64>, top_k: Option<i64>) -> Self {
        Self::from_core(SamplingParams {
            temperature,
            top_p,
            top_k,
        })
    }

    #[getter]
    fn temperature(&self) -> Option<f64> {
        self.inner.temperature
    }

    #[getter]
    fn top_p(&self) -> Option<f64> {
        self.inner.top_p
    }

    #[getter]
    fn top_k(&self) -> Option<i64> {
        self.inner.top_k
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "OutputParams",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyOutputParams {
    inner: OutputParams,
}

impl PyOutputParams {
    fn from_core(inner: OutputParams) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOutputParams {
    #[new]
    #[pyo3(signature = (*, max_output_tokens=None, response_format=None))]
    fn new(
        max_output_tokens: Option<u64>,
        response_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(OutputParams {
            max_output_tokens,
            response_format: optional_json(response_format)?,
        }))
    }

    #[getter]
    fn max_output_tokens(&self) -> Option<u64> {
        self.inner.max_output_tokens
    }

    #[getter]
    fn response_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_or_null(py, self.inner.response_format.as_ref())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ReasoningParams",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyReasoningParams {
    inner: ReasoningParams,
}

impl PyReasoningParams {
    fn from_core(inner: ReasoningParams) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyReasoningParams {
    #[new]
    #[pyo3(signature = (*, effort=None, raw=None))]
    fn new(effort: Option<String>, raw: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self::from_core(ReasoningParams {
            effort,
            raw: optional_json(raw)?,
        }))
    }

    #[getter]
    fn effort(&self) -> Option<String> {
        self.inner.effort.clone()
    }

    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_or_null(py, self.inner.raw.as_ref())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ProviderExtensions",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyProviderExtensions {
    inner: ProviderExtensions,
}

impl PyProviderExtensions {
    fn from_core(inner: ProviderExtensions) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyProviderExtensions {
    #[new]
    #[pyo3(signature = (fields=None))]
    fn new(fields: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let fields = match optional_json(fields)? {
            None => Map::new(),
            Some(Value::Object(fields)) => fields,
            Some(_) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "fields must be a mapping",
                ))
            }
        };
        Ok(Self::from_core(ProviderExtensions { fields }))
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &Value::Object(self.inner.fields.clone()))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "PreservationMetadata",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyPreservationMetadata {
    inner: PreservationMetadata,
}

impl PyPreservationMetadata {
    fn from_core(inner: PreservationMetadata) -> Self {
        Self { inner }
    }
}

#[pyclass(
    name = "LlmRequest",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyLlmRequest {
    inner: LlmRequest,
}

impl PyLlmRequest {
    pub(crate) fn from_core(inner: LlmRequest) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> LlmRequest {
        self.inner.clone()
    }
}

#[pymethods]
impl PyLlmRequest {
    #[new]
    #[pyo3(signature = (*, model=None, instructions=None, messages=None, tools=None, tool_choice=None, sampling=None, output=None, reasoning=None, stream=false, extensions=None, preservation=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: Option<String>,
        instructions: Option<Vec<Py<PyInstructionBlock>>>,
        messages: Option<Vec<Py<PyMessage>>>,
        tools: Option<Vec<Py<PyToolDefinition>>>,
        tool_choice: Option<PyRef<'_, PyToolChoice>>,
        sampling: Option<PyRef<'_, PySamplingParams>>,
        output: Option<PyRef<'_, PyOutputParams>>,
        reasoning: Option<PyRef<'_, PyReasoningParams>>,
        stream: bool,
        extensions: Option<PyRef<'_, PyProviderExtensions>>,
        preservation: Option<PyRef<'_, PyPreservationMetadata>>,
    ) -> Self {
        Self::from_core(LlmRequest {
            model,
            instructions: instructions
                .unwrap_or_default()
                .into_iter()
                .map(|item| item.borrow(py).inner.clone())
                .collect(),
            messages: messages
                .unwrap_or_default()
                .into_iter()
                .map(|item| item.borrow(py).inner.clone())
                .collect(),
            tools: tools
                .unwrap_or_default()
                .into_iter()
                .map(|item| item.borrow(py).inner.clone())
                .collect(),
            tool_choice: tool_choice.map(|value| value.inner.clone()),
            sampling: sampling
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
            output: output.map(|value| value.inner.clone()).unwrap_or_default(),
            reasoning: reasoning
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
            stream,
            extensions: extensions
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
            preservation: preservation
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
        })
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyInstructionBlock>>> {
        self.inner
            .instructions
            .iter()
            .cloned()
            .map(|value| Py::new(py, PyInstructionBlock::from_core(value)))
            .collect()
    }

    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMessage>>> {
        self.inner
            .messages
            .iter()
            .cloned()
            .map(|value| Py::new(py, PyMessage::from_core(value)))
            .collect()
    }

    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Vec<Py<PyToolDefinition>>> {
        self.inner
            .tools
            .iter()
            .cloned()
            .map(|value| Py::new(py, PyToolDefinition::from_core(value)))
            .collect()
    }

    #[getter]
    fn tool_choice(&self, py: Python<'_>) -> PyResult<Option<Py<PyToolChoice>>> {
        self.inner
            .tool_choice
            .clone()
            .map(|value| Py::new(py, PyToolChoice::from_core(value)))
            .transpose()
    }

    #[getter]
    fn sampling(&self, py: Python<'_>) -> PyResult<Py<PySamplingParams>> {
        Py::new(py, PySamplingParams::from_core(self.inner.sampling.clone()))
    }

    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyOutputParams>> {
        Py::new(py, PyOutputParams::from_core(self.inner.output.clone()))
    }

    #[getter]
    fn reasoning(&self, py: Python<'_>) -> PyResult<Py<PyReasoningParams>> {
        Py::new(
            py,
            PyReasoningParams::from_core(self.inner.reasoning.clone()),
        )
    }

    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }

    #[getter]
    fn extensions(&self, py: Python<'_>) -> PyResult<Py<PyProviderExtensions>> {
        Py::new(
            py,
            PyProviderExtensions::from_core(self.inner.extensions.clone()),
        )
    }

    #[getter]
    fn preservation(&self, py: Python<'_>) -> PyResult<Py<PyPreservationMetadata>> {
        Py::new(
            py,
            PyPreservationMetadata::from_core(self.inner.preservation.clone()),
        )
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!("LlmRequest(model={:?})", self.inner.model)
    }
}

#[pyclass(
    name = "Usage",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyUsage {
    inner: Usage,
}

impl PyUsage {
    fn from_core(inner: Usage) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyUsage {
    #[new]
    #[pyo3(signature = (*, input_tokens=None, output_tokens=None, total_tokens=None, reasoning_tokens=None))]
    fn new(
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        total_tokens: Option<u64>,
        reasoning_tokens: Option<u64>,
    ) -> Self {
        Self::from_core(Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            reasoning_tokens,
        })
    }

    #[getter]
    fn input_tokens(&self) -> Option<u64> {
        self.inner.input_tokens
    }

    #[getter]
    fn output_tokens(&self) -> Option<u64> {
        self.inner.output_tokens
    }

    #[getter]
    fn total_tokens(&self) -> Option<u64> {
        self.inner.total_tokens
    }

    #[getter]
    fn reasoning_tokens(&self) -> Option<u64> {
        self.inner.reasoning_tokens
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "ResponseOutput",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyResponseOutput {
    inner: ResponseOutput,
}

impl PyResponseOutput {
    fn from_core(inner: ResponseOutput) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyResponseOutput {
    #[new]
    #[pyo3(signature = (role, content, *, stop_reason=None))]
    fn new(
        py: Python<'_>,
        role: PyRole,
        content: Vec<Py<PyContentBlock>>,
        stop_reason: Option<PyStopReason>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(ResponseOutput {
            role: role.into(),
            content: content_blocks_to_core(py, content)?,
            stop_reason: stop_reason.map(Into::into),
        }))
    }

    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        content_blocks_from_core(py, self.inner.content.clone())
    }

    #[getter]
    fn stop_reason(&self) -> Option<PyStopReason> {
        self.inner.stop_reason.map(Into::into)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}

#[pyclass(
    name = "AggLlmResponse",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyAggLlmResponse {
    inner: AggLlmResponse,
}

impl PyAggLlmResponse {
    pub(crate) fn from_core(inner: AggLlmResponse) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> AggLlmResponse {
        self.inner.clone()
    }
}

#[pymethods]
impl PyAggLlmResponse {
    #[new]
    #[pyo3(signature = (*, id=None, model=None, outputs=None, usage=None, extensions=None, preservation=None))]
    fn new(
        py: Python<'_>,
        id: Option<String>,
        model: Option<String>,
        outputs: Option<Vec<Py<PyResponseOutput>>>,
        usage: Option<PyRef<'_, PyUsage>>,
        extensions: Option<PyRef<'_, PyProviderExtensions>>,
        preservation: Option<PyRef<'_, PyPreservationMetadata>>,
    ) -> Self {
        Self::from_core(AggLlmResponse {
            id,
            model,
            outputs: outputs
                .unwrap_or_default()
                .into_iter()
                .map(|value| value.borrow(py).inner.clone())
                .collect(),
            usage: usage.map(|value| value.inner.clone()).unwrap_or_default(),
            extensions: extensions
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
            preservation: preservation
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
        })
    }

    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    fn outputs(&self, py: Python<'_>) -> PyResult<Vec<Py<PyResponseOutput>>> {
        self.inner
            .outputs
            .iter()
            .cloned()
            .map(|value| Py::new(py, PyResponseOutput::from_core(value)))
            .collect()
    }

    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Py<PyUsage>> {
        Py::new(py, PyUsage::from_core(self.inner.usage.clone()))
    }

    #[getter]
    fn extensions(&self, py: Python<'_>) -> PyResult<Py<PyProviderExtensions>> {
        Py::new(
            py,
            PyProviderExtensions::from_core(self.inner.extensions.clone()),
        )
    }

    #[getter]
    fn preservation(&self, py: Python<'_>) -> PyResult<Py<PyPreservationMetadata>> {
        Py::new(
            py,
            PyPreservationMetadata::from_core(self.inner.preservation.clone()),
        )
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!("AggLlmResponse(model={:?})", self.inner.model)
    }
}

#[pyclass(
    name = "LlmResponseChunk",
    module = "switchyard.libsy.protocol",
    frozen
)]
pub(crate) enum PyLlmResponseChunk {
    MessageStart {
        id: Option<String>,
        model: Option<String>,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ReasoningDelta {
        index: usize,
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    #[pyo3(name = "UsageUpdate")]
    Usage {
        usage: Py<PyUsage>,
    },
    MessageStop {
        reason: Option<String>,
    },
    Error {
        message: String,
    },
}

impl PyLlmResponseChunk {
    pub(crate) fn from_core(py: Python<'_>, inner: LlmResponseChunk) -> PyResult<Self> {
        match inner {
            LlmResponseChunk::MessageStart { id, model } => Ok(Self::MessageStart { id, model }),
            LlmResponseChunk::TextDelta { index, text } => Ok(Self::TextDelta { index, text }),
            LlmResponseChunk::ReasoningDelta { index, text } => {
                Ok(Self::ReasoningDelta { index, text })
            }
            LlmResponseChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => Ok(Self::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            }),
            LlmResponseChunk::Usage(usage) => Ok(Self::Usage {
                usage: Py::new(py, PyUsage::from_core(usage))?,
            }),
            LlmResponseChunk::MessageStop { reason } => Ok(Self::MessageStop { reason }),
            LlmResponseChunk::Error { message } => Ok(Self::Error { message }),
        }
    }

    pub(crate) fn clone_core(&self, py: Python<'_>) -> LlmResponseChunk {
        match self {
            Self::MessageStart { id, model } => LlmResponseChunk::MessageStart {
                id: id.clone(),
                model: model.clone(),
            },
            Self::TextDelta { index, text } => LlmResponseChunk::TextDelta {
                index: *index,
                text: text.clone(),
            },
            Self::ReasoningDelta { index, text } => LlmResponseChunk::ReasoningDelta {
                index: *index,
                text: text.clone(),
            },
            Self::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => LlmResponseChunk::ToolCallDelta {
                index: *index,
                id: id.clone(),
                name: name.clone(),
                arguments_delta: arguments_delta.clone(),
            },
            Self::Usage { usage } => LlmResponseChunk::Usage(usage.borrow(py).inner.clone()),
            Self::MessageStop { reason } => LlmResponseChunk::MessageStop {
                reason: reason.clone(),
            },
            Self::Error { message } => LlmResponseChunk::Error {
                message: message.clone(),
            },
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDecision>()?;
    module.add_class::<PyWireFormat>()?;
    module.add_class::<PyFormatId>()?;
    module.add_class::<PyRole>()?;
    module.add_class::<PyStopReason>()?;
    module.add_class::<PyImageSource>()?;
    module.add_class::<PyFileSource>()?;
    module.add_class::<PyMediaSource>()?;
    module.add_class::<PyToolCall>()?;
    module.add_class::<PyToolResult>()?;
    module.add_class::<PyContentBlock>()?;
    module.add_class::<PyInstructionBlock>()?;
    module.add_class::<PyMessage>()?;
    module.add_class::<PyToolDefinition>()?;
    module.add_class::<PyToolChoice>()?;
    module.add_class::<PySamplingParams>()?;
    module.add_class::<PyOutputParams>()?;
    module.add_class::<PyReasoningParams>()?;
    module.add_class::<PyProviderExtensions>()?;
    module.add_class::<PyPreservationMetadata>()?;
    module.add_class::<PyLlmRequest>()?;
    module.add_class::<PyUsage>()?;
    module.add_class::<PyResponseOutput>()?;
    module.add_class::<PyAggLlmResponse>()?;
    module.add_class::<PyLlmResponseChunk>()?;
    Ok(())
}

fn preservation_map(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<BTreeMap<FormatId, Value>> {
    let Some(value) = optional_json(value)? else {
        return Ok(BTreeMap::new());
    };
    let Value::Object(value) = value else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be a mapping"
        )));
    };
    Ok(value
        .into_iter()
        .map(|(key, value)| (FormatId::new(key), value))
        .collect())
}

fn preservation_map_to_value(value: &BTreeMap<FormatId, Value>) -> Value {
    Value::Object(
        value
            .iter()
            .map(|(key, value)| (key.as_str().to_string(), value.clone()))
            .collect(),
    )
}

#[pymethods]
impl PyPreservationMetadata {
    #[new]
    #[pyo3(signature = (*, requests=None, responses=None))]
    fn new(
        requests: Option<&Bound<'_, PyAny>>,
        responses: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(PreservationMetadata {
            requests: preservation_map(requests, "requests")?,
            responses: preservation_map(responses, "responses")?,
        }))
    }

    #[getter]
    fn requests(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &preservation_map_to_value(&self.inner.requests))
    }

    #[getter]
    fn responses(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &preservation_map_to_value(&self.inner.responses))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialized_to_python(py, &self.inner)
    }
}
