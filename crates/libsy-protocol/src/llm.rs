// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral conversation types shared by routing, clients, and translation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::format::FormatId;

/// Actor role normalized across provider APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// Instruction content separated from normal conversation messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstructionBlock {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// One normalized conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a text-only message for the given role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Concatenates text-like content blocks when the message has any.
    pub fn text_content(&self, separator: &str) -> Option<String> {
        let parts = self
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                ContentBlock::Refusal { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(separator))
        }
    }
}

/// Normalized content block variants carried by messages and tool results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    Image {
        source: ImageSource,
    },
    Audio {
        source: MediaSource,
    },
    Video {
        source: MediaSource,
    },
    File {
        source: FileSource,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Refusal {
        text: String,
    },
    Unknown {
        provider: FormatId,
        raw: Value,
    },
}

/// Image payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImageSource {
    Url {
        url: String,
        detail: Option<String>,
    },
    Base64 {
        media_type: Option<String>,
        data: String,
    },
    Raw(Value),
}

/// File payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FileSource {
    FileId(String),
    FileData {
        data: String,
        filename: Option<String>,
    },
    Raw(Value),
}

/// Audio and video payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MediaSource {
    Url {
        url: String,
        media_type: Option<String>,
    },
    Base64 {
        media_type: Option<String>,
        data: String,
    },
    Raw(Value),
}

/// Normalized assistant tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Normalized tool result message content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: Vec<ContentBlock>,
    pub is_error: Option<bool>,
}

/// Normalized tool definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

/// Normalized tool choice policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Tool { name: String },
    Raw(Value),
}

/// Provider sampling parameters with common cross-provider names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
}

/// Output budget and structured-output options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputParams {
    pub max_output_tokens: Option<u64>,
    pub response_format: Option<Value>,
}

/// Provider reasoning controls preserved by translation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningParams {
    pub effort: Option<String>,
    pub raw: Option<Value>,
}

/// Provider-specific fields that do not have first-class conversation fields.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderExtensions {
    pub fields: Map<String, Value>,
}

/// Exact source payloads retained for lossless round trips.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PreservationMetadata {
    pub requests: BTreeMap<FormatId, Value>,
    pub responses: BTreeMap<FormatId, Value>,
}

/// Normalized request representation shared by Switchyard components.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: Option<String>,
    pub instructions: Vec<InstructionBlock>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub sampling: SamplingParams,
    pub output: OutputParams,
    pub reasoning: ReasoningParams,
    pub stream: bool,
    pub extensions: ProviderExtensions,
    pub preservation: PreservationMetadata,
}

/// Normalized token usage counts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

/// Normalized reason a model stopped producing output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    ContentFilter,
    Error,
    Unknown,
}

/// One assistant output item in a normalized response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutput {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
}

/// Normalized response representation shared by Switchyard components.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub outputs: Vec<ResponseOutput>,
    pub usage: Usage,
    pub extensions: ProviderExtensions,
    pub preservation: PreservationMetadata,
}

impl LlmResponse {
    /// Returns the first output item when a response has any output.
    pub fn first_output(&self) -> Option<&ResponseOutput> {
        self.outputs.first()
    }
}
