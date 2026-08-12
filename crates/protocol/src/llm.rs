// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral conversation types shared by routing, clients, and translation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::format::FormatId;

/// Actor role normalized across provider APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System-level instructions.
    System,
    /// Developer-level instructions supported by some provider APIs.
    Developer,
    /// End-user input.
    User,
    /// Model-generated output.
    Assistant,
    /// Tool execution output.
    Tool,
}

/// Instruction content separated from normal conversation messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstructionBlock {
    /// Instruction authority, normally [`Role::System`] or [`Role::Developer`].
    pub role: Role,
    /// Ordered instruction content.
    pub content: Vec<ContentBlock>,
}

/// One normalized conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Actor that produced the message.
    pub role: Role,
    /// Ordered message content.
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text {
        /// Text value.
        text: String,
    },
    /// Model reasoning or thinking content.
    Reasoning {
        /// Reasoning text.
        text: String,
        /// Provider signature used to validate or continue the reasoning block.
        signature: Option<String>,
    },
    /// Image content.
    Image {
        /// Image location or inline payload.
        source: ImageSource,
    },
    /// Audio content.
    Audio {
        /// Audio location or inline payload.
        source: MediaSource,
    },
    /// Video content.
    Video {
        /// Video location or inline payload.
        source: MediaSource,
    },
    /// File content.
    File {
        /// File identifier or inline payload.
        source: FileSource,
    },
    /// Tool invocation requested by the assistant.
    ToolCall(ToolCall),
    /// Result of an earlier tool invocation.
    ToolResult(ToolResult),
    /// Provider refusal content.
    Refusal {
        /// Human-readable refusal text.
        text: String,
    },
    /// Provider block that has no normalized representation.
    Unknown {
        /// Wire format that supplied the block.
        provider: FormatId,
        /// Exact provider block.
        raw: Value,
    },
}

/// Image payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ImageSource {
    /// Image fetched from a URL.
    Url {
        /// Image URL.
        url: String,
        /// Optional provider-specific image detail level.
        detail: Option<String>,
    },
    /// Base64-encoded image bytes.
    Base64 {
        /// MIME type, when supplied.
        media_type: Option<String>,
        /// Base64-encoded bytes.
        data: String,
    },
    /// Provider image source with no normalized representation.
    Raw(Value),
}

/// File payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum FileSource {
    /// Provider-managed file identifier.
    FileId(String),
    /// Inline file payload.
    FileData {
        /// Encoded or textual file data as supplied by the provider format.
        data: String,
        /// Original filename, when supplied.
        filename: Option<String>,
    },
    /// Provider file source with no normalized representation.
    Raw(Value),
}

/// Audio and video payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum MediaSource {
    /// Media fetched from a URL.
    Url {
        /// Media URL.
        url: String,
        /// MIME type, when supplied.
        media_type: Option<String>,
    },
    /// Base64-encoded media bytes.
    Base64 {
        /// MIME type, when supplied.
        media_type: Option<String>,
        /// Base64-encoded bytes.
        data: String,
    },
    /// Provider media source with no normalized representation.
    Raw(Value),
}

/// Normalized assistant tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider tool-call identifier used to pair the result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: Value,
}

/// Normalized tool result message content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Identifier of the [`ToolCall`] this result answers.
    pub tool_call_id: String,
    /// Ordered tool output content.
    pub content: Vec<ContentBlock>,
    /// Whether tool execution failed, when the provider reports it.
    pub is_error: Option<bool>,
}

/// Normalized tool definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name exposed to the model.
    pub name: String,
    /// Human-readable tool description.
    pub description: Option<String>,
    /// JSON Schema describing accepted arguments.
    pub parameters: Value,
    /// Whether the provider should enforce the schema strictly.
    pub strict: Option<bool>,
}

/// Normalized tool choice policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Require at least one tool call.
    Required,
    /// Disallow tool calls.
    None,
    /// Require a specific tool.
    Tool {
        /// Required tool name.
        name: String,
    },
    /// Provider tool-choice value with no normalized representation.
    Raw(Value),
}

/// Provider sampling parameters with common cross-provider names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f64>,
    /// Maximum number of candidate tokens considered at each step.
    pub top_k: Option<i64>,
}

/// Output budget and structured-output options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputParams {
    /// Maximum tokens the model may generate.
    pub max_output_tokens: Option<u64>,
    /// Provider-neutral or provider-specific structured-output configuration.
    pub response_format: Option<Value>,
}

/// Provider reasoning controls preserved by translation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningParams {
    /// Requested reasoning effort or level.
    pub effort: Option<String>,
    /// Provider reasoning controls without a normalized field.
    pub raw: Option<Value>,
}

/// Provider-specific fields that do not have first-class conversation fields.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderExtensions {
    /// Provider fields keyed by their original wire names.
    ///
    /// Codecs use these fields when translating to another format. Exact
    /// same-format replay is controlled separately by [`PreservationMetadata`].
    pub fields: Map<String, Value>,
}

/// Exact source payloads retained for lossless same-format round trips.
///
/// Translation's default preservation policy prefers a stored same-format body
/// over reconstructing one from normalized fields. A caller that mutates the IR
/// must clear the corresponding entry or use a policy with preservation disabled
/// when those mutations must be encoded.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreservationMetadata {
    /// Original request bodies keyed by source format.
    pub requests: BTreeMap<FormatId, Value>,
    /// Original response bodies keyed by source format.
    pub responses: BTreeMap<FormatId, Value>,
}

/// Normalized request representation shared by Switchyard components.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmRequest {
    /// Model currently addressed by the request.
    ///
    /// This initially contains the name supplied by the inbound client. A routing host may
    /// replace it with the selected target before serving the request.
    pub model: Option<String>,
    /// System and developer instructions separated from conversation turns.
    pub instructions: Vec<InstructionBlock>,
    /// Ordered conversation messages.
    pub messages: Vec<Message>,
    /// Tools available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Policy controlling model tool selection.
    pub tool_choice: Option<ToolChoice>,
    /// Common sampling controls.
    pub sampling: SamplingParams,
    /// Output budget and shape controls.
    pub output: OutputParams,
    /// Reasoning controls.
    pub reasoning: ReasoningParams,
    /// Whether the caller requested a streamed response.
    pub stream: bool,
    /// Provider fields without first-class normalized equivalents.
    pub extensions: ProviderExtensions,
    /// Exact provider bodies used by codecs for lossless same-format round trips.
    /// This is separate from a host's optional
    /// [`Request::raw_request`](crate::Request::raw_request).
    pub preservation: PreservationMetadata,
}

/// Normalized token usage counts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Non-cached input tokens. Provider codecs normalize aggregate OpenAI
    /// input counts by subtracting the cache detail fields.
    pub input_tokens: Option<u64>,
    /// Cache-read and cache-creation token detail, when reported.
    #[serde(flatten)]
    pub cache: Option<Box<InputCacheUsage>>,
    /// Generated output tokens, excluding reasoning detail when the provider reports it separately.
    pub output_tokens: Option<u64>,
    /// Provider-reported or codec-computed total token count.
    ///
    /// Codecs normalize OpenAI aggregate input counts before computing totals, so
    /// this may equal non-cached input plus cache detail plus output.
    pub total_tokens: Option<u64>,
    /// Reasoning output tokens, when reported separately.
    pub reasoning_tokens: Option<u64>,
}

/// Optional cache-token detail kept out of the common, cache-free usage allocation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputCacheUsage {
    /// Input tokens read from a provider cache.
    pub cached_input_tokens: Option<u64>,
    /// Input tokens written into a provider cache.
    pub cache_creation_input_tokens: Option<u64>,
}

impl Usage {
    /// Builds cache detail when at least one count is present.
    pub fn cache_details(
        cached_input_tokens: Option<u64>,
        cache_creation_input_tokens: Option<u64>,
    ) -> Option<Box<InputCacheUsage>> {
        if cached_input_tokens.is_none() && cache_creation_input_tokens.is_none() {
            return None;
        }
        Some(Box::new(InputCacheUsage {
            cached_input_tokens,
            cache_creation_input_tokens,
        }))
    }

    /// Returns the cache-read input-token count.
    pub fn cached_input_tokens(&self) -> Option<u64> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.cached_input_tokens)
    }

    /// Returns the cache-creation input-token count.
    pub fn cache_creation_input_tokens(&self) -> Option<u64> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.cache_creation_input_tokens)
    }

    /// Sets the cache-read count, allocating cache detail when needed.
    pub fn set_cached_input_tokens(&mut self, value: u64) {
        self.cache
            .get_or_insert_with(|| Box::new(InputCacheUsage::default()))
            .cached_input_tokens = Some(value);
    }

    /// Sets the cache-creation count, allocating cache detail when needed.
    pub fn set_cache_creation_input_tokens(&mut self, value: u64) {
        self.cache
            .get_or_insert_with(|| Box::new(InputCacheUsage::default()))
            .cache_creation_input_tokens = Some(value);
    }
}

/// Normalized reason a model stopped producing output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model completed its turn normally.
    EndTurn,
    /// Model reached the configured output limit.
    MaxTokens,
    /// Model stopped to request a tool call.
    ToolUse,
    /// Provider safety or content filtering stopped generation.
    ContentFilter,
    /// Generation terminated because of an error.
    Error,
    /// Provider stop reason with no normalized equivalent.
    Unknown,
}

/// One assistant output item in a normalized response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutput {
    /// Actor that produced the output, normally [`Role::Assistant`].
    pub role: Role,
    /// Ordered output content.
    pub content: Vec<ContentBlock>,
    /// Why this output stopped, when known.
    pub stop_reason: Option<StopReason>,
}

/// Normalized, fully-buffered response — the aggregate of a completed generation.
/// This is the terminal form of a streamed [`LlmResponse`](crate::LlmResponse).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AggLlmResponse {
    /// Provider response identifier.
    pub id: Option<String>,
    /// Model reported by the provider.
    pub model: Option<String>,
    /// Ordered response output items.
    pub outputs: Vec<ResponseOutput>,
    /// Normalized token usage.
    pub usage: Usage,
    /// Provider response fields without normalized equivalents.
    pub extensions: ProviderExtensions,
    /// Exact provider bodies retained for lossless round trips.
    pub preservation: PreservationMetadata,
}

impl AggLlmResponse {
    /// Returns the first output item when a response has any output.
    pub fn first_output(&self) -> Option<&ResponseOutput> {
        self.outputs.first()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn serde_uses_python_friendly_dictionary_shapes() -> Result<(), serde_json::Error> {
        let request: LlmRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }]
        }))?;
        assert_eq!(request.messages[0], Message::text(Role::User, "hello"));

        let tool_call = ContentBlock::ToolCall(ToolCall {
            id: "call-1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({"query": "rust"}),
        });
        assert_eq!(
            serde_json::to_value(tool_call)?,
            json!({
                "type": "tool_call",
                "id": "call-1",
                "name": "lookup",
                "arguments": {"query": "rust"}
            })
        );
        Ok(())
    }
}
