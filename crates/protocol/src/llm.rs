// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral conversation types shared by routing, clients, and translation.
//!
//! Debug output redacts MCP URLs and authorization tokens and hides raw provider tool definitions,
//! extensions, and preserved bodies. Conversation content, including unknown provider
//! blocks, remains visible and may contain secrets. Serialization retains the stored values.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::format::FormatId;

/// Actor role normalized across provider APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, strum_macros::Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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
        /// Structured reasoning details, such as an encrypted `{ "type":
        /// "reasoning.encrypted", "data": "..." }` object, replayed without modification.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        details: Vec<Value>,
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
    /// MCP activity executed by the backend, rather than by the calling agent.
    Mcp(McpContent),
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

/// A tool available to the model, in catalog order.
#[derive(Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    /// A remote MCP server and the tools it exposes.
    Mcp(McpToolset),
    /// A definition without a supported neutral representation.
    Provider {
        /// Source wire format; this definition is only replayed in that format.
        provider: FormatId,
        /// Complete provider definition.
        raw: Value,
    },
    /// An agent-executed function. Its JSON retains the existing flat definition shape.
    #[serde(untagged)]
    Function(ToolDefinition),
}

impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Keep these variants and the conversion below in sync with Tool.
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum TaggedTool {
            Mcp(McpToolset),
            Provider { provider: FormatId, raw: Value },
            Function(ToolDefinition),
        }

        let value = Value::deserialize(deserializer)?;
        // Only definitions without a type use the legacy function shape. A malformed
        // tagged definition must fail instead of falling back to a function.
        if value.get("type").is_none() {
            return serde_json::from_value(value)
                .map(Self::Function)
                .map_err(serde::de::Error::custom);
        }
        let tool = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(match tool {
            TaggedTool::Mcp(mcp) => Self::Mcp(mcp),
            TaggedTool::Provider { provider, raw } => Self::Provider { provider, raw },
            TaggedTool::Function(function) => Self::Function(function),
        })
    }
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function(definition) => f.debug_tuple("Function").field(definition).finish(),
            Self::Mcp(mcp) => f.debug_tuple("Mcp").field(mcp).finish(),
            // Raw provider definitions can contain credentials without normalized fields.
            Self::Provider { provider, .. } => f
                .debug_struct("Provider")
                .field("provider", provider)
                .finish_non_exhaustive(),
        }
    }
}

impl From<ToolDefinition> for Tool {
    fn from(definition: ToolDefinition) -> Self {
        Self::Function(definition)
    }
}

impl Tool {
    /// Returns the function definition, if this is an agent-executed function.
    pub fn function(&self) -> Option<&ToolDefinition> {
        match self {
            Self::Function(definition) => Some(definition),
            _ => None,
        }
    }
}

/// Configuration for a URL-based remote MCP toolset.
///
/// This type covers the supported neutral settings. Definitions with additional settings,
/// such as Responses headers, server descriptions, or conditional approval rules, belong
/// in [`Tool::Provider`] with the complete original definition.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolset {
    /// Server identity used by MCP calls.
    pub name: String,
    /// Remote MCP endpoint.
    pub url: String,
    /// Credential sent to the MCP server, when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_token: Option<String>,
    /// Defaults for tools without a per-tool setting in `configs`.
    #[serde(default, skip_serializing_if = "McpToolConfig::is_empty")]
    pub default_config: McpToolConfig,
    /// Per-tool settings, keyed by tool name. Each omitted setting inherits
    /// from `default_config`, then falls back to the provider's default.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub configs: BTreeMap<String, McpToolConfig>,
    /// Whether all calls require approval (`true`) or none do (`false`).
    /// Absence retains provider defaults. Definitions with conditional approval rules
    /// are stored in full as [`Tool::Provider`], rather than as this type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<bool>,
}

impl fmt::Debug for McpToolset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            name,
            url: _,
            authorization_token,
            default_config,
            configs,
            require_approval,
        } = self;
        f.debug_struct("McpToolset")
            .field("name", name)
            // Credentials can appear in any part of the URL, including its path.
            .field("url", &"[REDACTED]")
            .field(
                "authorization_token",
                &authorization_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("default_config", default_config)
            .field("configs", configs)
            .field("require_approval", require_approval)
            .finish()
    }
}

/// Settings for an individual MCP tool or the defaults for a toolset.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpToolConfig {
    /// Whether the tool is available for the model to call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Whether the tool definition is loaded through tool search instead of initially.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

impl McpToolConfig {
    fn is_empty(&self) -> bool {
        self.enabled.is_none() && self.defer_loading.is_none()
    }
}

/// A tool discovered on an MCP server.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpListedTool {
    /// Tool name as exposed by the server.
    pub name: String,
    /// Human-readable tool description, when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing accepted arguments.
    pub input_schema: Value,
    /// Additional tool metadata reported by the server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

/// MCP discovery and execution history. Results reference the matching call's server identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum McpContent {
    /// Tools discovered on a server at this point in the conversation.
    Listing {
        /// Identity of the MCP server.
        server_name: String,
        /// Listing identifier, when supplied by the provider.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Discovered tools in the order reported by the server.
        tools: Vec<McpListedTool>,
        /// Discovery failure, when reported. An empty tool list alone is not a failure.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A call executed by the backend's MCP client.
    Call {
        /// Identity of the MCP server.
        server_name: String,
        /// Ordinary call ID, name, and arguments.
        call: ToolCall,
    },
    /// Output from a backend-executed MCP call.
    Result(ToolResult),
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
    /// Whether the provider must enforce the response schema, overriding native
    /// flags when set and retaining native behavior when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_schema_enforced: Option<bool>,
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
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderExtensions {
    /// Provider fields keyed by their original wire names.
    ///
    /// Codecs use these fields when translating to another format. Exact
    /// same-format replay is controlled separately by [`PreservationMetadata`].
    pub fields: Map<String, Value>,
}

// Extension values can include credentials or embedded original request bodies.
impl fmt::Debug for ProviderExtensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderExtensions")
            .field("field_count", &self.fields.len())
            .finish_non_exhaustive()
    }
}

/// Exact source payloads retained for lossless same-format round trips.
///
/// Translation's default preservation policy prefers a stored same-format body
/// over reconstructing one from normalized fields. A caller that mutates the IR
/// must clear the corresponding entry or use a policy with preservation disabled
/// when those mutations must be encoded.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreservationMetadata {
    /// Original request bodies keyed by source format.
    pub requests: BTreeMap<FormatId, Value>,
    /// Original response bodies keyed by source format.
    pub responses: BTreeMap<FormatId, Value>,
}

// Original payloads can contain credentials even when normalized fields redact them.
impl fmt::Debug for PreservationMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreservationMetadata")
            .field("request_formats", &self.requests.keys())
            .field("response_formats", &self.responses.keys())
            .finish_non_exhaustive()
    }
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
    pub tools: Vec<Tool>,
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

/// A source URL cited by generated text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UrlCitation {
    /// Start character offset in the concatenated text and refusal blocks.
    pub start_index: usize,
    /// Exclusive end character offset in the same text.
    pub end_index: usize,
    /// Source URL.
    pub url: String,
    /// Source title.
    pub title: String,
}

/// One assistant output item in a normalized response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutput {
    /// Actor that produced the output, normally [`Role::Assistant`].
    pub role: Role,
    /// Ordered output content.
    pub content: Vec<ContentBlock>,
    /// URL citations in the concatenated text and refusal blocks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub url_citations: Vec<UrlCitation>,
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

    #[test]
    fn tool_deserialization_rejects_invalid_tagged_definitions() {
        for definition in [
            json!({"type": "mcp", "name": "inv", "parameters": {}}),
            json!({"type": "provider", "provider": "x", "name": "n", "parameters": {}}),
            json!({"type": "unknown", "name": "n", "parameters": {}}),
            json!({"type": null, "name": "n", "parameters": {}}),
        ] {
            assert!(serde_json::from_value::<Tool>(definition).is_err());
        }
        let error = serde_json::from_value::<Tool>(json!({
            "type": "mcp", "name": "inv", "url": "https://example.invalid/mcp",
            "authorization_token": 5, "parameters": {}
        }))
        .expect_err("an invalid credential must not become a function")
        .to_string();
        assert!(error.contains("string"), "field error was lost: {error}");
    }

    #[test]
    fn tool_catalog_preserves_function_json_and_backend_history() -> Result<(), serde_json::Error> {
        let function = json!({"name": "lookup", "description": null, "parameters": {"type": "object"}, "strict": null});
        let definitions = json!([
            function,
            {"type": "mcp", "name": "inventory", "url": "https://example.invalid/mcp", "require_approval": false},
            {"type": "provider", "provider": "openai_responses", "raw": {"type": "mcp", "connector_id": "connector_example", "server_label": "example"}},
            {
                "type": "mcp", "name": "configured", "url": "https://example.invalid/mcp",
                "default_config": {"enabled": false, "defer_loading": true},
                "configs": {"lookup": {"enabled": true}, "search": {"defer_loading": false}}
            }
        ]);
        let tools: Vec<Tool> = serde_json::from_value(definitions.clone())?;
        assert!(matches!(&tools[0], Tool::Function(_)));
        assert!(matches!(&tools[1], Tool::Mcp(_)));
        assert!(matches!(&tools[2], Tool::Provider { .. }));
        assert_eq!(serde_json::to_value(&tools)?, definitions);
        let explicit_function: Tool = serde_json::from_value(json!({
            "type": "function", "name": "lookup", "parameters": {"type": "object"}
        }))?;
        assert_eq!(explicit_function, tools[0]);

        let history = json!([
            {
                "type": "mcp", "kind": "listing", "server_name": "inventory",
                "id": "listing_1",
                "tools": [{
                    "name": "lookup", "description": "Look up stock",
                    "input_schema": {"type": "object"},
                    "annotations": {"readOnlyHint": true}
                }]
            },
            {"type": "mcp", "kind": "listing", "server_name": "empty", "tools": []},
            {
                "type": "mcp", "kind": "listing", "server_name": "unavailable",
                "tools": [], "error": "Server unavailable"
            },
            {
                "type": "mcp", "kind": "call", "server_name": "inventory",
                "call": {"id": "call_1", "name": "lookup", "arguments": {}}
            },
            {
                "type": "mcp", "kind": "result", "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "in stock"}], "is_error": false
            }
        ]);
        let content: Vec<ContentBlock> = serde_json::from_value(history.clone())?;
        assert_eq!(serde_json::to_value(&content)?, history);
        Ok(())
    }

    #[test]
    fn mcp_credentials_are_omitted_from_debug() -> Result<(), serde_json::Error> {
        let token = "synthetic-mcp-debug-secret";
        let url_secret = "synthetic-mcp-url-secret";
        let url = format!("https://example.invalid/mcp?api_key={url_secret}");
        let mcp: McpToolset = serde_json::from_value(json!({
            "name": "inventory", "url": url,
            "authorization_token": token
        }))?;
        let raw = json!({"mcp_servers": [{"authorization_token": token}]});
        let mut request = LlmRequest {
            tools: vec![
                Tool::Mcp(mcp.clone()),
                Tool::Provider {
                    provider: "openai_responses".into(),
                    raw: json!({"type": "mcp", "authorization": token}),
                },
            ],
            ..LlmRequest::default()
        };
        request
            .preservation
            .requests
            .insert("anthropic_messages".into(), raw.clone());
        request.extensions.fields.insert("metadata".into(), raw);
        let serialized = serde_json::to_value(&request)?;
        for debug in [
            format!("{mcp:?}"),
            format!("{mcp:#?}"),
            format!("{:?}", request.tools),
            format!("{:?}", request.extensions),
            format!("{:?}", request.preservation),
            format!("{request:?}"),
            format!("{request:#?}"),
        ] {
            assert!(!debug.contains(token), "Debug exposed an MCP credential");
            assert!(
                !debug.contains(url_secret),
                "Debug exposed an MCP URL secret"
            );
        }
        assert_eq!(serde_json::to_value(&request)?, serialized);
        assert_eq!(serialized["tools"][0]["url"], url);
        assert_eq!(serialized["tools"][0]["authorization_token"], token);
        assert_eq!(serialized["tools"][1]["raw"]["authorization"], token);
        Ok(())
    }
}
