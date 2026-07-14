// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for Anthropic Messages request and response JSON.

use serde_json::{json, Map, Value};

use crate::codecs::common::{is_known_role_name, provider_extensions, text_from_blocks};
use crate::codecs::openai_chat::{decode_file_source, decode_image_source};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::conversation::{
    ContentBlock, ConversationRequest, ConversationResponse, FileSource, ImageSource,
    InstructionBlock, MediaSource, Message, OutputParams, ProviderExtensions, ReasoningParams,
    ResponseOutput, Role, SamplingParams, StopReason, ToolCall, ToolChoice, ToolDefinition,
    ToolResult, Usage,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::sanitize_anthropic_tool_use_id;
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response,
};
use crate::util::{
    json_string, push_lossy, stable_id, string_value, validate_request_capabilities,
};

/// Format codec for Anthropic Messages payloads.
pub struct AnthropicMessagesCodec;

impl FormatCodec for AnthropicMessagesCodec {
    fn format(&self) -> FormatId {
        WireFormat::AnthropicMessages.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let mut request = ConversationRequest {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned),
            output: OutputParams {
                max_output_tokens: body.get("max_tokens").and_then(Value::as_u64),
                response_format: None,
            },
            sampling: SamplingParams {
                temperature: body.get("temperature").and_then(Value::as_f64),
                top_p: body.get("top_p").and_then(Value::as_f64),
                top_k: body.get("top_k").and_then(Value::as_i64),
            },
            reasoning: ReasoningParams {
                effort: body
                    .get("output_config")
                    .and_then(Value::as_object)
                    .and_then(|object| object.get("effort"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                raw: body.get("thinking").cloned(),
            },
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            preservation: capture_request_preservation(
                WireFormat::AnthropicMessages,
                &Value::Object(body.clone()),
                policy,
            ),
            ..ConversationRequest::default()
        };
        if let Some(system) = body.get("system") {
            if let Some(content) = decode_anthropic_system(system, &mut diagnostics, policy)? {
                request.instructions.push(InstructionBlock {
                    role: Role::System,
                    content,
                });
            }
        }
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            let mut generated_id = 0;
            for (index, message) in messages.iter().enumerate() {
                let Some(message) = message.as_object() else {
                    push_lossy(
                        &mut diagnostics,
                        policy,
                        format!("Anthropic message at index {index} is not an object"),
                    )?;
                    continue;
                };
                // Request decoding enforces the provider contract: an unknown
                // role is rejected rather than coerced to `user`. Anthropic
                // Messages only defines `user`/`assistant`, but other known
                // role names stay lenient (mapped to `user`) to preserve
                // historical cross-format behaviour.
                let role = match message.get("role").and_then(Value::as_str) {
                    Some("assistant") => Role::Assistant,
                    None => Role::User,
                    Some(other) if is_known_role_name(other) => Role::User,
                    Some(other) => {
                        return Err(TranslationError::unsupported_role(
                            format!("$.messages[{index}].role"),
                            other,
                        ));
                    }
                };
                generated_id += 1;
                let content = decode_anthropic_content(
                    message
                        .get("content")
                        .unwrap_or(&Value::String(String::new())),
                    role,
                    generated_id,
                    &mut diagnostics,
                    policy,
                )?;
                request.messages.push(Message { role, content });
            }
        }
        request.tools = decode_anthropic_tools(body.get("tools"));
        request.tool_choice = body.get("tool_choice").map(decode_anthropic_tool_choice);
        request.extensions.fields = provider_extensions(
            body,
            &[
                "model",
                "messages",
                "system",
                "tools",
                "tool_choice",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "thinking",
                "output_config",
                "stream",
            ],
        );

        Ok(DecodedRequest {
            request,
            diagnostics,
        })
    }

    fn encode_request(
        &self,
        request: &ConversationRequest,
        policy: &TranslationPolicy,
    ) -> Result<EncodedRequest> {
        if let Some(body) =
            exact_preserved_request(&request.preservation, WireFormat::AnthropicMessages, policy)
        {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, policy)?;
        let mut body = Map::new();
        if let Some(model) = &request.model {
            body.insert("model".to_string(), Value::String(model.clone()));
        }
        let system_text = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        if !system_text.is_empty() {
            body.insert("system".to_string(), Value::String(system_text));
        }

        body.insert(
            "messages".to_string(),
            Value::Array(encode_anthropic_messages(
                &request.messages,
                &mut diagnostics,
                policy,
            )?),
        );

        if !request.tools.is_empty() {
            body.insert("tools".to_string(), encode_anthropic_tools(&request.tools));
        }
        if let Some(choice) = &request.tool_choice {
            body.insert(
                "tool_choice".to_string(),
                encode_anthropic_tool_choice(choice),
            );
        }
        if let Some(stop_sequences) =
            anthropic_stop_sequences_from_extensions(&request.extensions.fields)
        {
            body.insert("stop_sequences".to_string(), stop_sequences);
        }
        if let Some(max_tokens) = request.output.max_output_tokens {
            body.insert("max_tokens".to_string(), json!(max_tokens));
        } else {
            body.insert("max_tokens".to_string(), json!(128_000));
        }
        if let Some(value) = request.sampling.temperature {
            body.insert("temperature".to_string(), json!(value));
        }
        if let Some(value) = request.sampling.top_p {
            body.insert("top_p".to_string(), json!(value));
        }
        if let Some(value) = request.sampling.top_k {
            body.insert("top_k".to_string(), json!(value));
        }
        if request.stream {
            body.insert("stream".to_string(), Value::Bool(true));
        }
        if let Some(effort) = &request.reasoning.effort {
            body.insert("thinking".to_string(), json!({"type": "adaptive"}));
            body.insert("output_config".to_string(), json!({"effort": effort}));
        }

        let body = embed_preservation(Value::Object(body), &request.preservation, policy);
        Ok(EncodedRequest { body, diagnostics })
    }

    fn decode_response(
        &self,
        body: &Value,
        _policy: &TranslationPolicy,
    ) -> Result<DecodedResponse> {
        let body = crate::util::object(body, "$")?;
        let mut content = Vec::new();
        if let Some(blocks) = body.get("content").and_then(Value::as_array) {
            for (index, block) in blocks.iter().enumerate() {
                if let Some(block) = block.as_object() {
                    content.extend(decode_anthropic_content_block(
                        block,
                        Role::Assistant,
                        index + 1,
                        &mut Vec::new(),
                        &TranslationPolicy::default(),
                    )?);
                }
            }
        }
        if content.is_empty() {
            content.push(ContentBlock::Text {
                text: String::new(),
            });
        }
        let response = ConversationResponse {
            id: body
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: Some(map_anthropic_stop_reason(
                    body.get("stop_reason").and_then(Value::as_str),
                )),
            }],
            usage: decode_anthropic_usage(body.get("usage")),
            extensions: ProviderExtensions {
                fields: provider_extensions(
                    body,
                    &[
                        "id",
                        "type",
                        "role",
                        "model",
                        "content",
                        "stop_reason",
                        "usage",
                    ],
                ),
            },
            preservation: capture_response_preservation(
                WireFormat::AnthropicMessages,
                &Value::Object(body.clone()),
                _policy,
            ),
        };
        Ok(DecodedResponse {
            response,
            diagnostics: Vec::new(),
        })
    }

    fn encode_response(
        &self,
        response: &ConversationResponse,
        _policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) = exact_preserved_response(
            &response.preservation,
            WireFormat::AnthropicMessages,
            _policy,
        ) {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let output = response.first_output();
        let content = output
            .map(|output| encode_anthropic_content(&output.content))
            .unwrap_or_else(|| vec![json!({"type": "text", "text": ""})]);
        let body = json!({
            "id": response.id.clone().unwrap_or_else(|| "msg_switchyard".to_string()),
            "type": "message",
            "role": "assistant",
            "model": response.model.clone().unwrap_or_else(|| "unknown".to_string()),
            "content": content,
            "stop_reason": output
                .and_then(|output| output.stop_reason)
                .map(anthropic_stop_reason)
                .unwrap_or("end_turn"),
            "stop_sequence": Value::Null,
            "usage": encode_anthropic_usage(&response.usage),
        });
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, _policy),
            diagnostics: Vec::new(),
        })
    }
}

// Decodes Anthropic's `system` field into instruction blocks.
fn decode_anthropic_system(
    value: &Value,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Vec<ContentBlock>>> {
    match value {
        Value::String(text) if !text.is_empty() => {
            Ok(Some(vec![ContentBlock::Text { text: text.clone() }]))
        }
        Value::String(_) | Value::Null => Ok(None),
        Value::Array(blocks) => {
            let mut content = Vec::new();
            for block in blocks {
                if let Some(block) = block.as_object() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        content.push(ContentBlock::Text { text });
                    }
                }
            }
            Ok((!content.is_empty()).then_some(content))
        }
        other => {
            push_lossy(diagnostics, policy, "Anthropic system field was not text")?;
            Ok(Some(vec![ContentBlock::Text {
                text: string_value(other).unwrap_or_default(),
            }]))
        }
    }
}

// Decodes Anthropic message content into normalized content blocks.
fn decode_anthropic_content(
    value: &Value,
    role: Role,
    generated_counter: usize,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    match value {
        Value::String(text) => Ok(vec![ContentBlock::Text { text: text.clone() }]),
        Value::Null => Ok(vec![ContentBlock::Text {
            text: String::new(),
        }]),
        Value::Array(blocks) => {
            let mut content = Vec::new();
            for (index, block) in blocks.iter().enumerate() {
                let Some(block) = block.as_object() else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("Anthropic content block {index} is not an object"),
                    )?;
                    continue;
                };
                content.extend(decode_anthropic_content_block(
                    block,
                    role,
                    generated_counter + index,
                    diagnostics,
                    policy,
                )?);
            }
            if content.is_empty() {
                content.push(ContentBlock::Text {
                    text: String::new(),
                });
            }
            Ok(content)
        }
        other => Ok(vec![ContentBlock::Text {
            text: string_value(other).unwrap_or_default(),
        }]),
    }
}

// Decodes one Anthropic content block into one or more IR blocks.
fn decode_anthropic_content_block(
    block: &Map<String, Value>,
    _role: Role,
    generated_counter: usize,
    _diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    Ok(match block.get("type").and_then(Value::as_str) {
        Some("text") => vec![ContentBlock::Text {
            text: block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }],
        Some("thinking") => vec![ContentBlock::Reasoning {
            text: block
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            signature: block
                .get("signature")
                .and_then(Value::as_str)
                .filter(|signature| !signature.is_empty())
                .map(ToOwned::to_owned),
        }],
        Some("tool_use") => vec![ContentBlock::ToolCall(ToolCall {
            id: block
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| match &policy.deterministic_ids {
                    DeterministicIdPolicy::GenerateStable { prefix } => {
                        stable_id(prefix, generated_counter)
                    }
                    DeterministicIdPolicy::Preserve => String::new(),
                }),
            name: block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
        })],
        Some("tool_result") => vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: block
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content: decode_tool_result_content(block.get("content").unwrap_or(&Value::Null)),
            is_error: block.get("is_error").and_then(Value::as_bool),
        })],
        Some("image") => {
            let source = block
                .get("source")
                .cloned()
                .map(ImageSource::Raw)
                .unwrap_or_else(|| ImageSource::Raw(Value::Object(block.clone())));
            vec![ContentBlock::Image { source }]
        }
        Some("input_image") | Some("image_url") => decode_image_source(block)
            .map(|source| vec![ContentBlock::Image { source }])
            .unwrap_or_default(),
        Some("input_file") | Some("file") => vec![ContentBlock::File {
            source: decode_file_source(block),
        }],
        _ => vec![ContentBlock::Unknown {
            provider: WireFormat::AnthropicMessages.into(),
            raw: Value::Object(block.clone()),
        }],
    })
}

// Converts Anthropic tool-result content into text-like IR blocks.
fn decode_tool_result_content(value: &Value) -> Vec<ContentBlock> {
    match value {
        Value::String(text) => vec![ContentBlock::Text { text: text.clone() }],
        Value::Array(blocks) => {
            let mut text = Vec::new();
            for block in blocks {
                if let Some(block) = block.as_object() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        text.push(
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        );
                    } else {
                        text.push(json_string(&Value::Object(block.clone())));
                    }
                }
            }
            vec![ContentBlock::Text {
                text: text.join(" "),
            }]
        }
        Value::Null => vec![ContentBlock::Text {
            text: String::new(),
        }],
        other => vec![ContentBlock::Text {
            text: json_string(other),
        }],
    }
}

// Decodes Anthropic tool definitions into normalized tool definitions.
fn decode_anthropic_tools(value: Option<&Value>) -> Vec<ToolDefinition> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(|tool| {
            let name = tool.get("name").and_then(Value::as_str)?.to_string();
            (!name.is_empty()).then(|| ToolDefinition {
                name,
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                parameters: tool
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                strict: None,
            })
        })
        .collect()
}

// Decodes Anthropic tool-choice values into normalized policy.
fn decode_anthropic_tool_choice(value: &Value) -> ToolChoice {
    match value {
        Value::String(text) if text == "auto" => ToolChoice::Auto,
        Value::String(text) if text == "any" => ToolChoice::Required,
        Value::String(text) if text == "none" => ToolChoice::None,
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("auto") => ToolChoice::Auto,
            Some("any") => ToolChoice::Required,
            Some("none") => ToolChoice::None,
            Some("tool") => object
                .get("name")
                .and_then(Value::as_str)
                .map(|name| ToolChoice::Tool {
                    name: name.to_string(),
                })
                .unwrap_or_else(|| ToolChoice::Raw(value.clone())),
            _ => ToolChoice::Raw(value.clone()),
        },
        _ => ToolChoice::Raw(value.clone()),
    }
}

// Encodes one normalized message into Anthropic message JSON.
fn encode_anthropic_message(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    let role = match message.role {
        Role::Assistant => "assistant",
        Role::User | Role::Tool | Role::System | Role::Developer => "user",
    };
    let content = encode_anthropic_content_with_policy(&message.content, diagnostics, policy)?;
    let simple_text = content.len() == 1
        && content
            .first()
            .and_then(Value::as_object)
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            == Some("text");
    let content = if simple_text {
        content
            .first()
            .and_then(Value::as_object)
            .and_then(|object| object.get("text"))
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()))
    } else {
        Value::Array(content)
    };
    Ok(json!({"role": role, "content": content}))
}

// Encodes messages while grouping adjacent tool-result-only messages correctly.
fn encode_anthropic_messages(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut encoded = Vec::new();
    let mut index = 0;

    while let Some(message) = messages.get(index) {
        if !message_is_tool_result_only(message) {
            encoded.push(encode_anthropic_message(message, diagnostics, policy)?);
            index += 1;
            continue;
        }

        let mut content = Vec::new();
        while let Some(tool_message) = messages.get(index) {
            if !message_is_tool_result_only(tool_message) {
                break;
            }
            content.extend(encode_anthropic_content_with_policy(
                &tool_message.content,
                diagnostics,
                policy,
            )?);
            index += 1;
        }
        encoded.push(json!({"role": "user", "content": content}));
    }

    Ok(encoded)
}

// Maps preserved OpenAI-style stop extensions to Anthropic stop sequences.
fn anthropic_stop_sequences_from_extensions(extensions: &Map<String, Value>) -> Option<Value> {
    match extensions.get("stop") {
        Some(Value::String(stop)) => Some(json!([stop])),
        Some(Value::Array(stops)) => Some(Value::Array(stops.clone())),
        _ => None,
    }
}

// Checks whether a message contains only tool-result blocks.
fn message_is_tool_result_only(message: &Message) -> bool {
    (message.role == Role::Tool || message.role == Role::User)
        && !message.content.is_empty()
        && message
            .content
            .iter()
            .all(|block| matches!(block, ContentBlock::ToolResult(_)))
}

// Encodes content while applying lossy-conversion policy to unknown blocks.
fn encode_anthropic_content_with_policy(
    content: &[ContentBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Unknown { raw, .. } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "unknown content block encoded as text for Anthropic",
                )?;
                blocks.push(json!({"type": "text", "text": json_string(raw)}));
            }
            other => blocks.extend(encode_one_anthropic_block(other)),
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    Ok(blocks)
}

// Encodes content without producing diagnostics for response paths.
fn encode_anthropic_content(content: &[ContentBlock]) -> Vec<Value> {
    let mut blocks = content
        .iter()
        .flat_map(encode_one_anthropic_response_block)
        .collect::<Vec<_>>();
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

// Encodes response content, where synthetic reasoning may be shown to clients.
fn encode_one_anthropic_response_block(block: &ContentBlock) -> Vec<Value> {
    match block {
        ContentBlock::Reasoning {
            text,
            signature: None,
        } => vec![json!({
            "type": "thinking",
            "thinking": text,
            "signature": "",
        })],
        other => encode_one_anthropic_block(other),
    }
}

// Encodes a single normalized content block into Anthropic block JSON.
fn encode_one_anthropic_block(block: &ContentBlock) -> Vec<Value> {
    match block {
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
            vec![json!({"type": "text", "text": text})]
        }
        ContentBlock::Reasoning {
            text,
            signature: Some(signature),
        } if !signature.is_empty() => vec![json!({
            "type": "thinking",
            "thinking": text,
            "signature": signature,
        })],
        ContentBlock::Reasoning { .. } => Vec::new(),
        ContentBlock::ToolCall(call) => vec![json!({
            "type": "tool_use",
            "id": sanitize_anthropic_tool_use_id(&call.id),
            "name": call.name,
            "input": anthropic_tool_input(&call.arguments),
        })],
        ContentBlock::ToolResult(result) => vec![json!({
            "type": "tool_result",
            "tool_use_id": sanitize_anthropic_tool_use_id(&result.tool_call_id),
            "content": text_from_blocks(&result.content, " "),
        })],
        ContentBlock::Image { source } => vec![match source {
            ImageSource::Url { url, .. } => {
                json!({"type": "image", "source": {"type": "url", "url": url}})
            }
            ImageSource::Base64 { media_type, data } => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type.clone().unwrap_or_else(|| "image/png".to_string()),
                    "data": data,
                },
            }),
            ImageSource::Raw(raw) => raw.clone(),
        }],
        ContentBlock::File { source } => vec![match source {
            FileSource::FileId(file_id) => {
                json!({"type": "document", "source": {"type": "file", "file_id": file_id}})
            }
            FileSource::FileData { data, filename } => json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "data": data,
                    "filename": filename,
                },
            }),
            FileSource::Raw(raw) => raw.clone(),
        }],
        ContentBlock::Audio { source } => vec![match source {
            MediaSource::Url { url, media_type } => {
                json!({"type": "audio", "source": {"type": "url", "url": url, "media_type": media_type}})
            }
            MediaSource::Base64 { media_type, data } => json!({
                "type": "audio",
                "source": {
                    "type": "base64",
                    "media_type": media_type.clone().unwrap_or_else(|| "audio/mpeg".to_string()),
                    "data": data,
                },
            }),
            MediaSource::Raw(raw) => raw.clone(),
        }],
        ContentBlock::Video { source } => vec![match source {
            MediaSource::Url { url, media_type } => {
                json!({"type": "video", "source": {"type": "url", "url": url, "media_type": media_type}})
            }
            MediaSource::Base64 { media_type, data } => json!({
                "type": "video",
                "source": {
                    "type": "base64",
                    "media_type": media_type.clone().unwrap_or_else(|| "video/mp4".to_string()),
                    "data": data,
                },
            }),
            MediaSource::Raw(raw) => raw.clone(),
        }],
        ContentBlock::Unknown { raw, .. } => vec![raw.clone()],
    }
}

// Anthropic requires `tool_use.input` to be object-shaped, while OpenAI and
// Responses commonly carry function-call arguments as JSON strings.
fn anthropic_tool_input(arguments: &Value) -> Value {
    match arguments {
        Value::Object(object) => Value::Object(object.clone()),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .map_or_else(|_| json!({"raw": text}), ensure_anthropic_tool_input_object),
        Value::Null => json!({}),
        other => json!({"value": other}),
    }
}

// Preserve valid objects and wrap every other JSON shape in a dictionary so
// translated requests satisfy Anthropic's schema without discarding payloads.
fn ensure_anthropic_tool_input_object(arguments: Value) -> Value {
    match arguments {
        Value::Object(_) => arguments,
        Value::Null => json!({}),
        other => json!({"value": other}),
    }
}

// Encodes normalized tool definitions into Anthropic tool JSON.
fn encode_anthropic_tools(tools: &[ToolDefinition]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "input_schema": tool.parameters,
                })
            })
            .collect(),
    )
}

// Encodes normalized tool choice into Anthropic tool-choice JSON.
fn encode_anthropic_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
        ToolChoice::Raw(value) => value.clone(),
    }
}

// Normalizes Anthropic usage fields.
fn decode_anthropic_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.and_then(Value::as_object) else {
        return Usage::default();
    };
    let input_tokens = value.get("input_tokens").and_then(Value::as_u64);
    let output_tokens = value.get("output_tokens").and_then(Value::as_u64);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens
            .zip(output_tokens)
            .map(|(input, output)| input + output),
        reasoning_tokens: value
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64),
    }
}

// Encodes normalized usage into Anthropic usage JSON.
fn encode_anthropic_usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
    })
}

// Maps Anthropic stop reasons to normalized stop reasons.
fn map_anthropic_stop_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("max_tokens") => StopReason::MaxTokens,
        Some("tool_use") => StopReason::ToolUse,
        Some("end_turn") | None => StopReason::EndTurn,
        _ => StopReason::Unknown,
    }
}

// Maps normalized stop reasons back to Anthropic's vocabulary.
fn anthropic_stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::EndTurn
        | StopReason::ContentFilter
        | StopReason::Error
        | StopReason::Unknown => "end_turn",
    }
}
