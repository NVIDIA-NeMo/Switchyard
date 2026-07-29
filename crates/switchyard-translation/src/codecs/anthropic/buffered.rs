// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for Anthropic Messages request and response JSON.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::codecs::common::{is_known_role_name, provider_extensions, text_from_blocks};
use crate::codecs::openai_chat::{decode_file_source, decode_image_source};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, FileSource, ImageSource, InstructionBlock, LlmRequest,
    MediaSource, Message, OutputParams, ProviderExtensions, ProviderPayload, ReasoningParams,
    ResponseOutput, Role, SamplingParams, StopReason, ToolCall, ToolChoice, ToolDefinition,
    ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response,
};
use crate::util::{
    json_string, push_lossy, stable_id, string_value, validate_request_capabilities,
};
use crate::util::{mapped_tool_id, sanitize_anthropic_tool_use_id};

/// Format codec for Anthropic Messages payloads.
pub struct AnthropicMessagesCodec;

impl FormatCodec for AnthropicMessagesCodec {
    fn format(&self) -> FormatId {
        WireFormat::AnthropicMessages.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let mut request = LlmRequest {
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
                    .or_else(|| body.get("reasoning_effort"))
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
            ..LlmRequest::default()
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
                // System-like compatibility roles become typed instructions so
                // the Anthropic encoder can place them at the top level.
                let role = match message.get("role").and_then(Value::as_str) {
                    Some("assistant") => Role::Assistant,
                    Some("system") => Role::System,
                    Some("developer") => Role::Developer,
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
                match role {
                    Role::System | Role::Developer => {
                        request
                            .instructions
                            .push(InstructionBlock { role, content });
                    }
                    Role::User | Role::Assistant | Role::Tool => {
                        request.messages.push(Message { role, content });
                    }
                }
            }
        }
        request.tools = decode_anthropic_tools(body.get("tools"));
        request.tool_choice = body.get("tool_choice").map(decode_anthropic_tool_choice);
        request.source_format = Some(WireFormat::AnthropicMessages.into());
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
                "reasoning_effort",
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
        request: &LlmRequest,
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
        let source_is_anthropic = request
            .source_format
            .as_ref()
            .is_some_and(|source| source.as_str() == WireFormat::AnthropicMessages.as_str());
        let mut body = if source_is_anthropic {
            request.extensions.fields.clone()
        } else {
            Map::new()
        };
        if let Some(model) = &request.model {
            body.insert("model".to_string(), Value::String(model.clone()));
        }
        if let Some(system) =
            encode_anthropic_system(&request.instructions, &mut diagnostics, policy)?
        {
            body.insert("system".to_string(), system);
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
            body.insert(
                "tools".to_string(),
                encode_anthropic_tools(&request.tools, source_is_anthropic),
            );
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
        if source_is_anthropic {
            if let Some(thinking) = &request.reasoning.raw {
                body.insert("thinking".to_string(), thinking.clone());
            }
        }
        if let Some(effort) = &request.reasoning.effort {
            body.entry("thinking".to_string())
                .or_insert_with(|| json!({"type": "adaptive"}));
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
        let response = AggLlmResponse {
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
        response: &AggLlmResponse,
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
            for (index, block) in blocks.iter().enumerate() {
                if let Some(block) = block.as_object() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        content.push(preserve_anthropic_block(
                            block,
                            &["type", "text"],
                            ContentBlock::Text { text },
                        ));
                    } else {
                        push_lossy(
                            diagnostics,
                            policy,
                            format!("Anthropic system block at index {index} was not text"),
                        )?;
                    }
                } else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("Anthropic system block at index {index} was not an object"),
                    )?;
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

// Retains provider-owned block fields without hiding the normalized semantics.
fn preserve_anthropic_block(
    raw: &Map<String, Value>,
    known: &[&str],
    normalized: ContentBlock,
) -> ContentBlock {
    if provider_extensions(raw, known).is_empty() {
        return normalized;
    }
    ContentBlock::Provider {
        payload: ProviderPayload {
            provider: WireFormat::AnthropicMessages.into(),
            raw: Value::Object(raw.clone()),
        },
        normalized: Box::new(normalized),
    }
}

// Decodes Anthropic's nested image source while retaining unrecognized source shapes.
fn decode_anthropic_image_source(source: Option<&Value>) -> ImageSource {
    let Some(source) = source else {
        return ImageSource::Raw(Value::Null);
    };
    let Some(object) = source.as_object() else {
        return ImageSource::Raw(source.clone());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("url") => object
            .get("url")
            .and_then(Value::as_str)
            .map(|url| ImageSource::Url {
                url: url.to_string(),
                detail: None,
            })
            .unwrap_or_else(|| ImageSource::Raw(source.clone())),
        Some("base64") => object
            .get("data")
            .and_then(Value::as_str)
            .map(|data| ImageSource::Base64 {
                media_type: object
                    .get("media_type")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                data: data.to_string(),
            })
            .unwrap_or_else(|| ImageSource::Raw(source.clone())),
        _ => ImageSource::Raw(source.clone()),
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
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    Ok(match block.get("type").and_then(Value::as_str) {
        Some("text") | Some("input_text") => vec![preserve_anthropic_block(
            block,
            &["type", "text"],
            ContentBlock::Text {
                text: block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
        )],
        Some("thinking") => vec![preserve_anthropic_block(
            block,
            &["type", "thinking", "signature"],
            ContentBlock::Reasoning {
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
            },
        )],
        Some("tool_use") => vec![preserve_anthropic_block(
            block,
            &["type", "id", "name", "input"],
            ContentBlock::ToolCall(ToolCall {
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
            }),
        )],
        Some("tool_result") => vec![preserve_anthropic_block(
            block,
            &["type", "tool_use_id", "content", "is_error"],
            ContentBlock::ToolResult(ToolResult {
                tool_call_id: block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                content: decode_tool_result_content(
                    block.get("content").unwrap_or(&Value::Null),
                    generated_counter,
                    diagnostics,
                    policy,
                )?,
                is_error: block.get("is_error").and_then(Value::as_bool),
            }),
        )],
        Some("image") => {
            let source = decode_anthropic_image_source(block.get("source"));
            vec![preserve_anthropic_block(
                block,
                &["type", "source"],
                ContentBlock::Image { source },
            )]
        }
        Some("input_image") | Some("image_url") => decode_image_source(block)
            .map(|source| vec![ContentBlock::Image { source }])
            .unwrap_or_default(),
        Some("document") => vec![preserve_anthropic_block(
            block,
            &["type", "source"],
            ContentBlock::File {
                source: FileSource::Raw(block.get("source").cloned().unwrap_or(Value::Null)),
            },
        )],
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
fn decode_tool_result_content(
    value: &Value,
    generated_counter: usize,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    match value {
        Value::String(text) => Ok(vec![ContentBlock::Text { text: text.clone() }]),
        Value::Array(blocks) => {
            let mut content = Vec::new();
            for (index, block) in blocks.iter().enumerate() {
                let Some(block) = block.as_object() else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("Anthropic tool-result block at index {index} was not an object"),
                    )?;
                    continue;
                };
                content.extend(decode_anthropic_content_block(
                    block,
                    Role::User,
                    generated_counter + index,
                    diagnostics,
                    policy,
                )?);
            }
            Ok(content)
        }
        Value::Null => Ok(vec![ContentBlock::Text {
            text: String::new(),
        }]),
        other => Ok(vec![ContentBlock::Text {
            text: json_string(other),
        }]),
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
                provider_payload: Some(ProviderPayload {
                    provider: WireFormat::AnthropicMessages.into(),
                    raw: Value::Object(tool.clone()),
                }),
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

// Encodes instructions as text unless provider-owned fields require structured blocks.
fn encode_anthropic_system(
    instructions: &[InstructionBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    let mut blocks = Vec::new();
    let mut needs_structured_blocks = false;
    for block in instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
    {
        let Some((encoded, preserved)) = encode_anthropic_system_block(block, diagnostics, policy)?
        else {
            continue;
        };
        blocks.push(encoded);
        needs_structured_blocks |= preserved;
    }
    if blocks.is_empty() {
        return Ok(None);
    }
    if needs_structured_blocks {
        return Ok(Some(Value::Array(blocks)));
    }
    let text = blocks
        .iter()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok((!text.is_empty()).then_some(Value::String(text)))
}

// Encodes one instruction block and reapplies same-provider fields when present.
fn encode_anthropic_system_block(
    block: &ContentBlock,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<(Value, bool)>> {
    match block {
        ContentBlock::Provider {
            payload,
            normalized,
        } => {
            let Some((encoded, preserved)) =
                encode_anthropic_system_block(normalized, diagnostics, policy)?
            else {
                return Ok(None);
            };
            if is_anthropic_payload(payload) {
                Ok(Some((merge_provider_payload(payload, encoded), true)))
            } else {
                Ok(Some((encoded, preserved)))
            }
        }
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
            Ok(Some((json!({"type": "text", "text": text}), false)))
        }
        _ => {
            push_lossy(
                diagnostics,
                policy,
                "non-text instruction block omitted from Anthropic system",
            )?;
            Ok(None)
        }
    }
}

// Encodes one normalized message into Anthropic message JSON.
fn encode_anthropic_message(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> Result<Option<Value>> {
    let role = match message.role {
        Role::Assistant => "assistant",
        Role::User | Role::Tool | Role::System | Role::Developer => "user",
    };
    let content = encode_anthropic_content_with_policy(
        &message.content,
        diagnostics,
        policy,
        id_map,
        used_ids,
    )?;
    if content.is_empty() {
        push_lossy(
            diagnostics,
            policy,
            "Anthropic message omitted after all content blocks were removed",
        )?;
        return Ok(None);
    }
    let content = simple_anthropic_text(&content).unwrap_or(Value::Array(content));
    Ok(Some(json!({"role": role, "content": content})))
}

// Encodes messages while grouping adjacent tool-result-only messages correctly.
fn encode_anthropic_messages(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut encoded = Vec::new();
    let mut id_map = BTreeMap::new();
    let mut used_ids = BTreeMap::new();
    let mut index = 0;

    while let Some(message) = messages.get(index) {
        if !message_is_tool_result_only(message) {
            if let Some(message) =
                encode_anthropic_message(message, diagnostics, policy, &mut id_map, &mut used_ids)?
            {
                encoded.push(message);
            }
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
                &mut id_map,
                &mut used_ids,
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
            .all(|block| matches!(block.normalized(), ContentBlock::ToolResult(_)))
}

// Encodes content while applying lossy-conversion policy to unknown blocks.
fn encode_anthropic_content_with_policy(
    content: &[ContentBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> Result<Vec<Value>> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Unknown { provider, raw }
                if provider.as_str() == WireFormat::AnthropicMessages.as_str() =>
            {
                blocks.push(raw.clone());
            }
            ContentBlock::Unknown { raw, .. } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "unknown content block encoded as text for Anthropic",
                )?;
                blocks.push(json!({"type": "text", "text": json_string(raw)}));
            }
            ContentBlock::Provider {
                payload,
                normalized,
            } => {
                let encoded = encode_anthropic_content_with_policy(
                    std::slice::from_ref(normalized),
                    diagnostics,
                    policy,
                    id_map,
                    used_ids,
                )?;
                if is_anthropic_payload(payload) {
                    blocks.extend(
                        encoded
                            .into_iter()
                            .map(|block| merge_provider_payload(payload, block)),
                    );
                } else {
                    blocks.extend(encoded);
                }
            }
            other => {
                blocks.extend(encode_one_anthropic_request_block(
                    other,
                    diagnostics,
                    policy,
                    id_map,
                    used_ids,
                )?);
            }
        }
    }
    Ok(blocks)
}

// Encodes request blocks while keeping sanitized tool call/result IDs aligned.
fn encode_one_anthropic_request_block(
    block: &ContentBlock,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> Result<Vec<Value>> {
    Ok(match block {
        ContentBlock::ToolCall(call) => vec![json!({
            "type": "tool_use",
            "id": mapped_tool_id(&call.id, id_map, used_ids),
            "name": call.name,
            "input": anthropic_tool_input(&call.arguments),
        })],
        ContentBlock::ToolResult(result) => {
            let mut tool_result = Map::new();
            tool_result.insert("type".to_string(), Value::String("tool_result".to_string()));
            tool_result.insert(
                "tool_use_id".to_string(),
                Value::String(mapped_tool_id(&result.tool_call_id, id_map, used_ids)),
            );
            let content = encode_anthropic_content_with_policy(
                &result.content,
                diagnostics,
                policy,
                id_map,
                used_ids,
            )?;
            tool_result.insert(
                "content".to_string(),
                simple_anthropic_text(&content).unwrap_or(Value::Array(content)),
            );
            if let Some(is_error) = result.is_error {
                tool_result.insert("is_error".to_string(), Value::Bool(is_error));
            }
            vec![Value::Object(tool_result)]
        }
        ContentBlock::Reasoning { signature, .. }
            if signature.as_deref().is_none_or(str::is_empty) =>
        {
            push_lossy(
                diagnostics,
                policy,
                "unsigned Anthropic thinking block omitted",
            )?;
            Vec::new()
        }
        other => encode_one_anthropic_block(other),
    })
}

// Uses Anthropic's compact string form only when no block metadata would be lost.
fn simple_anthropic_text(blocks: &[Value]) -> Option<Value> {
    let object = blocks.first()?.as_object()?;
    (blocks.len() == 1
        && object.len() == 2
        && object.get("type").and_then(Value::as_str) == Some("text"))
    .then(|| object.get("text").cloned())
    .flatten()
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
        ContentBlock::Provider {
            payload,
            normalized,
        } => {
            let encoded = encode_one_anthropic_response_block(normalized);
            if is_anthropic_payload(payload) {
                encoded
                    .into_iter()
                    .map(|block| merge_provider_payload(payload, block))
                    .collect()
            } else {
                encoded
            }
        }
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
            ImageSource::Raw(raw) => encode_anthropic_raw_image(raw),
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
            FileSource::Raw(raw) => encode_anthropic_raw_document(raw),
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
        ContentBlock::Provider {
            payload,
            normalized,
        } => {
            let encoded = encode_one_anthropic_block(normalized);
            if is_anthropic_payload(payload) {
                encoded
                    .into_iter()
                    .map(|block| merge_provider_payload(payload, block))
                    .collect()
            } else {
                encoded
            }
        }
    }
}

// Restores the outer Anthropic image block around a raw source object.
fn encode_anthropic_raw_image(raw: &Value) -> Value {
    if raw.get("type").and_then(Value::as_str) == Some("image") {
        raw.clone()
    } else {
        json!({"type": "image", "source": raw})
    }
}

// Restores the outer Anthropic document block around a raw source object.
fn encode_anthropic_raw_document(raw: &Value) -> Value {
    if raw.get("type").and_then(Value::as_str) == Some("document") {
        raw.clone()
    } else {
        json!({"type": "document", "source": raw})
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

// Encodes normalized tools while retaining Anthropic-native server-tool fields.
fn encode_anthropic_tools(tools: &[ToolDefinition], source_is_anthropic: bool) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                let preserved = source_is_anthropic
                    .then_some(tool.provider_payload.as_ref())
                    .flatten()
                    .filter(|payload| is_anthropic_payload(payload))
                    .and_then(|payload| payload.raw.as_object())
                    .cloned();
                let mut encoded = preserved.unwrap_or_default();
                encoded.insert("name".to_string(), Value::String(tool.name.clone()));
                if encoded.contains_key("input_schema") || !encoded.contains_key("type") {
                    encoded.insert("input_schema".to_string(), tool.parameters.clone());
                    if let Some(description) = &tool.description {
                        encoded.insert(
                            "description".to_string(),
                            Value::String(description.clone()),
                        );
                    } else {
                        encoded.remove("description");
                    }
                }
                Value::Object(encoded)
            })
            .collect(),
    )
}

fn is_anthropic_payload(payload: &ProviderPayload) -> bool {
    payload.provider.as_str() == WireFormat::AnthropicMessages.as_str()
}

// Reapplies provider-owned fields while letting canonical fields win.
fn merge_provider_payload(payload: &ProviderPayload, encoded: Value) -> Value {
    match (&payload.raw, encoded) {
        (Value::Object(raw), Value::Object(encoded)) => {
            let mut merged = raw.clone();
            merged.extend(encoded);
            Value::Object(merged)
        }
        (_, encoded) => encoded,
    }
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
    let cached_input_tokens = value.get("cache_read_input_tokens").and_then(Value::as_u64);
    let cache_creation_input_tokens = value
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64);
    let output_tokens = value.get("output_tokens").and_then(Value::as_u64);
    Usage {
        input_tokens,
        cache: Usage::cache_details(cached_input_tokens, cache_creation_input_tokens),
        output_tokens,
        total_tokens: input_tokens.zip(output_tokens).map(|(input, output)| {
            input
                + cached_input_tokens.unwrap_or(0)
                + cache_creation_input_tokens.unwrap_or(0)
                + output
        }),
        reasoning_tokens: value
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64),
    }
}

// Encodes normalized usage into Anthropic usage JSON.
fn encode_anthropic_usage(usage: &Usage) -> Value {
    let mut value = json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
    });
    if let Some(cached_tokens) = usage.cached_input_tokens() {
        value["cache_read_input_tokens"] = json!(cached_tokens);
    }
    if let Some(cache_creation_tokens) = usage.cache_creation_input_tokens() {
        value["cache_creation_input_tokens"] = json!(cache_creation_tokens);
    }
    value
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
