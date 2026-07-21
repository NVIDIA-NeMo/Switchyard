// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for OpenAI Responses request and response JSON.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use crate::codecs::common::{
    is_known_role_name, provider_extensions, reasoning_text_from_blocks, text_from_blocks,
};
use crate::codecs::openai_chat::{decode_file_source, decode_image_source};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, LlmRequest, MediaSource, Message, OutputParams,
    ProviderExtensions, ReasoningParams, ResponseOutput, Role, SamplingParams, StopReason,
    ToolCall, ToolChoice, ToolDefinition, ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, json_string, push_lossy, stable_id,
    string_value, validate_request_capabilities,
};

/// Format codec for OpenAI Responses payloads.
pub struct OpenAiResponsesCodec;

impl FormatCodec for OpenAiResponsesCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiResponses.into()
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
                max_output_tokens: body.get("max_output_tokens").and_then(Value::as_u64),
                response_format: decode_responses_text_format(body.get("text")),
            },
            reasoning: ReasoningParams {
                effort: body
                    .get("reasoning")
                    .and_then(Value::as_object)
                    .and_then(|object| object.get("effort"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                raw: body.get("reasoning").cloned(),
            },
            sampling: SamplingParams {
                temperature: body.get("temperature").and_then(Value::as_f64),
                top_p: body.get("top_p").and_then(Value::as_f64),
                top_k: None,
            },
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            preservation: capture_request_preservation(
                WireFormat::OpenAiResponses,
                &Value::Object(body.clone()),
                policy,
            ),
            ..LlmRequest::default()
        };
        if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
            if !instructions.is_empty() {
                request.instructions.push(crate::llm::InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: instructions.to_string(),
                    }],
                });
            }
        }
        request.messages = decode_responses_input(
            body.get("input").unwrap_or(&Value::String(String::new())),
            &mut diagnostics,
            policy,
        )?;
        request.tools = decode_responses_tools(body.get("tools"));
        request.tool_choice = body
            .get("tool_choice")
            .and_then(decode_responses_tool_choice);
        request.extensions.fields = provider_extensions(
            body,
            &[
                "model",
                "instructions",
                "input",
                "tools",
                "tool_choice",
                "max_output_tokens",
                "text",
                "reasoning",
                "temperature",
                "top_p",
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
        _policy: &TranslationPolicy,
    ) -> Result<EncodedRequest> {
        if let Some(body) =
            exact_preserved_request(&request.preservation, WireFormat::OpenAiResponses, _policy)
        {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, _policy)?;
        let mut body = Map::new();
        if let Some(model) = &request.model {
            body.insert("model".to_string(), Value::String(model.clone()));
        }
        let instructions = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        if !instructions.is_empty() {
            body.insert("instructions".to_string(), Value::String(instructions));
        }
        body.insert(
            "input".to_string(),
            encode_responses_input(&request.messages, &mut diagnostics, _policy)?,
        );
        if !request.tools.is_empty() {
            body.insert("tools".to_string(), encode_responses_tools(&request.tools));
        }
        if let Some(choice) = &request.tool_choice {
            if let Some(choice) = encode_responses_tool_choice(choice) {
                body.insert("tool_choice".to_string(), choice);
            }
        }
        if let Some(max_output_tokens) = request.output.max_output_tokens {
            body.insert("max_output_tokens".to_string(), json!(max_output_tokens));
        }
        if let Some(response_format) = &request.output.response_format {
            body.insert("text".to_string(), json!({"format": response_format}));
        }
        if let Some(effort) = &request.reasoning.effort {
            body.insert("reasoning".to_string(), json!({"effort": effort}));
        }
        if let Some(value) = request.sampling.temperature {
            body.insert("temperature".to_string(), json!(value));
        }
        if let Some(value) = request.sampling.top_p {
            body.insert("top_p".to_string(), json!(value));
        }
        if request.stream {
            body.insert("stream".to_string(), Value::Bool(true));
        }
        let body = embed_preservation(Value::Object(body), &request.preservation, _policy);
        Ok(EncodedRequest { body, diagnostics })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let mut outputs = Vec::new();
        if let Some(items) = body.get("output").and_then(Value::as_array) {
            for item in items {
                if let Some(output) = decode_responses_output_item(item, &mut diagnostics, policy)?
                {
                    outputs.push(output);
                }
            }
        }
        if outputs.is_empty() {
            outputs.push(ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: String::new(),
                }],
                stop_reason: Some(StopReason::EndTurn),
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
            outputs,
            usage: decode_responses_usage(body.get("usage")),
            extensions: ProviderExtensions {
                fields: provider_extensions(
                    body,
                    &["id", "model", "object", "output", "usage", "status"],
                ),
            },
            preservation: capture_response_preservation(
                WireFormat::OpenAiResponses,
                &Value::Object(body.clone()),
                policy,
            ),
        };
        Ok(DecodedResponse {
            response,
            diagnostics,
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
        _policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) =
            exact_preserved_response(&response.preservation, WireFormat::OpenAiResponses, _policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        Ok(EncodedResponse {
            body: embed_preservation(
                json!({
                    "id": response.id.clone().unwrap_or_else(|| "resp_switchyard".to_string()),
                    "object": "response",
                    "created_at": 0,
                    "model": response.model.clone().unwrap_or_else(|| "unknown".to_string()),
                    "status": "completed",
                    "output": encode_responses_output(&response.outputs),
                    "usage": encode_responses_usage(&response.usage),
                    "parallel_tool_calls": true,
                    "tool_choice": "auto",
                    "tools": [],
                }),
                &response.preservation,
                _policy,
            ),
            diagnostics: Vec::new(),
        })
    }
}

// Decodes Responses `input` into ordered normalized messages.
fn decode_responses_input(
    value: &Value,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Message>> {
    match value {
        Value::String(text) => Ok(vec![Message::text(Role::User, text)]),
        Value::Array(items) => {
            let mut messages = Vec::new();
            let mut pending_tool_calls = Vec::new();
            let mut pending_tool_outputs = Vec::new();
            let mut deferred_messages = Vec::new();
            let mut pending_reasoning: Vec<ContentBlock> = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let Some(item) = item.as_object() else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("Responses input item {index} is not an object"),
                    )?;
                    continue;
                };
                match item.get("type").and_then(Value::as_str) {
                    Some("message") => {
                        let role = request_role_from_responses(
                            item.get("role").and_then(Value::as_str),
                            &format!("$.input[{index}].role"),
                        )?;
                        let mut content =
                            decode_responses_content(item.get("content").unwrap_or(&Value::Null));
                        // Reasoning that precedes an assistant message belongs to
                        // that turn; fold it in so it never surfaces as its own
                        // (empty-looking) assistant message downstream.
                        if role == Role::Assistant
                            && pending_tool_calls.is_empty()
                            && !pending_reasoning.is_empty()
                        {
                            let mut merged = std::mem::take(&mut pending_reasoning);
                            merged.append(&mut content);
                            content = merged;
                        }
                        if role != Role::Assistant {
                            flush_unattached_responses_reasoning(
                                &mut messages,
                                &mut pending_tool_calls,
                                &mut pending_tool_outputs,
                                &mut deferred_messages,
                                &mut pending_reasoning,
                            );
                        }
                        push_responses_non_tool_message(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_tool_outputs,
                            &mut deferred_messages,
                            &mut pending_reasoning,
                            Message { role, content },
                        );
                    }
                    Some("reasoning") => {
                        // A reasoning item after a completed tool block opens the
                        // next assistant turn: flush the finished block so this
                        // reasoning attaches to the turn that produced it.
                        if !pending_tool_outputs.is_empty() {
                            flush_responses_tool_block(
                                &mut messages,
                                &mut pending_tool_calls,
                                &mut pending_tool_outputs,
                                &mut deferred_messages,
                                &mut pending_reasoning,
                            );
                        }
                        pending_reasoning.extend(decode_responses_reasoning_item(item));
                    }
                    Some("function_call") => {
                        if !pending_tool_outputs.is_empty() {
                            flush_responses_tool_block(
                                &mut messages,
                                &mut pending_tool_calls,
                                &mut pending_tool_outputs,
                                &mut deferred_messages,
                                &mut pending_reasoning,
                            );
                        }
                        let id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| match &policy.deterministic_ids {
                                DeterministicIdPolicy::GenerateStable { prefix } => {
                                    stable_id(prefix, index + 1)
                                }
                                DeterministicIdPolicy::Preserve => String::new(),
                            });
                        pending_tool_calls.push(ToolCall {
                            id,
                            name: item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: item.get("arguments").cloned().unwrap_or_else(|| json!({})),
                        });
                    }
                    Some("function_call_output") => {
                        let tool_call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let output_text = item.get("output").map(json_string).unwrap_or_default();
                        pending_tool_outputs.push(ToolResult {
                            tool_call_id,
                            content: vec![ContentBlock::Text { text: output_text }],
                            is_error: None,
                        });
                    }
                    _ => {
                        let message = Message {
                            role: Role::User,
                            content: vec![ContentBlock::Unknown {
                                provider: WireFormat::OpenAiResponses.into(),
                                raw: Value::Object(item.clone()),
                            }],
                        };
                        flush_unattached_responses_reasoning(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_tool_outputs,
                            &mut deferred_messages,
                            &mut pending_reasoning,
                        );
                        push_responses_non_tool_message(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_tool_outputs,
                            &mut deferred_messages,
                            &mut pending_reasoning,
                            message,
                        );
                    }
                }
            }
            if !pending_tool_calls.is_empty() || !pending_tool_outputs.is_empty() {
                flush_responses_tool_block(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_tool_outputs,
                    &mut deferred_messages,
                    &mut pending_reasoning,
                );
            }
            // Trailing reasoning with no assistant turn to join keeps the
            // historical standalone-message shape.
            if !pending_reasoning.is_empty() {
                messages.push(Message {
                    role: Role::Assistant,
                    content: pending_reasoning,
                });
            }
            Ok(messages)
        }
        other => Ok(vec![Message::text(
            Role::User,
            string_value(other).unwrap_or_default(),
        )]),
    }
}

// Places non-tool input items without breaking Responses tool-call adjacency.
fn push_responses_non_tool_message(
    messages: &mut Vec<Message>,
    pending_tool_calls: &mut Vec<ToolCall>,
    pending_tool_outputs: &mut Vec<ToolResult>,
    deferred_messages: &mut Vec<Message>,
    pending_reasoning: &mut Vec<ContentBlock>,
    message: Message,
) {
    if !pending_tool_calls.is_empty() && pending_tool_outputs.is_empty() {
        deferred_messages.push(message);
        return;
    }
    if !pending_tool_calls.is_empty() || !pending_tool_outputs.is_empty() {
        flush_responses_tool_block(
            messages,
            pending_tool_calls,
            pending_tool_outputs,
            deferred_messages,
            pending_reasoning,
        );
    }
    messages.push(message);
}

// Emits reasoning that cannot join an assistant turn as a standalone assistant
// message at its original position. While tool calls are pending, the
// reasoning belongs to the in-flight turn and stays pending for the flush.
fn flush_unattached_responses_reasoning(
    messages: &mut Vec<Message>,
    pending_tool_calls: &mut Vec<ToolCall>,
    pending_tool_outputs: &mut Vec<ToolResult>,
    deferred_messages: &mut Vec<Message>,
    pending_reasoning: &mut Vec<ContentBlock>,
) {
    if pending_reasoning.is_empty() || !pending_tool_calls.is_empty() {
        return;
    }
    let message = Message {
        role: Role::Assistant,
        content: std::mem::take(pending_reasoning),
    };
    push_responses_non_tool_message(
        messages,
        pending_tool_calls,
        pending_tool_outputs,
        deferred_messages,
        pending_reasoning,
        message,
    );
}

// Flushes pending function calls and matching outputs while preserving adjacency.
// Pending reasoning rides in the same assistant message as the tool calls so a
// Codex reasoning + function_call turn stays one assistant turn downstream.
fn flush_responses_tool_block(
    messages: &mut Vec<Message>,
    pending_tool_calls: &mut Vec<ToolCall>,
    pending_tool_outputs: &mut Vec<ToolResult>,
    deferred_messages: &mut Vec<Message>,
    pending_reasoning: &mut Vec<ContentBlock>,
) {
    let tool_call_ids = pending_tool_calls
        .iter()
        .map(|call| call.id.clone())
        .collect::<HashSet<_>>();

    if !pending_tool_calls.is_empty() {
        let mut content = std::mem::take(pending_reasoning);
        content.extend(
            std::mem::take(pending_tool_calls)
                .into_iter()
                .map(ContentBlock::ToolCall),
        );
        messages.push(Message {
            role: Role::Assistant,
            content,
        });
    }

    for output in std::mem::take(pending_tool_outputs) {
        if tool_call_ids.contains(&output.tool_call_id) {
            messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(output)],
            });
        } else {
            let tool_call_id = output.tool_call_id;
            let output_text = text_from_blocks(&output.content, " ");
            messages.push(Message::text(
                Role::User,
                format!("Tool result {tool_call_id}: {output_text}"),
            ));
        }
    }

    messages.append(deferred_messages);
}

// Decodes a Responses reasoning item into private reasoning IR content.
fn decode_responses_reasoning_item(item: &Map<String, Value>) -> Vec<ContentBlock> {
    let mut parts = Vec::new();
    collect_responses_reasoning_text(item.get("content"), &mut parts);
    collect_responses_reasoning_text(item.get("summary"), &mut parts);
    if let Some(text) = item.get("text").and_then(Value::as_str) {
        parts.push(text.to_string());
    }
    vec![ContentBlock::Reasoning {
        text: parts.join("\n"),
        signature: None,
    }]
}

// Collects text from the known Responses reasoning content/summary shapes.
fn collect_responses_reasoning_text(value: Option<&Value>, out: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) if !text.is_empty() => out.push(text.clone()),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) if !text.is_empty() => out.push(text.clone()),
                    Value::Object(object) => {
                        if matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("reasoning_text" | "summary_text" | "text")
                        ) {
                            if let Some(text) = object.get("text").and_then(Value::as_str) {
                                if !text.is_empty() {
                                    out.push(text.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

// Decodes Responses content arrays or strings into normalized content blocks.
fn decode_responses_content(value: &Value) -> Vec<ContentBlock> {
    match value {
        Value::String(text) => vec![ContentBlock::Text { text: text.clone() }],
        Value::Array(blocks) => {
            let mut out = Vec::new();
            for block in blocks {
                let Some(block) = block.as_object() else {
                    continue;
                };
                match block.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        out.push(ContentBlock::Text {
                            text: block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        });
                    }
                    Some("refusal") => out.push(ContentBlock::Refusal {
                        text: block
                            .get("refusal")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    Some("reasoning_text") | Some("summary_text") => {
                        out.push(ContentBlock::Reasoning {
                            text: block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            signature: None,
                        });
                    }
                    Some("input_image") => {
                        if let Some(source) = decode_image_source(block) {
                            out.push(ContentBlock::Image { source });
                        }
                    }
                    Some("input_file") => out.push(ContentBlock::File {
                        source: decode_file_source(block),
                    }),
                    _ => out.push(ContentBlock::Unknown {
                        provider: WireFormat::OpenAiResponses.into(),
                        raw: Value::Object(block.clone()),
                    }),
                }
            }
            if out.is_empty() {
                out.push(ContentBlock::Text {
                    text: String::new(),
                });
            }
            out
        }
        Value::Null => vec![ContentBlock::Text {
            text: String::new(),
        }],
        other => vec![ContentBlock::Text {
            text: string_value(other).unwrap_or_default(),
        }],
    }
}

// Decodes Responses role strings into normalized roles. Used for decoding
// provider *responses*, which stay lenient — an unexpected role from upstream
// is coerced to `user` rather than failing the whole response.
fn role_from_responses(role: Option<&str>) -> Role {
    match role {
        Some("assistant") => Role::Assistant,
        Some("system") => Role::System,
        Some("developer") => Role::Developer,
        _ => Role::User,
    }
}

// Decodes a Responses *request* message role, enforcing the provider contract.
//
// Unlike `role_from_responses` (used for provider responses), request decoding
// rejects an unknown role such as "api" with [`TranslationError::InvalidValue`]
// so the router surfaces the same `invalid_value` error the provider would,
// instead of silently coercing it to `user`. A missing role and
// known-but-unmapped roles keep their historical mapping to `user`.
fn request_role_from_responses(role: Option<&str>, path: &str) -> Result<Role> {
    match role {
        Some("assistant") => Ok(Role::Assistant),
        Some("system") => Ok(Role::System),
        Some("developer") => Ok(Role::Developer),
        None => Ok(Role::User),
        Some(other) if is_known_role_name(other) => Ok(Role::User),
        Some(other) => Err(TranslationError::unsupported_role(path, other)),
    }
}

// Decodes Responses tool shapes, including Codex-style tool entries.
fn decode_responses_tools(value: Option<&Value>) -> Vec<ToolDefinition> {
    let Some(tools) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if tool.get("type").and_then(Value::as_str) == Some("function") {
            if let Some(function) = tool.get("function").and_then(Value::as_object) {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    if !name.is_empty() {
                        out.push(ToolDefinition {
                            name: name.to_string(),
                            description: function
                                .get("description")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                            parameters: function
                                .get("parameters")
                                .cloned()
                                .unwrap_or_else(|| json!({})),
                            strict: function.get("strict").and_then(Value::as_bool),
                        });
                    }
                }
            } else {
                if !push_responses_function_tool(&mut out, tool) {
                    push_responses_id_tool(&mut out, tool);
                }
            }
        } else if tool.get("type").is_none() && tool.contains_key("name") {
            push_responses_function_tool(&mut out, tool);
        } else {
            push_responses_id_tool(&mut out, tool);
        }
    }
    out
}

// Adds an OpenAI-style function tool to the normalized tool list.
fn push_responses_function_tool(
    out: &mut Vec<ToolDefinition>,
    tool: &serde_json::Map<String, Value>,
) -> bool {
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return false;
    };
    if name.is_empty() {
        return false;
    }

    out.push(ToolDefinition {
        name: name.to_string(),
        description: tool
            .get("description")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        parameters: tool.get("parameters").cloned().unwrap_or_else(|| json!({})),
        strict: tool.get("strict").and_then(Value::as_bool),
    });
    true
}

// Adds a Codex-style ID tool to the normalized tool list.
fn push_responses_id_tool(
    out: &mut Vec<ToolDefinition>,
    tool: &serde_json::Map<String, Value>,
) -> bool {
    let Some(id) = tool.get("id").and_then(Value::as_str) else {
        return false;
    };
    if id.is_empty() {
        return false;
    }

    out.push(ToolDefinition {
        name: id.to_string(),
        description: tool
            .get("description")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        parameters: tool
            .get("inputSchema")
            .and_then(Value::as_object)
            .and_then(|schema| schema.get("jsonSchema"))
            .cloned()
            .unwrap_or_else(|| json!({})),
        strict: None,
    });
    true
}

// Decodes Responses tool-choice values into normalized policy.
fn decode_responses_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value {
        Value::String(text) if text == "auto" => Some(ToolChoice::Auto),
        Value::String(text) if text == "required" => Some(ToolChoice::Required),
        Value::String(text) if text == "none" => Some(ToolChoice::None),
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("function") => {
            object
                .get("name")
                .and_then(Value::as_str)
                .map(|name| ToolChoice::Tool {
                    name: name.to_string(),
                })
        }
        Value::Object(_) => None,
        _ => Some(ToolChoice::Raw(value.clone())),
    }
}

// Maps Responses text format options into Chat-compatible response_format JSON.
fn decode_responses_text_format(value: Option<&Value>) -> Option<Value> {
    let format = value
        .and_then(Value::as_object)
        .and_then(|object| object.get("format"))
        .and_then(Value::as_object)?;

    match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => {
            if let Some(json_schema) = format.get("json_schema").and_then(Value::as_object) {
                return Some(json!({
                    "type": "json_schema",
                    "json_schema": Value::Object(json_schema.clone()),
                }));
            }

            let mut json_schema = Map::new();
            for field in ["name", "description", "schema", "strict"] {
                if let Some(value) = format.get(field) {
                    json_schema.insert(field.to_string(), value.clone());
                }
            }
            (!json_schema.is_empty()).then_some(json!({
                "type": "json_schema",
                "json_schema": Value::Object(json_schema),
            }))
        }
        Some("json_object") => Some(json!({"type": "json_object"})),
        Some("text") => Some(json!({"type": "text"})),
        _ => None,
    }
}

// Encodes normalized messages into the Responses `input` shape.
fn encode_responses_input(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    if messages.len() == 1
        && matches!(messages[0].role, Role::User)
        && messages[0].content.len() == 1
        && matches!(messages[0].content[0], ContentBlock::Text { .. })
    {
        if let ContentBlock::Text { text } = &messages[0].content[0] {
            return Ok(Value::String(text.clone()));
        }
    }
    let mut encoded = Vec::new();
    for message in messages {
        if message.content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
            )
        }) {
            encoded.extend(
                message
                    .content
                    .iter()
                    .filter_map(encode_responses_special_input),
            );
            continue;
        }
        let mut visible_content = Vec::new();
        let mut emitted_special = false;
        for block in &message.content {
            if let Some(item) = encode_responses_special_input(block) {
                encoded.push(item);
                emitted_special = true;
            } else {
                visible_content.push(block.clone());
            }
        }
        if !visible_content.is_empty() || !emitted_special {
            let content = encode_responses_content(&visible_content, diagnostics, policy)?;
            encoded.push(json!({
                "type": "message",
                "role": role_to_responses(message.role),
                "content": content,
            }));
        }
    }
    Ok(Value::Array(encoded))
}

// Encodes IR blocks that Responses represents as top-level input items.
fn encode_responses_special_input(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Reasoning { text, .. } => Some(json!({
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": text}],
            "summary": [],
        })),
        ContentBlock::ToolCall(call) => Some(json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        })),
        ContentBlock::ToolResult(result) => Some(json!({
            "type": "function_call_output",
            "call_id": result.tool_call_id,
            "output": text_from_blocks(&result.content, " "),
        })),
        _ => None,
    }
}

// Maps normalized roles back to Responses role strings.
fn role_to_responses(role: Role) -> &'static str {
    match role {
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Developer => "developer",
        Role::User | Role::Tool => "user",
    }
}

// Encodes normalized content into Responses message content.
fn encode_responses_content(
    content: &[ContentBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    let has_non_text = content.iter().any(|block| {
        !matches!(
            block,
            ContentBlock::Text { .. }
                | ContentBlock::Refusal { .. }
                | ContentBlock::Reasoning { .. }
        )
    });
    if !has_non_text {
        return Ok(Value::String(text_from_blocks(content, "\n")));
    }
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text } => blocks.push(json!({"type": "input_text", "text": text})),
            ContentBlock::Refusal { text } => {
                blocks.push(json!({"type": "refusal", "refusal": text}));
            }
            ContentBlock::Image { source } => {
                blocks.push(json!({"type": "input_image", "image_url": source}));
            }
            ContentBlock::Audio { source } => blocks.push(match source {
                MediaSource::Raw(raw) => json!({"type": "input_text", "text": json_string(raw)}),
                MediaSource::Url { url, media_type } => json!({
                    "type": "input_audio",
                    "audio_url": url,
                    "media_type": media_type,
                }),
                MediaSource::Base64 { media_type, data } => json!({
                    "type": "input_audio",
                    "audio": {"media_type": media_type, "data": data},
                }),
            }),
            ContentBlock::Video { source } => blocks.push(match source {
                MediaSource::Raw(raw) => json!({"type": "input_text", "text": json_string(raw)}),
                MediaSource::Url { url, media_type } => json!({
                    "type": "input_video",
                    "video_url": url,
                    "media_type": media_type,
                }),
                MediaSource::Base64 { media_type, data } => json!({
                    "type": "input_video",
                    "video": {"media_type": media_type, "data": data},
                }),
            }),
            ContentBlock::File { source } => {
                blocks.push(json!({"type": "input_file", "file": source}));
            }
            ContentBlock::Unknown { raw, .. } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "unknown content block encoded as text for Responses",
                )?;
                blocks.push(json!({"type": "input_text", "text": json_string(raw)}));
            }
            ContentBlock::Reasoning { .. }
            | ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_) => {}
        }
    }
    Ok(Value::Array(blocks))
}

// Encodes normalized tool definitions into Responses tool JSON.
fn encode_responses_tools(tools: &[ToolDefinition]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                let mut item = json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "parameters": tool.parameters,
                });
                if let Some(strict) = tool.strict {
                    item["strict"] = Value::Bool(strict);
                }
                item
            })
            .collect(),
    )
}

// Encodes normalized tool choice into Responses JSON.
fn encode_responses_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(Value::String("auto".to_string())),
        ToolChoice::Required => Some(Value::String("required".to_string())),
        ToolChoice::None => Some(Value::String("none".to_string())),
        ToolChoice::Tool { name } => Some(json!({"type": "function", "name": name})),
        ToolChoice::Raw(value) => Some(value.clone()),
    }
}

// Decodes one Responses output item into a normalized response output.
fn decode_responses_output_item(
    item: &Value,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<ResponseOutput>> {
    let Some(item) = item.as_object() else {
        push_lossy(
            diagnostics,
            policy,
            "Responses output item is not an object",
        )?;
        return Ok(None);
    };
    match item.get("type").and_then(Value::as_str) {
        Some("message") => Ok(Some(ResponseOutput {
            role: role_from_responses(item.get("role").and_then(Value::as_str)),
            content: decode_responses_content(item.get("content").unwrap_or(&Value::Null)),
            stop_reason: Some(StopReason::EndTurn),
        })),
        Some("function_call") => Ok(Some(ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: item.get("arguments").cloned().unwrap_or_else(|| json!({})),
            })],
            stop_reason: Some(StopReason::ToolUse),
        })),
        Some("reasoning") => Ok(Some(ResponseOutput {
            role: Role::Assistant,
            content: decode_responses_reasoning_item(item),
            stop_reason: None,
        })),
        _ => Ok(None),
    }
}

// Encodes normalized response outputs into Responses output items.
fn encode_responses_output(outputs: &[ResponseOutput]) -> Value {
    Value::Array(
        outputs
            .iter()
            .flat_map(|output| {
                let has_tool_calls = output
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolCall(_)));
                let text = text_from_blocks(&output.content, "");
                let reasoning = reasoning_text_from_blocks(&output.content, "\n");
                let mut items = Vec::new();

                if !reasoning.is_empty() {
                    items.push(encode_responses_reasoning_output(&reasoning));
                }

                if !text.is_empty() || (!has_tool_calls && reasoning.is_empty()) {
                    items.push(json!({
                        "type": "message",
                        "id": "msg_switchyard",
                        "status": "completed",
                        "role": role_to_responses(output.role),
                        "content": [{
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }],
                    }));
                }

                items.extend(output.content.iter().filter_map(|block| match block {
                    ContentBlock::ToolCall(call) => Some(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json_string_python_style(&call.arguments),
                    })),
                    _ => None,
                }));

                items
            })
            .collect(),
    )
}

// Encodes private reasoning as a separate Responses output item.
fn encode_responses_reasoning_output(text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_switchyard",
        "status": "completed",
        "content": [{
            "type": "reasoning_text",
            "text": text,
        }],
        "summary": [],
    })
}

// Serializes JSON with Python-like spacing to match legacy converter behavior.
fn json_string_python_style(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| value.clone()),
        Value::Array(items) => {
            let body = items
                .iter()
                .map(json_string_python_style)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{body}]")
        }
        Value::Object(object) => {
            let body = object
                .iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                    format!("{key}: {}", json_string_python_style(value))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{body}}}")
        }
    }
}

// Normalizes Responses usage fields.
fn decode_responses_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.and_then(Value::as_object) else {
        return Usage::default();
    };
    let input_tokens = value
        .get("input_tokens")
        .or_else(|| value.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let output_tokens = value
        .get("output_tokens")
        .or_else(|| value.get("completion_tokens"))
        .and_then(Value::as_u64);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                input_tokens
                    .zip(output_tokens)
                    .map(|(input, output)| input + output)
            }),
        reasoning_tokens: value
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .or_else(|| {
                value
                    .get("completion_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
            })
            .and_then(Value::as_u64),
    }
}

// Encodes normalized usage into Responses usage JSON.
fn encode_responses_usage(usage: &Usage) -> Value {
    let mut value = json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
        "total_tokens": usage
            .total_tokens
            .or_else(|| Some(usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0)))
            .unwrap_or(0),
    });
    if let Some(reasoning_tokens) = usage.reasoning_tokens {
        value["output_tokens_details"] = json!({
            "reasoning_tokens": reasoning_tokens,
        });
    }
    value
}
