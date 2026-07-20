// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for OpenAI Chat Completions request and response JSON.

use serde_json::{json, Map, Value};

use crate::codecs::common::{
    is_known_role_name, provider_extensions, reasoning_text_from_blocks, text_from_blocks,
};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, FileSource, ImageSource, InstructionBlock, LlmRequest,
    MediaSource, Message, OutputParams, ProviderExtensions, ReasoningParams, ResponseOutput, Role,
    SamplingParams, StopReason, ToolCall, ToolChoice, ToolDefinition, ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, json_string, object, push_lossy, stable_id,
    string_value, validate_request_capabilities,
};

/// Format codec for OpenAI Chat Completions payloads.
pub struct OpenAiChatCodec;

impl FormatCodec for OpenAiChatCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiChat.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = object(body, "$")?;
        let mut diagnostics = Vec::new();
        let mut request = LlmRequest {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            sampling: SamplingParams {
                temperature: body.get("temperature").and_then(Value::as_f64),
                top_p: body.get("top_p").and_then(Value::as_f64),
                top_k: None,
            },
            output: OutputParams {
                max_output_tokens: body
                    .get("max_completion_tokens")
                    .or_else(|| body.get("max_tokens"))
                    .and_then(Value::as_u64),
                response_format: body.get("response_format").cloned(),
            },
            reasoning: ReasoningParams {
                effort: body
                    .get("reasoning_effort")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                raw: None,
            },
            preservation: capture_request_preservation(
                WireFormat::OpenAiChat,
                &Value::Object(body.clone()),
                policy,
            ),
            ..LlmRequest::default()
        };

        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            let mut generated_id = 0;
            for (index, message) in messages.iter().enumerate() {
                let Some(message) = message.as_object() else {
                    push_lossy(
                        &mut diagnostics,
                        policy,
                        format!("OpenAI message at index {index} is not an object"),
                    )?;
                    continue;
                };
                let role = role_from_openai(
                    message.get("role").and_then(Value::as_str),
                    &format!("$.messages[{index}].role"),
                )?;
                let mut content = decode_openai_content(
                    message.get("content").unwrap_or(&Value::Null),
                    WireFormat::OpenAiChat,
                    &mut diagnostics,
                    policy,
                    format!("$.messages[{index}].content"),
                )?;
                prepend_openai_reasoning_blocks(&mut content, message);
                if role == Role::Assistant {
                    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                        if is_empty_text_only(&content) {
                            content.clear();
                        }
                        for tool_call in tool_calls {
                            generated_id += 1;
                            if let Some(call) =
                                decode_openai_tool_call(tool_call, generated_id, policy)?
                            {
                                content.push(ContentBlock::ToolCall(call));
                            }
                        }
                    }
                }
                if role == Role::Tool {
                    let text = content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let tool_call_id = message
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    request.messages.push(Message {
                        role: Role::User,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id,
                            content: vec![ContentBlock::Text { text }],
                            is_error: None,
                        })],
                    });
                    continue;
                }
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

        request.tools = decode_openai_tools(body.get("tools"), &mut diagnostics, policy)?;
        request.tool_choice = body.get("tool_choice").map(decode_openai_tool_choice);
        request.extensions.fields = provider_extensions(
            body,
            &[
                "model",
                "messages",
                "stream",
                "temperature",
                "top_p",
                "max_completion_tokens",
                "max_tokens",
                "response_format",
                "reasoning_effort",
                "tools",
                "tool_choice",
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
            exact_preserved_request(&request.preservation, WireFormat::OpenAiChat, policy)
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

        let mut messages = Vec::new();
        for instruction in &request.instructions {
            let role = match instruction.role {
                Role::Developer => "developer",
                _ => "system",
            };
            messages.push(json!({
                "role": role,
                "content": text_from_blocks(&instruction.content, "\n\n"),
            }));
        }
        for message in &request.messages {
            messages.extend(encode_message_to_openai(message, &mut diagnostics, policy)?);
        }
        body.insert("messages".to_string(), Value::Array(messages));

        if !request.tools.is_empty() {
            body.insert("tools".to_string(), encode_openai_tools(&request.tools));
            if let Some(choice) = &request.tool_choice {
                body.insert("tool_choice".to_string(), encode_openai_tool_choice(choice));
            }
        }
        if let Some(value) = request.output.max_output_tokens {
            body.insert("max_completion_tokens".to_string(), json!(value));
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
        if let Some(effort) = &request.reasoning.effort {
            body.insert(
                "reasoning_effort".to_string(),
                Value::String(effort.clone()),
            );
        }
        if let Some(format) = &request.output.response_format {
            body.insert("response_format".to_string(), format.clone());
        }
        copy_openai_chat_request_extensions(&mut body, &request.extensions.fields);

        let body = embed_preservation(Value::Object(body), &request.preservation, policy);
        Ok(EncodedRequest { body, diagnostics })
    }

    fn decode_response(
        &self,
        body: &Value,
        _policy: &TranslationPolicy,
    ) -> Result<DecodedResponse> {
        let object = object(body, "$")?;
        let mut response = AggLlmResponse {
            id: object
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: object
                .get("model")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            outputs: Vec::new(),
            usage: decode_openai_usage(object.get("usage")),
            extensions: ProviderExtensions {
                fields: provider_extensions(object, &["id", "model", "choices", "usage"]),
            },
            preservation: capture_response_preservation(
                WireFormat::OpenAiChat,
                &Value::Object(object.clone()),
                _policy,
            ),
        };
        if let Some(choice) = object
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(Value::as_object)
        {
            let message = choice
                .get("message")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let mut content = decode_openai_content(
                message.get("content").unwrap_or(&Value::Null),
                WireFormat::OpenAiChat,
                &mut Vec::new(),
                &TranslationPolicy::default(),
                "$.choices[0].message.content",
            )?;
            prepend_openai_reasoning_blocks(&mut content, &message);
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for (index, tool_call) in tool_calls.iter().enumerate() {
                    if let Some(call) = decode_openai_tool_call(
                        tool_call,
                        index + 1,
                        &TranslationPolicy::default(),
                    )? {
                        content.push(ContentBlock::ToolCall(call));
                    }
                }
            }
            response.outputs.push(ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: Some(map_openai_finish_reason(
                    choice.get("finish_reason").and_then(Value::as_str),
                )),
            });
        }

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
        if let Some(body) =
            exact_preserved_response(&response.preservation, WireFormat::OpenAiChat, _policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let output = response.first_output();
        let content = output
            .map(|output| text_from_blocks(&output.content, ""))
            .unwrap_or_default();
        let tool_calls = output
            .map(|output| {
                output
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json_string(&call.arguments),
                            },
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut message = json!({
            "role": "assistant",
            "content": if content.is_empty() && !tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(content)
            },
        });
        if let Some(reasoning) = output
            .map(|output| reasoning_text_from_blocks(&output.content, "\n"))
            .filter(|reasoning| !reasoning.is_empty())
        {
            message["reasoning_content"] = Value::String(reasoning);
        }
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }

        let body = json!({
            "id": response.id.clone().unwrap_or_else(|| "chatcmpl_switchyard".to_string()),
            "object": "chat.completion",
            "created": 0,
            "model": response.model.clone().unwrap_or_else(|| "unknown".to_string()),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": output
                    .and_then(|output| output.stop_reason)
                    .map(openai_finish_reason)
                    .unwrap_or("stop"),
            }],
            "usage": encode_openai_usage(&response.usage),
        });
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, _policy),
            diagnostics: Vec::new(),
        })
    }
}

// Pulls OpenAI-compatible reasoning fields into private reasoning IR blocks.
fn prepend_openai_reasoning_blocks(content: &mut Vec<ContentBlock>, object: &Map<String, Value>) {
    let reasoning = ["reasoning_content", "reasoning"]
        .into_iter()
        .filter_map(|key| object.get(key).and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .map(|text| ContentBlock::Reasoning {
            text: text.to_string(),
            signature: None,
        })
        .collect::<Vec<_>>();
    if reasoning.is_empty() {
        return;
    }

    let mut merged = reasoning;
    merged.append(content);
    *content = merged;
}

/// Decodes OpenAI role strings into normalized roles.
///
/// Unknown role strings are rejected with [`TranslationError::InvalidValue`] so
/// the router returns the same `invalid_value` error the provider would, rather
/// than silently coercing an invalid role to `user`. A missing role and
/// known-but-unmapped roles (e.g. the legacy `function` role) keep their
/// historical mapping to `user`. `path` points at the offending field.
pub(crate) fn role_from_openai(role: Option<&str>, path: &str) -> Result<Role> {
    match role {
        Some("system") => Ok(Role::System),
        Some("developer") => Ok(Role::Developer),
        Some("assistant") => Ok(Role::Assistant),
        Some("tool") => Ok(Role::Tool),
        None => Ok(Role::User),
        Some(other) if is_known_role_name(other) => Ok(Role::User),
        Some(other) => Err(TranslationError::unsupported_role(path, other)),
    }
}

/// Decodes OpenAI-style content into normalized content blocks.
pub(crate) fn decode_openai_content(
    content: &Value,
    provider: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    path: impl Into<String>,
) -> Result<Vec<ContentBlock>> {
    let path = path.into();
    match content {
        Value::Null => Ok(vec![ContentBlock::Text {
            text: String::new(),
        }]),
        Value::String(text) => Ok(vec![ContentBlock::Text { text: text.clone() }]),
        Value::Array(blocks) => {
            let mut content = Vec::new();
            for (index, block) in blocks.iter().enumerate() {
                let Some(block) = block.as_object() else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("content block at {path}[{index}] is not an object"),
                    )?;
                    continue;
                };
                match block.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        content.push(ContentBlock::Text {
                            text: block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        });
                    }
                    Some("refusal") => {
                        content.push(ContentBlock::Refusal {
                            text: block
                                .get("refusal")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        });
                    }
                    Some("image_url") | Some("input_image") => {
                        if let Some(source) = decode_image_source(block) {
                            content.push(ContentBlock::Image { source });
                        }
                    }
                    Some("file") | Some("input_file") => {
                        content.push(ContentBlock::File {
                            source: decode_file_source(block),
                        });
                    }
                    _ => content.push(ContentBlock::Unknown {
                        provider: provider.into(),
                        raw: Value::Object(block.clone()),
                    }),
                }
            }
            if content.is_empty() {
                Ok(vec![ContentBlock::Text {
                    text: String::new(),
                }])
            } else {
                Ok(content)
            }
        }
        other => Ok(vec![ContentBlock::Text {
            text: string_value(other).unwrap_or_default(),
        }]),
    }
}

/// Decodes OpenAI image block shapes into normalized image sources.
pub(crate) fn decode_image_source(block: &Map<String, Value>) -> Option<ImageSource> {
    if let Some(image_url) = block.get("image_url") {
        if let Some(url) = image_url.as_str() {
            return Some(ImageSource::Url {
                url: url.to_string(),
                detail: block
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            });
        }
        if let Some(payload) = image_url.as_object() {
            return payload
                .get("url")
                .and_then(Value::as_str)
                .map(|url| ImageSource::Url {
                    url: url.to_string(),
                    detail: payload
                        .get("detail")
                        .or_else(|| block.get("detail"))
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                });
        }
    }
    if let Some(image_url) = block.get("image_url").and_then(Value::as_str) {
        return Some(ImageSource::Url {
            url: image_url.to_string(),
            detail: block
                .get("detail")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        });
    }
    None
}

/// Decodes OpenAI file block shapes into normalized file sources.
pub(crate) fn decode_file_source(block: &Map<String, Value>) -> FileSource {
    if let Some(file) = block.get("file").and_then(Value::as_object) {
        if let Some(file_id) = file.get("file_id").and_then(Value::as_str) {
            return FileSource::FileId(file_id.to_string());
        }
        if let Some(file_data) = file.get("file_data").and_then(Value::as_str) {
            return FileSource::FileData {
                data: file_data.to_string(),
                filename: file
                    .get("filename")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            };
        }
        return FileSource::Raw(Value::Object(file.clone()));
    }
    if let Some(file_id) = block.get("file_id").and_then(Value::as_str) {
        return FileSource::FileId(file_id.to_string());
    }
    FileSource::Raw(Value::Object(block.clone()))
}

/// Decodes one OpenAI tool call into a normalized tool call.
pub(crate) fn decode_openai_tool_call(
    tool_call: &Value,
    generated_counter: usize,
    policy: &TranslationPolicy,
) -> Result<Option<ToolCall>> {
    let Some(tool_call) = tool_call.as_object() else {
        return Ok(None);
    };
    let function = tool_call
        .get("function")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let id = tool_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| match &policy.deterministic_ids {
            DeterministicIdPolicy::GenerateStable { prefix } => {
                stable_id(prefix, generated_counter)
            }
            DeterministicIdPolicy::Preserve => String::new(),
        });
    let arguments = function
        .get("arguments")
        .map(parse_arguments)
        .unwrap_or_else(|| json!({}));
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(Some(ToolCall {
        id,
        name,
        arguments,
    }))
}

/// Parses stringified tool-call arguments when possible.
pub(crate) fn parse_arguments(value: &Value) -> Value {
    match value {
        Value::String(text) => serde_json::from_str(text).unwrap_or_else(|_| json!({"raw": text})),
        other => other.clone(),
    }
}

/// Decodes OpenAI tool definitions into normalized tool definitions.
pub(crate) fn decode_openai_tools(
    tools: Option<&Value>,
    _diagnostics: &mut Vec<TranslationDiagnostic>,
    _policy: &TranslationPolicy,
) -> Result<Vec<ToolDefinition>> {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut definitions = Vec::new();
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| tool.clone());
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if name.is_empty() {
            continue;
        }
        definitions.push(ToolDefinition {
            name,
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
    Ok(definitions)
}

/// Decodes OpenAI tool-choice values into normalized policy.
pub(crate) fn decode_openai_tool_choice(value: &Value) -> ToolChoice {
    match value {
        Value::String(text) if text == "auto" => ToolChoice::Auto,
        Value::String(text) if text == "required" => ToolChoice::Required,
        Value::String(text) if text == "none" => ToolChoice::None,
        Value::Object(object) => object
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .map(|name| ToolChoice::Tool {
                name: name.to_string(),
            })
            .unwrap_or_else(|| ToolChoice::Raw(value.clone())),
        _ => ToolChoice::Raw(value.clone()),
    }
}

/// Encodes one normalized message into one or more OpenAI messages.
pub(crate) fn encode_message_to_openai(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    if message_has_tool_results(message) {
        return encode_message_with_tool_results_to_openai(message, diagnostics, policy);
    }

    Ok(vec![encode_message_without_tool_results_to_openai(
        message,
        diagnostics,
        policy,
    )?])
}

// Splits mixed content so OpenAI tool-result messages stay separate.
fn encode_message_with_tool_results_to_openai(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut pending_content = Vec::new();

    for block in &message.content {
        if let ContentBlock::ToolResult(result) = block {
            push_pending_openai_message(
                &mut out,
                message.role,
                &mut pending_content,
                diagnostics,
                policy,
            )?;
            out.push(json!({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": text_from_blocks(&result.content, " "),
            }));
        } else {
            pending_content.push(block.clone());
        }
    }

    push_pending_openai_message(
        &mut out,
        message.role,
        &mut pending_content,
        diagnostics,
        policy,
    )?;
    Ok(out)
}

// Flushes accumulated non-tool-result content as a normal OpenAI message.
fn push_pending_openai_message(
    out: &mut Vec<Value>,
    role: Role,
    pending_content: &mut Vec<ContentBlock>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    if pending_content.is_empty() {
        return Ok(());
    }

    let message = Message {
        role,
        content: std::mem::take(pending_content),
    };
    out.push(encode_message_without_tool_results_to_openai(
        &message,
        diagnostics,
        policy,
    )?);
    Ok(())
}

// Encodes a normalized message that has no tool-result blocks.
fn encode_message_without_tool_results_to_openai(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    let role = match message.role {
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        _ => "user",
    };
    let tool_calls = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json_string(&call.arguments),
                },
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    let content_blocks = message
        .content
        .iter()
        .filter(|block| {
            !matches!(
                block,
                ContentBlock::ToolCall(_) | ContentBlock::Reasoning { .. }
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut message_json = json!({
        "role": role,
        "content": encode_openai_content(&content_blocks, message.role, diagnostics, policy)?,
    });
    if !tool_calls.is_empty() {
        message_json["tool_calls"] = Value::Array(tool_calls);
        if message_json["content"] == Value::String(String::new()) {
            message_json["content"] = Value::Null;
        }
    }
    Ok(message_json)
}

// Checks whether any block in a message is a tool result.
fn message_has_tool_results(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult(_)))
}

// Copies Chat-compatible extension fields preserved in the IR.
fn copy_openai_chat_request_extensions(
    body: &mut Map<String, Value>,
    extensions: &Map<String, Value>,
) {
    for field in [
        "metadata",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "store",
        "stream_options",
        "top_logprobs",
        "user",
    ] {
        if let Some(value) = extensions.get(field) {
            body.entry(field.to_string())
                .or_insert_with(|| value.clone());
        }
    }
    if let Some(stop_sequences) = extensions.get("stop_sequences") {
        body.entry("stop").or_insert_with(|| stop_sequences.clone());
    }
}

// Detects placeholder empty text generated while decoding assistant tool calls.
fn is_empty_text_only(content: &[ContentBlock]) -> bool {
    matches!(content, [ContentBlock::Text { text }] if text.is_empty())
}

/// Encodes normalized content into OpenAI Chat content JSON.
pub(crate) fn encode_openai_content(
    content: &[ContentBlock],
    role: Role,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    let has_non_text = content.iter().any(|block| {
        matches!(
            block,
            ContentBlock::Image { .. }
                | ContentBlock::Audio { .. }
                | ContentBlock::Video { .. }
                | ContentBlock::File { .. }
                | ContentBlock::Unknown { .. }
        )
    });
    if !has_non_text {
        return Ok(Value::String(text_from_blocks(content, "\n")));
    }
    if role != Role::User {
        push_lossy(
            diagnostics,
            policy,
            "OpenAI Chat only supports text content for non-user messages",
        )?;
        return Ok(Value::String(text_from_blocks(content, "\n")));
    }
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text } => blocks.push(json!({"type": "text", "text": text})),
            ContentBlock::Refusal { text } => blocks.push(json!({"type": "text", "text": text})),
            ContentBlock::Image { source } => match openai_image_part(source) {
                Some(part) => blocks.push(part),
                None => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "OpenAI Chat codec could not map image content",
                    )?;
                    blocks.push(openai_text_part(&image_source_text(source)));
                }
            },
            ContentBlock::File { source } => match openai_file_part(source) {
                Some(part) => blocks.push(part),
                None => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "OpenAI Chat codec could not map file content",
                    )?;
                    blocks.push(openai_text_part(&file_source_text(source)));
                }
            },
            ContentBlock::Audio { source } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "OpenAI Chat codec does not have a stable audio request mapping yet",
                )?;
                blocks.push(openai_text_part(&media_source_text(source)));
            }
            ContentBlock::Video { source } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "OpenAI Chat codec does not have a stable video request mapping yet",
                )?;
                blocks.push(openai_text_part(&media_source_text(source)));
            }
            ContentBlock::Unknown { raw, .. } => {
                push_lossy(
                    diagnostics,
                    policy,
                    "unknown content block encoded as text for OpenAI Chat",
                )?;
                blocks.push(openai_text_part(&json_string(raw)));
            }
            ContentBlock::Reasoning { .. }
            | ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_) => {}
        }
    }
    Ok(Value::Array(blocks))
}

// Builds an OpenAI text content part.
fn openai_text_part(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

// Maps IR image sources to OpenAI Chat image content parts when possible.
fn openai_image_part(source: &ImageSource) -> Option<Value> {
    match source {
        ImageSource::Url { url, detail } => {
            let mut image_url = json!({"url": url});
            if let Some(detail) = detail {
                image_url["detail"] = Value::String(detail.clone());
            }
            Some(json!({"type": "image_url", "image_url": image_url}))
        }
        ImageSource::Base64 { media_type, data } => media_type.as_ref().map(|media_type| {
            json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{media_type};base64,{data}")},
            })
        }),
        ImageSource::Raw(raw) => openai_raw_image_part(raw),
    }
}

// Recognizes common raw image shapes emitted by Anthropic and Responses.
fn openai_raw_image_part(raw: &Value) -> Option<Value> {
    let object = raw.as_object()?;
    if let Some(url) = object.get("url").and_then(Value::as_str) {
        return Some(json!({"type": "image_url", "image_url": {"url": url}}));
    }
    if let Some(url) = object.get("image_url").and_then(Value::as_str) {
        return Some(json!({"type": "image_url", "image_url": {"url": url}}));
    }
    let data = object.get("data").and_then(Value::as_str)?;
    let media_type = object
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    Some(json!({
        "type": "image_url",
        "image_url": {"url": format!("data:{media_type};base64,{data}")},
    }))
}

// Converts image sources to deterministic text fallback content.
fn image_source_text(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url, detail } => json_string(&json!({
            "url": url,
            "detail": detail,
        })),
        ImageSource::Base64 { media_type, data } => json_string(&json!({
            "media_type": media_type,
            "data": data,
        })),
        ImageSource::Raw(raw) => json_string(raw),
    }
}

// Maps IR file sources to OpenAI Chat file content parts when possible.
fn openai_file_part(source: &FileSource) -> Option<Value> {
    match source {
        FileSource::FileId(file_id) => Some(json!({"type": "file", "file": {"file_id": file_id}})),
        FileSource::FileData { data, filename } => {
            let mut file = json!({"file_data": data});
            if let Some(filename) = filename {
                file["filename"] = Value::String(filename.clone());
            }
            Some(json!({"type": "file", "file": file}))
        }
        FileSource::Raw(_) => None,
    }
}

// Converts file sources to deterministic text fallback content.
fn file_source_text(source: &FileSource) -> String {
    match source {
        FileSource::FileId(file_id) => json_string(&json!({"file_id": file_id})),
        FileSource::FileData { data, filename } => json_string(&json!({
            "file_data": data,
            "filename": filename,
        })),
        FileSource::Raw(raw) => json_string(raw),
    }
}

// Converts unsupported media sources to deterministic text fallback content.
fn media_source_text(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url, media_type } => json_string(&json!({
            "url": url,
            "media_type": media_type,
        })),
        MediaSource::Base64 { media_type, data } => json_string(&json!({
            "media_type": media_type,
            "data": data,
        })),
        MediaSource::Raw(raw) => json_string(raw),
    }
}

/// Encodes normalized tool definitions into OpenAI tool JSON.
pub(crate) fn encode_openai_tools(tools: &[ToolDefinition]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                let mut function = json!({
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "parameters": tool.parameters,
                });
                if let Some(strict) = tool.strict {
                    function["strict"] = Value::Bool(strict);
                }
                json!({"type": "function", "function": function})
            })
            .collect(),
    )
}

/// Encodes normalized tool choice into OpenAI Chat JSON.
pub(crate) fn encode_openai_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::Required => Value::String("required".to_string()),
        ToolChoice::None => Value::String("none".to_string()),
        ToolChoice::Tool { name } => json!({"type": "function", "function": {"name": name}}),
        ToolChoice::Raw(value) => value.clone(),
    }
}

/// Normalizes OpenAI usage fields.
pub(crate) fn decode_openai_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.and_then(Value::as_object) else {
        return Usage::default();
    };
    Usage {
        input_tokens: value.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: value.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        reasoning_tokens: value
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .or_else(|| {
                value
                    .get("output_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
            })
            .and_then(Value::as_u64),
    }
}

/// Encodes normalized usage into OpenAI Chat usage JSON.
pub(crate) fn encode_openai_usage(usage: &Usage) -> Value {
    let mut value = json!({
        "prompt_tokens": usage.input_tokens.unwrap_or(0),
        "completion_tokens": usage.output_tokens.unwrap_or(0),
        "total_tokens": usage
            .total_tokens
            .or_else(|| Some(usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0)))
            .unwrap_or(0),
    });
    if let Some(reasoning_tokens) = usage.reasoning_tokens {
        value["completion_tokens_details"] = json!({
            "reasoning_tokens": reasoning_tokens,
        });
    }
    value
}

/// Maps OpenAI finish reasons to normalized stop reasons.
pub(crate) fn map_openai_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some("stop") | None => StopReason::EndTurn,
        _ => StopReason::Unknown,
    }
}

/// Maps normalized stop reasons back to OpenAI's vocabulary.
pub(crate) fn openai_finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "length",
        StopReason::ToolUse => "tool_calls",
        StopReason::ContentFilter => "content_filter",
        StopReason::EndTurn | StopReason::Unknown | StopReason::Error => "stop",
    }
}
