// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for OpenAI Responses request and response JSON.

use serde_json::{Map, Value, json};

use crate::codecs::common::{
    collect_responses_reasoning_text, encrypted_reasoning_data, encrypted_reasoning_item_id,
    is_anthropic_request, is_known_role_name, provider_extensions, reasoning_text_from_blocks,
    text_from_blocks,
};
use crate::codecs::openai_chat::{decode_file_source, decode_image_source};
use crate::codecs::openai_media::{
    ImagePayload, file_payload, file_source_text, image_payload, image_source_text,
};
use crate::codecs::structured_output::{decode_openai_schema_enforcement, encode_openai_format};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, FileSource, ImageSource, LlmRequest, MediaSource, Message,
    OutputParams, ProviderExtensions, ReasoningParams, ResponseOutput, ResponsesContentKind,
    ResponsesItemMetadata, ResponsesTextMetadata, Role, SamplingParams, StopReason, Tool, ToolCall,
    ToolChoice, ToolDefinition, ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, is_responses_native_input_item, json_string,
    push_lossy, stable_id, string_value, validate_request_capabilities,
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
        let response_format = decode_responses_text_format(body.get("text"));
        let mut request = LlmRequest {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned),
            output: OutputParams {
                max_output_tokens: body.get("max_output_tokens").and_then(Value::as_u64),
                is_schema_enforced: decode_openai_schema_enforcement(response_format.as_ref()),
                response_format,
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
            request.instructions.push(crate::llm::InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    responses: None,
                    text: instructions.to_string(),
                }],
            });
        }
        let mut custom_call_outputs = Vec::new();
        request.messages = decode_responses_input(
            body.get("input").unwrap_or(&Value::String(String::new())),
            &mut custom_call_outputs,
            &mut diagnostics,
            policy,
        )?;
        let mut tool_namespaces = Map::new();
        let mut custom_tools = Map::new();
        request.tools =
            decode_responses_tools(body.get("tools"), &mut tool_namespaces, &mut custom_tools)
                .into_iter()
                .map(Tool::from)
                .collect();
        // Responses-lite clients (Codex with a GPT-5 model) carry the tool definitions inside
        // `input` as an `additional_tools` developer item instead of top-level `tools`. Those
        // definitions are the request's tools; the item itself is kept verbatim so the request
        // can be re-emitted in the shape the client used.
        let mut additional_tools = Vec::new();
        if let Some(items) = body.get("input").and_then(Value::as_array) {
            for item in items {
                if let Some(item) = item.as_object()
                    && item.get("type").and_then(Value::as_str) == Some("additional_tools")
                    && let Some(tools) = item.get("tools").and_then(Value::as_array)
                {
                    request.tools.extend(
                        decode_responses_tools(
                            Some(&Value::Array(tools.clone())),
                            &mut tool_namespaces,
                            &mut custom_tools,
                        )
                        .into_iter()
                        .map(Tool::from),
                    );
                    additional_tools.extend(tools.iter().cloned());
                }
            }
        }
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
        crate::codex_namespaces::attach_tool_namespaces(&mut request.extensions, tool_namespaces);
        crate::codex_custom_tools::attach_custom_tools(&mut request.extensions, custom_tools);
        crate::codex_custom_tools::attach_custom_call_outputs(
            &mut request.extensions,
            custom_call_outputs,
        );
        crate::codex_custom_tools::attach_additional_tools(
            &mut request.extensions,
            additional_tools,
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
            exact_preserved_request(&request.preservation, WireFormat::OpenAiResponses, policy)
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
        let instructions = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } | ContentBlock::Refusal { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !instructions.is_empty() {
            body.insert(
                "instructions".to_string(),
                Value::String(instructions.join("\n\n")),
            );
        }
        body.insert(
            "input".to_string(),
            encode_responses_input(
                &request.messages,
                &mut diagnostics,
                policy,
                crate::codex_namespaces::tool_namespaces(&request.extensions),
                &crate::codex_custom_tools::custom_tool_names(&request.extensions),
                crate::codex_custom_tools::custom_call_outputs(&request.extensions),
            )?,
        );
        if let Some(additional) = crate::codex_custom_tools::additional_tools(&request.extensions) {
            let has_native_item = request.messages.iter().any(|message| {
                message.responses.as_ref().is_some_and(|source| {
                    source.fields.get("type").and_then(Value::as_str) == Some("additional_tools")
                })
            });
            if !has_native_item {
                // Saved inline tools remain in `input` when no native item supplies them.
                // The top-level `tools` field remains absent, as in the caller's request.
                let item = json!({
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": additional,
                });
                match body.get_mut("input") {
                    Some(Value::Array(input)) => input.insert(0, item),
                    _ => body
                        .insert("input".to_string(), Value::Array(vec![item]))
                        .map_or((), |_| ()),
                }
            }
        } else if !request.tools.is_empty() {
            body.insert(
                "tools".to_string(),
                encode_responses_tools(
                    &request.tools,
                    crate::codex_namespaces::tool_namespaces(&request.extensions),
                    crate::codex_custom_tools::custom_tools(&request.extensions),
                )?,
            );
        }
        if let Some(choice) = &request.tool_choice
            && let Some(choice) = encode_responses_tool_choice(
                choice,
                crate::codex_namespaces::tool_namespaces(&request.extensions),
            )
        {
            body.insert("tool_choice".to_string(), choice);
        }
        if let Some(max_output_tokens) = request.output.max_output_tokens {
            body.insert("max_output_tokens".to_string(), json!(max_output_tokens));
        }
        if let Some(response_format) =
            encode_openai_format(&request.output, &mut diagnostics, policy)?
        {
            body.insert(
                "text".to_string(),
                json!({"format": encode_responses_text_format(&response_format)}),
            );
        }
        // An Anthropic request's raw reasoning is its `thinking` object, which Responses
        // does not accept. Keep only whether thinking is disabled.
        let mut reasoning = if is_anthropic_request(request) {
            let mut reasoning = Map::new();
            if request
                .reasoning
                .raw
                .as_ref()
                .and_then(|thinking| thinking.get("type"))
                .and_then(Value::as_str)
                == Some("disabled")
            {
                reasoning.insert("effort".to_string(), json!("none"));
            }
            reasoning
        } else {
            request
                .reasoning
                .raw
                .as_ref()
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default()
        };
        if let Some(effort) = &request.reasoning.effort {
            reasoning.insert("effort".to_string(), json!(effort));
        }
        if !reasoning.is_empty() {
            body.insert("reasoning".to_string(), Value::Object(reasoning));
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
        copy_responses_request_extensions(&mut body, &request.extensions.fields);

        let body = embed_preservation(Value::Object(body), &request.preservation, policy);
        Ok(EncodedRequest { body, diagnostics })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let body = crate::util::object(body, "$")?;
        // Providers can return HTTP 200 when generation fails. Check `status`
        // before treating an empty `output` as a completed turn.
        if body.get("status").and_then(Value::as_str) == Some("failed") {
            return Err(TranslationError::UpstreamFailure {
                error: body
                    .get("error")
                    .filter(|error| error.is_object())
                    .cloned()
                    .unwrap_or_else(|| {
                        json!({ "message": "provider reported status \"failed\" without error details" })
                    }),
            });
        }
        let mut diagnostics = Vec::new();
        let mut content = Vec::new();
        let mut role = Role::Assistant;
        let mut stop_reason = None;
        let mut has_output = false;
        let mut url_citations = Vec::new();
        let mut text_offset = 0;
        if let Some(items) = body.get("output").and_then(Value::as_array) {
            for item in items {
                if let Some(output) = decode_responses_output_item(item, &mut diagnostics, policy)?
                {
                    has_output = true;
                    role = output.role;
                    for mut citation in output.url_citations {
                        citation.start_index += text_offset;
                        citation.end_index += text_offset;
                        url_citations.push(citation);
                    }
                    text_offset += output
                        .content
                        .iter()
                        .map(|block| match block {
                            ContentBlock::Text { text, .. } | ContentBlock::Refusal { text } => {
                                text.chars().count()
                            }
                            _ => 0,
                        })
                        .sum::<usize>();
                    content.extend(output.content);
                    if output.stop_reason.is_some() {
                        stop_reason = output.stop_reason;
                    }
                }
            }
        }
        // The truncation signal is on the response, not the output items.
        if body.get("status").and_then(Value::as_str) == Some("incomplete")
            && body
                .get("incomplete_details")
                .and_then(|details| details.get("reason"))
                .and_then(Value::as_str)
                == Some("max_output_tokens")
        {
            stop_reason = Some(StopReason::MaxTokens);
        }
        let outputs = vec![if has_output {
            ResponseOutput {
                url_citations,
                role,
                content,
                stop_reason,
            }
        } else {
            ResponseOutput {
                url_citations: Vec::new(),
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    responses: None,
                    text: String::new(),
                }],
                stop_reason: Some(StopReason::EndTurn),
            }
        }];
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
        let is_truncated = matches!(
            response
                .first_output()
                .and_then(|output| output.stop_reason),
            Some(StopReason::MaxTokens)
        );
        let status = if is_truncated {
            "incomplete"
        } else {
            "completed"
        };
        let incomplete_details = is_truncated.then(|| json!({ "reason": "max_output_tokens" }));
        Ok(EncodedResponse {
            body: embed_preservation(
                json!({
                    "id": response.id.clone().unwrap_or_else(|| "resp_switchyard".to_string()),
                    "object": "response",
                    "created_at": 0,
                    "model": response.model.clone().unwrap_or_else(|| "unknown".to_string()),
                    "status": status,
                    "incomplete_details": incomplete_details,
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

/// This decoder retains native Responses input items in their supplied order.
fn decode_responses_input(
    value: &Value,
    custom_call_outputs: &mut Vec<String>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Message>> {
    match value {
        Value::String(text) => {
            let mut message = Message::text(Role::User, text);
            message.responses = Some(ResponsesItemMetadata {
                fields: Map::new(),
                content_kind: ResponsesContentKind::InputString,
            });
            Ok(vec![message])
        }
        Value::Array(items) => {
            let mut messages = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let Some(item) = item.as_object() else {
                    push_lossy(
                        diagnostics,
                        policy,
                        format!("Responses input item {index} is not an object"),
                    )?;
                    continue;
                };
                let kind = match item.get("type") {
                    Some(Value::String(kind)) => Some(kind.as_str()),
                    Some(_) => {
                        return Err(TranslationError::InvalidType {
                            path: format!("$.input[{index}].type"),
                            expected: "a string",
                        });
                    }
                    None => None,
                };
                let is_message = kind == Some("message")
                    || (kind.is_none()
                        && item.contains_key("role")
                        && item.contains_key("content"));
                let mut fields = item.clone();
                let mut content_kind = ResponsesContentKind::Missing;
                let (role, content) = if is_message {
                    let role = request_role_from_responses(
                        item.get("role").and_then(Value::as_str),
                        &format!("$.input[{index}].role"),
                    )?;
                    let value = item.get("content").unwrap_or(&Value::Null);
                    if is_responses_native_input_item(value) {
                        return Err(TranslationError::InvalidValue {
                            path: format!("$.input[{index}].content.type"),
                            message: "native Responses items must be top-level input items".into(),
                        });
                    }
                    if let Some(content_index) = value
                        .as_array()
                        .and_then(|blocks| blocks.iter().position(is_responses_native_input_item))
                    {
                        return Err(TranslationError::InvalidValue {
                            path: format!("$.input[{index}].content[{content_index}].type"),
                            message: "native Responses items must be top-level input items".into(),
                        });
                    }
                    content_kind = responses_content_kind(item.get("content"));
                    fields.remove("role");
                    fields.remove("content");
                    (role, decode_responses_content(value, true))
                } else {
                    match kind {
                        Some("reasoning") => {
                            fields.retain(|key, _| key == "type");
                            (Role::Assistant, decode_responses_reasoning_item(item, true))
                        }
                        Some("function_call" | "custom_tool_call") => {
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
                            let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                            let name = item
                                .get("namespace")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                                .map_or_else(
                                    || name.to_string(),
                                    |namespace| {
                                        crate::codex_namespaces::qualified_tool_name(
                                            namespace, name,
                                        )
                                    },
                                );
                            let arguments = if kind == Some("custom_tool_call") {
                                json!({crate::codex_custom_tools::INPUT_ARGUMENT: item.get("input").and_then(Value::as_str).unwrap_or_default()})
                            } else {
                                item.get("arguments").cloned().unwrap_or_else(|| json!({}))
                            };
                            for key in ["call_id", "name", "arguments", "input"] {
                                fields.remove(key);
                            }
                            (
                                Role::Assistant,
                                vec![ContentBlock::ToolCall(ToolCall {
                                    id,
                                    name,
                                    arguments,
                                })],
                            )
                        }
                        Some("function_call_output" | "custom_tool_call_output") => {
                            let tool_call_id = item
                                .get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            if kind == Some("custom_tool_call_output") {
                                custom_call_outputs.push(tool_call_id.clone());
                            }
                            content_kind = responses_content_kind(item.get("output"));
                            let content = match item.get("output") {
                                Some(value @ Value::Array(_)) => {
                                    decode_responses_content(value, true)
                                }
                                value => vec![ContentBlock::text(
                                    value.map(json_string).unwrap_or_default(),
                                )],
                            };
                            fields.remove("call_id");
                            fields.remove("output");
                            (
                                Role::User,
                                vec![ContentBlock::ToolResult(ToolResult {
                                    tool_call_id,
                                    content,
                                    is_error: None,
                                })],
                            )
                        }
                        None => {
                            return Err(TranslationError::InvalidValue {
                                path: format!("$.input[{index}].type"),
                                message: "missing type discriminator on a non-message input item"
                                    .into(),
                            });
                        }
                        _ => {
                            fields.retain(|key, _| key == "type");
                            let role = match kind {
                                Some(
                                    "shell_call"
                                    | "apply_patch_call"
                                    | "computer_call"
                                    | "image_generation_call",
                                ) => Role::Assistant,
                                Some(
                                    "shell_call_output"
                                    | "apply_patch_call_output"
                                    | "computer_call_output",
                                ) => Role::Tool,
                                Some("additional_tools") => Role::Developer,
                                _ => Role::User,
                            };
                            (
                                role,
                                vec![ContentBlock::Unknown {
                                    provider: WireFormat::OpenAiResponses.into(),
                                    raw: Value::Object(item.clone()),
                                }],
                            )
                        }
                    }
                };
                messages.push(Message {
                    role,
                    content,
                    responses: Some(ResponsesItemMetadata {
                        fields,
                        content_kind,
                    }),
                });
            }
            Ok(messages)
        }
        _ => Err(TranslationError::InvalidType {
            path: "$.input".into(),
            expected: "string or array",
        }),
    }
}

fn responses_content_kind(value: Option<&Value>) -> ResponsesContentKind {
    match value {
        None => ResponsesContentKind::Missing,
        Some(Value::Null) => ResponsesContentKind::Null,
        Some(Value::Array(_)) => ResponsesContentKind::Array,
        _ => ResponsesContentKind::String,
    }
}

// Request reconstruction needs the original reasoning fields as well as decoded text.
fn decode_responses_reasoning_item(
    item: &Map<String, Value>,
    preserve_metadata: bool,
) -> Vec<ContentBlock> {
    if let Some((text, signature)) = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(super::decode_anthropic_thinking)
    {
        return vec![ContentBlock::Reasoning {
            text,
            signature: Some(signature),
            details: Vec::new(),
        }];
    }
    let mut parts = Vec::new();
    collect_responses_reasoning_text(item.get("content"), &mut parts);
    collect_responses_reasoning_text(item.get("summary"), &mut parts);
    if let Some(text) = item.get("text").and_then(Value::as_str) {
        parts.push(text.to_string());
    }
    // The original item retains summary boundaries and any provider-issued encrypted content.
    let details = if preserve_metadata
        || item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some()
    {
        vec![Value::Object(item.clone())]
    } else {
        Vec::new()
    };
    vec![ContentBlock::Reasoning {
        text: parts.join("\n"),
        signature: None,
        details,
    }]
}

// Request decoding keeps metadata on each text block so it stays with edits and reordering.
fn decode_responses_content(value: &Value, preserve_metadata: bool) -> Vec<ContentBlock> {
    match value {
        Value::String(text) => vec![ContentBlock::text(text.clone())],
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
                            responses: preserve_metadata.then(|| ResponsesTextMetadata {
                                fields: block
                                    .iter()
                                    .filter(|(key, _)| key.as_str() != "text")
                                    .map(|(key, value)| (key.clone(), value.clone()))
                                    .collect(),
                            }),
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
                            details: Vec::new(),
                        });
                    }
                    Some("input_image") => {
                        if let Some(source) = decode_image_source(block) {
                            out.push(ContentBlock::Image { source });
                        }
                    }
                    Some("input_audio") => out.push(ContentBlock::Audio {
                        source: MediaSource::Raw(Value::Object(block.clone())),
                    }),
                    Some("input_file") => out.push(ContentBlock::File {
                        source: decode_file_source(block),
                    }),
                    _ => out.push(ContentBlock::Unknown {
                        provider: WireFormat::OpenAiResponses.into(),
                        raw: Value::Object(block.clone()),
                    }),
                }
            }
            if out.is_empty() && !preserve_metadata {
                out.push(ContentBlock::text(String::new()));
            }
            out
        }
        Value::Null => vec![ContentBlock::Text {
            responses: None,
            text: String::new(),
        }],
        other => vec![ContentBlock::Text {
            responses: None,
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
// Decodes Responses tool shapes, including Codex-style tool entries.
//
// Codex groups tools into non-standard ``namespace`` containers that
// OpenAI-compatible upstreams do not accept. Each child is exposed under a
// `<namespace>__<tool>` name, and `namespaces` records the mapping so the
// response can split it back apart.
fn decode_responses_tools(
    value: Option<&Value>,
    namespaces: &mut Map<String, Value>,
    custom_tools: &mut Map<String, Value>,
) -> Vec<ToolDefinition> {
    let Some(tools) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if tool.get("type").and_then(Value::as_str) == Some("namespace") {
            // An unnamed container carries nothing to dispatch on, so its
            // children stay bare rather than gaining an empty prefix.
            let container = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty());
            let description = tool.get("description").and_then(Value::as_str);
            for mut child in decode_responses_tools(tool.get("tools"), namespaces, custom_tools) {
                // A nested container already qualified its own children, and the
                // innermost name is the one that identifies the tool.
                let already_qualified = namespaces.contains_key(&child.name);
                if let Some(container) = container
                    && !already_qualified
                {
                    let qualified =
                        crate::codex_namespaces::qualified_tool_name(container, &child.name);
                    crate::codex_namespaces::record_tool_namespace(
                        namespaces,
                        &qualified,
                        container,
                        description,
                    );
                    child.name = qualified;
                }
                out.push(child);
            }
        } else if tool.get("type").and_then(Value::as_str) == Some("custom") {
            // A freeform tool takes a raw string, not JSON arguments. The IR sees it as a
            // function with a single `input` argument; the verbatim definition is kept so a
            // Responses upstream still receives the freeform tool. The definition is keyed by
            // the tool's own name: Codex defines its freeform tools at the top level, so a
            // custom tool nested in a namespace container (which would be renamed above) is
            // not restored verbatim and goes upstream as the equivalent function tool.
            if let Some(name) = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                custom_tools.insert(name.to_string(), Value::Object(tool.clone()));
                out.push(ToolDefinition {
                    name: name.to_string(),
                    description: tool
                        .get("description")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    parameters: crate::codex_custom_tools::input_schema(),
                    strict: None,
                });
            }
        } else if tool.get("type").and_then(Value::as_str) == Some("function") {
            if let Some(function) = tool.get("function").and_then(Value::as_object) {
                if let Some(name) = function.get("name").and_then(Value::as_str)
                    && !name.is_empty()
                {
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
            // A forced tool has to name the same thing its definition does, or
            // the upstream rejects a choice naming a tool it was never offered.
            object.get("name").and_then(Value::as_str).map(|name| {
                match object
                    .get("namespace")
                    .and_then(Value::as_str)
                    .filter(|namespace| !namespace.is_empty())
                {
                    Some(namespace) => ToolChoice::Tool {
                        name: crate::codex_namespaces::qualified_tool_name(namespace, name),
                    },
                    None => ToolChoice::Tool {
                        name: name.to_string(),
                    },
                }
            })
        }
        Value::Object(object)
            if object.get("type").and_then(Value::as_str) == Some("allowed_tools") =>
        {
            Some(ToolChoice::Raw(value.clone()))
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

// Flattens Chat-compatible JSON schema fields into the Responses text format shape.
fn encode_responses_text_format(response_format: &Value) -> Value {
    let Some(format) = response_format.as_object() else {
        return response_format.clone();
    };
    let Some(response_type) = format
        .get("type")
        .filter(|value| value.as_str() == Some("json_schema"))
    else {
        return response_format.clone();
    };

    let Some(Value::Object(json_schema)) = format.get("json_schema") else {
        return response_format.clone();
    };
    if !matches!(json_schema.get("name"), Some(Value::String(_)))
        || !matches!(json_schema.get("schema"), Some(Value::Object(_)))
        || json_schema
            .get("description")
            .is_some_and(|value| !value.is_string())
        || json_schema
            .get("strict")
            .is_some_and(|value| !value.is_boolean())
    {
        return response_format.clone();
    }

    let mut output = Map::new();
    output.insert("type".to_string(), response_type.clone());
    for field in ["name", "description", "schema", "strict"] {
        if let Some(value) = json_schema.get(field) {
            output.insert(field.to_string(), value.clone());
        }
    }
    Value::Object(output)
}

// Native metadata supplies fields outside the decoded role and content.
// This encoder reads edited values from the current role and content.
fn encode_native_responses_item(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    let Some(source) = &message.responses else {
        return Ok(None);
    };
    let kind = source.fields.get("type").and_then(Value::as_str);
    let mut item = source.fields.clone();
    match (kind, message.content.as_slice()) {
        (Some("function_call" | "custom_tool_call"), [ContentBlock::ToolCall(call)])
            if message.role == Role::Assistant =>
        {
            item.insert("call_id".into(), Value::String(call.id.clone()));
            let namespace = item
                .get("namespace")
                .and_then(Value::as_str)
                .filter(|namespace| !namespace.is_empty());
            let name = if let Some(namespace) = namespace {
                if let Some(name) = call.name.strip_prefix(&format!(
                    "{namespace}{}",
                    crate::codex_namespaces::NAMESPACE_SEPARATOR
                )) {
                    name.to_owned()
                } else {
                    item.remove("namespace");
                    call.name.clone()
                }
            } else {
                call.name.clone()
            };
            item.insert("name".into(), Value::String(name));
            if kind == Some("custom_tool_call") {
                item.insert(
                    "input".into(),
                    Value::String(crate::codex_custom_tools::input_from_arguments(
                        &call.arguments,
                    )),
                );
            } else {
                item.insert(
                    "arguments".into(),
                    Value::String(json_string(&call.arguments)),
                );
            }
        }
        (
            Some("function_call_output" | "custom_tool_call_output"),
            [ContentBlock::ToolResult(result)],
        ) if matches!(message.role, Role::User | Role::Tool) => {
            item.insert("call_id".into(), Value::String(result.tool_call_id.clone()));
            if let Some(output) = encode_native_responses_content(
                &result.content,
                source.content_kind,
                diagnostics,
                policy,
            )? {
                item.insert("output".into(), output);
            }
        }
        (
            Some("reasoning"),
            [
                ContentBlock::Reasoning {
                    text,
                    signature: None,
                    details,
                },
            ],
        ) if message.role == Role::Assistant => {
            let Some(original_item) = details.iter().find_map(|value| {
                value
                    .as_object()
                    .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
            }) else {
                return Ok(None);
            };
            item = original_item.clone();
            let original = decode_responses_reasoning_item(original_item, false);
            if !matches!(original.as_slice(), [ContentBlock::Reasoning { text: original_text, .. }] if original_text == text)
            {
                // Rewritten reasoning cannot retain an encrypted payload or ID for the old content.
                for key in ["id", "content", "text", "encrypted_content"] {
                    item.remove(key);
                }
                item.insert(
                    "summary".into(),
                    json!([{"type":"summary_text", "text":text}]),
                );
            }
        }
        (Some("message") | None, _) => {
            if message.content.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolCall(_)
                        | ContentBlock::ToolResult(_)
                        | ContentBlock::Reasoning { .. }
                )
            }) {
                return Ok(None);
            }
            item.insert(
                "role".into(),
                Value::String(role_to_responses(message.role).into()),
            );
            if let Some(content) = encode_native_responses_content(
                &message.content,
                source.content_kind,
                diagnostics,
                policy,
            )? {
                item.insert("content".into(), content);
            }
        }
        (_, [ContentBlock::Unknown { provider, raw }])
            if provider.as_str() == WireFormat::OpenAiResponses.as_str() =>
        {
            return Ok(Some(raw.clone()));
        }
        _ => return Ok(None),
    }
    Ok(Some(Value::Object(item)))
}

// The saved representation keeps empty strings, empty arrays, null, and omitted fields distinct.
fn encode_native_responses_content(
    content: &[ContentBlock],
    kind: ResponsesContentKind,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    let is_empty = content.is_empty()
        || matches!(content, [ContentBlock::Text { text, .. }] if text.is_empty());
    match kind {
        ResponsesContentKind::Missing if is_empty => return Ok(None),
        ResponsesContentKind::Null if is_empty => return Ok(Some(Value::Null)),
        ResponsesContentKind::String | ResponsesContentKind::InputString => {
            if let [ContentBlock::Text { text, .. }] = content {
                return Ok(Some(Value::String(text.clone())));
            }
        }
        _ => {}
    }
    let mut blocks = Vec::new();
    for block in content {
        match encode_responses_content(std::slice::from_ref(block), diagnostics, policy)? {
            Value::Array(items) => blocks.extend(items),
            Value::String(text) => blocks.push(json!({"type":"input_text","text":text})),
            _ => {}
        }
    }
    Ok(Some(Value::Array(blocks)))
}

// Encodes normalized messages into the Responses `input` shape.
fn encode_responses_input(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    namespaces: Option<&Map<String, Value>>,
    custom_tools: &std::collections::HashSet<String>,
    mut custom_call_ids: std::collections::HashSet<String>,
) -> Result<Value> {
    if let [message] = messages
        && message
            .responses
            .as_ref()
            .is_some_and(|source| source.content_kind == ResponsesContentKind::InputString)
        && message.role == Role::User
        && let [ContentBlock::Text { text, .. }] = message.content.as_slice()
    {
        return Ok(Value::String(text.clone()));
    }
    // `custom_call_ids` starts with IDs from stored-state custom outputs and gains local call IDs.
    // These IDs determine the type of outputs that need reconstruction.
    // New IR input uses an item list, even for one user text, for Codex compatibility.
    // The early return above preserves string input decoded from a native Responses request.
    let mut encoded = Vec::new();
    for message in messages {
        if let Some(item) = encode_native_responses_item(message, diagnostics, policy)? {
            if item.get("type").and_then(Value::as_str) == Some("custom_tool_call")
                && let Some(id) = item.get("call_id").and_then(Value::as_str)
            {
                custom_call_ids.insert(id.to_owned());
            }
            encoded.push(item);
            continue;
        }

        // Native items kept as opaque Responses blocks must retain their position in `input`.
        if let [ContentBlock::Unknown { provider, raw }] = message.content.as_slice()
            && provider.as_str() == WireFormat::OpenAiResponses.as_str()
            && is_responses_native_input_item(raw)
        {
            encoded.push(raw.clone());
            continue;
        }
        // Anthropic-signed thinking cannot be sent as Responses input.
        let content = message
            .content
            .iter()
            .filter(|block| {
                !matches!(
                    block,
                    ContentBlock::Reasoning {
                        signature: Some(_),
                        ..
                    }
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        if content.is_empty() {
            continue;
        }
        if content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
            )
        }) {
            let mut visible_content = Vec::new();
            for block in &content {
                if let Some(item) = encode_responses_special_input(
                    block,
                    namespaces,
                    custom_tools,
                    &mut custom_call_ids,
                    diagnostics,
                    policy,
                )? {
                    if !visible_content.is_empty() {
                        let content =
                            encode_responses_content(&visible_content, diagnostics, policy)?;
                        encoded.push(json!({
                            "type": "message",
                            "role": role_to_responses(message.role),
                            "content": content,
                        }));
                        visible_content.clear();
                    }
                    encoded.push(item);
                } else if !matches!(block, ContentBlock::Reasoning { .. }) {
                    visible_content.push(block.clone());
                }
            }
            if !visible_content.is_empty() {
                let content = encode_responses_content(&visible_content, diagnostics, policy)?;
                encoded.push(json!({
                    "type": "message",
                    "role": role_to_responses(message.role),
                    "content": content,
                }));
            }
            continue;
        }
        let mut visible_content = Vec::new();
        let mut emitted_special = false;
        let mut omitted_reasoning = false;
        for block in &content {
            if let Some(item) = encode_responses_special_input(
                block,
                namespaces,
                custom_tools,
                &mut custom_call_ids,
                diagnostics,
                policy,
            )? {
                encoded.push(item);
                emitted_special = true;
            } else if !matches!(block, ContentBlock::Reasoning { .. }) {
                visible_content.push(block.clone());
            } else {
                omitted_reasoning = true;
            }
        }
        if !visible_content.is_empty() || (!emitted_special && !omitted_reasoning) {
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
fn encode_responses_special_input(
    block: &ContentBlock,
    namespaces: Option<&Map<String, Value>>,
    custom_tools: &std::collections::HashSet<String>,
    custom_call_ids: &mut std::collections::HashSet<String>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    Ok(match block {
        ContentBlock::Reasoning {
            text,
            signature: None,
            details,
        } => encode_responses_reasoning_input(text, details),
        ContentBlock::ToolCall(call) if custom_tools.contains(&call.name) => {
            // A freeform tool call replays as `custom_tool_call` with its raw input.
            custom_call_ids.insert(call.id.clone());
            Some(json!({
                "type": "custom_tool_call",
                "call_id": call.id,
                "name": call.name,
                "input": crate::codex_custom_tools::input_from_arguments(&call.arguments),
            }))
        }
        ContentBlock::ToolCall(call) => {
            // A Responses client dispatches on name plus namespace, so undo the
            // qualification this request applied for a flat upstream.
            let (name, namespace) = namespaces
                .and_then(|namespaces| {
                    crate::codex_namespaces::split_qualified_name(namespaces, &call.name)
                })
                .map_or((call.name.as_str(), None), |(name, namespace)| {
                    (name, Some(namespace))
                });
            let mut item = json!({
                "type": "function_call",
                "call_id": call.id,
                "name": name,
                "arguments": json_string(&call.arguments),
            });
            if let Some(namespace) = namespace {
                item["namespace"] = Value::String(namespace.to_string());
            }
            Some(item)
        }
        ContentBlock::ToolResult(result) => {
            // A result answering a freeform call replays as `custom_tool_call_output`; anything
            // else is a `function_call_output`.
            let item = json!({
                "type": if custom_call_ids.contains(&result.tool_call_id) {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                },
                "call_id": result.tool_call_id,
                "output": encode_responses_tool_output(&result.content, diagnostics, policy)?,
            });
            Some(item)
        }
        _ => None,
    })
}

// This helper encodes reasoning for input history when native item reconstruction does not apply.
// Response output items may carry `content`, but this encoder omits it from input items.
fn encode_responses_reasoning_input(text: &str, details: &[Value]) -> Option<Value> {
    for detail in details {
        let Some(detail) = detail.as_object() else {
            continue;
        };
        if detail.get("type").and_then(Value::as_str) != Some("reasoning") {
            continue;
        }
        // Edited text invalidates the saved ID and encrypted content, even in a mixed message.
        let original = decode_responses_reasoning_item(detail, false);
        if !matches!(original.as_slice(), [ContentBlock::Reasoning { text: original_text, .. }] if original_text == text)
        {
            break;
        }
        let mut item = Map::new();
        item.insert("type".to_string(), Value::String("reasoning".to_string()));
        for field in ["id", "summary", "encrypted_content"] {
            if let Some(value) = detail.get(field) {
                item.insert(field.to_string(), value.clone());
            }
        }
        item.entry("summary".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        return Some(Value::Object(item));
    }

    if text.is_empty() {
        return None;
    }
    Some(json!({
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    }))
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
            ContentBlock::Text {
                responses: None,
                ..
            } | ContentBlock::Refusal { .. }
                | ContentBlock::Reasoning { .. }
        )
    });
    if !has_non_text {
        return Ok(Value::String(text_from_blocks(content, "\n")));
    }
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text, responses } => {
                let mut fields = responses
                    .as_ref()
                    .map(|source| source.fields.clone())
                    .unwrap_or_default();
                fields.entry("type").or_insert_with(|| json!("input_text"));
                fields.insert("text".into(), Value::String(text.clone()));
                blocks.push(Value::Object(fields));
            }
            ContentBlock::Refusal { text } => {
                blocks.push(json!({"type": "refusal", "refusal": text}));
            }
            ContentBlock::Image { source } => match responses_image_part(source) {
                Some(part) => blocks.push(part),
                None => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Responses codec could not map image content",
                    )?;
                    blocks.push(json!({"type": "input_text", "text": image_source_text(source)}));
                }
            },
            ContentBlock::Audio { source } => {
                blocks.push(crate::codecs::openai_media::audio_part(source)?);
            }
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
            ContentBlock::File { source } => match responses_file_part(source) {
                Some(part) => blocks.push(part),
                None => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Responses codec could not map file content",
                    )?;
                    blocks.push(json!({"type": "input_text", "text": file_source_text(source)}));
                }
            },
            ContentBlock::Mcp(_) => {
                return Err(TranslationError::LossyConversion(
                    "MCP history encoding is not implemented".into(),
                ));
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

// Encodes tool-result content as a `function_call_output.output`: a string when it is
// text only, otherwise the typed `input_text` / `input_image` / `input_file` parts.
fn encode_responses_tool_output(
    content: &[ContentBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    let text_only = content.iter().all(|block| {
        matches!(
            block,
            ContentBlock::Text {
                responses: None,
                ..
            } | ContentBlock::Refusal { .. }
        )
    });
    if text_only {
        return Ok(Value::String(text_from_blocks(content, " ")));
    }
    encode_responses_content(content, diagnostics, policy)
}

fn responses_image_part(source: &ImageSource) -> Option<Value> {
    if let ImageSource::Raw(raw) = source
        && raw.get("type").and_then(Value::as_str) == Some("input_image")
        && raw.get("file_id").and_then(Value::as_str).is_some()
    {
        return Some(raw.clone());
    }
    let ImagePayload { url, detail } = image_payload(source)?;
    let mut part = json!({"type": "input_image"});
    part["image_url"] = Value::String(url);
    if let Some(detail) = detail {
        part["detail"] = Value::String(detail);
    }
    Some(part)
}

fn responses_file_part(source: &FileSource) -> Option<Value> {
    let mut part = file_payload(source)?;
    part.insert("type".to_string(), Value::String("input_file".to_string()));
    Some(Value::Object(part))
}

// Encodes normalized tool definitions into Responses tool JSON.
// Encodes normalized tools as Responses entries.
//
// A Responses client understands Codex containers, so a tool whose name this
// request qualified is regrouped under the container it came from.
fn encode_responses_tools(
    tools: &[Tool],
    namespaces: Option<&Map<String, Value>>,
    custom_tools: Option<&Map<String, Value>>,
) -> Result<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut containers: Vec<(String, Vec<Value>)> = Vec::new();
    for tool in tools {
        let tool = tool.function().ok_or_else(|| {
            TranslationError::LossyConversion(
                "MCP and provider tool encoding is not implemented".into(),
            )
        })?;
        // A freeform tool goes back out exactly as the client defined it.
        if let Some(custom) = custom_tools.and_then(|custom| custom.get(&tool.name)) {
            out.push(custom.clone());
            continue;
        }
        let mut item = json!({
            "type": "function",
            "name": tool.name,
            "description": tool.description.clone().unwrap_or_default(),
            "parameters": tool.parameters,
        });
        if let Some(strict) = tool.strict {
            item["strict"] = Value::Bool(strict);
        }
        let split = namespaces.and_then(|namespaces| {
            crate::codex_namespaces::split_qualified_name(namespaces, &tool.name)
        });
        match split {
            None => out.push(item),
            Some((name, namespace)) => {
                item["name"] = Value::String(name.to_string());
                match containers
                    .iter_mut()
                    .find(|(existing, _)| *existing == namespace)
                {
                    Some((_, children)) => children.push(item),
                    None => containers.push((namespace.to_string(), vec![item])),
                }
            }
        }
    }
    for (namespace, children) in containers {
        let description = namespaces
            .and_then(|namespaces| {
                crate::codex_namespaces::namespace_description(namespaces, &namespace)
            })
            .unwrap_or_default();
        out.push(json!({
            "type": "namespace",
            "name": namespace,
            "description": description,
            "tools": children,
        }));
    }
    Ok(Value::Array(out))
}

// Encodes normalized tool choice into Responses JSON.
fn encode_responses_tool_choice(
    choice: &ToolChoice,
    namespaces: Option<&Map<String, Value>>,
) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(Value::String("auto".to_string())),
        ToolChoice::Required => Some(Value::String("required".to_string())),
        ToolChoice::None => Some(Value::String("none".to_string())),
        ToolChoice::Tool { name } => {
            let split = namespaces.and_then(|namespaces| {
                crate::codex_namespaces::split_qualified_name(namespaces, name)
            });
            Some(match split {
                Some((tool, namespace)) => {
                    json!({"type": "function", "name": tool, "namespace": namespace})
                }
                None => json!({"type": "function", "name": name}),
            })
        }
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
            url_citations: decode_responses_url_citations(item.get("content")),
            role: role_from_responses(item.get("role").and_then(Value::as_str)),
            content: decode_responses_content(item.get("content").unwrap_or(&Value::Null), false),
            stop_reason: Some(StopReason::EndTurn),
        })),
        Some("function_call") => Ok(Some(ResponseOutput {
            url_citations: Vec::new(),
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
        Some("custom_tool_call") => Ok(Some(ResponseOutput {
            url_citations: Vec::new(),
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
                arguments: json!({
                    crate::codex_custom_tools::INPUT_ARGUMENT:
                        item.get("input").and_then(Value::as_str).unwrap_or_default()
                }),
            })],
            stop_reason: Some(StopReason::ToolUse),
        })),
        Some("reasoning") => Ok(Some(ResponseOutput {
            url_citations: Vec::new(),
            role: Role::Assistant,
            content: decode_responses_reasoning_item(item, false),
            stop_reason: None,
        })),
        Some(kind) if super::is_native_output(kind) => Ok(Some(ResponseOutput {
            url_citations: Vec::new(),
            role: Role::Assistant,
            content: vec![ContentBlock::Unknown {
                provider: WireFormat::OpenAiResponses.into(),
                raw: Value::Object(item.clone()),
            }],
            stop_reason: None,
        })),
        _ => Ok(None),
    }
}

fn decode_responses_url_citations(content: Option<&Value>) -> Vec<crate::UrlCitation> {
    let mut citations = Vec::new();
    let mut text_offset = 0;
    for part in content.and_then(Value::as_array).into_iter().flatten() {
        for annotation in part
            .get("annotations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if annotation["type"] == "url_citation"
                && let Ok(mut citation) =
                    serde_json::from_value::<crate::UrlCitation>(annotation.clone())
            {
                citation.start_index += text_offset;
                citation.end_index += text_offset;
                citations.push(citation);
            }
        }
        let text = match part.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text") => part.get("text"),
            Some("refusal") => part.get("refusal"),
            _ => None,
        };
        text_offset += text
            .and_then(Value::as_str)
            .unwrap_or_default()
            .chars()
            .count();
    }
    citations
}

// Encodes normalized response outputs into Responses output items.
fn encode_responses_output(outputs: &[ResponseOutput]) -> Value {
    Value::Array(
        outputs
            .iter()
            .flat_map(|output| {
                let has_output_items = output.content.iter().any(|block| {
                    matches!(block, ContentBlock::ToolCall(_))
                        || matches!(block, ContentBlock::Unknown { provider, raw }
                            if provider.as_str() == WireFormat::OpenAiResponses.as_str()
                                && raw["type"].as_str().is_some_and(super::is_native_output))
                });
                let text = text_from_blocks(&output.content, "");
                let reasoning = reasoning_text_from_blocks(&output.content, "\n");
                let status = if matches!(output.stop_reason, Some(StopReason::MaxTokens)) {
                    "incomplete"
                } else {
                    "completed"
                };
                let mut items = Vec::new();

                let encrypted_reasoning = output.content.iter().find_map(|block| match block {
                    ContentBlock::Reasoning { details, .. } => encrypted_reasoning_data(details),
                    _ => None,
                });
                let encrypted_reasoning_id = output.content.iter().find_map(|block| match block {
                    ContentBlock::Reasoning { details, .. } => encrypted_reasoning_item_id(details),
                    _ => None,
                });
                let has_signed_reasoning = output.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::Reasoning {
                            signature: Some(_),
                            ..
                        }
                    )
                });
                if has_signed_reasoning {
                    for (index, block) in output.content.iter().enumerate() {
                        if let ContentBlock::Reasoning {
                            text,
                            signature,
                            details,
                        } = block
                        {
                            let encrypted = signature
                                .as_deref()
                                .map(|signature| super::encode_anthropic_thinking(text, signature))
                                .or_else(|| encrypted_reasoning_data(details));
                            let id = encrypted_reasoning_item_id(details)
                                .unwrap_or_else(|| format!("rs_switchyard_{index}"));
                            items.push(encode_responses_reasoning_output(
                                text,
                                encrypted.as_deref(),
                                Some(&id),
                            ));
                        }
                    }
                } else if !reasoning.is_empty() || encrypted_reasoning.is_some() {
                    items.push(encode_responses_reasoning_output(
                        &reasoning,
                        encrypted_reasoning.as_deref(),
                        encrypted_reasoning_id.as_deref(),
                    ));
                }

                if !text.is_empty() || (!has_output_items && reasoning.is_empty()) {
                    items.push(json!({
                        "type": "message",
                        "id": "msg_switchyard",
                        "status": status,
                        "role": role_to_responses(output.role),
                        "content": [{
                            "type": "output_text",
                            "text": text,
                            "annotations": output.url_citations.iter().map(|citation| json!({
                                "type": "url_citation",
                                "start_index": citation.start_index,
                                "end_index": citation.end_index,
                                "url": citation.url,
                                "title": citation.title,
                            })).collect::<Vec<_>>(),
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
                    ContentBlock::Unknown { provider, raw }
                        if provider.as_str() == WireFormat::OpenAiResponses.as_str()
                            && raw["type"].as_str().is_some_and(super::is_native_output) =>
                    {
                        Some(raw.clone())
                    }
                    _ => None,
                }));

                items
            })
            .collect(),
    )
}

// Encodes private reasoning as a separate Responses output item. An encrypted-only item
// carries no text part but keeps `encrypted_content` so the client can replay it.
fn encode_responses_reasoning_output(
    text: &str,
    encrypted: Option<&str>,
    item_id: Option<&str>,
) -> Value {
    // Standard Responses shape: text as `summary_text` parts, which is what clients record.
    let mut summary = Vec::new();
    if !text.is_empty() {
        summary.push(json!({"type": "summary_text", "text": text}));
    }
    // Encrypted reasoning verifies only under the id it was issued with, so reuse it.
    let mut item = json!({
        "type": "reasoning",
        "id": item_id.unwrap_or("rs_switchyard"),
        "status": "completed",
        "summary": summary,
    });
    if let Some(encrypted) = encrypted {
        item["encrypted_content"] = Value::String(encrypted.to_string());
    }
    item
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
    let aggregate_input_tokens = value
        .get("input_tokens")
        .or_else(|| value.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let cached_input_tokens = value
        .get("input_tokens_details")
        .or_else(|| value.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
    let cache_creation_input_tokens = value
        .get("input_tokens_details")
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            value.get("prompt_tokens_details").and_then(|details| {
                details
                    .get("cache_write_tokens")
                    .and_then(Value::as_u64)
                    .or_else(|| details.get("cache_creation_tokens").and_then(Value::as_u64))
            })
        });
    let input_tokens = aggregate_input_tokens.map(|tokens| {
        tokens
            .saturating_sub(cached_input_tokens.unwrap_or(0))
            .saturating_sub(cache_creation_input_tokens.unwrap_or(0))
    });
    let output_tokens = value
        .get("output_tokens")
        .or_else(|| value.get("completion_tokens"))
        .and_then(Value::as_u64);
    Usage {
        provider_cost: crate::codecs::common::provider_cost(value),
        input_tokens,
        cache: crate::codecs::common::cache_usage_details(
            value,
            cached_input_tokens,
            cache_creation_input_tokens,
        ),
        output_tokens,
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                aggregate_input_tokens
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
pub(super) fn encode_responses_usage(usage: &Usage) -> Value {
    let input_tokens = usage.input_tokens.unwrap_or(0)
        + usage.cached_input_tokens().unwrap_or(0)
        + usage.cache_creation_input_tokens().unwrap_or(0);
    // The detail objects are required by the Responses schema, so both are always present. An
    // upstream that reports no cache or reasoning breakdown means zero, not "unknown"; omitting
    // them instead yields a payload that fails to deserialize into the OpenAI SDK's
    // ResponseUsage, which types both as non-optional.
    crate::codecs::common::encode_usage_metadata(
        json!({
            "input_tokens": input_tokens,
            "output_tokens": usage.output_tokens.unwrap_or(0),
            "total_tokens": usage
                .total_tokens
                .or_else(|| Some(input_tokens + usage.output_tokens.unwrap_or(0)))
                .unwrap_or(0),
            "input_tokens_details": {
                "cached_tokens": usage.cached_input_tokens().unwrap_or(0),
                "cache_write_tokens": usage.cache_creation_input_tokens().unwrap_or(0),
            },
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
        }),
        usage,
    )
}

/// Re-emits captured request extensions that the Responses format accepts.
///
/// The allowlist is Responses-specific: Chat-only fields such as
/// `stream_options` and `top_logprobs` are deliberately absent, so they stay
/// dropped on a Chat-to-Responses hop. Fields already present in `body` are
/// left untouched — anything the encoder generated is authoritative over a
/// captured extension of the same name.
fn copy_responses_request_extensions(
    body: &mut Map<String, Value>,
    extensions: &Map<String, Value>,
) {
    for field in [
        "conversation",
        "include",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "store",
        "user",
    ] {
        if let Some(value) = extensions.get(field) {
            body.entry(field.to_string())
                .or_insert_with(|| value.clone());
        }
    }
}
