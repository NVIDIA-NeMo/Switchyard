// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for Gemini generateContent request and response JSON.
//!
//! The internal Gemini wire body carries synthetic top-level `model` and
//! `stream` fields because the real API encodes both in the URL path
//! (`/v1beta/models/{model}:generateContent` vs `:streamGenerateContent`).
//! Endpoints inject them on decode and the Gemini backend moves them back
//! into the URL before the upstream call.

use serde_json::{json, Map, Value};

use crate::codecs::common::provider_extensions;
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::ir::{
    is_known_role_name, ContentBlock, ConversationRequest, ConversationResponse, FileSource,
    ImageSource, InstructionBlock, MediaSource, Message, OutputParams, ProviderExtensions,
    ReasoningParams, ResponseOutput, Role, SamplingParams, StopReason, ToolCall, ToolChoice,
    ToolDefinition, ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response,
};
use crate::util::{json_string, push_lossy, stable_id, validate_request_capabilities};

/// Format codec for Gemini generateContent payloads.
pub struct GeminiGenerateContentCodec;

impl FormatCodec for GeminiGenerateContentCodec {
    fn format(&self) -> FormatId {
        WireFormat::GeminiGenerateContent.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let generation = camel_or_snake(body, "generationConfig", "generation_config")
            .and_then(Value::as_object);
        let mut request = ConversationRequest {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned),
            output: OutputParams {
                max_output_tokens: generation
                    .and_then(|config| config.get("maxOutputTokens"))
                    .and_then(Value::as_u64),
                response_format: generation.and_then(decode_gemini_response_format),
            },
            sampling: SamplingParams {
                temperature: generation
                    .and_then(|config| config.get("temperature"))
                    .and_then(Value::as_f64),
                top_p: generation
                    .and_then(|config| config.get("topP"))
                    .and_then(Value::as_f64),
                top_k: generation
                    .and_then(|config| config.get("topK"))
                    .and_then(Value::as_i64),
            },
            reasoning: ReasoningParams {
                effort: None,
                raw: generation
                    .and_then(|config| config.get("thinkingConfig"))
                    .cloned(),
            },
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            preservation: capture_request_preservation(
                WireFormat::GeminiGenerateContent,
                &Value::Object(body.clone()),
                policy,
            ),
            ..ConversationRequest::default()
        };
        if let Some(stop) = generation
            .and_then(|config| config.get("stopSequences"))
            .filter(|stop| stop.is_array())
        {
            // Stored under both raw keys: the Anthropic encoder reads the
            // OpenAI-style `stop` extension and the OpenAI encoder reads the
            // Anthropic-style `stop_sequences` extension.
            request
                .extensions
                .fields
                .insert("stop".to_string(), stop.clone());
            request
                .extensions
                .fields
                .insert("stop_sequences".to_string(), stop.clone());
        }
        if let Some(system) = camel_or_snake(body, "systemInstruction", "system_instruction") {
            if let Some(content) = decode_gemini_system(system) {
                request.instructions.push(InstructionBlock {
                    role: Role::System,
                    content,
                });
            }
        }
        if let Some(contents) = body.get("contents").and_then(Value::as_array) {
            request.messages = decode_gemini_contents(contents, &mut diagnostics, policy)?;
        }
        request.tools = decode_gemini_tools(body.get("tools"), &mut diagnostics, policy)?;
        request.tool_choice =
            camel_or_snake(body, "toolConfig", "tool_config").map(decode_gemini_tool_choice);
        request.extensions.fields.extend(provider_extensions(
            body,
            &[
                "model",
                "stream",
                "contents",
                "systemInstruction",
                "system_instruction",
                "tools",
                "toolConfig",
                "tool_config",
                "generationConfig",
                "generation_config",
                "safetySettings",
                "safety_settings",
            ],
        ));

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
        if let Some(body) = exact_preserved_request(
            &request.preservation,
            WireFormat::GeminiGenerateContent,
            policy,
        ) {
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
        if request.stream {
            body.insert("stream".to_string(), Value::Bool(true));
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
            body.insert(
                "systemInstruction".to_string(),
                json!({"parts": [{"text": system_text}]}),
            );
        }

        body.insert(
            "contents".to_string(),
            Value::Array(encode_gemini_contents(
                &request.messages,
                &mut diagnostics,
                policy,
            )?),
        );

        if !request.tools.is_empty() {
            body.insert("tools".to_string(), encode_gemini_tools(&request.tools));
        }
        if let Some(config) = request
            .tool_choice
            .as_ref()
            .and_then(encode_gemini_tool_choice)
        {
            body.insert("toolConfig".to_string(), config);
        }

        let generation = encode_generation_config(request);
        if !generation.is_empty() {
            body.insert("generationConfig".to_string(), Value::Object(generation));
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
        let candidate = body
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(Value::as_object);
        let mut content = Vec::new();
        if let Some(parts) = candidate
            .and_then(|candidate| candidate.get("content"))
            .and_then(|value| value.get("parts"))
            .and_then(Value::as_array)
        {
            let mut generated_id = 0;
            for part in parts {
                if let Some(part) = part.as_object() {
                    generated_id += 1;
                    content.extend(decode_gemini_part(
                        part,
                        generated_id,
                        &mut Vec::new(),
                        &TranslationPolicy::default(),
                    )?);
                }
            }
        }
        let has_tool_calls = content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolCall(_)));
        if content.is_empty() {
            content.push(ContentBlock::Text {
                text: String::new(),
            });
        }
        let finish_reason = candidate
            .and_then(|candidate| candidate.get("finishReason"))
            .and_then(Value::as_str);
        // A fully blocked prompt has no candidates but carries a block reason.
        let blocked = candidate.is_none()
            && body
                .get("promptFeedback")
                .and_then(|feedback| feedback.get("blockReason"))
                .is_some();
        let stop_reason = if blocked {
            StopReason::ContentFilter
        } else {
            map_gemini_finish_reason(finish_reason, has_tool_calls)
        };
        let response = ConversationResponse {
            id: body
                .get("responseId")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: body
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: Some(stop_reason),
            }],
            usage: decode_gemini_usage(body.get("usageMetadata")),
            extensions: ProviderExtensions {
                fields: provider_extensions(
                    body,
                    &["candidates", "usageMetadata", "modelVersion", "responseId"],
                ),
            },
            preservation: capture_response_preservation(
                WireFormat::GeminiGenerateContent,
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
            WireFormat::GeminiGenerateContent,
            _policy,
        ) {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let output = response.first_output();
        let parts = output
            .map(|output| encode_gemini_response_parts(&output.content))
            .unwrap_or_else(|| vec![json!({"text": ""})]);
        let body = json!({
            "candidates": [{
                "content": {"parts": parts, "role": "model"},
                "finishReason": output
                    .and_then(|output| output.stop_reason)
                    .map(gemini_finish_reason)
                    .unwrap_or("STOP"),
                "index": 0,
            }],
            "usageMetadata": encode_gemini_usage(&response.usage),
            "modelVersion": response.model.clone().unwrap_or_else(|| "unknown".to_string()),
            "responseId": response
                .id
                .clone()
                .unwrap_or_else(|| "gemini_switchyard".to_string()),
        });
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, _policy),
            diagnostics: Vec::new(),
        })
    }
}

// Reads a field that Gemini clients may spell in camelCase or snake_case.
fn camel_or_snake<'a>(
    object: &'a Map<String, Value>,
    camel: &str,
    snake: &str,
) -> Option<&'a Value> {
    object.get(camel).or_else(|| object.get(snake))
}

// Decodes Gemini's `systemInstruction` content into instruction blocks.
fn decode_gemini_system(value: &Value) -> Option<Vec<ContentBlock>> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Object(object) => object
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            })
            .unwrap_or_default(),
        _ => String::new(),
    };
    (!text.is_empty()).then(|| vec![ContentBlock::Text { text }])
}

// Decodes `contents[]` while pairing functionResponse parts with the
// functionCall that produced them. Gemini has no tool-call IDs on the wire;
// calls and responses pair by function name and order, so decoding assigns
// stable synthetic IDs and lets each response consume the earliest
// unconsumed call with a matching name.
fn decode_gemini_contents(
    contents: &[Value],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    let mut open_calls: Vec<(String, String, bool)> = Vec::new();
    let mut generated_id = 0;
    for (index, content) in contents.iter().enumerate() {
        let Some(content) = content.as_object() else {
            push_lossy(
                diagnostics,
                policy,
                format!("Gemini content at index {index} is not an object"),
            )?;
            continue;
        };
        // Gemini only defines `user`/`model`; other known role names stay
        // lenient (mapped to `user`) to match cross-format decode behavior,
        // while genuinely unknown roles are rejected.
        let role = match content.get("role").and_then(Value::as_str) {
            Some("model") => Role::Assistant,
            Some("user") | None => Role::User,
            Some(other) if is_known_role_name(other) => Role::User,
            Some(other) => {
                return Err(TranslationError::unsupported_role(
                    format!("$.contents[{index}].role"),
                    other,
                ));
            }
        };
        let mut blocks = Vec::new();
        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts {
                let Some(part) = part.as_object() else {
                    push_lossy(diagnostics, policy, "Gemini part is not an object")?;
                    continue;
                };
                if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
                    generated_id += 1;
                    let decoded = decode_gemini_function_call(part, call, generated_id, policy);
                    if let Some(ContentBlock::ToolCall(call)) = decoded.last() {
                        open_calls.push((call.name.clone(), call.id.clone(), false));
                    }
                    blocks.extend(decoded);
                    continue;
                }
                if let Some(response) = part.get("functionResponse").and_then(Value::as_object) {
                    generated_id += 1;
                    blocks.push(decode_gemini_function_response(
                        response,
                        &mut open_calls,
                        generated_id,
                        policy,
                    ));
                    continue;
                }
                generated_id += 1;
                blocks.extend(decode_gemini_part(part, generated_id, diagnostics, policy)?);
            }
        }
        if blocks.is_empty() {
            blocks.push(ContentBlock::Text {
                text: String::new(),
            });
        }
        messages.push(Message {
            role,
            content: blocks,
        });
    }
    Ok(messages)
}

// Decodes one non-tool Gemini part into zero or more IR blocks.
fn decode_gemini_part(
    part: &Map<String, Value>,
    generated_counter: usize,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
        return Ok(decode_gemini_function_call(
            part,
            call,
            generated_counter,
            policy,
        ));
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            return Ok(vec![ContentBlock::Reasoning {
                text: text.to_string(),
                signature: part_thought_signature(part),
            }]);
        }
        return Ok(vec![ContentBlock::Text {
            text: text.to_string(),
        }]);
    }
    if let Some(inline) = part.get("inlineData").and_then(Value::as_object) {
        let media_type = inline
            .get("mimeType")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let data = inline
            .get("data")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        return Ok(vec![
            match media_type.as_deref().unwrap_or("").split('/').next() {
                Some("image") => ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data },
                },
                Some("audio") => ContentBlock::Audio {
                    source: MediaSource::Base64 { media_type, data },
                },
                Some("video") => ContentBlock::Video {
                    source: MediaSource::Base64 { media_type, data },
                },
                _ => ContentBlock::File {
                    source: FileSource::FileData {
                        data,
                        filename: None,
                    },
                },
            },
        ]);
    }
    if let Some(file) = part.get("fileData").and_then(Value::as_object) {
        let url = file
            .get("fileUri")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let media_type = file
            .get("mimeType")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return Ok(vec![
            match media_type.as_deref().unwrap_or("").split('/').next() {
                Some("image") => ContentBlock::Image {
                    source: ImageSource::Url { url, detail: None },
                },
                Some("audio") => ContentBlock::Audio {
                    source: MediaSource::Url { url, media_type },
                },
                Some("video") => ContentBlock::Video {
                    source: MediaSource::Url { url, media_type },
                },
                _ => ContentBlock::Unknown {
                    provider: WireFormat::GeminiGenerateContent.into(),
                    raw: Value::Object(part.clone()),
                },
            },
        ]);
    }
    push_lossy(diagnostics, policy, "unsupported Gemini part kind")?;
    Ok(vec![ContentBlock::Unknown {
        provider: WireFormat::GeminiGenerateContent.into(),
        raw: Value::Object(part.clone()),
    }])
}

// Decodes a functionCall part; a thought signature rides along as a
// signature-only reasoning block so it can be replayed on encode.
fn decode_gemini_function_call(
    part: &Map<String, Value>,
    call: &Map<String, Value>,
    generated_counter: usize,
    policy: &TranslationPolicy,
) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    if let Some(signature) = part_thought_signature(part) {
        blocks.push(ContentBlock::Reasoning {
            text: String::new(),
            signature: Some(signature),
        });
    }
    blocks.push(ContentBlock::ToolCall(ToolCall {
        id: generated_tool_id(generated_counter, policy),
        name: call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        arguments: call.get("args").cloned().unwrap_or_else(|| json!({})),
    }));
    blocks
}

// Decodes a functionResponse part, consuming the matching open call ID.
fn decode_gemini_function_response(
    response: &Map<String, Value>,
    open_calls: &mut [(String, String, bool)],
    generated_counter: usize,
    policy: &TranslationPolicy,
) -> ContentBlock {
    let name = response
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let tool_call_id = open_calls
        .iter_mut()
        .find(|(call_name, _, consumed)| !consumed && call_name == name)
        .map(|entry| {
            entry.2 = true;
            entry.1.clone()
        })
        .unwrap_or_else(|| generated_tool_id(generated_counter, policy));
    ContentBlock::ToolResult(ToolResult {
        tool_call_id,
        content: decode_function_response_content(response.get("response")),
        is_error: None,
    })
}

// Converts a functionResponse `response` payload into IR content blocks.
fn decode_function_response_content(value: Option<&Value>) -> Vec<ContentBlock> {
    match value {
        Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
        Some(Value::Object(object)) => {
            // Accept the `{parts: [...]}` shape this codec emits, so tool
            // results round-trip; any other object is preserved as JSON text.
            if let Some(parts) = object.get("parts").and_then(Value::as_array) {
                let mut blocks = Vec::new();
                for part in parts {
                    let Some(part) = part.as_object() else {
                        continue;
                    };
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        blocks.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    } else if let Some(inline) = part.get("inlineData").and_then(Value::as_object) {
                        blocks.push(ContentBlock::Image {
                            source: ImageSource::Base64 {
                                media_type: inline
                                    .get("mimeType")
                                    .and_then(Value::as_str)
                                    .map(ToOwned::to_owned),
                                data: inline
                                    .get("data")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                        });
                    }
                }
                if !blocks.is_empty() {
                    return blocks;
                }
            }
            vec![ContentBlock::Text {
                text: json_string(&Value::Object(object.clone())),
            }]
        }
        Some(other) => vec![ContentBlock::Text {
            text: json_string(other),
        }],
        None => vec![ContentBlock::Text {
            text: String::new(),
        }],
    }
}

// Reads a non-empty thoughtSignature from a Gemini part.
fn part_thought_signature(part: &Map<String, Value>) -> Option<String> {
    part.get("thoughtSignature")
        .and_then(Value::as_str)
        .filter(|signature| !signature.is_empty())
        .map(ToOwned::to_owned)
}

// Generates a tool-call ID for Gemini's ID-less function calls.
fn generated_tool_id(counter: usize, policy: &TranslationPolicy) -> String {
    match &policy.deterministic_ids {
        DeterministicIdPolicy::GenerateStable { prefix } => stable_id(prefix, counter),
        DeterministicIdPolicy::Preserve => String::new(),
    }
}

// Decodes Gemini tool declarations into normalized tool definitions.
fn decode_gemini_tools(
    value: Option<&Value>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ToolDefinition>> {
    let mut tools = Vec::new();
    for entry in value.and_then(Value::as_array).into_iter().flatten() {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let Some(declarations) =
            camel_or_snake(entry, "functionDeclarations", "function_declarations")
                .and_then(Value::as_array)
        else {
            // Built-in tools such as googleSearch or codeExecution have no
            // cross-provider equivalent.
            push_lossy(
                diagnostics,
                policy,
                "unsupported non-function Gemini tool entry",
            )?;
            continue;
        };
        for declaration in declarations.iter().filter_map(Value::as_object) {
            let Some(name) = declaration
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            else {
                continue;
            };
            tools.push(ToolDefinition {
                name: name.to_string(),
                description: declaration
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                parameters: declaration
                    .get("parameters")
                    .or_else(|| declaration.get("parametersJsonSchema"))
                    .map(schema_from_gemini)
                    .unwrap_or_else(|| json!({})),
                strict: None,
            });
        }
    }
    Ok(tools)
}

// Decodes Gemini toolConfig into normalized tool-choice policy.
fn decode_gemini_tool_choice(value: &Value) -> ToolChoice {
    let Some(config) = value
        .as_object()
        .and_then(|object| {
            camel_or_snake(object, "functionCallingConfig", "function_calling_config")
        })
        .and_then(Value::as_object)
    else {
        return ToolChoice::Raw(value.clone());
    };
    let allowed = config
        .get("allowedFunctionNames")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    match config.get("mode").and_then(Value::as_str) {
        Some("AUTO") | None => ToolChoice::Auto,
        Some("NONE") => ToolChoice::None,
        Some("ANY") if allowed.len() == 1 => ToolChoice::Tool {
            name: allowed.into_iter().next().unwrap_or_default(),
        },
        Some("ANY") => ToolChoice::Required,
        _ => ToolChoice::Raw(value.clone()),
    }
}

// Maps generationConfig JSON-mode fields onto OpenAI-style response_format.
fn decode_gemini_response_format(generation: &Map<String, Value>) -> Option<Value> {
    if generation.get("responseMimeType").and_then(Value::as_str) != Some("application/json") {
        return None;
    }
    let schema = generation
        .get("responseSchema")
        .or_else(|| generation.get("responseJsonSchema"));
    Some(match schema {
        Some(schema) => json!({
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema_from_gemini(schema)},
        }),
        None => json!({"type": "json_object"}),
    })
}

// Encodes normalized messages into Gemini `contents[]`.
fn encode_gemini_contents(
    messages: &[Message],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut contents = Vec::new();
    for message in messages {
        let role = match message.role {
            Role::Assistant => "model",
            Role::User | Role::Tool | Role::System | Role::Developer => "user",
        };
        // Thought signatures echoed by the client (or produced by response
        // decode) attach back onto functionCall parts in order; Gemini
        // requires them when replaying thinking-model tool calls.
        let signatures = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Reasoning {
                    signature: Some(signature),
                    ..
                } if !signature.is_empty() => Some(signature.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut signature_index = 0;
        let mut parts = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                    if !text.is_empty() {
                        parts.push(json!({"text": text}));
                    }
                }
                // Reasoning text is never replayed in requests; only its
                // signature matters (attached to functionCall parts above).
                ContentBlock::Reasoning { .. } => {}
                ContentBlock::ToolCall(call) => {
                    let mut part = Map::new();
                    part.insert(
                        "functionCall".to_string(),
                        json!({"name": call.name, "args": gemini_tool_args(&call.arguments)}),
                    );
                    let signature = signatures
                        .get(signature_index)
                        .or_else(|| signatures.last());
                    if let Some(signature) = signature {
                        part.insert(
                            "thoughtSignature".to_string(),
                            Value::String(signature.clone()),
                        );
                    }
                    signature_index += 1;
                    parts.push(Value::Object(part));
                }
                ContentBlock::ToolResult(result) => {
                    parts.push(encode_gemini_function_response(result, messages));
                }
                ContentBlock::Image { source } => match source {
                    ImageSource::Base64 { media_type, data } => parts.push(json!({
                        "inlineData": {
                            "mimeType": media_type.clone().unwrap_or_else(|| "image/png".to_string()),
                            "data": data,
                        },
                    })),
                    ImageSource::Url { .. } | ImageSource::Raw(_) => {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini does not accept URL or raw image sources; image dropped",
                        )?;
                    }
                },
                ContentBlock::Audio { source } => {
                    encode_gemini_media(source, "audio/mpeg", &mut parts, diagnostics, policy)?;
                }
                ContentBlock::Video { source } => {
                    encode_gemini_media(source, "video/mp4", &mut parts, diagnostics, policy)?;
                }
                ContentBlock::File { source } => match source {
                    // The IR file source has no media type; PDF is the
                    // dominant cross-provider document payload.
                    FileSource::FileData { data, .. } => parts.push(json!({
                        "inlineData": {"mimeType": "application/pdf", "data": data},
                    })),
                    FileSource::FileId(_) | FileSource::Raw(_) => {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini cannot reference foreign file IDs; file dropped",
                        )?;
                    }
                },
                ContentBlock::Unknown { raw, .. } => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "unknown content block encoded as text for Gemini",
                    )?;
                    parts.push(json!({"text": json_string(raw)}));
                }
            }
        }
        // Gemini rejects contents with an empty parts array.
        if parts.is_empty() {
            continue;
        }
        contents.push(json!({"role": role, "parts": parts}));
    }
    Ok(contents)
}

// Encodes base64 and URL media sources into Gemini parts.
fn encode_gemini_media(
    source: &MediaSource,
    default_media_type: &str,
    parts: &mut Vec<Value>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    match source {
        MediaSource::Base64 { media_type, data } => parts.push(json!({
            "inlineData": {
                "mimeType": media_type.clone().unwrap_or_else(|| default_media_type.to_string()),
                "data": data,
            },
        })),
        MediaSource::Url { url, media_type } => {
            let mut file = Map::new();
            file.insert("fileUri".to_string(), Value::String(url.clone()));
            if let Some(media_type) = media_type {
                file.insert("mimeType".to_string(), Value::String(media_type.clone()));
            }
            parts.push(json!({"fileData": Value::Object(file)}));
        }
        MediaSource::Raw(_) => {
            push_lossy(diagnostics, policy, "raw media source dropped for Gemini")?;
        }
    }
    Ok(())
}

// Encodes a tool result as a functionResponse part, recovering the function
// name from the tool call it answers (Gemini pairs by name, not ID).
fn encode_gemini_function_response(result: &ToolResult, messages: &[Message]) -> Value {
    let name = find_tool_name(messages, &result.tool_call_id).unwrap_or_else(|| "tool".to_string());
    let mut parts = Vec::new();
    for block in &result.content {
        match block {
            ContentBlock::Text { text } => parts.push(json!({"text": text})),
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            } => parts.push(json!({
                "inlineData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "image/png".to_string()),
                    "data": data,
                },
            })),
            other => parts.push(json!({"text": json_string(&json!(other))})),
        }
    }
    if parts.is_empty() {
        parts.push(json!({"text": ""}));
    }
    json!({
        "functionResponse": {"name": name, "response": {"parts": parts}},
    })
}

// Finds the function name for a tool-call ID anywhere in the conversation.
fn find_tool_name(messages: &[Message], tool_call_id: &str) -> Option<String> {
    messages
        .iter()
        .flat_map(|message| message.content.iter())
        .find_map(|block| match block {
            ContentBlock::ToolCall(call) if call.id == tool_call_id => Some(call.name.clone()),
            _ => None,
        })
}

// Gemini requires `functionCall.args` to be object-shaped, while OpenAI and
// Responses commonly carry function-call arguments as JSON strings.
fn gemini_tool_args(arguments: &Value) -> Value {
    match arguments {
        Value::Object(object) => Value::Object(object.clone()),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({"raw": text})),
        Value::Null => json!({}),
        other => json!({"value": other}),
    }
}

// Encodes normalized tool definitions into a single Gemini tool entry.
fn encode_gemini_tools(tools: &[ToolDefinition]) -> Value {
    json!([{
        "functionDeclarations": tools
            .iter()
            .map(|tool| {
                let mut declaration = Map::new();
                declaration.insert("name".to_string(), Value::String(tool.name.clone()));
                if let Some(description) = &tool.description {
                    declaration.insert(
                        "description".to_string(),
                        Value::String(description.clone()),
                    );
                }
                declaration.insert("parameters".to_string(), schema_to_gemini(&tool.parameters));
                Value::Object(declaration)
            })
            .collect::<Vec<_>>(),
    }])
}

// Encodes normalized tool choice into Gemini toolConfig JSON.
fn encode_gemini_tool_choice(choice: &ToolChoice) -> Option<Value> {
    let config = match choice {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::Required => json!({"mode": "ANY"}),
        ToolChoice::None => json!({"mode": "NONE"}),
        ToolChoice::Tool { name } => json!({"mode": "ANY", "allowedFunctionNames": [name]}),
        ToolChoice::Raw(value) => {
            // Only replay raw values that are already Gemini-shaped; foreign
            // provider shapes would be rejected with a 400.
            return value
                .as_object()
                .is_some_and(|object| object.contains_key("functionCallingConfig"))
                .then(|| value.clone());
        }
    };
    Some(json!({"functionCallingConfig": config}))
}

// Builds generationConfig from IR sampling, output, and reasoning params.
fn encode_generation_config(request: &ConversationRequest) -> Map<String, Value> {
    let mut generation = Map::new();
    if let Some(max_tokens) = request.output.max_output_tokens {
        generation.insert("maxOutputTokens".to_string(), json!(max_tokens));
    }
    if let Some(value) = request.sampling.temperature {
        generation.insert("temperature".to_string(), json!(value));
    }
    if let Some(value) = request.sampling.top_p {
        generation.insert("topP".to_string(), json!(value));
    }
    if let Some(value) = request.sampling.top_k {
        generation.insert("topK".to_string(), json!(value));
    }
    // OpenAI sources preserve `stop`; Anthropic sources preserve `stop_sequences`.
    let stop = request
        .extensions
        .fields
        .get("stop")
        .or_else(|| request.extensions.fields.get("stop_sequences"));
    match stop {
        Some(Value::String(stop)) => {
            generation.insert("stopSequences".to_string(), json!([stop]));
        }
        Some(Value::Array(stops)) => {
            generation.insert("stopSequences".to_string(), Value::Array(stops.clone()));
        }
        _ => {}
    }
    if let Some(format) = &request.output.response_format {
        encode_gemini_response_format(format, &mut generation);
    }
    if let Some(config) = thinking_config(&request.reasoning) {
        generation.insert("thinkingConfig".to_string(), config);
    }
    generation
}

// Maps OpenAI-style response_format onto Gemini JSON-mode fields.
fn encode_gemini_response_format(format: &Value, generation: &mut Map<String, Value>) {
    match format.get("type").and_then(Value::as_str) {
        Some("json_object") => {
            generation.insert(
                "responseMimeType".to_string(),
                Value::String("application/json".to_string()),
            );
        }
        Some("json_schema") => {
            generation.insert(
                "responseMimeType".to_string(),
                Value::String("application/json".to_string()),
            );
            if let Some(schema) = format
                .get("json_schema")
                .and_then(|json_schema| json_schema.get("schema"))
            {
                generation.insert("responseSchema".to_string(), schema_to_gemini(schema));
            }
        }
        _ => {}
    }
}

// Builds thinkingConfig from preserved Gemini config or a reasoning effort.
fn thinking_config(reasoning: &ReasoningParams) -> Option<Value> {
    if let Some(raw) = &reasoning.raw {
        if raw.as_object().is_some_and(|config| {
            config.contains_key("thinkingBudget") || config.contains_key("includeThoughts")
        }) {
            return Some(raw.clone());
        }
    }
    let budget = match reasoning.effort.as_deref() {
        Some("minimal") | Some("low") => 1024,
        Some("medium") => 8192,
        Some("high") => 24576,
        _ => return None,
    };
    Some(json!({"thinkingBudget": budget}))
}

// JSON-Schema keywords Gemini's OpenAPI-style schema subset rejects with
// `400 Unknown name`; they are stripped rather than forwarded.
const GEMINI_DROPPED_SCHEMA_KEYS: &[&str] = &[
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "additionalProperties",
    "unevaluatedProperties",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "allOf",
    "not",
    "const",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "uniqueItems",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "readOnly",
    "writeOnly",
    "deprecated",
    "examples",
    "default",
];

const GEMINI_SCHEMA_TYPES: &[&str] = &[
    "string", "number", "integer", "boolean", "array", "object", "null",
];

// Sanitizes a JSON Schema into Gemini's OpenAPI-style subset: uppercase
// `type`, dropped unsupported keywords, union types collapsed to a single
// type plus `nullable`, and enums coerced to STRING values.
fn schema_to_gemini(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        // Boolean schemas (`true`) have no Gemini equivalent; accept anything.
        return json!({});
    };
    let mut sanitized = Map::new();
    let has_enum = object.get("enum").and_then(Value::as_array).is_some();
    for (key, value) in object {
        if GEMINI_DROPPED_SCHEMA_KEYS.contains(&key.as_str())
            || key.starts_with("x-")
            || key.starts_with("x_")
        {
            continue;
        }
        match key.as_str() {
            "type" => {
                let (schema_type, nullable) = gemini_schema_type(value, has_enum);
                sanitized.insert("type".to_string(), Value::String(schema_type));
                if nullable {
                    sanitized.insert("nullable".to_string(), Value::Bool(true));
                }
            }
            "enum" => {
                let values = value
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .map(|value| match value {
                                Value::String(text) => Value::String(text.clone()),
                                Value::Null => Value::String(String::new()),
                                other => Value::String(json_string(other)),
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                sanitized.insert("enum".to_string(), Value::Array(values));
                // Gemini only allows enums on STRING-typed schemas.
                sanitized.insert("type".to_string(), Value::String("STRING".to_string()));
            }
            "properties" => {
                let properties = value
                    .as_object()
                    .map(|properties| {
                        properties
                            .iter()
                            .map(|(name, schema)| (name.clone(), schema_to_gemini(schema)))
                            .collect::<Map<_, _>>()
                    })
                    .unwrap_or_default();
                sanitized.insert("properties".to_string(), Value::Object(properties));
            }
            "items" => {
                sanitized.insert("items".to_string(), schema_to_gemini(value));
            }
            "anyOf" | "oneOf" => {
                let members = value
                    .as_array()
                    .map(|members| members.iter().map(schema_to_gemini).collect::<Vec<_>>())
                    .unwrap_or_default();
                sanitized.insert("anyOf".to_string(), Value::Array(members));
            }
            _ => {
                sanitized.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(sanitized)
}

// Resolves a JSON Schema `type` value into Gemini's single uppercase type
// plus a nullable flag, collapsing Draft-2020 union arrays.
fn gemini_schema_type(value: &Value, has_enum: bool) -> (String, bool) {
    match value {
        Value::String(schema_type) => (uppercase_schema_type(schema_type, has_enum), false),
        Value::Array(members) => {
            let nullable = members.iter().any(|member| member.as_str() == Some("null"));
            let concrete = members
                .iter()
                .filter_map(Value::as_str)
                .find(|member| *member != "null");
            match concrete {
                Some(schema_type) => (uppercase_schema_type(schema_type, has_enum), nullable),
                None => ("STRING".to_string(), nullable),
            }
        }
        _ => ("STRING".to_string(), false),
    }
}

fn uppercase_schema_type(schema_type: &str, has_enum: bool) -> String {
    if has_enum {
        return "STRING".to_string();
    }
    if GEMINI_SCHEMA_TYPES.contains(&schema_type) {
        schema_type.to_ascii_uppercase()
    } else {
        "STRING".to_string()
    }
}

// Lowercases Gemini's uppercase schema types back into JSON Schema form so
// declarations survive translation toward OpenAI or Anthropic targets.
fn schema_from_gemini(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let mut restored = Map::new();
    for (key, value) in object {
        match key.as_str() {
            "type" => {
                let lowered = value
                    .as_str()
                    .map(|schema_type| Value::String(schema_type.to_ascii_lowercase()))
                    .unwrap_or_else(|| value.clone());
                restored.insert("type".to_string(), lowered);
            }
            "properties" => {
                let properties = value
                    .as_object()
                    .map(|properties| {
                        properties
                            .iter()
                            .map(|(name, schema)| (name.clone(), schema_from_gemini(schema)))
                            .collect::<Map<_, _>>()
                    })
                    .unwrap_or_default();
                restored.insert("properties".to_string(), Value::Object(properties));
            }
            "items" => {
                restored.insert("items".to_string(), schema_from_gemini(value));
            }
            "anyOf" => {
                let members = value
                    .as_array()
                    .map(|members| members.iter().map(schema_from_gemini).collect::<Vec<_>>())
                    .unwrap_or_default();
                restored.insert("anyOf".to_string(), Value::Array(members));
            }
            _ => {
                restored.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(restored)
}

// Encodes assistant output blocks into Gemini response parts.
fn encode_gemini_response_parts(content: &[ContentBlock]) -> Vec<Value> {
    let mut parts = Vec::new();
    let mut pending_signature: Option<String> = None;
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                parts.push(json!({"text": text}));
            }
            ContentBlock::Reasoning { text, signature } => {
                if text.is_empty() {
                    // Signature-only reasoning pairs with the next tool call.
                    if signature.is_some() {
                        pending_signature = signature.clone();
                    }
                    continue;
                }
                let mut part = Map::new();
                part.insert("text".to_string(), Value::String(text.clone()));
                part.insert("thought".to_string(), Value::Bool(true));
                if let Some(signature) = signature.as_ref().filter(|value| !value.is_empty()) {
                    part.insert(
                        "thoughtSignature".to_string(),
                        Value::String(signature.clone()),
                    );
                }
                parts.push(Value::Object(part));
            }
            ContentBlock::ToolCall(call) => {
                let mut part = Map::new();
                part.insert(
                    "functionCall".to_string(),
                    json!({"name": call.name, "args": gemini_tool_args(&call.arguments)}),
                );
                if let Some(signature) = pending_signature.take() {
                    part.insert("thoughtSignature".to_string(), Value::String(signature));
                }
                parts.push(Value::Object(part));
            }
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            } => parts.push(json!({
                "inlineData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "image/png".to_string()),
                    "data": data,
                },
            })),
            other => parts.push(json!({"text": json_string(&json!(other))})),
        }
    }
    if parts.is_empty() {
        parts.push(json!({"text": ""}));
    }
    parts
}

// Maps Gemini finish reasons to normalized stop reasons. Gemini reports
// `STOP` even for function-call turns, so tool-call presence wins.
fn map_gemini_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    match reason {
        Some("STOP") | Some("FINISH_REASON_UNSPECIFIED") | None if has_tool_calls => {
            StopReason::ToolUse
        }
        Some("STOP") | Some("FINISH_REASON_UNSPECIFIED") | None => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some("SAFETY")
        | Some("RECITATION")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII")
        | Some("IMAGE_SAFETY") => StopReason::ContentFilter,
        Some("MALFORMED_FUNCTION_CALL") => StopReason::Error,
        _ => StopReason::Unknown,
    }
}

// Maps normalized stop reasons back to Gemini's vocabulary.
fn gemini_finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn | StopReason::ToolUse => "STOP",
        StopReason::MaxTokens => "MAX_TOKENS",
        StopReason::ContentFilter => "SAFETY",
        StopReason::Error | StopReason::Unknown => "OTHER",
    }
}

// Normalizes Gemini usageMetadata counters. Gemini reports thinking tokens
// outside candidatesTokenCount, so they are folded back into output tokens
// to match the other providers' accounting.
fn decode_gemini_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.and_then(Value::as_object) else {
        return Usage::default();
    };
    let input_tokens = value.get("promptTokenCount").and_then(Value::as_u64);
    let candidates_tokens = value.get("candidatesTokenCount").and_then(Value::as_u64);
    let thoughts_tokens = value.get("thoughtsTokenCount").and_then(Value::as_u64);
    let output_tokens = match (candidates_tokens, thoughts_tokens) {
        (None, None) => None,
        (candidates, thoughts) => Some(
            candidates
                .unwrap_or(0)
                .saturating_add(thoughts.unwrap_or(0)),
        ),
    };
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: value
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .or_else(|| {
                input_tokens
                    .zip(output_tokens)
                    .map(|(input, output)| input + output)
            }),
        reasoning_tokens: thoughts_tokens,
    }
}

// Encodes normalized usage into Gemini usageMetadata JSON.
fn encode_gemini_usage(usage: &Usage) -> Value {
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    let thoughts = usage.reasoning_tokens.unwrap_or(0);
    let mut metadata = Map::new();
    metadata.insert("promptTokenCount".to_string(), json!(input));
    metadata.insert(
        "candidatesTokenCount".to_string(),
        json!(output.saturating_sub(thoughts)),
    );
    if thoughts > 0 {
        metadata.insert("thoughtsTokenCount".to_string(), json!(thoughts));
    }
    metadata.insert(
        "totalTokenCount".to_string(),
        json!(usage.total_tokens.unwrap_or(input + output)),
    );
    Value::Object(metadata)
}
