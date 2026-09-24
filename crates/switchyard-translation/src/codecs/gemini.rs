// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered Gemini generateContent bodies; the model belongs to the HTTP URL.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::FormatId;
use crate::llm::*;
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, object, push_lossy, push_unknown_field,
    validate_request_capabilities,
};

/// Identifier for the buffered Gemini generateContent codec.
pub const GEMINI_GENERATE_CONTENT: &str = "gemini_generate_content";

/// Converts native Gemini generateContent JSON to and from the neutral IR.
pub struct GeminiGenerateContentCodec;

fn invalid(path: &str, message: &str) -> TranslationError {
    TranslationError::InvalidValue {
        path: path.into(),
        message: message.into(),
    }
}

fn array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| TranslationError::InvalidType {
            path: path.into(),
            expected: "array",
        })
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(key, "expected a string"))
}

fn unsupported_fields(
    value: &Value,
    known: &[&str],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    for key in object(value, "$")?.keys() {
        if !known.contains(&key.as_str()) {
            push_unknown_field(diagnostics, policy, key)?;
            push_lossy(
                diagnostics,
                policy,
                format!("Gemini field {key} is retained only by exact preservation"),
            )?;
        }
    }
    Ok(())
}

fn schema(value: &Value) -> Value {
    let mut normalized = match value {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, value)| {
                    let value = if key == "type" {
                        value
                            .as_str()
                            .map(|v| json!(v.to_ascii_lowercase()))
                            .unwrap_or_else(|| value.clone())
                    } else if key == "properties" {
                        value
                            .as_object()
                            .map(|props| {
                                Value::Object(
                                    props
                                        .iter()
                                        .map(|(name, field)| (name.clone(), schema(field)))
                                        .collect(),
                                )
                            })
                            .unwrap_or_else(|| value.clone())
                    } else if matches!(
                        key.as_str(),
                        "items" | "anyOf" | "allOf" | "oneOf" | "additionalProperties"
                    ) {
                        schema(value)
                    } else {
                        value.clone()
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(schema).collect()),
        _ => value.clone(),
    };
    if let Some(fields) = normalized.as_object_mut() {
        let nullable = fields.remove("nullable");
        fields.remove("propertyOrdering");
        if nullable == Some(Value::Bool(true)) {
            return json!({"anyOf":[normalized,{"type":"null"}]});
        }
    }
    normalized
}

#[derive(Default)]
struct Calls {
    names: BTreeMap<String, String>,
    pending: Vec<(String, String)>,
    counter: usize,
    reserved: BTreeSet<String>,
}

impl Calls {
    fn for_body(body: &Value) -> Self {
        let mut calls = Self::default();
        let contents = body
            .get("contents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(
                body.get("candidates")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|c| c.get("content")),
            );
        for content in contents {
            for part in content
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(id) = part.pointer("/functionCall/id").and_then(Value::as_str) {
                    calls.reserved.insert(id.into());
                }
            }
        }
        calls
    }

    fn decode_parts(
        &mut self,
        content: &Value,
        policy: &TranslationPolicy,
        diagnostics: &mut Vec<TranslationDiagnostic>,
    ) -> Result<Vec<ContentBlock>> {
        unsupported_fields(content, &["role", "parts"], diagnostics, policy)?;
        let parts = array(
            content
                .get("parts")
                .ok_or_else(|| invalid("parts", "missing parts"))?,
            "parts",
        )?;
        let mut blocks = Vec::new();
        for part in parts {
            object(part, "parts[]")?;
            let union = [
                "text",
                "functionCall",
                "functionResponse",
                "inlineData",
                "fileData",
                "executableCode",
                "codeExecutionResult",
            ];
            if union.iter().filter(|key| part.get(**key).is_some()).count() > 1 {
                return Err(invalid("parts[]", "multiple content union members"));
            }
            unsupported_fields(
                part,
                &[
                    "text",
                    "functionCall",
                    "functionResponse",
                    "inlineData",
                    "fileData",
                    "thought",
                    "thoughtSignature",
                ],
                diagnostics,
                policy,
            )?;
            if part.get("thought").is_some_and(|v| v.as_bool().is_none())
                || part
                    .get("thoughtSignature")
                    .is_some_and(|v| v.as_str().is_none())
            {
                return Err(invalid("parts[]", "invalid thought metadata"));
            }
            if part.get("thought").and_then(Value::as_bool) == Some(true)
                && part.get("text").is_none()
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "Non-text Gemini thought parts cannot be projected as visible content",
                )?;
                blocks.push(ContentBlock::Unknown {
                    provider: GEMINI_GENERATE_CONTENT.into(),
                    raw: part.clone(),
                });
                continue;
            }
            if part.get("thoughtSignature").is_some()
                && part.get("text").is_none()
                && part.get("functionCall").is_none()
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "Gemini part signatures require exact native preservation",
                )?;
            }
            if let Some(text) = part.get("text") {
                let text = text
                    .as_str()
                    .ok_or_else(|| invalid("text", "expected a string"))?
                    .to_owned();
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    if part.get("thoughtSignature").is_some() {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini thought signatures require exact native preservation",
                        )?;
                    }
                    blocks.push(ContentBlock::Reasoning {
                        text,
                        signature: None,
                        details: vec![],
                    });
                } else {
                    if part.get("thoughtSignature").is_some() {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini signature on visible text has no neutral representation",
                        )?;
                    }
                    blocks.push(ContentBlock::Text { text });
                }
            } else if let Some(call) = part.get("functionCall") {
                unsupported_fields(call, &["name", "args", "id"], diagnostics, policy)?;
                let name = string(call, "name")?.to_owned();
                self.counter += 1;
                let id = match call.get("id").and_then(Value::as_str) {
                    Some(id) => id.to_owned(),
                    None => match &policy.deterministic_ids {
                        DeterministicIdPolicy::GenerateStable { prefix } => loop {
                            let id = format!("{prefix}_gemini_{}", self.counter);
                            if !self.names.contains_key(&id) && !self.reserved.contains(&id) {
                                break id;
                            }
                            self.counter += 1;
                        },
                        _ => {
                            return Err(invalid(
                                "functionCall.id",
                                "missing tool ID with generation disabled",
                            ));
                        }
                    },
                };
                let arguments = call.get("args").cloned().unwrap_or_else(|| json!({}));
                object(&arguments, "functionCall.args")?;
                if self.names.contains_key(&id) {
                    return Err(invalid("functionCall.id", "duplicate function call ID"));
                }
                self.names.insert(id.clone(), name.clone());
                self.pending.push((id.clone(), name.clone()));
                if part.get("thoughtSignature").is_some() {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Gemini function-call thought signature is retained only by exact preservation",
                    )?;
                }
                blocks.push(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments,
                }));
            } else if let Some(result) = part.get("functionResponse") {
                unsupported_fields(result, &["name", "response", "id"], diagnostics, policy)?;
                let name = string(result, "name")?;
                if let Some(id) = result.get("id").and_then(Value::as_str)
                    && self.names.get(id).is_some_and(|expected| expected != name)
                {
                    return Err(invalid(
                        "functionResponse.name",
                        "name does not match function call ID",
                    ));
                }
                let index = self.pending.iter().position(|(id, n)| {
                    n == name
                        && result
                            .get("id")
                            .and_then(Value::as_str)
                            .is_none_or(|wanted| id == wanted)
                });
                let id = if let Some(index) = index {
                    self.pending.remove(index).0
                } else {
                    result
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("functionResponse", "no matching function call"))?
                        .to_owned()
                };
                let response = result
                    .get("response")
                    .ok_or_else(|| invalid("functionResponse.response", "missing object"))?;
                object(response, "functionResponse.response")?;
                blocks.push(ContentBlock::ToolResult(ToolResult {
                    tool_call_id: id,
                    content: vec![ContentBlock::Text {
                        text: response.to_string(),
                    }],
                    is_error: response.get("error").map(|_| true),
                }));
            } else if let Some(blob) = part.get("inlineData") {
                unsupported_fields(blob, &["mimeType", "data"], diagnostics, policy)?;
                let mime = string(blob, "mimeType")?.to_owned();
                let data = string(blob, "data")?.to_owned();
                blocks.push(if mime.starts_with("image/") {
                    ContentBlock::Image {
                        source: ImageSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                    }
                } else if mime.starts_with("audio/") {
                    ContentBlock::Audio {
                        source: MediaSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                    }
                } else if mime.starts_with("video/") {
                    ContentBlock::Video {
                        source: MediaSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                    }
                } else {
                    ContentBlock::Unknown {
                        provider: GEMINI_GENERATE_CONTENT.into(),
                        raw: part.clone(),
                    }
                });
            } else if let Some(file) = part.get("fileData") {
                unsupported_fields(file, &["mimeType", "fileUri"], diagnostics, policy)?;
                let mime = string(file, "mimeType")?.to_owned();
                let url = string(file, "fileUri")?.to_owned();
                blocks.push(if mime.starts_with("audio/") {
                    ContentBlock::Audio {
                        source: MediaSource::Url {
                            url,
                            media_type: Some(mime),
                        },
                    }
                } else if mime.starts_with("video/") {
                    ContentBlock::Video {
                        source: MediaSource::Url {
                            url,
                            media_type: Some(mime),
                        },
                    }
                } else {
                    ContentBlock::Unknown {
                        provider: GEMINI_GENERATE_CONTENT.into(),
                        raw: part.clone(),
                    }
                });
            } else {
                push_lossy(
                    diagnostics,
                    policy,
                    "Gemini part has no normalized representation",
                )?;
                blocks.push(ContentBlock::Unknown {
                    provider: GEMINI_GENERATE_CONTENT.into(),
                    raw: part.clone(),
                });
            }
        }
        Ok(blocks)
    }

    fn encode_parts(
        &mut self,
        blocks: &[ContentBlock],
        policy: &TranslationPolicy,
        diagnostics: &mut Vec<TranslationDiagnostic>,
    ) -> Result<Vec<Value>> {
        let mut parts = Vec::new();
        for block in blocks {
            let part = match block {
                ContentBlock::Text { text } => json!({"text":text}),
                ContentBlock::Reasoning {
                    text,
                    signature,
                    details,
                } => {
                    if !details.is_empty() {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini cannot encode foreign reasoning details",
                        )?;
                    }
                    let part = json!({"text":text,"thought":true});
                    if signature.is_some() {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Opaque thought signatures require exact native preservation",
                        )?;
                    }
                    part
                }
                ContentBlock::ToolCall(call) => {
                    if self.names.contains_key(&call.id) {
                        return Err(invalid("functionCall.id", "duplicate function call ID"));
                    }
                    self.names.insert(call.id.clone(), call.name.clone());
                    object(&call.arguments, "functionCall.args")?;
                    json!({"functionCall":{"id":call.id,"name":call.name,"args":call.arguments}})
                }
                ContentBlock::ToolResult(result) => {
                    let name = self.names.get(&result.tool_call_id).ok_or_else(|| {
                        invalid("functionResponse", "tool result has no matching named call")
                    })?;
                    let text = crate::codecs::common::text_from_blocks(&result.content, "\n");
                    if result
                        .content
                        .iter()
                        .any(|b| !matches!(b, ContentBlock::Text { .. }))
                    {
                        push_lossy(
                            diagnostics,
                            policy,
                            "Gemini function response requires JSON/text tool output",
                        )?;
                    }
                    let mut value = serde_json::from_str::<Value>(&text)
                        .ok()
                        .filter(Value::is_object)
                        .unwrap_or_else(|| {
                            if result.is_error == Some(true) {
                                json!({"error":text})
                            } else {
                                json!({"output":text})
                            }
                        });
                    if result.is_error == Some(true) && value.get("error").is_none() {
                        value = json!({"error":value});
                    }
                    json!({"functionResponse":{"id":result.tool_call_id,"name":name,"response":value}})
                }
                ContentBlock::Image {
                    source:
                        ImageSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                }
                | ContentBlock::Audio {
                    source:
                        MediaSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                }
                | ContentBlock::Video {
                    source:
                        MediaSource::Base64 {
                            media_type: Some(mime),
                            data,
                        },
                } => json!({"inlineData":{"mimeType":mime,"data":data}}),
                ContentBlock::Audio {
                    source:
                        MediaSource::Url {
                            url,
                            media_type: Some(mime),
                        },
                }
                | ContentBlock::Video {
                    source:
                        MediaSource::Url {
                            url,
                            media_type: Some(mime),
                        },
                } => json!({"fileData":{"mimeType":mime,"fileUri":url}}),
                ContentBlock::Unknown { provider, raw }
                    if provider.as_str() == GEMINI_GENERATE_CONTENT =>
                {
                    raw.clone()
                }
                _ => {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Content block cannot be represented as a Gemini part",
                    )?;
                    continue;
                }
            };
            parts.push(part);
        }
        Ok(parts)
    }
}

impl FormatCodec for GeminiGenerateContentCodec {
    fn format(&self) -> FormatId {
        GEMINI_GENERATE_CONTENT.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        object(body, "$")?;
        let mut diagnostics = Vec::new();
        unsupported_fields(
            body,
            &[
                "contents",
                "systemInstruction",
                "tools",
                "toolConfig",
                "generationConfig",
            ],
            &mut diagnostics,
            policy,
        )?;
        let mut calls = Calls::for_body(body);
        let mut request = LlmRequest {
            preservation: capture_request_preservation(self.format(), body, policy),
            ..Default::default()
        };
        if let Some(system) = body.get("systemInstruction") {
            let content = calls.decode_parts(system, policy, &mut diagnostics)?;
            if content
                .iter()
                .any(|block| !matches!(block, ContentBlock::Text { .. }))
            {
                return Err(invalid(
                    "systemInstruction",
                    "Gemini system instructions support text only",
                ));
            }
            request.instructions.push(InstructionBlock {
                role: Role::System,
                content,
            });
        }
        for content in array(
            body.get("contents")
                .ok_or_else(|| invalid("contents", "missing contents"))?,
            "contents",
        )? {
            let role = match content.get("role").and_then(Value::as_str) {
                Some("model") => Role::Assistant,
                Some("user") | None => Role::User,
                Some(_) => return Err(invalid("contents.role", "expected user or model")),
            };
            let content = calls.decode_parts(content, policy, &mut diagnostics)?;
            request.messages.push(Message { role, content });
        }
        if let Some(tools) = body.get("tools") {
            for tool in array(tools, "tools")? {
                unsupported_fields(tool, &["functionDeclarations"], &mut diagnostics, policy)?;
                if let Some(declarations) = tool.get("functionDeclarations") {
                    for declaration in array(declarations, "functionDeclarations")? {
                        unsupported_fields(
                            declaration,
                            &["name", "description", "parameters", "parametersJsonSchema"],
                            &mut diagnostics,
                            policy,
                        )?;
                        request.tools.push(ToolDefinition {
                            name: string(declaration, "name")?.into(),
                            description: declaration
                                .get("description")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            parameters: schema(
                                &declaration
                                    .get("parametersJsonSchema")
                                    .or_else(|| declaration.get("parameters"))
                                    .cloned()
                                    .unwrap_or_else(|| json!({"type":"object","properties":{}})),
                            ),
                            strict: None,
                        });
                    }
                } else {
                    push_lossy(
                        &mut diagnostics,
                        policy,
                        "Gemini built-in tool has no neutral function declaration",
                    )?;
                }
            }
        }
        if let Some(config) = body.get("toolConfig") {
            unsupported_fields(config, &["functionCallingConfig"], &mut diagnostics, policy)?;
        }
        if let Some(config) = body.pointer("/toolConfig/functionCallingConfig") {
            unsupported_fields(
                config,
                &["mode", "allowedFunctionNames"],
                &mut diagnostics,
                policy,
            )?;
            request.tool_choice = match config.get("mode").and_then(Value::as_str) {
                Some("AUTO") => Some(ToolChoice::Auto),
                Some("NONE") => Some(ToolChoice::None),
                Some("ANY") => match config.get("allowedFunctionNames") {
                    None => Some(ToolChoice::Required),
                    Some(names) if names.as_array().is_some_and(|a| a.len() == 1) => {
                        Some(ToolChoice::Tool {
                            name: names[0]
                                .as_str()
                                .ok_or_else(|| invalid("allowedFunctionNames", "expected string"))?
                                .into(),
                        })
                    }
                    Some(_) => {
                        push_lossy(
                            &mut diagnostics,
                            policy,
                            "Gemini multiple allowed function names cannot be normalized",
                        )?;
                        None
                    }
                },
                _ => {
                    push_lossy(
                        &mut diagnostics,
                        policy,
                        "Gemini function calling mode cannot be normalized",
                    )?;
                    None
                }
            };
        }
        if let Some(config) = body.get("generationConfig") {
            unsupported_fields(
                config,
                &[
                    "temperature",
                    "topP",
                    "topK",
                    "maxOutputTokens",
                    "responseMimeType",
                    "responseJsonSchema",
                    "responseSchema",
                    "thinkingConfig",
                ],
                &mut diagnostics,
                policy,
            )?;
            for key in ["temperature", "topP"] {
                if config.get(key).is_some_and(|v| v.as_f64().is_none()) {
                    return Err(invalid(key, "expected a number"));
                }
            }
            for key in ["maxOutputTokens", "topK"] {
                if config.get(key).is_some_and(|v| v.as_u64().is_none()) {
                    return Err(invalid(key, "expected a non-negative integer"));
                }
            }
            request.sampling = SamplingParams {
                temperature: config.get("temperature").and_then(Value::as_f64),
                top_p: config.get("topP").and_then(Value::as_f64),
                top_k: config.get("topK").and_then(Value::as_i64),
            };
            request.output.max_output_tokens =
                config.get("maxOutputTokens").and_then(Value::as_u64);
            if config.get("responseMimeType").and_then(Value::as_str) == Some("application/json") {
                request.output.response_format = Some(
                    match config
                        .get("responseJsonSchema")
                        .or_else(|| config.get("responseSchema"))
                    {
                        Some(schema) => {
                            json!({"type":"json_schema","json_schema":{"name":"response","schema":self::schema(schema)}})
                        }
                        None => json!({"type":"json_object"}),
                    },
                );
            }
            if config
                .get("responseMimeType")
                .is_some_and(|v| !matches!(v.as_str(), Some("application/json" | "text/plain")))
                || (config.get("responseMimeType").and_then(Value::as_str)
                    != Some("application/json")
                    && (config.get("responseSchema").is_some()
                        || config.get("responseJsonSchema").is_some()))
            {
                push_lossy(
                    &mut diagnostics,
                    policy,
                    "Gemini output MIME/schema constraint has no portable representation",
                )?;
            }
            if let Some(thinking) = config.get("thinkingConfig") {
                request.reasoning.raw = Some(thinking.clone());
                push_lossy(
                    &mut diagnostics,
                    policy,
                    "Gemini thinking configuration has no portable reasoning control",
                )?;
            }
        }
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
        if let Some(body) = exact_preserved_request(&request.preservation, self.format(), policy) {
            return Ok(EncodedRequest {
                body,
                diagnostics: vec![],
            });
        }
        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, policy)?;
        if request.stream {
            push_lossy(
                &mut diagnostics,
                policy,
                "Buffered Gemini codec cannot request streaming",
            )?;
        }
        for key in request.extensions.fields.keys() {
            if key == crate::codecs::common::ANTHROPIC_REQUEST_KEY {
                continue;
            }
            push_lossy(
                &mut diagnostics,
                policy,
                format!("Request extension {key} is not a Gemini field"),
            )?;
        }
        if request.tools.iter().any(|tool| tool.strict.is_some()) {
            push_lossy(
                &mut diagnostics,
                policy,
                "Gemini function declarations do not support strict",
            )?;
        }
        let mut calls = Calls::default();
        let mut body = json!({"contents":[]});
        let instructions: Vec<_> = request
            .instructions
            .iter()
            .flat_map(|i| i.content.clone())
            .collect();
        if instructions
            .iter()
            .any(|block| !matches!(block, ContentBlock::Text { .. }))
        {
            return Err(invalid(
                "systemInstruction",
                "Gemini system instructions support text only",
            ));
        }
        if !instructions.is_empty() {
            body["systemInstruction"] =
                json!({"parts":calls.encode_parts(&instructions, policy, &mut diagnostics)?});
        }
        let mut contents = Vec::new();
        for message in &request.messages {
            let role = match message.role {
                Role::Assistant => "model",
                Role::User | Role::Tool => "user",
                _ => {
                    return Err(invalid(
                        "messages.role",
                        "instructions must be separate from conversation messages",
                    ));
                }
            };
            contents.push(json!({"role":role,"parts":calls.encode_parts(&message.content, policy, &mut diagnostics)?}));
        }
        body["contents"] = json!(contents);
        if !request.tools.is_empty() {
            body["tools"] = json!([{"functionDeclarations":request.tools.iter().map(|t| {
                let mut tool = json!({"name":t.name,"parametersJsonSchema":t.parameters});
                if let Some(description) = &t.description { tool["description"] = json!(description); }
                tool
            }).collect::<Vec<_>>()}]);
        }
        if let Some(choice) = &request.tool_choice {
            body["toolConfig"] = json!({"functionCallingConfig":match choice {
                ToolChoice::Auto => json!({"mode":"AUTO"}), ToolChoice::Required => json!({"mode":"ANY"}), ToolChoice::None => json!({"mode":"NONE"}), ToolChoice::Tool { name } => json!({"mode":"ANY","allowedFunctionNames":[name]}),
                ToolChoice::Raw(_) => { push_lossy(&mut diagnostics, policy, "Raw tool choice cannot be encoded as Gemini function calling configuration")?; json!({}) },
            }});
        }
        let mut config = serde_json::Map::new();
        for (key, value) in [
            (
                "temperature",
                request.sampling.temperature.map(|v| json!(v)),
            ),
            ("topP", request.sampling.top_p.map(|v| json!(v))),
            ("topK", request.sampling.top_k.map(|v| json!(v))),
            (
                "maxOutputTokens",
                request.output.max_output_tokens.map(|v| json!(v)),
            ),
        ] {
            if let Some(value) = value {
                config.insert(key.into(), value);
            }
        }
        if let Some(format) = &request.output.response_format {
            match format.get("type").and_then(Value::as_str) {
                Some("json_object") => {
                    config.insert("responseMimeType".into(), json!("application/json"));
                }
                Some("json_schema") => {
                    config.insert("responseMimeType".into(), json!("application/json"));
                    config.insert(
                        "responseJsonSchema".into(),
                        format
                            .pointer("/json_schema/schema")
                            .ok_or_else(|| invalid("response_format", "missing JSON schema"))?
                            .clone(),
                    );
                }
                _ => push_lossy(
                    &mut diagnostics,
                    policy,
                    "Response format cannot be encoded as Gemini JSON output",
                )?,
            }
        }
        if request.reasoning.raw.is_some() || request.reasoning.effort.is_some() {
            push_lossy(
                &mut diagnostics,
                policy,
                "Reasoning controls require native Gemini preservation",
            )?;
        }
        if !config.is_empty() {
            body["generationConfig"] = Value::Object(config);
        }
        Ok(EncodedRequest {
            body: embed_preservation(body, &request.preservation, policy),
            diagnostics,
        })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        object(body, "$")?;
        if let Some(error) = body.get("error") {
            return Err(TranslationError::UpstreamFailure {
                error: error.clone(),
            });
        }
        let mut diagnostics = Vec::new();
        unsupported_fields(
            body,
            &[
                "candidates",
                "responseId",
                "modelVersion",
                "usageMetadata",
                "promptFeedback",
            ],
            &mut diagnostics,
            policy,
        )?;
        let mut response = AggLlmResponse {
            id: body
                .get("responseId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            model: body
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(str::to_owned),
            preservation: capture_response_preservation(self.format(), body, policy),
            ..Default::default()
        };
        let mut calls = Calls::for_body(body);
        if let Some(candidates) = body.get("candidates") {
            for candidate in array(candidates, "candidates")? {
                unsupported_fields(
                    candidate,
                    &["index", "content", "finishReason"],
                    &mut diagnostics,
                    policy,
                )?;
                let content = candidate
                    .get("content")
                    .map(|c| calls.decode_parts(c, policy, &mut diagnostics))
                    .transpose()?
                    .unwrap_or_default();
                let stop_reason = candidate.get("finishReason").and_then(Value::as_str).map(
                    |reason| match reason {
                        "STOP"
                            if content
                                .iter()
                                .any(|b| matches!(b, ContentBlock::ToolCall(_))) =>
                        {
                            StopReason::ToolUse
                        }
                        "STOP" => StopReason::EndTurn,
                        "MAX_TOKENS" => StopReason::MaxTokens,
                        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
                        | "IMAGE_SAFETY" => StopReason::ContentFilter,
                        "MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" => StopReason::Error,
                        _ => StopReason::Unknown,
                    },
                );
                response.outputs.push(ResponseOutput {
                    role: Role::Assistant,
                    content,
                    stop_reason,
                    url_citations: vec![],
                });
            }
        }
        if let Some(usage) = body.get("usageMetadata") {
            let cached = usage.get("cachedContentTokenCount").and_then(Value::as_u64);
            response.usage = Usage {
                input_tokens: usage
                    .get("promptTokenCount")
                    .and_then(Value::as_u64)
                    .map(|n| n.saturating_sub(cached.unwrap_or(0))),
                cache: Usage::cache_details(cached, None),
                output_tokens: usage.get("candidatesTokenCount").and_then(Value::as_u64),
                reasoning_tokens: usage.get("thoughtsTokenCount").and_then(Value::as_u64),
                total_tokens: usage.get("totalTokenCount").and_then(Value::as_u64),
            };
        }
        if body.pointer("/promptFeedback/blockReason").is_some() && response.outputs.is_empty() {
            response.outputs.push(ResponseOutput {
                role: Role::Assistant,
                content: vec![],
                stop_reason: Some(StopReason::ContentFilter),
                url_citations: vec![],
            });
        }
        Ok(DecodedResponse {
            response,
            diagnostics,
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) = exact_preserved_response(&response.preservation, self.format(), policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: vec![],
            });
        }
        let mut diagnostics = Vec::new();
        let mut calls = Calls::default();
        for key in response.extensions.fields.keys() {
            push_lossy(
                &mut diagnostics,
                policy,
                format!("Response extension {key} is not a Gemini field"),
            )?;
        }
        if response.usage.cache_creation_input_tokens().is_some() {
            push_lossy(
                &mut diagnostics,
                policy,
                "Gemini usage cannot report cache creation separately",
            )?;
        }
        let mut candidates = Vec::new();
        for (index, output) in response.outputs.iter().enumerate() {
            if output.role != Role::Assistant {
                return Err(invalid(
                    "outputs.role",
                    "Gemini candidates must be model output",
                ));
            }
            let mut candidate = json!({"index":index,"content":{"role":"model","parts":calls.encode_parts(&output.content, policy, &mut diagnostics)?}});
            if !output.url_citations.is_empty() {
                push_lossy(
                    &mut diagnostics,
                    policy,
                    "Neutral URL citations cannot be encoded as Gemini grounding metadata",
                )?;
            }
            if let Some(reason) = output.stop_reason {
                candidate["finishReason"] = json!(match reason {
                    StopReason::EndTurn | StopReason::ToolUse => "STOP",
                    StopReason::MaxTokens => "MAX_TOKENS",
                    StopReason::ContentFilter => "SAFETY",
                    StopReason::Error | StopReason::Unknown => "OTHER",
                });
            }
            candidates.push(candidate);
        }
        let mut body = json!({"candidates":candidates});
        if let Some(id) = &response.id {
            body["responseId"] = json!(id);
        }
        if let Some(model) = &response.model {
            body["modelVersion"] = json!(model);
        }
        let usage = &response.usage;
        let mut counts = serde_json::Map::new();
        for (key, count) in [
            (
                "promptTokenCount",
                usage.input_tokens.map(|n| {
                    n.saturating_add(usage.cached_input_tokens().unwrap_or(0))
                        .saturating_add(usage.cache_creation_input_tokens().unwrap_or(0))
                }),
            ),
            ("cachedContentTokenCount", usage.cached_input_tokens()),
            ("candidatesTokenCount", usage.output_tokens),
            ("thoughtsTokenCount", usage.reasoning_tokens),
            ("totalTokenCount", usage.total_tokens),
        ] {
            if let Some(count) = count {
                counts.insert(key.into(), json!(count));
            }
        }
        if !counts.is_empty() {
            body["usageMetadata"] = Value::Object(counts);
        }
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, policy),
            diagnostics,
        })
    }
}
