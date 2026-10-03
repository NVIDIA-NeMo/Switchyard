// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered Bedrock Converse JSON conversion. The host supplies the URL-owned model ID.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use crate::codecs::common::provider_extensions;
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, ImageSource, InstructionBlock, LlmRequest, Message,
    ProviderExtensions, ResponseOutput, Role, StopReason, ToolCall, ToolChoice, ToolDefinition,
    ToolResult, Usage,
};
use crate::policy::TranslationPolicy;
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, object, push_lossy, push_unknown_field,
    reject_responses_builtin_tool_item, validate_request_capabilities,
};

const NATIVE_REQUEST: &str = "_bedrock_converse_request";
const NATIVE_RESPONSE: &str = "_bedrock_converse_response";

/// Buffered codec for Bedrock Converse bodies.
pub struct BedrockConverseCodec;

impl FormatCodec for BedrockConverseCodec {
    fn format(&self) -> FormatId {
        WireFormat::BedrockConverse.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let obj = object(body, "$")?;
        let mut diagnostics = Vec::new();
        check_unknown(
            obj,
            &[
                "messages",
                "system",
                "inferenceConfig",
                "toolConfig",
                "additionalModelRequestFields",
                "additionalModelResponseFieldPaths",
                "guardrailConfig",
                "promptVariables",
                "requestMetadata",
                "performanceConfig",
                "serviceTier",
                "outputConfig",
                "metadata",
            ],
            "$",
            &mut diagnostics,
            policy,
        )?;
        let mut request = LlmRequest {
            preservation: capture_request_preservation(WireFormat::BedrockConverse, body, policy),
            ..LlmRequest::default()
        };
        if let Some(system) = obj.get("system") {
            let content = decode_content(
                array(system, "$.system")?,
                "$.system",
                &mut diagnostics,
                policy,
            )?;
            request.instructions.push(InstructionBlock {
                role: Role::System,
                content,
            });
        }
        let messages = array(required(obj, "messages", "$")?, "$.messages")?;
        for (index, value) in messages.iter().enumerate() {
            let path = format!("$.messages[{index}]");
            let message = object(value, &path)?;
            let role = decode_role(required_string(message, "role", &path)?, &path)?;
            let content = decode_content(
                array(
                    required(message, "content", &path)?,
                    &format!("{path}.content"),
                )?,
                &format!("{path}.content"),
                &mut diagnostics,
                policy,
            )?;
            request.messages.push(Message { role, content });
        }
        if let Some(config) = obj.get("inferenceConfig") {
            let config = object(config, "$.inferenceConfig")?;
            check_unknown(
                config,
                &["maxTokens", "temperature", "topP", "stopSequences"],
                "$.inferenceConfig",
                &mut diagnostics,
                policy,
            )?;
            request.output.max_output_tokens =
                optional_u64(config, "maxTokens", "$.inferenceConfig")?;
            request.sampling.temperature =
                optional_f64(config, "temperature", "$.inferenceConfig")?;
            request.sampling.top_p = optional_f64(config, "topP", "$.inferenceConfig")?;
            if let Some(stop) = config.get("stopSequences") {
                if array(stop, "$.inferenceConfig.stopSequences")?
                    .iter()
                    .any(|v| !v.is_string())
                {
                    return Err(invalid(
                        "$.inferenceConfig.stopSequences",
                        "expected strings",
                    ));
                }
                request
                    .extensions
                    .fields
                    .insert("stop".into(), stop.clone());
                request
                    .extensions
                    .fields
                    .insert("stop_sequences".into(), stop.clone());
            }
        }
        if let Some(config) = obj.get("toolConfig") {
            let config = object(config, "$.toolConfig")?;
            for (index, value) in array(
                required(config, "tools", "$.toolConfig")?,
                "$.toolConfig.tools",
            )?
            .iter()
            .enumerate()
            {
                let path = format!("$.toolConfig.tools[{index}].toolSpec");
                let tool = object(value, &format!("$.toolConfig.tools[{index}]"))?;
                if tool.len() != 1 {
                    return Err(invalid(
                        "$.toolConfig.tools",
                        "tool must contain one union member",
                    ));
                }
                if let Some(spec) = value.get("toolSpec") {
                    let spec = object(spec, &path)?;
                    let schema = object(
                        required(spec, "inputSchema", &path)?,
                        &format!("{path}.inputSchema"),
                    )?;
                    let strict = spec
                        .get("strict")
                        .map(|v| {
                            v.as_bool().ok_or_else(|| {
                                invalid(&format!("{path}.strict"), "expected boolean")
                            })
                        })
                        .transpose()?;
                    request.tools.push(ToolDefinition {
                        name: required_string(spec, "name", &path)?.into(),
                        description: spec
                            .get("description")
                            .map(|v| string(v, &format!("{path}.description")).map(str::to_owned))
                            .transpose()?,
                        parameters: required(schema, "json", &format!("{path}.inputSchema"))?
                            .clone(),
                        strict,
                    });
                } else {
                    push_lossy(
                        &mut diagnostics,
                        policy,
                        "Bedrock non-function tool has no neutral tool definition",
                    )?;
                }
            }
            request.tool_choice = config
                .get("toolChoice")
                .map(decode_tool_choice)
                .transpose()?;
        }
        request.extensions.fields.insert(
            NATIVE_REQUEST.into(),
            Value::Object(provider_extensions(
                obj,
                &[
                    "messages",
                    "system",
                    "inferenceConfig",
                    "toolConfig",
                    "metadata",
                ],
            )),
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
        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, policy)?;
        if let Some(body) =
            exact_preserved_request(&request.preservation, WireFormat::BedrockConverse, policy)
        {
            return Ok(EncodedRequest { body, diagnostics });
        }
        for (unsupported, label) in [
            (request.sampling.top_k.is_some(), "top_k"),
            (
                request.reasoning.effort.is_some() || request.reasoning.raw.is_some(),
                "reasoning controls",
            ),
            (
                request.output.response_format.is_some()
                    || request.output.is_schema_enforced == Some(true),
                "response format",
            ),
            (
                request
                    .extensions
                    .fields
                    .contains_key("parallel_tool_calls"),
                "parallel_tool_calls",
            ),
        ] {
            if unsupported {
                push_lossy(
                    &mut diagnostics,
                    policy,
                    format!("Bedrock Converse has no portable mapping for {label}"),
                )?;
            }
        }
        let mut body = Map::new();
        let mut system = Vec::new();
        for instruction in &request.instructions {
            for block in &instruction.content {
                system.push(encode_system_block(block, &mut diagnostics, policy)?);
            }
        }
        let mut messages = Vec::new();
        for message in &request.messages {
            if matches!(message.role, Role::System | Role::Developer) {
                for block in &message.content {
                    system.push(encode_system_block(block, &mut diagnostics, policy)?);
                }
            } else {
                messages.push(json!({"role": encode_role(message.role),
                    "content": encode_content(&message.content, &mut diagnostics, policy)?}));
            }
        }
        if !system.is_empty() {
            body.insert("system".into(), Value::Array(system));
        }
        body.insert("messages".into(), Value::Array(messages));
        let mut inference = Map::new();
        if let Some(v) = request.output.max_output_tokens {
            inference.insert("maxTokens".into(), v.into());
        }
        if let Some(v) = request.sampling.temperature {
            inference.insert("temperature".into(), v.into());
        }
        if let Some(v) = request.sampling.top_p {
            inference.insert("topP".into(), v.into());
        }
        if let Some(v) = request
            .extensions
            .fields
            .get("stop_sequences")
            .or_else(|| request.extensions.fields.get("stop"))
        {
            let v = if v.is_string() { json!([v]) } else { v.clone() };
            inference.insert("stopSequences".into(), v);
        }
        if !inference.is_empty() {
            body.insert("inferenceConfig".into(), Value::Object(inference));
        }
        // Converse has no `none` tool-choice variant: disabling tools means omitting the config.
        if request.tool_choice != Some(ToolChoice::None) && !request.tools.is_empty() {
            let tools = request
                .tools
                .iter()
                .map(|tool| {
                    let mut spec =
                        json!({"name": tool.name, "inputSchema": {"json": tool.parameters}});
                    if let Some(v) = &tool.description {
                        spec["description"] = v.clone().into();
                    }
                    if let Some(v) = tool.strict {
                        spec["strict"] = v.into();
                    }
                    json!({"toolSpec": spec})
                })
                .collect::<Vec<_>>();
            let mut config = json!({"tools": tools});
            if let Some(choice) = &request.tool_choice {
                config["toolChoice"] = encode_tool_choice(choice)?;
            }
            body.insert("toolConfig".into(), config);
        } else if request.tools.is_empty()
            && matches!(
                request.tool_choice,
                Some(ToolChoice::Required | ToolChoice::Tool { .. })
            )
        {
            return Err(invalid("$.toolConfig", "required tool choice needs tools"));
        }
        if let Some(native) = request
            .extensions
            .fields
            .get(NATIVE_REQUEST)
            .and_then(Value::as_object)
        {
            for key in [
                "additionalModelRequestFields",
                "additionalModelResponseFieldPaths",
                "guardrailConfig",
                "promptVariables",
                "requestMetadata",
                "performanceConfig",
                "serviceTier",
                "outputConfig",
            ] {
                if let Some(value) = native.get(key) {
                    body.insert(key.into(), value.clone());
                }
            }
        }
        Ok(EncodedRequest {
            body: embed_preservation(Value::Object(body), &request.preservation, policy),
            diagnostics,
        })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let obj = object(body, "$")?;
        if let Some(error) = obj.iter().find(|(key, _)| key.ends_with("Exception")) {
            return Err(TranslationError::UpstreamFailure {
                error: json!({error.0: error.1}),
            });
        }
        let mut diagnostics = Vec::new();
        check_unknown(
            obj,
            &[
                "output",
                "stopReason",
                "usage",
                "metrics",
                "additionalModelResponseFields",
                "trace",
                "performanceConfig",
                "serviceTier",
                "metadata",
            ],
            "$",
            &mut diagnostics,
            policy,
        )?;
        let output = object(required(obj, "output", "$")?, "$.output")?;
        let message = object(required(output, "message", "$.output")?, "$.output.message")?;
        let role = decode_role(
            required_string(message, "role", "$.output.message")?,
            "$.output.message",
        )?;
        if role != Role::Assistant {
            return Err(invalid("$.output.message.role", "expected assistant"));
        }
        let content = decode_content(
            array(
                required(message, "content", "$.output.message")?,
                "$.output.message.content",
            )?,
            "$.output.message.content",
            &mut diagnostics,
            policy,
        )?;
        let reason = required_string(obj, "stopReason", "$")?;
        let response = AggLlmResponse {
            outputs: vec![ResponseOutput {
                role,
                content,
                stop_reason: Some(decode_stop_reason(reason)),
                url_citations: Vec::new(),
            }],
            usage: decode_usage(required(obj, "usage", "$")?)?,
            extensions: ProviderExtensions {
                fields: Map::from_iter([(
                    NATIVE_RESPONSE.into(),
                    Value::Object(provider_extensions(obj, &["output", "usage", "metadata"])),
                )]),
            },
            preservation: capture_response_preservation(WireFormat::BedrockConverse, body, policy),
            ..AggLlmResponse::default()
        };
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
        super::super::responses::validate_response_output(response, WireFormat::BedrockConverse)?;
        if let Some(body) =
            exact_preserved_response(&response.preservation, WireFormat::BedrockConverse, policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        if response.outputs.len() > 1 {
            push_lossy(
                &mut diagnostics,
                policy,
                "Bedrock Converse returns one output message",
            )?;
        }
        let output = response
            .outputs
            .first()
            .ok_or_else(|| invalid("$.output", "expected an assistant output"))?;
        if output.role != Role::Assistant {
            return Err(invalid("$.output.message.role", "expected assistant"));
        }
        if !output.url_citations.is_empty() {
            push_lossy(
                &mut diagnostics,
                policy,
                "Bedrock output does not encode neutral URL citations",
            )?;
        }
        let mut body = json!({"output": {"message": {"role": "assistant",
            "content": encode_content(&output.content, &mut diagnostics, policy)?}},
            "stopReason": encode_stop_reason(output.stop_reason.unwrap_or(StopReason::EndTurn))?,
            "usage": encode_usage(&response.usage)?});
        if let Some(native) = response
            .extensions
            .fields
            .get(NATIVE_RESPONSE)
            .and_then(Value::as_object)
        {
            for key in [
                "metrics",
                "additionalModelResponseFields",
                "trace",
                "performanceConfig",
                "serviceTier",
            ] {
                if let Some(v) = native.get(key) {
                    body[key] = v.clone();
                }
            }
            if let Some(reason) = native.get("stopReason").and_then(Value::as_str)
                && Some(decode_stop_reason(reason)) == output.stop_reason
            {
                body["stopReason"] = reason.into();
            }
        }
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, policy),
            diagnostics,
        })
    }
}

fn decode_content(
    blocks: &[Value],
    path: &str,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    blocks
        .iter()
        .enumerate()
        .map(|(i, block)| decode_block(block, &format!("{path}[{i}]"), diagnostics, policy))
        .collect()
}

fn decode_block(
    block: &Value,
    path: &str,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<ContentBlock> {
    let obj = object(block, path)?;
    if obj.len() != 1 {
        return Err(invalid(path, "content block must contain one union member"));
    }
    if let Some(text) = obj.get("text") {
        return Ok(ContentBlock::Text {
            text: string(text, path)?.into(),
        });
    }
    if let Some(tool) = obj.get("toolUse") {
        let tool = object(tool, path)?;
        return Ok(ContentBlock::ToolCall(ToolCall {
            id: required_string(tool, "toolUseId", path)?.into(),
            name: required_string(tool, "name", path)?.into(),
            arguments: required(tool, "input", path)?.clone(),
        }));
    }
    if let Some(tool) = obj.get("toolResult") {
        let tool = object(tool, path)?;
        let mut content = Vec::new();
        for (i, value) in array(required(tool, "content", path)?, path)?
            .iter()
            .enumerate()
        {
            let p = format!("{path}.toolResult.content[{i}]");
            if object(value, &p)?.len() != 1 {
                return Err(invalid(&p, "tool result must contain one union member"));
            }
            if value.get("json").is_some() {
                content.push(ContentBlock::Text {
                    text: value["json"].to_string(),
                });
            } else {
                content.push(decode_block(value, &p, diagnostics, policy)?);
            }
        }
        let is_error = match tool.get("status").map(|v| string(v, path)).transpose()? {
            Some("error") => Some(true),
            Some("success") => Some(false),
            None => None,
            _ => return Err(invalid(path, "invalid tool result status")),
        };
        return Ok(ContentBlock::ToolResult(ToolResult {
            tool_call_id: required_string(tool, "toolUseId", path)?.into(),
            content,
            is_error,
        }));
    }
    if let Some(reasoning) = obj.get("reasoningContent") {
        let reasoning = object(reasoning, path)?;
        if reasoning.len() != 1 {
            return Err(invalid(path, "reasoning must contain one union member"));
        }
        if let Some(text) = reasoning.get("reasoningText") {
            let text = object(text, path)?;
            return Ok(ContentBlock::Reasoning {
                text: required_string(text, "text", path)?.into(),
                signature: text
                    .get("signature")
                    .map(|v| string(v, path).map(str::to_owned))
                    .transpose()?,
                details: Vec::new(),
            });
        }
        if reasoning.contains_key("redactedContent") {
            return Ok(ContentBlock::Unknown {
                provider: WireFormat::BedrockConverse.into(),
                raw: block.clone(),
            });
        }
        return Err(invalid(path, "invalid reasoning content"));
    }
    if let Some(image) = obj.get("image") {
        let image = object(image, path)?;
        let format = required_string(image, "format", path)?;
        if let Some(data) = image.get("source").and_then(|s| s.get("bytes")) {
            return Ok(ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: Some(format!("image/{format}")),
                    data: string(data, path)?.into(),
                },
            });
        }
    }
    push_unknown_field(diagnostics, policy, path)?;
    Ok(ContentBlock::Unknown {
        provider: WireFormat::BedrockConverse.into(),
        raw: block.clone(),
    })
}

fn encode_content(
    blocks: &[ContentBlock],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    blocks
        .iter()
        .map(|b| encode_block(b, diagnostics, policy))
        .collect()
}

fn encode_system_block(
    block: &ContentBlock,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    match block {
        ContentBlock::Text { .. } => encode_block(block, diagnostics, policy),
        ContentBlock::Unknown { provider, raw }
            if provider.as_str() == WireFormat::BedrockConverse.as_str() =>
        {
            Ok(raw.clone())
        }
        _ => Err(invalid("$.system", "unsupported system content")),
    }
}

fn encode_block(
    block: &ContentBlock,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    Ok(match block {
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => json!({"text": text}),
        ContentBlock::Reasoning {
            text,
            signature,
            details,
        } => {
            let mut native_signature = String::new();
            let mut redacted = Vec::new();
            for detail in details {
                match detail.get("type").and_then(Value::as_str) {
                    Some("bedrock.signature_delta") => {
                        if let Some(s) = detail.get("signature").and_then(Value::as_str) {
                            native_signature.push_str(s);
                        }
                    }
                    Some("bedrock.redacted_content") => {
                        if let Some(v) = detail.get("data") {
                            redacted.push(v.clone());
                        }
                    }
                    _ => push_lossy(
                        diagnostics,
                        policy,
                        "opaque reasoning details have no Bedrock mapping",
                    )?,
                }
            }
            if !redacted.is_empty() {
                if !text.is_empty() || signature.is_some() || !native_signature.is_empty() {
                    return Err(invalid(
                        "$.reasoningContent",
                        "redacted reasoning cannot be combined with reasoning text or signatures",
                    ));
                }
                let mut bytes = Vec::new();
                for fragment in redacted {
                    let encoded = string(&fragment, "$.reasoningContent.redactedContent")?;
                    bytes.extend(STANDARD.decode(encoded).map_err(|_| {
                        invalid("$.reasoningContent.redactedContent", "invalid base64")
                    })?);
                }
                json!({"reasoningContent": {"redactedContent": STANDARD.encode(bytes)}})
            } else {
                let mut reasoning = json!({"text": text});
                if let Some(signature) = signature {
                    reasoning["signature"] = signature.clone().into();
                } else if !native_signature.is_empty() {
                    reasoning["signature"] = native_signature.into();
                }
                json!({"reasoningContent": {"reasoningText": reasoning}})
            }
        }
        ContentBlock::ToolCall(call) => {
            let input = if let Value::String(raw) = &call.arguments {
                serde_json::from_str::<Value>(raw)
                    .map_err(|_| invalid("$.toolUse.input", "invalid tool arguments JSON"))?
            } else {
                call.arguments.clone()
            };
            json!({"toolUse": {"toolUseId": call.id, "name": call.name, "input": input}})
        }
        ContentBlock::ToolResult(result) => {
            let mut tool = json!({"toolUseId": result.tool_call_id, "content": encode_content(&result.content, diagnostics, policy)?});
            if let Some(is_error) = result.is_error {
                tool["status"] = if is_error { "error" } else { "success" }.into();
            }
            json!({"toolResult": tool})
        }
        ContentBlock::Image {
            source: ImageSource::Base64 { media_type, data },
        } => {
            let format = media_type
                .as_deref()
                .and_then(|m| m.strip_prefix("image/"))
                .ok_or_else(|| invalid("$.content.image", "inline image requires its MIME type"))?;
            let format = if format == "jpg" { "jpeg" } else { format };
            if !matches!(format, "png" | "jpeg" | "gif" | "webp") {
                return Err(invalid("$.content.image", "unsupported image format"));
            }
            json!({"image": {"format": format, "source": {"bytes": data}}})
        }
        ContentBlock::Image {
            source: ImageSource::Url { url, .. },
        } if url.starts_with("data:") => {
            let (media, data) = url[5..]
                .split_once(";base64,")
                .ok_or_else(|| invalid("$.content.image", "expected a base64 data URL"))?;
            return encode_block(
                &ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: Some(media.into()),
                        data: data.into(),
                    },
                },
                diagnostics,
                policy,
            );
        }
        ContentBlock::Unknown { provider, raw }
            if provider.as_str() == WireFormat::BedrockConverse.as_str() =>
        {
            raw.clone()
        }
        ContentBlock::Unknown { provider, raw } => {
            reject_responses_builtin_tool_item(provider, raw, WireFormat::BedrockConverse)?;
            return Err(TranslationError::UnsupportedTranslation {
                from: provider.clone(),
                to: WireFormat::BedrockConverse.into(),
            });
        }
        _ => {
            return Err(invalid(
                "$.content",
                "content has no Bedrock Converse mapping",
            ));
        }
    })
}

fn decode_tool_choice(value: &Value) -> Result<ToolChoice> {
    let obj = object(value, "$.toolConfig.toolChoice")?;
    if obj.len() != 1 {
        return Err(invalid(
            "$.toolConfig.toolChoice",
            "expected one union member",
        ));
    }
    if obj.get("auto").is_some_and(Value::is_object) {
        Ok(ToolChoice::Auto)
    } else if obj.get("any").is_some_and(Value::is_object) {
        Ok(ToolChoice::Required)
    } else if let Some(tool) = obj.get("tool") {
        Ok(ToolChoice::Tool {
            name: required_string(
                object(tool, "$.toolConfig.toolChoice.tool")?,
                "name",
                "$.toolConfig.toolChoice.tool",
            )?
            .into(),
        })
    } else {
        Err(invalid("$.toolConfig.toolChoice", "unknown tool choice"))
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Result<Value> {
    match choice {
        ToolChoice::Auto => Ok(json!({"auto": {}})),
        ToolChoice::Required => Ok(json!({"any": {}})),
        ToolChoice::Tool { name } => Ok(json!({"tool": {"name": name}})),
        _ => Err(invalid(
            "$.toolConfig.toolChoice",
            "unsupported tool choice",
        )),
    }
}

fn decode_role(role: &str, path: &str) -> Result<Role> {
    match role {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        _ => Err(TranslationError::unsupported_role(
            format!("{path}.role"),
            role,
        )),
    }
}
fn encode_role(role: Role) -> &'static str {
    if role == Role::Assistant {
        "assistant"
    } else {
        "user"
    }
}

pub(super) fn decode_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "content_filtered" | "guardrail_intervened" => StopReason::ContentFilter,
        "malformed_model_output" | "malformed_tool_use" => StopReason::Error,
        _ => StopReason::Unknown,
    }
}
fn encode_stop_reason(reason: StopReason) -> Result<&'static str> {
    match reason {
        StopReason::EndTurn => Ok("end_turn"),
        StopReason::MaxTokens => Ok("max_tokens"),
        StopReason::ToolUse => Ok("tool_use"),
        StopReason::ContentFilter => Ok("content_filtered"),
        StopReason::Error => Ok("malformed_model_output"),
        StopReason::Unknown => Err(invalid(
            "$.stopReason",
            "unknown stop reason has no Bedrock mapping",
        )),
    }
}

pub(super) fn decode_usage(value: &Value) -> Result<Usage> {
    let obj = object(value, "$.usage")?;
    Ok(Usage {
        input_tokens: Some(unsigned(
            required(obj, "inputTokens", "$.usage")?,
            "$.usage.inputTokens",
        )?),
        output_tokens: Some(unsigned(
            required(obj, "outputTokens", "$.usage")?,
            "$.usage.outputTokens",
        )?),
        total_tokens: Some(unsigned(
            required(obj, "totalTokens", "$.usage")?,
            "$.usage.totalTokens",
        )?),
        cache: Usage::cache_details(
            optional_u64(obj, "cacheReadInputTokens", "$.usage")?,
            optional_u64(obj, "cacheWriteInputTokens", "$.usage")?,
        ),
        ..Usage::default()
    })
}
pub(super) fn encode_usage(usage: &Usage) -> Result<Value> {
    let input = usage.input_tokens.ok_or_else(|| {
        invalid(
            "$.usage.inputTokens",
            "Bedrock requires reported input tokens",
        )
    })?;
    let output = usage.output_tokens.ok_or_else(|| {
        invalid(
            "$.usage.outputTokens",
            "Bedrock requires reported output tokens",
        )
    })?;
    let total = usage.total_tokens.unwrap_or_else(|| {
        input
            .saturating_add(output)
            .saturating_add(usage.cached_input_tokens().unwrap_or(0))
            .saturating_add(usage.cache_creation_input_tokens().unwrap_or(0))
    });
    let mut value = json!({"inputTokens": input, "outputTokens": output, "totalTokens": total});
    if let Some(n) = usage.cached_input_tokens() {
        value["cacheReadInputTokens"] = n.into();
    }
    if let Some(n) = usage.cache_creation_input_tokens() {
        value["cacheWriteInputTokens"] = n.into();
    }
    Ok(value)
}

pub(crate) fn request_projection_diagnostics(
    request: &LlmRequest,
    target: &FormatId,
    policy: &TranslationPolicy,
) -> Result<Vec<TranslationDiagnostic>> {
    let mut diagnostics = Vec::new();
    if target.as_str() == WireFormat::BedrockConverse.as_str() {
        return Ok(diagnostics);
    }
    if let Some(native) = request
        .extensions
        .fields
        .get(NATIVE_REQUEST)
        .and_then(Value::as_object)
    {
        for key in native.keys().filter(|key| {
            !matches!(
                key.as_str(),
                "messages" | "system" | "inferenceConfig" | "toolConfig" | "metadata"
            )
        }) {
            push_lossy(
                &mut diagnostics,
                policy,
                format!("Bedrock {key} has no mapping to {target}"),
            )?;
        }
    }
    Ok(diagnostics)
}

fn check_unknown(
    obj: &Map<String, Value>,
    known: &[&str],
    path: &str,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    for key in obj.keys() {
        if !known.contains(&key.as_str()) {
            push_unknown_field(diagnostics, policy, format!("{path}.{key}"))?;
        }
    }
    Ok(())
}
fn required<'a>(obj: &'a Map<String, Value>, key: &str, path: &str) -> Result<&'a Value> {
    obj.get(key)
        .ok_or_else(|| invalid(&format!("{path}.{key}"), "missing required field"))
}
fn required_string<'a>(obj: &'a Map<String, Value>, key: &str, path: &str) -> Result<&'a str> {
    string(required(obj, key, path)?, &format!("{path}.{key}"))
}
fn string<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| invalid(path, "expected string"))
}
fn array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(path, "expected array"))
}
fn unsigned(value: &Value, path: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| invalid(path, "expected non-negative integer"))
}

fn optional_u64(obj: &Map<String, Value>, key: &str, path: &str) -> Result<Option<u64>> {
    obj.get(key)
        .map(|v| unsigned(v, &format!("{path}.{key}")))
        .transpose()
}
fn optional_f64(obj: &Map<String, Value>, key: &str, path: &str) -> Result<Option<f64>> {
    obj.get(key)
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| invalid(&format!("{path}.{key}"), "expected number"))
        })
        .transpose()
}
fn invalid(path: &str, message: &str) -> TranslationError {
    TranslationError::InvalidValue {
        path: path.into(),
        message: message.into(),
    }
}
