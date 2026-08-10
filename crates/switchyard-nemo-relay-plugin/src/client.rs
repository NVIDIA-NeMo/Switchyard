// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard-owned HTTP clients bound to one semantic routing target.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as Json;
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_protocol::{
    ContentBlock, Context, Decision, LlmClientError, Message, Request, Response, Role,
    RoutedLlmClient, ToolCall, ToolResult, WireFormat,
};
use switchyard_translation::TranslationEngine;

use crate::translation;

/// A provider client bound to one configured Switchyard target.
///
/// libsy routes with a stable semantic name (for example `fast`). The provider
/// still expects its own model id (for example `meta/llama-3.1-8b-instruct`).
/// Keeping that mapping here prevents an algorithm's semantic labels from
/// leaking into provider requests.
pub(crate) struct TargetClient {
    provider_model: String,
    target_format: WireFormat,
    drop_caller_extra_body: bool,
    inner: TranslatingLlmClient,
    translation: TranslationEngine,
}

impl TargetClient {
    pub(crate) fn new(
        provider_model: String,
        target_format: WireFormat,
        dispatch_url: String,
        headers: BTreeMap<String, String>,
        extra_body: BTreeMap<String, Json>,
        drop_caller_extra_body: bool,
    ) -> Result<Self, LlmClientError> {
        let backend_config = HttpBackendConfig {
            // `dispatch_url` is already resolved by configuration. Backend URL
            // joining accepts a complete canonical endpoint as well as a base
            // URL/prefix.
            base_url: dispatch_url,
            api_key: None,
            extra_headers: headers,
            extra_body,
            // Routing retries belong to the plugin: every retry must start a
            // fresh libsy run and obtain a fresh decision.
            max_retries: 0,
        };
        let backend = match target_format {
            WireFormat::OpenAiChat => Backend::OpenAiChat(backend_config),
            WireFormat::OpenAiResponses => Backend::OpenAiResponses(backend_config),
            WireFormat::AnthropicMessages => Backend::Anthropic(backend_config),
        };
        let model = ModelConfig::new(provider_model.clone(), backend, None);
        let inner = TranslatingLlmClient::new(&[model])?;
        Ok(Self {
            provider_model,
            target_format,
            drop_caller_extra_body,
            inner,
            translation: TranslationEngine::default(),
        })
    }

    /// Retargets only the provider-facing transport metadata.
    ///
    /// Correlation and agent identity remain available to libsy, while inbound
    /// HTTP headers are deliberately removed. Provider credentials come solely
    /// from this target's `header_env` configuration.
    fn prepare_request(&self, mut request: Request, decision: &Decision) -> Request {
        if !decision.is_answer_call() {
            sanitize_judge_request(&mut request);
        }
        if decision.reasoning() == Some("escalation classifier: efficient tier") {
            // Escalation always buffers this draft before judging it. Asking the
            // provider for a buffered response preserves normalized usage for ATOF;
            // libsy reconstructs a caller stream when the weak draft wins.
            request.llm_request.stream = false;
            request.llm_request.preservation.requests.clear();
        }
        let metadata = request.metadata.get_or_insert_default();
        metadata.wire_format = Some(self.target_format);
        metadata.http_headers = None;
        if self.drop_caller_extra_body {
            request.llm_request.extensions.fields.remove("extra_body");
            for preserved in request.llm_request.preservation.requests.values_mut() {
                if let Some(body) = preserved.as_object_mut() {
                    body.remove("extra_body");
                }
            }
        }
        request
    }
}

#[async_trait]
impl RoutedLlmClient for TargetClient {
    async fn call(
        &self,
        ctx: Context,
        request: Request,
        decision: Arc<Decision>,
    ) -> Result<Response, LlmClientError> {
        let request = self.prepare_request(request, decision.as_ref());
        translation::validate_target_request(
            &self.translation,
            self.target_format,
            &request.llm_request,
        )
        .map_err(LlmClientError::RequestEncoding)?;
        self.inner
            .call_rewrite_model(ctx, request, Some(&self.provider_model))
            .await
    }
}

/// Maximum plain-text context retained from one native tool block in a judge request.
const MAX_JUDGE_TOOL_CONTEXT_CHARS: usize = 4_096;

/// Keep judge requests provider-neutral. Native tool turns without their original
/// definitions are rejected by some OpenAI-compatible Bedrock gateways, while the
/// text evidence is still valuable to the classifier.
fn sanitize_judge_request(request: &mut Request) {
    request.llm_request.messages = request
        .llm_request
        .messages
        .drain(..)
        .map(|message| Message {
            role: if message.role == Role::Tool {
                Role::User
            } else {
                message.role
            },
            content: message
                .content
                .into_iter()
                .map(|block| match block {
                    ContentBlock::ToolCall(call) => ContentBlock::Text {
                        text: bounded_tool_context(tool_call_text(call)),
                    },
                    ContentBlock::ToolResult(result) => ContentBlock::Text {
                        text: bounded_tool_context(tool_result_text(result)),
                    },
                    ordinary => ordinary,
                })
                .collect(),
        })
        .collect();
    request.llm_request.tools.clear();
    request.llm_request.tool_choice = None;
    if let Some(response_format) = request.llm_request.output.response_format.as_mut() {
        remove_numeric_schema_bounds(response_format);
    }
    request.llm_request.preservation.requests.clear();
}

fn tool_call_text(call: ToolCall) -> String {
    format!(
        "[tool call]\nid: {}\nname: {}\narguments: {}",
        Json::String(call.id),
        Json::String(call.name),
        call.arguments
    )
}

fn tool_result_text(result: ToolResult) -> String {
    let content = result
        .content
        .into_iter()
        .map(tool_content_text)
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "[tool result]\ncall_id: {}\nis_error: {}\ncontent:\n{}",
        Json::String(result.tool_call_id),
        result
            .is_error
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        content
    )
}

fn tool_content_text(block: ContentBlock) -> String {
    match block {
        ContentBlock::Text { text }
        | ContentBlock::Reasoning { text, .. }
        | ContentBlock::Refusal { text } => text,
        ContentBlock::ToolCall(call) => tool_call_text(call),
        ContentBlock::ToolResult(result) => tool_result_text(result),
        ContentBlock::Image { .. } => "[image omitted]".to_string(),
        ContentBlock::Audio { .. } => "[audio omitted]".to_string(),
        ContentBlock::Video { .. } => "[video omitted]".to_string(),
        ContentBlock::File { .. } => "[file omitted]".to_string(),
        ContentBlock::Unknown { provider, .. } => {
            format!("[unsupported {provider} content omitted]")
        }
    }
}

fn bounded_tool_context(text: String) -> String {
    const TRUNCATED: &str = "\n[truncated]";
    let keep = MAX_JUDGE_TOOL_CONTEXT_CHARS - TRUNCATED.chars().count();
    let mut chars = text.chars();
    let prefix = chars.by_ref().take(keep).collect::<String>();
    if chars.next().is_some() {
        prefix + TRUNCATED
    } else {
        prefix
    }
}

fn remove_numeric_schema_bounds(value: &mut Json) {
    match value {
        Json::Object(object) => {
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
                object.remove(key);
            }
            for child in object.values_mut() {
                remove_numeric_schema_bounds(child);
            }
        }
        Json::Array(values) => {
            for child in values {
                remove_numeric_schema_bounds(child);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{
        LlmRequest, Metadata, PreservationMetadata, ProviderExtensions, ToolChoice, ToolDefinition,
    };

    fn decision() -> Decision {
        Decision::new("target", None, true)
    }

    fn client(format: WireFormat) -> TargetClient {
        TargetClient::new(
            "provider/model".into(),
            format,
            match format {
                WireFormat::OpenAiChat => "https://provider.example/v1/chat/completions".into(),
                WireFormat::OpenAiResponses => "https://provider.example/v1/responses".into(),
                WireFormat::AnthropicMessages => "https://provider.example/v1/messages".into(),
            },
            BTreeMap::new(),
            BTreeMap::new(),
            false,
        )
        .unwrap()
    }

    #[test]
    fn target_preparation_forces_format_and_removes_inbound_headers() {
        let client = client(WireFormat::AnthropicMessages);
        let request = Request {
            metadata: Some(Metadata {
                correlation_id: Some("request-123".into()),
                wire_format: Some(WireFormat::OpenAiChat),
                http_headers: Some(http::HeaderMap::from_iter([
                    (
                        http::HeaderName::from_static("authorization"),
                        http::HeaderValue::from_static("Bearer caller-secret"),
                    ),
                    (
                        http::HeaderName::from_static("x-caller-only"),
                        http::HeaderValue::from_static("must-not-forward"),
                    ),
                ])),
                ..Metadata::default()
            }),
            ..Request::default()
        };

        let prepared = client.prepare_request(request, &decision());
        let metadata = prepared.metadata.unwrap();
        assert_eq!(metadata.wire_format, Some(WireFormat::AnthropicMessages));
        assert_eq!(metadata.correlation_id.as_deref(), Some("request-123"));
        assert!(metadata.http_headers.is_none());
    }

    #[test]
    fn missing_metadata_is_created_for_the_target_format() {
        let client = client(WireFormat::OpenAiResponses);
        let prepared = client.prepare_request(Request::default(), &decision());
        assert_eq!(
            prepared.metadata.and_then(|metadata| metadata.wire_format),
            Some(WireFormat::OpenAiResponses)
        );
    }

    #[test]
    fn configured_target_drops_intercepted_caller_extra_body() {
        let client = TargetClient::new(
            "provider/model".into(),
            WireFormat::OpenAiChat,
            "https://provider.example/v1/chat/completions".into(),
            BTreeMap::new(),
            BTreeMap::new(),
            true,
        )
        .unwrap();
        let request = Request {
            llm_request: LlmRequest {
                extensions: ProviderExtensions {
                    fields: serde_json::Map::from_iter([(
                        "extra_body".into(),
                        json!({"reasoning": {"effort": "medium"}}),
                    )]),
                },
                preservation: PreservationMetadata {
                    requests: BTreeMap::from([(
                        WireFormat::OpenAiChat.into(),
                        json!({
                            "model": "route",
                            "messages": [{"role": "user", "content": "hello"}],
                            "extra_body": {
                                "reasoning": {"effort": "medium"},
                                "session_id": "hermes-session"
                            }
                        }),
                    )]),
                    ..PreservationMetadata::default()
                },
                ..LlmRequest::default()
            },
            ..Request::default()
        };

        let prepared = client.prepare_request(request, &decision());
        assert!(
            !prepared
                .llm_request
                .extensions
                .fields
                .contains_key("extra_body")
        );
        assert!(
            prepared
                .llm_request
                .preservation
                .requests
                .values()
                .all(|body| body.get("extra_body").is_none())
        );
    }

    #[test]
    fn judge_preparation_sanitizes_tool_history_and_schema_dialect() {
        let client = client(WireFormat::OpenAiChat);
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message::text(Role::User, "inspect the workspace"),
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".into(),
                            name: "terminal".into(),
                            arguments: json!({"command": "pwd"}),
                        })],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "call-1".into(),
                            content: vec![ContentBlock::Text {
                                text: format!("result {} TAIL", "x".repeat(5_000)),
                            }],
                            is_error: Some(false),
                        })],
                    },
                ],
                tools: vec![ToolDefinition {
                    name: "terminal".into(),
                    description: None,
                    parameters: json!({"type": "object"}),
                    strict: None,
                }],
                tool_choice: Some(ToolChoice::Required),
                output: switchyard_protocol::OutputParams {
                    max_output_tokens: Some(64),
                    response_format: Some(json!({
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {
                                "properties": {
                                    "p_solve": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0
                                    }
                                }
                            }
                        }
                    })),
                },
                ..LlmRequest::default()
            },
            ..Request::default()
        };

        let prepared = client.prepare_request(
            request,
            &Decision::new("judge", Some("structured judge".into()), false),
        );

        assert!(prepared.llm_request.tools.is_empty());
        assert_eq!(prepared.llm_request.tool_choice, None);
        assert!(
            prepared
                .llm_request
                .messages
                .iter()
                .all(|message| message.role != Role::Tool)
        );
        let text = prepared
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("[tool call]"));
        assert!(text.contains("terminal"));
        assert!(text.contains("[tool result]"));
        assert!(text.contains("[truncated]"));
        assert!(!text.contains("TAIL"));
        let schema = prepared.llm_request.output.response_format.unwrap();
        let p_solve = schema
            .pointer("/json_schema/schema/properties/p_solve")
            .unwrap();
        assert!(p_solve.get("minimum").is_none());
        assert!(p_solve.get("maximum").is_none());
    }

    #[test]
    fn escalation_candidate_is_buffered_for_usage_accounting() {
        let client = client(WireFormat::OpenAiChat);
        let mut request = Request::default();
        request.llm_request.stream = true;
        request.llm_request.preservation.requests.insert(
            WireFormat::OpenAiChat.into(),
            json!({"model": "route", "stream": true}),
        );
        let decision = Decision::new(
            "weak",
            Some("escalation classifier: efficient tier".into()),
            true,
        );

        let prepared = client.prepare_request(request, &decision);

        assert!(!prepared.llm_request.stream);
        assert!(prepared.llm_request.preservation.requests.is_empty());
    }
}
