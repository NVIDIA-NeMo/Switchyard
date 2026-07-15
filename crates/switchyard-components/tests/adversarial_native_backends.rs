// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial integration tests for native LLM backends.

mod support;

use futures_util::StreamExt;
use serde_json::{json, Value};
use switchyard_components::{
    AnthropicNativeBackend, BackendSelection, OpenAiNativeBackend, OpenAiPassthroughBackend,
};
use switchyard_core::{
    BackendFormat, ChatRequest, ChatRequestType, ChatResponse, ChatResponseType, EndpointConfig,
    LlmBackend, LlmTarget, LlmTargetId, ModelId, ProxyContext, Result, StreamEvent,
    SwitchyardError,
};

use support::{CapturedRequest, OneShotServer};

// Reads the selected model stamped by the backend into context.
fn selected_model(ctx: &ProxyContext) -> Option<&ModelId> {
    ctx.get::<BackendSelection>()
        .map(|selection| &selection.model)
}

// Reads the selected target stamped by native backends into context.
fn selected_target(ctx: &ProxyContext) -> Option<&LlmTargetId> {
    ctx.get::<BackendSelection>()
        .and_then(|selection| selection.target_id.as_ref())
}

// Mirrors runtime telemetry header resolution for assertions.
fn expected_switchyard_version_header() -> Option<String> {
    if env_value_opts_out(
        std::env::var("SWITCHYARD_TELEMETRY_OPT_OUT")
            .ok()
            .as_deref(),
    ) || env_value_opts_out(
        std::env::var("NEMO_SWITCHYARD_TELEMETRY_OPT_OUT")
            .ok()
            .as_deref(),
    ) {
        return None;
    }
    Some(
        std::env::var("SWITCHYARD_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()),
    )
}

// Parses telemetry opt-out environment values like the production helper.
fn env_value_opts_out(value: Option<&str>) -> bool {
    let Some(value) = value.map(str::trim) else {
        return false;
    };
    !matches!(
        value.to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no"
    )
}

// Builds a target with explicit endpoint credentials for native backend tests.
fn target(
    format: BackendFormat,
    base_url: String,
    api_key: &str,
    model: &str,
) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(
        LlmTargetId::from_static("primary"),
        ModelId::new(model).map_err(|error| SwitchyardError::Other(error.to_string()))?,
    );
    target.format = format;
    target.endpoint = EndpointConfig {
        base_url: Some(base_url),
        api_key: Some(api_key.to_string()),
        timeout_secs: None,
    };
    Ok(target)
}

// Builds an OpenAI target fixture.
fn openai_target(base_url: String) -> Result<LlmTarget> {
    target(
        BackendFormat::OpenAi,
        base_url,
        "openai-secret",
        "target-gpt",
    )
}

// Builds an OpenAI Responses target fixture.
fn responses_target(base_url: String) -> Result<LlmTarget> {
    target(
        BackendFormat::Responses,
        base_url,
        "openai-secret",
        "target-responses",
    )
}

// Builds an Anthropic target fixture.
fn anthropic_target(base_url: String) -> Result<LlmTarget> {
    target(
        BackendFormat::Anthropic,
        base_url,
        "anthropic-secret",
        "target-claude",
    )
}

// Builds a passthrough OpenAI endpoint fixture.
fn openai_endpoint(base_url: String) -> EndpointConfig {
    EndpointConfig {
        base_url: Some(base_url),
        api_key: Some("openai-secret".to_string()),
        timeout_secs: None,
    }
}

// Calls the OpenAI backend against a one-shot mock server.
async fn openai_call(body: Value) -> Result<(ChatResponse, CapturedRequest, ProxyContext)> {
    let server = OneShotServer::json(
        200,
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": []
        }),
    )?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();
    let response = backend
        .call(&mut ctx, &ChatRequest::openai_chat(body))
        .await?;
    Ok((response, server.captured()?, ctx))
}

// Drains JSON stream events and rejects unexpected text frames.
async fn collect_json_events(response: ChatResponse) -> Result<Vec<Value>> {
    let mut stream = match response {
        ChatResponse::OpenAiStream(stream)
        | ChatResponse::OpenAiResponsesStream(stream)
        | ChatResponse::AnthropicStream(stream) => stream,
        other => {
            return Err(SwitchyardError::Other(format!(
                "expected streaming response, got {:?}",
                other.response_type()
            )));
        }
    };
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Json(value) => events.push(value),
            StreamEvent::Text(text) => {
                return Err(SwitchyardError::Other(format!(
                    "unexpected text stream event: {text}"
                )));
            }
        }
    }
    Ok(events)
}

// OpenAI native backends should only advertise OpenAI Chat input.
#[test]
fn openai_backend_is_openai_chat_only() -> Result<()> {
    let backend = OpenAiNativeBackend::new(openai_target("http://127.0.0.1:1/v1".to_string())?)?;

    assert_eq!(
        backend.supported_request_types(),
        &[ChatRequestType::OpenAiChat]
    );
    Ok(())
}

// Responses-format OpenAI targets should advertise OpenAI Responses input.
#[test]
fn openai_backend_can_be_responses_only() -> Result<()> {
    let backend = OpenAiNativeBackend::new(responses_target("http://127.0.0.1:1/v1".to_string())?)?;

    assert_eq!(
        backend.supported_request_types(),
        &[ChatRequestType::OpenAiResponses]
    );
    Ok(())
}

// OpenAI passthrough keeps the same OpenAI Chat-only role contract.
#[test]
fn openai_passthrough_backend_is_openai_chat_only() -> Result<()> {
    let backend =
        OpenAiPassthroughBackend::new(openai_endpoint("http://127.0.0.1:1/v1".to_string()))?;

    assert_eq!(
        backend.supported_request_types(),
        &[ChatRequestType::OpenAiChat]
    );
    Ok(())
}

// Non-streaming OpenAI calls should preserve body fields and stamp context.
#[tokio::test]
async fn openai_non_streaming_posts_configured_body_and_records_context() -> Result<()> {
    let (response, request, ctx) = openai_call(json!({
        "model": "client-gpt",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "made_up_beta_field": {"kept": true},
        "stream": false
    }))
    .await?;

    assert_eq!(response.response_type(), ChatResponseType::OpenAiCompletion);
    assert_eq!(
        response
            .body()
            .ok_or_else(|| SwitchyardError::Other("buffered response".to_string()))?["id"],
        "chatcmpl-test"
    );
    assert_eq!(ctx.inbound_format, Some(ChatRequestType::OpenAiChat));
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("target-gpt"))
    );
    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("primary"))
    );

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(
        request.header("authorization"),
        Some("Bearer openai-secret")
    );
    let expected_version = expected_switchyard_version_header();
    assert_eq!(
        request.header("x-switchyard-version"),
        expected_version.as_deref()
    );
    assert_eq!(request.body["model"], "target-gpt");
    assert_eq!(request.body["messages"][0]["content"], "hello");
    assert_eq!(request.body["temperature"], 0.2);
    assert_eq!(request.body["made_up_beta_field"], json!({"kept": true}));
    assert!(request.body.get("stream_options").is_none());
    Ok(())
}

// Responses-format OpenAI targets should call /v1/responses without translating through Chat.
#[tokio::test]
async fn openai_responses_target_posts_responses_body_and_records_context() -> Result<()> {
    let server = OneShotServer::json(
        200,
        json!({
            "id": "resp-test",
            "object": "response",
            "output": []
        }),
    )?;
    let backend = OpenAiNativeBackend::new(responses_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_responses(json!({
                "model": "client-gpt",
                "input": "hello",
                "stream": false
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(
        response.response_type(),
        ChatResponseType::OpenAiResponsesCompletion
    );
    assert_eq!(ctx.inbound_format, Some(ChatRequestType::OpenAiResponses));
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("target-responses"))
    );
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(request.body["model"], "target-responses");
    assert_eq!(request.body["input"], "hello");
    assert!(request.body.get("messages").is_none());
    assert!(request.body.get("stream_options").is_none());
    Ok(())
}

// Endpoint-specific base URLs should normalize to the endpoint selected by target format.
#[tokio::test]
async fn openai_specific_base_url_uses_selected_endpoint() -> Result<()> {
    let chat_server = OneShotServer::json(200, json!({"id": "chatcmpl-test", "choices": []}))?;
    let chat_backend = OpenAiNativeBackend::new(openai_target(format!(
        "{}/v1/responses",
        chat_server.base_url()
    ))?)?;
    let mut chat_ctx = ProxyContext::new();

    chat_backend
        .call(
            &mut chat_ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "hello"}]
            })),
        )
        .await?;
    assert_eq!(chat_server.captured()?.path, "/v1/chat/completions");

    let responses_server = OneShotServer::json(200, json!({"id": "resp-test", "output": []}))?;
    let responses_backend = OpenAiNativeBackend::new(responses_target(format!(
        "{}/v1/chat/completions",
        responses_server.base_url()
    ))?)?;
    let mut responses_ctx = ProxyContext::new();

    responses_backend
        .call(
            &mut responses_ctx,
            &ChatRequest::openai_responses(json!({
                "model": "client-gpt",
                "input": "hello"
            })),
        )
        .await?;
    assert_eq!(responses_server.captured()?.path, "/v1/responses");
    Ok(())
}

// Chat-format OpenAI targets should keep translating Responses requests to Chat fallback.
#[tokio::test]
async fn openai_chat_target_translates_responses_to_chat_fallback() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "chatcmpl-test", "choices": []}))?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_responses(json!({
                "model": "client-gpt",
                "input": "translate me"
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(response.response_type(), ChatResponseType::OpenAiCompletion);
    assert_eq!(ctx.inbound_format, Some(ChatRequestType::OpenAiResponses));
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.body["model"], "target-gpt");
    assert_eq!(request.body["messages"][0]["content"], "translate me");
    Ok(())
}

// Native OpenAI targets should apply per-target body and header overrides.
#[tokio::test]
async fn openai_native_applies_target_extra_body_and_headers() -> Result<()> {
    let server = OneShotServer::json(
        200,
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": []
        }),
    )?;
    let mut target = openai_target(format!("{}/v1", server.base_url()))?;
    target.extra_body = Some(json!({
        "chat_template_kwargs": {"enable_thinking": false}
    }));
    target
        .extra_headers
        .insert("X-Inference-Priority".to_string(), "batch".to_string());
    let backend = OpenAiNativeBackend::new(target)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(
        request.body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert_eq!(request.header("x-inference-priority"), Some("batch"));
    Ok(())
}

// Passthrough OpenAI calls should not rewrite caller model names.
#[tokio::test]
async fn openai_passthrough_preserves_client_model_and_records_context() -> Result<()> {
    let server = OneShotServer::json(
        200,
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": []
        }),
    )?;
    let backend =
        OpenAiPassthroughBackend::new(openai_endpoint(format!("{}/v1", server.base_url())))?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(response.response_type(), ChatResponseType::OpenAiCompletion);
    assert_eq!(ctx.inbound_format, Some(ChatRequestType::OpenAiChat));
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("client-gpt"))
    );
    assert!(selected_target(&ctx).is_none());
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(
        request.header("authorization"),
        Some("Bearer openai-secret")
    );
    assert_eq!(request.body["model"], "client-gpt");
    Ok(())
}

// OpenAI streams should request usage and stop cleanly on `[DONE]`.
#[tokio::test]
async fn openai_streaming_injects_usage_opt_in_and_parses_done() -> Result<()> {
    let server = OneShotServer::sse(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
         data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n\
         data: [DONE]\n\n",
    )?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "stream"}],
                "stream": true
            })),
        )
        .await?;
    assert_eq!(response.response_type(), ChatResponseType::OpenAiStream);
    let events = collect_json_events(response).await?;
    let request = server.captured()?;

    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "hi");
    assert_eq!(events[1]["usage"]["total_tokens"], 3);
    assert_eq!(
        request.body["stream_options"],
        json!({"include_usage": true})
    );
    Ok(())
}

// Existing stream options should win over backend usage defaults.
#[tokio::test]
async fn openai_streaming_respects_usage_opt_out_and_preserves_other_options() -> Result<()> {
    let server = OneShotServer::sse("data: [DONE]\n\n")?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "stream"}],
                "stream": true,
                "stream_options": {
                    "include_usage": false,
                    "continuous_usage_stats": true
                }
            })),
        )
        .await?;
    let events = collect_json_events(response).await?;
    let request = server.captured()?;

    assert!(events.is_empty());
    assert_eq!(
        request.body["stream_options"],
        json!({
            "include_usage": false,
            "continuous_usage_stats": true
        })
    );
    Ok(())
}

// OpenAI native should translate Anthropic requests before upstream dispatch.
#[tokio::test]
async fn openai_translates_anthropic_requests_before_native_call() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "chatcmpl-test", "choices": []}))?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "translate me"}]
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(ctx.inbound_format, Some(ChatRequestType::Anthropic));
    assert_eq!(request.body["model"], "target-gpt");
    assert_eq!(request.body["messages"][0]["role"], "user");
    assert_eq!(request.body["messages"][0]["content"], "translate me");
    Ok(())
}

// OpenAI passthrough should translate Anthropic shape without model rewriting.
#[tokio::test]
async fn openai_passthrough_translates_anthropic_without_rewriting_model() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "chatcmpl-test", "choices": []}))?;
    let backend =
        OpenAiPassthroughBackend::new(openai_endpoint(format!("{}/v1", server.base_url())))?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "translate me"}]
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(ctx.inbound_format, Some(ChatRequestType::Anthropic));
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("client-claude"))
    );
    assert_eq!(request.body["model"], "client-claude");
    assert_eq!(request.body["messages"][0]["content"], "translate me");
    Ok(())
}

// Responses streams should call /responses and return Responses stream variants.
#[tokio::test]
async fn openai_responses_streaming_uses_responses_stream_variant() -> Result<()> {
    let server = OneShotServer::sse(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-stream\"}}\n\n",
    )?;
    let backend = OpenAiNativeBackend::new(responses_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_responses(json!({
                "model": "client-gpt",
                "input": "stream",
                "stream": true
            })),
        )
        .await?;
    assert_eq!(
        response.response_type(),
        ChatResponseType::OpenAiResponsesStream
    );
    let events = collect_json_events(response).await?;
    let request = server.captured()?;

    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(request.path, "/v1/responses");
    assert!(request.body.get("stream_options").is_none());
    Ok(())
}

// OpenAI native must reject targets configured for Anthropic format.
#[test]
fn openai_backend_rejects_anthropic_targets() -> Result<()> {
    let Err(error) = OpenAiNativeBackend::new(target(
        BackendFormat::Anthropic,
        "http://127.0.0.1:1".to_string(),
        "secret",
        "claude",
    )?) else {
        return Err(SwitchyardError::Other(
            "OpenAI backend should reject Anthropic targets".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    Ok(())
}

// OpenAI upstream error responses should preserve status and body in the error.
#[tokio::test]
async fn openai_error_status_includes_status_and_body() -> Result<()> {
    let server = OneShotServer::json(
        429,
        json!({"error": {"message": "rate limited", "type": "rate_limit"}}),
    )?;
    let backend = OpenAiNativeBackend::new(openai_target(format!("{}/v1", server.base_url()))?)?;
    let mut ctx = ProxyContext::new();

    let Err(error) = backend
        .call(
            &mut ctx,
            &ChatRequest::openai_chat(json!({
                "model": "client-gpt",
                "messages": [{"role": "user", "content": "hello"}]
            })),
        )
        .await
    else {
        return Err(SwitchyardError::Other(
            "OpenAI backend should propagate HTTP errors".to_string(),
        ));
    };
    let captured = server.captured()?;

    assert!(matches!(error, SwitchyardError::UpstreamHttp { .. }));
    assert!(error.to_string().contains("HTTP 429"));
    assert!(error.to_string().contains("rate limited"));
    assert_eq!(captured.path, "/v1/chat/completions");
    Ok(())
}

// Anthropic native backends should only advertise Anthropic input.
#[test]
fn anthropic_backend_is_anthropic_only() -> Result<()> {
    let backend = AnthropicNativeBackend::new(anthropic_target("http://127.0.0.1:1".to_string())?)?;

    assert_eq!(
        backend.supported_request_types(),
        &[ChatRequestType::Anthropic]
    );
    Ok(())
}

// Non-streaming native Anthropic calls forward the body verbatim and stamp context.
#[tokio::test]
async fn anthropic_native_passthrough_forwards_body_verbatim() -> Result<()> {
    let server = OneShotServer::json(
        200,
        json!({
            "id": "msg-test",
            "type": "message",
            "content": []
        }),
    )?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "high",
                "context_management": {"strategy": "auto"},
                "made_up_beta_field": {"kept": true},
                "extra_body": {"caller": "value"},
                "stream": false
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(
        response.response_type(),
        ChatResponseType::AnthropicCompletion
    );
    assert_eq!(
        response
            .body()
            .ok_or_else(|| SwitchyardError::Other("buffered response".to_string()))?["id"],
        "msg-test"
    );
    assert_eq!(ctx.inbound_format, Some(ChatRequestType::Anthropic));
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("target-claude"))
    );
    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("primary"))
    );

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(request.header("x-api-key"), Some("anthropic-secret"));
    assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
    let expected_version = expected_switchyard_version_header();
    assert_eq!(
        request.header("x-switchyard-version"),
        expected_version.as_deref()
    );
    assert_eq!(request.body["model"], "target-claude");
    assert_eq!(request.body["messages"][0]["content"], "hello");
    // Native Anthropic is a verbatim passthrough: client fields are preserved,
    // not stripped (only the model id is rewritten for routing). Stripping these
    // belongs to the translated path, not a real /v1/messages request.
    assert_eq!(request.body["reasoning_effort"], "high");
    assert_eq!(
        request.body["context_management"],
        json!({"strategy": "auto"})
    );
    assert_eq!(request.body["made_up_beta_field"], json!({"kept": true}));
    assert_eq!(request.body["extra_body"], json!({"caller": "value"}));
    Ok(())
}

// Native Anthropic passthrough must NOT relocate message-level system/developer turns.
#[tokio::test]
async fn anthropic_native_passthrough_does_not_lift_system_messages() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [
                    {"role": "system", "content": "System rules."},
                    {"role": "user", "content": "hello"},
                    {
                        "role": "developer",
                        "content": [
                            {"type": "text", "text": "Developer rules."}
                        ]
                    },
                    {"role": "assistant", "content": "ready"}
                ]
            })),
        )
        .await?;
    let request = server.captured()?;

    // Verbatim: no top-level system is synthesized, and the system/developer
    // turns stay exactly where the client put them.
    assert!(request.body.get("system").is_none());
    assert_eq!(
        request.body["messages"],
        json!([
            {"role": "system", "content": "System rules."},
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": [{"type": "text", "text": "Developer rules."}]},
            {"role": "assistant", "content": "ready"}
        ])
    );
    Ok(())
}

// Interleaved system/developer turns are preserved in place (no lifting) on passthrough.
#[tokio::test]
async fn anthropic_native_passthrough_preserves_interleaved_system_messages() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "system": "Top-level rules.",
                "messages": [
                    {"role": "system", "content": "First lifted system."},
                    {"role": "user", "content": "first user"},
                    {"role": "system", "content": "Second lifted system."},
                    {"role": "assistant", "content": "assistant reply"},
                    {"role": "developer", "content": "Developer lifted system."},
                    {"role": "user", "content": "second user"}
                ]
            })),
        )
        .await?;
    let request = server.captured()?;

    // Verbatim: top-level system unchanged, all message-level turns preserved in place.
    assert_eq!(request.body["system"], "Top-level rules.");
    assert_eq!(
        request.body["messages"],
        json!([
            {"role": "system", "content": "First lifted system."},
            {"role": "user", "content": "first user"},
            {"role": "system", "content": "Second lifted system."},
            {"role": "assistant", "content": "assistant reply"},
            {"role": "developer", "content": "Developer lifted system."},
            {"role": "user", "content": "second user"}
        ])
    );
    Ok(())
}

// Structured system prompt and message-level system (incl. non-text blocks) pass through untouched.
#[tokio::test]
async fn anthropic_native_passthrough_preserves_structured_system_and_messages() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "system": [{"type": "text", "text": "Existing system."}],
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "Lifted system."},
                            {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
                            {"type": "input_text", "text": "Lifted input text."}
                        ]
                    },
                    {"role": "user", "content": "hello"}
                ]
            })),
        )
        .await?;
    let request = server.captured()?;

    // Verbatim: structured system kept as-is; the message-level system turn
    // (including its image block) is preserved, not downgraded into system.
    assert_eq!(
        request.body["system"],
        json!([{"type": "text", "text": "Existing system."}])
    );
    assert_eq!(
        request.body["messages"],
        json!([
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Lifted system."},
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
                    {"type": "input_text", "text": "Lifted input text."}
                ]
            },
            {"role": "user", "content": "hello"}
        ])
    );
    Ok(())
}

// Responses requests should translate into Anthropic Messages with default max_tokens.
#[tokio::test]
async fn anthropic_translates_responses_requests_with_default_max_tokens() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::openai_responses(json!({
                "model": "client-gpt",
                "input": "translate me"
            })),
        )
        .await?;
    let request = server.captured()?;

    assert_eq!(ctx.inbound_format, Some(ChatRequestType::OpenAiResponses));
    assert_eq!(request.body["model"], "target-claude");
    assert_eq!(request.body["max_tokens"], 128000);
    assert_eq!(request.body["messages"][0]["role"], "user");
    assert_eq!(request.body["messages"][0]["content"], "translate me");
    Ok(())
}

// Native Anthropic passthrough preserves client tool-use IDs verbatim (no sanitization).
#[tokio::test]
async fn anthropic_native_passthrough_preserves_tool_use_ids() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [
                    {"role": "user", "content": "use the tool"},
                    {
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": "toolu_01*bad:id",
                            "name": "lookup",
                            "input": {}
                        }]
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": "toolu_01*bad:id",
                            "content": "done"
                        }]
                    }
                ]
            })),
        )
        .await?;
    let request = server.captured()?;

    // Verbatim: the client's tool-use id is forwarded unchanged (sanitization
    // belongs to the translated path; a real Anthropic client sends valid ids).
    let tool_use_id = &request.body["messages"][1]["content"][0]["id"];
    assert_eq!(tool_use_id, "toolu_01*bad:id");
    assert_eq!(
        &request.body["messages"][2]["content"][0]["tool_use_id"],
        tool_use_id
    );
    Ok(())
}

// Native Anthropic passthrough preserves thinking blocks verbatim (no stripping).
#[tokio::test]
async fn anthropic_native_passthrough_preserves_thinking_blocks() -> Result<()> {
    let server = OneShotServer::json(200, json!({"id": "msg-test", "content": []}))?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "synthetic", "signature": ""},
                            {"type": "tool_use", "id": "toolu_ok", "name": "lookup", "input": {}}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "real", "signature": "signed"},
                            {"type": "text", "text": "visible"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "only synthetic"}
                        ]
                    }
                ]
            })),
        )
        .await?;
    let request = server.captured()?;

    // Verbatim: thinking blocks (signed or not) are preserved; the upstream API
    // decides what to accept. Stripping unsigned blocks belongs to the translated
    // path, where they are synthetic translation artifacts.
    assert_eq!(
        request.body["messages"][0]["content"]
            .as_array()
            .ok_or_else(|| SwitchyardError::Other("content should be an array".to_string()))?
            .len(),
        2
    );
    assert_eq!(
        request.body["messages"][0]["content"][0]["type"],
        "thinking"
    );
    assert_eq!(
        request.body["messages"][0]["content"][1]["type"],
        "tool_use"
    );
    assert_eq!(
        request.body["messages"][1]["content"][0]["type"],
        "thinking"
    );
    assert_eq!(
        request.body["messages"][1]["content"][0]["thinking"],
        "real"
    );
    assert_eq!(
        request.body["messages"][2]["content"],
        json!([{"type": "thinking", "thinking": "only synthetic"}])
    );
    Ok(())
}

// Anthropic SSE should produce JSON stream events.
#[tokio::test]
async fn anthropic_streaming_returns_stream_events() -> Result<()> {
    let server = OneShotServer::sse(
        "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-stream\"}}\n\n",
    )?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    let response = backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "stream"}],
                "stream": true
            })),
        )
        .await?;
    assert_eq!(response.response_type(), ChatResponseType::AnthropicStream);
    let events = collect_json_events(response).await?;
    let request = server.captured()?;

    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "message_start");
    assert_eq!(events[0]["message"]["id"], "msg-stream");
    assert_eq!(request.path, "/v1/messages");
    Ok(())
}

// Anthropic native must reject targets configured for OpenAI format.
#[test]
fn anthropic_backend_rejects_openai_targets() -> Result<()> {
    let Err(error) = AnthropicNativeBackend::new(target(
        BackendFormat::OpenAi,
        "http://127.0.0.1:1".to_string(),
        "secret",
        "gpt",
    )?) else {
        return Err(SwitchyardError::Other(
            "Anthropic backend should reject OpenAI targets".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    Ok(())
}

// Anthropic upstream error responses should preserve status and body in the error.
#[tokio::test]
async fn anthropic_error_status_includes_status_and_body() -> Result<()> {
    let server = OneShotServer::json(
        400,
        json!({"error": {"message": "invalid request", "type": "bad_request"}}),
    )?;
    let backend = AnthropicNativeBackend::new(anthropic_target(server.base_url().to_string())?)?;
    let mut ctx = ProxyContext::new();

    let Err(error) = backend
        .call(
            &mut ctx,
            &ChatRequest::anthropic(json!({
                "model": "client-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}]
            })),
        )
        .await
    else {
        return Err(SwitchyardError::Other(
            "Anthropic backend should propagate HTTP errors".to_string(),
        ));
    };
    let captured = server.captured()?;

    assert!(matches!(error, SwitchyardError::UpstreamHttp { .. }));
    assert!(error.to_string().contains("HTTP 400"));
    assert!(error.to_string().contains("invalid request"));
    assert_eq!(captured.path, "/v1/messages");
    Ok(())
}

// Native backends should reject unresolved Auto formats until config resolves them.
#[test]
fn native_backends_reject_unresolved_auto_targets() -> Result<()> {
    let auto_openai = target(
        BackendFormat::Auto,
        "http://127.0.0.1:1".to_string(),
        "secret",
        "gpt",
    )?;
    let Err(openai_error) = OpenAiNativeBackend::new(auto_openai) else {
        return Err(SwitchyardError::Other(
            "OpenAI backend should reject Auto targets".to_string(),
        ));
    };
    assert!(matches!(openai_error, SwitchyardError::InvalidConfig(_)));

    let auto_anthropic = target(
        BackendFormat::Auto,
        "http://127.0.0.1:1".to_string(),
        "secret",
        "claude",
    )?;
    let Err(anthropic_error) = AnthropicNativeBackend::new(auto_anthropic) else {
        return Err(SwitchyardError::Other(
            "Anthropic backend should reject Auto targets".to_string(),
        ));
    };
    assert!(matches!(anthropic_error, SwitchyardError::InvalidConfig(_)));
    Ok(())
}
