// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial tests for compatibility identifiers, context storage, and wire wrappers.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::json;
use switchyard_components::{
    BackendFormat, ChatRequest, ChatRequestType, ChatResponse, ChatResponseType, LlmTarget,
    LlmTargetId, ModelId, ProxyContext, StreamEvent,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Eq, PartialEq)]
struct ContextMarker(&'static str);

// Empty stream fixture used to prove streams do not expose buffered JSON bodies.
struct EmptyStream;

impl Stream for EmptyStream {
    type Item = switchyard_components::Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

// Verifies every ID constructor and serde path rejects empty identifiers.
#[test]
fn ids_reject_empty_and_whitespace_values_from_constructors_and_serde() -> TestResult {
    assert!(ModelId::new("").is_err());
    assert!(ModelId::new("   ").is_err());
    assert!(serde_json::from_value::<ModelId>(json!("")).is_err());
    assert!(serde_json::from_value::<ModelId>(json!("   ")).is_err());

    let parsed = serde_json::from_value::<ModelId>(json!("real-model"))?;
    assert_eq!(parsed.as_str(), "real-model");
    Ok(())
}

// Verifies typed context extensions are keyed by type, not string names.
#[test]
fn typed_context_extensions_do_not_collide_or_require_string_keys() -> TestResult {
    let mut ctx = ProxyContext::new();

    assert!(ctx.insert(ContextMarker("first")).is_none());
    assert_eq!(ctx.insert(String::from("unrelated")), None);
    assert_eq!(
        ctx.insert(ContextMarker("second")),
        Some(ContextMarker("first"))
    );

    match ctx.get::<ContextMarker>() {
        Some(marker) => assert_eq!(marker, &ContextMarker("second")),
        None => panic!("marker should be present"),
    }
    assert_eq!(ctx.get::<String>().map(String::as_str), Some("unrelated"));
    assert_eq!(ctx.remove::<ContextMarker>(), Some(ContextMarker("second")));
    assert!(ctx.get::<ContextMarker>().is_none());
    assert_eq!(ctx.get::<String>().map(String::as_str), Some("unrelated"));
    Ok(())
}

// Verifies model rewriting can recover malformed request bodies.
#[test]
fn set_model_recovers_from_malformed_non_object_request_bodies() {
    let mut request = ChatRequest::openai_chat(json!("not-an-object"));

    assert_eq!(request.model(), None);
    request.set_model("recovered-model");

    assert_eq!(request.request_type(), ChatRequestType::OpenAiChat);
    assert_eq!(request.model(), Some("recovered-model"));
    assert_eq!(request.body(), &json!({"model": "recovered-model"}));
}

// Verifies serialized wire values stay compatible with the Python side.
#[test]
fn request_and_response_wire_enum_serialization_stays_stable() -> TestResult {
    assert_eq!(
        serde_json::to_value(ChatRequestType::OpenAiResponses)?,
        json!("openai_responses")
    );
    assert_eq!(
        serde_json::to_value(ChatResponseType::AnthropicStream)?,
        json!("anthropic_stream")
    );

    let request = ChatRequest::anthropic(json!({
        "model": "claude",
        "messages": [],
    }));
    assert_eq!(
        serde_json::to_value(request)?,
        json!({
            "request_type": "anthropic",
            "request": {
                "body": {
                    "model": "claude",
                    "messages": [],
                },
            },
        })
    );
    Ok(())
}

// Verifies OpenAI backend format remains `openai`, not Rust enum snake_case.
#[test]
fn backend_format_stays_wire_compatible_with_python_configs() -> TestResult {
    assert_eq!(
        serde_json::to_value(BackendFormat::OpenAi)?,
        json!("openai")
    );
    assert_eq!(
        serde_json::to_value(BackendFormat::Responses)?,
        json!("responses")
    );
    assert_eq!(
        serde_json::from_value::<BackendFormat>(json!("openai"))?,
        BackendFormat::OpenAi
    );
    assert_eq!(
        serde_json::from_value::<BackendFormat>(json!("responses"))?,
        BackendFormat::Responses
    );
    assert!(serde_json::from_value::<BackendFormat>(json!("open_ai")).is_err());
    Ok(())
}

// Verifies Rust LLM targets reject stale provider tuning fields instead of dropping typos.
#[test]
fn llm_target_rejects_provider_tuning_fields() -> TestResult {
    assert!(serde_json::from_value::<LlmTarget>(json!({
        "id": "primary",
        "model": "gpt-5",
        "format": "openai",
        "endpoint": {
            "base_url": "https://example.test/v1",
            "api_key": null,
            "timeout_secs": 30
        },
        "tuning": {
            "max_output_tokens": 4096,
            "reasoning_effort": "xhigh"
        }
    }))
    .is_err());

    let target = serde_json::from_value::<LlmTarget>(json!({
        "id": "primary",
        "model": "gpt-5",
        "format": "openai",
        "endpoint": {
            "base_url": "https://example.test/v1",
            "api_key": null,
            "timeout_secs": 30
        }
    }))?;
    assert_eq!(target.id, LlmTargetId::from_static("primary"));
    assert_eq!(target.model, ModelId::from_static("gpt-5"));
    assert_eq!(target.format, BackendFormat::OpenAi);
    let serialized = serde_json::to_value(target)?;
    assert_eq!(
        serialized["endpoint"]["base_url"],
        "https://example.test/v1"
    );
    assert!(serialized.get("tuning").is_none());
    Ok(())
}

// Verifies streaming responses are distinguishable from buffered JSON responses.
#[test]
fn streaming_responses_are_not_mistaken_for_buffered_json_bodies() {
    let response = ChatResponse::OpenAiStream(Box::pin(EmptyStream));

    assert_eq!(response.response_type(), ChatResponseType::OpenAiStream);
    assert!(response.body().is_none());
    assert!(format!("{response:?}").contains("<stream>"));
}
