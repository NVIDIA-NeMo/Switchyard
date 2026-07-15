// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake payload tests for chat-completions ingest payload shape and metadata.

mod support;

use serde_json::{json, Value};
use switchyard_components::{
    IntakePayloadBuilder, IntakeRequestMetadata, IntakeRequestState, IntakeSinkConfig,
    RandomRoutingDecision, RandomRoutingTier, RequestMetadata, StatsRouteLabel, SubModelCall,
};
use switchyard_core::{ChatRequest, ChatRequestType, LlmTargetId, ModelId, ProxyContext, Result};

use support::intake::{
    completion, completion_with_usage, openai_chat_request, record_backend_selection,
};

// Test helper mirrors runtime telemetry version resolution.
fn expected_switchyard_version() -> String {
    std::env::var("SWITCHYARD_VERSION")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn assert_chat_completions_ingest_shape(payload: &Value) {
    assert!(payload.get("request").is_some());
    assert!(payload.get("response").is_some());
    assert_eq!(payload["provider"], "switchyard");

    // The nemo-platform chat-completions ingest schema is strict. These legacy
    // entry-envelope fields do not belong on chat-completions ingest payloads.
    for key in ["data", "context", "external_id", "usage"] {
        assert!(payload.get(key).is_none(), "unexpected top-level key {key}");
    }
}

// Responses requests are normalized to Chat payloads and synthetic stream IDs are hidden.
#[test]
fn payload_builder_normalizes_responses_request_and_strips_synthetic_response_id() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        workspace: Some("default".to_string()),
        capture_content: true,
        ..IntakeSinkConfig::default()
    });
    let request = ChatRequest::openai_responses(json!({
        "model": "gpt-4o",
        "input": "say hi"
    }));
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("gpt-4o"));
    ctx.insert(RequestMetadata {
        intake: IntakeRequestMetadata {
            app: Some("codex".to_string()),
            task: Some("developer-session".to_string()),
            ..IntakeRequestMetadata::default()
        },
        ..RequestMetadata::default()
    });
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiResponses,
        session_id: Some("session-123".to_string()),
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx = switchyard_components::intake::IntakePayloadContext::from_proxy_context(
        &ctx,
        Some(1_700_000_001_840),
    );

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion("chatcmpl-switchyard-stream", "hello"),
        true,
    )?;

    assert_eq!(payload["request"]["messages"][0]["content"], "say hi");
    assert_eq!(
        payload["request"]["switchyard"]["inbound_format"],
        "openai_responses"
    );
    assert_eq!(payload["request"]["stream"], false);
    assert_eq!(payload["request"]["switchyard"]["stream"], true);
    assert!(payload["request"]["switchyard"].get("app").is_none());
    assert!(payload["request"]["switchyard"].get("task").is_none());
    assert_eq!(
        payload["evaluation_context"]["evaluation_run_id"],
        "session-123"
    );
    assert_eq!(
        payload["evaluation_context"]["test_case_id"],
        "developer-session"
    );
    assert_eq!(payload["request"]["switchyard"]["latency_ms"], 1840);
    assert!(payload["response"].get("id").is_none());
    assert_eq!(payload["provider"], "switchyard");
    assert!(payload["response"].get("switchyard").is_none());
    Ok(())
}

// Response payloads carry token usage; Switchyard timing stays under request metadata.
#[test]
fn payload_carries_response_usage_and_switchyard_timing() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        workspace: Some("default".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("openai/openai/gpt-5.2");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("openai/openai/gpt-5.2"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx = switchyard_components::intake::IntakePayloadContext::from_proxy_context(
        &ctx,
        Some(1_700_000_001_840),
    );

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "openai/openai/gpt-5.2",
            "hello",
            Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        ),
        false,
    )?;

    assert_eq!(payload["response"]["model"], "openai/openai/gpt-5.2");
    assert_eq!(payload["response"]["usage"]["prompt_tokens"], 10);
    assert_eq!(payload["response"]["usage"]["completion_tokens"], 5);
    assert_eq!(payload["cost_usd"], 0.000088);
    assert_eq!(payload["cost_input_usd"], 0.000018);
    assert_eq!(payload["cost_output_usd"], 0.00007);
    assert_eq!(payload["cost_details"]["base_input"], 0.000018);
    assert_eq!(payload["cost_details"]["cached_input"], 0.0);
    assert_eq!(payload["cost_details"]["cache_write"], 0.0);
    assert_chat_completions_ingest_shape(&payload);
    let request_switchyard = &payload["request"]["switchyard"];
    assert!(request_switchyard.get("served_model").is_none());
    assert!(request_switchyard.get("started_at_ms").is_none());
    assert!(request_switchyard.get("ended_at_ms").is_none());
    assert!(request_switchyard.get("duration_ms").is_none());
    assert_eq!(request_switchyard["latency_ms"], 1840);
    assert_eq!(request_switchyard["stream"], false);
    assert_eq!(request_switchyard["inbound_format"], "openai_chat");
    assert_eq!(request_switchyard["version"], expected_switchyard_version());
    assert!(payload["response"].get("switchyard").is_none());
    Ok(())
}

// Random routing decisions should survive into intake routing metadata.
#[test]
fn payload_carries_random_routing_metadata_from_typed_context() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig::default());
    let request = openai_chat_request("client-model");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("weak-model"));
    ctx.insert(RandomRoutingDecision {
        tier: RandomRoutingTier::Weak,
        selected_target: LlmTargetId::from_static("weak"),
        selected_model: ModelId::from_static("weak-model"),
        original_model: Some("client-model".to_string()),
        strong_probability: 0.5,
        draw: 0.75,
    });
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx =
        switchyard_components::intake::IntakePayloadContext::from_proxy_context(&ctx, None);

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage("chatcmpl-test", "weak-model", "hello", None),
        false,
    )?;

    let routing = &payload["request"]["switchyard"]["routing"];
    assert_eq!(routing["router_type"], "random");
    assert_eq!(routing["routed_to"], "weak");
    Ok(())
}

// Generic stats route labels fill the routing payload when no router decision exists.
#[test]
fn payload_carries_custom_route_label_when_no_random_decision_exists() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig::default());
    let request = openai_chat_request("client-model");
    let mut ctx = ProxyContext::new();
    ctx.insert(StatsRouteLabel::new("plugin-tier"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx =
        switchyard_components::intake::IntakePayloadContext::from_proxy_context(&ctx, None);

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage("chatcmpl-test", "plugin-model", "hello", None),
        false,
    )?;

    let routing = &payload["request"]["switchyard"]["routing"];
    assert_eq!(routing["router_type"], "custom");
    assert_eq!(routing["routed_to"], "plugin-tier");
    Ok(())
}

// Unknown models still report token usage but do not invent chat-completions cost fields.
#[test]
fn payload_usage_omits_cost_for_unknown_model_even_with_tokens() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        workspace: Some("default".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("made-up-model");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("made-up-model"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx =
        switchyard_components::intake::IntakePayloadContext::from_proxy_context(&ctx, None);

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "made-up-model",
            "hello",
            Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        ),
        false,
    )?;

    assert_eq!(payload["response"]["model"], "made-up-model");
    assert_eq!(payload["response"]["usage"]["prompt_tokens"], 10);
    assert_eq!(payload["response"]["usage"]["completion_tokens"], 5);
    assert!(payload.get("cost_usd").is_none());
    assert!(payload.get("cost_input_usd").is_none());
    assert!(payload.get("cost_output_usd").is_none());
    assert!(payload.get("cost_details").is_none());
    assert_chat_completions_ingest_shape(&payload);
    Ok(())
}

// Known models without token usage should not produce partial chat-completions cost fields.
#[test]
fn payload_usage_omits_cost_when_tokens_are_missing_for_known_model() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        workspace: Some("default".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("openai/openai/gpt-5.2");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("openai/openai/gpt-5.2"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx =
        switchyard_components::intake::IntakePayloadContext::from_proxy_context(&ctx, None);

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage("chatcmpl-test", "openai/openai/gpt-5.2", "hello", None),
        false,
    )?;

    assert_eq!(payload["response"]["model"], "openai/openai/gpt-5.2");
    assert!(payload["response"].get("usage").is_none());
    assert_chat_completions_ingest_shape(&payload);
    Ok(())
}

// Missing backend selection should be represented as null, not as the request model.
#[test]
fn payload_usage_uses_null_model_when_served_model_is_missing() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        workspace: Some("default".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("openai/openai/gpt-5.2");
    let mut ctx = ProxyContext::new();
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx =
        switchyard_components::intake::IntakePayloadContext::from_proxy_context(&ctx, None);

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "openai/openai/gpt-5.2",
            "hello",
            Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        ),
        false,
    )?;

    assert!(payload["request"]["switchyard"]
        .get("served_model")
        .is_none());
    Ok(())
}

// NVDataflow mode flattens the record into top-level, type-prefixed fields.
#[test]
fn payload_builds_flat_nvdataflow_document_when_project_set() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        user_id: "0badf00d".to_string(),
        nvdataflow_project: Some("sandbox-switchyard".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("openai/openai/gpt-5.2");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("openai/openai/gpt-5.2"));
    ctx.insert(RandomRoutingDecision {
        tier: RandomRoutingTier::Weak,
        selected_target: LlmTargetId::from_static("weak"),
        selected_model: ModelId::from_static("openai/openai/gpt-5.2"),
        original_model: Some("openai/openai/gpt-5.2".to_string()),
        strong_probability: 0.5,
        draw: 0.75,
    });
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: Some("claude-smoke-0001".to_string()),
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx = switchyard_components::intake::IntakePayloadContext::from_proxy_context(
        &ctx,
        Some(1_700_000_001_840),
    );

    let payload = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "openai/openai/gpt-5.2",
            "hello",
            Some(json!({"prompt_tokens": 25000, "completion_tokens": 6580, "total_tokens": 31580})),
        ),
        false,
    )?;

    // Flat schema: no nested request/response envelope, all metrics top-level.
    assert!(payload.get("request").is_none());
    assert!(payload.get("response").is_none());
    assert_eq!(payload["s_source"], "switchyard");
    assert_eq!(payload["s_record_type"], "switchyard_request");
    assert_eq!(payload["l_schema_version"], 1);
    assert_eq!(payload["ts_created"], 1_700_000_001_840i64);
    assert_eq!(payload["_id"], "claude-smoke-0001-1700000001840");
    assert_eq!(payload["s_switchyard_session_id"], "claude-smoke-0001");
    assert_eq!(payload["s_switchyard_user_id"], "0badf00d");
    assert_eq!(
        payload["s_switchyard_served_model"],
        "openai/openai/gpt-5.2"
    );
    assert_eq!(payload["s_switchyard_inbound_format"], "openai_chat");
    assert_eq!(payload["s_switchyard_router_type"], "random");
    assert_eq!(payload["s_switchyard_routed_to"], "weak");
    assert_eq!(payload["b_switchyard_routed"], true);
    assert_eq!(payload["l_switchyard_input_tokens"], 25000);
    assert_eq!(payload["l_switchyard_output_tokens"], 6580);
    assert_eq!(payload["l_switchyard_total_tokens"], 31580);
    assert_eq!(payload["l_switchyard_latency_ms"], 1840);
    assert!(payload["f_switchyard_cost_usd"].is_number());
    assert!(payload["text_switchyard_record_json"].is_string());
    Ok(())
}

// Sub-model routing records reuse the turn's context but carry the router's own
// model/usage/route, and each gets a distinct NVDataflow _id so the primary
// record and every routing call coexist instead of overwriting one another.
#[test]
fn submodel_records_get_distinct_nvdataflow_ids_and_router_fields() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        user_id: "0badf00d".to_string(),
        nvdataflow_project: Some("sandbox-switchyard".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = openai_chat_request("openai/openai/gpt-5.2");
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("openai/openai/gpt-5.2"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: Some("sess-1".to_string()),
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx = switchyard_components::intake::IntakePayloadContext::from_proxy_context(
        &ctx,
        Some(1_700_000_001_840),
    );

    // Primary turn record and two routing calls share one session + timestamp.
    let primary = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "openai/openai/gpt-5.2",
            "hello",
            Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        ),
        false,
    )?;
    let call = SubModelCall {
        model: "gcp/google/gemini-3.5-flash".to_string(),
        prompt_tokens: 1200,
        completion_tokens: 80,
        cached_tokens: 0,
        router_type: "deterministic".to_string(),
        routed_to: "classifier".to_string(),
    };
    let first = builder.build_submodel_record(&payload_ctx, &call, 0)?;
    let second = builder.build_submodel_record(&payload_ctx, &call, 1)?;

    // Distinct _id per record so nothing overwrites another in the data lake.
    assert_eq!(primary["_id"], "sess-1-1700000001840");
    assert_eq!(first["_id"], "sess-1-1700000001840-classifier-0");
    assert_eq!(second["_id"], "sess-1-1700000001840-classifier-1");

    // The routing record carries the router's own model, usage, and route label.
    assert_eq!(
        first["s_switchyard_requested_model"],
        "gcp/google/gemini-3.5-flash"
    );
    assert_eq!(
        first["s_switchyard_served_model"],
        "gcp/google/gemini-3.5-flash"
    );
    assert_eq!(first["s_switchyard_router_type"], "deterministic");
    assert_eq!(first["s_switchyard_routed_to"], "classifier");
    assert_eq!(first["l_switchyard_input_tokens"], 1200);
    assert_eq!(first["l_switchyard_output_tokens"], 80);
    // Start stamp is dropped for routing calls, so no turn-wide latency leaks in.
    assert!(first.get("l_switchyard_latency_ms").is_none());
    Ok(())
}

// Default is metadata-only: no prompt/response text in the document, metrics kept.
#[test]
fn nvdataflow_document_is_metadata_only_by_default() -> Result<()> {
    let builder = IntakePayloadBuilder::new(IntakeSinkConfig {
        nvdataflow_project: Some("sandbox-switchyard".to_string()),
        ..IntakeSinkConfig::default()
    });
    let request = ChatRequest::openai_chat(json!({
        "model": "openai/openai/gpt-5.2",
        "messages": [
            {"role": "system", "content": "SENTINEL_SYSTEM_PROMPT"},
            {"role": "user", "content": "SENTINEL_USER_PROMPT"}
        ],
        "tools": [{"type": "function", "function": {"name": "SENTINEL_TOOL"}}],
        "functions": [{"name": "SENTINEL_LEGACY_FUNCTION"}],
        "function_call": {"name": "SENTINEL_FUNCTION_CALL"}
    }));
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("openai/openai/gpt-5.2"));
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: Some("sess-1".to_string()),
        skip: false,
        request_snapshot: Some(request.clone()),
    });
    let payload_ctx = switchyard_components::intake::IntakePayloadContext::from_proxy_context(
        &ctx,
        Some(1_700_000_001_840),
    );

    let doc = builder.build(
        &payload_ctx,
        &request,
        &completion_with_usage(
            "chatcmpl-test",
            "openai/openai/gpt-5.2",
            "SENTINEL_RESPONSE_TEXT",
            Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        ),
        false,
    )?;

    let serialized = serde_json::to_string(&doc).unwrap();
    for sentinel in [
        "SENTINEL_SYSTEM_PROMPT",
        "SENTINEL_USER_PROMPT",
        "SENTINEL_TOOL",
        "SENTINEL_LEGACY_FUNCTION",
        "SENTINEL_FUNCTION_CALL",
        "SENTINEL_RESPONSE_TEXT",
    ] {
        assert!(!serialized.contains(sentinel), "leaked content: {sentinel}");
    }

    assert_eq!(doc["s_switchyard_served_model"], "openai/openai/gpt-5.2");
    assert_eq!(doc["l_switchyard_input_tokens"], 10);
    assert_eq!(doc["l_switchyard_output_tokens"], 5);
    assert!(doc["f_switchyard_cost_usd"].is_number());
    Ok(())
}
