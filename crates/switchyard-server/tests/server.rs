// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the components-v2 Rust profile server.

use std::io::ErrorKind;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{HeaderName, HeaderValue, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use switchyard_components_v2::{
    parse_profile_config_str, profile_stats_accumulator, Profile, ProfileConfigFormat,
    ProfileInput, ProfileResponse, RelayIdentityKey, RelaySnapshotLimits, RoutingMetadata,
    PROXY_SESSION_ID_HEADER, RELAY_SESSION_ID_HEADER, ROUTING_DECISION_SCHEMA_VERSION,
    ROUTING_REQUEST_SCHEMA_VERSION,
};
use switchyard_core::{
    ChatRequestType, ChatResponse, ModelId, Result, StreamEvent, SwitchyardError,
};
use switchyard_server::{build_switchyard_router, ProfileRegistry, ServerState};
use tower::ServiceExt;

#[tokio::test]
async fn minimal_noop_config_boots_and_serves_core_routes() -> TestResult {
    let _stats_guard = stats_guard().await;
    reset_stats()?;
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let health = app
        .clone()
        .oneshot(request("GET", "/health", None)?)
        .await?;
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(json_body(health).await?, json!({"status": "ok"}));

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    assert_eq!(models.status(), StatusCode::OK);
    let models = json_body(models).await?;
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "bench");
    assert_eq!(models["default_model"], "bench");
    assert_eq!(models["model_pool"], json!(["bench"]));

    let chat = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "bench",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(chat.status(), StatusCode::OK);
    assert_eq!(
        json_body(chat).await?["choices"][0]["message"]["content"],
        "ok"
    );
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_uses_configured_random_profile_without_backend_dispatch() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
    base_url: http://127.0.0.1:1/v1
  efficient:
    model: upstream-efficient
    format: responses
    base_url: http://127.0.0.1:1/v1
profiles:
  remote-random:
    type: random-routing
    strong: capable
    weak: efficient
    strong_probability: 1.0
    rng_seed: 7
"#,
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json("remote-random", "none", None)),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let decision = json_body(response).await?;
    assert_eq!(decision["schema_version"], ROUTING_DECISION_SCHEMA_VERSION);
    assert_eq!(decision["router"]["name"], "random-routing");
    assert_eq!(decision["route"]["backend_id"], "capable");
    assert_eq!(decision["route"]["target_model"], "upstream-capable");
    assert_eq!(decision["route"]["target_protocol_profile"], "openai_chat");
    Ok(())
}

#[tokio::test]
async fn stage_router_decision_warms_from_one_relay_record_without_target_dispatch() -> TestResult {
    let state = state_from_yaml(stage_router_decision_yaml())?;
    let app = build_switchyard_router(state);
    let event = tool_event(
        "decision-oom",
        "end",
        Some("session-1"),
        None,
        json!({"output": "process failed: out of memory"}),
    );

    let ingest = app
        .clone()
        .oneshot(ndjson_request(&format!("{event}\n"), None)?)
        .await?;
    assert_eq!(ingest.status(), StatusCode::OK);
    assert_eq!(json_body(ingest).await?["accepted_events"], 1);

    // The legacy Relay fixture includes decision_profile.router and intentionally
    // sends no session header. Both selected targets are unreachable; a 200 proves
    // the decision-only path did not dispatch either target.
    let warm = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json("remote-stage_router", "summary_only", None)),
        )?)
        .await?;
    assert_eq!(warm.status(), StatusCode::OK);
    let warm = json_body(warm).await?;
    assert_eq!(warm["router"]["name"], "stage_router");
    assert_eq!(warm["route"]["backend_id"], "capable");
    assert_eq!(warm["route"]["target_model"], "provider/capable");
    assert_eq!(warm["reason_code"], "override");
    assert_eq!(warm["confidence"], 1.0);
    assert_eq!(warm["metadata"]["feature_state"], "fresh");
    assert_eq!(warm["metadata"]["source"], "override");

    let mut cold_request = routing_request_json("remote-stage_router", "summary_only", None);
    cold_request["identity"]["session_id"] = json!("different-session");
    let cold = app
        .oneshot(request("POST", "/v1/routing/decision", Some(cold_request))?)
        .await?;
    assert_eq!(cold.status(), StatusCode::OK);
    let cold = json_body(cold).await?;
    assert_eq!(cold["route"]["backend_id"], "efficient");
    assert_eq!(cold["reason_code"], "stage_router_feature_cold_default");
    assert!(cold["confidence"].is_null());
    assert_eq!(cold["metadata"]["feature_state"], "cold");
    assert_eq!(cold["metadata"]["source"], "fall_open");
    Ok(())
}

#[tokio::test]
async fn stage_router_decision_uses_exact_owner_identity_and_keeps_turn_only_state_cold() -> TestResult {
    let state = state_from_yaml(stage_router_decision_yaml())?;
    let app = build_switchyard_router(state);
    let owner_event = tool_event(
        "owner-oom",
        "end",
        Some("owner-session"),
        Some("owner-a"),
        json!({"output": "CUDA out of memory"}),
    );
    let session_only_event = tool_event(
        "session-oom",
        "end",
        Some("owner-session"),
        None,
        json!({"output": "CUDA out of memory"}),
    );
    let turn_only = turn_event("turn-only-start", "turn-session", None);
    for event in [owner_event, session_only_event, turn_only] {
        let response = app
            .clone()
            .oneshot(ndjson_request(&format!("{event}\n"), None)?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let mut exact = routing_request_json("remote-stage_router", "summary_only", None);
    exact["identity"]["session_id"] = json!("owner-session");
    exact["identity"]["owner_id"] = json!("owner-a");
    let exact = app
        .clone()
        .oneshot(request("POST", "/v1/routing/decision", Some(exact))?)
        .await?;
    assert_eq!(
        json_body(exact).await?["metadata"]["feature_state"],
        "fresh"
    );

    let mut padded_owner = routing_request_json("remote-stage_router", "summary_only", None);
    padded_owner["identity"]["session_id"] = json!("owner-session");
    padded_owner["identity"]["owner_id"] = json!(" owner-a ");

    let mut session_only = routing_request_json("remote-stage_router", "summary_only", None);
    session_only["identity"]["session_id"] = json!("owner-session");

    let mut blank_owner = session_only.clone();
    blank_owner["identity"]["owner_id"] = json!("   ");

    for canonical_match in [padded_owner, session_only, blank_owner] {
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/v1/routing/decision",
                Some(canonical_match),
            )?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            json_body(response).await?["metadata"]["feature_state"],
            "fresh"
        );
    }

    let mut owner_mismatch = routing_request_json("remote-stage_router", "summary_only", None);
    owner_mismatch["identity"]["session_id"] = json!("owner-session");
    owner_mismatch["identity"]["owner_id"] = json!("owner-b");
    owner_mismatch["identity"]["parent_scope_id"] = json!("owner-a");
    owner_mismatch["identity"]["root_scope_id"] = json!("owner-a");

    let mut turn_only = routing_request_json("remote-stage_router", "summary_only", None);
    turn_only["identity"]["session_id"] = json!("turn-session");

    for cold_request in [owner_mismatch, turn_only] {
        let response = app
            .clone()
            .oneshot(request("POST", "/v1/routing/decision", Some(cold_request))?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let decision = json_body(response).await?;
        assert_eq!(decision["route"]["backend_id"], "efficient");
        assert_eq!(decision["reason_code"], "stage_router_feature_cold_default");
        assert_eq!(decision["metadata"]["feature_state"], "cold");
    }
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_reconciles_all_session_header_aliases() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(random_decision_yaml())?);
    let accepted: Vec<Vec<(&'static str, &'static [u8])>> = vec![
        vec![(PROXY_SESSION_ID_HEADER, b"session-1")],
        vec![(RELAY_SESSION_ID_HEADER, b"session-1")],
        vec![
            (PROXY_SESSION_ID_HEADER, b"session-1"),
            (RELAY_SESSION_ID_HEADER, b"session-1"),
        ],
        vec![
            (PROXY_SESSION_ID_HEADER, b"session-1"),
            (PROXY_SESSION_ID_HEADER, b"session-1"),
        ],
        vec![(RELAY_SESSION_ID_HEADER, b" session-1 ")],
    ];

    for headers in accepted {
        let response = app
            .clone()
            .oneshot(request_with_headers(
                "POST",
                "/v1/routing/decision",
                Some(routing_request_json("remote-random", "none", None)),
                &headers,
            )?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK, "headers={headers:?}");
    }
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_rejects_invalid_or_conflicting_session_headers() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(random_decision_yaml())?);
    let rejected: Vec<Vec<(&'static str, &'static [u8])>> = vec![
        vec![
            (PROXY_SESSION_ID_HEADER, b"session-1"),
            (RELAY_SESSION_ID_HEADER, b"session-2"),
        ],
        vec![
            (PROXY_SESSION_ID_HEADER, b"session-1"),
            (PROXY_SESSION_ID_HEADER, b"session-2"),
        ],
        vec![(PROXY_SESSION_ID_HEADER, b"")],
        vec![(RELAY_SESSION_ID_HEADER, b"   ")],
        vec![(PROXY_SESSION_ID_HEADER, b"different-session")],
        vec![(RELAY_SESSION_ID_HEADER, b"\xff")],
    ];

    for headers in rejected {
        let response = app
            .clone()
            .oneshot(request_with_headers(
                "POST",
                "/v1/routing/decision",
                Some(routing_request_json("remote-random", "none", None)),
                &headers,
            )?)
            .await?;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "headers={headers:?}"
        );
        assert_eq!(
            json_body(response).await?["error"]["code"],
            "invalid_request_error"
        );
    }
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_classifies_materialized_llm_request_without_target_dispatch(
) -> TestResult {
    let classifier_response = json!({
        "id": "chatcmpl-classifier",
        "object": "chat.completion",
        "model": "classifier-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-route",
                    "type": "function",
                    "function": {
                        "name": "select_route",
                        "arguments": {
                            "recommended_tier": "complex",
                            "confidence": 0.95,
                            "abstain": false,
                            "turn_type": "debug",
                            "code_modification_scope": "cross_module",
                            "tool_call_count_estimate": 4,
                            "requires_codebase_context": true
                        }
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let Some(classifier) = HttpStub::start_with_response(1, classifier_response)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
    base_url: http://127.0.0.1:1/v1
  efficient:
    model: upstream-efficient
    format: openai
    base_url: http://127.0.0.1:1/v1
  classifier:
    model: classifier-model
    format: openai
    base_url: {classifier_url}
profiles:
  remote-llm:
    type: llm-routing
    strong: capable
    weak: efficient
    classifier: classifier
    fallback_target_on_evict: capable
    profile_name: coding_agent
    classifier_min_confidence: 0.0
"#,
        classifier_url = classifier.base_url
    ))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json(
                "remote-llm",
                "full_body",
                Some(json!({
                    "body": {
                        "model": "client/model",
                        "messages": [{"role": "user", "content": "debug failing tests"}]
                    }
                })),
            )),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let decision = json_body(response).await?;
    assert_eq!(decision["router"]["name"], "llm-routing");
    assert_eq!(decision["route"]["backend_id"], "capable");
    assert_eq!(decision["confidence"], 0.95);
    assert_eq!(classifier.requests()?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_distinguishes_malformed_unknown_and_unsupported_profiles() -> TestResult
{
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let malformed = app
        .clone()
        .oneshot(raw_request("POST", "/v1/routing/decision", "{")?)
        .await?;
    assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_body(malformed).await?["error"]["code"], "invalid_body");

    let mut missing_id = routing_request_json("bench", "summary_only", None);
    missing_id["decision_profile"]
        .as_object_mut()
        .ok_or("decision_profile was not an object")?
        .remove("profile_id");
    let missing_id = app
        .clone()
        .oneshot(request("POST", "/v1/routing/decision", Some(missing_id))?)
        .await?;
    assert_eq!(missing_id.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(missing_id).await?["error"]["code"],
        "invalid_body"
    );

    let unknown = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json("missing", "summary_only", None)),
        )?)
        .await?;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        json_body(unknown).await?["error"]["code"],
        "decision_profile_not_found"
    );

    let unsupported = app
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json("bench", "summary_only", None)),
        )?)
        .await?;
    assert_eq!(unsupported.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        json_body(unsupported).await?["error"]["code"],
        "decision_not_supported"
    );
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_validates_schema_and_attempt_before_unknown_profile_lookup() -> TestResult
{
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let mut invalid_schema = routing_request_json("missing", "summary_only", None);
    invalid_schema["schema_version"] = json!("switchyard.routing_request.v999");
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(invalid_schema),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = json_body(response).await?;
    assert_eq!(error["error"]["code"], "invalid_request_error");
    assert!(error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("schema_version")));

    for (routing_attempt, max_routing_attempts) in [(0, 1), (2, 1), (1, 0)] {
        let mut invalid_attempt = routing_request_json("missing", "summary_only", None);
        invalid_attempt["attempt"]["routing_attempt"] = json!(routing_attempt);
        invalid_attempt["attempt"]["max_routing_attempts"] = json!(max_routing_attempts);
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/v1/routing/decision",
                Some(invalid_attempt),
            )?)
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error = json_body(response).await?;
        assert_eq!(error["error"]["code"], "invalid_request_error");
        assert!(error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("routing_attempt")));
    }
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_rejects_unknown_inbound_before_unknown_profile_lookup() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
  efficient:
    model: upstream-efficient
    format: openai
profiles:
  remote-random:
    type: random-routing
    strong: capable
    weak: efficient
    strong_probability: 1.0
"#,
    )?);
    let mut body = routing_request_json("missing", "summary_only", None);
    body["protocol"]["inbound_profile"] = json!("unrecognized-chat-format");

    let response = app
        .oneshot(request("POST", "/v1/routing/decision", Some(body))?)
        .await?;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = json_body(response).await?;
    assert_eq!(error["error"]["code"], "invalid_request_error");
    assert!(error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("unsupported inbound_profile")));
    Ok(())
}

#[tokio::test]
async fn llm_decision_rejects_summary_only_materialization_before_classifier_call() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
  efficient:
    model: upstream-efficient
    format: openai
  classifier:
    model: classifier-model
    format: openai
    base_url: http://127.0.0.1:1/v1
profiles:
  remote-llm:
    type: llm-routing
    strong: capable
    weak: efficient
    classifier: classifier
    fallback_target_on_evict: capable
    profile_name: coding_agent
"#,
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/routing/decision",
            Some(routing_request_json("remote-llm", "summary_only", None)),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = json_body(response).await?;
    assert!(error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("materialized current_request.body")));
    Ok(())
}

#[tokio::test]
async fn relay_http_post_events_build_one_router_neutral_snapshot() -> TestResult {
    let state = state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?;
    let app = build_switchyard_router(state.clone());
    let start = tool_event(
        "tool-1",
        "start",
        Some("session-1"),
        Some("owner-a"),
        json!({"command": "cargo test"}),
    );
    let end = tool_event(
        "tool-1",
        "end",
        Some("session-1"),
        Some("owner-a"),
        json!({"output": "ok"}),
    );

    // Relay's selected `http_post` transport sends one NDJSON record per POST.
    for (event, expected_count) in [(start, 1), (end, 2)] {
        let response = app
            .clone()
            .oneshot(ndjson_request(&format!("{event}\n"), None)?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await?;
        assert_eq!(body["accepted_events"], 1);
        assert_eq!(body["accumulator_ingests"], 1);
        assert_eq!(body["batch"]["ingested_events"], 1);
        assert_eq!(body["cumulative"]["ingested_events"], expected_count);
    }

    let key = RelayIdentityKey::new("session-1", Some("owner-a".to_string()));
    let snapshot = state
        .relay_snapshots()
        .snapshot(&key)
        .ok_or("expected Relay snapshot")?;
    assert_eq!(snapshot.identity, key);
    assert_eq!(snapshot.messages.len(), 2);
    assert_eq!(snapshot.messages[0]["role"], "assistant");
    assert_eq!(snapshot.messages[1]["role"], "tool");
    assert_eq!(snapshot.messages[1]["content"], "ok");
    assert_eq!(snapshot.event_count, 2);
    Ok(())
}

#[tokio::test]
async fn atof_batch_parse_and_validation_fail_without_partial_ingest() -> TestResult {
    let state = state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?;
    let app = build_switchyard_router(state.clone());
    let event = tool_event("tool-1", "start", Some("session-1"), None, json!({}));

    let malformed = app
        .clone()
        .oneshot(ndjson_request(&format!("{event}\n{{not-json}}\n"), None)?)
        .await?;
    assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_body(malformed).await?["error"]["code"], "invalid_body");
    assert!(state
        .relay_snapshots()
        .snapshot(&RelayIdentityKey::session_only("session-1"))
        .is_none());

    let non_object = app
        .clone()
        .oneshot(ndjson_request(&format!("{event}\n[]\n"), None)?)
        .await?;
    assert_eq!(non_object.status(), StatusCode::BAD_REQUEST);
    assert!(state
        .relay_snapshots()
        .snapshot(&RelayIdentityKey::session_only("session-1"))
        .is_none());

    let mut malformed_scope = event.clone();
    malformed_scope
        .as_object_mut()
        .ok_or("test ATOF event must be an object")?
        .remove("name");
    let semantic = app
        .oneshot(ndjson_request(
            &format!("{event}\n{malformed_scope}\n"),
            None,
        )?)
        .await?;
    assert_eq!(semantic.status(), StatusCode::BAD_REQUEST);
    assert!(json_body(semantic).await?["error"]["message"]
        .as_str()
        .ok_or("semantic error message missing")?
        .contains("canonical recognized scope"));
    assert!(state
        .relay_snapshots()
        .snapshot(&RelayIdentityKey::session_only("session-1"))
        .is_none());
    assert_eq!(state.relay_snapshots().counters().batches, 0);
    Ok(())
}

#[tokio::test]
async fn atof_endpoint_reports_duplicate_and_drop_categories() -> TestResult {
    let state = state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?;
    let app = build_switchyard_router(state);
    let event = tool_event("tool-1", "start", Some("session-1"), None, json!({}));
    let missing_identity = tool_event("tool-2", "start", None, None, json!({}));
    let body = format!(
        "{event}\n{event}\n{missing_identity}\n{}\n",
        json!({"kind": "mark", "uuid": "mark-1"})
    );

    let response = app.oneshot(ndjson_request(&body, None)?).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await?;
    assert_eq!(body["batch"]["received_events"], 4);
    assert_eq!(body["batch"]["ingested_events"], 1);
    assert_eq!(body["accepted_events"], 1);
    assert_eq!(body["received_events"], 4);
    assert_eq!(body["batch"]["duplicate_events"], 1);
    assert_eq!(body["batch"]["dropped_events"], 3);
    assert_eq!(body["batch"]["dropped_missing_identity_events"], 1);
    assert_eq!(body["batch"]["dropped_unrecognized_events"], 1);
    Ok(())
}

#[tokio::test]
async fn atof_endpoint_enforces_optional_bearer_token() -> TestResult {
    let state = state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?
    .with_atof_bearer_token(Some("expected-token".to_string()))?;
    let app = build_switchyard_router(state);
    let event = tool_event("tool-1", "start", Some("session-1"), None, json!({}));
    let body = format!("{event}\n");

    let missing = app.clone().oneshot(ndjson_request(&body, None)?).await?;
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(header(&missing, "www-authenticate"), Some("Bearer"));

    let wrong = app
        .clone()
        .oneshot(ndjson_request(&body, Some("Bearer wrong"))?)
        .await?;
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);

    let accepted = app
        .oneshot(ndjson_request(&body, Some("bearer expected-token"))?)
        .await?;
    assert_eq!(accepted.status(), StatusCode::OK);
    Ok(())
}

#[tokio::test]
async fn decision_endpoint_enforces_relay_bearer_token_end_to_end() -> TestResult {
    let state = state_from_yaml(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
    base_url: http://127.0.0.1:1/v1
  efficient:
    model: upstream-efficient
    format: openai
    base_url: http://127.0.0.1:1/v1
profiles:
  remote-random:
    type: random-routing
    strong: capable
    weak: efficient
    strong_probability: 1.0
    rng_seed: 7
"#,
    )?
    .with_atof_bearer_token(Some("expected-token".to_string()))?;
    let app = build_switchyard_router(state);
    let body = routing_request_json("remote-random", "none", None);

    let missing = app
        .clone()
        .oneshot(request("POST", "/v1/routing/decision", Some(body.clone()))?)
        .await?;
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(header(&missing, "www-authenticate"), Some("Bearer"));

    let wrong = app
        .clone()
        .oneshot(request_with_headers(
            "POST",
            "/v1/routing/decision",
            Some(body.clone()),
            &[("authorization", b"Bearer wrong")],
        )?)
        .await?;
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);

    let accepted = app
        .oneshot(request_with_headers(
            "POST",
            "/v1/routing/decision",
            Some(body),
            &[("authorization", b"bearer expected-token")],
        )?)
        .await?;
    assert_eq!(accepted.status(), StatusCode::OK);
    assert_eq!(json_body(accepted).await?["route"]["backend_id"], "capable");
    Ok(())
}

#[test]
fn explicit_blank_atof_bearer_token_is_invalid_config() -> TestResult {
    for token in ["", "   ", "\t\n"] {
        let state = state_from_yaml(
            r#"
profiles:
  bench:
    type: noop
"#,
        )?;
        let error = state
            .with_atof_bearer_token(Some(token.to_string()))
            .err()
            .ok_or("expected blank ATOF bearer token error")?;
        assert!(error.to_string().contains("cannot be blank"));
    }
    Ok(())
}

#[tokio::test]
async fn atof_endpoint_enforces_content_type_event_and_batch_limits() -> TestResult {
    let state = state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?
    .with_relay_snapshot_limits(RelaySnapshotLimits {
        max_identities: 4,
        max_history_per_identity: 8,
        max_dedupe_entries: 16,
        max_retained_bytes: 4_096,
        max_event_bytes: 256,
        max_batch_bytes: 512,
    })?;
    let app = build_switchyard_router(state.clone());

    let wrong_type = app
        .clone()
        .oneshot(raw_request("POST", "/v1/atof/events", "{}")?)
        .await?;
    assert_eq!(wrong_type.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let oversized_event = tool_event(
        "tool-1",
        "end",
        Some("session-1"),
        None,
        json!({"output": "x".repeat(300)}),
    );
    let event_response = app
        .clone()
        .oneshot(ndjson_request(&format!("{oversized_event}\n"), None)?)
        .await?;
    assert_eq!(event_response.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let batch_response = app.oneshot(ndjson_request(&" ".repeat(513), None)?).await?;
    assert_eq!(batch_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(state.relay_snapshots().identity_count(), 0);
    Ok(())
}

#[tokio::test]
async fn all_request_endpoints_route_through_selected_profile() -> TestResult {
    let _stats_guard = stats_guard().await;
    reset_stats()?;
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let anthropic = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "bench",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(anthropic.status(), StatusCode::OK);
    let anthropic = json_body(anthropic).await?;
    assert_eq!(anthropic["type"], "message");
    assert_eq!(anthropic["content"][0]["text"], "ok");

    let responses = app
        .oneshot(request(
            "POST",
            "/v1/responses",
            Some(json!({"model": "bench", "input": "hi"})),
        )?)
        .await?;
    assert_eq!(responses.status(), StatusCode::OK);
    let responses = json_body(responses).await?;
    assert_eq!(responses["object"], "response");
    assert_eq!(responses["output"][0]["content"][0]["text"], "ok");
    Ok(())
}

#[tokio::test]
async fn missing_and_unknown_models_return_client_errors() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let missing = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({"messages": [{"role": "user", "content": "hi"}]})),
        )?)
        .await?;
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(missing).await?["error"]["type"],
        "invalid_request_error"
    );

    let unknown = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "missing-route",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        json_body(unknown).await?["error"]["type"],
        "model_not_found"
    );
    Ok(())
}

#[tokio::test]
async fn malformed_and_non_object_json_return_shared_client_errors() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    for uri in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
        let malformed = app.clone().oneshot(raw_request("POST", uri, "{")?).await?;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(malformed).await?["error"]["code"], "invalid_body");

        let non_object = app.clone().oneshot(raw_request("POST", uri, "[]")?).await?;
        assert_eq!(non_object.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(non_object).await?["error"]["code"],
            "invalid_body"
        );
    }
    Ok(())
}

#[tokio::test]
async fn translation_errors_do_not_emit_routing_metadata_headers() -> TestResult {
    let app = build_switchyard_router(state_from_profile("bad", Arc::new(BadTranslationProfile))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "bad",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            })),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!response
        .headers()
        .keys()
        .any(|name| name.as_str().starts_with("x-model-router-")));
    Ok(())
}

#[tokio::test]
async fn target_id_and_target_model_aliases_are_advertised_and_routable() -> TestResult {
    let _stats_guard = stats_guard().await;
    let Some(stub) = HttpStub::start(2)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  direct:
    model: upstream-direct
    format: openai
    base_url: {base_url}
profiles:
  direct-profile:
    type: passthrough
    target: direct
"#,
        base_url = stub.base_url
    ))?);

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    let model_ids = json_body(models).await?["model_pool"].clone();
    assert_eq!(
        model_ids,
        json!(["direct-profile", "direct", "upstream-direct"])
    );

    for public_model in ["direct", "upstream-direct"] {
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                Some(json!({
                    "model": public_model,
                    "messages": [{"role": "user", "content": "hi"}],
                })),
            )?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["model"], "upstream-direct");
    assert_eq!(seen[1]["model"], "upstream-direct");
    Ok(())
}

#[tokio::test]
async fn target_with_same_id_and_model_is_registered_once() -> TestResult {
    let _stats_guard = stats_guard().await;
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  upstream-direct:
    model: upstream-direct
    format: openai
    base_url: {base_url}
profiles:
  direct-profile:
    type: passthrough
    target: upstream-direct
"#,
        base_url = stub.base_url
    ))?);

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    assert_eq!(
        json_body(models).await?["model_pool"],
        json!(["direct-profile", "upstream-direct"])
    );

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "upstream-direct",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(stub.requests()?[0]["model"], "upstream-direct");
    Ok(())
}

#[tokio::test]
async fn random_routing_profile_reaches_selected_backend_path() -> TestResult {
    let _stats_guard = stats_guard().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  capable:
    model: upstream-capable
    format: openai
    base_url: {base_url}
  efficient:
    model: upstream-efficient
    format: openai
    base_url: {base_url}
profiles:
  random:
    type: random-routing
    strong: capable
    weak: efficient
    strong_probability: 0.0000004
    rng_seed: 7
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "random",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    for (name, expected) in [
        ("x-model-router-selected-model", "upstream-efficient"),
        ("x-model-router-selected-tier", "weak"),
        ("x-model-router-version", "random-routing:v1"),
        ("x-model-router-tolerance", "0.0000004"),
    ] {
        assert_eq!(header(&response, name), Some(expected));
    }
    assert!(header(&response, "x-model-router-rationale")
        .is_some_and(|value| value.contains("strong_probability 0.0000004; selected weak")));

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["model"], "upstream-efficient");
    Ok(())
}

#[tokio::test]
async fn latency_service_profile_reaches_configured_backend_path() -> TestResult {
    let _stats_guard = stats_guard().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  fast:
    model: upstream-fast
    format: openai
    base_url: {base_url}
profiles:
  latency:
    type: latency-service
    latency_service_url: http://latency.local
    targets: [fast]
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "latency",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["model"], "upstream-fast");
    Ok(())
}

#[tokio::test]
async fn duplicate_public_model_ids_are_rejected() -> TestResult {
    let err = state_from_yaml(
        r#"
targets:
  direct:
    model: same
    format: openai
    base_url: http://127.0.0.1:9/v1
profiles:
  same:
    type: noop
"#,
    )
    .err()
    .ok_or("expected duplicate public model id failure")?;

    assert!(err.to_string().contains("same"));
    assert!(err.to_string().contains("already registered"));
    Ok(())
}

#[tokio::test]
async fn stats_endpoints_use_components_v2_global_accumulator() -> TestResult {
    let _stats_guard = stats_guard().await;
    reset_stats()?;
    profile_stats_accumulator().record_success("served-model", Some(12.0), Some("capable"))?;
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let stats = app
        .clone()
        .oneshot(request("GET", "/v1/routing/stats", None)?)
        .await?;
    assert_eq!(stats.status(), StatusCode::OK);
    assert_eq!(
        json_body(stats).await?["models"]["served-model"]["calls"],
        1
    );

    let reset = app
        .clone()
        .oneshot(request("POST", "/v1/routing/stats/reset", None)?)
        .await?;
    assert_eq!(reset.status(), StatusCode::OK);
    assert_eq!(json_body(reset).await?, json!({"status": "reset"}));

    let after = app.oneshot(request("GET", "/v1/stats", None)?).await?;
    assert_eq!(json_body(after).await?["models"], json!({}));
    Ok(())
}

#[tokio::test]
async fn openai_streams_are_sse_framed_with_done() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::OpenAi,
            routing_metadata: None,
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "stream",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await?;
    assert!(body.contains("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}"));
    assert!(body.contains("data: [DONE]"));
    Ok(())
}

#[tokio::test]
async fn openai_streams_include_routing_metadata_headers() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::OpenAi,
            routing_metadata: Some(RoutingMetadata {
                selected_model: Some("served-model".to_string()),
                selected_tier: Some("efficient".to_string()),
                confidence: Some(0.0000004),
                router_version: Some("test-router:v1".to_string()),
                tolerance: Some(0.0000004),
                rationale: Some("line\nbreak".to_string()),
            }),
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "stream",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    for (name, expected) in [
        ("x-model-router-selected-model", "served-model"),
        ("x-model-router-confidence", "0.0000004"),
        ("x-model-router-rationale", "line break"),
    ] {
        assert_eq!(header(&response, name), Some(expected));
    }
    let body = text_body(response).await?;
    assert!(body.contains("data: [DONE]"));
    Ok(())
}

#[tokio::test]
async fn anthropic_streams_are_named_sse_without_done() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::Anthropic,
            routing_metadata: None,
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "stream",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await?;
    assert!(body.contains("event: message_start"));
    assert!(body.contains("\"type\":\"message_start\""));
    assert!(!body.contains("[DONE]"));
    Ok(())
}

#[tokio::test]
async fn endpoint_metadata_is_passed_to_profiles() -> TestResult {
    let captured = Arc::new(Mutex::new(None));
    let app = build_switchyard_router(state_from_profile(
        "capture",
        Arc::new(CaptureProfile {
            captured: Arc::clone(&captured),
        }),
    )?);

    let response = app
        .oneshot(request_with_headers(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "capture",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })),
            &[
                ("x-request-id", b"req-123"),
                ("x-switchyard-trace", b"trace-a"),
                (PROXY_SESSION_ID_HEADER, b" session-normal "),
                (PROXY_SESSION_ID_HEADER, b"session-normal"),
                (RELAY_SESSION_ID_HEADER, b"session-normal"),
            ],
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let input = captured
        .lock()
        .map_err(|_| "captured input mutex poisoned")?
        .clone()
        .ok_or("profile should have received input")?;
    assert_eq!(input.request.model(), Some("capture"));
    assert_eq!(
        input.metadata.request_id.as_ref().map(|id| id.as_str()),
        Some("req-123")
    );
    assert_eq!(
        input.metadata.inbound_format,
        Some(ChatRequestType::Anthropic)
    );
    assert_eq!(input.metadata.session_id.as_deref(), Some("session-normal"));
    assert_eq!(
        input
            .metadata
            .headers
            .get(PROXY_SESSION_ID_HEADER)
            .map(Vec::as_slice),
        Some(&[" session-normal ".to_string(), "session-normal".to_string()][..])
    );
    assert_eq!(
        input
            .metadata
            .headers
            .get("x-switchyard-trace")
            .map(Vec::as_slice),
        Some(&["trace-a".to_string()][..])
    );
    Ok(())
}

#[tokio::test]
async fn inference_endpoint_rejects_invalid_or_conflicting_session_headers() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);
    let rejected: Vec<Vec<(&'static str, &'static [u8])>> = vec![
        vec![
            (PROXY_SESSION_ID_HEADER, b"session-1"),
            (RELAY_SESSION_ID_HEADER, b"session-2"),
        ],
        vec![
            (RELAY_SESSION_ID_HEADER, b"session-1"),
            (RELAY_SESSION_ID_HEADER, b"session-2"),
        ],
        vec![(PROXY_SESSION_ID_HEADER, b"")],
        vec![(RELAY_SESSION_ID_HEADER, b"\xff")],
    ];

    for headers in rejected {
        let response = app
            .clone()
            .oneshot(request_with_headers(
                "POST",
                "/v1/chat/completions",
                Some(json!({
                    "model": "bench",
                    "messages": [{"role": "user", "content": "hi"}],
                })),
                &headers,
            )?)
            .await?;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "headers={headers:?}"
        );
    }
    Ok(())
}

/// Test profile that emits one deterministic stream event for SSE framing checks.
#[derive(Clone)]
struct StreamProfile {
    kind: StreamKind,
    routing_metadata: Option<RoutingMetadata>,
}

/// Stream format variant emitted by `StreamProfile`.
#[derive(Clone, Copy)]
enum StreamKind {
    OpenAi,
    Anthropic,
}

#[async_trait]
impl Profile for StreamProfile {
    async fn run(&self, _input: ProfileInput) -> Result<ProfileResponse> {
        let response = match self.kind {
            StreamKind::OpenAi => ChatResponse::OpenAiStream(Box::pin(futures_util::stream::iter(
                [Ok(StreamEvent::Json(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": "served-model",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "hello"},
                        "finish_reason": null,
                    }],
                })))],
            ))),
            StreamKind::Anthropic => ChatResponse::AnthropicStream(Box::pin(
                futures_util::stream::iter([Ok(StreamEvent::Json(json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-test",
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                })))]),
            )),
        };
        Ok(match &self.routing_metadata {
            Some(metadata) => ProfileResponse::with_routing_metadata(response, metadata.clone()),
            None => ProfileResponse::from(response),
        })
    }
}

struct CaptureProfile {
    captured: Arc<Mutex<Option<ProfileInput>>>,
}

struct BadTranslationProfile;

#[async_trait]
impl Profile for BadTranslationProfile {
    async fn run(&self, _input: ProfileInput) -> Result<ProfileResponse> {
        Ok(ProfileResponse::with_routing_metadata(
            ChatResponse::openai_completion(json!("not an OpenAI response object")),
            RoutingMetadata {
                selected_model: Some("bad-upstream".to_string()),
                selected_tier: Some("efficient".to_string()),
                router_version: Some("test-router:v1".to_string()),
                ..RoutingMetadata::default()
            },
        ))
    }
}

#[async_trait]
impl Profile for CaptureProfile {
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        {
            let mut captured = self
                .captured
                .lock()
                .map_err(|_| SwitchyardError::Other("captured input mutex poisoned".to_string()))?;
            *captured = Some(input);
        }
        Ok(ChatResponse::anthropic_completion(json!({
            "id": "msg-capture",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "captured"}],
            "model": "capture",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }))
        .into())
    }
}

/// In-process HTTP stub that records JSON request bodies from backend calls.
struct HttpStub {
    base_url: String,
    addr: SocketAddr,
    expected_requests: usize,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl HttpStub {
    /// Binds an ephemeral port and accepts the expected number of stub requests.
    fn start(expected_requests: usize) -> TestResult<Option<Self>> {
        Self::start_with_response(
            expected_requests,
            json!({
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": "stub",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "stub-ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }),
        )
    }

    /// Binds an ephemeral port and serves a caller-supplied JSON response.
    fn start_with_response(expected_requests: usize, response: Value) -> TestResult<Option<Self>> {
        let listener = match StdTcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == ErrorKind::PermissionDenied => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let addr = listener.local_addr()?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let Ok((mut stream, _addr)) = listener.accept() else {
                    return;
                };
                if let Ok(body) = read_http_body(&mut stream) {
                    if let Ok(value) = serde_json::from_slice::<Value>(&body) {
                        if let Ok(mut requests) = thread_requests.lock() {
                            requests.push(value);
                        }
                    }
                }
                let response = response.to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
            }
        });

        Ok(Some(Self {
            base_url: format!("http://{addr}/v1"),
            addr,
            expected_requests,
            requests,
            handle: Some(handle),
        }))
    }

    /// Returns the JSON request bodies captured by the stub thread.
    fn requests(&self) -> TestResult<Vec<Value>> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .map_err(|_| "stub request mutex poisoned".into())
    }
}

impl Drop for HttpStub {
    fn drop(&mut self) {
        // Wake pending accepts before joining the stub thread.
        for _ in 0..self.expected_requests {
            let _ = TcpStream::connect(self.addr);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_http_body(stream: &mut std::net::TcpStream) -> TestResult<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut header_end = None;
    while header_end.is_none() {
        let mut chunk = [0; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        header_end = find_bytes(&buffer, b"\r\n\r\n").map(|index| index + 4);
    }

    let Some(body_start) = header_end else {
        return Err("HTTP request headers were incomplete".into());
    };
    let headers = std::str::from_utf8(&buffer[..body_start])?;
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);

    while buffer.len() < body_start + content_length {
        let mut chunk = [0; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    Ok(buffer[body_start..body_start + content_length].to_vec())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn routing_request_json(
    profile_id: &str,
    request_materialization: &str,
    current_request: Option<Value>,
) -> Value {
    let mut request = json!({
        "schema_version": ROUTING_REQUEST_SCHEMA_VERSION,
        "decision_profile": {
            "profile_id": profile_id,
            "router": "legacy-relay-router",
            "request_materialization": request_materialization
        },
        "identity": {
            "session_id": "session-1",
            "request_id": "request-1",
            "harness": "server-test",
            "source": "nemo-relay",
            "quality": "native"
        },
        "protocol": {
            "inbound_profile": "openai_chat",
            "inbound_endpoint": "/v1/chat/completions",
            "desired_response_profile": "openai_chat"
        },
        "request_summary": {
            "client_requested_model": "client/model",
            "tool_count_in_payload": 0,
            "has_system_prompt": false
        },
        "attempt": {
            "routing_attempt": 1,
            "max_routing_attempts": 1
        }
    });
    if let Some(current_request) = current_request {
        request["current_request"] = current_request;
    }
    request
}

fn tool_event(
    uuid: &str,
    phase: &str,
    session_id: Option<&str>,
    owner_id: Option<&str>,
    data: Value,
) -> Value {
    let mut metadata = serde_json::Map::new();
    if let Some(session_id) = session_id {
        metadata.insert("hermes_session_id".to_string(), json!(session_id));
    }
    if let Some(owner_id) = owner_id {
        metadata.insert("switchyard_owner_id".to_string(), json!(owner_id));
    }
    json!({
        "kind": "scope",
        "uuid": uuid,
        "scope_category": phase,
        "name": "Bash",
        "category": "tool",
        "category_profile": {"tool_call_id": uuid},
        "data": data,
        "metadata": metadata,
    })
}

fn turn_event(uuid: &str, session_id: &str, owner_id: Option<&str>) -> Value {
    let mut metadata = serde_json::Map::from_iter([
        ("hermes_session_id".to_string(), json!(session_id)),
        ("nemo_relay_scope_role".to_string(), json!("turn")),
    ]);
    if let Some(owner_id) = owner_id {
        metadata.insert("switchyard_owner_id".to_string(), json!(owner_id));
    }
    json!({
        "kind": "scope",
        "uuid": uuid,
        "scope_category": "start",
        "name": "turn",
        "category": "agent",
        "metadata": metadata,
    })
}

fn stage_router_decision_yaml() -> &'static str {
    r#"
targets:
  capable:
    model: provider/capable
    format: openai
    base_url: http://127.0.0.1:1/v1
  efficient:
    model: provider/efficient
    format: openai
    base_url: http://127.0.0.1:1/v1
profiles:
  remote-stage_router:
    type: stage_router
    capable: capable
    efficient: efficient
    fallback_target_on_evict: capable
    picker: efficient_first
    confidence_threshold: 0.7
"#
}

fn random_decision_yaml() -> &'static str {
    r#"
targets:
  capable:
    model: provider/capable
    format: openai
    base_url: http://127.0.0.1:1/v1
  efficient:
    model: provider/efficient
    format: openai
    base_url: http://127.0.0.1:1/v1
profiles:
  remote-random:
    type: random-routing
    strong: capable
    weak: efficient
    strong_probability: 1.0
    rng_seed: 7
"#
}

fn state_from_yaml(input: &str) -> TestResult<ServerState> {
    let plan = parse_profile_config_str(input, ProfileConfigFormat::Yaml)?.resolve()?;
    Ok(ServerState::from_plan(&plan)?)
}

fn state_from_profile(model: &'static str, profile: Arc<dyn Profile>) -> TestResult<ServerState> {
    let registry = ProfileRegistry::from_profiles([(
        ModelId::from_static(model),
        profile,
        model.to_string(),
    )])?;
    Ok(ServerState::new(registry))
}

fn request(method: &str, uri: &str, body: Option<Value>) -> TestResult<Request<Body>> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    let body = body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    Ok(builder.body(body)?)
}

fn request_with_headers(
    method: &str,
    uri: &str,
    body: Option<Value>,
    headers: &[(&'static str, &[u8])],
) -> TestResult<Request<Body>> {
    let mut request = request(method, uri, body)?;
    for (name, value) in headers {
        request.headers_mut().append(
            HeaderName::from_static(name),
            HeaderValue::from_bytes(value)?,
        );
    }
    Ok(request)
}

fn raw_request(method: &str, uri: &str, body: &str) -> TestResult<Request<Body>> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    Ok(builder.body(Body::from(body.to_string()))?)
}

fn ndjson_request(body: &str, authorization: Option<&str>) -> TestResult<Request<Body>> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/atof/events")
        .header("content-type", "application/x-ndjson");
    if let Some(authorization) = authorization {
        builder = builder.header("authorization", authorization);
    }
    Ok(builder.body(Body::from(body.to_string()))?)
}

async fn json_body(response: axum::response::Response) -> TestResult<Value> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(serde_json::from_slice(&bytes)?)
}

async fn text_body(response: axum::response::Response) -> TestResult<String> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(String::from_utf8(bytes.to_vec())?)
}

fn header<'a>(response: &'a axum::response::Response, name: &str) -> Option<&'a str> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

fn reset_stats() -> Result<()> {
    profile_stats_accumulator().reset()
}

/// Serializes tests that touch the global profile stats accumulator.
///
/// The lock is initialized once for the test process and held by each caller
/// until the returned guard is dropped.
async fn stats_guard() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn log_loopback_bind_skip() {
    eprintln!("SKIP: permission denied binding loopback socket");
}

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
