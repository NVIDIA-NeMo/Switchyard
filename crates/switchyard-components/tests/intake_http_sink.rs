// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use serde_json::json;
use switchyard_components::{
    HttpIntakeSink, IntakeFormat, IntakeSink, IntakeSinkConfig, IntakeTarget,
};
use switchyard_core::{Result, SwitchyardError};

use support::{OneShotServer, SequenceServer};

#[tokio::test]
async fn http_intake_sink_posts_to_workspace_chat_completions_path_with_bearer_auth() -> Result<()>
{
    let server = OneShotServer::json(200, json!({"ok": true}))?;
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        intake_base_url: Some(server.base_url().to_string()),
        workspace: Some("team space".to_string()),
        api_key: Some("secret-token".to_string()),
        ..IntakeSinkConfig::default()
    })?;

    sink.enqueue(json!({"request": {"ok": true}, "response": {"choices": []}}))
        .await?;
    sink.shutdown().await?;
    let request = server.captured()?;

    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/apis/intake/v2/workspaces/team%20space/ingest/chat-completions"
    );
    assert_eq!(request.header("authorization"), Some("Bearer secret-token"));
    assert_eq!(request.body["request"]["ok"], true);
    Ok(())
}

#[test]
fn intake_sink_config_debug_redacts_api_key() -> Result<()> {
    let config = IntakeSinkConfig {
        api_key: Some("secret-token".to_string()),
        ..IntakeSinkConfig::default()
    };

    let debug = format!("{config:?}");

    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("secret-token"));
    Ok(())
}

#[test]
fn http_intake_sink_rejects_missing_base_url_at_construction() -> Result<()> {
    let error = HttpIntakeSink::new(IntakeSinkConfig::default())
        .err()
        .ok_or_else(|| SwitchyardError::Other("missing base URL should fail".to_string()))?;

    match error {
        SwitchyardError::InvalidConfig(message) => {
            assert!(message.contains("intake_base_url is required"));
        }
        other => {
            return Err(SwitchyardError::Other(format!(
                "expected invalid config error, got {other:?}"
            )));
        }
    }
    Ok(())
}

#[tokio::test]
async fn http_intake_sink_error_status_is_fail_open() -> Result<()> {
    let server = OneShotServer::json(503, json!({"error": "down"}))?;
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        intake_base_url: Some(server.base_url().to_string()),
        workspace: Some("default".to_string()),
        ..IntakeSinkConfig::default()
    })?;

    sink.enqueue(json!({"request": {"ok": true}, "response": {"choices": []}}))
        .await?;
    sink.shutdown().await?;
    let request = server.captured()?;

    assert_eq!(request.method, "POST");
    assert_eq!(request.body["request"]["ok"], true);
    Ok(())
}

#[tokio::test]
async fn http_intake_sink_retries_transient_status_failures() -> Result<()> {
    let server = SequenceServer::json(vec![
        (503, json!({"error": "try again"})),
        (200, json!({"ok": true})),
    ])?;
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        intake_base_url: Some(server.base_url().to_string()),
        max_retries: 1,
        ..IntakeSinkConfig::default()
    })?;

    sink.enqueue(json!({"request": {"retry": true}, "response": {"choices": []}}))
        .await?;
    sink.shutdown().await?;
    let requests = server.captured()?;

    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].body["request"]["retry"], true);
    assert_eq!(requests[1].body["request"]["retry"], true);
    Ok(())
}

#[test]
fn http_intake_sink_rejects_non_positive_timeout() -> Result<()> {
    let error = HttpIntakeSink::new(IntakeSinkConfig {
        intake_base_url: Some("http://localhost:1".to_string()),
        request_timeout_s: 0.0,
        ..IntakeSinkConfig::default()
    })
    .err()
    .ok_or_else(|| SwitchyardError::Other("zero timeout should fail".to_string()))?;

    match error {
        SwitchyardError::InvalidConfig(message) => {
            assert!(message.contains("request_timeout_s must be finite and positive"));
        }
        other => {
            return Err(SwitchyardError::Other(format!(
                "expected invalid config error, got {other:?}"
            )));
        }
    }
    Ok(())
}

#[test]
fn http_intake_sink_rejects_zero_queue_size() -> Result<()> {
    let error = HttpIntakeSink::new(IntakeSinkConfig {
        intake_base_url: Some("http://localhost:1".to_string()),
        max_queue_size: 0,
        ..IntakeSinkConfig::default()
    })
    .err()
    .ok_or_else(|| SwitchyardError::Other("zero queue size should fail".to_string()))?;

    match error {
        SwitchyardError::InvalidConfig(message) => {
            assert!(message.contains("max_queue_size must be positive"));
        }
        other => {
            return Err(SwitchyardError::Other(format!(
                "expected invalid config error, got {other:?}"
            )));
        }
    }
    Ok(())
}

// An unauthenticated flat-document target posts to its full URL and must not
// send the bearer token even when an api_key is configured.
#[tokio::test]
async fn http_intake_sink_posts_to_flat_target_url_without_auth() -> Result<()> {
    let server = OneShotServer::json(201, json!({"status": "Created"}))?;
    let sink = HttpIntakeSink::new(IntakeSinkConfig {
        target: Some(IntakeTarget {
            url: format!("{}/lake/switchyard/posting", server.base_url()),
            format: IntakeFormat::FlatDocument,
            authenticated: false,
        }),
        // Set but must not be sent for an unauthenticated target.
        api_key: Some("secret-token".to_string()),
        ..IntakeSinkConfig::default()
    })?;

    sink.enqueue(json!({"_id": "x", "s_source": "switchyard", "l_switchyard_input_tokens": 19}))
        .await?;
    sink.shutdown().await?;
    let request = server.captured()?;

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/lake/switchyard/posting");
    assert_eq!(request.header("authorization"), None);
    assert_eq!(request.body["s_source"], "switchyard");
    assert_eq!(request.body["l_switchyard_input_tokens"], 19);
    Ok(())
}
