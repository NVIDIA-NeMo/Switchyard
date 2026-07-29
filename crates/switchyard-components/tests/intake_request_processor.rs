// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use std::collections::BTreeMap;

use serde_json::json;
use switchyard_components::{ChatRequest, ChatRequestType, ProxyContext, Result, SwitchyardError};
use switchyard_components::{
    IntakeRequestMetadata, IntakeRequestProcessor, IntakeRequestState, RequestMetadata,
};

use support::intake::request;

#[test]
fn request_metadata_extracts_only_explicit_intake_fields_case_insensitively() -> Result<()> {
    let headers = BTreeMap::from([
        ("Proxy_X_Session_ID".to_string(), "sess-1".to_string()),
        (
            "X-Switchyard-Intake-Enabled".to_string(),
            "true".to_string(),
        ),
        (
            "x-switchyard-intake-app".to_string(),
            "log2/codex".to_string(),
        ),
        (
            "x-switchyard-intake-task".to_string(),
            "developer-session".to_string(),
        ),
        ("authorization".to_string(), "Bearer secret".to_string()),
    ]);

    let metadata = RequestMetadata::from_headers(&headers);

    assert_eq!(metadata.session_id.as_deref(), Some("sess-1"));
    assert_eq!(metadata.intake.enabled, Some(true));
    assert_eq!(metadata.intake.app.as_deref(), Some("log2/codex"));
    assert_eq!(metadata.intake.task.as_deref(), Some("developer-session"));
    Ok(())
}

#[tokio::test]
async fn request_processor_skips_by_default_and_stamps_session() -> Result<()> {
    let processor = IntakeRequestProcessor;
    let mut ctx = ProxyContext::new();
    ctx.insert(RequestMetadata {
        session_id: Some("session-123".to_string()),
        ..RequestMetadata::default()
    });

    let processed = processor.process(&mut ctx, request()).await?;

    assert_eq!(processed.model(), Some("gpt-4o"));
    let state = ctx
        .get::<IntakeRequestState>()
        .ok_or_else(|| SwitchyardError::Other("intake state missing".to_string()))?;
    assert_eq!(state.session_id.as_deref(), Some("session-123"));
    assert_eq!(state.inbound_format, ChatRequestType::OpenAiChat);
    assert!(state.skip);
    assert!(state.request_snapshot.is_none());
    Ok(())
}

#[tokio::test]
async fn request_processor_store_true_opts_in_when_header_absent() -> Result<()> {
    let processor = IntakeRequestProcessor;
    let mut ctx = ProxyContext::new();
    let original = ChatRequest::openai_chat(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "store": true
    }));

    let processed = processor.process(&mut ctx, original.clone()).await?;
    let state = ctx
        .get::<IntakeRequestState>()
        .ok_or_else(|| SwitchyardError::Other("intake state missing".to_string()))?;

    assert_eq!(processed, original);
    assert!(!state.skip);
    assert_eq!(state.request_snapshot.as_ref(), Some(&original));
    Ok(())
}

#[tokio::test]
async fn request_processor_header_overrides_store_toggle() -> Result<()> {
    let processor = IntakeRequestProcessor;
    let mut ctx = ProxyContext::new();
    ctx.insert(RequestMetadata {
        intake: IntakeRequestMetadata {
            enabled: Some(false),
            ..IntakeRequestMetadata::default()
        },
        ..RequestMetadata::default()
    });
    let request = ChatRequest::openai_chat(json!({
        "model": "gpt-4o",
        "messages": [],
        "store": true
    }));

    processor.process(&mut ctx, request).await?;

    let state = ctx
        .get::<IntakeRequestState>()
        .ok_or_else(|| SwitchyardError::Other("intake state missing".to_string()))?;
    assert!(state.skip);
    assert!(state.request_snapshot.is_none());
    Ok(())
}

#[tokio::test]
async fn request_processor_header_true_opts_in_without_store_toggle() -> Result<()> {
    let processor = IntakeRequestProcessor;
    let mut ctx = ProxyContext::new();
    ctx.insert(RequestMetadata {
        intake: IntakeRequestMetadata {
            enabled: Some(true),
            ..IntakeRequestMetadata::default()
        },
        ..RequestMetadata::default()
    });
    let original = request();

    let processed = processor.process(&mut ctx, original.clone()).await?;

    let state = ctx
        .get::<IntakeRequestState>()
        .ok_or_else(|| SwitchyardError::Other("intake state missing".to_string()))?;
    assert_eq!(processed, original);
    assert!(!state.skip);
    assert_eq!(state.request_snapshot.as_ref(), Some(&original));
    Ok(())
}

#[tokio::test]
async fn request_processor_store_false_skips_when_header_absent() -> Result<()> {
    let processor = IntakeRequestProcessor;
    let mut ctx = ProxyContext::new();
    let request = ChatRequest::openai_chat(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "store": false
    }));

    processor.process(&mut ctx, request).await?;

    let state = ctx
        .get::<IntakeRequestState>()
        .ok_or_else(|| SwitchyardError::Other("intake state missing".to_string()))?;
    assert!(state.skip);
    assert!(state.request_snapshot.is_none());
    Ok(())
}
