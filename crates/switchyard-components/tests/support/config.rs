// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers that do not depend on the removed config graph.

use std::collections::BTreeMap;

use switchyard_components::{
    BackendFormat, EndpointConfig, LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError,
};

/// Test result alias that works across async and blocking integration tests.
pub type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

/// Asserts Switchyard validation fails with a stable, useful error fragment.
pub fn assert_invalid<T>(result: Result<T>, message: &'static str, expected: &str) -> TestResult {
    let error = error_from(result, message);
    assert!(
        error.to_string().contains(expected),
        "expected {error:?} to contain {expected:?}"
    );
    Ok(())
}

/// Builds an OpenAI-compatible runtime target for component tests.
pub fn openai_target(id: &'static str, model: &'static str, base_url: &str) -> Result<LlmTarget> {
    Ok(LlmTarget {
        id: LlmTargetId::from_static(id),
        model: ModelId::from_static(model),
        format: BackendFormat::OpenAi,
        endpoint: EndpointConfig {
            base_url: Some(base_url.to_string()),
            api_key: Some("test-key".to_string()),
            timeout_secs: Some(5.0),
        },
        extra_body: None,
        extra_headers: BTreeMap::new(),
    })
}

/// Builds an Anthropic-compatible runtime target for component tests.
pub fn anthropic_target(
    id: &'static str,
    model: &'static str,
    base_url: &str,
) -> Result<LlmTarget> {
    Ok(LlmTarget {
        id: LlmTargetId::from_static(id),
        model: ModelId::from_static(model),
        format: BackendFormat::Anthropic,
        endpoint: EndpointConfig {
            base_url: Some(base_url.to_string()),
            api_key: Some("test-key".to_string()),
            timeout_secs: Some(5.0),
        },
        extra_body: None,
        extra_headers: BTreeMap::new(),
    })
}

// Extracts the error from a result that is expected to fail.
fn error_from<T>(result: Result<T>, message: &'static str) -> SwitchyardError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}
