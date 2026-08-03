// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial tests for the random routing engine.

use serde_json::json;
use switchyard_components::{
    BackendFormat, ChatRequest, EndpointConfig, LlmTarget, LlmTargetId, ModelId, Result,
    SwitchyardError,
};
use switchyard_components::{RandomRoutingEngine, RandomRoutingProcessorConfig, RandomRoutingTier};

// Builds a deterministic strong/weak config for routing tests.
fn config(strong_probability: f64, rng_seed: u64) -> Result<RandomRoutingProcessorConfig> {
    Ok(RandomRoutingProcessorConfig::new(
        LlmTarget::new(
            LlmTargetId::from_static("strong-target"),
            ModelId::from_static("strong-model"),
        ),
        LlmTarget::new(
            LlmTargetId::from_static("weak-target"),
            ModelId::from_static("weak-model"),
        ),
    )
    .with_strong_probability(strong_probability)?
    .with_rng_seed(Some(rng_seed)))
}

// Builds an OpenAI Chat request whose non-model fields must be preserved.
fn openai_request(model: &str) -> ChatRequest {
    ChatRequest::openai_chat(json!({
        "model": model,
        "messages": [{"role": "user", "content": "keep me"}],
        "temperature": 0.3,
    }))
}

// Runs one request through the engine and applies the route-local model rewrite.
fn route_once(engine: &RandomRoutingEngine, mut request: ChatRequest) -> Result<ChatRequest> {
    let decision = engine.select(request.model().map(str::to_owned))?;
    request.set_model(decision.selected_model.as_str());
    Ok(request)
}

// Probability zero is a hard weak route and should preserve request payload fields.
#[test]
fn probability_zero_always_routes_to_weak_and_preserves_body_fields() -> Result<()> {
    let engine = RandomRoutingEngine::new(config(0.0, 7)?)?;
    let request = route_once(&engine, openai_request("client-model"))?;
    let decision = engine.select(Some("client-model".to_string()))?;

    assert_eq!(request.model(), Some("weak-model"));
    assert_eq!(request.body()["messages"][0]["content"], "keep me");
    assert_eq!(request.body()["temperature"], 0.3);
    assert_eq!(decision.tier, RandomRoutingTier::Weak);
    assert_eq!(decision.selected_model, ModelId::from_static("weak-model"));
    assert_eq!(decision.original_model.as_deref(), Some("client-model"));
    assert_eq!(decision.strong_probability, 0.0);
    assert!((0.0..1.0).contains(&decision.draw));
    Ok(())
}

// Probability one is a hard strong route.
#[test]
fn probability_one_always_routes_to_strong() -> Result<()> {
    let engine = RandomRoutingEngine::new(config(1.0, 7)?)?;
    let request = route_once(&engine, openai_request("client-model"))?;
    let decision = engine.select(Some("client-model".to_string()))?;

    assert_eq!(request.model(), Some("strong-model"));
    assert_eq!(
        decision.selected_target,
        LlmTargetId::from_static("strong-target")
    );
    assert_eq!(decision.tier, RandomRoutingTier::Strong);
    Ok(())
}

// Equal seeds should produce identical routing sequences across processors.
#[test]
fn seeded_engines_produce_the_same_routing_sequence() -> Result<()> {
    let left = RandomRoutingEngine::new(config(0.5, 42)?)?;
    let right = RandomRoutingEngine::new(config(0.5, 42)?)?;

    let mut left_sequence = Vec::new();
    let mut right_sequence = Vec::new();
    for _ in 0..32 {
        left_sequence.push(left.select(Some("client-model".to_string()))?.tier);
        right_sequence.push(right.select(Some("client-model".to_string()))?.tier);
    }

    assert_eq!(left_sequence, right_sequence);
    assert!(left_sequence.contains(&RandomRoutingTier::Strong));
    assert!(left_sequence.contains(&RandomRoutingTier::Weak));
    Ok(())
}

// Malformed request bodies should be repaired into an object with the selected model.
#[test]
fn malformed_non_object_request_body_is_recovered_with_selected_model() -> Result<()> {
    let engine = RandomRoutingEngine::new(config(1.0, 11)?)?;
    let request = route_once(&engine, ChatRequest::openai_chat(json!("bad")))?;

    assert_eq!(request.body(), &json!({"model": "strong-model"}));
    Ok(())
}

// Invalid probabilities must fail during engine construction, not at call time.
#[test]
fn invalid_probability_is_rejected_before_routing_requests() -> Result<()> {
    for value in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        let Err(error) = RandomRoutingEngine::new(RandomRoutingProcessorConfig {
            strong: LlmTarget::new(
                LlmTargetId::from_static("strong"),
                ModelId::from_static("strong-model"),
            ),
            weak: LlmTarget::new(
                LlmTargetId::from_static("weak"),
                ModelId::from_static("weak-model"),
            ),
            strong_probability: value,
            rng_seed: Some(1),
        }) else {
            return Err(SwitchyardError::Other(
                "invalid probability should be rejected".to_string(),
            ));
        };

        assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    }
    Ok(())
}

// LLM target serde remains compatible with Python-authored configs.
#[test]
fn llm_target_format_wire_values_match_python_config_contract() -> Result<()> {
    assert_eq!(
        serde_json::to_value(BackendFormat::Auto)
            .map_err(|error| SwitchyardError::Other(error.to_string()))?,
        json!("auto")
    );
    assert_eq!(
        serde_json::to_value(BackendFormat::OpenAi)
            .map_err(|error| SwitchyardError::Other(error.to_string()))?,
        json!("openai")
    );
    assert_eq!(
        serde_json::to_value(BackendFormat::Responses)
            .map_err(|error| SwitchyardError::Other(error.to_string()))?,
        json!("responses")
    );
    assert_eq!(
        serde_json::to_value(BackendFormat::Anthropic)
            .map_err(|error| SwitchyardError::Other(error.to_string()))?,
        json!("anthropic")
    );
    assert!(serde_json::from_value::<BackendFormat>(json!("unknown")).is_err());

    let minimal = LlmTarget::new(
        LlmTargetId::from_static("minimal"),
        ModelId::from_static("model"),
    );
    assert_eq!(minimal.format, BackendFormat::Auto);
    assert_eq!(minimal.endpoint, EndpointConfig::default());

    let explicit: LlmTarget = serde_json::from_value(json!({
        "id": "explicit",
        "model": "model",
        "format": "openai",
        "endpoint": {
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "timeout_secs": 2.5
        }
    }))
    .map_err(|error| SwitchyardError::Other(error.to_string()))?;
    assert_eq!(explicit.format, BackendFormat::OpenAi);
    assert_eq!(
        explicit.endpoint.base_url.as_deref(),
        Some("https://example.test/v1")
    );
    assert_eq!(explicit.endpoint.api_key.as_deref(), Some("secret"));
    assert_eq!(explicit.endpoint.timeout_secs, Some(2.5));
    Ok(())
}

// Random routing config defaults remain stable and accept inclusive boundaries.
#[test]
fn random_routing_config_defaults_and_accepts_boundary_probabilities() -> Result<()> {
    let base = RandomRoutingProcessorConfig::new(
        LlmTarget::new(
            LlmTargetId::from_static("strong"),
            ModelId::from_static("strong-model"),
        ),
        LlmTarget::new(
            LlmTargetId::from_static("weak"),
            ModelId::from_static("weak-model"),
        ),
    );

    assert_eq!(base.strong_probability, 0.5);
    assert_eq!(base.rng_seed, None);

    for probability in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let configured = base.clone().with_strong_probability(probability)?;
        assert_eq!(configured.strong_probability, probability);
    }
    Ok(())
}
