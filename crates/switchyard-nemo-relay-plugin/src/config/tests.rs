// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{Value, json};

fn binding(protocol: WireFormat, model: &str) -> TargetBinding {
    TargetBinding {
        model: model.into(),
        protocol,
        endpoint: String::new(),
        base_url: "https://provider.example/v1".into(),
        weight: 1.0,
        drop_caller_extra_body: false,
        header_env: BTreeMap::new(),
        extra_body: BTreeMap::new(),
    }
}

fn config() -> SwitchyardConfig {
    SwitchyardConfig {
        version: 2,
        priority: 0,
        algorithm: AlgorithmConfig::Random { seed: Some(42) },
        targets: BTreeMap::from([
            (
                "chat".into(),
                binding(WireFormat::OpenAiChat, "provider/chat"),
            ),
            (
                "responses".into(),
                binding(WireFormat::OpenAiResponses, "provider/responses"),
            ),
            (
                "anthropic".into(),
                binding(WireFormat::AnthropicMessages, "provider/anthropic"),
            ),
        ]),
        default_targets: BTreeMap::from([
            (WireFormat::OpenAiChat, "chat".into()),
            (WireFormat::OpenAiResponses, "responses".into()),
            (WireFormat::AnthropicMessages, "anthropic".into()),
        ]),
    }
}

#[test]
fn version_two_random_configuration_builds_clients_without_a_service() {
    let config = config();
    config.validate().unwrap();
    let prepared = config.prepare().unwrap();
    assert_eq!(prepared.algorithm.name(), "random");
    assert_eq!(prepared.targets.len(), 3);
    assert!(
        prepared
            .targets
            .values()
            .all(|target| Arc::strong_count(&target.client) == 1)
    );
}

#[test]
fn target_endpoints_must_be_canonical_for_the_current_http_client() {
    let mut config = config();
    config.targets.get_mut("chat").unwrap().endpoint = "/custom/chat".into();
    let error = config.validate().unwrap_err();
    assert!(error.contains("ending in \"/chat/completions\""));

    config.targets.get_mut("chat").unwrap().endpoint = "/custom/chat/completions".into();
    config.validate().unwrap();
    assert_eq!(
        config.targets["chat"].dispatch_url(),
        "https://provider.example/v1/custom/chat/completions"
    );
}

#[test]
fn complete_provider_endpoint_is_not_appended_twice() {
    let mut config = config();
    let chat = config.targets.get_mut("chat").unwrap();
    chat.base_url = "https://provider.example/v1/chat/completions/".into();
    assert_eq!(
        chat.dispatch_url(),
        "https://provider.example/v1/chat/completions"
    );
    config.validate().unwrap();
}

#[test]
fn absolute_urls_cannot_embed_credentials_or_query_parameters() {
    let mut config = config();
    config.targets.get_mut("chat").unwrap().base_url =
        "https://user:password@provider.example/v1".into();
    assert!(
        config
            .validate()
            .unwrap_err()
            .contains("embedded credentials")
    );

    config.targets.get_mut("chat").unwrap().base_url =
        "https://provider.example/v1?api-version=1".into();
    assert!(config.validate().unwrap_err().contains("query parameters"));
}

#[test]
fn transport_owned_and_case_duplicate_environment_headers_are_rejected() {
    let mut host_header_config = config();
    let chat = host_header_config.targets.get_mut("chat").unwrap();
    chat.header_env.insert("Host".into(), "TARGET_HOST".into());
    assert!(
        host_header_config
            .validate()
            .unwrap_err()
            .contains("HTTP transport")
    );

    let mut duplicate_config = config();
    let chat = duplicate_config.targets.get_mut("chat").unwrap();
    chat.header_env
        .insert("X-Tenant".into(), "TARGET_TENANT_A".into());
    chat.header_env
        .insert("x-tenant".into(), "TARGET_TENANT_B".into());
    assert!(
        duplicate_config
            .validate()
            .unwrap_err()
            .contains("more than once")
    );
}

#[test]
fn only_canonical_relay_execution_names_resolve_protocols() {
    assert_eq!(
        protocol_from_call("openai.chat_completions"),
        Some(WireFormat::OpenAiChat)
    );
    assert_eq!(
        protocol_from_call("openai.responses"),
        Some(WireFormat::OpenAiResponses)
    );
    assert_eq!(
        protocol_from_call("anthropic.messages"),
        Some(WireFormat::AnthropicMessages)
    );
    assert_eq!(protocol_from_call("openai_chat"), None);
}

#[test]
fn schema_required_contract_fields_do_not_default_during_deserialization() {
    let base = json!({
        "version": 2,
        "algorithm": {"kind": "random"},
        "targets": {
            "chat": {
                "model": "provider/chat",
                "protocol": "openai_chat",
                "base_url": "https://provider.example/v1"
            }
        },
        "default_targets": {"openai_chat": "chat"}
    });
    for field in ["version", "algorithm", "default_targets"] {
        let mut value = base.clone();
        value.as_object_mut().unwrap().remove(field);
        let error = serde_json::from_value::<SwitchyardConfig>(value)
            .err()
            .expect("required field must not default");
        assert!(error.to_string().contains(field), "field={field}: {error}");
    }
}

#[test]
fn plugin_retry_budget_is_not_configurable() {
    let value = json!({
        "version": 2,
        "max_retries": 3,
        "algorithm": {"kind": "random"},
        "targets": {
            "chat": {
                "model": "provider/chat",
                "protocol": "openai_chat",
                "base_url": "https://provider.example/v1"
            }
        },
        "default_targets": {"openai_chat": "chat"}
    });

    let error = serde_json::from_value::<SwitchyardConfig>(value)
        .err()
        .expect("plugin retry budget must be rejected");
    assert!(error.to_string().contains("max_retries"));
}

#[test]
fn unknown_target_fields_are_rejected() {
    let value = json!({
        "version": 2,
        "algorithm": {"kind": "random"},
        "targets": {
            "chat": {
                "model": "provider/chat",
                "protocol": "openai_chat",
                "base_url": "https://provider.example/v1",
                "unexpected_setting": true
            }
        },
        "default_targets": {"openai_chat": "chat"}
    });
    let error = serde_json::from_value::<SwitchyardConfig>(value)
        .err()
        .expect("unknown target field must be rejected");
    assert!(error.to_string().contains("unexpected_setting"));
}

#[test]
fn literal_target_headers_are_rejected() {
    let value = json!({
        "version": 2,
        "algorithm": {"kind": "random"},
        "targets": {
            "chat": {
                "model": "provider/chat",
                "protocol": "openai_chat",
                "base_url": "https://provider.example/v1",
                "headers": {"x-provider-token": "plaintext-secret"}
            }
        },
        "default_targets": {"openai_chat": "chat"}
    });
    let error = serde_json::from_value::<SwitchyardConfig>(value)
        .err()
        .expect("literal target headers must be rejected")
        .to_string();
    assert!(error.contains("unknown field `headers`"));
    assert!(!error.contains("plaintext-secret"));
}

#[test]
fn unknown_algorithm_fields_are_rejected() {
    let error = serde_json::from_value::<AlgorithmConfig>(json!({
        "kind": "random",
        "seed": 42,
        "unexpected_setting": true
    }))
    .err()
    .expect("unknown algorithm field must be rejected");
    assert!(error.to_string().contains("unexpected_setting"));
}

#[test]
fn classifier_prepares_clients_for_judge_and_routed_targets() {
    let mut config = config();
    config.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic",
        "base_threshold": 0.5,
        "recent_turn_window": 4,
        "max_output_tokens": 512
    }))
    .unwrap();
    config.validate().unwrap();
    let prepared = config.prepare().unwrap();
    assert_eq!(prepared.algorithm.name(), "llm_task_classifier");
    assert!(
        prepared
            .targets
            .values()
            .all(|target| Arc::strong_count(&target.client) == 1)
    );
}

#[test]
fn target_provider_defaults_are_accepted_for_judge_controls() {
    let mut config = config();
    config.targets.get_mut("chat").unwrap().extra_body =
        BTreeMap::from([("think".into(), json!(false))]);

    config.validate().unwrap();
    config.prepare().unwrap();
}

#[test]
fn classifier_rejects_anthropic_judge_targets_before_dispatch() {
    let mut config = config();
    config.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "classifier_target": "anthropic",
        "weak_target": "responses",
        "strong_target": "chat",
        "base_threshold": 0.5
    }))
    .unwrap();

    let error = config.validate().unwrap_err();
    assert!(error.contains("classifier target \"anthropic\" uses anthropic_messages"));
}

#[test]
fn validation_does_not_resolve_environment_backed_headers() {
    let mut config = config();
    config.targets.get_mut("chat").unwrap().header_env = BTreeMap::from([(
        "authorization".into(),
        "SWITCHYARD_TEST_ENVIRONMENT_VARIABLE_THAT_IS_NOT_SET".into(),
    )]);

    config.validate().unwrap();
    let error = config
        .prepare()
        .err()
        .expect("preparation must resolve headers");
    assert!(error.contains("SWITCHYARD_TEST_ENVIRONMENT_VARIABLE_THAT_IS_NOT_SET"));
}

#[test]
fn invalid_environment_variable_names_are_rejected_before_resolution() {
    for variable in ["INVALID=VARIABLE", "INVALID\0VARIABLE"] {
        let mut config = config();
        config.targets.get_mut("chat").unwrap().header_env =
            BTreeMap::from([("authorization".into(), variable.into())]);

        let error = config.validate().unwrap_err();
        assert!(error.contains("must not contain '=' or NUL"));
    }
}

#[test]
fn static_validation_preserves_algorithm_constructor_checks() {
    let mut random = config();
    for target in random.targets.values_mut() {
        target.weight = 0.0;
    }
    assert!(
        random
            .validate()
            .unwrap_err()
            .contains("at least one positive target weight")
    );

    let mut classifier = config();
    classifier.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic",
        "base_threshold": 1.1
    }))
    .unwrap();
    assert!(
        classifier
            .validate()
            .unwrap_err()
            .contains("base_threshold must be between 0 and 1")
    );
}

#[test]
fn escalation_classifier_builds_with_defaulted_policy_settings() {
    let mut config = config();
    config.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "mode": "escalation",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic",
        "prompt": "Judge the completed trajectory.",
        "max_output_tokens": 256,
        "escalation": {}
    }))
    .unwrap();

    config.validate().unwrap();
    let prepared = config.prepare().unwrap();
    assert_eq!(prepared.algorithm.name(), "llm_task_classifier");
    assert!(
        prepared
            .targets
            .values()
            .all(|target| Arc::strong_count(&target.client) == 1)
    );
}

#[test]
fn classifier_modes_reject_mixed_or_missing_settings() {
    let mut capability = config();
    capability.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic",
        "base_threshold": 0.5,
        "escalation": {}
    }))
    .unwrap();
    assert!(
        capability
            .validate()
            .unwrap_err()
            .contains("capability mode does not accept escalation")
    );

    let mut escalation = config();
    escalation.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "mode": "escalation",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic",
        "base_threshold": 0.5,
        "escalation": {}
    }))
    .unwrap();
    assert!(
        escalation
            .validate()
            .unwrap_err()
            .contains("escalation mode does not accept capability")
    );

    let mut missing = config();
    missing.algorithm = serde_json::from_value(json!({
        "kind": "llm_classifier",
        "mode": "escalation",
        "classifier_target": "chat",
        "weak_target": "responses",
        "strong_target": "anthropic"
    }))
    .unwrap();
    assert!(
        missing
            .validate()
            .unwrap_err()
            .contains("requires escalation settings")
    );
}

#[test]
fn escalation_settings_are_validated_by_the_libsy_constructor() {
    for (settings, expected) in [
        (
            json!({"confirmations": 0}),
            "confirmations must be at least 1",
        ),
        (
            json!({"recent_turn_window": 0}),
            "recent_turn_window must be at least 1",
        ),
        (
            json!({"window_message_chars": 49}),
            "window_message_chars must be at least 50",
        ),
    ] {
        let mut config = config();
        config.algorithm = serde_json::from_value(json!({
            "kind": "llm_classifier",
            "mode": "escalation",
            "classifier_target": "chat",
            "weak_target": "responses",
            "strong_target": "anthropic",
            "escalation": settings
        }))
        .unwrap();
        assert!(config.validate().unwrap_err().contains(expected));
    }
}

#[test]
fn full_stage_router_configuration_builds_all_clients() {
    let mut config = config();
    config.algorithm = serde_json::from_value(json!({
        "kind": "stage_router",
        "capable_target": "anthropic",
        "efficient_target": "responses",
        "picker": "efficient_first",
        "confidence_threshold": 0.5,
        "recent_turn_window": 3,
        "capable_system_prompt": "Diagnose before editing.",
        "efficient_system_prompt": "Follow the settled plan.",
        "handoff_notes": {
            "escalation_note": "The previous model was stalling.",
            "deescalation_note": "The task is settled.",
            "only_on_wrong_signal_escalation": true
        },
        "classifier": {
            "target": "chat",
            "base_threshold": 0.5,
            "threshold_step": 0.1,
            "recent_turn_window": 3,
            "prompt": "Can the efficient tier finish this turn?",
            "max_output_tokens": 256
        }
    }))
    .unwrap();

    config.validate().unwrap();
    let prepared = config.prepare().unwrap();
    assert_eq!(prepared.algorithm.name(), "stage_router");
    assert!(
        prepared
            .targets
            .values()
            .all(|target| Arc::strong_count(&target.client) == 1)
    );
}

#[test]
fn stage_router_validates_threshold_targets_and_judge_protocol() {
    let stage = |classifier: Value, threshold: f64| {
        serde_json::from_value(json!({
            "kind": "stage_router",
            "capable_target": "anthropic",
            "efficient_target": "responses",
            "picker": "capable_first",
            "confidence_threshold": threshold,
            "classifier": classifier
        }))
        .unwrap()
    };

    let mut invalid_threshold = config();
    invalid_threshold.algorithm = stage(Value::Null, 1.1);
    assert!(
        invalid_threshold
            .validate()
            .unwrap_err()
            .contains("confidence_threshold must be between 0 and 1")
    );

    let mut missing_target = config();
    missing_target.algorithm = serde_json::from_value(json!({
        "kind": "stage_router",
        "capable_target": "missing",
        "efficient_target": "responses",
        "picker": "capable_first",
        "confidence_threshold": 0.5
    }))
    .unwrap();
    assert!(
        missing_target
            .validate()
            .unwrap_err()
            .contains("algorithm target \"missing\" is not configured")
    );

    let mut anthropic_judge = config();
    anthropic_judge.algorithm = stage(json!({"target": "anthropic", "base_threshold": 0.5}), 0.5);
    assert!(
        anthropic_judge
            .validate()
            .unwrap_err()
            .contains("classifier target \"anthropic\" uses anthropic_messages")
    );
}

#[test]
fn zero_weight_random_targets_are_fallback_only() {
    let mut config = config();
    config.targets.get_mut("anthropic").unwrap().weight = 0.0;
    let prepared = config.prepare().unwrap();

    assert_eq!(Arc::strong_count(&prepared.targets["anthropic"].client), 1);
    assert_eq!(Arc::strong_count(&prepared.targets["chat"].client), 1);
    assert_eq!(Arc::strong_count(&prepared.targets["responses"].client), 1);
}
