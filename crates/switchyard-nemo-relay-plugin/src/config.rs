// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use http::header::{HeaderName, HeaderValue};
use serde::Deserialize;
use switchyard_libsy::{
    Algorithm, LlmTarget, LlmTargetSet, LlmTaskClassifier, Random, TaskClassifierConfig,
};
use switchyard_protocol::WireFormat;

pub(crate) fn protocol_from_call(name: &str) -> Option<WireFormat> {
    match name {
        "openai.chat_completions" => Some(WireFormat::OpenAiChat),
        "openai.responses" => Some(WireFormat::OpenAiResponses),
        "anthropic.messages" => Some(WireFormat::AnthropicMessages),
        _ => None,
    }
}

const fn default_endpoint(protocol: WireFormat) -> &'static str {
    match protocol {
        WireFormat::OpenAiChat => "/v1/chat/completions",
        WireFormat::OpenAiResponses => "/v1/responses",
        WireFormat::AnthropicMessages => "/v1/messages",
    }
}

#[derive(Deserialize)]
struct TargetBinding {
    model: String,
    protocol: WireFormat,
    #[serde(default)]
    endpoint: String,
    base_url: String,
    #[serde(default = "default_weight")]
    weight: f64,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    header_env: BTreeMap<String, String>,
}

impl TargetBinding {
    fn dispatch_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        let endpoint = if self.endpoint.is_empty() {
            default_endpoint(self.protocol)
        } else {
            &self.endpoint
        };
        let endpoint = if base.ends_with("/v1") && endpoint.starts_with("/v1/") {
            &endpoint[3..]
        } else {
            endpoint
        };
        format!("{base}{endpoint}")
    }

    fn validate_headers(&self) -> Result<(), String> {
        for (name, value) in &self.headers {
            validate_header(name, value)?;
        }
        for (name, variable) in &self.header_env {
            validate_header_name(name)?;
            if self
                .headers
                .keys()
                .any(|configured| configured.eq_ignore_ascii_case(name))
            {
                return Err(format!(
                    "target header {name:?} cannot appear in both headers and header_env"
                ));
            }
            if variable.is_empty() {
                return Err(format!(
                    "environment variable name for target header {name:?} must not be empty"
                ));
            }
        }
        Ok(())
    }

    fn into_prepared(self) -> Result<PreparedTargetBinding, String> {
        let dispatch_url = self.dispatch_url();
        let mut headers = self.headers;
        for (name, variable) in self.header_env {
            let value = std::env::var(&variable)
                .map_err(|_| format!("environment variable {variable:?} is not set"))?;
            validate_header(&name, &value)?;
            headers.insert(name, value);
        }
        Ok(PreparedTargetBinding {
            model: self.model,
            protocol: self.protocol,
            dispatch_url,
            headers,
        })
    }
}

pub(crate) struct PreparedTargetBinding {
    pub(crate) model: String,
    pub(crate) protocol: WireFormat,
    dispatch_url: String,
    pub(crate) headers: BTreeMap<String, String>,
}

impl PreparedTargetBinding {
    pub(crate) fn dispatch_url(&self) -> &str {
        &self.dispatch_url
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AlgorithmConfig {
    Random {
        #[serde(default)]
        seed: Option<u64>,
    },
    LlmClassifier {
        classifier_target: String,
        weak_target: String,
        strong_target: String,
        base_threshold: f64,
        #[serde(default)]
        min_confidence: f64,
        #[serde(default)]
        capability_elevated_floor: Option<f64>,
        #[serde(default)]
        session_affinity: bool,
        #[serde(default)]
        message_hash_fallback: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwitchyardConfig {
    version: u32,
    #[serde(default)]
    pub(crate) priority: i32,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
    algorithm: AlgorithmConfig,
    targets: BTreeMap<String, TargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
}

pub(crate) struct PreparedConfig {
    pub(crate) max_retries: u32,
    pub(crate) algorithm: Arc<dyn Algorithm>,
    pub(crate) targets: BTreeMap<String, PreparedTargetBinding>,
    pub(crate) default_targets: BTreeMap<WireFormat, String>,
}

impl SwitchyardConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.validate_structure()?;
        self.build_algorithm().map(drop)
    }

    fn validate_structure(&self) -> Result<(), String> {
        if self.version != 2 {
            return Err(format!(
                "unsupported Switchyard config version {}; version 1 used switchyard-server; migrate to version = 2",
                self.version
            ));
        }
        if self.max_retries > 10 {
            return Err("max_retries must not exceed 10".into());
        }
        if self.targets.is_empty() {
            return Err("targets must not be empty".into());
        }
        if self.default_targets.is_empty() {
            return Err("default_targets must not be empty".into());
        }
        for (name, target) in &self.targets {
            if name.trim().is_empty() || target.model.trim().is_empty() {
                return Err("target names and models must be non-empty".into());
            }
            if !target.base_url.starts_with("http://") && !target.base_url.starts_with("https://") {
                return Err(format!("target {name:?} base_url must use http or https"));
            }
            if !target.weight.is_finite() || target.weight < 0.0 {
                return Err(format!(
                    "target {name:?} weight must be finite and nonnegative"
                ));
            }
            target.validate_headers()?;
        }
        for (protocol, fallback) in &self.default_targets {
            let target = self
                .targets
                .get(fallback)
                .ok_or_else(|| format!("default target {fallback:?} is not configured"))?;
            if target.protocol != *protocol {
                return Err(format!(
                    "default target {fallback:?} must use protocol {}",
                    protocol.as_str()
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn prepare(self) -> Result<PreparedConfig, String> {
        self.validate_structure()?;
        let algorithm = self.build_algorithm()?;
        let targets = self
            .targets
            .into_iter()
            .map(|(name, target)| target.into_prepared().map(|prepared| (name, prepared)))
            .collect::<Result<_, _>>()?;
        Ok(PreparedConfig {
            max_retries: self.max_retries,
            algorithm,
            targets,
            default_targets: self.default_targets,
        })
    }

    fn build_algorithm(&self) -> Result<Arc<dyn Algorithm>, String> {
        let target = |name: &str| {
            self.targets
                .contains_key(name)
                .then(|| LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: None,
                })
                .ok_or_else(|| format!("algorithm target {name:?} is not configured"))
        };
        match &self.algorithm {
            AlgorithmConfig::Random { seed } => {
                let targets = self
                    .targets
                    .keys()
                    .map(|name| LlmTarget {
                        semantic_name: name.clone(),
                        llm_client: None,
                    })
                    .collect();
                let weights = self
                    .targets
                    .values()
                    .map(|target| target.weight)
                    .collect::<Vec<_>>();
                Random::new(LlmTargetSet::new(targets), Some(weights), *seed)
                    .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                    .map_err(|error| error.to_string())
            }
            AlgorithmConfig::LlmClassifier {
                classifier_target,
                weak_target,
                strong_target,
                base_threshold,
                min_confidence,
                capability_elevated_floor,
                session_affinity,
                message_hash_fallback,
            } => LlmTaskClassifier::new(
                target(classifier_target)?,
                target(weak_target)?,
                target(strong_target)?,
                TaskClassifierConfig {
                    base_threshold: *base_threshold,
                    min_confidence: *min_confidence,
                    capability_elevated_floor: *capability_elevated_floor,
                    session_affinity: *session_affinity,
                    message_hash_fallback: *message_hash_fallback,
                    recent_turn_window: None,
                },
            )
            .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
            .map_err(|error| error.to_string()),
        }
    }
}

fn validate_header_name(name: &str) -> Result<(), String> {
    HeaderName::from_bytes(name.as_bytes())
        .map(|_| ())
        .map_err(|error| format!("invalid target header name {name:?}: {error}"))
}

fn validate_header(name: &str, value: &str) -> Result<(), String> {
    validate_header_name(name)?;
    HeaderValue::from_str(value)
        .map_err(|error| format!("invalid target header value for {name:?}: {error}"))?;
    Ok(())
}

const fn default_max_retries() -> u32 {
    3
}

const fn default_weight() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn binding(protocol: WireFormat, model: &str) -> TargetBinding {
        TargetBinding {
            model: model.into(),
            protocol,
            endpoint: String::new(),
            base_url: "https://provider.example/v1".into(),
            weight: 1.0,
            headers: BTreeMap::new(),
            header_env: BTreeMap::new(),
        }
    }

    fn config() -> SwitchyardConfig {
        SwitchyardConfig {
            version: 2,
            priority: 0,
            max_retries: 3,
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
    fn version_two_random_configuration_builds_without_a_service() {
        let config = config();
        config.validate().unwrap();
        assert_eq!(
            config.targets["chat"].dispatch_url(),
            "https://provider.example/v1/chat/completions"
        );
        assert_eq!(config.prepare().unwrap().algorithm.name(), "random");
    }

    #[test]
    fn version_one_reports_the_service_to_library_migration() {
        let mut config = config();
        config.version = 1;
        let error = config.validate().unwrap_err();
        assert!(error.contains("version 1 used switchyard-server"));
        assert!(error.contains("version = 2"));
    }

    #[test]
    fn default_target_keys_define_the_managed_protocols() {
        let mut config = config();
        config
            .default_targets
            .retain(|protocol, _| *protocol == WireFormat::OpenAiChat);
        config.validate().unwrap();
        assert_eq!(config.default_targets.len(), 1);
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
        for alias in [
            "openai_chat",
            "openai_chat_completions",
            "openai_responses",
            "anthropic",
            "anthropic_messages",
        ] {
            assert_eq!(protocol_from_call(alias), None, "alias={alias}");
        }
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
    fn removed_enabled_profile_list_is_not_silently_ignored() {
        let value = json!({
            "version": 2,
            "algorithm": {"kind": "random"},
            "targets": {
                "chat": {
                    "model": "provider/chat",
                    "protocol": "openai_chat",
                    "base_url": "https://provider.example/v1"
                }
            },
            "default_targets": {"openai_chat": "chat"},
            "enabled_inbound_profiles": ["openai_chat"]
        });
        let error = serde_json::from_value::<SwitchyardConfig>(value)
            .err()
            .expect("removed field must produce a migration error");
        assert!(error.to_string().contains("enabled_inbound_profiles"));
    }

    #[test]
    fn classifier_targets_are_semantic_names_not_provider_models() {
        let mut config = config();
        config.algorithm = AlgorithmConfig::LlmClassifier {
            classifier_target: "chat".into(),
            weak_target: "responses".into(),
            strong_target: "anthropic".into(),
            base_threshold: 0.5,
            min_confidence: 0.0,
            capability_elevated_floor: None,
            session_affinity: false,
            message_hash_fallback: false,
        };
        config.validate().unwrap();
        assert_eq!(
            config.prepare().unwrap().algorithm.name(),
            "llm_task_classifier"
        );
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
    fn static_validation_preserves_algorithm_constructor_checks() {
        let mut random = config();
        for target in random.targets.values_mut() {
            target.weight = 0.0;
        }
        assert!(random
            .validate()
            .unwrap_err()
            .contains("at least one weight must be positive"));

        let mut classifier = config();
        classifier.algorithm = AlgorithmConfig::LlmClassifier {
            classifier_target: "chat".into(),
            weak_target: "responses".into(),
            strong_target: "anthropic".into(),
            base_threshold: 1.1,
            min_confidence: 0.0,
            capability_elevated_floor: None,
            session_affinity: false,
            message_hash_fallback: false,
        };
        assert!(classifier
            .validate()
            .unwrap_err()
            .contains("base_threshold must be between 0 and 1"));
    }
}
