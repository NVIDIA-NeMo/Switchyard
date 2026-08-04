// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use http::Uri;
use http::header::{HeaderName, HeaderValue};
use serde::Deserialize;
use switchyard_libsy::{
    Algorithm, LlmTarget, LlmTargetSet, LlmTaskClassifier, Random, TaskClassifierConfig,
};
use switchyard_protocol::{RoutedLlmClient, WireFormat};

use crate::client::TargetClient;

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
#[serde(deny_unknown_fields)]
struct TargetBinding {
    model: String,
    protocol: WireFormat,
    #[serde(default)]
    endpoint: String,
    base_url: String,
    #[serde(default = "default_weight")]
    weight: f64,
    #[serde(default)]
    header_env: BTreeMap<String, String>,
}

impl TargetBinding {
    fn dispatch_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        let default = default_endpoint(self.protocol);
        if self.endpoint.is_empty() && base.ends_with(default) {
            return base.to_string();
        }
        let endpoint = if self.endpoint.is_empty() {
            default
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

    fn validate(&self, name: &str) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err(format!("target {name:?} model must be non-empty"));
        }
        if !self.endpoint.is_empty() && !self.endpoint.starts_with('/') {
            return Err(format!(
                "target {name:?} endpoint must be empty or begin with '/'"
            ));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(format!(
                "target {name:?} weight must be finite and nonnegative"
            ));
        }
        validate_dispatch_url(name, self.protocol, &self.dispatch_url())?;
        self.validate_headers(name)
    }

    fn validate_headers(&self, target_name: &str) -> Result<(), String> {
        let mut normalized = BTreeSet::new();
        for (name, variable) in &self.header_env {
            let canonical = validate_header_name(name)?;
            if !normalized.insert(canonical) {
                return Err(format!(
                    "target {target_name:?} configures header {name:?} more than once (header names are case-insensitive)"
                ));
            }
            if variable.trim().is_empty() {
                return Err(format!(
                    "environment variable name for target header {name:?} must not be empty"
                ));
            }
            if variable.as_bytes().contains(&b'=') || variable.as_bytes().contains(&b'\0') {
                return Err(format!(
                    "environment variable name for target header {name:?} must not contain '=' or NUL"
                ));
            }
        }
        Ok(())
    }

    fn prepare(&self) -> Result<PreparedTargetBinding, String> {
        let mut headers = BTreeMap::new();
        for (name, variable) in &self.header_env {
            let value = std::env::var(variable)
                .map_err(|_| format!("environment variable {variable:?} is not set"))?;
            validate_header(name, &value)?;
            headers.insert(name.clone(), value);
        }
        let dispatch_url = self.dispatch_url();
        let client = TargetClient::new(self.model.clone(), self.protocol, dispatch_url, headers)
            .map_err(|error| format!("failed to create target HTTP client: {error}"))?;
        Ok(PreparedTargetBinding {
            client: Arc::new(client),
        })
    }
}

pub(crate) struct PreparedTargetBinding {
    pub(crate) client: Arc<dyn RoutedLlmClient>,
}

impl PreparedTargetBinding {
    fn as_llm_target(&self, semantic_name: &str) -> LlmTarget {
        LlmTarget {
            semantic_name: semantic_name.to_string(),
            llm_client: Some(self.client.clone()),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AlgorithmConfig {
    Random {
        #[serde(default)]
        seed: Option<u64>,
    },
    LlmClassifier {
        classifier_target: String,
        weak_target: String,
        strong_target: String,
        #[serde(flatten)]
        config: TaskClassifierConfig,
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
        self.build_algorithm(None).map(drop)
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
            if name.trim().is_empty() {
                return Err("target names must be non-empty".into());
            }
            target.validate(name)?;
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
        let targets = self
            .targets
            .iter()
            .map(|(name, target)| target.prepare().map(|prepared| (name.clone(), prepared)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let algorithm = self.build_algorithm(Some(&targets))?;
        Ok(PreparedConfig {
            max_retries: self.max_retries,
            algorithm,
            targets,
            default_targets: self.default_targets,
        })
    }

    fn build_algorithm(
        &self,
        prepared: Option<&BTreeMap<String, PreparedTargetBinding>>,
    ) -> Result<Arc<dyn Algorithm>, String> {
        let target = |name: &str| {
            if !self.targets.contains_key(name) {
                return Err(format!("algorithm target {name:?} is not configured"));
            }
            Ok(match prepared {
                Some(targets) => targets
                    .get(name)
                    .ok_or_else(|| format!("algorithm target {name:?} was not prepared"))?
                    .as_llm_target(name),
                None => LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: None,
                },
            })
        };

        match &self.algorithm {
            AlgorithmConfig::Random { seed } => {
                let routable = self
                    .targets
                    .iter()
                    .filter(|(_, binding)| binding.weight > 0.0)
                    .collect::<Vec<_>>();
                if routable.is_empty() {
                    return Err(
                        "random routing requires at least one positive target weight".into(),
                    );
                }
                let targets = routable
                    .iter()
                    .map(|(name, _)| target(name))
                    .collect::<Result<Vec<_>, _>>()?;
                let weights = routable
                    .iter()
                    .map(|(_, binding)| binding.weight)
                    .collect::<Vec<_>>();
                Random::new(LlmTargetSet::new(targets), Some(weights), *seed)
                    .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                    .map_err(|error| error.to_string())
            }
            AlgorithmConfig::LlmClassifier {
                classifier_target,
                weak_target,
                strong_target,
                config,
            } => {
                let classifier_binding = self.targets.get(classifier_target).ok_or_else(|| {
                    format!("algorithm target {classifier_target:?} is not configured")
                })?;
                if classifier_binding.protocol == WireFormat::AnthropicMessages {
                    return Err(format!(
                        "classifier target {classifier_target:?} uses anthropic_messages, which cannot encode the required JSON-schema response format without loss; use an openai_chat or openai_responses target"
                    ));
                }
                LlmTaskClassifier::new(
                    target(classifier_target)?,
                    target(weak_target)?,
                    target(strong_target)?,
                    config.clone(),
                )
                .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                .map_err(|error| error.to_string())
            }
        }
    }
}

fn validate_dispatch_url(
    target_name: &str,
    protocol: WireFormat,
    dispatch_url: &str,
) -> Result<(), String> {
    let uri = dispatch_url
        .parse::<Uri>()
        .map_err(|error| format!("target {target_name:?} has invalid URL: {error}"))?;
    if !matches!(uri.scheme_str(), Some("http" | "https")) {
        return Err(format!(
            "target {target_name:?} base_url must use http or https"
        ));
    }
    let authority = uri
        .authority()
        .ok_or_else(|| format!("target {target_name:?} URL must include a host"))?;
    if authority.host().is_empty() {
        return Err(format!("target {target_name:?} URL must include a host"));
    }
    if authority.as_str().contains('@') {
        return Err(format!(
            "target {target_name:?} URL must not contain embedded credentials"
        ));
    }
    if uri.query().is_some() {
        return Err(format!(
            "target {target_name:?} URL query parameters are not supported"
        ));
    }

    // The current switchyard-llm-client accepts provider base URLs and complete
    // canonical endpoints. Reject a custom terminal route to avoid
    // allowing Backend::url() to append another provider suffix silently.
    let expected_suffix = match protocol {
        WireFormat::OpenAiChat => "/chat/completions",
        WireFormat::OpenAiResponses => "/responses",
        WireFormat::AnthropicMessages => "/v1/messages",
    };
    if !uri.path().ends_with(expected_suffix) {
        return Err(format!(
            "target {target_name:?} endpoint must resolve to a canonical {protocol} route ending in {expected_suffix:?}"
        ));
    }
    Ok(())
}

fn validate_header_name(name: &str) -> Result<String, String> {
    let parsed = HeaderName::from_bytes(name.as_bytes())
        .map_err(|error| format!("invalid target header name {name:?}: {error}"))?;
    let canonical = parsed.as_str().to_ascii_lowercase();
    if is_forbidden_target_header(&canonical) {
        return Err(format!(
            "target header {name:?} is controlled by the HTTP transport and cannot be configured"
        ));
    }
    Ok(canonical)
}

fn validate_header(name: &str, value: &str) -> Result<String, String> {
    let canonical = validate_header_name(name)?;
    HeaderValue::from_str(value)
        .map_err(|error| format!("invalid target header value for {name:?}: {error}"))?;
    Ok(canonical)
}

fn is_forbidden_target_header(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "content-length"
            | "host"
            | "keep-alive"
            | "proxy-connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || name.starts_with("x-nemo-relay-internal-")
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
                .all(|target| Arc::strong_count(&target.client) >= 2)
        );
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
    fn classifier_attaches_clients_to_judge_and_routed_targets() {
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
                .all(|target| Arc::strong_count(&target.client) >= 2)
        );
    }

    #[test]
    fn classifier_rejects_anthropic_judge_targets_before_dispatch() {
        let mut config = config();
        config.algorithm = AlgorithmConfig::LlmClassifier {
            classifier_target: "anthropic".into(),
            weak_target: "responses".into(),
            strong_target: "chat".into(),
            config: TaskClassifierConfig {
                base_threshold: 0.5,
                ..Default::default()
            },
        };

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
        classifier.algorithm = AlgorithmConfig::LlmClassifier {
            classifier_target: "chat".into(),
            weak_target: "responses".into(),
            strong_target: "anthropic".into(),
            config: TaskClassifierConfig {
                base_threshold: 1.1,
                ..Default::default()
            },
        };
        assert!(
            classifier
                .validate()
                .unwrap_err()
                .contains("base_threshold must be between 0 and 1")
        );
    }

    #[test]
    fn zero_weight_random_targets_are_fallback_only() {
        let mut config = config();
        config.targets.get_mut("anthropic").unwrap().weight = 0.0;
        let prepared = config.prepare().unwrap();

        assert_eq!(Arc::strong_count(&prepared.targets["anthropic"].client), 1);
        assert!(Arc::strong_count(&prepared.targets["chat"].client) >= 2);
        assert!(Arc::strong_count(&prepared.targets["responses"].client) >= 2);
    }
}
