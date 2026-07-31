// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use http::header::{HeaderName, HeaderValue};
use nemo_relay_plugin::LlmContinuationRouteV2;
use serde::{Deserialize, Serialize};
use switchyard_libsy::algorithms::{LlmTaskClassifier, Random, TaskClassifierConfig};
use switchyard_libsy::{Algorithm, LlmTarget, LlmTargetSet};
use switchyard_protocol::WireFormat;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    OpenaiChat,
    OpenaiResponses,
    AnthropicMessages,
}

impl WireProtocol {
    pub const fn label(self) -> &'static str {
        match self {
            Self::OpenaiChat => "openai_chat",
            Self::OpenaiResponses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
        }
    }

    pub const fn endpoint(self) -> &'static str {
        match self {
            Self::OpenaiChat => "/v1/chat/completions",
            Self::OpenaiResponses => "/v1/responses",
            Self::AnthropicMessages => "/v1/messages",
        }
    }

    pub fn from_call(name: &str) -> Option<Self> {
        match name {
            "openai.chat_completions" | "openai_chat" | "openai_chat_completions" => {
                Some(Self::OpenaiChat)
            }
            "openai.responses" | "openai_responses" => Some(Self::OpenaiResponses),
            "anthropic.messages" | "anthropic" | "anthropic_messages" => {
                Some(Self::AnthropicMessages)
            }
            _ => None,
        }
    }

    pub const fn relay_route(self) -> LlmContinuationRouteV2 {
        match self {
            Self::OpenaiChat => LlmContinuationRouteV2::OpenaiChat,
            Self::OpenaiResponses => LlmContinuationRouteV2::OpenaiResponses,
            Self::AnthropicMessages => LlmContinuationRouteV2::AnthropicMessages,
        }
    }

    pub const fn wire_format(self) -> WireFormat {
        match self {
            Self::OpenaiChat => WireFormat::OpenAiChat,
            Self::OpenaiResponses => WireFormat::OpenAiResponses,
            Self::AnthropicMessages => WireFormat::AnthropicMessages,
        }
    }

    pub fn from_wire_format(format: &WireFormat) -> Option<Self> {
        match format {
            WireFormat::OpenAiChat => Some(Self::OpenaiChat),
            WireFormat::OpenAiResponses => Some(Self::OpenaiResponses),
            WireFormat::AnthropicMessages => Some(Self::AnthropicMessages),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TargetBinding {
    pub model: String,
    pub protocol: WireProtocol,
    #[serde(default)]
    pub endpoint: String,
    pub base_url: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub header_env: BTreeMap<String, String>,
}

impl TargetBinding {
    pub fn dispatch_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        let endpoint = if self.endpoint.is_empty() {
            self.protocol.endpoint()
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

    pub fn resolved_headers(&self) -> Result<BTreeMap<String, String>, String> {
        let mut headers = BTreeMap::new();
        for (name, value) in &self.headers {
            validate_header(name, value)?;
            headers.insert(name.clone(), value.clone());
        }
        for (name, variable) in &self.header_env {
            if self
                .headers
                .keys()
                .any(|configured| configured.eq_ignore_ascii_case(name))
            {
                return Err(format!(
                    "target header {name:?} cannot appear in both headers and header_env"
                ));
            }
            let value = std::env::var(variable)
                .map_err(|_| format!("environment variable {variable:?} is not set"))?;
            validate_header(name, &value)?;
            headers.insert(name.clone(), value);
        }
        Ok(headers)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ProtocolDefaults {
    #[serde(default)]
    pub openai_chat: String,
    #[serde(default)]
    pub openai_responses: String,
    #[serde(default)]
    pub anthropic_messages: String,
}

impl ProtocolDefaults {
    pub fn target(&self, protocol: WireProtocol) -> &str {
        match protocol {
            WireProtocol::OpenaiChat => &self.openai_chat,
            WireProtocol::OpenaiResponses => &self.openai_responses,
            WireProtocol::AnthropicMessages => &self.anthropic_messages,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AlgorithmConfig {
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

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self::Random { seed: None }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SwitchyardConfig {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub algorithm: AlgorithmConfig,
    pub targets: BTreeMap<String, TargetBinding>,
    #[serde(default)]
    pub default_targets: ProtocolDefaults,
    #[serde(default = "default_enabled_protocols")]
    pub enabled_inbound_profiles: BTreeSet<WireProtocol>,
}

impl SwitchyardConfig {
    pub fn validate(&self) -> Result<(), String> {
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
        if self.enabled_inbound_profiles.is_empty() {
            return Err("enabled_inbound_profiles must not be empty".into());
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
            target.resolved_headers()?;
        }
        for protocol in &self.enabled_inbound_profiles {
            let fallback = self.default_targets.target(*protocol);
            let target = self
                .targets
                .get(fallback)
                .ok_or_else(|| format!("default target {fallback:?} is not configured"))?;
            if target.protocol != *protocol {
                return Err(format!(
                    "default target {fallback:?} must use protocol {}",
                    protocol.label()
                ));
            }
        }
        self.build_algorithm().map(|_| ())
    }

    pub fn build_algorithm(&self) -> Result<Arc<dyn Algorithm>, String> {
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
                    .map(|name| target(name))
                    .collect::<Result<Vec<_>, _>>()?;
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

fn validate_header(name: &str, value: &str) -> Result<(), String> {
    HeaderName::from_bytes(name.as_bytes())
        .map_err(|error| format!("invalid target header name {name:?}: {error}"))?;
    HeaderValue::from_str(value)
        .map_err(|error| format!("invalid target header value for {name:?}: {error}"))?;
    Ok(())
}

const fn default_version() -> u32 {
    2
}

const fn default_max_retries() -> u32 {
    3
}

const fn default_weight() -> f64 {
    1.0
}

fn default_enabled_protocols() -> BTreeSet<WireProtocol> {
    BTreeSet::from([
        WireProtocol::OpenaiChat,
        WireProtocol::OpenaiResponses,
        WireProtocol::AnthropicMessages,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(protocol: WireProtocol, model: &str) -> TargetBinding {
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
                    binding(WireProtocol::OpenaiChat, "provider/chat"),
                ),
                (
                    "responses".into(),
                    binding(WireProtocol::OpenaiResponses, "provider/responses"),
                ),
                (
                    "anthropic".into(),
                    binding(WireProtocol::AnthropicMessages, "provider/anthropic"),
                ),
            ]),
            default_targets: ProtocolDefaults {
                openai_chat: "chat".into(),
                openai_responses: "responses".into(),
                anthropic_messages: "anthropic".into(),
            },
            enabled_inbound_profiles: default_enabled_protocols(),
        }
    }

    #[test]
    fn version_two_random_configuration_builds_without_a_service() {
        let config = config();
        config.validate().unwrap();
        assert_eq!(config.build_algorithm().unwrap().name(), "random");
        assert_eq!(
            config.targets["chat"].dispatch_url(),
            "https://provider.example/v1/chat/completions"
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
            config.build_algorithm().unwrap().name(),
            "llm_task_classifier"
        );
    }
}
