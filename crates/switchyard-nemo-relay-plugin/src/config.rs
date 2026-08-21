// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use http::Uri;
use http::header::{HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value as Json;
use switchyard_libsy::{
    Algorithm, ClassifierContractConfig, EscalationJudgeConfig, HandoffNoteConfig,
    LlmClassifierConfig, LlmFallback, LlmTaskClassifier, PickerMode, Random, StageRouter,
    StageRouterConfig, TargetPrompts, TaskClassifierConfig,
};
use switchyard_llm_client::{ModelConfig, TranslatingLlmClient};
use switchyard_protocol::{ModelId, RoutedLlmClient, WireFormat};

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
    drop_caller_extra_body: bool,
    #[serde(default)]
    header_env: BTreeMap<String, String>,
    #[serde(default)]
    extra_body: BTreeMap<String, Json>,
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

    fn prepare(&self) -> Result<PreparedTargetTransport, String> {
        let mut headers = BTreeMap::new();
        for (name, variable) in &self.header_env {
            let value = std::env::var(variable)
                .map_err(|_| format!("environment variable {variable:?} is not set"))?;
            validate_header(name, &value)?;
            headers.insert(name.clone(), value);
        }
        let model_config = TargetClient::model_config(
            self.model.clone(),
            self.protocol,
            self.dispatch_url(),
            headers,
            self.extra_body.clone(),
        );
        Ok(PreparedTargetTransport {
            provider_model: ModelId::from(self.model.clone()),
            protocol: self.protocol,
            drop_caller_extra_body: self.drop_caller_extra_body,
            model_config,
        })
    }
}

struct PreparedTargetTransport {
    provider_model: ModelId,
    protocol: WireFormat,
    drop_caller_extra_body: bool,
    model_config: ModelConfig,
}

pub(crate) struct PreparedTargetBinding {
    pub(crate) client: Arc<dyn RoutedLlmClient>,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LlmClassifierMode {
    #[default]
    Capability,
    Escalation,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClassifierAlgorithmConfig {
    #[serde(default)]
    mode: LlmClassifierMode,
    classifier_target: String,
    weak_target: String,
    strong_target: String,
    #[serde(default)]
    base_threshold: Option<f64>,
    #[serde(default)]
    threshold_step: Option<f64>,
    #[serde(default)]
    session_affinity: Option<bool>,
    #[serde(default)]
    message_hash_fallback: Option<bool>,
    #[serde(default)]
    recent_turn_window: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default = "default_classifier_max_output_tokens")]
    max_output_tokens: u64,
    #[serde(default)]
    escalation: Option<EscalationJudgeConfig>,
}

impl LlmClassifierAlgorithmConfig {
    fn capability_config(&self) -> Result<TaskClassifierConfig, String> {
        if self.escalation.is_some() {
            return Err(
                "llm_classifier capability mode does not accept escalation settings".into(),
            );
        }
        let base_threshold = self
            .base_threshold
            .ok_or_else(|| "llm_classifier capability mode requires base_threshold".to_string())?;
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = &self.prompt {
            contract = contract.with_prompt(prompt.clone());
        }
        Ok(TaskClassifierConfig {
            base_threshold,
            threshold_step: self.threshold_step.unwrap_or_default(),
            session_affinity: self.session_affinity.unwrap_or_default(),
            message_hash_fallback: self.message_hash_fallback.unwrap_or_default(),
            recent_turn_window: self.recent_turn_window,
            contract,
            max_output_tokens: self.max_output_tokens,
        })
    }

    fn escalation_config(
        &self,
    ) -> Result<(ClassifierContractConfig, EscalationJudgeConfig), String> {
        if self.base_threshold.is_some()
            || self.threshold_step.is_some()
            || self.session_affinity.is_some()
            || self.message_hash_fallback.is_some()
            || self.recent_turn_window.is_some()
        {
            return Err(
                "llm_classifier escalation mode does not accept capability settings".into(),
            );
        }
        let config = self.escalation.clone().ok_or_else(|| {
            "llm_classifier escalation mode requires escalation settings".to_string()
        })?;
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = &self.prompt {
            contract = contract.with_prompt(prompt.clone());
        }
        Ok((contract, config))
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageFallbackConfig {
    target: String,
    base_threshold: f64,
    #[serde(default)]
    threshold_step: f64,
    #[serde(default)]
    recent_turn_window: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default = "default_classifier_max_output_tokens")]
    max_output_tokens: u64,
}

impl StageFallbackConfig {
    fn classifier_config(&self) -> TaskClassifierConfig {
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = &self.prompt {
            contract = contract.with_prompt(prompt.clone());
        }
        TaskClassifierConfig {
            base_threshold: self.base_threshold,
            threshold_step: self.threshold_step,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: self.recent_turn_window,
            contract,
            max_output_tokens: self.max_output_tokens,
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
        #[serde(flatten)]
        config: LlmClassifierAlgorithmConfig,
    },
    StageRouter {
        capable_target: String,
        efficient_target: String,
        picker: PickerMode,
        confidence_threshold: f64,
        #[serde(default)]
        recent_turn_window: Option<usize>,
        #[serde(default)]
        capable_system_prompt: Option<String>,
        #[serde(default)]
        efficient_system_prompt: Option<String>,
        #[serde(default)]
        handoff_notes: Option<HandoffNoteConfig>,
        #[serde(default)]
        classifier: Option<StageFallbackConfig>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwitchyardConfig {
    version: u32,
    #[serde(default)]
    pub(crate) priority: i32,
    algorithm: AlgorithmConfig,
    targets: BTreeMap<String, TargetBinding>,
    default_targets: BTreeMap<WireFormat, String>,
}

pub(crate) struct PreparedConfig {
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
        let transports = self
            .targets
            .iter()
            .map(|(name, target)| target.prepare().map(|prepared| (name.clone(), prepared)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let models = transports
            .values()
            .map(|transport| transport.model_config.clone())
            .collect::<Vec<_>>();
        // One multi-model client shares reqwest connection pools across targets.
        let client = Arc::new(
            TranslatingLlmClient::new(&models)
                .map_err(|error| format!("failed to create target HTTP client: {error}"))?,
        );
        let targets = transports
            .into_iter()
            .map(|(name, transport)| {
                let target = TargetClient::new(
                    transport.provider_model,
                    transport.protocol,
                    transport.drop_caller_extra_body,
                    client.clone(),
                );
                (
                    name,
                    PreparedTargetBinding {
                        client: Arc::new(target),
                    },
                )
            })
            .collect();
        let algorithm = self.build_algorithm(Some(&targets))?;
        Ok(PreparedConfig {
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
                Some(targets) => {
                    targets
                        .get(name)
                        .ok_or_else(|| format!("algorithm target {name:?} was not prepared"))?;
                    ModelId::from(name)
                }
                None => ModelId::from(name),
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
                Random::new(targets, Some(weights), *seed)
                    .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                    .map_err(|error| error.to_string())
            }
            AlgorithmConfig::LlmClassifier { config } => {
                self.validate_judge_target(&config.classifier_target)?;
                let algorithm = match config.mode {
                    LlmClassifierMode::Capability => LlmClassifierConfig::Capability {
                        judge_target: target(&config.classifier_target)?,
                        efficient_target: target(&config.weak_target)?,
                        capable_target: target(&config.strong_target)?,
                        config: config.capability_config()?,
                    },
                    LlmClassifierMode::Escalation => {
                        let (contract, escalation) = config.escalation_config()?;
                        LlmClassifierConfig::Escalation {
                            judge_target: target(&config.classifier_target)?,
                            efficient_target: target(&config.weak_target)?,
                            capable_target: target(&config.strong_target)?,
                            contract,
                            config: escalation,
                            max_output_tokens: config.max_output_tokens,
                        }
                    }
                };
                LlmTaskClassifier::new(algorithm)
                    .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                    .map_err(|error| error.to_string())
            }
            AlgorithmConfig::StageRouter {
                capable_target,
                efficient_target,
                picker,
                confidence_threshold,
                recent_turn_window,
                capable_system_prompt,
                efficient_system_prompt,
                handoff_notes,
                classifier,
            } => {
                let capable = target(capable_target)?;
                let efficient = target(efficient_target)?;
                let mut config = StageRouterConfig::new(*picker, *confidence_threshold);
                config.recent_window = *recent_turn_window;
                config.handoff_notes = handoff_notes.clone();
                let mut prompts = TargetPrompts::default();
                if let Some(prompt) = capable_system_prompt {
                    prompts = prompts.with(capable_target.as_str(), prompt);
                }
                if let Some(prompt) = efficient_system_prompt {
                    prompts = prompts.with(efficient_target.as_str(), prompt);
                }
                config.tier_prompts = prompts;
                if let Some(classifier) = classifier {
                    self.validate_judge_target(&classifier.target)?;
                    config.llm_fallback = Some(LlmFallback {
                        judge_target: target(&classifier.target)?,
                        config: classifier.classifier_config(),
                    });
                }
                StageRouter::new(capable, efficient, config)
                    .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                    .map_err(|error| error.to_string())
            }
        }
    }

    fn validate_judge_target(&self, name: &str) -> Result<(), String> {
        let binding = self
            .targets
            .get(name)
            .ok_or_else(|| format!("algorithm target {name:?} is not configured"))?;
        if binding.protocol == WireFormat::AnthropicMessages {
            return Err(format!(
                "classifier target {name:?} uses anthropic_messages, which cannot encode the required JSON-schema response format without loss; use an openai_chat or openai_responses target"
            ));
        }
        Ok(())
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

const fn default_weight() -> f64 {
    1.0
}

fn default_classifier_max_output_tokens() -> u64 {
    TaskClassifierConfig::default().max_output_tokens
}

#[cfg(test)]
mod tests;
