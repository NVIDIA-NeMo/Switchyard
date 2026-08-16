// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed TOML configuration and explicit construction for the Rust server.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use libsy::{
    Algorithm, ClassifierContractConfig, ClassifierResponseFormat, CustomClassifierConfig,
    CustomClassifierPolicy, EscalationJudgeConfig, HandoffNoteConfig, LlmClassifierConfig,
    LlmFallback, LlmTaskClassifier, Noop, Passthrough, PickerMode, Random, StageRouter,
    StageRouterConfig, TargetPrompts, TaskClassifierConfig,
};
use serde::Deserialize;
use serde_json::Value;
use switchyard_llm_client::{
    Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    TranslatingLlmClient,
};
use switchyard_protocol::{ModelId, RoutedLlmClient};

use crate::{
    CallerAuthKind, CountTokensTarget, ModelCapabilities, ServerError, ServerResult, ServerState,
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const MAX_CONFIGURED_RETRIES: u32 = 10;

/// Loads a TOML deployment file and constructs the complete server state.
pub fn load_server_state(path: impl AsRef<Path>) -> ServerResult<ServerState> {
    let path = path.as_ref();
    let toml = fs::read_to_string(path).map_err(|error| {
        ServerError::new(format!(
            "failed to read server config {}: {error}",
            path.display()
        ))
    })?;
    server_state_from_toml(&toml).map_err(|error| {
        ServerError::new(format!("invalid server config {}: {error}", path.display()))
    })
}

fn server_state_from_toml(toml: &str) -> ServerResult<ServerState> {
    let config: ServerConfig = toml::from_str(toml)
        .map_err(|error| ServerError::new(format!("failed to parse TOML: {error}")))?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerConfig {
    schema_version: u32,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
}

impl ServerConfig {
    fn build(&self) -> ServerResult<ServerState> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(ServerError::new(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        // An llm client keys its models by id, so two targets that share an id on one
        // client end up as one entry: the client keeps one target and drops the other. Warn
        // so the drop is visible at startup instead of silent, and still build both routes.
        // The set only tells us the pair was already seen, not whether the two targets
        // differ, so it warns for a harmless duplicate and for one whose extra_body would be
        // dropped alike. The same id on two different clients never collides: each client
        // keeps its own model.
        let mut seen_client_model_ids = HashSet::new();
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            if !seen_client_model_ids.insert((target.llm_client.as_str(), target.id.as_str())) {
                tracing::warn!(
                    "target {target_name} reuses model id {} on llm client {}; only one target per id is kept and the other is dropped. Give each target a unique model id, or point both routes at one target.",
                    target.id,
                    target.llm_client
                );
            }
        }

        let clients = self.build_clients()?;
        let targets = self.build_targets()?;
        let mut routes = Vec::with_capacity(self.routes.len());
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), config.id())?;
            let capabilities = config.capabilities();
            if capabilities.context_window == Some(0) {
                return Err(ServerError::new(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            let algorithm = build_algorithm(route_name, config, &targets)?;
            let (client, caller_auth) = self.build_route_clients(route_name, config, &clients)?;
            let count_tokens_target = self.build_count_tokens_target(config, &clients);
            routes.push((
                config.id().clone(),
                algorithm,
                client,
                caller_auth,
                capabilities,
                count_tokens_target,
            ));
        }
        ServerState::new_with_capabilities(routes)
    }

    fn build_clients(&self) -> ServerResult<BTreeMap<String, Arc<TranslatingLlmClient>>> {
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for name in self.llm_clients.keys() {
            validate_value("llm client name", name)?;
        }
        for (target_name, target) in &self.targets {
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                ServerError::new(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| ServerError::new("validated llm client was not initialized"))?;
            model_configs.push(ModelConfig::new(
                target.id.clone(),
                build_backend(&target.llm_client, client_config, &target.extra_body)?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let client = Arc::new(
                TranslatingLlmClient::new(&model_configs)
                    .map_err(|error| ServerError::new(error.to_string()))?,
            );
            clients.insert(name, client);
        }
        Ok(clients)
    }

    fn build_targets(&self) -> ServerResult<BTreeMap<String, ModelId>> {
        Ok(self
            .targets
            .iter()
            .map(|(name, config)| (name.clone(), config.id.clone()))
            .collect())
    }

    /// Maps every model this route can select to the client configured for that target.
    ///
    /// A route is a synthetic model (`switchyard/random`) with no upstream of its own — the
    /// algorithm picks a real target, and the *target* names the `[llm_clients.*]` section
    /// that serves it. Two targets in one route may therefore sit on different clients, and
    /// one request can emit calls for both.
    ///
    /// The map is built per route because targets are only reachable through the routes that
    /// name them, so the same model id may resolve to different clients in two routes without
    /// colliding.
    fn build_route_clients(
        &self,
        route_name: &str,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> ServerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let mut by_model = HashMap::new();
        let mut caller_auth = None;
        for name in route.callable_target_names() {
            let target = self.targets.get(name).ok_or_else(|| {
                ServerError::new(format!("route references unknown target {name}"))
            })?;
            let client = clients.get(&target.llm_client).ok_or_else(|| {
                ServerError::new(format!("target {name} has no constructed llm client"))
            })?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                ServerError::new(format!(
                    "target {name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                if caller_auth.is_some_and(|kind| kind != target_auth) {
                    return Err(ServerError::new(format!(
                        "route {route_name} cannot forward both Anthropic and OpenAI caller credentials"
                    )));
                }
                caller_auth = Some(target_auth);
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(target.id.clone(), client);
        }
        Ok((ClientRouter::new(by_model), caller_auth))
    }

    fn build_count_tokens_target(
        &self,
        route_config: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<CountTokensTarget> {
        route_config
            .routing_target_names()
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let target = self.targets.get(name)?;
                let client = clients.get(&target.llm_client)?;
                client.supports_count_tokens(&target.id).then_some((
                    count_tokens_priority(name, &target.id),
                    index,
                    target,
                    client,
                ))
            })
            .min_by_key(|(priority, index, _, _)| (*priority, *index))
            .map(|(_, _, target, client)| CountTokensTarget {
                model: target.id.clone(),
                client: client.clone(),
            })
    }
}

// Prefer known Claude families, then preserve the route's target order.
fn count_tokens_priority(target_name: &str, model_id: &ModelId) -> usize {
    let target_name = target_name.to_ascii_lowercase();
    let model_id = model_id.to_ascii_lowercase();
    ["opus", "sonnet", "haiku"]
        .iter()
        .position(|hint| target_name.contains(hint) || model_id.contains(hint))
        .unwrap_or(3)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: String,
    api_key_env: Option<String>,
    #[serde(default)]
    forward_auth: bool,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum ClientFormat {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

impl ClientFormat {
    const fn caller_auth_kind(self) -> CallerAuthKind {
        match self {
            Self::AnthropicMessages => CallerAuthKind::Anthropic,
            Self::OpenAiChat | Self::OpenAiResponses => CallerAuthKind::OpenAi,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ClassifierPolicyConfig {
    TargetSelector { selector: String },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClassifierMode {
    Capability,
    Escalation,
    Custom,
}

impl ClassifierPolicyConfig {
    fn into_libsy(self) -> CustomClassifierPolicy {
        match self {
            Self::TargetSelector { selector } => CustomClassifierPolicy::target_selector(selector),
        }
    }
}

#[derive(Debug)]
enum LlmClassifierModeConfig {
    Capability(CapabilityClassifierRouteConfig),
    Escalation(EscalationClassifierRouteConfig),
    Custom(CustomClassifierRouteConfig),
}

#[derive(Debug)]
struct CapabilityClassifierRouteConfig {
    strong_target: String,
    weak_target: String,
    base_threshold: f64,
    threshold_step: f64,
    session_affinity: bool,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
}

#[derive(Debug)]
struct EscalationClassifierRouteConfig {
    strong_target: String,
    weak_target: String,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
    judge: EscalationJudgeConfig,
}

#[derive(Debug)]
struct CustomClassifierRouteConfig {
    targets: Vec<String>,
    default_target: String,
    prompt: String,
    response_schema: String,
    policy: ClassifierPolicyConfig,
    session_affinity: bool,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    max_output_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RouteConfig {
    Noop {
        id: ModelId,
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        tool_calling: Option<bool>,
        #[serde(default)]
        reasoning: Option<bool>,
    },
    Random {
        id: ModelId,
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        tool_calling: Option<bool>,
        #[serde(default)]
        reasoning: Option<bool>,
        targets: Vec<String>,
        weights: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    Passthrough {
        id: ModelId,
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        tool_calling: Option<bool>,
        #[serde(default)]
        reasoning: Option<bool>,
        target: String,
    },
    LlmClassifier {
        id: ModelId,
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        tool_calling: Option<bool>,
        #[serde(default)]
        reasoning: Option<bool>,
        classifier_target: String,
        #[serde(default)]
        mode: Option<ClassifierMode>,
        #[serde(default)]
        strong_target: Option<String>,
        #[serde(default)]
        weak_target: Option<String>,
        #[serde(default)]
        base_threshold: Option<f64>,
        #[serde(default)]
        threshold_step: Option<f64>,
        #[serde(default)]
        session_affinity: bool,
        #[serde(default)]
        message_hash_fallback: bool,
        #[serde(default)]
        recent_turn_window: Option<usize>,
        #[serde(default)]
        prompt: Option<String>,
        #[serde(default)]
        response_format_type: ClassifierResponseFormat,
        #[serde(default = "default_classifier_max_output_tokens")]
        max_output_tokens: u64,
        #[serde(default)]
        escalation: Option<EscalationJudgeConfig>,
        #[serde(default)]
        targets: Option<Vec<String>>,
        #[serde(default)]
        default_target: Option<String>,
        #[serde(default)]
        response_schema: Option<String>,
        #[serde(default)]
        policy: Option<ClassifierPolicyConfig>,
    },
    StageRouter {
        id: ModelId,
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        tool_calling: Option<bool>,
        #[serde(default)]
        reasoning: Option<bool>,
        capable_target: String,
        efficient_target: String,
        /// Tier a turn falls back to when the signals are not confident.
        picker: PickerMode,
        confidence_threshold: f64,
        /// Trailing tool results the signals are computed over.
        #[serde(default)]
        recent_turn_window: Option<usize>,
        /// Note handed to the model a signal-driven switch routes to.
        #[serde(default)]
        handoff_notes: Option<HandoffNoteConfig>,
        /// System prompt handed to each tier on every turn it serves.
        #[serde(default)]
        capable_system_prompt: Option<String>,
        #[serde(default)]
        efficient_system_prompt: Option<String>,
        /// Capability judge consulted on turns the signals leave undecided.
        #[serde(default)]
        classifier: Option<StageClassifierConfig>,
    },
}

/// The judge a `stage_router` route falls through to, and how it routes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageClassifierConfig {
    /// Target the judge is called through. Not a routing destination.
    target: String,
    base_threshold: f64,
    #[serde(default)]
    threshold_step: f64,
    #[serde(default)]
    session_affinity: bool,
    #[serde(default)]
    message_hash_fallback: bool,
    #[serde(default)]
    recent_turn_window: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    response_format_type: ClassifierResponseFormat,
    #[serde(default = "default_classifier_max_output_tokens")]
    max_output_tokens: u64,
}

impl StageClassifierConfig {
    fn task_classifier_config(&self) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold: self.base_threshold,
            threshold_step: self.threshold_step,
            session_affinity: self.session_affinity,
            message_hash_fallback: self.message_hash_fallback,
            recent_turn_window: self.recent_turn_window,
            contract: classifier_contract(self.prompt.as_deref())
                .with_response_format_type(self.response_format_type),
            max_output_tokens: self.max_output_tokens,
        }
    }
}

impl RouteConfig {
    fn id(&self) -> &ModelId {
        use RouteConfig::*;
        match self {
            Noop { id, .. }
            | Random { id, .. }
            | LlmClassifier { id, .. }
            | Passthrough { id, .. }
            | StageRouter { id, .. } => id,
        }
    }

    // Completion targets in algorithm order; judge-only targets are excluded.
    fn routing_target_names(&self) -> Vec<&str> {
        match self {
            Self::Noop { .. } => Vec::new(),
            Self::Random { targets, .. } => targets.iter().map(String::as_str).collect(),
            Self::Passthrough { target, .. } => vec![target],
            Self::LlmClassifier {
                mode,
                strong_target,
                weak_target,
                escalation,
                targets,
                ..
            } => match mode.unwrap_or(if escalation.is_some() {
                ClassifierMode::Escalation
            } else {
                ClassifierMode::Capability
            }) {
                ClassifierMode::Capability => weak_target
                    .iter()
                    .chain(strong_target)
                    .map(String::as_str)
                    .collect(),
                ClassifierMode::Escalation => strong_target
                    .iter()
                    .chain(weak_target)
                    .map(String::as_str)
                    .collect(),
                ClassifierMode::Custom => targets.iter().flatten().map(String::as_str).collect(),
            },
            Self::StageRouter {
                capable_target,
                efficient_target,
                ..
            } => vec![capable_target, efficient_target],
        }
    }

    /// Every target the algorithm may call, including judge-only targets.
    ///
    /// [`routing_target_names`](Self::routing_target_names) covers completion destinations;
    /// a classifier also calls its judge, and that call needs a client too.
    fn callable_target_names(&self) -> Vec<&str> {
        let mut names = self.routing_target_names();
        match self {
            Self::LlmClassifier {
                classifier_target, ..
            } => names.push(classifier_target),
            Self::StageRouter {
                classifier: Some(classifier),
                ..
            } => names.push(&classifier.target),
            _ => {}
        }
        names
    }

    fn capabilities(&self) -> ModelCapabilities {
        use RouteConfig::*;
        match self {
            Noop {
                context_window,
                tool_calling,
                reasoning,
                ..
            }
            | Random {
                context_window,
                tool_calling,
                reasoning,
                ..
            }
            | Passthrough {
                context_window,
                tool_calling,
                reasoning,
                ..
            }
            | LlmClassifier {
                context_window,
                tool_calling,
                reasoning,
                ..
            }
            | StageRouter {
                context_window,
                tool_calling,
                reasoning,
                ..
            } => ModelCapabilities {
                context_window: *context_window,
                tool_calling: *tool_calling,
                reasoning: *reasoning,
            },
        }
    }

    fn classifier_mode(&self, route_name: &str) -> ServerResult<LlmClassifierModeConfig> {
        let Self::LlmClassifier {
            mode,
            strong_target,
            weak_target,
            base_threshold,
            threshold_step,
            session_affinity,
            message_hash_fallback,
            recent_turn_window,
            prompt,
            response_format_type,
            max_output_tokens,
            escalation,
            targets,
            default_target,
            response_schema,
            policy,
            ..
        } = self
        else {
            return Err(ServerError::new("route is not an llm_classifier"));
        };

        let selected_mode = match (mode, escalation.is_some()) {
            (Some(mode), _) => *mode,
            (None, true) => ClassifierMode::Escalation,
            (None, false) => ClassifierMode::Capability,
        };

        match selected_mode {
            ClassifierMode::Capability => {
                if escalation.is_some() {
                    return Err(classifier_field_error(
                        route_name,
                        "escalation",
                        "capability",
                    ));
                }
                reject_custom_fields(
                    route_name,
                    "capability",
                    targets,
                    default_target,
                    response_schema,
                    policy,
                )?;
                Ok(LlmClassifierModeConfig::Capability(
                    CapabilityClassifierRouteConfig {
                        strong_target: required_classifier_field(
                            route_name,
                            "strong_target",
                            strong_target,
                        )?,
                        weak_target: required_classifier_field(
                            route_name,
                            "weak_target",
                            weak_target,
                        )?,
                        base_threshold: required_classifier_field(
                            route_name,
                            "base_threshold",
                            base_threshold,
                        )?,
                        threshold_step: threshold_step.unwrap_or_default(),
                        session_affinity: *session_affinity,
                        message_hash_fallback: *message_hash_fallback,
                        recent_turn_window: *recent_turn_window,
                        prompt: prompt.clone(),
                        response_format_type: *response_format_type,
                        max_output_tokens: *max_output_tokens,
                    },
                ))
            }
            ClassifierMode::Escalation => {
                reject_custom_fields(
                    route_name,
                    "escalation",
                    targets,
                    default_target,
                    response_schema,
                    policy,
                )?;
                if mode.is_some()
                    && (base_threshold.is_some()
                        || threshold_step.is_some()
                        || *session_affinity
                        || *message_hash_fallback
                        || recent_turn_window.is_some())
                {
                    return Err(ServerError::new(format!(
                        "llm_classifier route {route_name} mode escalation cannot use capability routing settings"
                    )));
                }
                Ok(LlmClassifierModeConfig::Escalation(
                    EscalationClassifierRouteConfig {
                        strong_target: required_classifier_field(
                            route_name,
                            "strong_target",
                            strong_target,
                        )?,
                        weak_target: required_classifier_field(
                            route_name,
                            "weak_target",
                            weak_target,
                        )?,
                        prompt: prompt.clone(),
                        response_format_type: *response_format_type,
                        max_output_tokens: *max_output_tokens,
                        judge: required_classifier_field(route_name, "escalation", escalation)?,
                    },
                ))
            }
            ClassifierMode::Custom => {
                if strong_target.is_some()
                    || weak_target.is_some()
                    || base_threshold.is_some()
                    || threshold_step.is_some()
                    || escalation.is_some()
                    || *response_format_type != ClassifierResponseFormat::JsonSchema
                {
                    return Err(ServerError::new(format!(
                        "llm_classifier route {route_name} mode custom cannot use capability or escalation fields"
                    )));
                }
                Ok(LlmClassifierModeConfig::Custom(
                    CustomClassifierRouteConfig {
                        targets: required_classifier_field(route_name, "targets", targets)?,
                        default_target: required_classifier_field(
                            route_name,
                            "default_target",
                            default_target,
                        )?,
                        prompt: required_classifier_field(route_name, "prompt", prompt)?,
                        response_schema: required_classifier_field(
                            route_name,
                            "response_schema",
                            response_schema,
                        )?,
                        policy: required_classifier_field(route_name, "policy", policy)?,
                        session_affinity: *session_affinity,
                        message_hash_fallback: *message_hash_fallback,
                        recent_turn_window: *recent_turn_window,
                        max_output_tokens: *max_output_tokens,
                    },
                ))
            }
        }
    }
}

fn reject_custom_fields(
    route_name: &str,
    mode: &str,
    targets: &Option<Vec<String>>,
    default_target: &Option<String>,
    response_schema: &Option<String>,
    policy: &Option<ClassifierPolicyConfig>,
) -> ServerResult<()> {
    if targets.is_some()
        || default_target.is_some()
        || response_schema.is_some()
        || policy.is_some()
    {
        return Err(ServerError::new(format!(
            "llm_classifier route {route_name} mode {mode} cannot use custom classifier fields"
        )));
    }
    Ok(())
}

fn classifier_field_error(route_name: &str, field: &str, mode: &str) -> ServerError {
    ServerError::new(format!(
        "llm_classifier route {route_name} mode {mode} cannot use {field}"
    ))
}

fn required_classifier_field<T: Clone>(
    route_name: &str,
    field: &str,
    value: &Option<T>,
) -> ServerResult<T> {
    value.clone().ok_or_else(|| {
        ServerError::new(format!(
            "llm_classifier route {route_name} requires {field}"
        ))
    })
}

fn build_backend(
    client_name: &str,
    config: &LlmClientConfig,
    extra_body: &BTreeMap<String, Value>,
) -> ServerResult<Backend> {
    let base_url = config.base_url.trim();
    if base_url.is_empty() {
        return Err(ServerError::new(format!(
            "llm client {client_name} base_url must not be empty"
        )));
    }
    if config.max_retries > MAX_CONFIGURED_RETRIES {
        return Err(ServerError::new(format!(
            "llm client {client_name} max_retries must be at most {MAX_CONFIGURED_RETRIES}"
        )));
    }
    if config.forward_auth && config.api_key_env.is_some() {
        return Err(ServerError::new(format!(
            "llm client {client_name} cannot set both forward_auth and api_key_env"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| {
            if variable.trim().is_empty() {
                return Err(ServerError::new(format!(
                    "llm client {client_name} api_key_env must not be empty"
                )));
            }
            let api_key = std::env::var(variable).map_err(|error| {
                ServerError::new(format!(
                    "llm client {client_name} could not read api_key_env {variable}: {error}"
                ))
            })?;
            if api_key.trim().is_empty() {
                return Err(ServerError::new(format!(
                    "llm client {client_name} api_key_env {variable} is empty"
                )));
            }
            Ok(api_key)
        })
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key,
        forward_auth: config.forward_auth,
        extra_headers: config.extra_headers.clone(),
        extra_body: extra_body.clone(),
        max_retries: config.max_retries,
    };
    Ok(match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    })
}

const fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

fn build_algorithm(
    route_name: &str,
    config: &RouteConfig,
    targets: &BTreeMap<String, ModelId>,
) -> ServerResult<Arc<dyn Algorithm>> {
    match config {
        RouteConfig::Noop { .. } => Ok(Arc::new(Noop {})),
        RouteConfig::Random {
            targets: names,
            weights,
            seed,
            ..
        } => {
            let target_set =
                resolve_targets(route_name, names.iter().map(String::as_str), targets)?;
            let algorithm = Random::new(target_set, weights.clone(), *seed)
                .map_err(|error| ServerError::new(format!("random route {route_name}: {error}")))?;
            Ok(Arc::new(algorithm))
        }
        RouteConfig::Passthrough { target, .. } => {
            let target = resolve_target_model_id(route_name, target, targets)?;
            Ok(Arc::new(Passthrough::new(target)))
        }
        RouteConfig::LlmClassifier {
            classifier_target, ..
        } => {
            let classifier = resolve_target_model_id(route_name, classifier_target, targets)?;
            let mode = config.classifier_mode(route_name)?;
            let algorithm = match mode {
                LlmClassifierModeConfig::Capability(config) => {
                    let strong =
                        resolve_target_model_id(route_name, &config.strong_target, targets)?;
                    let weak = resolve_target_model_id(route_name, &config.weak_target, targets)?;
                    let classifier_config = TaskClassifierConfig {
                        base_threshold: config.base_threshold,
                        threshold_step: config.threshold_step,
                        session_affinity: config.session_affinity,
                        message_hash_fallback: config.message_hash_fallback,
                        recent_turn_window: config.recent_turn_window,
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        max_output_tokens: config.max_output_tokens,
                    };
                    LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                        judge_target: classifier,
                        efficient_target: weak,
                        capable_target: strong,
                        config: classifier_config,
                    })
                }
                LlmClassifierModeConfig::Escalation(config) => {
                    let strong =
                        resolve_target_model_id(route_name, &config.strong_target, targets)?;
                    let weak = resolve_target_model_id(route_name, &config.weak_target, targets)?;
                    LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
                        judge_target: classifier,
                        efficient_target: weak,
                        capable_target: strong,
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        config: config.judge,
                        max_output_tokens: config.max_output_tokens,
                    })
                }
                LlmClassifierModeConfig::Custom(config) => {
                    let resolved_targets = config
                        .targets
                        .iter()
                        .map(|name| {
                            resolve_target_model_id(route_name, name, targets)
                                .map(|target| (name.clone(), target))
                        })
                        .collect::<ServerResult<Vec<_>>>()?;
                    let response_schema = serde_json::from_str(&config.response_schema).map_err(
                        |error| {
                            ServerError::new(format!(
                                "llm_classifier route {route_name}: response_schema is invalid JSON: {error}"
                            ))
                        },
                    )?;
                    let mut classifier_config = CustomClassifierConfig::new(
                        config.prompt,
                        response_schema,
                        config.policy.into_libsy(),
                    );
                    classifier_config.session_affinity = config.session_affinity;
                    classifier_config.message_hash_fallback = config.message_hash_fallback;
                    classifier_config.recent_turn_window = config.recent_turn_window;
                    classifier_config.max_output_tokens = config.max_output_tokens;
                    LlmTaskClassifier::new(LlmClassifierConfig::Custom {
                        judge_target: classifier,
                        targets: resolved_targets,
                        default_target: config.default_target,
                        config: classifier_config,
                    })
                }
            }
            .map_err(|error| {
                ServerError::new(format!("llm_classifier route {route_name}: {error}"))
            })?;
            Ok(Arc::new(algorithm))
        }
        RouteConfig::StageRouter {
            capable_target,
            efficient_target,
            picker,
            confidence_threshold,
            recent_turn_window,
            handoff_notes,
            capable_system_prompt,
            efficient_system_prompt,
            classifier,
            ..
        } => {
            if matches!(picker, PickerMode::CapableFirst) {
                tracing::warn!(
                    "stage_router route {route_name} uses picker \"capable_first\", which is experimental: published thresholds and routing results all come from \"efficient_first\", so there is no calibrated confidence_threshold for it and no measured accuracy or cost. Use \"efficient_first\" unless you are running your own calibration."
                );
            }
            let capable = resolve_target_model_id(route_name, capable_target, targets)?;
            let efficient = resolve_target_model_id(route_name, efficient_target, targets)?;
            let mut config = StageRouterConfig::new(*picker, *confidence_threshold);
            config.recent_window = *recent_turn_window;
            config.handoff_notes = handoff_notes.clone();
            config.tier_prompts = tier_prompts(
                &capable,
                capable_system_prompt.as_deref(),
                &efficient,
                efficient_system_prompt.as_deref(),
            );
            // The judge is called through its own target, so it is not a routing
            // destination and stays out of the tier pair.
            config.llm_fallback = classifier
                .as_ref()
                .map(|classifier| {
                    resolve_target_model_id(route_name, &classifier.target, targets).map(
                        |judge_target| LlmFallback {
                            judge_target,
                            config: classifier.task_classifier_config(),
                        },
                    )
                })
                .transpose()?;
            let algorithm = StageRouter::new(capable, efficient, config).map_err(|error| {
                ServerError::new(format!("stage_router route {route_name}: {error}"))
            })?;
            Ok(Arc::new(algorithm))
        }
    }
}

fn classifier_contract(prompt: Option<&str>) -> ClassifierContractConfig {
    prompt.map_or_else(ClassifierContractConfig::default, |prompt| {
        ClassifierContractConfig::default().with_prompt(prompt)
    })
}

fn default_classifier_max_output_tokens() -> u64 {
    TaskClassifierConfig::default().max_output_tokens
}

/// Keys each configured system prompt by the target it belongs to.
fn tier_prompts(
    capable: &str,
    capable_prompt: Option<&str>,
    efficient: &str,
    efficient_prompt: Option<&str>,
) -> TargetPrompts {
    let mut prompts = TargetPrompts::default();
    if let Some(prompt) = capable_prompt {
        prompts = prompts.with(capable, prompt);
    }
    if let Some(prompt) = efficient_prompt {
        prompts = prompts.with(efficient, prompt);
    }
    prompts
}

fn resolve_targets<'a>(
    route_name: &str,
    names: impl IntoIterator<Item = &'a str>,
    targets: &BTreeMap<String, ModelId>,
) -> ServerResult<Vec<ModelId>> {
    names
        .into_iter()
        .map(|name| resolve_target_model_id(route_name, name, targets))
        .collect()
}

fn resolve_target_model_id(
    route_name: &str,
    name: &str,
    targets: &BTreeMap<String, ModelId>,
) -> ServerResult<ModelId> {
    targets.get(name).cloned().ok_or_else(|| {
        ServerError::new(format!(
            "route {route_name} references unknown target {name}"
        ))
    })
}

fn validate_value(label: &str, value: &str) -> ServerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(ServerError::new(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const VALID_CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.classifier]
id = "classifier/model"
llm_client = "primary"

[targets.strong]
id = "strong/model"
llm_client = "responses"

[targets.weak]
id = "weak/model"
llm_client = "anthropic"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["strong", "weak"]

[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "weak"
"#;

    fn error_message(toml: &str) -> String {
        match server_state_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> ServerResult<()> {
        let state = server_state_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state.models().collect::<Vec<_>>(),
            [
                "switchyard/classifier",
                "switchyard/noop",
                "switchyard/passthrough",
                "switchyard/random",
            ]
        );
        Ok(())
    }

    #[test]
    fn an_escalation_table_switches_the_classifier_route_to_escalation() -> ServerResult<()> {
        // Present: the classifier target judges the weak tier's reply each turn instead of
        // picking a tier ahead of it. The route builds either way, so the assertion is that
        // the knob parses and its settings reach the algorithm's validation.
        let escalating = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2 }",
        );
        server_state_from_toml(&escalating)?;

        // A setting that would starve the judge is rejected here rather than on the first
        // request, the same as any other unusable route configuration.
        let starved = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 0 }",
        );
        assert!(error_message(&starved).contains("confirmations must be at least 1"));
        Ok(())
    }

    #[test]
    fn classifier_judge_completion_caps_are_configurable() -> ServerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 512",
        );
        server_state_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 256\nescalation = { confirmations = 2 }",
        );
        server_state_from_toml(&escalation)?;
        Ok(())
    }

    #[test]
    fn classifier_prompts_are_configurable_in_both_modes() -> ServerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom capability rubric\"",
        );
        server_state_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom trajectory rubric\"\nescalation = { confirmations = 2 }",
        );
        server_state_from_toml(&escalation)?;

        let empty = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"   \"",
        );
        assert!(error_message(&empty).contains("classifier prompt must not be empty"));

        let schema_placeholder = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"{{RESPONSE_SCHEMA}}\"",
        );
        assert!(
            error_message(&schema_placeholder)
                .contains("Switchyard supplies the schema automatically")
        );
        Ok(())
    }

    #[test]
    fn mode_custom_rejects_capability_fields() {
        let mixed = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "mode = \"custom\"\nbase_threshold = 0.5",
        );

        assert!(
            error_message(&mixed)
                .contains("mode custom cannot use capability or escalation fields")
        );
    }

    #[test]
    fn rejects_unknown_fields_and_algorithm_types() {
        let unknown_field =
            VALID_CONFIG.replace("schema_version = 1", "schema_version = 1\nmagic = true");
        assert!(error_message(&unknown_field).contains("unknown field"));

        let nested_completion_cap = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { max_output_tokens = 256 }",
        );
        assert!(error_message(&nested_completion_cap).contains("unknown field"));

        let unknown_classifier_field = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nclassifier_magic = true",
        );
        assert!(error_message(&unknown_classifier_field).contains("unknown field"));

        let target_capability = VALID_CONFIG.replace(
            "llm_client = \"responses\"",
            "llm_client = \"responses\"\ncontext_window = 1000000",
        );
        assert!(error_message(&target_capability).contains("unknown field `context_window`"));

        let unknown_algorithm = VALID_CONFIG.replace("type = \"noop\"", "type = \"imaginary\"");
        assert!(error_message(&unknown_algorithm).contains("unknown variant"));
    }

    #[test]
    fn rejects_unknown_stage_classifier_fields() {
        // Nested classifier typos must fail instead of silently using a default.
        let config = format!(
            r#"{VALID_CONFIG}

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
classifier_magic = true
"#
        );

        let error = error_message(&config);
        assert!(
            error.contains("unknown field `classifier_magic`"),
            "{error}"
        );
    }

    #[test]
    fn rejects_invalid_references_and_parameters() {
        let cases = [
            (
                VALID_CONFIG.replace("llm_client = \"primary\"", "llm_client = \"missing\""),
                "unknown llm client missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"missing\"]",
                ),
                "unknown target missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"strong\"]",
                ),
                "random targets must be unique",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [1]",
                ),
                "expected 2 weights, got 1",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [0, 0]",
                ),
                "at least one weight must be positive",
            ),
            (
                VALID_CONFIG.replace("base_threshold = 0.5", "base_threshold = 1.5"),
                "base_threshold must be between 0 and 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nthreshold_step = -0.1",
                ),
                "threshold_step must be finite and greater than or equal to 0",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.8\nthreshold_step = 0.11",
                ),
                "base_threshold + 2 * threshold_step must be at most 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmax_output_tokens = 0\nescalation = { confirmations = 2 }",
                ),
                "max_output_tokens must be at least 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmessage_hash_fallback = true",
                ),
                "message_hash_fallback requires session_affinity",
            ),
            (
                VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"),
                "unsupported schema_version 2",
            ),
            (
                VALID_CONFIG.replace("[targets.strong]", "[targets.\" strong \"]"),
                "target name must be non-empty and have no surrounding whitespace",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\ncontext_window = 0",
                ),
                "route random context_window must be greater than zero",
            ),
        ];

        for (toml, expected) in cases {
            assert!(
                error_message(&toml).contains(expected),
                "expected error containing {expected}"
            );
        }
    }

    #[test]
    fn accepts_duplicate_target_model_ids_on_one_client() -> ServerResult<()> {
        // Two targets share one model id on one client. The client keeps one and drops the
        // other, so the build warns and still succeeds, and both routes resolve. Serving one
        // model under two route names this way is allowed; pointing both routes at one target
        // is the tidier form.
        const SAME_MODEL_TWO_ROUTES: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.fast]
id = "gpt-4o"
llm_client = "primary"

[targets.smart]
id = "gpt-4o"
llm_client = "primary"

[routes.fast]
id = "switchyard/fast"
type = "passthrough"
target = "fast"

[routes.smart]
id = "switchyard/smart"
type = "passthrough"
target = "smart"
"#;
        let state = server_state_from_toml(SAME_MODEL_TWO_ROUTES)?;
        assert_eq!(
            state.models().collect::<Vec<_>>(),
            ["switchyard/fast", "switchyard/smart"]
        );
        Ok(())
    }

    #[test]
    fn accepts_same_model_id_on_different_llm_clients() -> ServerResult<()> {
        // The same model id served by two llm clients never collides (each client keys its own
        // models), so cross-provider A/B builds with no warning; only a repeat within one client
        // warns.
        const CROSS_PROVIDER: &str = r#"
schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.azure]
format = "openai_chat"
base_url = "https://azure.test/v1"

[targets.openai]
id = "gpt-4o"
llm_client = "openai"

[targets.azure]
id = "gpt-4o"
llm_client = "azure"

[routes.openai]
id = "switchyard/openai-gpt4o"
type = "passthrough"
target = "openai"

[routes.azure]
id = "switchyard/azure-gpt4o"
type = "passthrough"
target = "azure"
"#;
        server_state_from_toml(CROSS_PROVIDER)?;
        Ok(())
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> ServerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        server_state_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_session_affinity_with_message_hash_fallback() -> ServerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nthreshold_step = 0.1\nsession_affinity = true\nmessage_hash_fallback = true",
        );
        server_state_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn target_extra_body_is_parsed_and_applied_to_its_backend() -> ServerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             extra_body = { service_tier = \"priority\", \
             chat_template_kwargs = { enable_thinking = false } }",
            1,
        );
        let config: ServerConfig = toml::from_str(&configured)
            .map_err(|error| ServerError::new(format!("failed to parse config: {error}")))?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(ServerError::new("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(ServerError::new("primary llm client is missing"));
        };
        let backend = build_backend("primary", client, &target.extra_body)?;

        assert_eq!(
            backend.extra_body().get("service_tier"),
            Some(&json!("priority"))
        );
        assert_eq!(
            backend
                .extra_body()
                .get("chat_template_kwargs")
                .and_then(|value| value.get("enable_thinking")),
            Some(&json!(false))
        );
        Ok(())
    }

    #[test]
    fn retry_budget_defaults_and_accepts_an_override() -> ServerResult<()> {
        let default: ServerConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            ServerError::new(format!("failed to parse default config: {error}"))
        })?;
        let Some(primary) = default.llm_clients.get("primary") else {
            return Err(ServerError::new("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, DEFAULT_MAX_RETRIES);

        let explicit = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 0",
            1,
        );
        let config: ServerConfig = toml::from_str(&explicit).map_err(|error| {
            ServerError::new(format!("failed to parse explicit retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(ServerError::new("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, 0);
        Ok(())
    }

    #[test]
    fn rejects_headers_that_switchyard_sets() {
        let cases = [
            (
                "base_url = \"https://example.test/v1\"",
                "base_url = \"https://example.test/v1\"\n\
                 extra_headers = { AUTHORIZATION = \"Bearer custom-key\" }",
                "AUTHORIZATION",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"X-Api-Key\" = \"custom-key\" }",
                "X-Api-Key",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"ANTHROPIC-VERSION\" = \"custom-version\" }",
                "ANTHROPIC-VERSION",
            ),
        ];

        for (original, replacement, header) in cases {
            let configured = VALID_CONFIG.replacen(original, replacement, 1);
            let error = error_message(&configured);
            assert!(
                error.contains(&format!("extra_headers cannot set {header:?}")),
                "expected {header} to be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn accepts_additional_headers() -> ServerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { X-Inference-Priority = \"batch\" }",
            1,
        );

        server_state_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn retry_budget_rejects_negative_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = -1",
            1,
        );
        assert!(error_message(&invalid).contains("max_retries"));
    }

    #[test]
    fn retry_budget_rejects_excessive_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 11",
            1,
        );
        assert!(
            error_message(&invalid).contains("llm client primary max_retries must be at most 10")
        );
    }

    #[test]
    fn api_key_environment_reference_is_validated() {
        let missing = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\napi_key_env = \"SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET\"",
            1,
        );
        assert!(error_message(&missing).contains("SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET"));

        const EMPTY_KEY_ENV: &str = "SWITCHYARD_CONFIG_TEST_EMPTY_KEY";
        unsafe {
            // "unsafe" is for concurrent reads and writes, very rare
            std::env::set_var(EMPTY_KEY_ENV, "");
        }
        let empty = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{EMPTY_KEY_ENV}\""),
            1,
        );
        let message = error_message(&empty);
        unsafe {
            std::env::remove_var(EMPTY_KEY_ENV);
        }
        assert!(message.contains("is empty"));
    }

    #[test]
    fn forward_auth_rejects_conflicting_credentials() {
        let competing_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
                 forward_auth = true\n\
                 api_key_env = \"UNUSED_TEST_KEY\"",
            1,
        );
        assert!(
            error_message(&competing_auth).contains("cannot set both forward_auth and api_key_env")
        );

        let static_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test\"",
            "base_url = \"https://example.test\"\n\
                 forward_auth = true\n\
                 extra_headers = { Authorization = \"static-value\" }",
            1,
        );
        assert!(error_message(&static_auth).contains("extra_headers cannot set \"Authorization\""));

        let static_beta = static_auth.replace("Authorization", "anthropic-beta");
        assert!(
            error_message(&static_beta).contains("extra_headers cannot set \"anthropic-beta\"")
        );

        for header in ["chatgpt-account-id", "x-openai-fedramp"] {
            let static_context = VALID_CONFIG.replacen(
                "base_url = \"https://example.test/v1\"",
                &format!(
                    "base_url = \"https://example.test/v1\"\n\
                     forward_auth = true\n\
                     extra_headers = {{ \"{header}\" = \"static-value\" }}"
                ),
                1,
            );
            assert!(
                error_message(&static_context)
                    .contains(&format!("extra_headers cannot set \"{header}\""))
            );
        }
    }
}
