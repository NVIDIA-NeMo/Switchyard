// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-1 TOML deployment loading for the shared runner.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_llm_client::{
    AuxiliaryOperation, Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    SystemOneClient, TranslatingLlmClient,
};
use switchyard_protocol::{ModelId, RoutedDecisionClient, RoutedLlmClient, WireFormat};

use crate::eligibility::{EligibilityPolicy, Requirements, validate_labels};
use crate::route::ExecutionLane;
use crate::runner::ModelPropertiesProbe;
use crate::{
    AlgorithmSpec, AuxiliaryTarget, CallerAuthKind, DecisionTarget, ModelCapabilities, Route,
    Runner, RunnerError,
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const MAX_CONFIGURED_RETRIES: u32 = 10;

/// This function returns a deployment's declared credential variable names without reading their values.
/// It parses the deployment and checks its schema version before returning names from LLM and decision clients.
pub fn api_key_variables(source: &str) -> Result<BTreeSet<String>, RunnerError> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|_| {
        RunnerError::configuration(
            "Could not parse the server config to read credential references.",
        )
    })?;
    if config.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(RunnerError::configuration(
            "Unsupported server config schema version.",
        ));
    }
    let mut variables: BTreeSet<_> = config
        .llm_clients
        .values()
        .filter_map(|client| client.api_key_env.clone())
        .collect();
    for client in config.decision_clients.values() {
        let DecisionClientConfig::SystemOne { api_key_env, .. } = client;
        variables.insert(api_key_env.clone());
    }
    if variables
        .iter()
        .any(|name| name.trim().is_empty() || name.contains(['\0', '=']))
    {
        return Err(RunnerError::configuration(
            "Invalid credential environment variable name.",
        ));
    }
    Ok(variables)
}

type RunnerResult<T> = Result<T, RunnerError>;

pub(crate) fn load_runner(path: impl AsRef<Path>) -> RunnerResult<Runner> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|error| {
        RunnerError::configuration_source(
            format!("failed to read server config {}: {error}", path.display()),
            error,
        )
    })?;
    runner_from_toml(&source).map_err(|error| {
        RunnerError::configuration_source(
            format!("invalid server config {}: {error}", path.display()),
            error,
        )
    })
}

pub(crate) fn runner_from_toml(source: &str) -> RunnerResult<Runner> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|error| {
        RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
    })?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeploymentConfig {
    schema_version: u32,
    fallback_client: Option<String>,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    #[serde(default)]
    decision_clients: BTreeMap<String, DecisionClientConfig>,
    #[serde(default)]
    decision_targets: BTreeMap<String, DecisionTargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
}

#[derive(Debug)]
struct RouteConfig {
    id: ModelId,
    context_window: Option<u32>,
    tool_calling: Option<bool>,
    reasoning: Option<bool>,
    vision: Option<bool>,
    eligibility: Option<EligibilityConfig>,
    algorithm: AlgorithmSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EligibilityConfig {
    #[serde(default)]
    require_explicit_requirements: bool,
    accepted_requirement_sets: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    target_overrides: BTreeMap<String, TargetOverrides>,
    #[serde(default)]
    requirement_scope: RequirementScope,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequirementScope {
    #[default]
    Request,
    Task,
}

struct TargetPromptPolicy {
    prompts: HashMap<ModelId, String>,
    routing_answer_target: Option<ModelId>,
}

type LaneTargets<'a> = BTreeMap<&'a str, &'a TargetConfig>;
type TargetOverrides = BTreeMap<String, String>;

// Replacements bind logical names once; mappings never chain.
fn bound_target<'a>(name: &'a str, overrides: &'a TargetOverrides) -> &'a str {
    overrides.get(name).map_or(name, String::as_str)
}

#[derive(Clone, Copy)]
struct ResolvedDecisionTarget<'a> {
    model: &'a ModelId,
    client: &'a Arc<dyn RoutedDecisionClient>,
}

impl<'de> Deserialize<'de> for RouteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut table = toml::Table::deserialize(deserializer)?;
        let id = take_required(&mut table, "id")?;
        let context_window = take_optional(&mut table, "context_window")?;
        let tool_calling = take_optional(&mut table, "tool_calling")?;
        let reasoning = take_optional(&mut table, "reasoning")?;
        let vision = take_optional(&mut table, "vision")?;
        let eligibility = take_optional(&mut table, "eligibility")?;
        let algorithm = AlgorithmSpec::deserialize(toml::Value::Table(table))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id,
            context_window,
            tool_calling,
            reasoning,
            vision,
            eligibility,
            algorithm,
        })
    }
}

fn take_required<T, E>(table: &mut toml::Table, name: &'static str) -> Result<T, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    let value = table.remove(name).ok_or_else(|| E::missing_field(name))?;
    T::deserialize(value).map_err(E::custom)
}

fn take_optional<T, E>(table: &mut toml::Table, name: &'static str) -> Result<Option<T>, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    table
        .remove(name)
        .map(|value| T::deserialize(value).map_err(E::custom))
        .transpose()
}

impl RouteConfig {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            context_window: self.context_window,
            tool_calling: self.tool_calling,
            reasoning: self.reasoning,
            vision: self.vision,
        }
    }
}

impl DeploymentConfig {
    fn decision_target(&self, name: &str) -> Option<DecisionTarget> {
        let target = self.targets.get(name)?;
        let client = self.llm_clients.get(&target.llm_client)?;
        Some(DecisionTarget {
            target: name.to_string(),
            model: target.id.clone(),
            format: client.format.wire_format(),
            base_url: client.base_url.as_str().to_string(),
            extra_body: target.extra_body.clone(),
        })
    }

    fn build(self) -> RunnerResult<Runner> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(RunnerError::configuration(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        for (name, target) in &self.targets {
            validate_labels(&target.labels).map_err(|error| {
                RunnerError::configuration_source(
                    format!("target {name} has invalid labels: {error}"),
                    error,
                )
            })?;
        }
        for (name, target) in &self.decision_targets {
            validate_labels(&target.labels).map_err(|error| {
                RunnerError::configuration_source(
                    format!("decision target {name} has invalid labels: {error}"),
                    error,
                )
            })?;
        }

        let mut route_names_by_id = HashMap::new();
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), &config.id)?;
            if let Some(first_route_name) =
                route_names_by_id.insert(config.id.as_str(), route_name.as_str())
            {
                return Err(RunnerError::configuration(format!(
                    "routes {first_route_name} and {route_name} both use id {}; route ids must be unique",
                    config.id
                )));
            }
        }

        // The LLM client keeps one backend per model id, so two targets naming the same model on
        // the same client share it. That is harmless when their request settings agree (an alias
        // for a different system prompt, say) and silently wrong when they do not: the second
        // target's reasoning_effort or extra_body would never reach the wire.
        let mut seen_client_model_ids: HashMap<(&str, &str), (&String, &TargetConfig)> =
            HashMap::new();
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            match seen_client_model_ids.entry((target.llm_client.as_str(), target.id.as_str())) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert((target_name, target));
                }
                std::collections::hash_map::Entry::Occupied(slot) => {
                    let (first_name, first) = slot.get();
                    if first.labels != target.labels {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} share a model and llm client but disagree on labels"
                        )));
                    }
                    if first.reasoning_effort != target.reasoning_effort
                        || first.extra_body != target.extra_body
                        || first.omit_body_fields != target.omit_body_fields
                    {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} both name model {} on llm client {} but with different reasoning_effort, extra_body, or omit_body_fields; one target per model id is kept, so give each its own model id or llm client",
                            target.id, target.llm_client
                        )));
                    }
                    tracing::warn!(
                        "target {target_name} reuses model id {} on llm client {}; only one target per id is kept and the other is dropped. Give each target a unique model id, or point both routes at one target.",
                        target.id,
                        target.llm_client
                    );
                }
            }
        }

        let mut provider_api_keys = Vec::new();
        let clients = self.build_clients(&mut provider_api_keys)?;
        let decision_clients = self.build_decision_clients(&mut provider_api_keys)?;
        let mut decision_approvals = HashMap::new();
        for (name, target) in &self.decision_targets {
            validate_value("decision target name", name)?;
            validate_value(&format!("decision target {name} id"), &target.id)?;
            if self.targets.contains_key(name) {
                return Err(RunnerError::configuration(format!(
                    "target {name} is defined as both an LLM and decision target"
                )));
            }
            if !decision_clients.contains_key(&target.decision_client) {
                return Err(RunnerError::configuration(format!(
                    "decision target {name} references unknown decision client {}",
                    target.decision_client
                )));
            }
            if decision_approvals
                .insert((&target.decision_client, &target.id), &target.labels)
                .is_some_and(|approved| approved != &target.labels)
            {
                return Err(RunnerError::configuration(format!(
                    "decision target {name} shares a model and client with an alias but disagrees on labels"
                )));
            }
        }
        let fallback_base_url = self.fallback_base_url()?;
        let mut routes = Vec::with_capacity(self.routes.len());
        let mut model_properties_probes = BTreeMap::new();
        let no_overrides = TargetOverrides::new();
        for (route_name, config) in &self.routes {
            let standard_targets = self.resolve_lane_targets(&config.algorithm, &no_overrides)?;
            let decision = config
                .algorithm
                .decision_judge()
                .map(|(name, _)| self.resolve_decision_target(route_name, name, &decision_clients))
                .transpose()?;
            let capabilities = config.capabilities();
            if capabilities.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            if let AlgorithmSpec::Vgr { config: vgr } = &config.algorithm
                && let Some(probe) = self.build_model_properties_probe(&vgr.local_target, &clients)
            {
                model_properties_probes.insert(config.id.clone(), probe);
            }
            let eligibility = self.build_eligibility(
                route_name,
                config,
                &standard_targets,
                &decision_clients,
                &clients,
            )?;
            let (standard, caller_auth) = self.build_lane(
                route_name,
                &config.algorithm,
                &standard_targets,
                decision,
                &clients,
                config.eligibility.is_some(),
            )?;
            let completion_names = config.algorithm.routing_target_names();
            let mut target_names = completion_names.clone();
            if let Some(eligibility) = &config.eligibility {
                for overrides in eligibility.target_overrides.values() {
                    target_names.extend(
                        completion_names
                            .iter()
                            .filter_map(|name| overrides.get(*name).map(String::as_str)),
                    );
                }
            }
            let mut model_clients = HashMap::new();
            for name in &target_names {
                let target = &self.targets[*name];
                if model_clients
                    .insert(&target.id, &target.llm_client)
                    .is_some_and(|first| first != &target.llm_client)
                {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} uses model {} on different clients across eligibility groups; use distinct model ids",
                        target.id
                    )));
                }
            }
            let decision_targets = target_names
                .into_iter()
                .filter_map(|name| self.decision_target(name))
                .collect();
            let mut route = Route::from_lane(standard, caller_auth, capabilities, decision_targets);
            if let Some((policy, lanes)) = eligibility {
                route = route.with_eligibility(policy, lanes);
            }
            routes.push((config.id.clone(), route));
        }
        let runner = Runner::new(routes)
            .with_fallback_url(fallback_base_url)
            .with_provider_api_keys(provider_api_keys)
            .with_model_properties_probes(model_properties_probes);
        Ok(runner)
    }

    // Build only configured requirement sets; every callable target must satisfy every label.
    fn build_eligibility(
        &self,
        route_name: &str,
        route_config: &RouteConfig,
        standard_targets: &LaneTargets<'_>,
        decision_clients: &BTreeMap<String, Arc<dyn RoutedDecisionClient>>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> RunnerResult<Option<(EligibilityPolicy, BTreeMap<Requirements, ExecutionLane>)>> {
        let Some(config) = &route_config.eligibility else {
            return Ok(None);
        };
        if !route_config.algorithm.supports_target_restrictions() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot use eligibility with prefill_router"
            )));
        }
        if self.uses_forward_auth(standard_targets) {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot use eligibility with forward_auth"
            )));
        }
        let policy = EligibilityPolicy::new(
            config.require_explicit_requirements,
            config.accepted_requirement_sets.values().cloned().collect(),
            matches!(&config.requirement_scope, RequirementScope::Task),
        )
        .map_err(|error| {
            RunnerError::configuration_source(
                format!("route {route_name} has invalid eligibility: {error}"),
                error,
            )
        })?;
        let empty_overrides = TargetOverrides::new();
        let mut overrides_by_set = BTreeMap::new();
        if let Some(name) = config
            .target_overrides
            .keys()
            .find(|name| !config.accepted_requirement_sets.contains_key(*name))
        {
            return Err(RunnerError::configuration(format!(
                "route {route_name} target_overrides references unknown requirement set {name}"
            )));
        }
        for (name, labels) in &config.accepted_requirement_sets {
            let labels = labels.iter().cloned().collect::<Requirements>();
            let overrides = config
                .target_overrides
                .get(name)
                .unwrap_or(&empty_overrides);
            if labels.is_empty() && !overrides.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} cannot override targets for the empty requirement set"
                )));
            }
            if overrides_by_set
                .insert(labels, overrides)
                .is_some_and(|previous| previous != overrides)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} duplicate requirement sets have different target_overrides"
                )));
            }
        }
        let mut lanes = BTreeMap::new();
        let mut callable = route_config
            .algorithm
            .callable_target_names()
            .into_iter()
            .collect::<BTreeSet<_>>();
        if let Some((_, decision)) = route_config.algorithm.decision_judge() {
            callable.extend(decision.candidates.values().map(String::as_str));
        }
        for requirements in policy
            .accepted_requirement_sets()
            .iter()
            .filter(|labels| !labels.is_empty())
        {
            let overrides = overrides_by_set
                .get(requirements)
                .copied()
                .unwrap_or(&empty_overrides);
            for (source, replacement) in overrides {
                if !callable.contains(source.as_str()) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} target_overrides source {source} is not used by the algorithm"
                    )));
                }
                let same_kind = (self.targets.contains_key(source)
                    && self.targets.contains_key(replacement))
                    || (self.decision_targets.contains_key(source)
                        && self.decision_targets.contains_key(replacement));
                if !same_kind {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} target_overrides replacement {replacement} must exist and match the target kind of {source}"
                    )));
                }
                let labels = self
                    .targets
                    .get(replacement)
                    .map(|target| &target.labels)
                    .or_else(|| {
                        self.decision_targets
                            .get(replacement)
                            .map(|target| &target.labels)
                    });
                if !labels.is_some_and(|labels| labels.is_superset(requirements)) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} target_overrides replacement {replacement} does not satisfy target requirements"
                    )));
                }
            }
            let approved = callable
                .iter()
                .copied()
                .filter(|name| {
                    let name = bound_target(name, overrides);
                    self.targets
                        .get(name)
                        .map(|target| &target.labels)
                        .or_else(|| self.decision_targets.get(name).map(|target| &target.labels))
                        .is_some_and(|labels| labels.is_superset(requirements))
                })
                .collect();
            let algorithm = route_config
                .algorithm
                .restricted_to(&approved)
                .map_err(|error| {
                    RunnerError::configuration(format!(
                        "route {route_name} eligibility requirement set {requirements:?}: {error}"
                    ))
                })?;
            let targets = self.resolve_lane_targets(&algorithm, overrides)?;
            if self.uses_forward_auth(&targets) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} cannot use eligibility with forward_auth replacements"
                )));
            }
            if matches!(algorithm, AlgorithmSpec::Random { .. }) && !overrides.is_empty() {
                let mut models = HashMap::new();
                for name in algorithm.routing_target_names() {
                    let model = &targets[name].id;
                    if models
                        .insert(model, name)
                        .is_some_and(|previous| self.targets[previous].id != self.targets[name].id)
                    {
                        return Err(RunnerError::configuration(format!(
                            "route {route_name} target_overrides collapse distinct random targets onto model {model}"
                        )));
                    }
                }
            }
            let decision = algorithm
                .decision_judge()
                .map(|(name, _)| {
                    self.resolve_decision_target(
                        route_name,
                        bound_target(name, overrides),
                        decision_clients,
                    )
                })
                .transpose()?;
            let (lane, _) =
                self.build_lane(route_name, &algorithm, &targets, decision, clients, true)?;
            lanes.insert(requirements.clone(), lane);
        }
        Ok(Some((policy, lanes)))
    }

    fn uses_forward_auth(&self, targets: &LaneTargets<'_>) -> bool {
        targets.values().any(|target| {
            self.llm_clients
                .get(&target.llm_client)
                .is_some_and(|client| client.forward_auth)
        })
    }

    fn resolve_lane_targets<'a>(
        &'a self,
        algorithm: &'a AlgorithmSpec,
        overrides: &TargetOverrides,
    ) -> RunnerResult<LaneTargets<'a>> {
        let decision = algorithm.decision_judge().map(|(target, _)| target);
        let mut callable = algorithm.callable_target_names();
        if let Some((_, decision)) = algorithm.decision_judge() {
            callable.extend(
                decision
                    .candidates
                    .values()
                    .filter(|name| overrides.contains_key(*name))
                    .map(String::as_str),
            );
        }
        callable.retain(|name| Some(*name) != decision);
        callable.sort_unstable();
        callable.dedup();
        callable
            .into_iter()
            .map(|name| {
                let bound = bound_target(name, overrides);
                let config = self.targets.get(bound).ok_or_else(|| {
                    RunnerError::configuration(format!("route references unknown target {bound}"))
                })?;
                Ok((name, config))
            })
            .collect()
    }

    fn resolve_decision_target<'a>(
        &'a self,
        route_name: &str,
        name: &str,
        clients: &'a BTreeMap<String, Arc<dyn RoutedDecisionClient>>,
    ) -> RunnerResult<ResolvedDecisionTarget<'a>> {
        let target = self.decision_targets.get(name).ok_or_else(|| {
            RunnerError::configuration(format!(
                "route {route_name} requires unknown decision target {name}"
            ))
        })?;
        let client = clients.get(&target.decision_client).ok_or_else(|| {
            RunnerError::configuration(format!(
                "decision target {name} references unknown decision client {}",
                target.decision_client
            ))
        })?;
        Ok(ResolvedDecisionTarget {
            model: &target.id,
            client,
        })
    }

    fn build_lane(
        &self,
        route_name: &str,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
        decision: Option<ResolvedDecisionTarget<'_>>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        restrict_targets: bool,
    ) -> RunnerResult<(ExecutionLane, Option<CallerAuthKind>)> {
        let mut model_ids = targets
            .iter()
            .map(|(name, target)| ((*name).to_string(), target.id.clone()))
            .collect::<BTreeMap<_, _>>();
        if let Some((_, config)) = route.decision_judge() {
            for candidate in config.candidates.values() {
                if !model_ids.contains_key(candidate) {
                    let target = self.targets.get(candidate).ok_or_else(|| {
                        RunnerError::configuration(format!(
                            "route {route_name} completion and candidate target {candidate} must be an LLM target"
                        ))
                    })?;
                    model_ids.insert(candidate.clone(), target.id.clone());
                }
            }
        }
        if let (Some((name, _)), Some(decision)) = (route.decision_judge(), decision) {
            model_ids.insert(name.to_string(), decision.model.clone());
        }
        let (algorithm, mut models) = route
            .build_with_runtime_models(route_name, &model_ids)
            .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
        let (route_clients, caller_auth) =
            self.build_route_clients(route_name, route, targets, decision, clients)?;
        let anthropic_auxiliary_target =
            self.build_anthropic_auxiliary_target(route, targets, clients);
        let responses_auxiliary_target =
            self.build_responses_auxiliary_target(route, targets, clients);
        if restrict_targets {
            models = models.with_target_restriction();
        }
        Ok((
            ExecutionLane::new(
                algorithm,
                route_clients,
                anthropic_auxiliary_target,
                responses_auxiliary_target,
                models,
            ),
            caller_auth,
        ))
    }

    fn build_clients(
        &self,
        provider_api_keys: &mut Vec<String>,
    ) -> RunnerResult<BTreeMap<String, Arc<TranslatingLlmClient>>> {
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for (name, client_config) in &self.llm_clients {
            validate_value("llm client name", name)?;
            let backend = build_backend(
                name,
                client_config,
                &BTreeMap::new(),
                &BTreeSet::new(),
                None,
            )?;
            let (Backend::OpenAiChat(config)
            | Backend::OpenAiResponses(config)
            | Backend::Anthropic(config)) = backend;
            if let Some(key) = config.api_key {
                provider_api_keys.push(key);
            }
        }
        for (target_name, target) in &self.targets {
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| {
                    RunnerError::configuration("validated llm client was not initialized")
                })?;
            if let Some(effort) = &target.reasoning_effort {
                if effort.trim().is_empty() {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort must not be empty"
                    )));
                }
                if matches!(client_config.format, ClientFormat::AnthropicMessages) {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort is only supported on openai_chat and openai_responses clients"
                    )));
                }
            }
            model_configs.push(ModelConfig::new(
                target.id.clone(),
                build_backend(
                    &target.llm_client,
                    client_config,
                    &target.extra_body,
                    &target.omit_body_fields,
                    target.reasoning_effort.clone(),
                )?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let mut client = TranslatingLlmClient::new(&model_configs)
                .map_err(|error| RunnerError::configuration(error.to_string()))?;
            if !self.llm_clients[&name].fallback_on_rate_limit {
                client = client.without_rate_limit_cooldown();
            }
            clients.insert(name, Arc::new(client));
        }
        Ok(clients)
    }

    fn build_decision_clients(
        &self,
        provider_api_keys: &mut Vec<String>,
    ) -> RunnerResult<BTreeMap<String, Arc<dyn RoutedDecisionClient>>> {
        self.decision_clients
            .iter()
            .map(|(name, config)| {
                validate_value("decision client name", name)?;
                let DecisionClientConfig::SystemOne {
                    endpoint,
                    api_key_env,
                    timeout_ms,
                } = config;
                if *timeout_ms == 0 {
                    return Err(RunnerError::configuration(format!(
                        "decision client {name} timeout_ms must be at least 1"
                    )));
                }
                let api_key = read_api_key(&format!("decision client {name}"), api_key_env)?;
                let client = SystemOneClient::new(
                    endpoint.0.clone(),
                    api_key.clone(),
                    Duration::from_millis(*timeout_ms),
                )
                .map_err(|error| RunnerError::configuration(error.to_string()))?;
                provider_api_keys.push(api_key);
                Ok((
                    name.clone(),
                    Arc::new(client) as Arc<dyn RoutedDecisionClient>,
                ))
            })
            .collect()
    }

    /// Builds the client router for one route. The second value is the caller credential family
    /// that the route's forwarding clients need, or `None` when no client forwards the caller's
    /// credential. A request through the other family's APIs fails before any upstream call.
    fn build_route_clients(
        &self,
        route_name: &str,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
        decision: Option<ResolvedDecisionTarget<'_>>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> RunnerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let TargetPromptPolicy {
            prompts,
            routing_answer_target,
        } = self.build_route_target_prompts(route_name, route, targets)?;
        let mut by_model = HashMap::new();
        let mut targets_by_model: HashMap<&str, (&str, &TargetConfig)> = HashMap::new();
        let mut caller_auth = None;
        let mut has_mixed_families = false;
        let mut forwarding_origins = BTreeSet::new();
        for (&name, &target) in targets {
            if let Some((first_name, first)) = targets_by_model.insert(&target.id, (name, target))
                && first.llm_client != target.llm_client
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} targets {first_name} and {} use model {} on different llm clients; execution is keyed by model id, so use distinct model ids within this route or put these targets in separate routes",
                    name, target.id
                )));
            }
            let client = clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!("target {} has no constructed llm client", name))
            })?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {} references unknown llm client {}",
                    name, target.llm_client
                ))
            })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                has_mixed_families |= caller_auth.is_some_and(|kind| kind != target_auth);
                caller_auth = Some(target_auth);
                forwarding_origins.insert(client_config.base_url.0.origin().ascii_serialization());
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(target.id.clone(), client);
        }
        // Only forwarding clients count: an api_key_env client sends the server's own key.
        // Forwarding clients must share one credential family unless they all use the same
        // scheme, host, and port, such as one LLM gateway that accepts the caller's gateway key
        // on every endpoint. Such a route serves Chat Completions and Responses callers because
        // Anthropic clients forward the caller's `authorization` header unchanged.
        if has_mixed_families {
            if forwarding_origins.len() > 1 {
                let origins = Vec::from_iter(forwarding_origins).join(", ");
                return Err(RunnerError::configuration(format!(
                    "route {route_name} cannot forward both Anthropic and OpenAI caller credentials to different origins ({origins}); point all of its forwarding clients at one origin (same scheme, host, and port), such as an LLM gateway, or set api_key_env instead of forward_auth on one provider's clients"
                )));
            }
            caller_auth = Some(CallerAuthKind::OpenAi);
        }
        let completion_targets = route
            .routing_target_names()
            .into_iter()
            .map(|name| targets[name].id.clone())
            .collect::<Vec<_>>();
        let decisions_by_model = decision
            .map(|decision| HashMap::from([(decision.model.clone(), Arc::clone(decision.client))]))
            .unwrap_or_default();
        let router = ClientRouter::new_with_decision_clients(
            by_model,
            decisions_by_model,
            prompts,
            routing_answer_target,
            &completion_targets,
        )
        .without_rate_limit_fallback(
            route
                .routing_target_names()
                .into_iter()
                .filter_map(|name| {
                    let target = targets[name];
                    (!self.llm_clients[&target.llm_client].fallback_on_rate_limit)
                        .then(|| target.id.clone())
                })
                .collect(),
        );
        Ok((router, caller_auth))
    }

    /// Builds the effective prompt policy for this route's completion targets.
    fn build_route_target_prompts(
        &self,
        route_name: &str,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
    ) -> RunnerResult<TargetPromptPolicy> {
        let mut prompts = HashMap::new();
        let mut aliases = HashMap::<&ModelId, Option<&str>>::new();
        for name in route.routing_target_names() {
            let target = targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let prompt = target.system_prompt.as_deref();
            if aliases
                .insert(&target.id, prompt)
                .is_some_and(|configured| configured != prompt)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} maps completion target aliases to model {} with different system_prompt values",
                    target.id
                )));
            }
            if let Some(prompt) = prompt {
                prompts.insert(target.id.clone(), prompt.to_string());
            }
        }
        let mut policy = TargetPromptPolicy {
            prompts,
            routing_answer_target: None,
        };
        let Some((response_name, dependency_name)) = route.routing_response_and_dependency() else {
            return Ok(policy);
        };
        let response = targets.get(response_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {response_name}"))
        })?;
        if !policy.prompts.contains_key(&response.id) {
            return Ok(policy);
        }
        let dependency = targets.get(dependency_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {dependency_name}"))
        })?;
        if response.id == dependency.id {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot apply system_prompt to target {response_name}: model {} is also used by routing-only target {dependency_name}",
                response.id,
            )));
        }
        policy.routing_answer_target = Some(response.id.clone());
        Ok(policy)
    }

    fn fallback_base_url(&self) -> RunnerResult<Option<String>> {
        let Some(name) = &self.fallback_client else {
            return Ok(None);
        };
        let config = self.llm_clients.get(name).ok_or_else(|| {
            RunnerError::configuration(format!(
                "fallback_client references unknown llm client {name}"
            ))
        })?;
        Ok(Some(config.base_url.as_str().to_string()))
    }

    fn build_model_properties_probe(
        &self,
        local_target_name: &str,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<ModelPropertiesProbe> {
        let local_target = self.targets.get(local_target_name)?;
        let client_config = self.llm_clients.get(&local_target.llm_client)?;
        if !matches!(
            client_config.context_discovery,
            Some(ContextDiscovery::LlamaCpp)
        ) {
            return None;
        }
        let client = clients.get(&local_target.llm_client)?;
        Some(ModelPropertiesProbe {
            model: local_target.id.clone(),
            format: client_config.format.wire_format(),
            client: Arc::clone(client),
        })
    }

    fn build_anthropic_auxiliary_target(
        &self,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route
            .routing_target_names()
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let config = targets.get(name)?;
                let target = self.build_auxiliary_target(
                    config,
                    clients,
                    AuxiliaryOperation::AnthropicCountTokens,
                )?;
                Some((count_tokens_priority(name, &target.model), index, target))
            })
            .min_by_key(|(priority, index, _)| (*priority, *index))
            .map(|(_, _, target)| target)
    }

    fn build_responses_auxiliary_target(
        &self,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route.routing_target_names().into_iter().find_map(|name| {
            let target = *targets.get(name)?;
            self.build_auxiliary_target(target, clients, AuxiliaryOperation::ResponsesInputTokens)
        })
    }

    fn build_auxiliary_target(
        &self,
        target: &TargetConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        operation: AuxiliaryOperation,
    ) -> Option<AuxiliaryTarget> {
        let client = clients.get(&target.llm_client)?;
        client
            .supports_auxiliary(&target.id, operation)
            .then(|| AuxiliaryTarget {
                model: target.id.clone(),
                client: client.clone(),
            })
    }
}

fn count_tokens_priority(target_name: &str, model_id: &ModelId) -> usize {
    let target_name = target_name.to_ascii_lowercase();
    let model_id = model_id.to_ascii_lowercase();
    ["opus", "sonnet", "haiku"]
        .iter()
        .position(|hint| target_name.contains(hint) || model_id.contains(hint))
        .unwrap_or(3)
}

/// A client endpoint, parsed when the config loads rather than checked afterwards.
///
/// Holding a `HttpBaseUrl` is proof the value is an absolute HTTP(S) URL, so no
/// later stage has to re-check it or can forget to.
#[derive(Clone, Debug)]
struct HttpBaseUrl(reqwest::Url);

impl HttpBaseUrl {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl<'de> Deserialize<'de> for HttpBaseUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let url = reqwest::Url::parse(raw.trim()).map_err(|error| {
            serde::de::Error::custom(format!("base_url must be an absolute HTTP(S) URL: {error}"))
        })?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(serde::de::Error::custom(
                "base_url must be an absolute HTTP(S) URL",
            ));
        }
        Ok(Self(url))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: HttpBaseUrl,
    /// Opt-in context discovery for VGR's local target. Unset disables discovery.
    context_discovery: Option<ContextDiscovery>,
    api_key_env: Option<String>,
    #[serde(default)]
    forward_auth: bool,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
    #[serde(default = "default_fallback_on_rate_limit")]
    fallback_on_rate_limit: bool,
    /// Cooldown after an exhausted transient completion failure. Defaults to 5 seconds; zero disables it.
    #[serde(default = "default_failure_cooldown_ms")]
    failure_cooldown_ms: u64,
    /// Deadline in milliseconds for all attempts and the complete response. Unset is unbounded.
    timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ContextDiscovery {
    LlamaCpp,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case", deny_unknown_fields)]
enum DecisionClientConfig {
    SystemOne {
        endpoint: HttpBaseUrl,
        api_key_env: String,
        timeout_ms: u64,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionTargetConfig {
    id: ModelId,
    decision_client: String,
    #[serde(default)]
    labels: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    labels: BTreeSet<String>,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
    /// Top-level request fields dropped from the outbound body before `extra_body` is merged.
    /// For providers that reject an otherwise standard field.
    #[serde(default)]
    omit_body_fields: BTreeSet<String>,
    system_prompt: Option<String>,
    /// Reasoning effort forced on every request to this target, replacing the caller's value.
    /// Only meaningful on `openai_chat` and `openai_responses` clients.
    reasoning_effort: Option<String>,
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
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::OpenAiChat => WireFormat::OpenAiChat,
            Self::OpenAiResponses => WireFormat::OpenAiResponses,
            Self::AnthropicMessages => WireFormat::AnthropicMessages,
        }
    }

    const fn caller_auth_kind(self) -> CallerAuthKind {
        match self {
            Self::AnthropicMessages => CallerAuthKind::Anthropic,
            Self::OpenAiChat | Self::OpenAiResponses => CallerAuthKind::OpenAi,
        }
    }
}

fn build_backend(
    client_name: &str,
    config: &LlmClientConfig,
    extra_body: &BTreeMap<String, Value>,
    omit_body_fields: &BTreeSet<String>,
    reasoning_effort: Option<String>,
) -> RunnerResult<Backend> {
    if config.max_retries > MAX_CONFIGURED_RETRIES {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} max_retries must be at most {MAX_CONFIGURED_RETRIES}"
        )));
    }
    if config.timeout_ms == Some(0) {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} timeout_ms must be at least 1"
        )));
    }
    if config.forward_auth && config.api_key_env.is_some() {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} cannot set both forward_auth and api_key_env"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| read_api_key(&format!("llm client {client_name}"), variable))
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: config.base_url.as_str().to_string(),
        api_key,
        forward_auth: config.forward_auth,
        extra_headers: config.extra_headers.clone(),
        extra_body: extra_body.clone(),
        omit_body_fields: omit_body_fields.clone(),
        reasoning_effort,
        max_retries: config.max_retries,
        failure_cooldown: Duration::from_millis(config.failure_cooldown_ms),
        timeout: config.timeout_ms.map(Duration::from_millis),
    };
    let backend = match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    };
    Ok(backend)
}

fn read_api_key(client: &str, variable: &str) -> RunnerResult<String> {
    if variable.trim().is_empty() {
        return Err(RunnerError::configuration(format!(
            "{client} api_key_env must not be empty"
        )));
    }
    // A name with `=` is often a pasted `NAME=key` line, so this error omits the name.
    if variable.contains(['\0', '=']) {
        return Err(RunnerError::configuration(format!(
            "{client} api_key_env must not contain '=' or NUL"
        )));
    }
    let api_key = std::env::var(variable).map_err(|error| {
        // VarError can contain the raw credential, so errors report only its category.
        let reason = match error {
            std::env::VarError::NotPresent => "environment variable not found",
            std::env::VarError::NotUnicode(_) => "environment variable is not valid Unicode",
        };
        RunnerError::configuration(format!(
            "{client} could not read api_key_env {variable}: {reason}"
        ))
    })?;
    if api_key.trim().is_empty() {
        return Err(RunnerError::configuration(format!(
            "{client} api_key_env {variable} is empty"
        )));
    }
    Ok(api_key)
}

// A function so that serde default can use it.
const fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

const fn default_fallback_on_rate_limit() -> bool {
    true
}

const fn default_failure_cooldown_ms() -> u64 {
    5000
}

fn validate_value(label: &str, value: &str) -> RunnerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(RunnerError::configuration(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_presentation_fields_are_split_from_the_algorithm() {
        let route: RouteConfig = toml::from_str(
            r#"
type = "random"
id = "switchyard/random"
context_window = 128000
tool_calling = true
reasoning = false
targets = ["fast", "strong"]
weights = [1.0, 2.0]
seed = 7
"#,
        )
        .expect("route should deserialize");

        assert_eq!(route.id, "switchyard/random");
        assert_eq!(route.context_window, Some(128_000));
        assert_eq!(route.tool_calling, Some(true));
        assert_eq!(route.reasoning, Some(false));
        assert_eq!(route.algorithm.routing_target_names(), ["fast", "strong"]);
    }

    #[test]
    fn route_algorithm_unknown_fields_are_rejected() {
        let error = toml::from_str::<RouteConfig>(
            r#"
type = "passthrough"
id = "switchyard/fast"
target = "fast"
bogus = true
"#,
        )
        .expect_err("unknown route keys must be rejected");

        assert!(error.to_string().contains("bogus"), "{error}");
    }

    #[test]
    fn deployment_errors_keep_one_path_context() {
        let path = Path::new("/definitely/missing/switchyard-routes.toml");
        let error = match load_runner(path) {
            Ok(_) => panic!("missing config should fail"),
            Err(error) => error,
        };
        let message = error.to_string();

        assert_eq!(message.matches(&path.display().to_string()).count(), 1);
        assert!(message.starts_with("failed to read server config"));
    }
}
#[cfg(test)]
mod deployment_tests {
    use super::*;
    use crate::mark_privacy_restricted;
    use serde_json::json;
    use switchyard_protocol::{Category, Message, Request, Role};

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

[routes.plan_execute]
id = "switchyard/plan-execute"
type = "plan_execute"
capable_target = "strong"
efficient_target = "weak"
"#;

    fn decision_privacy_config() -> String {
        let config = VALID_CONFIG
            .replace(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nlabels = [\"private\"]",
            )
            .replace(
                "id = \"weak/model\"",
                "id = \"weak/model\"\nlabels = [\"private\"]",
            );
        format!(
            r#"{config}

[decision_clients.privacy]
format = "system_one"
endpoint = "https://example.test/v1/systemone"
api_key_env = "PATH"
timeout_ms = 5000

[decision_targets.standard]
id = "decision/standard"
decision_client = "privacy"
labels = ["private"]

[routes.typed]
id = "switchyard/typed"
type = "llm_classifier"
mode = "capability"
classifier_target = "standard"
strong_target = "strong"
weak_target = "weak"

[routes.typed.decision]
cutoff = 0.4
candidates = {{ a = "strong", b = "weak" }}
evidence = {{}}

[routes.typed.eligibility]
accepted_requirement_sets = {{ private = ["private"] }}
"#
        )
    }

    fn external_privacy_config() -> String {
        let config = VALID_CONFIG
            .replace("id = \"strong/model\"", "id = \"strong/model\"\nlabels = [\"private\"]")
            .replace(
                "type = \"passthrough\"\ntarget = \"weak\"",
                "type = \"random\"\ntargets = [\"weak\", \"strong\"]\nweights = [3.0, 1.0]\nseed = 7",
            );
        format!(
            r#"{config}

[routes.passthrough.eligibility]
accepted_requirement_sets = {{ private = ["private"] }}
"#
        )
    }

    fn external_eligibility_config() -> String {
        external_privacy_config()
            .replace(
                "accepted_requirement_sets = { private = [\"private\"] }",
                "accepted_requirement_sets = { private_eu = [\"private\", \"eu\"] }",
            )
            .replace(
                "id = \"strong/model\"\nlabels = [\"private\"]",
                "id = \"strong/model\"\nlabels = [\"private\", \"eu\"]",
            )
            .replace(
                "id = \"weak/model\"",
                "id = \"weak/model\"\nlabels = [\"private\"]",
            )
    }

    fn task_retained_privacy_config() -> String {
        external_privacy_config().replace(
            "[routes.passthrough.eligibility]",
            "[routes.passthrough.eligibility]\nrequirement_scope = \"task\"",
        )
    }

    #[test]
    fn public_runner_from_toml_builds_a_deployment() -> RunnerResult<()> {
        let runner = Runner::from_toml(VALID_CONFIG)?;

        let classifier = runner
            .route("switchyard/classifier")
            .expect("classifier route should exist");
        let models = classifier.models();
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("classifier/model")]
        );
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("weak/model")]
        );
        assert_eq!(
            models.models_for(&Category::Capable),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("weak/model"), ModelId::from("strong/model")]
        );
        let plan_execute = runner
            .route("switchyard/plan-execute")
            .expect("plan-execute route should exist");
        assert_eq!(
            plan_execute.models().models_for(&Category::Capable),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            plan_execute.models().models_for(&Category::Efficient),
            [ModelId::from("weak/model")]
        );
        assert!(runner.route("switchyard/passthrough").is_some());
        Ok(())
    }

    #[test]
    fn privacy_requires_approved_roles_and_decision_targets() -> RunnerResult<()> {
        let configured = decision_privacy_config();
        let runner = Runner::from_toml(&configured)?;
        let route = runner
            .route("switchyard/typed")
            .expect("typed privacy route should exist");
        assert_eq!(
            route.models().models_for(&Category::Judge),
            [ModelId::from("decision/standard")]
        );

        for (config, expected) in [
            (
                configured.replace(
                    "decision_client = \"privacy\"\nlabels = [\"private\"]",
                    "decision_client = \"privacy\"",
                ),
                "required target standard does not satisfy target requirements",
            ),
            (
                configured.replace(
                    "id = \"weak/model\"\nlabels = [\"private\"]",
                    "id = \"weak/model\"",
                ),
                "decision candidate weak does not satisfy target requirements",
            ),
        ] {
            let message = error_message(&config);
            assert!(
                message.contains(expected),
                "expected {expected:?}, got {message}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn external_privacy_selects_the_restricted_lane() -> RunnerResult<()> {
        let configured = external_privacy_config();
        let runner = Runner::from_toml(&configured)?;
        let route = runner
            .route("switchyard/passthrough")
            .expect("privacy route should exist");

        let mut request = Request::default();
        request.llm_request.messages.push(Message::text(
            Role::User,
            "Contact user@example.test. Authorization: Bearer abcdefghijklmnop",
        ));
        let outcome = route.decide(request.clone()).await?;
        assert!(
            route
                .models()
                .models_for(&Category::Any)
                .contains(outcome.selected_model_id()?)
        );
        assert_eq!(
            &runner
                .describe_decision(&ModelId::from("switchyard/passthrough"), &outcome)
                .expect("selected target metadata")
                .selected
                .model,
            outcome.selected_model_id()?
        );
        mark_privacy_restricted(&mut request);
        assert_eq!(
            route.decide(request).await?.selected_model_id()?,
            "strong/model"
        );

        let no_private_target = configured.replace("labels = [\"private\"]", "labels = []");
        assert!(error_message(&no_private_target).contains("without an eligible target"));
        let missing_tier = format!(
            "{configured}\n[routes.plan_execute.eligibility]\naccepted_requirement_sets = {{ private = [\"private\"] }}\n"
        );
        assert!(
            error_message(&missing_tier)
                .contains("required target weak does not satisfy target requirements")
        );
        let conflicting_alias = format!(
            "{configured}\n[targets.alias]\nid = \"strong/model\"\nllm_client = \"responses\"\n"
        );
        assert!(error_message(&conflicting_alias).contains("disagree on labels"));
        Ok(())
    }

    #[tokio::test]
    async fn eligibility_matches_labels_not_set_names() -> RunnerResult<()> {
        let configured =
            external_eligibility_config().replace("private_eu =", "customer_records =");
        let runner = Runner::from_toml(&configured)?;
        let route = runner
            .route("switchyard/passthrough")
            .expect("eligibility route");
        assert_eq!(route.models().models_for(&Category::Any).len(), 2);
        let ordinary = runner.route("switchyard/random").expect("ordinary route");
        for labels in [&[][..], &["private"][..]] {
            let mut request = Request::default();
            crate::eligibility::require_target_labels(&mut request, labels)?;
            let error = ordinary
                .decide(request)
                .await
                .err()
                .expect("no eligibility");
            assert!(
                error
                    .to_string()
                    .contains("does not accept target requirements")
            );
        }
        let mut request = Request::default();
        crate::eligibility::require_target_labels(&mut request, ["eu", "private"])?;
        let outcome = route.decide(request).await?;
        assert_eq!(outcome.selected_model_id()?, "strong/model");
        let description = runner
            .describe_decision(&ModelId::from("switchyard/passthrough"), &outcome)
            .expect("eligible target metadata");
        assert_eq!(description.selected.target, "strong");
        assert_eq!(description.selected.base_url, "https://example.test/v1");
        let mut named = Request::default();
        crate::eligibility::require_target_labels(&mut named, ["customer_records"])?;
        let error = route
            .decide(named)
            .await
            .err()
            .expect("set name is not a label");
        assert!(error.to_string().contains("unknown label"));
        Ok(())
    }

    #[tokio::test]
    async fn eligibility_requires_signal_when_configured() -> RunnerResult<()> {
        let configured = external_eligibility_config().replace(
            "[routes.passthrough.eligibility]",
            "[routes.passthrough.eligibility]\nrequire_explicit_requirements = true",
        );
        let runner = Runner::from_toml(&configured)?;
        let route = runner.route("switchyard/passthrough").expect("route");
        let mut request = Request::default();
        let error = route
            .decide(request.clone())
            .await
            .err()
            .expect("missing signal");
        assert!(
            error
                .to_string()
                .contains("requires explicit external target requirements")
        );
        crate::eligibility::require_target_labels(&mut request, std::iter::empty::<&str>())?;
        assert_eq!(route.decide(request).await?.selected_model_ids.len(), 2);
        Ok(())
    }

    fn overridden_eligibility_config(algorithm: &str) -> String {
        let globals = VALID_CONFIG.split("[routes.noop]").next().expect("globals");
        let mut config = globals.to_string();
        for name in ["weak", "strong", "classifier"] {
            config.push_str(&format!(
                "\n[targets.approved_{name}]\nid = 'approved/{name}'\nllm_client = 'primary'\nlabels = ['private', 'eu']\n"
            ));
        }
        config.push_str(&format!(
            "\n[routes.test]\nid = 'route'\n{algorithm}\n[routes.test.eligibility]\naccepted_requirement_sets = {{ private_eu = ['private', 'eu'] }}\n[routes.test.eligibility.target_overrides.private_eu]\n"
        ));
        let spec: AlgorithmSpec = toml::from_str(algorithm).expect("test algorithm");
        for name in spec
            .callable_target_names()
            .into_iter()
            .collect::<BTreeSet<_>>()
        {
            config.push_str(&format!("{name} = 'approved_{name}'\n"));
        }
        config
    }

    // Fixed roles keep their meaning while each lane resolves its own models and metadata.
    #[tokio::test]
    async fn eligibility_overrides_preserve_router_roles() -> RunnerResult<()> {
        for (algorithm, initial) in [
            ("type = 'passthrough'\ntarget = 'weak'", Some("weak")),
            (
                "type = 'auto'\ncapable_target = 'strong'\nefficient_target = 'weak'",
                Some("weak"),
            ),
            (
                "type = 'plan_execute'\ncapable_target = 'strong'\nefficient_target = 'weak'",
                Some("strong"),
            ),
            (
                "type = 'stage_router'\ncapable_target = 'strong'\nefficient_target = 'weak'\npicker = 'efficient_first'\nconfidence_threshold = 1.0",
                Some("weak"),
            ),
            (
                "type = 'advisor'\nexecutor_target = 'weak'\nadvisor_target = 'classifier'",
                None,
            ),
            (
                "type = 'llm_classifier'\nclassifier_target = 'classifier'\nstrong_target = 'strong'\nweak_target = 'weak'\nbase_threshold = 0.5",
                None,
            ),
            (
                "type = 'llm_classifier'\nmode = 'escalation'\nclassifier_target = 'classifier'\nstrong_target = 'strong'\nweak_target = 'weak'\nescalation = { confirmations = 1 }",
                None,
            ),
            (
                "type = 'composite'\nclassifier = { target = 'classifier', base_threshold = 0.5, classify_trigger = 'user_turn' }\nstage = { capable_target = 'strong', efficient_target = 'weak', confidence_threshold = 1.0 }",
                None,
            ),
            (
                "type = 'vgr'\nlocal_target = 'weak'\ncloud_target = 'strong'",
                None,
            ),
            (
                "type = 'vgr'\nlocal_target = 'weak'\ncloud_target = 'strong'\njudge_target = 'classifier'\ncloud_judge_target = 'classifier'",
                None,
            ),
            (
                "type = 'passthrough'\ntarget = 'weak'\nsubagents = { type = 'passthrough', target = 'strong' }",
                Some("weak"),
            ),
        ] {
            let runner = Runner::from_toml(&overridden_eligibility_config(algorithm))?;
            let Some(initial) = initial else { continue };
            let route = runner.route("route").expect("route");
            let standard = route.decide(Request::default()).await?;
            assert_eq!(standard.selected_model_id()?, &format!("{initial}/model"));
            let mut request = Request::default();
            crate::require_target_labels(&mut request, ["private", "eu"])?;
            let outcome = route.decide(request).await?;
            assert_eq!(outcome.selected_model_id()?, &format!("approved/{initial}"));
            assert!(
                outcome
                    .selected_model_ids
                    .iter()
                    .all(|model| model.starts_with("approved/"))
            );
            let description = runner
                .describe_decision(&ModelId::from("route"), &outcome)
                .expect("mapped metadata");
            assert_eq!(description.selected.target, format!("approved_{initial}"));
        }
        Ok(())
    }

    // Invalid bindings fail at startup instead of being ignored or weakening requirements.
    #[test]
    fn eligibility_overrides_reject_invalid_bindings() {
        let config = overridden_eligibility_config(
            "type = 'auto'\ncapable_target = 'strong'\nefficient_target = 'weak'",
        );
        for (invalid, expected) in [
            (config.replace("target_overrides.private_eu", "target_overrides.typo"), "unknown requirement set"),
            (format!("{config}typo = 'approved_weak'\n"), "not used by the algorithm"),
            (config.replace("weak = 'approved_weak'", "weak = 'missing'"), "must exist and match the target kind"),
            (config.replace("weak = 'approved_weak'", "weak = 'weak'"), "does not satisfy target requirements"),
            (config.replace("accepted_requirement_sets = { private_eu = ['private', 'eu'] }", "accepted_requirement_sets = { private_eu = [] }"), "empty requirement set"),
            (config.replace("accepted_requirement_sets = { private_eu = ['private', 'eu'] }", "accepted_requirement_sets = { private_eu = ['private', 'eu'], duplicate = ['eu', 'private'] }"), "different target_overrides"),
            (config.replace("[llm_clients.primary]", "[llm_clients.primary]\nforward_auth = true"), "forward_auth replacements"),
            (config.replace("id = 'approved/weak'", "id = 'weak/model'"), "different clients across eligibility groups"),
        ] {
            let message = error_message(&invalid);
            assert!(message.contains(expected), "expected {expected:?}, got {message}");
        }
        let random = overridden_eligibility_config(
            "type = 'random'\ntargets = ['weak', 'strong']\nweights = [0, 1]",
        )
        .replace("strong = 'approved_strong'", "strong = 'approved_weak'");
        assert!(error_message(&random).contains("collapse distinct random targets"));
        let prompt = config.replace(
            "[targets.approved_weak]",
            "[targets.approved_weak]\nsystem_prompt = 'private prompt'",
        );
        assert!(Runner::from_toml(&prompt).is_ok());
    }

    #[test]
    fn eligibility_rejects_unknown_and_invalid_config() {
        let configured = external_eligibility_config();
        for (invalid, expected) in [
            (
                configured.replace("accepted_requirement_sets =", "requirement_sets ="),
                "unknown field `requirement_sets`",
            ),
            (
                configured.replace(
                    "accepted_requirement_sets = { private_eu = [\"private\", \"eu\"] }",
                    "",
                ),
                "missing field",
            ),
            (
                configured.replace("labels = [\"private\", \"eu\"]", "labels = [\" \"]"),
                "has invalid labels",
            ),
            (
                configured.replace("[targets.strong]", "[targets.strong]\nprivate = true"),
                "unknown field `private`",
            ),
            (
                decision_privacy_config().replace(
                    "[decision_targets.standard]",
                    "[decision_targets.standard]\nprivate = false",
                ),
                "unknown field `private`",
            ),
            (
                format!("{configured}\n[routes.passthrough.privacy]\nrequirement_scope = 'task'\n"),
                "unknown field `privacy`",
            ),
            (
                format!("{configured}\naccept_external_signal = true\n"),
                "unknown field `accept_external_signal`",
            ),
            (
                format!("{configured}\nrequire_external_signal = true\n"),
                "unknown field `require_external_signal`",
            ),
            (
                format!("{configured}\nrestriction_scope = 'task'\n"),
                "unknown field `restriction_scope`",
            ),
        ] {
            let message = error_message(&invalid);
            assert!(
                message.contains(expected),
                "expected {expected:?}, got {message}"
            );
        }
    }

    #[test]
    fn eligibility_rejects_unservable_requirement_sets_and_approval_aliases() {
        let configured = external_eligibility_config();
        for (invalid, expected) in [
            (
                configured.replace(
                    "accepted_requirement_sets = { private_eu = [\"private\", \"eu\"] }",
                    "accepted_requirement_sets = { missing = [\"missing\"] }",
                ),
                "eligibility requirement set",
            ),
            (
                format!(
                    "{configured}\n[routes.plan_execute.eligibility]\naccepted_requirement_sets = {{ private_eu = [\"private\", \"eu\"] }}\n"
                ),
                "required target weak does not satisfy target requirements",
            ),
            (
                format!(
                    "{configured}\n[targets.alias]\nid = \"strong/model\"\nllm_client = \"responses\"\nlabels = [\"private\"]\n"
                ),
                "disagree on labels",
            ),
            (
                format!(
                    "{}\n[decision_targets.alias]\nid = \"decision/standard\"\ndecision_client = \"privacy\"\nlabels = [\"private\", \"eu\"]\n",
                    decision_privacy_config()
                ),
                "disagrees on labels",
            ),
            (
                configured.replace(
                    "[llm_clients.responses]\nformat = \"openai_responses\"",
                    "[llm_clients.responses]\nformat = \"openai_responses\"\nforward_auth = true",
                ),
                "cannot use eligibility with forward_auth",
            ),
        ] {
            let message = error_message(&invalid);
            assert!(
                message.contains(expected),
                "expected {expected:?}, got {message}"
            );
        }
    }

    #[tokio::test]
    async fn task_retention_is_sticky_per_task() -> RunnerResult<()> {
        let runner = Runner::from_toml(&task_retained_privacy_config())?;
        let route = runner
            .route("switchyard/passthrough")
            .expect("privacy route should exist");
        let request = |task_id: Option<&str>, text: &str, restricted: bool| {
            let mut request = Request::default();
            request
                .llm_request
                .messages
                .push(Message::text(Role::User, text));
            request.metadata.get_or_insert_default().task_id = task_id.map(str::to_string);
            if restricted {
                mark_privacy_restricted(&mut request);
            }
            request
        };

        for (task_id, text, restricted, expected) in [
            (Some("task-a"), "ordinary turn", false, "standard"),
            (
                Some("task-a"),
                "Authorization: Bearer abcdefghijklmnop",
                true,
                "restricted",
            ),
            (Some("task-a"), "later clear turn", false, "restricted"),
            (Some("task-b"), "independent clear task", false, "standard"),
        ] {
            let outcome = route.decide(request(task_id, text, restricted)).await?;
            assert_eq!(
                outcome.selected_model_ids.len(),
                if expected == "restricted" { 1 } else { 2 },
                "{text}"
            );
            if expected == "restricted" {
                assert_eq!(outcome.selected_model_id()?, "strong/model");
            }
        }
        for task_id in [None, Some(" ")] {
            assert!(
                route
                    .decide(request(task_id, "invalid identity", false))
                    .await
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn duplicate_route_ids_are_rejected() {
        let config = format!(
            r#"{VALID_CONFIG}

[routes.duplicate]
id = "switchyard/passthrough"
type = "passthrough"
target = "strong"
"#
        );

        let error = error_message(&config);

        assert!(
            error.contains(
                "routes duplicate and passthrough both use id switchyard/passthrough; route ids must be unique"
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn plan_execute_applies_custom_tool_semantics() -> RunnerResult<()> {
        use switchyard_protocol::{ContentBlock, Message, Request, Role, ToolCall};

        let configured = format!(
            "{VALID_CONFIG}\n[routes.plan_execute.tool_semantics]\nmutate = [\"persist_source_file\"]"
        );
        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "write-1".to_string(),
                name: "mcp__files__Persist_Source_File".to_string(),
                arguments: json!({"path": "task.py", "content": "print('hello')"}),
            })],
        });

        for (config, expected) in [
            (VALID_CONFIG, "strong/model"),
            (configured.as_str(), "weak/model"),
        ] {
            let runner = Runner::from_toml(config)?;
            let route = runner.route("switchyard/plan-execute").unwrap();
            let outcome = route.decide(request.clone()).await?;
            assert_eq!(outcome.selected_model_ids[0], expected);
        }
        Ok(())
    }

    fn error_message(toml: &str) -> String {
        match runner_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    fn vgr_config() -> String {
        format!(
            r#"{VALID_CONFIG}

[routes.vgr]
id = "switchyard/vgr"
type = "vgr"
local_target = "weak"
cloud_target = "strong"
judge_target = "classifier"
mode = "active"
active_approval = "prospective-validation-and-canary-approved"
task_typing = false
"#
        )
    }

    #[test]
    fn vgr_route_builds_and_rejects_unsafe_configuration() -> RunnerResult<()> {
        let runner = runner_from_toml(&vgr_config())?;
        assert!(runner.route("switchyard/vgr").is_some());

        let compact = format!("{}\ncompact_handoff = true\n", vgr_config());
        assert!(error_message(&compact).contains("unknown field `compact_handoff`"));

        let identical_ids =
            vgr_config().replace("local_target = \"weak\"", "local_target = \"strong\"");
        assert!(error_message(&identical_ids).contains("distinct local and cloud model IDs"));

        let bad_approval =
            vgr_config().replace("prospective-validation-and-canary-approved", "approved");
        assert!(error_message(&bad_approval).contains("approval attestation"));

        let missing_target =
            vgr_config().replace("local_target = \"weak\"", "local_target = \"missing\"");
        assert!(error_message(&missing_target).contains("unknown target missing"));
        Ok(())
    }

    #[test]
    fn vgr_local_server_example_builds_and_rejects_unknown_discovery() -> RunnerResult<()> {
        let example = include_str!("../../../examples/vgr-local-servers.toml");
        let runner = runner_from_toml(example)?;
        assert!(runner.route("switchyard/vgr").is_some());
        let invalid = example.replace(
            "context_discovery = \"llama_cpp\"",
            "context_discovery = \"guess\"",
        );
        assert!(error_message(&invalid).contains("unknown variant `guess`"));
        Ok(())
    }

    fn with_subagent_llm_classifier(config: &str, route: &str, extra: &str) -> String {
        let mut configured = config.to_string();
        configured.push_str(&format!("\n[routes.{route}.subagents]\n"));
        configured.push_str(
            r#"type = "llm_classifier"
mode = "custom"
models = { judge = ["classifier"], capable = ["strong"], efficient = ["weak"], any = ["strong", "weak"] }
default_target = "efficient"
prompt = "Select a target for this delegated task."
response_schema = '{"type":"object","properties":{"target":{"type":"string","enum":["capable","efficient"]}},"required":["target"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/target" }
classify_trigger = "new_session""#,
        );
        configured.push_str(extra);
        configured
    }

    #[test]
    fn subagent_models_stay_separate_from_the_parent_tiers() -> RunnerResult<()> {
        // The sub-agent target is also the parent's capable tier. Merged into one group it
        // would be indistinguishable from that tier, and delegated work would follow the
        // parent's ordering instead of its own configured target.
        let runner = runner_from_toml(&with_subagent_passthrough(&stage_config(), "stage"))?;
        let models = runner
            .route("switchyard/stage")
            .expect("stage route should exist")
            .models();

        assert_eq!(
            models.subagent_models_for(&Category::Any),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("strong/model"), ModelId::from("weak/model")]
        );
        Ok(())
    }

    fn with_subagent_passthrough(config: &str, route: &str) -> String {
        format!("{config}\n[routes.{route}.subagents]\ntype = \"passthrough\"\ntarget = \"strong\"")
    }

    fn stage_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.stage_judge]
id = "stage-judge/model"
llm_client = "primary"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0
capable_hold_turns = 2

[routes.stage.tool_semantics]
observe = ["lookup_customer"]
mutate = ["send_payment"]
plan = ["create_workflow"]
new = ["send_message"]

[routes.stage.classifier]
target = "stage_judge"
base_threshold = 0.5
"#
        )
    }

    fn composite_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.tier_judge]
id = "tier-judge/model"
llm_client = "primary"

[routes.composed]
id = "switchyard/hier"
type = "composite"

[routes.composed.classifier]
target = "tier_judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.composed.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5
capable_hold_turns = 2

[routes.composed.stage.tool_semantics]
new = ["send_message"]
"#
        )
    }

    #[test]
    fn composite_decision_judge_reuses_capability_validation() {
        let source = composite_config().replace(
            "[routes.composed.classifier]\ntarget = \"tier_judge\"\nbase_threshold = 0.5",
            "[routes.composed.classifier]\ntarget = \"tier_judge\"",
        ) + r#"
[routes.composed.classifier.decision]
cutoff = 0.4
candidates = { a = "strong", b = "weak" }
evidence = { description = "candidate comparison" }
"#;
        for (source, valid) in [
            (source.clone(), true),
            (
                source.replace(
                    "target = \"tier_judge\"",
                    "target = \"tier_judge\"\nbase_threshold = 0.5",
                ),
                false,
            ),
            (source.replace("b = \"weak\"", "b = \"strong\""), false),
        ] {
            let deployment: DeploymentConfig = toml::from_str(&source).unwrap();
            let targets = deployment
                .targets
                .iter()
                .map(|(name, target)| (name.clone(), target.id.clone()))
                .collect();
            let spec = &deployment.routes["composed"].algorithm;
            assert_eq!(spec.decision_judge().unwrap().0, "tier_judge");
            assert_eq!(
                spec.build_with_runtime_models("composed", &targets).is_ok(),
                valid
            );
        }
    }

    #[test]
    fn composite_custom_judge_builds_and_rejects_capability_settings() -> RunnerResult<()> {
        let source = composite_config().replace(
            "base_threshold = 0.5\nclassify_trigger",
            r#"mode = "custom"
prompt = 'Select efficient or capable.'
response_schema = '{"type":"object","properties":{"decision":{"type":"object","properties":{"target":{"type":"string","enum":["efficient","capable"]}},"required":["target"],"additionalProperties":false}},"required":["decision"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/decision/target" }
classify_trigger"#,
        );
        runner_from_toml(&source)?;
        runner_from_toml(
            &source.replace("mode = \"custom\"", "mode = \"custom\"\nfail_open = true"),
        )?;
        for (setting, expected) in [
            (
                "base_threshold = 0.5",
                "composite custom classifier cannot use capability thresholds",
            ),
            (
                "fail_open = false",
                "composite custom classifiers always fail open",
            ),
        ] {
            let invalid = source.replace(
                "mode = \"custom\"",
                &format!("mode = \"custom\"\n{setting}"),
            );
            assert!(error_message(&invalid).contains(expected));
        }
        Ok(())
    }

    #[test]
    fn composite_route_builds_and_claims_both_tiers_and_its_judge() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_config())?;
        assert!(
            runner
                .models()
                .any(|model| model.id.as_str() == "switchyard/hier")
        );
        Ok(())
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> RunnerResult<()> {
        let state = runner_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            [
                "switchyard/classifier",
                "switchyard/noop",
                "switchyard/passthrough",
                "switchyard/plan-execute",
                "switchyard/random",
            ]
        );
        Ok(())
    }

    #[test]
    fn stage_rejects_ambiguous_or_non_additive_tool_semantics() {
        for (configured, expected) in [
            (
                stage_config().replace(
                    "mutate = [\"send_payment\"]",
                    "mutate = [\"LOOKUP_CUSTOMER\"]",
                ),
                "appears in both tool_semantics.observe and tool_semantics.mutate",
            ),
            (
                stage_config().replace("new = [\"send_message\"]", "new = [\"Read\"]"),
                "already has built-in semantics and cannot be reclassified",
            ),
            (
                stage_config().replace("observe = [\"lookup_customer\"]", "observe = [\" \"]"),
                "contains an empty tool name",
            ),
        ] {
            let message = error_message(&configured);
            assert!(
                message.contains(expected),
                "expected {expected:?} in error, got: {message}"
            );
        }
    }

    #[test]
    fn passthrough_and_stage_accept_subagent_routing() -> RunnerResult<()> {
        let stage = stage_config();
        let stage_with_classifier = with_subagent_llm_classifier(&stage, "stage", "");
        let parsed: DeploymentConfig = toml::from_str(&stage_with_classifier).map_err(|error| {
            RunnerError::configuration(format!("failed to parse stage config: {error}"))
        })?;
        let Some(stage_route) = parsed.routes.get("stage") else {
            return Err(RunnerError::configuration("stage route is missing"));
        };
        let callable_targets = stage_route.algorithm.callable_target_names();
        for expected in ["strong", "weak", "stage_judge", "classifier"] {
            assert!(callable_targets.contains(&expected));
        }

        for configured in [
            with_subagent_llm_classifier(VALID_CONFIG, "passthrough", ""),
            with_subagent_passthrough(VALID_CONFIG, "passthrough"),
            stage_with_classifier,
            with_subagent_passthrough(&stage, "stage"),
        ] {
            runner_from_toml(&configured)?;
        }
        Ok(())
    }

    #[test]
    fn aliased_completion_targets_reject_prompt_conflicts() {
        let configured = stage_config()
            .replace(
                "id = \"strong/model\"\nllm_client = \"responses\"",
                "id = \"strong/model\"\nllm_client = \"responses\"\nsystem_prompt = \"capable\"",
            )
            .replace(
                "[routes.stage]",
                "[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\"\n\n[routes.stage]",
            )
            .replace("efficient_target = \"weak\"", "efficient_target = \"strong_alias\"");
        let message = error_message(&configured);
        assert!(
            message.contains("completion target aliases to model strong/model with different system_prompt values"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn prompted_routing_response_cannot_share_a_model_with_a_dependency() {
        let configured = VALID_CONFIG
            .replace(
                "id = \"classifier/model\"\nllm_client = \"primary\"",
                "id = \"weak/model\"\nllm_client = \"primary\"",
            )
            .replace(
                "id = \"weak/model\"\nllm_client = \"anthropic\"",
                "id = \"weak/model\"\nllm_client = \"anthropic\"\nsystem_prompt = \"answer prompt\"",
            )
            .replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nescalation = { confirmations = 1 }",
            );

        let message = error_message(&configured);

        assert!(
            message.contains("cannot apply system_prompt to target weak: model weak/model is also used by routing-only target classifier"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn rejects_invalid_unreferenced_llm_client() {
        for (settings, expected) in [
            (
                "base_url = \"not a url\"",
                "base_url must be an absolute HTTP(S) URL",
            ),
            (
                "base_url = \"https://example.test\"\ntimeout_ms = 0",
                "timeout_ms must be at least 1",
            ),
        ] {
            let invalid = format!(
                "{VALID_CONFIG}\n[llm_clients.unused]\nformat = \"openai_chat\"\n{settings}"
            );
            let message = error_message(&invalid);
            assert!(message.contains(expected), "unexpected error: {message}");
        }
    }

    #[test]
    fn an_escalation_table_switches_the_classifier_route_to_escalation() -> RunnerResult<()> {
        // Present: the classifier target judges the weak tier's reply each turn instead of
        // picking a tier ahead of it. The route builds either way, so the assertion is that
        // the knob parses and its settings reach the algorithm's validation.
        let escalating = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalating)?;

        let reversible = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2, deescalation = { strong_min_calls = 3, confirmations = 2, strong_max_calls = 6, weak_cooldown_calls = 8 } }",
        );
        runner_from_toml(&reversible)?;

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
    fn a_target_reasoning_effort_parses_and_is_rejected_where_unsupported() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let weak = "[targets.weak]\nid = \"weak/model\"\nllm_client = \"anthropic\"";
        assert!(VALID_CONFIG.contains(strong) && VALID_CONFIG.contains(weak));

        let forced = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \"max\""));
        runner_from_toml(&forced)?;

        let blank = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \" \""));
        assert!(error_message(&blank).contains("reasoning_effort must not be empty"));

        let anthropic = VALID_CONFIG.replace(weak, &format!("{weak}\nreasoning_effort = \"high\""));
        assert!(
            error_message(&anthropic)
                .contains("only supported on openai_chat and openai_responses")
        );
        Ok(())
    }

    #[test]
    fn duplicate_targets_with_conflicting_settings_are_rejected() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(VALID_CONFIG.contains(strong));
        // Same model, same client, different effort: the second target could never take effect.
        let conflicting = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\n\n[targets.strong_max]\nid = \"strong/model\"\nllm_client = \"responses\"\nreasoning_effort = \"max\""
            ),
        );
        assert!(
            error_message(&conflicting)
                .contains("different reasoning_effort, extra_body, or omit_body_fields"),
            "{}",
            error_message(&conflicting)
        );
        // An alias with identical settings is still allowed (it only warns).
        let alias = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\n\n[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\""),
        );
        runner_from_toml(&alias)?;
        Ok(())
    }

    #[test]
    fn classifier_judge_completion_caps_are_configurable() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 512",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 256\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;
        Ok(())
    }

    #[test]
    fn classifier_prompts_are_configurable_in_both_modes() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom capability rubric\"",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom trajectory rubric\"\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;

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
    fn stage_router_rejects_an_unknown_field() {
        let config = stage_config().replace(
            "picker = \"efficient_first\"",
            "picker = \"efficient_first\"\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn auto_route_builds_a_stage_router_with_no_extra_fields() -> RunnerResult<()> {
        let config = format!(
            r#"{VALID_CONFIG}
[routes.auto]
id = "switchyard/auto"
type = "auto"
capable_target = "strong"
efficient_target = "weak"
"#
        );
        let runner = runner_from_toml(&config)?;
        assert!(runner.route("switchyard/auto").is_some());
        Ok(())
    }

    #[test]
    fn composite_stage_block_rejects_an_unknown_field() {
        let config = composite_config().replace(
            "confidence_threshold = 0.5",
            "confidence_threshold = 0.5\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
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
                    "targets = [\"strong\", \"weak\"]\nweights = [0, 0]",
                ),
                "at least one weight must be positive",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"strong\"]",
                ),
                "targets must be unique, strong is repeated",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [1]",
                ),
                "expected 2 weights, got 1",
            ),
            (
                VALID_CONFIG.replace("base_threshold = 0.5", "base_threshold = 1.5"),
                "base_threshold must be between 0 and 1",
            ),
            (
                VALID_CONFIG.replace("classifier_target = \"classifier\"\n", ""),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"\"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"   \"",
                ),
                "route references unknown target",
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
                "message_hash_fallback requires classify_trigger = new_session",
            ),
            (
                with_subagent_llm_classifier(
                    VALID_CONFIG,
                    "passthrough",
                    "\nmessage_hash_fallback = true",
                ),
                "cannot use message_hash_fallback",
            ),
            (
                with_subagent_llm_classifier(VALID_CONFIG, "passthrough", "")
                    .replace("mode = \"custom\"", "mode = \"capability\""),
                "mode capability cannot use custom classifier fields",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "escalation = { confirmations = 2 }\nclassify_trigger = \"user_turn\"",
                ),
                "mode escalation cannot use classify_trigger",
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
            let error = error_message(&toml);
            assert!(
                error.contains(expected),
                "expected error containing {expected}, got {error}"
            );
        }
    }

    #[test]
    fn accepts_duplicate_target_model_ids_on_one_client() -> RunnerResult<()> {
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
        let state = runner_from_toml(SAME_MODEL_TWO_ROUTES)?;
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/fast", "switchyard/smart"]
        );
        Ok(())
    }

    #[test]
    fn accepts_same_model_id_on_different_llm_clients() -> RunnerResult<()> {
        // Separate routes may serve the same model through different clients.
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
        runner_from_toml(CROSS_PROVIDER)?;
        let shared_route = format!(
            r#"{CROSS_PROVIDER}

[routes.shared]
id = "switchyard/shared"
type = "stage_router"
capable_target = "openai"
efficient_target = "azure"
picker = "efficient_first"
confidence_threshold = 0.5
"#
        );
        let message = error_message(&shared_route);
        assert!(
            message.contains("route shared")
                && message.contains("openai")
                && message.contains("azure")
                && message.contains("gpt-4o")
                && message.contains("different llm clients"),
            "{message}"
        );
        Ok(())
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> RunnerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        runner_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_new_session_trigger_with_message_hash_fallback() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nthreshold_step = 0.1\nclassify_trigger = \"new_session\"\nmessage_hash_fallback = true",
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn target_extra_body_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             extra_body = { service_tier = \"priority\", \
             chat_template_kwargs = { enable_thinking = false } }",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend(
            "primary",
            client,
            &target.extra_body,
            &target.omit_body_fields,
            None,
        )?;

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
    fn target_omit_body_fields_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             omit_body_fields = [\"max_output_tokens\"]",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend(
            "primary",
            client,
            &target.extra_body,
            &target.omit_body_fields,
            None,
        )?;

        assert!(backend.omit_body_fields().contains("max_output_tokens"));
        Ok(())
    }

    #[test]
    fn client_deadline_defaults_and_rejects_zero() -> RunnerResult<()> {
        for (setting, expected) in [
            ("", None),
            ("timeout_ms = 1500", Some(1500)),
            ("timeout_ms = 0", Some(0)),
        ] {
            let source = format!(
                "format = \"openai_chat\"\nbase_url = \"https://example.test/v1\"\n{setting}"
            );
            let config: LlmClientConfig = toml::from_str(&source).expect("valid deadline config");
            let backend = build_backend("test", &config, &BTreeMap::new(), &BTreeSet::new(), None);
            if expected == Some(0) {
                assert!(backend.is_err());
            } else {
                assert_eq!(backend?.timeout(), expected.map(Duration::from_millis));
            }
        }
        Ok(())
    }

    #[test]
    fn retry_budget_defaults_and_accepts_an_override() -> RunnerResult<()> {
        let default: DeploymentConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            RunnerError::configuration(format!("failed to parse default config: {error}"))
        })?;
        let Some(primary) = default.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, DEFAULT_MAX_RETRIES);

        let explicit = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 0",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&explicit).map_err(|error| {
            RunnerError::configuration(format!("failed to parse explicit retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, 0);

        let maximum = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!(
                "base_url = \"https://example.test/v1\"\nmax_retries = {MAX_CONFIGURED_RETRIES}"
            ),
            1,
        );
        let config: DeploymentConfig = toml::from_str(&maximum).map_err(|error| {
            RunnerError::configuration(format!("failed to parse maximum retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, MAX_CONFIGURED_RETRIES);
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
    fn accepts_additional_headers() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { X-Inference-Priority = \"batch\" }",
            1,
        );

        runner_from_toml(&configured)?;
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

        // macOS reads `FOO` for the name `FOO=x`, so the config would use another variable. A
        // name with `=` is often a pasted `NAME=key` line, so the error leaves it out.
        let with_equals = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\napi_key_env = \"NVIDIA_API_KEY=DUMMY_SECRET_DO_NOT_LOG\"",
            1,
        );
        let message = error_message(&with_equals);
        assert!(message.contains("llm client primary api_key_env must not contain '=' or NUL"));
        assert!(!message.contains("DUMMY_SECRET_DO_NOT_LOG"), "{message}");
    }

    // `VarError::NotUnicode` holds the variable's value, so printing that error prints the key.
    #[cfg(any(unix, windows))]
    #[test]
    fn invalid_unicode_api_key_is_not_printed() {
        const RAW_KEY_ENV: &str = "SWITCHYARD_CONFIG_TEST_RAW_KEY";
        #[cfg(unix)]
        let raw = {
            use std::os::unix::ffi::OsStringExt;
            let mut bytes = b"DUMMY_SECRET_DO_NOT_LOG".to_vec();
            bytes.push(0xff);
            std::ffi::OsString::from_vec(bytes)
        };
        #[cfg(windows)]
        let raw = {
            use std::os::windows::ffi::OsStringExt;
            let mut words: Vec<_> = "DUMMY_SECRET_DO_NOT_LOG".encode_utf16().collect();
            words.push(0xd800);
            std::ffi::OsString::from_wide(&words)
        };
        unsafe {
            // "unsafe" is for concurrent reads and writes, very rare
            std::env::set_var(RAW_KEY_ENV, raw);
        }
        let result = read_api_key("llm client fixture", RAW_KEY_ENV);
        unsafe {
            std::env::remove_var(RAW_KEY_ENV);
        }
        let error = result.expect_err("invalid Unicode credential").to_string();
        assert!(error.contains("not valid Unicode"), "{error}");
        assert!(!error.contains("DUMMY_SECRET_DO_NOT_LOG"), "{error}");
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

    /// Returns a config whose `mixed` route has a GPT target on a Responses client and a Claude
    /// target on a Messages client at `messages_url`. Its `claude` route uses Claude only.
    fn mixed_forwarding_config(messages_url: &str) -> String {
        format!(
            r#"
schema_version = 1

[llm_clients.responses]
format = "openai_responses"
base_url = "https://gateway.example.test/v1"
forward_auth = true

[llm_clients.messages]
format = "anthropic_messages"
base_url = "{messages_url}"
forward_auth = true

[targets]
gpt = {{ id = "gpt/model", llm_client = "responses" }}
claude = {{ id = "claude/model", llm_client = "messages" }}

[routes.mixed]
id = "switchyard/mixed"
type = "random"
targets = ["gpt", "claude"]

[routes.claude]
id = "switchyard/claude"
type = "passthrough"
target = "claude"
"#
        )
    }

    /// A forwarding route can mix OpenAI and Anthropic clients only when they all use the same
    /// scheme, host, and port.
    #[test]
    fn forwarding_route_mixes_formats_only_on_one_host() -> RunnerResult<()> {
        // Same host, different paths: the mixed route serves OpenAI callers, and the route that
        // uses only the Messages client keeps serving Messages callers.
        let runner = runner_from_toml(&mixed_forwarding_config("https://gateway.example.test"))?;
        let caller_auth = |id| runner.route(id).and_then(Route::caller_auth);
        assert_eq!(
            caller_auth("switchyard/mixed"),
            Some(CallerAuthKind::OpenAi)
        );
        assert_eq!(
            caller_auth("switchyard/claude"),
            Some(CallerAuthKind::Anthropic)
        );

        // A different host, port, or scheme fails, and the error names it.
        for other in [
            "https://api.anthropic.test",
            "https://gateway.example.test:8443",
            "http://gateway.example.test",
        ] {
            let error = error_message(&mixed_forwarding_config(other));
            assert!(
                error.contains("different origins") && error.contains(other),
                "{error}"
            );
        }
        Ok(())
    }

    const ADVISOR_CONFIG: &str = r#"
schema_version = 1

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.executor]
id = "executor/model"
llm_client = "anthropic"

[targets.advisor]
id = "advisor/model"
llm_client = "anthropic"

[routes.gated]
id = "switchyard/advisor"
type = "advisor"
executor_target = "executor"
advisor_target = "advisor"
"#;

    #[test]
    fn advisor_route_parses_with_defaults_and_builds() -> RunnerResult<()> {
        let state = runner_from_toml(ADVISOR_CONFIG)?;
        let route = state
            .route("switchyard/advisor")
            .expect("advisor route should exist");
        let models = route.models();
        // The gate calls the executor through `efficient`; `any` keeps it in the
        // route's last-resort pool.
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("advisor/model")]
        );
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/advisor"]
        );
        Ok(())
    }

    #[test]
    fn advisor_route_accepts_every_gate_knob() -> RunnerResult<()> {
        let tuned = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            concat!(
                "advisor_target = \"advisor\"\n",
                "reviewer_system_prompt = \"review it\"\n",
                "redo_feedback_prefix = \"REVIEWER SAYS: \"\n",
                "gate_trigger = \"pattern\"\n",
                "gate_trigger_pattern = 'task_complete[\"\\s>:]*true'\n",
                "max_reviews = 2\n",
                "gate_stall_turns = 40\n",
                "gate_min_tool_results = 1\n",
                "advisor_max_tokens = 1024\n",
                "advisor_temperature = 0.0\n",
                "transcript_max_chars = 100000\n",
                "fail_open = false\n",
                "context_window = 200000\n",
                "tool_calling = true\n",
                "reasoning = true",
            ),
        );
        runner_from_toml(&tuned)?;
        Ok(())
    }

    #[test]
    fn advisor_route_rejects_unknown_keys() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nbogus_field = 1",
        );
        assert!(error_message(&invalid).contains("bogus_field"));
    }

    #[test]
    fn advisor_route_requires_both_targets() {
        let missing = ADVISOR_CONFIG.replace("advisor_target = \"advisor\"\n", "");
        assert!(error_message(&missing).contains("advisor_target"));
    }

    #[test]
    fn advisor_route_rejects_unknown_target() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"missing\"",
        );
        assert!(error_message(&invalid).contains("missing"));
    }

    #[test]
    fn advisor_route_rejects_invalid_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"\ngate_trigger_pattern = \"(unclosed\"",
        );
        assert!(error_message(&invalid).contains("not a valid regex"));
    }

    #[test]
    fn advisor_route_rejects_pattern_without_pattern_trigger() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger_pattern = \"done\"",
        );
        assert!(
            error_message(&invalid)
                .contains("gate_trigger_pattern requires gate_trigger = \"pattern\"")
        );
    }

    #[test]
    fn advisor_route_pattern_trigger_requires_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"",
        );
        assert!(error_message(&invalid).contains("non-empty gate_trigger_pattern"));
    }

    #[test]
    fn advisor_route_rejects_zero_max_reviews() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nmax_reviews = 0",
        );
        assert!(error_message(&invalid).contains("max_reviews must be at least 1"));
    }
}
