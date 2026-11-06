// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-1 TOML deployment loading for the shared runner.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_llm_client::{
    AuxiliaryOperation, Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    OpenAiDecisionClient, ResponsesReasoningPolicy, SystemOneClient, TranslatingLlmClient,
};
use switchyard_protocol::{
    LlmClientError, ModelId, RoutedDecisionClient, RoutedLlmClient, WireFormat,
};

use crate::algorithm_factory::AlgorithmAndModels;
use crate::eligibility::{EligibilityPolicy, validate_labels};
use crate::route::ExecutionLane;
use crate::runner::ModelPropertiesProbe;
use crate::{
    AlgorithmFactory, AlgorithmSpec, AuxiliaryTarget, CallerAuthKind, DecisionTarget,
    ModelCapabilities, Route, Runner, RunnerError,
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
        variables.insert(client.api_key_env.clone());
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

/// RouteTargets lists the targets one route calls, as the runner reads them from the server config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteTargets {
    /// This field names the route's table under `[routes]`.
    pub name: String,
    /// This field holds the public model ID that callers send.
    pub id: ModelId,
    /// This field holds the route's `type`, such as `stage_router`.
    pub kind: &'static str,
    /// This field lists the targets that answer requests, in algorithm order. It includes
    /// targets that only delegated sub-agent work uses.
    pub answer_targets: Vec<ConfiguredTarget>,
    /// This field lists the targets the route calls only to decide, such as a judge or an advisor.
    pub judge_targets: Vec<ConfiguredTarget>,
}

/// ConfiguredTarget names one target that a route uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfiguredTarget {
    pub name: String,
    /// This field holds the client and model from the `[targets]` or `[decision_targets]`
    /// entry with this name. It is `None` when no entry has the name; the server refuses to
    /// start with such a config.
    pub model: Option<TargetModel>,
}

/// TargetModel identifies the model a target sends requests to and the client that serves it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetModel {
    pub client: String,
    pub id: ModelId,
}

/// This function lists each route's targets without building clients or reading credentials.
/// It returns the same parse error the server reports for a config that does not match the schema.
pub fn route_targets(source: &str) -> Result<Vec<RouteTargets>, RunnerError> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|error| {
        RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
    })?;
    config.check_schema_version()?;
    Ok(config
        .routes
        .iter()
        .map(|(name, route)| RouteTargets {
            name: name.clone(),
            id: route.id.clone(),
            kind: (&route.algorithm).into(),
            answer_targets: config.configured_targets(route.routing_target_names()),
            judge_targets: config.configured_targets(route.algorithm.judge_target_names()),
        })
        .collect())
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
    Arc::new(AlgorithmFactory::from_toml(source)?.config).build()
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
pub(crate) struct RouteConfig {
    pub(crate) id: ModelId,
    context_window: Option<u32>,
    tool_calling: Option<bool>,
    reasoning: Option<bool>,
    vision: Option<bool>,
    algorithm: AlgorithmSpec,
    infra: Option<libsy::InfraPolicy>,
    eligibility: Option<TargetEligibilityConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetEligibilityConfig {
    #[serde(default)]
    require_explicit_requirements: bool,
    #[serde(default)]
    target_alternatives: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    scope: RequirementScope,
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

// Algorithms keep their role names; calls use the selected target's name.
struct ResolvedLlmTarget<'a> {
    name: &'a str,
    config: &'a TargetConfig,
}

type LaneTargets<'a> = BTreeMap<&'a str, ResolvedLlmTarget<'a>>;
type TargetOverrides<'a> = BTreeMap<&'a str, &'a str>;
type LaneDecisions<'a> = BTreeMap<&'a str, ResolvedDecisionTarget<'a>>;

// Replacements bind logical names once; mappings never chain.
fn bound_target<'a>(name: &'a str, overrides: &TargetOverrides<'a>) -> &'a str {
    overrides.get(name).copied().unwrap_or(name)
}

fn callable_targets(algorithm: &AlgorithmSpec) -> BTreeSet<&str> {
    algorithm.callable_target_names().into_iter().collect()
}

type TargetBindings = Vec<(String, String)>;

// Cache by resolved targets, so equivalent requirements reuse algorithm-owned state.
pub(crate) struct EligibleLanes {
    deployment: Arc<DeploymentConfig>,
    route_name: String,
    clients: Arc<BTreeMap<String, Arc<TranslatingLlmClient>>>,
    decision_clients: Arc<BTreeMap<String, Arc<dyn RoutedDecisionClient>>>,
    lanes: Mutex<BTreeMap<TargetBindings, Arc<ExecutionLane>>>,
}

impl EligibleLanes {
    const MAX_LANES: usize = 128;

    pub(crate) fn select(
        &self,
        requirements: &BTreeSet<String>,
        standard: &Arc<ExecutionLane>,
    ) -> RunnerResult<Arc<ExecutionLane>> {
        let route = &self.deployment.routes[&self.route_name];
        let config = route
            .eligibility
            .as_ref()
            .ok_or_else(|| RunnerError::configuration("route has no eligibility configuration"))?;
        let callable = callable_targets(&route.algorithm);
        let mut allowed = BTreeSet::new();
        let mut overrides = TargetOverrides::new();
        for name in &callable {
            let candidates = std::iter::once(*name).chain(
                config
                    .target_alternatives
                    .get(*name)
                    .into_iter()
                    .flatten()
                    .map(String::as_str),
            );
            if let Some(target) = candidates.into_iter().find(|name| {
                self.deployment
                    .target_labels(name)
                    .is_some_and(|labels| requirements.is_subset(labels))
            }) {
                allowed.insert(*name);
                if target != *name {
                    overrides.insert(*name, target);
                }
            }
        }
        if overrides.is_empty() && allowed.len() == callable.len() {
            return Ok(Arc::clone(standard));
        }
        let key = allowed
            .iter()
            .map(|name| {
                (
                    (*name).to_owned(),
                    bound_target(name, &overrides).to_owned(),
                )
            })
            .collect();
        let mut lanes = self.lanes.lock();
        if let Some(lane) = lanes.get(&key) {
            return Ok(Arc::clone(lane));
        }
        if lanes.len() >= Self::MAX_LANES {
            return Err(LlmClientError::TemporarilyUnavailable.into());
        }
        let algorithm = route.algorithm.restricted_to(&allowed).map_err(|error| {
            LlmClientError::InvalidRequest {
                message: error.to_string(),
            }
        })?;
        let targets = self
            .deployment
            .resolve_lane_targets(&algorithm, &overrides)?;
        if matches!(algorithm, AlgorithmSpec::Random { .. }) && !overrides.is_empty() {
            let mut models = BTreeSet::new();
            for name in algorithm.routing_target_names() {
                let target = targets[name].name;
                if !models.insert(target) {
                    return Err(LlmClientError::InvalidRequest {
                        message: format!(
                            "random route alternatives map multiple pool entries to target {target}; use distinct targets"
                        ),
                    }
                    .into());
                }
            }
        }
        let decisions = self.deployment.resolve_decision_targets(
            &self.route_name,
            &algorithm,
            &overrides,
            &self.decision_clients,
        )?;
        // Construction is synchronous; no model I/O or await occurs under this lock.
        let (lane, _) = self.deployment.build_lane(
            &self.route_name,
            &algorithm,
            &targets,
            &decisions,
            &self.clients,
            None,
        )?;
        let lane = Arc::new(lane);
        lanes.insert(key, Arc::clone(&lane));
        Ok(lane)
    }
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
        let infra = take_optional(&mut table, "infra")?;
        let eligibility = take_optional(&mut table, "eligibility")?;
        let algorithm = AlgorithmSpec::deserialize(toml::Value::Table(table))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id,
            context_window,
            tool_calling,
            reasoning,
            vision,
            algorithm,
            infra,
            eligibility,
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

    fn routing_target_names(&self) -> Vec<&str> {
        self.algorithm.routing_target_names()
    }

    fn callable_target_names(&self) -> Vec<&str> {
        self.algorithm.callable_target_names()
    }
}

impl DeploymentConfig {
    /// Judges called through a decision target use the default decision settings
    /// when the route omits a `decision` table.
    pub(crate) fn infer_decision_judges(&mut self) {
        for route in self.routes.values_mut() {
            route
                .algorithm
                .infer_decision_judges(|name| self.decision_targets.contains_key(name));
        }
    }

    pub(crate) fn requires_target_admission(&self) -> bool {
        self.routes
            .values()
            .any(|route| route.eligibility.is_some())
    }

    fn target_labels(&self, name: &str) -> Option<&BTreeSet<String>> {
        self.targets
            .get(name)
            .map(|target| &target.labels)
            .or_else(|| self.decision_targets.get(name).map(|target| &target.labels))
    }

    fn check_schema_version(&self) -> RunnerResult<()> {
        if self.schema_version == SUPPORTED_SCHEMA_VERSION {
            return Ok(());
        }
        Err(RunnerError::configuration(format!(
            "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
            self.schema_version
        )))
    }

    /// This function looks up each name in `[targets]`, then in `[decision_targets]`. The server
    /// rejects a name defined in both, so the lookup order does not change the result.
    fn configured_targets(&self, names: Vec<&str>) -> Vec<ConfiguredTarget> {
        names
            .into_iter()
            .map(|name| ConfiguredTarget {
                name: name.to_string(),
                model: self
                    .targets
                    .get(name)
                    .map(|target| (&target.llm_client, &target.id))
                    .or_else(|| {
                        self.decision_targets
                            .get(name)
                            .map(|target| (&target.decision_client, &target.id))
                    })
                    .map(|(client, id)| TargetModel {
                        client: client.clone(),
                        id: id.clone(),
                    }),
            })
            .collect()
    }

    fn decision_target(&self, name: &str) -> Option<DecisionTarget> {
        let target = self.targets.get(name)?;
        let client = self.llm_clients.get(&target.llm_client)?;
        Some(DecisionTarget {
            target: name.to_string(),
            model: target.id.clone(),
            format: client.format.wire_format(),
            base_url: client.base_url.as_str().to_string(),
            extra_body: target.extra_body.clone(),
            reasoning_effort: target.reasoning_effort.clone(),
        })
    }

    /// Checks route identity and target references without constructing clients.
    pub(crate) fn validate(&self) -> RunnerResult<()> {
        self.check_schema_version()?;

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

        for (name, target) in &self.targets {
            validate_labels(&target.labels)
                .map_err(|error| RunnerError::configuration(error.to_string()))?;
            validate_value("target name", name)?;
            validate_value(&format!("target {name} id"), &target.id)?;
            if !self.llm_clients.contains_key(&target.llm_client) {
                return Err(RunnerError::configuration(format!(
                    "target {name} references unknown llm client {}",
                    target.llm_client
                )));
            }
        }
        let mut decision_approvals = HashMap::new();
        for (name, target) in &self.decision_targets {
            validate_labels(&target.labels)
                .map_err(|error| RunnerError::configuration(error.to_string()))?;
            if decision_approvals
                .insert((&target.decision_client, &target.id), &target.labels)
                .is_some_and(|approved| approved != &target.labels)
            {
                return Err(RunnerError::configuration(format!(
                    "decision target {name} shares a model and client with an alias but disagrees on approval labels"
                )));
            }
            validate_value("decision target name", name)?;
            validate_value(&format!("decision target {name} id"), &target.id)?;
            if self.targets.contains_key(name) {
                return Err(RunnerError::configuration(format!(
                    "target {name} is defined as both an LLM and decision target"
                )));
            }
            // Algorithms route LLM targets by name and decision targets by id.
            if self.targets.contains_key(target.id.as_str()) {
                return Err(RunnerError::configuration(format!(
                    "decision target {name} id {} is also an LLM target name; rename one of them",
                    target.id
                )));
            }
            if !self.decision_clients.contains_key(&target.decision_client) {
                return Err(RunnerError::configuration(format!(
                    "decision target {name} references unknown decision client {}",
                    target.decision_client
                )));
            }
        }
        for (route_name, config) in &self.routes {
            if let Some(eligibility) = &config.eligibility {
                self.validate_eligibility(route_name, &config.algorithm, eligibility)?;
            }
        }
        Ok(())
    }

    /// Resolves each route once; native and host-owned execution share this path.
    pub(crate) fn build_algorithms(
        &self,
    ) -> impl Iterator<Item = RunnerResult<(&str, &RouteConfig, AlgorithmAndModels)>> {
        let targets = self.build_targets();
        self.routes.iter().map(move |(route_name, config)| {
            let decision_targets = config.algorithm.decision_target_names();
            for name in config.callable_target_names() {
                let (exists, kind) = if decision_targets.contains(&name) {
                    (self.decision_targets.contains_key(name), "decision")
                } else {
                    (self.targets.contains_key(name), "LLM")
                };
                if !exists {
                    return Err(RunnerError::configuration(format!(
                        "route references unknown target {name}; route {route_name} requires target kind {kind}"
                    )));
                }
            }
            for name in config.routing_target_names() {
                if !self.targets.contains_key(name) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} completion target {name} must be an LLM target"
                    )));
                }
            }
            if config.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            let (algorithm, models) = config
                .algorithm
                .build_with_runtime_models(route_name, &targets)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let models = models.with_provider_models(
                config
                    .callable_target_names()
                    .into_iter()
                    .filter_map(|name| {
                        self.targets
                            .get(name)
                            .map(|target| (ModelId::from(name), target.id.clone()))
                    })
                    .collect(),
            );
            if let Some(infra) = &config.infra {
                if !matches!(&config.algorithm, AlgorithmSpec::LlmClassifier { config }
                    if matches!(config.mode, None | Some(crate::ClassifierMode::Capability))
                        && config.escalation.is_none())
                {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name}: infra requires capability llm_classifier without escalation"
                    )));
                }
                infra.validate().map_err(|error| {
                    RunnerError::configuration_source(format!("route {route_name}: {error}"), error)
                })?;
            }
            Ok((route_name.as_str(), config, (algorithm, models)))
        })
    }

    fn build(self: Arc<Self>) -> RunnerResult<Runner> {
        self.validate()?;

        let mut approvals = HashMap::new();
        for (target_name, target) in &self.targets {
            if let Some((first_name, labels)) = approvals.insert(
                (&target.llm_client, &target.id),
                (target_name, &target.labels),
            ) && labels != &target.labels
            {
                return Err(RunnerError::configuration(format!(
                    "targets {first_name} and {target_name} share a model and llm client but disagree on approval labels"
                )));
            }
        }

        let mut provider_api_keys = Vec::new();
        let clients = Arc::new(self.build_clients(&mut provider_api_keys)?);
        let decision_clients = Arc::new(self.build_decision_clients(&mut provider_api_keys)?);
        let fallback_base_url = self.fallback_base_url()?;
        let mut routes = Vec::with_capacity(self.routes.len());
        let mut model_properties_probes = BTreeMap::new();
        for entry in self.build_algorithms() {
            let (route_name, config, (algorithm, models)) = entry?;
            let capabilities = config.capabilities();
            if let AlgorithmSpec::Vgr { config: vgr } = &config.algorithm
                && let Some(probe) =
                    self.build_model_properties_probe(&vgr.efficient_target, &clients)
            {
                model_properties_probes.insert(config.id.clone(), probe);
            }
            let no_overrides = TargetOverrides::new();
            let standard_targets = self.resolve_lane_targets(&config.algorithm, &no_overrides)?;
            let decisions = self.resolve_decision_targets(
                route_name,
                &config.algorithm,
                &no_overrides,
                &decision_clients,
            )?;
            let (standard, caller_auth) = self.build_lane(
                route_name,
                &config.algorithm,
                &standard_targets,
                &decisions,
                &clients,
                Some((algorithm, models)),
            )?;
            // Judges are included so a judge answer maps to its provider model too.
            let mut target_names = callable_targets(&config.algorithm);
            if let Some(eligibility) = &config.eligibility {
                target_names.extend(
                    eligibility
                        .target_alternatives
                        .values()
                        .flatten()
                        .map(String::as_str),
                );
            }
            let route_targets = target_names
                .into_iter()
                .filter_map(|name| self.decision_target(name))
                .collect();
            let mut route = Route::from_lane(standard, caller_auth, capabilities, route_targets)
                .with_infra(config.infra.clone());
            if let Some(eligibility) = &config.eligibility {
                let policy = EligibilityPolicy::new(
                    eligibility.require_explicit_requirements,
                    matches!(eligibility.scope, RequirementScope::Task),
                );
                route = route.with_eligibility(
                    policy,
                    EligibleLanes {
                        deployment: Arc::clone(&self),
                        route_name: route_name.to_owned(),
                        clients: Arc::clone(&clients),
                        decision_clients: Arc::clone(&decision_clients),
                        lanes: Mutex::new(BTreeMap::new()),
                    },
                );
            }
            routes.push((config.id.clone(), route));
        }
        let runner = Runner::new(routes)
            .with_fallback_url(fallback_base_url)
            .with_provider_api_keys(provider_api_keys)
            .with_model_properties_probes(model_properties_probes);
        Ok(runner)
    }

    // Alternatives preserve the router's existing roles and never chain.
    fn validate_eligibility(
        &self,
        route_name: &str,
        algorithm: &AlgorithmSpec,
        config: &TargetEligibilityConfig,
    ) -> RunnerResult<()> {
        if !algorithm.supports_target_restrictions() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot use eligibility with prefill_router"
            )));
        }
        let callable = callable_targets(algorithm);
        for (source, alternatives) in &config.target_alternatives {
            if !callable.contains(source.as_str()) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} target_alternatives source {source} is not used by the algorithm"
                )));
            }
            if alternatives.is_empty() {
                return Err(RunnerError::configuration(
                    "target alternatives must not be empty",
                ));
            }
            for target in alternatives {
                let same_kind = (self.targets.contains_key(source)
                    && self.targets.contains_key(target))
                    || (self.decision_targets.contains_key(source)
                        && self.decision_targets.contains_key(target));
                if !same_kind {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} alternative {target} must exist and match the target kind of {source}"
                    )));
                }
            }
        }
        for name in callable.into_iter().chain(
            config
                .target_alternatives
                .values()
                .flatten()
                .map(String::as_str),
        ) {
            if let Some(target) = self.targets.get(name)
                && self
                    .llm_clients
                    .get(&target.llm_client)
                    .is_some_and(|client| client.forward_auth)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} cannot use eligibility with forwarded caller authentication"
                )));
            }
        }
        Ok(())
    }

    fn resolve_lane_targets<'a>(
        &'a self,
        algorithm: &'a AlgorithmSpec,
        overrides: &TargetOverrides<'_>,
    ) -> RunnerResult<LaneTargets<'a>> {
        let decisions = algorithm.decision_target_names();
        callable_targets(algorithm)
            .into_iter()
            .filter(|name| !decisions.contains(name))
            .map(|name| {
                let bound = bound_target(name, overrides);
                let (resolved_name, config) =
                    self.targets.get_key_value(bound).ok_or_else(|| {
                        RunnerError::configuration(format!(
                            "route references unknown target {bound}"
                        ))
                    })?;
                Ok((
                    name,
                    ResolvedLlmTarget {
                        name: resolved_name.as_str(),
                        config,
                    },
                ))
            })
            .collect()
    }

    fn resolve_decision_targets<'a>(
        &'a self,
        route_name: &str,
        algorithm: &'a AlgorithmSpec,
        overrides: &TargetOverrides<'_>,
        clients: &'a BTreeMap<String, Arc<dyn RoutedDecisionClient>>,
    ) -> RunnerResult<LaneDecisions<'a>> {
        algorithm
            .decision_target_names()
            .into_iter()
            .map(|name| {
                let bound = bound_target(name, overrides);
                let target = self.decision_targets.get(bound).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "route {route_name} requires unknown decision target {bound}"
                    ))
                })?;
                let client = clients.get(&target.decision_client).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "decision target {bound} references unknown decision client {}",
                        target.decision_client
                    ))
                })?;
                Ok((
                    name,
                    ResolvedDecisionTarget {
                        model: &target.id,
                        client,
                    },
                ))
            })
            .collect()
    }

    fn build_lane(
        &self,
        route_name: &str,
        route: &AlgorithmSpec,
        targets: &LaneTargets<'_>,
        decisions: &LaneDecisions<'_>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        built: Option<AlgorithmAndModels>,
    ) -> RunnerResult<(ExecutionLane, Option<CallerAuthKind>)> {
        let (algorithm, mut models) = if let Some(built) = built {
            built
        } else {
            let model_ids = targets
                .iter()
                .map(|(name, target)| ((*name).to_owned(), ModelId::from(target.name)))
                .chain(
                    decisions
                        .iter()
                        .map(|(name, target)| ((*name).to_owned(), target.model.clone())),
                )
                .collect();
            let (algorithm, models) = route
                .build_with_runtime_models(route_name, &model_ids)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let models = models.with_provider_models(
                targets
                    .values()
                    .map(|target| (ModelId::from(target.name), target.config.id.clone()))
                    .collect(),
            );
            (algorithm, models)
        };
        let (route_clients, caller_auth) =
            self.build_route_clients(route_name, route, targets, decisions, clients)?;
        let anthropic_auxiliary_target =
            self.build_anthropic_auxiliary_target(route, targets, clients);
        let responses_auxiliary_target =
            self.build_responses_auxiliary_target(route, targets, clients);
        if self
            .routes
            .get(route_name)
            .is_some_and(|config| config.eligibility.is_some())
        {
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
            if client_config.responses_reasoning.is_some()
                && !matches!(client_config.format, ClientFormat::OpenAiResponses)
            {
                return Err(RunnerError::configuration(format!(
                    "llm client {name} responses_reasoning is only valid for openai_responses"
                )));
            }
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
                // Anthropic rejects any other value with a 400 on every request.
                if matches!(client_config.format, ClientFormat::AnthropicMessages)
                    && !ANTHROPIC_EFFORTS.contains(&effort.as_str())
                {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort {effort:?} is not an anthropic_messages effort; use one of {}",
                        ANTHROPIC_EFFORTS.join(", ")
                    )));
                }
            }
            model_configs.push(
                ModelConfig::new(
                    target_name.as_str(),
                    build_backend(
                        &target.llm_client,
                        client_config,
                        &target.extra_body,
                        &target.omit_body_fields,
                        target.reasoning_effort.clone(),
                    )?,
                    None,
                )
                .with_responses_reasoning(client_config.responses_reasoning.unwrap_or_default())
                .with_provider_model(target.id.clone()),
            );
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
                let DecisionClientConfig {
                    format,
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
                let timeout = Duration::from_millis(*timeout_ms);
                let client = match format {
                    DecisionFormat::SystemOne => {
                        SystemOneClient::new(endpoint.0.clone(), api_key.clone(), timeout)
                            .map(|client| Arc::new(client) as Arc<dyn RoutedDecisionClient>)
                    }
                    DecisionFormat::OpenaiDecisions => {
                        OpenAiDecisionClient::new(endpoint.0.clone(), api_key.clone(), timeout)
                            .map(|client| Arc::new(client) as Arc<dyn RoutedDecisionClient>)
                    }
                }
                .map_err(|error| RunnerError::configuration(error.to_string()))?;
                provider_api_keys.push(api_key);
                Ok((name.clone(), client))
            })
            .collect()
    }

    /// Algorithms route on LLM target names, so targets that share a provider model keep
    /// their own settings. The LLM client sends each target's `id` upstream. Decision targets
    /// route on their `id`, which the decision client sends as is.
    fn build_targets(&self) -> BTreeMap<String, ModelId> {
        self.targets
            .keys()
            .map(|name| (name.clone(), ModelId::from(name.as_str())))
            .chain(
                self.decision_targets
                    .iter()
                    .map(|(name, config)| (name.clone(), config.id.clone())),
            )
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
        decisions: &LaneDecisions<'_>,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> RunnerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let TargetPromptPolicy {
            prompts,
            routing_answer_target,
        } = self.build_route_target_prompts(route_name, route, targets)?;
        let mut by_model = HashMap::new();
        let mut caller_auth = None;
        let mut has_mixed_families = false;
        let mut forwarding_origins = BTreeSet::new();
        let mut decisions_by_model = HashMap::new();
        for name in route.callable_target_names() {
            if let Some(target) = decisions.get(name) {
                if let Some(previous) =
                    decisions_by_model.insert(target.model.clone(), Arc::clone(target.client))
                    && !Arc::ptr_eq(&previous, target.client)
                {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} maps decision model {} to different clients",
                        target.model
                    )));
                }
                continue;
            }
            let target = targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let client = clients.get(&target.config.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!("target {name} has no constructed llm client"))
            })?;
            let client_config =
                self.llm_clients
                    .get(&target.config.llm_client)
                    .ok_or_else(|| {
                        RunnerError::configuration(format!(
                            "target {name} references unknown llm client {}",
                            target.config.llm_client
                        ))
                    })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                has_mixed_families |= caller_auth.is_some_and(|kind| kind != target_auth);
                caller_auth = Some(target_auth);
                forwarding_origins.insert(client_config.base_url.0.origin().ascii_serialization());
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(ModelId::from(target.name), client);
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
            .map(|name| ModelId::from(targets[name].name))
            .collect::<Vec<_>>();
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
                    let target = &targets[name];
                    (!self.llm_clients[&target.config.llm_client].fallback_on_rate_limit)
                        .then(|| ModelId::from(target.name))
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
        for name in route.routing_target_names() {
            let target = targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            if let Some(prompt) = &target.config.system_prompt {
                prompts.insert(ModelId::from(target.name), prompt.clone());
            }
        }
        let mut policy = TargetPromptPolicy {
            prompts,
            routing_answer_target: None,
        };
        let Some((response_name, dependency_name)) = route.routing_response_and_dependency() else {
            return Ok(policy);
        };
        let response = &targets[response_name];
        if !policy.prompts.contains_key(response.name) {
            return Ok(policy);
        }
        if std::iter::once(dependency_name)
            .chain(route.judge_target_names())
            .any(|name| {
                targets
                    .get(name)
                    .is_some_and(|dependency| dependency.name == response.name)
            })
        {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot apply system_prompt to target {response_name}: it is also the routing-only target"
            )));
        }
        policy.routing_answer_target = Some(ModelId::from(response.name));
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
        efficient_target_name: &str,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<ModelPropertiesProbe> {
        let efficient_target = self.targets.get(efficient_target_name)?;
        let client_config = self.llm_clients.get(&efficient_target.llm_client)?;
        if !matches!(
            client_config.context_discovery,
            Some(ContextDiscovery::LlamaCpp)
        ) {
            return None;
        }
        let client = clients.get(&efficient_target.llm_client)?;
        Some(ModelPropertiesProbe {
            model: ModelId::from(efficient_target_name),
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
                    config.name,
                    config.config,
                    clients,
                    AuxiliaryOperation::AnthropicCountTokens,
                )?;
                Some((
                    count_tokens_priority(config.name, &config.config.id),
                    index,
                    target,
                ))
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
            let target = targets.get(name)?;
            self.build_auxiliary_target(
                target.name,
                target.config,
                clients,
                AuxiliaryOperation::ResponsesInputTokens,
            )
        })
    }

    fn build_auxiliary_target(
        &self,
        name: &str,
        target: &TargetConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        operation: AuxiliaryOperation,
    ) -> Option<AuxiliaryTarget> {
        let client = clients.get(&target.llm_client)?;
        let model = ModelId::from(name);
        client
            .supports_auxiliary(&model, operation)
            .then(|| AuxiliaryTarget {
                model,
                client: client.clone(),
            })
    }
}

/// Values Anthropic accepts for `output_config.effort`.
const ANTHROPIC_EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

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
    /// Opt-in context discovery for VGR's efficient target. Unset disables discovery.
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
    responses_reasoning: Option<ResponsesReasoningPolicy>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ContextDiscovery {
    LlamaCpp,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionClientConfig {
    format: DecisionFormat,
    endpoint: HttpBaseUrl,
    api_key_env: String,
    timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DecisionFormat {
    SystemOne,
    OpenaiDecisions,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionTargetConfig {
    #[serde(default)]
    labels: BTreeSet<String>,
    id: ModelId,
    decision_client: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    #[serde(default)]
    labels: BTreeSet<String>,
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
    /// Top-level request fields dropped from the outbound body before `extra_body` is merged.
    /// For providers that reject an otherwise standard field.
    #[serde(default)]
    omit_body_fields: BTreeSet<String>,
    system_prompt: Option<String>,
    /// Reasoning effort forced on every request to this target, replacing the caller's value.
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
    use crate::require_target_labels;
    use serde_json::json;
    use switchyard_protocol::{Category, Request};

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
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "capable_first"
confidence_threshold = 0.0
efficient_hold_turns = -1
capable_system_prompt = "Inspect the task and plan before modifying files."
"#;

    fn decision_eligibility_config() -> String {
        let config = VALID_CONFIG
            .replace(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nlabels = ['private']",
            )
            .replace(
                "id = \"weak/model\"",
                "id = \"weak/model\"\nlabels = ['private']",
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

[routes.typed.eligibility]
"#
        )
    }

    fn eligible_pool_config() -> String {
        let config = VALID_CONFIG
            .replace("id = \"strong/model\"", "id = \"strong/model\"\nlabels = ['private']")
            .replace(
                "type = \"passthrough\"\ntarget = \"weak\"",
                "type = \"random\"\ntargets = [\"weak\", \"strong\"]\nweights = [3.0, 1.0]\nseed = 7",
            );
        format!(
            r#"{config}

[routes.passthrough.eligibility]
"#
        )
    }

    #[tokio::test]
    async fn eligibility_requires_approved_roles_and_decision_targets() -> RunnerResult<()> {
        let configured = decision_eligibility_config();
        let runner = Runner::from_toml(&configured)?;
        let route = runner
            .route("switchyard/typed")
            .expect("typed eligible route should exist");
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
                "required target standard",
            ),
            (
                configured.replace(
                    "id = \"weak/model\"\nlabels = ['private']",
                    "id = \"weak/model\"",
                ),
                "required target weak",
            ),
        ] {
            let runner = Runner::from_toml(&config)?;
            let mut request = Request::default();
            require_target_labels(&mut request, ["private"])?;
            let message = runner
                .route("switchyard/typed")
                .expect("typed route")
                .decide(request)
                .await
                .err()
                .expect("unapproved role must fail before model I/O")
                .to_string();
            assert!(
                message.contains(expected),
                "expected {expected:?}, got {message}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn label_requirements_select_only_approved_targets() -> RunnerResult<()> {
        let configured = eligible_pool_config();
        let runner = Runner::from_toml(&configured)?;
        let route = runner
            .route("switchyard/passthrough")
            .expect("eligible route should exist");

        let mut request = Request::default();
        require_target_labels(&mut request, ["private"])?;
        let outcome = route.decide(request).await?;
        assert_eq!(outcome.selected_model_id()?, "strong");
        assert_eq!(
            &runner
                .describe_decision(&ModelId::from("switchyard/passthrough"), &outcome)
                .expect("selected target metadata")
                .selected
                .target,
            outcome.selected_model_id()?.as_str()
        );
        for (config, route_id, expected) in [
            (
                configured.replace("labels = ['private']", "labels = []"),
                "switchyard/passthrough",
                "without an eligible target",
            ),
            (
                format!("{configured}\n[routes.plan_execute.eligibility]\n"),
                "switchyard/plan-execute",
                "required target weak",
            ),
        ] {
            let runner = Runner::from_toml(&config)?;
            let mut request = Request::default();
            require_target_labels(&mut request, ["private"])?;
            let error = runner
                .route(route_id)
                .expect("configured route")
                .decide(request)
                .await
                .err()
                .expect("no eligible setup");
            assert!(error.to_string().contains(expected), "{error}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_routes_reject_label_requirements() -> RunnerResult<()> {
        let runner = Runner::from_toml(VALID_CONFIG)?;
        let ordinary = runner.route("switchyard/random").expect("ordinary route");
        for labels in [&[][..], &["private"][..]] {
            let mut request = Request::default();
            require_target_labels(&mut request, labels)?;
            assert!(ordinary.decide(request).await.is_err());
        }
        Ok(())
    }

    fn alternative_config(algorithm: &str) -> String {
        let globals = VALID_CONFIG.split("[routes.noop]").next().expect("globals");
        let mut config = globals.to_string();
        for name in ["weak", "strong", "classifier"] {
            config.push_str(&format!(
                "\n[targets.approved_{name}]\nid = 'approved/{name}'\nllm_client = 'primary'\nlabels = ['private']\n"
            ));
        }
        config.push_str(&format!(
            "\n[routes.test]\nid = 'route'\n{algorithm}\n[routes.test.eligibility.target_alternatives]\n"
        ));
        let spec: AlgorithmSpec = toml::from_str(algorithm).expect("test algorithm");
        for name in spec
            .callable_target_names()
            .into_iter()
            .collect::<BTreeSet<_>>()
        {
            config.push_str(&format!("{name} = ['approved_{name}']\n"));
        }
        config
    }

    // Alternatives preserve fixed roles and selected-target metadata.
    #[tokio::test]
    async fn alternatives_preserve_router_roles() -> RunnerResult<()> {
        for (algorithm, initial) in [
            ("type = 'passthrough'\ntarget = 'weak'", "weak"),
            (
                "type = 'stage_router'\ncapable_target = 'strong'\nefficient_target = 'weak'\npicker = 'capable_first'\nconfidence_threshold = 0.0",
                "strong",
            ),
        ] {
            let runner = Runner::from_toml(&alternative_config(algorithm))?;
            let route = runner.route("route").expect("route");
            let standard = route.decide(Request::default()).await?;
            assert_eq!(standard.selected_model_id()?.as_str(), initial);
            let mut request = Request::default();
            require_target_labels(&mut request, ["private"])?;
            let outcome = route.decide(request).await?;
            assert_eq!(outcome.selected_model_id()?, &format!("approved_{initial}"));
            assert!(
                outcome
                    .selected_model_ids
                    .iter()
                    .all(|model| model.starts_with("approved_"))
            );
            let description = runner
                .describe_decision(&ModelId::from("route"), &outcome)
                .expect("mapped metadata");
            assert_eq!(description.selected.target, format!("approved_{initial}"));
        }
        Ok(())
    }

    // Invalid references fail at load; unavailable approved bindings fail at admission.
    #[tokio::test]
    async fn alternatives_reject_invalid_bindings() -> RunnerResult<()> {
        let config = alternative_config(
            "type = 'auto'\ncapable_target = 'strong'\nefficient_target = 'weak'",
        );
        for (invalid, expected) in [
            (
                format!("{config}typo = ['approved_weak']\n"),
                "not used by the algorithm",
            ),
            (
                config.replace("weak = ['approved_weak']", "weak = ['missing']"),
                "must exist and match the target kind",
            ),
            (
                config.replace("weak = ['approved_weak']", "weak = []"),
                "target alternatives must not be empty",
            ),
            (
                config.replace(
                    "[llm_clients.primary]",
                    "[llm_clients.primary]\nforward_auth = true",
                ),
                "forwarded caller authentication",
            ),
        ] {
            let message = error_message(&invalid);
            assert!(
                message.contains(expected),
                "expected {expected:?}, got {message}"
            );
        }
        let random =
            alternative_config("type = 'random'\ntargets = ['weak', 'strong']\nweights = [0, 1]")
                .replace("strong = ['approved_strong']", "strong = ['approved_weak']");
        for (source, expected) in [
            (
                config.replace("weak = ['approved_weak']", "weak = ['strong']"),
                "required target weak",
            ),
            (random, "multiple pool entries to target approved_weak"),
        ] {
            let runner = Runner::from_toml(&source)?;
            let mut request = Request::default();
            require_target_labels(&mut request, ["private"])?;
            let error = runner
                .route("route")
                .expect("configured route")
                .decide(request)
                .await
                .err()
                .expect("invalid eligible bindings");
            assert!(error.to_string().contains(expected), "{error}");
        }
        Runner::from_toml(&config.replace("id = 'approved/weak'", "id = 'weak/model'"))?;
        let shared_model =
            alternative_config("type = 'random'\ntargets = ['weak', 'strong']\nweights = [0, 1]")
                .replace("id = 'approved/strong'", "id = 'approved/weak'");
        let runner = Runner::from_toml(&shared_model)?;
        let mut request = Request::default();
        require_target_labels(&mut request, ["private"])?;
        assert_eq!(
            runner
                .route("route")
                .expect("route")
                .decide(request)
                .await?
                .selected_model_id()?,
            "approved_strong"
        );
        let wrong_kind = format!(
            "{}\n[routes.typed.eligibility.target_alternatives]\nstandard = ['weak']\n",
            decision_eligibility_config()
        );
        assert!(error_message(&wrong_kind).contains("must exist and match the target kind"));
        Ok(())
    }

    // A secondary verifier must not inherit the answer target's prompt after binding.
    #[tokio::test]
    async fn alternatives_reject_prompted_vgr_secondary_judge_alias() -> RunnerResult<()> {
        let source = alternative_config(
            "type = 'vgr'\nefficient_target = 'weak'\ncapable_target = 'strong'\njudge_target = 'strong'\ncapable_judge_target = 'classifier'",
        )
        .replace(
            "classifier = ['approved_classifier']",
            "classifier = ['approved_weak']",
        )
        .replace(
            "[targets.approved_weak]",
            "[targets.approved_weak]\nsystem_prompt = 'Answer the user.'",
        );
        let runner = Runner::from_toml(&source)?;
        let mut request = Request::default();
        require_target_labels(&mut request, ["private"])?;
        let error = runner
            .route("route")
            .expect("VGR route")
            .decide(request)
            .await
            .err()
            .expect("secondary judge alias must fail before model I/O");
        assert!(matches!(error, RunnerError::Configuration { .. }));
        assert!(error.to_string().contains("routing-only target"), "{error}");
        Ok(())
    }

    // Decision IDs remain provider keys, so bindings cannot give one ID two clients.
    #[tokio::test]
    async fn alternatives_reject_ambiguous_composite_decision_clients() -> RunnerResult<()> {
        let source = decision_eligibility_config().replace(
            "decision_client = \"privacy\"\nlabels = [\"private\"]",
            "decision_client = \"privacy\"",
        );
        let source = format!(
            r#"{source}
[decision_clients.other]
format = "system_one"
endpoint = "http://127.0.0.1:1/decisions"
api_key_env = "PATH"
timeout_ms = 1

[decision_targets.secondary]
id = "decision/secondary"
decision_client = "privacy"

[decision_targets.approved_primary]
id = "decision/shared"
decision_client = "privacy"
labels = ["private"]

[decision_targets.approved_secondary]
id = "decision/shared"
decision_client = "other"
labels = ["private"]

[routes.ambiguous]
id = "switchyard/ambiguous"
type = "composite"
classifier = {{ target = "standard", classify_trigger = "user_turn", decision = {{ cutoff = 0.4 }} }}
stage = {{ capable_target = "strong", efficient_target = "weak", confidence_threshold = 0.5 }}
efficient = {{ classifier = {{ target = "secondary", decision = {{ cutoff = 0.4 }} }} }}

[routes.ambiguous.eligibility.target_alternatives]
standard = ["approved_primary"]
secondary = ["approved_secondary"]
"#
        );
        let runner = Runner::from_toml(&source)?;
        let mut request = Request::default();
        require_target_labels(&mut request, ["private"])?;
        let error = runner
            .route("switchyard/ambiguous")
            .expect("composite route")
            .decide(request)
            .await
            .err()
            .expect("ambiguous decision clients must fail before model I/O");
        assert!(matches!(error, RunnerError::Configuration { .. }));
        assert!(
            error
                .to_string()
                .contains("maps decision model decision/shared to different clients"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn eligibility_rejects_unknown_fields_and_invalid_scope() {
        let configured = eligible_pool_config();
        for (invalid, expected) in [
            (
                configured.replace("labels = ['private']", "label = ['private']"),
                "unknown field `label`",
            ),
            (
                configured.replace(
                    "[routes.passthrough.eligibility]",
                    "[routes.passthrough.eligible]",
                ),
                "unknown field `eligible`",
            ),
            (
                decision_eligibility_config().replace(
                    "[decision_targets.standard]",
                    "[decision_targets.standard]\nlabel = ['private']",
                ),
                "unknown field `label`",
            ),
            (
                format!("{configured}\nrequire_explicit_requirement = true\n"),
                "unknown field `require_explicit_requirement`",
            ),
            (
                format!("{configured}\nscope = 'tenant'\n"),
                "unknown variant `tenant`",
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
    fn eligibility_rejects_conflicting_approvals_and_forwarded_auth() {
        let configured = eligible_pool_config();
        for (invalid, expected) in [
            (
                format!(
                    "{configured}\n[targets.alias]\nid = \"strong/model\"\nllm_client = \"responses\"\n"
                ),
                "disagree on approval labels",
            ),
            (
                format!(
                    "{}\n[decision_targets.alias]\nid = \"decision/standard\"\ndecision_client = \"privacy\"\n",
                    decision_eligibility_config()
                ),
                "disagrees on approval labels",
            ),
            (
                configured.replace(
                    "[llm_clients.responses]\nformat = \"openai_responses\"",
                    "[llm_clients.responses]\nformat = \"openai_responses\"\nforward_auth = true",
                ),
                "cannot use eligibility with forwarded caller authentication",
            ),
            (
                alternative_config(
                    "type = 'prefill_router'\ntargets = ['weak']\ncheckpoint = 'unused'",
                ),
                "cannot use eligibility with prefill_router",
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
    fn public_runner_from_toml_builds_a_deployment() -> RunnerResult<()> {
        let config = format!(
            "{VALID_CONFIG}\n[routes.classifier.infra]\n\
             max_signal_age_ms = 1000\nmin_waiting_requests = 4\nmax_capable_inflight = 2"
        );
        let runner = Runner::from_toml(&config)?;
        assert!(
            Runner::from_toml(
                &config.replace("min_waiting_requests = 4", "min_waiting_requests = 0")
            )
            .is_err()
        );

        let classifier = runner
            .route("switchyard/classifier")
            .expect("classifier route should exist");
        let models = classifier.models();
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("classifier")]
        );
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("weak")]
        );
        assert_eq!(
            models.models_for(&Category::Capable),
            [ModelId::from("strong")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("weak"), ModelId::from("strong")]
        );
        let plan_execute = runner
            .route("switchyard/plan-execute")
            .expect("plan-execute route should exist");
        assert_eq!(
            plan_execute.models().models_for(&Category::Capable),
            [ModelId::from("strong")]
        );
        assert_eq!(
            plan_execute.models().models_for(&Category::Efficient),
            [ModelId::from("weak")]
        );
        assert!(runner.route("switchyard/passthrough").is_some());
        Ok(())
    }

    /// Both the TOML and JSON parsers must reject the removed `plan_execute` type.
    #[test]
    fn removed_plan_execute_type_is_rejected() {
        let route = r#"
type = "plan_execute"
capable_target = "strong"
efficient_target = "weak"
"#;
        let config = format!("{VALID_CONFIG}\n[routes.obsolete]\nid = \"obsolete\"\n{route}");
        assert!(error_message(&config).contains("unknown variant `plan_execute`"));

        let error = serde_json::from_str::<AlgorithmSpec>(
            r#"{"type":"plan_execute","capable_target":"strong","efficient_target":"weak"}"#,
        )
        .expect_err("the removed algorithm must be rejected");
        assert!(error.to_string().contains("unknown variant `plan_execute`"));
    }

    /// TOML and JSON parsers must reject old `plan_execute` fields on `stage_router` routes.
    /// Changing only the route type must not load a route that ignores its handoff settings.
    #[test]
    fn stage_rejects_obsolete_plan_execute_settings() {
        let stage = json!({
            "type": "stage_router", "capable_target": "strong", "efficient_target": "weak",
            "picker": "capable_first", "confidence_threshold": 0.0,
            "efficient_hold_turns": -1, "capable_system_prompt": "Plan before editing.",
        });
        assert!(serde_json::from_str::<AlgorithmSpec>(&stage.to_string()).is_ok());
        for (field, value, toml_value) in [
            ("planning_prompt", json!("plan"), r#""plan""#),
            ("handoff_prompt", json!("execute"), r#""execute""#),
            ("planner_reasoning_as_text", json!(true), "true"),
            ("execution_trigger", json!("edit"), r#""edit""#),
            (
                "judge",
                json!({"target": "classifier"}),
                r#"{ target = "classifier" }"#,
            ),
        ] {
            let mut invalid = stage.clone();
            invalid[field] = value;
            let error = serde_json::from_str::<AlgorithmSpec>(&invalid.to_string())
                .expect_err("obsolete fields must be rejected");
            assert!(error.to_string().contains(field), "{error}");
            let error = error_message(&format!("{VALID_CONFIG}\n{field} = {toml_value}\n"));
            assert!(error.contains(field), "{error}");
        }
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
efficient_target = "weak"
capable_target = "strong"
judge_target = "classifier"
mode = "active"
active_approval = "prospective-validation-and-canary-approved"
task_typing = false
"#
        )
    }

    #[test]
    fn vgr_compaction_is_opt_in_from_toml() -> RunnerResult<()> {
        for setting in ["", "compact_handoff = false", "compact_handoff = true"] {
            let source = format!("{}\nagentic_handoff = true\n{setting}\n", vgr_config());
            let parsed: DeploymentConfig = toml::from_str(&source).expect("valid VGR config");
            let AlgorithmSpec::Vgr { config } = &parsed.routes["vgr"].algorithm else {
                panic!("expected VGR route");
            };
            assert_eq!(config.compact_handoff, setting.ends_with("= true"));
            assert!(runner_from_toml(&source)?.route("switchyard/vgr").is_some());
        }
        Ok(())
    }

    #[test]
    fn vgr_route_builds_and_rejects_unsafe_configuration() -> RunnerResult<()> {
        let runner = runner_from_toml(&vgr_config())?;
        assert!(runner.route("switchyard/vgr").is_some());

        let compact = format!("{}\ncompact_handoff = true\n", vgr_config());
        assert!(error_message(&compact).contains("compact_handoff requires agentic_handoff"));
        let compact = format!("{compact}\nagentic_handoff = true\n");
        assert!(
            runner_from_toml(&compact)?
                .route("switchyard/vgr")
                .is_some()
        );

        let identical_ids = vgr_config().replace(
            "efficient_target = \"weak\"",
            "efficient_target = \"strong\"",
        );
        assert!(error_message(&identical_ids).contains("distinct efficient and capable model IDs"));

        let bad_approval =
            vgr_config().replace("prospective-validation-and-canary-approved", "approved");
        assert!(error_message(&bad_approval).contains("approval attestation"));

        let missing_target = vgr_config().replace(
            "efficient_target = \"weak\"",
            "efficient_target = \"missing\"",
        );
        assert!(error_message(&missing_target).contains("unknown target missing"));

        let zero_cooldown = format!("{}breaker_cooldown_seconds = 0\n", vgr_config());
        assert!(
            error_message(&zero_cooldown).contains("breaker_cooldown_seconds must be a positive")
        );
        Ok(())
    }

    #[test]
    fn vgr_local_server_example_builds_and_rejects_unknown_discovery() -> RunnerResult<()> {
        let example = include_str!("../../../examples/vgr-local-servers.toml");
        let runner = runner_from_toml(example)?;
        assert!(runner.route("switchyard/vgr").is_some());
        assert!(runner.route("switchyard/only-efficient").is_some());
        let parsed: DeploymentConfig = toml::from_str(example).expect("valid VGR example");
        assert!(matches!(
            &parsed.routes["auxiliary"].algorithm,
            AlgorithmSpec::Passthrough { target, .. } if target == "efficient"
        ));
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
            [ModelId::from("strong")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("strong"), ModelId::from("weak")]
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
    fn stage_and_composite_validate_hold_turns() {
        let composite =
            composite_config().replace("capable_hold_turns = 2", "capable_hold_turns = 0");
        for source in [
            stage_config(),
            stage_config().replace("efficient_first", "capable_first"),
            composite_config(),
            format!("{composite}\n[routes.composed.efficient]\ncapable_hold_turns = 2"),
            format!("{composite}\n[routes.composed.capable]\ncapable_hold_turns = 2"),
        ] {
            for field in ["capable_hold_turns", "efficient_hold_turns"] {
                for (turns, valid) in [
                    (-2_i64, false),
                    (-1, true),
                    (0, true),
                    (3, true),
                    (i64::from(u32::MAX), true),
                    (i64::from(u32::MAX) + 1, false),
                ] {
                    let configured =
                        source.replace("capable_hold_turns = 2", &format!("{field} = {turns}"));
                    let result = AlgorithmFactory::from_toml(&configured)
                        .and_then(|factory| factory.build());
                    assert_eq!(result.is_ok(), valid, "{field} = {turns}");
                    if let Err(error) = result {
                        assert!(error.to_string().contains(field), "{error}");
                    }
                }
            }
        }
    }

    #[test]
    fn composite_decision_judge_reuses_capability_validation() {
        let source = composite_config().replace(
            "[routes.composed.classifier]\ntarget = \"tier_judge\"\nbase_threshold = 0.5",
            "[routes.composed.classifier]\ntarget = \"tier_judge\"",
        ) + r#"
[routes.composed.classifier.decision]
cutoff = 0.4
instructions = "candidate comparison"
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
            (source.replace("cutoff = 0.4", "cutoff = 1.1"), false),
        ] {
            let deployment: DeploymentConfig = toml::from_str(&source).unwrap();
            let spec = &deployment.routes["composed"].algorithm;
            assert_eq!(spec.classifier_targets().next().unwrap().0, "tier_judge");
            assert_eq!(
                spec.build_with_runtime_models("composed", &deployment.build_targets())
                    .is_ok(),
                valid
            );
        }
    }

    #[test]
    fn decision_target_judges_use_default_decision_settings() -> RunnerResult<()> {
        let source = format!(
            r#"{VALID_CONFIG}
[decision_clients.decisions]
format = "openai_decisions"
endpoint = "https://example.test/v1/decisions"
api_key_env = "PATH"
timeout_ms = 5000

[decision_targets.judge]
id = "decision/judge"
decision_client = "decisions"

[routes.judged]
id = "switchyard/judged"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"

[routes.staged]
id = "switchyard/staged"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.staged.classifier]
target = "judge"

[routes.composed]
id = "switchyard/composed"
type = "composite"

[routes.composed.classifier]
target = "judge"
classify_trigger = "user_turn"

[routes.composed.stage]
capable_target = "strong"
efficient_target = "weak"

[routes.composed.efficient.classifier]
target = "judge"
"#
        );
        let factory = AlgorithmFactory::from_toml(&source)?;
        let routes = &factory.config.routes;
        for route in ["judged", "staged", "composed"] {
            let judges: Vec<_> = routes[route].algorithm.classifier_targets().collect();
            assert!(!judges.is_empty(), "{route}");
            assert!(
                judges
                    .iter()
                    .all(|(target, decision)| *target == "judge" && decision.is_some()),
                "{route}"
            );
        }
        // LLM judges keep their capability settings.
        assert!(
            routes["classifier"]
                .algorithm
                .classifier_targets()
                .all(|(_, d)| d.is_none())
        );
        factory.build()?;
        runner_from_toml(&source)?;
        Ok(())
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
        let callable_targets = stage_route.callable_target_names();
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

    // A routing-only judge cannot inherit the answer target's system prompt.
    #[test]
    fn rejects_prompted_answer_target_used_as_routing_only_target() {
        let configured = VALID_CONFIG
            .replace(
                "classifier_target = \"classifier\"",
                "classifier_target = \"weak\"",
            )
            .replace(
                "id = \"weak/model\"\nllm_client = \"anthropic\"",
                "id = \"weak/model\"\nllm_client = \"anthropic\"\nsystem_prompt = \"answer prompt\"",
            )
            .replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nescalation = { confirmations = 1 }",
            );

        assert_eq!(
            error_message(&configured),
            "route classifier cannot apply system_prompt to target weak: it is also the routing-only target"
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
    fn unmatched_steps_is_a_capability_setting() -> RunnerResult<()> {
        // Accepted on a capability route; the value range is checked by the algorithm.
        runner_from_toml(&VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nunmatched_steps = 2",
        ))?;
        assert!(
            error_message(&VALID_CONFIG.replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nunmatched_steps = 3",
            ))
            .contains("unmatched_steps must be 0, 1, or 2")
        );

        // Rejected on an escalation route, including the implicit form where the
        // `escalation` table alone selects the mode: the key is new, so no existing
        // configuration relies on it being ignored there.
        assert!(
            error_message(&VALID_CONFIG.replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nunmatched_steps = 2\nescalation = { confirmations = 2 }",
            ))
            .contains("mode escalation cannot use capability routing settings")
        );
        Ok(())
    }

    #[test]
    fn a_target_reasoning_effort_parses_on_every_client_format() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let weak = "[targets.weak]\nid = \"weak/model\"\nllm_client = \"anthropic\"";
        assert!(VALID_CONFIG.contains(strong) && VALID_CONFIG.contains(weak));

        let forced = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \"max\""));
        runner_from_toml(&forced)?;

        let blank = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \" \""));
        assert!(error_message(&blank).contains("reasoning_effort must not be empty"));

        let anthropic = VALID_CONFIG.replace(weak, &format!("{weak}\nreasoning_effort = \"high\""));
        runner_from_toml(&anthropic)?;

        let minimal =
            VALID_CONFIG.replace(weak, &format!("{weak}\nreasoning_effort = \"minimal\""));
        assert!(
            error_message(&minimal).contains("is not an anthropic_messages effort"),
            "{}",
            error_message(&minimal)
        );
        Ok(())
    }

    #[test]
    fn decision_target_ids_must_not_match_llm_target_names() {
        let configured = format!(
            "{VALID_CONFIG}\n[decision_clients.decisions]\nformat = \"system_one\"\nendpoint = \"https://example.test/decisions\"\napi_key_env = \"PATH\"\ntimeout_ms = 5000\n\n[decision_targets.judge]\nid = \"strong\"\ndecision_client = \"decisions\"\n"
        );
        let message = error_message(&configured);
        assert!(
            message.contains("decision target judge id strong is also an LLM target name"),
            "{message}"
        );
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
    fn stage_router_validates_the_optional_balanced_target() -> RunnerResult<()> {
        let balanced = VALID_CONFIG.replace(
            "picker = \"capable_first\"",
            "picker = \"capable_first\"\nbalanced_target = \"classifier\"",
        );
        runner_from_toml(&balanced)?;
        runner_from_toml(&balanced.replace(
            "balanced_target = \"classifier\"",
            "balanced_target = \"classifier\"\ncapable_score_threshold = 0.55",
        ))?;
        assert!(
            error_message(&balanced.replace(
                "balanced_target = \"classifier\"",
                "balanced_target = \"missing\""
            ))
            .contains("missing")
        );
        for value in ["-0.1", "1.1", "nan", "inf"] {
            let invalid = balanced.replace(
                "balanced_target = \"classifier\"",
                &format!("balanced_target = \"classifier\"\ncapable_score_threshold = {value}"),
            );
            assert!(error_message(&invalid).contains("capable_score_threshold"));
        }
        let no_target = VALID_CONFIG.replace(
            "picker = \"capable_first\"",
            "picker = \"capable_first\"\ncapable_score_threshold = 0.55",
        );
        assert!(error_message(&no_target).contains("requires balanced_target"));
        let unreachable =
            balanced.replace("confidence_threshold = 0.0", "confidence_threshold = 0.6");
        assert!(error_message(&unreachable).contains("above confidence_threshold"));
        let no_hold_target = VALID_CONFIG.replace(
            "picker = \"capable_first\"",
            "picker = \"capable_first\"\nbalanced_hold_turns = 3",
        );
        assert!(error_message(&no_hold_target).contains("requires balanced_target"));
        // The balanced settings work alongside the existing hold keys.
        runner_from_toml(&balanced.replace(
            "balanced_target = \"classifier\"",
            "balanced_target = \"classifier\"\nbalanced_hold_turns = 3\ncapable_hold_turns = 4",
        ))?;
        Ok(())
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
            let error = AlgorithmFactory::from_toml(&toml)
                .and_then(|factory| factory.build())
                .err()
                .expect("invalid configuration should fail")
                .to_string();
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
        // One model may be served through different clients in separate routes.
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
    fn responses_reasoning_defaults_and_accepts_drop() -> RunnerResult<()> {
        let default: DeploymentConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let responses = default
            .llm_clients
            .get("responses")
            .ok_or_else(|| RunnerError::configuration("responses llm client is missing"))?;
        assert_eq!(responses.responses_reasoning, None);

        let configured = VALID_CONFIG.replacen(
            "[llm_clients.responses]\nformat = \"openai_responses\"\nbase_url = \"https://example.test/v1\"",
            "[llm_clients.responses]\nformat = \"openai_responses\"\nbase_url = \"https://example.test/v1\"\nresponses_reasoning = \"drop\"",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let responses = config
            .llm_clients
            .get("responses")
            .ok_or_else(|| RunnerError::configuration("responses llm client is missing"))?;
        assert_eq!(
            responses.responses_reasoning,
            Some(ResponsesReasoningPolicy::Drop)
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn responses_reasoning_rejects_other_formats() {
        let invalid = VALID_CONFIG.replacen(
            "format = \"openai_chat\"",
            "format = \"openai_chat\"\nresponses_reasoning = \"drop\"",
            1,
        );
        assert!(
            error_message(&invalid)
                .contains("responses_reasoning is only valid for openai_responses")
        );
    }

    #[test]
    fn responses_reasoning_rejects_unreferenced_other_format() {
        let invalid = format!(
            "{VALID_CONFIG}\n[llm_clients.unused]\nformat = \"openai_chat\"\nbase_url = \"https://example.test/v1\"\nresponses_reasoning = \"drop\"\n"
        );
        assert!(
            error_message(&invalid)
                .contains("responses_reasoning is only valid for openai_responses")
        );
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
            [ModelId::from("executor")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("executor")]
        );
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("advisor")]
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
