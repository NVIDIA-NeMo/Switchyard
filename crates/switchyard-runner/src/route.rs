// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured route execution and the clients that serve its targets.

use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome, RuntimeModels};
use serde_json::Value;
use switchyard_llm_client::{AuxiliaryOperation, ClientRouter, RunObserver, TranslatingLlmClient};
use switchyard_protocol::{
    LlmClientError, ModelId, Request, Response, RoutedLlmClient, WireFormat,
};
use thiserror::Error;

use crate::DecisionTarget;
use crate::TargetRequirements;
use crate::config::EligibleLanes;
use crate::eligibility::{EligibilityPolicy, has_target_requirements};

/// Capabilities declared for one route.
///
/// `GET /v1/models` includes `context_window`, `tool_calling`, and `vision` in each
/// standard `data` entry, using `null` for unset values. It omits `reasoning`.
/// The server rejects tool inputs, reasoning controls, or images when the
/// corresponding declaration is explicitly `false` for the selected route.
#[derive(Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub context_window: Option<u32>,
    pub tool_calling: Option<bool>,
    /// Whether the routed model accepts reasoning controls, as declared in config.
    pub reasoning: Option<bool>,
    /// Whether the routed model accepts image input, as declared in config.
    pub vision: Option<bool>,
}

/// Caller credential family required by a forwarded-auth route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerAuthKind {
    Anthropic,
    OpenAi,
}

impl CallerAuthKind {
    /// Stable provider name used by the server compatibility API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    const fn accepts(self, wire_format: WireFormat) -> bool {
        matches!(
            (self, wire_format),
            (Self::Anthropic, WireFormat::AnthropicMessages)
                | (
                    Self::OpenAi,
                    WireFormat::OpenAiChat | WireFormat::OpenAiResponses
                )
        )
    }
}

/// Exact upstream model and client used for an auxiliary provider operation.
#[derive(Clone)]
pub struct AuxiliaryTarget {
    pub model: ModelId,
    pub client: Arc<TranslatingLlmClient>,
}

/// Error returned while loading or executing configured routes.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("unknown route model {0:?}")]
    UnknownRouteModel(String),
    #[error("caller format is incompatible with {} credentials", .0.as_str())]
    IncompatibleCallerFormat(CallerAuthKind),
    #[error("route has no compatible target for the auxiliary operation")]
    AuxiliaryUnsupported,
    #[error(transparent)]
    Algorithm(#[from] LibsyError),
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

impl RunnerError {
    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn configuration_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

// Configured algorithm and targets for one execution lane.
pub(crate) struct ExecutionLane {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    anthropic_auxiliary_target: Option<AuxiliaryTarget>,
    responses_auxiliary_target: Option<AuxiliaryTarget>,
    models: Arc<RuntimeModels>,
}

impl ExecutionLane {
    pub(crate) fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        models: RuntimeModels,
    ) -> Self {
        Self {
            algorithm,
            clients,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
            models: Arc::new(models),
        }
    }
}

struct EligibilityExecution {
    policy: EligibilityPolicy,
    lanes: EligibleLanes,
}

struct LaneSelection {
    lane: Arc<ExecutionLane>,
    requirements: Option<TargetRequirements>,
}

/// One or more execution lanes behind a configured route.
pub struct Route {
    standard: Arc<ExecutionLane>,
    eligibility: Option<EligibilityExecution>,
    decision_targets: Vec<DecisionTarget>,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
    infra: Option<libsy::InfraPolicy>,
}

/// The selected model and untouched response produced by a route execution.
pub struct RunOutput {
    pub selected_model: ModelId,
    pub response: Response,
}

impl Route {
    /// Creates a fully configured execution route.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
        models: RuntimeModels,
    ) -> Self {
        Self::from_lane(
            ExecutionLane::new(
                algorithm,
                clients,
                anthropic_auxiliary_target,
                responses_auxiliary_target,
                models,
            ),
            caller_auth,
            capabilities,
            decision_targets,
        )
    }

    pub(crate) fn from_lane(
        lane: ExecutionLane,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        decision_targets: Vec<DecisionTarget>,
    ) -> Self {
        Self {
            standard: Arc::new(lane),
            eligibility: None,
            decision_targets,
            caller_auth,
            capabilities,
            infra: None,
        }
    }

    pub(crate) fn with_eligibility(
        mut self,
        policy: EligibilityPolicy,
        lanes: EligibleLanes,
    ) -> Self {
        self.eligibility = Some(EligibilityExecution { policy, lanes });
        self
    }

    fn select_lane(&self, request: &Request) -> Result<LaneSelection, RunnerError> {
        if let Some(eligibility) = &self.eligibility {
            let required = eligibility.policy.admit(request)?;
            let lane = if required.required_labels().is_empty() {
                Arc::clone(&self.standard)
            } else {
                eligibility
                    .lanes
                    .select(required.required_labels(), &self.standard)?
            };
            let requirements = Some(required);
            tracing::debug!(
                switchyard.route = request.llm_request.model.as_deref().unwrap_or_default(),
                switchyard.algorithm = self.algorithm_name(),
                privacy.lane = requirements.as_ref().map(TargetRequirements::lane),
                privacy.source = requirements.as_ref().map(TargetRequirements::source),
                privacy.reason_code = requirements.as_ref().map(TargetRequirements::reason_code),
                "eligible targets selected"
            );
            return Ok(LaneSelection { lane, requirements });
        }
        if has_target_requirements(request) {
            return Err(LlmClientError::InvalidRequest {
                message: "route does not accept target requirements".to_string(),
            }
            .into());
        }
        Ok(LaneSelection {
            lane: Arc::clone(&self.standard),
            requirements: None,
        })
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.standard.algorithm.name()
    }

    /// Returns model-list capability metadata.
    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Returns the forwarded caller credential family.
    pub fn caller_auth(&self) -> Option<CallerAuthKind> {
        self.caller_auth
    }

    /// Resolves a selected model to this route's non-secret target metadata.
    pub(crate) fn decision_target(&self, model: &ModelId) -> Option<DecisionTarget> {
        self.decision_targets
            .iter()
            .find(|target| target.model == *model)
            .cloned()
    }

    /// Returns the models grouped for the standard execution lane.
    pub fn models(&self) -> &RuntimeModels {
        &self.standard.models
    }

    /// Rejects a caller format incompatible with forwarded credentials.
    pub fn check_caller_format(&self, input_format: WireFormat) -> Result<(), RunnerError> {
        if let Some(kind) = self.caller_auth
            && !kind.accepts(input_format)
        {
            return Err(RunnerError::IncompatibleCallerFormat(kind));
        }
        Ok(())
    }

    /// Executes the configured route without consuming or proxying streamed responses.
    pub async fn execute(
        &self,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<RunOutput, RunnerError> {
        self.execute_with_privacy_observer(request, observer, |_| {})
            .await
    }

    /// Executes the route and reports standard or restricted admission before dispatch.
    pub async fn execute_with_privacy_observer(
        &self,
        request: Request,
        observer: Option<RunObserver>,
        on_privacy_decision: impl FnOnce(&TargetRequirements) + Send,
    ) -> Result<RunOutput, RunnerError> {
        let selection = self.select_lane(&request)?;
        if let Some(decision) = &selection.requirements {
            on_privacy_decision(decision);
        }
        let lane = selection.lane;
        let (selected_model, response) = switchyard_llm_client::run(
            Arc::clone(&lane.algorithm),
            lane.clients.clone(),
            request,
            Arc::clone(&lane.models),
            observer,
        )
        .await?;
        Ok(RunOutput {
            selected_model,
            response,
        })
    }

    /// Completes routing-time calls without serving a post-routing completion.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        let selection = self.select_lane(&request)?;
        let lane = selection.lane;
        switchyard_llm_client::decide(
            Arc::clone(&lane.algorithm),
            lane.clients.clone(),
            request,
            Arc::clone(&lane.models),
        )
        .await
        .map_err(RunnerError::from)
    }

    pub(crate) fn with_infra(mut self, infra: Option<libsy::InfraPolicy>) -> Self {
        self.infra = infra;
        self
    }

    /// Apply the optional queue policy to a completed task decision.
    ///
    /// The host must hold its in-flight accounting lock through this call and the
    /// increment for the selected model. Ordinary `decide` leaves task routing unchanged.
    pub fn apply_infra(
        &self,
        outcome: &mut RoutingOutcome,
        signals: &switchyard_protocol::InfraSignals,
    ) -> Result<bool, RunnerError> {
        match &self.infra {
            Some(policy) => {
                let selection = self.select_lane(&outcome.request)?;
                if outcome.selected_model_ids.iter().any(|model| {
                    !selection
                        .lane
                        .models
                        .models_for(&switchyard_protocol::Category::Any)
                        .contains(model)
                }) {
                    return Err(LlmClientError::InvalidRequest {
                        message: "target requirements changed after the routing decision".into(),
                    }
                    .into());
                }
                policy
                    .apply(outcome, &selection.lane.models, signals)
                    .map_err(Into::into)
            }
            None => Ok(false),
        }
    }

    /// Routes with a host-owned client for routing-time LLM calls, such as judges.
    ///
    /// The host owns transport, credentials, and target request settings. Configured
    /// HTTP clients, target prompts, and decision clients are not used on this path.
    /// The same algorithm instance retains its state across calls. No post-routing
    /// completion is dispatched. Target eligibility still applies to the host-owned path.
    pub async fn decide_with_client(
        &self,
        request: Request,
        client: Arc<dyn RoutedLlmClient>,
    ) -> Result<RoutingOutcome, RunnerError> {
        let selection = self.select_lane(&request)?;
        let lane = selection.lane;
        switchyard_llm_client::decide(
            Arc::clone(&lane.algorithm),
            ClientRouter::single(client),
            request,
            Arc::clone(&lane.models),
        )
        .await
        .map_err(RunnerError::from)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let selection = self.select_lane(&request)?;
        let lane = selection.lane;
        let target = match operation {
            AuxiliaryOperation::AnthropicCountTokens => &lane.anthropic_auxiliary_target,
            AuxiliaryOperation::ResponsesInputTokens | AuxiliaryOperation::ResponsesCompact => {
                &lane.responses_auxiliary_target
            }
        }
        .as_ref()
        .ok_or(RunnerError::AuxiliaryUnsupported)?;
        target
            .client
            .call_auxiliary(&target.model, request, operation)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_protocol::{InfraSignals, Message, ModelInfraSignals, PoolLoad, Role};

    use super::*;
    use crate::{Runner, require_restricted_targets, require_target_labels};

    const PASSTHROUGH: &str = "type = 'passthrough'\ntarget = 'fast'";

    fn config(algorithm: &str) -> String {
        let mut source = String::from(
            "schema_version = 1\n[llm_clients.primary]\nformat = 'openai_chat'\nbase_url = 'https://example.test/v1'\n",
        );
        for name in ["fast", "capable", "judge"] {
            source.push_str(&format!(
                "\n[targets.{name}]\nid = '{name}'\nllm_client = 'primary'\nlabels = []\n\
                 [targets.approved_{name}]\nid = 'approved/{name}'\nllm_client = 'primary'\nlabels = ['private', 'customer_data']\n"
            ));
        }
        source.push_str(&format!(
            "\n[routes.test]\nid = 'route'\n{algorithm}\n\
             [routes.test.eligibility]\nscope = 'request'\n\
             [routes.test.eligibility.target_alternatives]\n"
        ));
        let spec: crate::AlgorithmSpec = toml::from_str(algorithm).expect("test algorithm");
        for target in spec.callable_target_names() {
            source.push_str(&format!("{target} = ['approved_{target}']\n"));
        }
        source
    }

    fn request(task: &str, labels: &[&str]) -> Request {
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some(task.into());
        require_target_labels(&mut request, labels).expect("trusted requirements");
        request
    }

    // Different requirements that resolve to the same bindings share algorithm state.
    #[test]
    fn resolved_bindings_reuse_the_algorithm_and_standard_setup() {
        let source = config(PASSTHROUGH);
        let runner = Runner::from_toml(&source).expect("eligible route");
        let route = runner.route("route").expect("route");
        let private = route
            .select_lane(&request("one", &["private"]))
            .expect("private");
        let both = route
            .select_lane(&request("two", &["private", "customer_data"]))
            .expect("both labels");
        assert!(Arc::ptr_eq(&private.lane, &both.lane));
        assert!(Arc::ptr_eq(&private.lane.algorithm, &both.lane.algorithm));
        assert!(!Arc::ptr_eq(&private.lane, &route.standard));

        let runner = Runner::from_toml(&source.replace("labels = []", "labels = ['private']"))
            .expect("standard models already approved");
        let route = runner.route("route").expect("route");
        let approved = route
            .select_lane(&request("one", &["private"]))
            .expect("approved");
        assert!(Arc::ptr_eq(&approved.lane, &route.standard));
    }

    // Queue spillover cannot introduce targets outside this request's approved pool.
    #[test]
    fn infra_spillover_stays_within_request_requirements() {
        let algorithm = "type = 'llm_classifier'\nclassifier_target = 'judge'\n\
            strong_target = 'capable'\nweak_target = 'fast'\nbase_threshold = 0.5";
        let source = format!(
            "{}\n[routes.test.infra]\nmax_signal_age_ms = 1000\n\
             min_waiting_requests = 4\nmax_capable_inflight = 2\n",
            config(algorithm)
        );
        let runner = Runner::from_toml(&source).expect("eligible queue policy");
        let route = runner.route("route").expect("route");
        let signals: InfraSignals = [
            ("fast", 8),
            ("capable", 0),
            ("approved/fast", 8),
            ("approved/capable", 0),
        ]
        .into_iter()
        .map(|(model, waiting)| {
            (
                model.into(),
                ModelInfraSignals {
                    load: Some(PoolLoad {
                        sampled_at: std::time::Instant::now(),
                        min_waiting: waiting,
                        max_waiting: waiting,
                        running: 0,
                    }),
                    ..Default::default()
                },
            )
        })
        .collect();

        let mut stale =
            RoutingOutcome::route_to("fast".into(), vec![], request("request", &["private"]));
        let error = route
            .apply_infra(&mut stale, &signals)
            .expect_err("stale decision");
        assert!(
            error.to_string().contains("requirements changed"),
            "{error}"
        );

        for (labels, initial, expected) in [
            (&[] as &[&str], "fast", "capable"),
            (&["private"], "approved/fast", "approved/capable"),
        ] {
            let mut outcome =
                RoutingOutcome::route_to(initial.into(), vec![], request("request", labels));
            assert!(
                route
                    .apply_infra(&mut outcome, &signals)
                    .expect("approved spillover")
            );
            assert_eq!(
                outcome.selected_model_id().expect("selected model"),
                expected
            );
        }
    }

    // Payload content and provider-body metadata cannot supply trusted requirements.
    #[tokio::test]
    async fn content_and_provider_metadata_do_not_classify_the_request() {
        let runner = Runner::from_toml(&config(PASSTHROUGH)).expect("eligible route");
        let route = runner.route("route").expect("route");
        let mut request = Request::default();
        request.llm_request.messages.push(Message::text(
            Role::User,
            "Synthetic confidential customer records.",
        ));
        request.raw_request = Some(json!({
            "metadata": {"switchyard.internal.required_target_labels": "[\"private\"]"},
            "restricted": true
        }));
        let ordinary = route
            .decide(request.clone())
            .await
            .expect("unmarked request");
        assert_eq!(ordinary.selected_model_id().expect("model"), "fast");
        require_target_labels(&mut request, ["private"]).expect("trusted requirements");
        let approved = route.decide(request).await.expect("approved request");
        assert_eq!(
            approved.selected_model_id().expect("model"),
            "approved/fast"
        );
    }

    #[tokio::test]
    async fn binary_compatibility_maps_to_private_without_clearing_existing_requirements() {
        let runner = Runner::from_toml(&config(PASSTHROUGH)).expect("eligible route");
        let route = runner.route("route").expect("route");
        let mut request = Request::default();
        require_restricted_targets(&mut request, false).expect("explicit empty");
        assert_eq!(
            route
                .decide(request.clone())
                .await
                .expect("ordinary")
                .selected_model_id()
                .expect("model"),
            "fast"
        );
        require_restricted_targets(&mut request, true).expect("private");
        require_restricted_targets(&mut request, false).expect("cannot clear private");
        let approved = route.decide(request).await.expect("approved request");
        assert_eq!(
            approved.selected_model_id().expect("model"),
            "approved/fast"
        );
    }
}
