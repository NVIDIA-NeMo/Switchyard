// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::collections::HashMap;
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

/// A configured algorithm and the per-target clients its calls resolve through.
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

/// A configured route with optional target eligibility checks.
#[derive(Clone)]
pub struct Route {
    standard: Arc<ExecutionLane>,
    eligibility: Option<Arc<EligibilityExecution>>,
    targets: Arc<HashMap<ModelId, DecisionTarget>>,
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
        targets: Vec<DecisionTarget>,
        models: RuntimeModels,
    ) -> Self {
        let models = models.with_provider_models(
            targets
                .iter()
                .map(|target| (ModelId::from(target.target.as_str()), target.model.clone()))
                .collect(),
        );
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
            targets,
        )
    }

    pub(crate) fn from_lane(
        lane: ExecutionLane,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        targets: Vec<DecisionTarget>,
    ) -> Self {
        let targets = targets
            .into_iter()
            .map(|target| (ModelId::from(target.target.as_str()), target))
            .collect();
        Self {
            standard: Arc::new(lane),
            eligibility: None,
            targets: Arc::new(targets),
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
        self.eligibility = Some(Arc::new(EligibilityExecution { policy, lanes }));
        self
    }

    fn select_lane(&self, request: &Request) -> Result<Arc<ExecutionLane>, RunnerError> {
        if let Some(eligibility) = &self.eligibility {
            let required = eligibility.policy.admit(request)?;
            let lane = if required.required_labels().is_empty() {
                Arc::clone(&self.standard)
            } else {
                eligibility
                    .lanes
                    .select(required.required_labels(), &self.standard)?
            };
            tracing::debug!(
                switchyard.route = request.llm_request.model.as_deref().unwrap_or_default(),
                switchyard.algorithm = self.algorithm_name(),
                eligibility.required_label_count = required.required_labels().len(),
                "eligible targets selected"
            );
            return Ok(lane);
        }
        if has_target_requirements(request) {
            return Err(LlmClientError::InvalidRequest {
                message: "route does not accept target requirements".to_string(),
            }
            .into());
        }
        Ok(Arc::clone(&self.standard))
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

    /// Returns non-secret metadata for a configured LLM target, including judges.
    pub fn target(&self, selected: &ModelId) -> Option<&DecisionTarget> {
        self.targets.get(selected)
    }

    /// Resolves a selected model to this route's non-secret target metadata.
    pub(crate) fn decision_target(&self, model: &ModelId) -> Option<DecisionTarget> {
        self.target(model).cloned()
    }

    /// The provider model that serves `selected`, or `selected` itself when it names no
    /// target of this route.
    pub fn provider_model<'a>(&'a self, selected: &'a ModelId) -> &'a ModelId {
        self.targets
            .get(selected)
            .map(|target| &target.model)
            .unwrap_or(selected)
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
        let lane = self.select_lane(&request)?;
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
        let lane = self.select_lane(&request)?;
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
                let lane = self.select_lane(&outcome.request)?;
                if outcome.selected_model_ids.iter().any(|model| {
                    !lane
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
                    .apply(outcome, &lane.models, signals)
                    .map_err(Into::into)
            }
            None => Ok(false),
        }
    }

    /// Routes with a host-owned client for routing-time LLM calls, such as judges.
    ///
    /// The host owns transport, credentials, and target request settings. Configured
    /// HTTP clients, target prompts, and decision clients are not used on this path.
    /// Algorithm state is retained for each target setup. No post-routing completion
    /// is dispatched. Target eligibility still applies to the host-owned path.
    pub async fn decide_with_client(
        &self,
        request: Request,
        client: Arc<dyn RoutedLlmClient>,
    ) -> Result<RoutingOutcome, RunnerError> {
        let lane = self.select_lane(&request)?;
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
        let lane = self.select_lane(&request)?;
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
    use crate::{Runner, require_target_labels};

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

    #[test]
    fn resolved_bindings_reuse_the_algorithm_and_standard_setup() {
        let source = config(PASSTHROUGH);
        let runner = Runner::from_toml(&source).expect("eligible route");
        let route = runner.route("route").expect("route");
        let cloned = route.clone();
        let private = route
            .select_lane(&request("one", &["private"]))
            .expect("private");
        let both = route
            .select_lane(&request("two", &["private", "customer_data"]))
            .expect("both labels");
        assert!(Arc::ptr_eq(&private, &both));
        assert!(Arc::ptr_eq(&private.algorithm, &both.algorithm));
        assert!(!Arc::ptr_eq(&private, &route.standard));
        assert_eq!(
            private.models.provider_model(&"approved_fast".into()),
            "approved/fast"
        );
        let shared = cloned
            .select_lane(&request("three", &["private"]))
            .expect("cloned route");
        assert!(Arc::ptr_eq(&private, &shared));

        let runner = Runner::from_toml(&source.replace("labels = []", "labels = ['private']"))
            .expect("standard models already approved");
        let route = runner.route("route").expect("route");
        let approved = route
            .select_lane(&request("one", &["private"]))
            .expect("approved");
        assert!(Arc::ptr_eq(&approved, &route.standard));
    }

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
            ("approved_fast", 8),
            ("approved_capable", 0),
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
            (&["private"], "approved_fast", "approved_capable"),
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
            "approved_fast"
        );
    }
}
