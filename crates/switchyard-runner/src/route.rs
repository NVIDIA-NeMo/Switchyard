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
use crate::privacy::{LocalPolicy, PrivacyDecision, has_local_requirement};

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

struct LocalExecution {
    policy: LocalPolicy,
    lane: ExecutionLane,
}

struct LaneSelection<'a> {
    lane: &'a ExecutionLane,
    privacy: Option<PrivacyDecision>,
}

/// One or more execution lanes behind a configured route.
pub struct Route {
    standard: ExecutionLane,
    local: Option<LocalExecution>,
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
            standard: lane,
            local: None,
            decision_targets,
            caller_auth,
            capabilities,
            infra: None,
        }
    }

    pub(crate) fn with_local(mut self, policy: LocalPolicy, lane: ExecutionLane) -> Self {
        self.local = Some(LocalExecution { policy, lane });
        self
    }

    fn select_lane(&self, request: &Request) -> Result<LaneSelection<'_>, RunnerError> {
        if let Some(local) = &self.local {
            let local_only = local.policy.admit(request)?;
            let privacy = Some(PrivacyDecision::from_local_requirement(
                request, local_only,
            )?);
            let lane = if local_only {
                &local.lane
            } else {
                &self.standard
            };
            tracing::debug!(
                switchyard.route = request.llm_request.model.as_deref().unwrap_or_default(),
                switchyard.algorithm = self.algorithm_name(),
                local_only,
                privacy.lane = privacy.as_ref().map(PrivacyDecision::lane),
                privacy.source = privacy.as_ref().map(PrivacyDecision::source),
                privacy.reason_code = privacy.as_ref().map(PrivacyDecision::reason_code),
                "execution lane selected"
            );
            return Ok(LaneSelection { lane, privacy });
        }
        if has_local_requirement(request) {
            return Err(LlmClientError::InvalidRequest {
                message: "route does not accept a local-only signal".to_string(),
            }
            .into());
        }
        Ok(LaneSelection {
            lane: &self.standard,
            privacy: None,
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

    /// Executes the route and reports standard or local-only admission before dispatch.
    pub async fn execute_with_privacy_observer(
        &self,
        request: Request,
        observer: Option<RunObserver>,
        on_privacy_decision: impl FnOnce(&PrivacyDecision) + Send,
    ) -> Result<RunOutput, RunnerError> {
        let selection = self.select_lane(&request)?;
        if let Some(decision) = &selection.privacy {
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
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use libsy::{Driver, Passthrough};
    use reqwest::StatusCode;
    use switchyard_protocol::{
        Category, ContentBlock, DecisionRequest, DecisionResponse, LlmResponse, Message, Role,
        RoutedDecisionClient, RoutedLlmClient, Usage, text_response,
    };

    use super::*;
    use crate::require_local_targets;

    impl Route {
        fn with_private_targets(self, restricted: ExecutionLane) -> Self {
            self.with_local(LocalPolicy::new(false, false), restricted)
        }
    }

    enum RoutingCall {
        Llm,
        Decision,
    }

    struct CallThenRoute(RoutingCall);

    #[async_trait::async_trait]
    impl Algorithm for CallThenRoute {
        fn name(&self) -> &str {
            "call_then_route"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> libsy::Result<RoutingOutcome> {
            match &self.0 {
                RoutingCall::Llm => {
                    driver
                        .call_model(
                            request.clone(),
                            driver.models_for(&Category::Judge).to_vec(),
                        )
                        .await?;
                }
                RoutingCall::Decision => {
                    let judge = driver.first_model_for(&Category::Judge)?.clone();
                    driver
                        .call_decision(
                            DecisionRequest {
                                model: None,
                                context: Value::Null,
                                questions: BTreeMap::new(),
                            },
                            judge,
                        )
                        .await?;
                }
            }
            let (selected, fallbacks) = driver
                .models_for(&Category::Any)
                .split_first()
                .ok_or(LibsyError::NoTargets)?;
            Ok(RoutingOutcome::route_to(
                selected.clone(),
                fallbacks.to_vec(),
                request,
            ))
        }
    }

    struct RecordingClient {
        calls: Arc<Mutex<Vec<ModelId>>>,
    }

    struct RecordingDecisionClient {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for RecordingClient {
        async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
            let model = request
                .llm_request
                .model
                .map(ModelId::from)
                .expect("driver should stamp a model");
            self.calls.lock().expect("calls lock").push(model.clone());
            if model == "restricted-primary" {
                return Err(LlmClientError::UpstreamHttp {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    body: "unavailable".to_string(),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(Some(model.to_string()), "ok")),
                metadata: None,
                upstream_headers: Default::default(),
            })
        }
    }

    #[async_trait::async_trait]
    impl RoutedDecisionClient for RecordingDecisionClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(DecisionResponse {
                id: None,
                model: None,
                answers: BTreeMap::new(),
                usage: Usage::default(),
            })
        }
    }

    fn lane(
        algorithm: Arc<dyn Algorithm>,
        client: Arc<dyn RoutedLlmClient>,
        models: RuntimeModels,
        auxiliary_model: &str,
    ) -> ExecutionLane {
        let routed_models = models
            .models_for(&Category::Any)
            .iter()
            .chain(models.models_for(&Category::Judge))
            .cloned()
            .map(|model| (model, Arc::clone(&client)))
            .collect();
        let auxiliary_client =
            Arc::new(TranslatingLlmClient::new(&[]).expect("empty test client should be valid"));
        ExecutionLane::new(
            algorithm,
            ClientRouter::new(routed_models),
            Some(AuxiliaryTarget {
                model: auxiliary_model.into(),
                client: auxiliary_client,
            }),
            None,
            models,
        )
    }

    fn restricted_request() -> Request {
        let mut request = Request::default();
        require_local_targets(&mut request, true).expect("trusted restriction");
        request
    }

    // Queue spillover must use the admitted lane, even when both lanes share a model.
    #[test]
    fn infra_policy_stays_in_the_selected_lane() {
        let build_lane = |capable: &str| {
            ExecutionLane::new(
                Arc::new(Passthrough),
                ClientRouter::new(HashMap::new()),
                None,
                None,
                RuntimeModels::new(HashMap::from([
                    (Category::Efficient, vec!["local-fast".into()]),
                    (Category::Capable, vec![capable.into()]),
                ])),
            )
        };
        let route = Route::from_lane(
            build_lane("cloud-capable"),
            None,
            ModelCapabilities::default(),
            Vec::new(),
        )
        .with_local(LocalPolicy::new(false, true), build_lane("local-capable"))
        .with_infra(Some(libsy::InfraPolicy {
            max_signal_age_ms: 1_000,
            min_waiting_requests: 4,
            max_capable_inflight: 2,
        }));
        let signals = [
            ("local-fast", 8),
            ("local-capable", 0),
            ("cloud-capable", 0),
        ]
        .into_iter()
        .map(|(model, waiting)| {
            (
                model.into(),
                switchyard_protocol::ModelInfraSignals {
                    load: Some(switchyard_protocol::PoolLoad {
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
        for (task, local_only, expected) in [
            ("restricted", true, "local-capable"),
            ("restricted", false, "local-capable"),
            ("ordinary", false, "cloud-capable"),
        ] {
            let mut request = Request::default();
            request.metadata.get_or_insert_default().task_id = Some(task.into());
            require_local_targets(&mut request, local_only).expect("trusted signal");
            let mut outcome = RoutingOutcome::route_to("local-fast".into(), vec![], request);
            assert!(
                route
                    .apply_infra(&mut outcome, &signals)
                    .expect("spillover")
            );
            assert_eq!(outcome.selected_model_id().expect("model"), expected);
        }
    }

    fn decision_target(target: &str, model: &ModelId) -> DecisionTarget {
        DecisionTarget {
            target: target.to_string(),
            model: model.clone(),
            format: WireFormat::OpenAiChat,
            base_url: format!("https://{target}.example/v1"),
            extra_body: BTreeMap::new(),
        }
    }

    // Rejected content must not erase authoritative requirements supplied with that turn.
    #[tokio::test]
    async fn rejected_turn_retains_requirements() {
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let build_lane = |name: &str| {
            lane(
                Arc::new(Passthrough),
                Arc::clone(&client),
                RuntimeModels::new(HashMap::from([(Category::Any, vec![name.into()])])),
                name,
            )
        };
        let route = Route::from_lane(
            build_lane("standard"),
            None,
            ModelCapabilities::default(),
            Vec::new(),
        )
        .with_local(LocalPolicy::new(false, true), build_lane("local"));
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some("task".into());
        require_local_targets(&mut request, false).expect("trusted signal");
        let outcome = route
            .decide(request.clone())
            .await
            .expect("normal decision");
        assert_eq!(outcome.selected_model_id().expect("model"), "standard");

        require_local_targets(&mut request, true).expect("trusted restriction");
        request.llm_request.extensions.fields.insert(
            "previous_response_id".into(),
            Value::String("hidden".into()),
        );
        assert!(route.decide(request).await.is_err());
        let mut next = Request::default();
        next.metadata.get_or_insert_default().task_id = Some("task".into());
        let outcome = route.decide(next).await.expect("retained decision");
        assert_eq!(outcome.selected_model_id().expect("model"), "local");
    }

    #[tokio::test]
    async fn requirements_cover_judges_fallbacks_and_auxiliary_calls() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::clone(&calls),
        });
        let standard = lane(
            Arc::new(Passthrough),
            Arc::clone(&client),
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["standard-primary".into()],
            )])),
            "standard-auxiliary",
        );
        let restricted = lane(
            Arc::new(CallThenRoute(RoutingCall::Llm)),
            client,
            RuntimeModels::new(HashMap::from([
                (Category::Judge, vec!["restricted-judge".into()]),
                (
                    Category::Any,
                    vec!["restricted-primary".into(), "restricted-fallback".into()],
                ),
            ])),
            "restricted-auxiliary",
        );
        let route = Route::from_lane(standard, None, ModelCapabilities::default(), Vec::new());
        let route = route.with_private_targets(restricted);
        let request = restricted_request;

        let mut observed = None;
        let output = route
            .execute_with_privacy_observer(request(), None, |decision| {
                observed = Some((decision.lane(), decision.source(), decision.reason_code()));
            })
            .await
            .expect("restricted fallback should answer");
        assert_eq!(
            observed,
            Some(("restricted", "external_signal", "restricted"))
        );
        assert_eq!(output.selected_model, "restricted-primary");
        assert_eq!(
            output.response.served_model().map(ModelId::as_str),
            Some("restricted-fallback")
        );
        assert_eq!(
            *calls.lock().expect("calls lock"),
            [
                ModelId::from("restricted-judge"),
                ModelId::from("restricted-primary"),
                ModelId::from("restricted-fallback"),
            ]
        );

        let hosted_calls = Arc::new(Mutex::new(Vec::new()));
        let hosted_client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::clone(&hosted_calls),
        });
        let standard = route
            .decide_with_client(Request::default(), Arc::clone(&hosted_client))
            .await
            .expect("unmarked hosted routing should use the standard lane");
        assert_eq!(
            standard.selected_model_id().expect("selected model"),
            "standard-primary"
        );
        let restricted = route
            .decide_with_client(request(), hosted_client)
            .await
            .expect("marked hosted routing should use the restricted lane");
        assert_eq!(
            restricted.selected_model_id().expect("selected model"),
            "restricted-primary"
        );
        assert_eq!(
            *hosted_calls.lock().expect("calls lock"),
            [ModelId::from("restricted-judge")]
        );
        assert_eq!(calls.lock().expect("calls lock").len(), 3);

        let error = route
            .call_auxiliary(request(), AuxiliaryOperation::AnthropicCountTokens)
            .await
            .expect_err("empty auxiliary client should reject the model");
        assert!(
            error.to_string().contains("restricted-auxiliary"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn restricted_typed_decision_resolves_configured_target_metadata() {
        let answer: ModelId = "restricted-answer".into();
        let llm: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let standard = lane(
            Arc::new(Passthrough),
            Arc::clone(&llm),
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["standard-answer".into(), answer.clone()],
            )])),
            "standard-auxiliary",
        );

        let judge: ModelId = "restricted-decision".into();
        let decision_calls = Arc::new(AtomicUsize::new(0));
        let restricted = ExecutionLane::new(
            Arc::new(CallThenRoute(RoutingCall::Decision)),
            ClientRouter::single_with_decision_clients(
                llm,
                HashMap::from([(
                    judge.clone(),
                    Arc::new(RecordingDecisionClient {
                        calls: Arc::clone(&decision_calls),
                    }) as Arc<dyn RoutedDecisionClient>,
                )]),
            ),
            None,
            None,
            RuntimeModels::new(HashMap::from([
                (Category::Judge, vec![judge]),
                (Category::Any, vec![answer]),
            ])),
        );
        let route = Route::from_lane(
            standard,
            None,
            ModelCapabilities::default(),
            vec![
                decision_target("standard", &ModelId::from("standard-answer")),
                decision_target("restricted", &ModelId::from("restricted-answer")),
            ],
        )
        .with_private_targets(restricted);
        let route_id: ModelId = "switchyard/private".into();
        let runner = crate::Runner::new(vec![(route_id.clone(), route)]);

        let outcome = runner
            .route(route_id.as_str())
            .expect("route")
            .decide(restricted_request())
            .await
            .expect("restricted decision");
        assert_eq!(decision_calls.load(Ordering::Relaxed), 1);
        let description = runner
            .describe_decision(&route_id, &outcome)
            .expect("restricted target metadata");
        assert_eq!(description.selected.target, "restricted");
        assert_eq!(
            description.selected.base_url,
            "https://restricted.example/v1"
        );
    }

    // Content inspection belongs to the application; routing follows only its requirements.
    #[tokio::test]
    async fn request_content_does_not_classify_itself() {
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let build_lane = |name: &str| {
            lane(
                Arc::new(Passthrough),
                Arc::clone(&client),
                RuntimeModels::new(HashMap::from([(Category::Any, vec![name.into()])])),
                name,
            )
        };
        let route = Route::from_lane(
            build_lane("standard"),
            None,
            ModelCapabilities::default(),
            Vec::new(),
        )
        .with_private_targets(build_lane("restricted"));
        let mut request = Request::default();
        request.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: Value::Null,
            }],
        });
        let outcome = route
            .decide(request.clone())
            .await
            .expect("unmarked content");
        assert_eq!(
            outcome.selected_model_id().expect("selected model"),
            "standard"
        );

        require_local_targets(&mut request, true).expect("trusted restriction");
        let outcome = route.decide(request).await.expect("trusted restriction");
        assert_eq!(
            outcome.selected_model_id().expect("selected model"),
            "restricted"
        );
    }
}
