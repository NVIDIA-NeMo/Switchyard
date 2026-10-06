// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured route execution and the clients that serve its targets.

use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome, RuntimeModels};
use serde_json::Value;
use switchyard_llm_client::{AuxiliaryOperation, ClientRouter, RunObserver, TranslatingLlmClient};
use switchyard_protocol::{LlmClientError, ModelId, Request, Response, WireFormat};
use thiserror::Error;

use crate::DecisionTarget;
use crate::privacy::{
    PrivacyLane, PrivacyPolicy, external_signal_not_accepted, has_external_restriction,
    record_selected_lane, selected_lane, validate_mixed_request,
};

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
    decision_targets: Vec<DecisionTarget>,
    models: Arc<RuntimeModels>,
}

impl ExecutionLane {
    pub(crate) fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
        models: RuntimeModels,
    ) -> Self {
        Self {
            algorithm,
            clients,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
            decision_targets,
            models: Arc::new(models),
        }
    }
}

struct MixedTrafficPrivacy {
    policy: PrivacyPolicy,
    restricted: ExecutionLane,
}

/// One or more execution lanes behind a configured route.
pub struct Route {
    lane: ExecutionLane,
    privacy: Option<MixedTrafficPrivacy>,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
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
                decision_targets,
                models,
            ),
            caller_auth,
            capabilities,
        )
    }

    pub(crate) fn from_lane(
        lane: ExecutionLane,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            lane,
            privacy: None,
            caller_auth,
            capabilities,
        }
    }

    pub(crate) fn with_privacy(mut self, policy: PrivacyPolicy, restricted: ExecutionLane) -> Self {
        self.privacy = Some(MixedTrafficPrivacy { policy, restricted });
        self
    }

    async fn select_lane(
        &self,
        request: &Request,
    ) -> Result<(&ExecutionLane, Option<PrivacyLane>), RunnerError> {
        let Some(privacy) = &self.privacy else {
            if has_external_restriction(request) {
                return Err(external_signal_not_accepted().into());
            }
            return Ok((&self.lane, None));
        };
        validate_mixed_request(request)?;
        let decision = privacy.policy.decide(request).await?;
        tracing::info!(
            privacy.lane = decision.lane.as_str(),
            privacy.source = decision.source.as_str(),
            privacy.reason_code = decision.reason_code,
            privacy.clear_score = decision.clear_score,
            privacy.clear_threshold = decision.clear_threshold,
            "privacy lane selected"
        );
        let lane = match decision.lane {
            PrivacyLane::Standard => &self.lane,
            PrivacyLane::Restricted => &privacy.restricted,
        };
        Ok((lane, Some(decision.lane)))
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.lane.algorithm.name()
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
    pub(crate) fn decision_target(
        &self,
        model: &ModelId,
        request: &Request,
    ) -> Option<DecisionTarget> {
        let targets = match (&self.privacy, selected_lane(request)) {
            (None, _) | (Some(_), Some(PrivacyLane::Standard)) => &self.lane.decision_targets,
            (Some(privacy), Some(PrivacyLane::Restricted)) => &privacy.restricted.decision_targets,
            (Some(_), None) => return None,
        };
        targets
            .iter()
            .find(|target| target.model == *model)
            .cloned()
    }

    /// Returns the models grouped for the standard execution lane.
    pub fn models(&self) -> &RuntimeModels {
        &self.lane.models
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
        let (lane, _) = self.select_lane(&request).await?;
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
        let (lane, selected_privacy_lane) = self.select_lane(&request).await?;
        let mut outcome = switchyard_llm_client::decide(
            Arc::clone(&lane.algorithm),
            lane.clients.clone(),
            request,
            Arc::clone(&lane.models),
        )
        .await
        .map_err(RunnerError::from)?;
        if let Some(selected_privacy_lane) = selected_privacy_lane {
            record_selected_lane(&mut outcome.request, selected_privacy_lane);
        }
        Ok(outcome)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let (lane, _) = self.select_lane(&request).await?;
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
        Category, DecisionAnswer, DecisionRequest, DecisionResponse, DecisionValue, LlmResponse,
        Message, Probability, Role, RoutedDecisionClient, RoutedLlmClient, Usage, text_response,
    };

    use super::*;
    use crate::privacy::{
        DeterministicDetector, SemanticPrivacyClassifier, mark_privacy_restricted,
    };

    struct JudgeThenRoute;

    #[async_trait::async_trait]
    impl Algorithm for JudgeThenRoute {
        fn name(&self) -> &str {
            "judge_then_route"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> libsy::Result<RoutingOutcome> {
            driver
                .call_model(
                    request.clone(),
                    driver.models_for(&Category::Judge).to_vec(),
                )
                .await?;
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

    struct VerdictClient {
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
    impl RoutedDecisionClient for VerdictClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut probabilities = [
                "no_sensitive_content",
                "personal_data",
                "credentials",
                "confidential_data",
                "regulated_data",
                "uncertain",
            ]
            .into_iter()
            .map(|id| (id.into(), Probability(0.0)))
            .collect::<BTreeMap<_, _>>();
            probabilities.insert("no_sensitive_content".into(), Probability(0.95));
            probabilities.insert("uncertain".into(), Probability(0.05));
            Ok(DecisionResponse {
                id: None,
                model: None,
                answers: BTreeMap::from([(
                    "privacy".into(),
                    DecisionAnswer {
                        value: DecisionValue::Choice {
                            selected: "no_sensitive_content".into(),
                            probabilities: Some(probabilities),
                        },
                        provider_confidence: None,
                    },
                )]),
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
            Vec::new(),
            models,
        )
    }

    fn restricted_request() -> Request {
        let mut request = Request::default();
        mark_privacy_restricted(&mut request);
        request
    }

    #[tokio::test]
    async fn privacy_restricted_lane_covers_judge_fallbacks_and_auxiliary_calls() {
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
            Arc::new(JudgeThenRoute),
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
        let route = Route::from_lane(standard, None, ModelCapabilities::default()).with_privacy(
            PrivacyPolicy::new(true, None, None)
                .expect("empty detector configuration should compile"),
            restricted,
        );

        let output = route
            .execute(restricted_request(), None)
            .await
            .expect("restricted fallback should answer");
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

        let error = route
            .call_auxiliary(
                restricted_request(),
                AuxiliaryOperation::AnthropicCountTokens,
            )
            .await
            .expect_err("empty auxiliary client should reject the model");
        assert!(
            error.to_string().contains("restricted-auxiliary"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn semantic_privacy_runs_after_deterministic_checks() {
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
            Arc::new(Passthrough),
            client,
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["restricted-primary".into()],
            )])),
            "restricted-auxiliary",
        );
        let classifier_calls = Arc::new(AtomicUsize::new(0));
        let classifier = SemanticPrivacyClassifier::new(
            "privacy-judge".into(),
            Arc::new(VerdictClient {
                calls: Arc::clone(&classifier_calls),
            }),
            0.9,
        );
        let route = Route::from_lane(standard, None, ModelCapabilities::default()).with_privacy(
            PrivacyPolicy::new(
                true,
                Some(vec![DeterministicDetector::BearerToken]),
                Some(classifier),
            )
            .expect("static detector configuration should compile"),
            restricted,
        );

        assert_eq!(
            route
                .decide(Request::default())
                .await
                .expect("classifier verdict should select a lane")
                .selected_model_id()
                .expect("selected model"),
            "standard-primary"
        );

        let mut sensitive = Request::default();
        sensitive.llm_request.messages.push(Message::text(
            Role::User,
            "Authorization: Bearer abcdefghijklmnop",
        ));
        assert_eq!(
            route
                .decide(sensitive)
                .await
                .expect("deterministic restriction")
                .selected_model_id()
                .expect("selected model"),
            "restricted-primary"
        );
        assert_eq!(classifier_calls.load(Ordering::Relaxed), 1);
    }
}
