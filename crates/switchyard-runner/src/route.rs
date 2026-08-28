// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, CallModel, LibsyError, RoutingOutcome, drive};
use serde_json::Value;
use switchyard_llm_client::{ClientRouter, RunObserver, TranslatingLlmClient};
use switchyard_protocol::{LlmClientError, ModelId, Request, Response, WireFormat};
use thiserror::Error;

use crate::DecisionTarget;

/// Capabilities that one route advertises on `GET /v1/models`.
///
/// An unset capability is undeclared: it serializes as `null` in the OpenAI
/// `data` entry, and the Codex entry falls back to a safe default for it.
#[derive(Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub context_window: Option<u32>,
    pub tool_calling: Option<bool>,
    /// Whether the routed model takes reasoning controls. A serving surface cannot
    /// probe this, so a route opts in via config; undeclared routes advertise as
    /// non-reasoning to Codex (fail closed).
    pub reasoning: Option<bool>,
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

/// Exact upstream model used for Anthropic token counting.
#[derive(Clone)]
pub struct CountTokensTarget {
    pub model: ModelId,
    pub client: Arc<TranslatingLlmClient>,
}

/// Exact upstream model used for OpenAI Responses auxiliary operations.
#[derive(Clone)]
pub struct ResponsesTarget {
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
    #[error("route has no Anthropic target for token counting")]
    CountTokensUnsupported,
    #[error("route has no OpenAI Responses target")]
    ResponsesAuxiliaryUnsupported,
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
pub struct Route {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
    count_tokens_target: Option<CountTokensTarget>,
    responses_target: Option<ResponsesTarget>,
    decision_targets: Vec<DecisionTarget>,
}

/// The selected model and untouched response produced by a route execution.
pub struct RunOutput {
    pub selected_model: ModelId,
    pub response: Response,
}

impl Route {
    /// Creates a fully configured execution route.
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        count_tokens_target: Option<CountTokensTarget>,
        responses_target: Option<ResponsesTarget>,
        decision_targets: Vec<DecisionTarget>,
    ) -> Self {
        Self {
            algorithm,
            clients,
            caller_auth,
            capabilities,
            count_tokens_target,
            responses_target,
            decision_targets,
        }
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
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
        let (selected_model, response) = switchyard_llm_client::run(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            observer,
        )
        .await?;
        Ok(RunOutput {
            selected_model,
            response,
        })
    }

    /// Completes routing-time calls without serving the answer target.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        drive(Arc::clone(&self.algorithm), request, |call| {
            serve_decision_dependency(self.clients.clone(), call)
        })
        .await
        .map_err(Into::into)
    }

    /// Counts tokens using the configured Anthropic-capable target.
    pub async fn count_tokens(&self, request: Request) -> Result<Value, RunnerError> {
        let target = self
            .count_tokens_target
            .as_ref()
            .ok_or(RunnerError::CountTokensUnsupported)?;
        target
            .client
            .count_tokens(&target.model, request)
            .await
            .map_err(Into::into)
    }

    /// Counts tokens using the configured OpenAI Responses-capable target.
    pub async fn responses_input_tokens(&self, request: Request) -> Result<Value, RunnerError> {
        let target = self
            .responses_target
            .as_ref()
            .ok_or(RunnerError::ResponsesAuxiliaryUnsupported)?;
        target
            .client
            .responses_input_tokens(&target.model, request)
            .await
            .map_err(Into::into)
    }

    /// Compacts a request using the configured OpenAI Responses-capable target.
    pub async fn responses_compact(&self, request: Request) -> Result<Value, RunnerError> {
        let target = self
            .responses_target
            .as_ref()
            .ok_or(RunnerError::ResponsesAuxiliaryUnsupported)?;
        target
            .client
            .responses_compact(&target.model, request)
            .await
            .map_err(Into::into)
    }
}

async fn serve_decision_dependency(clients: ClientRouter, call: CallModel) -> libsy::Result<()> {
    let mut result = Err(LibsyError::NoTargets);
    for (index, model) in call.models.iter().enumerate() {
        // The driver stamps only the first candidate, so every fallback must replace it.
        let mut request = call.request.clone();
        request.llm_request.model = Some(model.to_string());
        let response = match clients.route(model) {
            Ok(client) => client.call(request).await,
            Err(source) => Err(source),
        };
        match response {
            Ok(response) => {
                result = Ok(response);
                break;
            }
            Err(source) => {
                let try_next = index + 1 < call.models.len() && eligible_routing_fallback(&source);
                result = Err(LibsyError::client_call(model.clone(), source));
                if !try_next {
                    break;
                }
            }
        }
    }
    call.respond(result)
}

/// Whether a routing-time candidate failure may fall through to the next model.
fn eligible_routing_fallback(error: &LlmClientError) -> bool {
    match error {
        LlmClientError::ContextWindowExceeded { .. }
        | LlmClientError::Transport { .. }
        | LlmClientError::Timeout { .. } => true,
        LlmClientError::UpstreamHttp { status, .. } => {
            matches!(
                *status,
                reqwest::StatusCode::FORBIDDEN
                    | reqwest::StatusCode::REQUEST_TIMEOUT
                    | reqwest::StatusCode::TOO_MANY_REQUESTS
            ) || status.is_server_error()
        }
        _ => false,
    }
}
