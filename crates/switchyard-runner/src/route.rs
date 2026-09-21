// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome};
use serde_json::Value;
use switchyard_llm_client::{AuxiliaryOperation, ClientRouter, RunObserver, TranslatingLlmClient};
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
    /// Whether the routed model accepts image input. Declared per route for the same
    /// reason as `reasoning`, and failing closed matters more here: a route may
    /// resolve to a target with no vision at all.
    ///
    /// This is not cosmetic metadata. Codex reads `input_modalities` from the model
    /// card and, when it reads text-only, replaces an attached image with the literal
    /// text `image content omitted because you do not support image input` *before
    /// sending*. An undeclared vision-capable route therefore loses the image in the
    /// client, and the proxy never receives one to forward.
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
pub struct Route {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
    anthropic_auxiliary_target: Option<AuxiliaryTarget>,
    responses_auxiliary_target: Option<AuxiliaryTarget>,
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
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
    ) -> Self {
        Self {
            algorithm,
            clients,
            caller_auth,
            capabilities,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
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

    /// Completes routing-time calls without serving a post-routing completion.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        switchyard_llm_client::decide(Arc::clone(&self.algorithm), self.clients.clone(), request)
            .await
            .map_err(Into::into)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let target = match operation {
            AuxiliaryOperation::AnthropicCountTokens => &self.anthropic_auxiliary_target,
            AuxiliaryOperation::ResponsesInputTokens | AuxiliaryOperation::ResponsesCompact => {
                &self.responses_auxiliary_target
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
