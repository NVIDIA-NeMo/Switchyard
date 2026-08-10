// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routed-call server trait and the routing decision it carries.
//!
//! [`RoutedLlmClient`] is the one piece of I/O the protocol does not own: a host
//! implements it to actually perform a model call. [`Decision`] is the routing
//! decision that produced the call, carried alongside so the client and any
//! observer can see which model was chosen and why. Both live here — rather than
//! in libsy's orchestration crate — so a client crate that depends only on the
//! protocol can serve routed calls without pulling in the orchestrator.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::{Context, Request, Response};

/// A boxed client-specific error preserved as the source of a routed call failure.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Failures a routed LLM client can surface to its caller.
///
/// The variants classify failures that routing hosts commonly need to handle,
/// while boxed sources preserve implementation-specific detail. `General` is the
/// escape hatch for failures that do not fit a shared category.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// The request cannot be served as supplied.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Human-readable request validation failure.
        message: String,
    },

    /// Decoding the inbound request failed in the translation engine.
    #[error("request translation failed: {0}")]
    RequestTranslation(String),

    /// Encoding the request for the upstream failed in the translation engine.
    #[error("outbound request encoding failed: {0}")]
    RequestEncoding(String),

    /// Decoding or encoding the response failed in the translation engine.
    #[error("response translation failed: {0}")]
    ResponseTranslation(String),

    /// The client is not configured to serve the selected target.
    #[error("client configuration error: {message}")]
    Configuration {
        /// Human-readable configuration failure.
        message: String,
    },

    /// The upstream could not be reached or the request could not be sent.
    #[error("upstream transport error: {source}")]
    Transport {
        /// Client-specific transport failure.
        #[source]
        source: BoxError,
    },

    /// The upstream request exceeded its timeout.
    #[error("upstream request timed out: {source}")]
    Timeout {
        /// Client-specific timeout failure.
        #[source]
        source: BoxError,
    },

    /// The upstream rejected the request because it exceeds the model's context window.
    #[error("context window exceeded for model {model}: {message}")]
    ContextWindowExceeded {
        /// Model whose context window was exceeded.
        model: String,
        /// Upstream error message.
        message: String,
    },

    /// The upstream returned a non-success HTTP response.
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamHttp {
        /// Upstream HTTP status code.
        status: u16,
        /// Raw upstream error body.
        body: String,
    },

    /// The upstream returned a response the client could not decode.
    #[error("invalid upstream response: {source}")]
    InvalidResponse {
        /// Client-specific decoding or validation failure.
        #[source]
        source: BoxError,
    },

    /// A call across a foreign-function boundary (e.g. a Python-implemented client)
    /// failed. The boxed source is the foreign error itself.
    #[error("foreign function interface error: {source}")]
    Ffi {
        /// Foreign-language failure, preserved verbatim.
        #[source]
        source: BoxError,
    },

    /// A string message. Useful in testing, but prefer adding variants over using this.
    #[error("{0}")]
    General(String),
}

/// Why routing replaced a selected target with another eligible target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingFallbackReason {
    /// The selected target rejected the request because its context window was too small.
    ContextWindow,
    /// The selected target was unavailable after its client retries finished.
    Unavailable,
}

impl RoutingFallbackReason {
    /// Stable value embedded in routing reasoning.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextWindow => "context_window",
            Self::Unavailable => "unavailable",
        }
    }
}

/// A routing choice produced by an algorithm.
#[derive(Clone, Debug)]
pub struct Decision {
    /// The model identifier selected for the call.
    selected_model_id: String,
    /// Why, for logs and traces.
    reasoning: Option<String>,
    /// True for an answer-generating call. False for classifier and judge calls.
    is_answer_call: bool,
}

impl Decision {
    /// Creates a decision and records whether its call produces the answer.
    pub fn new(
        selected_model_id: impl Into<String>,
        reasoning: Option<String>,
        is_answer_call: bool,
    ) -> Self {
        Self {
            selected_model_id: selected_model_id.into(),
            reasoning,
            is_answer_call,
        }
    }

    /// The model identifier selected for the call.
    pub fn selected_model_id(&self) -> &str {
        self.selected_model_id.as_str()
    }

    /// Why this decision was made.
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Whether this call generates an answer rather than a routing verdict.
    pub fn is_answer_call(&self) -> bool {
        self.is_answer_call
    }
}

/// Performs the actual model call for a target. This is the one piece of I/O the
/// library does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as a routed request's `default_client`.
///
/// # Concurrency
///
/// A client may be shared by many targets and concurrent algorithm runs. Calls may
/// overlap, so implementations must synchronize mutable state internally and should
/// not serialize requests unless their transport requires it.
#[async_trait]
pub trait RoutedLlmClient: Send + Sync {
    /// Serve the model identified by
    /// [`decision.selected_model_id()`](Decision::selected_model_id), resolving it to the
    /// provider model this client calls.
    /// `request.llm_request.model` is the agent's original name, carried through for
    /// reference, not a call target. `ctx` carries the request's cross-cutting state.
    async fn call(
        &self,
        ctx: Context,
        request: Request,
        decision: Arc<Decision>,
    ) -> Result<Response, LlmClientError>;
}
