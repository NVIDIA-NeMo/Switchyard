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

use std::{fmt, sync::Arc};

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
#[derive(Error)]
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
    #[error("context window exceeded for model {model}")]
    ContextWindowExceeded {
        /// Model whose context window was exceeded.
        model: String,
        /// Upstream error message.
        message: String,
    },

    /// The upstream returned a non-success HTTP response.
    #[error("upstream returned HTTP {status}")]
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

impl fmt::Debug for LlmClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { message } => formatter
                .debug_struct("InvalidRequest")
                .field("message", message)
                .finish(),
            Self::RequestTranslation(message) => formatter
                .debug_tuple("RequestTranslation")
                .field(message)
                .finish(),
            Self::RequestEncoding(message) => formatter
                .debug_tuple("RequestEncoding")
                .field(message)
                .finish(),
            Self::ResponseTranslation(message) => formatter
                .debug_tuple("ResponseTranslation")
                .field(message)
                .finish(),
            Self::Configuration { message } => formatter
                .debug_struct("Configuration")
                .field("message", message)
                .finish(),
            Self::Transport { source } => formatter
                .debug_struct("Transport")
                .field("source", source)
                .finish(),
            Self::Timeout { source } => formatter
                .debug_struct("Timeout")
                .field("source", source)
                .finish(),
            Self::ContextWindowExceeded { model, .. } => formatter
                .debug_struct("ContextWindowExceeded")
                .field("model", model)
                .field("message", &"[REDACTED]")
                .finish(),
            Self::UpstreamHttp { status, .. } => formatter
                .debug_struct("UpstreamHttp")
                .field("status", status)
                .field("body", &"[REDACTED]")
                .finish(),
            Self::InvalidResponse { source } => formatter
                .debug_struct("InvalidResponse")
                .field("source", source)
                .finish(),
            Self::Ffi { source } => formatter
                .debug_struct("Ffi")
                .field("source", source)
                .finish(),
            Self::General(message) => formatter.debug_tuple("General").field(message).finish(),
        }
    }
}

/// A decision/trace object produced by an algorithm.
///
/// Carried as a trait object (not a generic parameter) so a stream consumer can
/// inspect any algorithm's decision through this common interface without
/// knowing the concrete type. `as_any` is the escape hatch for a consumer that
/// *does* know the algo and wants to downcast to the concrete decision.
pub trait Decision: Send + Sync {
    /// The model this decision selected (e.g. the routed target's name).
    fn selected_model(&self) -> &str;
    /// Stable routing tier for the selected model, when the algorithm provides one.
    fn routing_tier(&self) -> Option<&str> {
        None
    }
    /// Whether this is the final call selected to serve the request.
    fn is_routed_call(&self) -> bool {
        true
    }
    /// A human-readable explanation of the decision, for logs and traces.
    fn reasoning(&self) -> Option<&str>;
    /// Downcast handle: a consumer that knows the algorithm can recover the
    /// concrete decision type via `as_any().downcast_ref::<ConcreteDecision>()`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A minimal [`Decision`] implementation for one-off calls that don't belong to a
/// named algorithm step — judge side calls, classifier side calls, etc.
pub struct SimpleDecision {
    /// Model or semantic target selected for the call.
    pub selected_model: String,
    /// Optional explanation recorded with the call.
    pub reasoning: Option<String>,
}

impl Decision for SimpleDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }

    fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    /// Serve the call, returning the model's response. Call the model named by
    /// [`decision.selected_model()`](Decision::selected_model) — the target the algorithm
    /// routed to — mapping it to whatever provider model id this client hits.
    /// `request.llm_request.model` is the agent's original name, carried through for
    /// reference, not a call target. `ctx` carries the request's cross-cutting state.
    async fn call(
        &self,
        ctx: Context,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, LlmClientError>;

    /// Whether this client can serve [`count_tokens`](Self::count_tokens) — i.e.
    /// it has an Anthropic upstream. The default is `false`.
    fn supports_count_tokens(&self) -> bool {
        false
    }

    /// Count the tokens `request` would use — a **direct passthrough**, not a
    /// routed call. Forwards `request` to this client's Anthropic
    /// `/v1/messages/count_tokens` endpoint (model restamped to the upstream
    /// target id) and returns the JSON verbatim. Token counting is a pre-flight
    /// estimate with no routing decision, so unlike [`call`](Self::call) it
    /// takes no [`Decision`]. The default errors; only an Anthropic-backed
    /// client overrides it.
    async fn count_tokens(&self, request: Request) -> Result<serde_json::Value, LlmClientError> {
        let _ = request;
        Err(LlmClientError::Configuration {
            message: "count_tokens is not supported by this client".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::LlmClientError;

    #[test]
    fn context_window_message_is_redacted_from_display_and_debug() {
        let secret = "provider context-window details";
        let error = LlmClientError::ContextWindowExceeded {
            model: "provider/model".to_string(),
            message: secret.to_string(),
        };

        assert!(!error.to_string().contains(secret));
        let debug = format!("{error:?}");
        assert!(!debug.contains(secret));
        assert_eq!(
            debug,
            "ContextWindowExceeded { model: \"provider/model\", message: \"[REDACTED]\" }"
        );
    }

    #[test]
    fn upstream_http_body_is_redacted_from_display_and_debug() {
        let secret = "provider error body";
        let error = LlmClientError::UpstreamHttp {
            status: 500,
            body: secret.to_string(),
        };

        assert!(!error.to_string().contains(secret));
        let debug = format!("{error:?}");
        assert!(!debug.contains(secret));
        assert_eq!(debug, "UpstreamHttp { status: 500, body: \"[REDACTED]\" }");
    }
}
