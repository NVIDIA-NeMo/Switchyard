// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe, structured summaries of route-execution failures for telemetry.

use libsy::LibsyError;
use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, ModelId};

use crate::RunnerError;

/// Stable class of a terminal route-execution failure.
///
/// This deliberately carries no provider message, response body, or source
/// error. It is suitable for logs and telemetry, not client-facing rendering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq)]
#[strum(serialize_all = "snake_case")]
pub enum RouteErrorKind {
    /// The upstream returned a non-success HTTP response.
    UpstreamHttp,
    /// The selected target rejected the request because its context window was exceeded.
    ContextWindowExceeded,
    /// The upstream request timed out.
    Timeout,
    /// The upstream could not be reached or the request could not be sent.
    Transport,
    /// The upstream response could not be decoded or validated.
    InvalidResponse,
    /// Decoding the inbound request failed in translation.
    RequestTranslation,
    /// Encoding the request for the upstream failed in translation.
    RequestEncoding,
    /// Decoding or encoding the response failed in translation.
    ResponseTranslation,
    /// The request cannot be served as supplied.
    InvalidRequest,
    /// The configured route or client cannot serve the request.
    Configuration,
    /// The routing algorithm or driver could not produce an outcome.
    Algorithm,
    /// A failure without a safe, more specific kind.
    Other,
}

impl RouteErrorKind {
    /// Returns the stable telemetry value for this kind.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

/// When a terminal failure occurred relative to response delivery.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteFailurePhase {
    /// The route failed before returning a response to its caller.
    BeforeResponse,
    /// A previously returned streaming response failed while it was consumed.
    DuringStream,
}

impl RouteFailurePhase {
    /// Returns the stable telemetry value for this phase.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BeforeResponse => "before_response",
            Self::DuringStream => "during_stream",
        }
    }
}

/// Safe, structured terminal-failure data for routing telemetry.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteFailureSummary {
    /// Stable failure classification.
    pub kind: RouteErrorKind,
    /// Whether the error preceded response delivery or occurred while streaming.
    pub phase: RouteFailurePhase,
    /// Upstream HTTP status, when directly available.
    pub upstream_status: Option<u16>,
    /// Selected target that failed, when the runner knows it.
    pub target: Option<ModelId>,
}

impl RunnerError {
    /// Returns a safe telemetry summary for a failure before response delivery.
    pub fn execution_failure_summary(&self) -> RouteFailureSummary {
        match self {
            Self::Algorithm(LibsyError::ClientCall { target, source }) => {
                client_failure_summary(source, RouteFailurePhase::BeforeResponse, Some(target))
            }
            Self::Client(source) => {
                client_failure_summary(source, RouteFailurePhase::BeforeResponse, None)
            }
            Self::Configuration { .. } => summary(
                RouteErrorKind::Configuration,
                RouteFailurePhase::BeforeResponse,
                None,
                None,
            ),
            Self::UnknownRouteModel(_)
            | Self::IncompatibleCallerFormat(_)
            | Self::CountTokensUnsupported => summary(
                RouteErrorKind::InvalidRequest,
                RouteFailurePhase::BeforeResponse,
                None,
                None,
            ),
            Self::Algorithm(_) => summary(
                RouteErrorKind::Algorithm,
                RouteFailurePhase::BeforeResponse,
                None,
                None,
            ),
        }
    }
}

/// Returns a safe telemetry summary for an error yielded by an active response stream.
///
/// `served_model` should be the target recorded on the response before its stream was returned.
pub fn stream_failure_summary(
    error: &LlmClientError,
    served_model: Option<&ModelId>,
) -> RouteFailureSummary {
    client_failure_summary(error, RouteFailurePhase::DuringStream, served_model)
}

fn client_failure_summary(
    error: &LlmClientError,
    phase: RouteFailurePhase,
    target: Option<&ModelId>,
) -> RouteFailureSummary {
    let target = target.or(match error {
        // Context-window errors carry the resolved target model when a caller has not
        // already supplied the route's selected or served target.
        LlmClientError::ContextWindowExceeded { model, .. } => Some(model),
        _ => None,
    });
    let (kind, upstream_status) = match error {
        LlmClientError::UpstreamHttp { status, .. } => {
            (RouteErrorKind::UpstreamHttp, Some(status.as_u16()))
        }
        LlmClientError::ContextWindowExceeded { .. } => {
            (RouteErrorKind::ContextWindowExceeded, None)
        }
        LlmClientError::Timeout { .. } => (RouteErrorKind::Timeout, None),
        LlmClientError::Transport { .. } => (RouteErrorKind::Transport, None),
        LlmClientError::InvalidResponse { .. } => (RouteErrorKind::InvalidResponse, None),
        LlmClientError::RequestTranslation(_) => (RouteErrorKind::RequestTranslation, None),
        LlmClientError::RequestEncoding(_) => (RouteErrorKind::RequestEncoding, None),
        LlmClientError::ResponseTranslation(_) => (RouteErrorKind::ResponseTranslation, None),
        LlmClientError::InvalidRequest { .. } => (RouteErrorKind::InvalidRequest, None),
        LlmClientError::Configuration { .. } => (RouteErrorKind::Configuration, None),
        LlmClientError::Ffi { .. } | LlmClientError::General(_) => (RouteErrorKind::Other, None),
        _ => (RouteErrorKind::Other, None),
    };
    summary(kind, phase, upstream_status, target)
}

fn summary(
    kind: RouteErrorKind,
    phase: RouteFailurePhase,
    upstream_status: Option<u16>,
    target: Option<&ModelId>,
) -> RouteFailureSummary {
    RouteFailureSummary {
        kind,
        phase,
        upstream_status,
        target: target.cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "patient name is Jane Doe";

    #[test]
    fn execution_summary_keeps_http_status_and_target_without_body() {
        let error = RunnerError::Algorithm(LibsyError::ClientCall {
            target: ModelId::from("strong"),
            source: LlmClientError::UpstreamHttp {
                status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                body: format!("upstream response: {SECRET}"),
            },
        });

        let summary = error.execution_failure_summary();

        assert_eq!(summary.kind, RouteErrorKind::UpstreamHttp);
        assert_eq!(summary.phase, RouteFailurePhase::BeforeResponse);
        assert_eq!(summary.upstream_status, Some(503));
        assert_eq!(summary.target.as_ref().map(ModelId::as_str), Some("strong"));
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn execution_summary_reduces_untrusted_messages_to_categories() {
        let error = RunnerError::Algorithm(LibsyError::AlgorithmError {
            message: SECRET.to_string(),
        });

        let summary = error.execution_failure_summary();

        assert_eq!(summary.kind, RouteErrorKind::Algorithm);
        assert_eq!(summary.target, None);
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn stream_summary_preserves_served_target_without_source_text() {
        let error = LlmClientError::Timeout {
            source: std::io::Error::other(SECRET).into(),
        };

        let summary = stream_failure_summary(&error, Some(&ModelId::from("fallback")));

        assert_eq!(summary.kind, RouteErrorKind::Timeout);
        assert_eq!(summary.phase, RouteFailurePhase::DuringStream);
        assert_eq!(
            summary.target.as_ref().map(ModelId::as_str),
            Some("fallback")
        );
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn stream_summary_uses_context_window_model_without_a_served_target() {
        let error = LlmClientError::ContextWindowExceeded {
            model: ModelId::from("weak"),
            message: SECRET.to_string(),
        };

        let summary = stream_failure_summary(&error, None);

        assert_eq!(summary.kind, RouteErrorKind::ContextWindowExceeded);
        assert_eq!(summary.phase, RouteFailurePhase::DuringStream);
        assert_eq!(summary.target.as_ref().map(ModelId::as_str), Some("weak"));
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn client_error_categories_have_stable_telemetry_values() {
        let cases = vec![
            (
                LlmClientError::ContextWindowExceeded {
                    model: ModelId::from("weak"),
                    message: SECRET.to_string(),
                },
                RouteErrorKind::ContextWindowExceeded,
                "context_window_exceeded",
            ),
            (
                LlmClientError::Transport {
                    source: std::io::Error::other(SECRET).into(),
                },
                RouteErrorKind::Transport,
                "transport",
            ),
            (
                LlmClientError::InvalidResponse {
                    source: std::io::Error::other(SECRET).into(),
                },
                RouteErrorKind::InvalidResponse,
                "invalid_response",
            ),
            (
                LlmClientError::RequestTranslation(SECRET.to_string()),
                RouteErrorKind::RequestTranslation,
                "request_translation",
            ),
            (
                LlmClientError::RequestEncoding(SECRET.to_string()),
                RouteErrorKind::RequestEncoding,
                "request_encoding",
            ),
            (
                LlmClientError::ResponseTranslation(SECRET.to_string()),
                RouteErrorKind::ResponseTranslation,
                "response_translation",
            ),
            (
                LlmClientError::Configuration {
                    message: SECRET.to_string(),
                },
                RouteErrorKind::Configuration,
                "configuration",
            ),
            (
                LlmClientError::InvalidRequest {
                    message: SECRET.to_string(),
                },
                RouteErrorKind::InvalidRequest,
                "invalid_request",
            ),
            (
                LlmClientError::General(SECRET.to_string()),
                RouteErrorKind::Other,
                "other",
            ),
        ];

        for (error, kind, value) in cases {
            let summary = stream_failure_summary(&error, None);
            assert_eq!(summary.kind, kind);
            assert_eq!(summary.kind.as_str(), value);
            assert!(!format!("{summary:?}").contains(SECRET));
        }
    }

    #[test]
    fn direct_client_errors_do_not_claim_a_target() {
        let error = RunnerError::Client(LlmClientError::Timeout {
            source: std::io::Error::other(SECRET).into(),
        });

        let summary = error.execution_failure_summary();

        assert_eq!(summary.kind, RouteErrorKind::Timeout);
        assert_eq!(summary.phase, RouteFailurePhase::BeforeResponse);
        assert_eq!(summary.target, None);
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn runner_request_and_configuration_errors_are_classified() {
        let configuration = RunnerError::configuration(SECRET);
        assert_eq!(
            configuration.execution_failure_summary().kind,
            RouteErrorKind::Configuration
        );

        let unsupported = RunnerError::CountTokensUnsupported;
        assert_eq!(
            unsupported.execution_failure_summary().kind,
            RouteErrorKind::InvalidRequest
        );
    }
}
