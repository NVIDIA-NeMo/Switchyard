// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe, structured summaries of route-execution failures for telemetry.

use libsy::LibsyError;
use switchyard_protocol::{LlmClientError, ModelId};

use crate::RunnerError;

/// Stable class of a terminal route-execution failure.
///
/// This deliberately carries no provider message, response body, or source
/// error. It is suitable for logs and telemetry, not client-facing rendering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteFailureCategory {
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
    /// A failure without a safe, more specific category.
    Other,
}

impl RouteFailureCategory {
    /// Returns the stable telemetry value for this category.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UpstreamHttp => "upstream_http",
            Self::ContextWindowExceeded => "context_window_exceeded",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::InvalidResponse => "invalid_response",
            Self::RequestTranslation => "request_translation",
            Self::RequestEncoding => "request_encoding",
            Self::ResponseTranslation => "response_translation",
            Self::InvalidRequest => "invalid_request",
            Self::Configuration => "configuration",
            Self::Algorithm => "algorithm",
            Self::Other => "other",
        }
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
    pub category: RouteFailureCategory,
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
                RouteFailureCategory::Configuration,
                RouteFailurePhase::BeforeResponse,
                None,
                None,
            ),
            Self::UnknownRouteModel(_)
            | Self::IncompatibleCallerFormat(_)
            | Self::CountTokensUnsupported => summary(
                RouteFailureCategory::InvalidRequest,
                RouteFailurePhase::BeforeResponse,
                None,
                None,
            ),
            Self::Algorithm(_) => summary(
                RouteFailureCategory::Algorithm,
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
    let (category, upstream_status) = match error {
        LlmClientError::UpstreamHttp { status, .. } => {
            (RouteFailureCategory::UpstreamHttp, Some(status.as_u16()))
        }
        LlmClientError::ContextWindowExceeded { .. } => {
            (RouteFailureCategory::ContextWindowExceeded, None)
        }
        LlmClientError::Timeout { .. } => (RouteFailureCategory::Timeout, None),
        LlmClientError::Transport { .. } => (RouteFailureCategory::Transport, None),
        LlmClientError::InvalidResponse { .. } => (RouteFailureCategory::InvalidResponse, None),
        LlmClientError::RequestTranslation(_) => (RouteFailureCategory::RequestTranslation, None),
        LlmClientError::RequestEncoding(_) => (RouteFailureCategory::RequestEncoding, None),
        LlmClientError::ResponseTranslation(_) => (RouteFailureCategory::ResponseTranslation, None),
        LlmClientError::InvalidRequest { .. } => (RouteFailureCategory::InvalidRequest, None),
        LlmClientError::Configuration { .. } => (RouteFailureCategory::Configuration, None),
        LlmClientError::Ffi { .. } | LlmClientError::General(_) => {
            (RouteFailureCategory::Other, None)
        }
        _ => (RouteFailureCategory::Other, None),
    };
    summary(category, phase, upstream_status, target)
}

fn summary(
    category: RouteFailureCategory,
    phase: RouteFailurePhase,
    upstream_status: Option<u16>,
    target: Option<&ModelId>,
) -> RouteFailureSummary {
    RouteFailureSummary {
        category,
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

        assert_eq!(summary.category, RouteFailureCategory::UpstreamHttp);
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

        assert_eq!(summary.category, RouteFailureCategory::Algorithm);
        assert_eq!(summary.target, None);
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn stream_summary_preserves_served_target_without_source_text() {
        let error = LlmClientError::Timeout {
            source: std::io::Error::other(SECRET).into(),
        };

        let summary = stream_failure_summary(&error, Some(&ModelId::from("fallback")));

        assert_eq!(summary.category, RouteFailureCategory::Timeout);
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

        assert_eq!(
            summary.category,
            RouteFailureCategory::ContextWindowExceeded
        );
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
                RouteFailureCategory::ContextWindowExceeded,
                "context_window_exceeded",
            ),
            (
                LlmClientError::Transport {
                    source: std::io::Error::other(SECRET).into(),
                },
                RouteFailureCategory::Transport,
                "transport",
            ),
            (
                LlmClientError::InvalidResponse {
                    source: std::io::Error::other(SECRET).into(),
                },
                RouteFailureCategory::InvalidResponse,
                "invalid_response",
            ),
            (
                LlmClientError::RequestTranslation(SECRET.to_string()),
                RouteFailureCategory::RequestTranslation,
                "request_translation",
            ),
            (
                LlmClientError::RequestEncoding(SECRET.to_string()),
                RouteFailureCategory::RequestEncoding,
                "request_encoding",
            ),
            (
                LlmClientError::ResponseTranslation(SECRET.to_string()),
                RouteFailureCategory::ResponseTranslation,
                "response_translation",
            ),
            (
                LlmClientError::Configuration {
                    message: SECRET.to_string(),
                },
                RouteFailureCategory::Configuration,
                "configuration",
            ),
            (
                LlmClientError::InvalidRequest {
                    message: SECRET.to_string(),
                },
                RouteFailureCategory::InvalidRequest,
                "invalid_request",
            ),
            (
                LlmClientError::General(SECRET.to_string()),
                RouteFailureCategory::Other,
                "other",
            ),
        ];

        for (error, category, value) in cases {
            let summary = stream_failure_summary(&error, None);
            assert_eq!(summary.category, category);
            assert_eq!(summary.category.as_str(), value);
            assert!(!format!("{summary:?}").contains(SECRET));
        }
    }

    #[test]
    fn direct_client_errors_do_not_claim_a_target() {
        let error = RunnerError::Client(LlmClientError::Timeout {
            source: std::io::Error::other(SECRET).into(),
        });

        let summary = error.execution_failure_summary();

        assert_eq!(summary.category, RouteFailureCategory::Timeout);
        assert_eq!(summary.phase, RouteFailurePhase::BeforeResponse);
        assert_eq!(summary.target, None);
        assert!(!format!("{summary:?}").contains(SECRET));
    }

    #[test]
    fn runner_request_and_configuration_errors_are_classified() {
        let configuration = RunnerError::configuration(SECRET);
        assert_eq!(
            configuration.execution_failure_summary().category,
            RouteFailureCategory::Configuration
        );

        let unsupported = RunnerError::CountTokensUnsupported;
        assert_eq!(
            unsupported.execution_failure_summary().category,
            RouteFailureCategory::InvalidRequest
        );
    }
}
