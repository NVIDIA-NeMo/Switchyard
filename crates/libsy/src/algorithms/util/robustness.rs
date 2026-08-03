// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hardening helpers shared by classifier-style inner-LLM callers.
//!
//! Consulting a model to make a routing decision fails in ways the routed call does
//! not, and those failures are logged on a path that runs for every turn. The
//! natural `Display` of an error is not safe there: an upstream error body is echoed
//! verbatim, and both bodies and decode failures can embed the conversation or the
//! model's reply. [`safe_error_summary`] is the redaction those log sites use.

use switchyard_protocol::LlmClientError;

use crate::LibsyError;

/// A loggable description of `error` that carries no conversation or response content.
///
/// Each case contributes only vetted operational detail — the failure's class, the
/// target or model it concerns, and a status code where one exists. Messages libsy
/// builds itself from static strings are reported in full; anything sourced from
/// outside the process is reduced to its class.
///
/// The match is deliberately exhaustive rather than defaulting to `to_string()`, so a
/// new [`LibsyError`] variant cannot start leaking content by omission.
pub(crate) fn safe_error_summary(error: &LibsyError) -> String {
    match error {
        LibsyError::ClientCall { target, source } => {
            format!(
                "client call to target {target:?} failed: {}",
                safe_client_error(source)
            )
        }
        // Built by libsy from static text plus a serde position, so safe verbatim.
        LibsyError::AlgorithmError { message } => message.clone(),
        // A task panic message is arbitrary text from wherever the panic was raised,
        // so only the fact of the failure is reported.
        LibsyError::AlgorithmTask { .. } => "algorithm task failed".to_string(),
        LibsyError::Driver(_) => "algorithm driver failed".to_string(),
        // The operation is a static label, but the boxed source comes from a user
        // extension and is unvetted.
        LibsyError::External { operation, .. } => format!("{operation} failed"),
        // The remaining variants render only structural detail — configured target
        // names and fixed strings — and carry nothing request-derived.
        LibsyError::TargetNotFound { .. }
        | LibsyError::NoTargets
        | LibsyError::MissingFinalResponse
        | LibsyError::AllTargetsExcluded => error.to_string(),
    }
}

/// The client half of [`safe_error_summary`].
///
/// `UpstreamHttp` is the sharp edge: its `Display` interpolates the raw upstream
/// body, which routinely quotes the request back. Boxed transport, decode, and FFI
/// sources are reduced for the same reason.
pub(crate) fn safe_client_error(error: &LlmClientError) -> String {
    match error {
        LlmClientError::UpstreamHttp { status, .. } => format!("upstream HTTP {status}"),
        LlmClientError::ContextWindowExceeded { model, .. } => {
            format!("context window exceeded for model {model}")
        }
        LlmClientError::Timeout { .. } => "upstream request timed out".to_string(),
        LlmClientError::Transport { .. } => "upstream transport error".to_string(),
        LlmClientError::InvalidResponse { .. } => "invalid upstream response".to_string(),
        LlmClientError::Ffi { .. } => "foreign function interface error".to_string(),
        LlmClientError::InvalidRequest { .. } => "invalid request".to_string(),
        LlmClientError::RequestTranslation(_) => "request translation failed".to_string(),
        LlmClientError::RequestEncoding(_) => "outbound request encoding failed".to_string(),
        LlmClientError::ResponseTranslation(_) => "response translation failed".to_string(),
        // Raised by the client from its own configuration, not from a request.
        LlmClientError::Configuration { message } => {
            format!("client configuration error: {message}")
        }
        // `General` and any future variant are unvetted by construction.
        _ => "client call failed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conversation text a leaking log line would expose.
    const SECRET: &str = "patient name is Jane Doe";

    #[test]
    fn an_upstream_body_never_reaches_the_summary() {
        let error = LibsyError::ClientCall {
            target: "weak".to_string(),
            source: LlmClientError::UpstreamHttp {
                status: 400,
                body: format!(r#"{{"error":{{"message":"bad request: {SECRET}"}}}}"#),
            },
        };

        let summary = safe_error_summary(&error);

        // The status and target survive; the body does not.
        assert!(summary.contains("upstream HTTP 400"), "{summary}");
        assert!(summary.contains("weak"), "{summary}");
        assert!(!summary.contains(SECRET), "{summary}");
        // Guard against the redaction silently regressing to `Display`.
        assert!(error.to_string().contains(SECRET));
    }

    #[test]
    fn boxed_sources_are_reduced_to_their_class() {
        for source in [
            LlmClientError::Transport {
                source: std::io::Error::other(SECRET).into(),
            },
            LlmClientError::InvalidResponse {
                source: std::io::Error::other(SECRET).into(),
            },
            LlmClientError::Timeout {
                source: std::io::Error::other(SECRET).into(),
            },
        ] {
            let summary = safe_client_error(&source);
            assert!(!summary.contains(SECRET), "{summary}");
            assert!(!summary.is_empty());
        }
    }

    #[test]
    fn context_overflow_keeps_the_model_but_not_the_message() {
        let summary = safe_client_error(&LlmClientError::ContextWindowExceeded {
            model: "weak".to_string(),
            message: SECRET.to_string(),
        });
        assert!(summary.contains("weak"), "{summary}");
        assert!(!summary.contains(SECRET), "{summary}");
    }

    #[test]
    fn libsy_authored_messages_are_reported_in_full() {
        // The parse path builds this from static text plus a serde position; it is
        // the one message operators need verbatim to diagnose a bad judge reply.
        let message = "judge reply did not parse as Verdict: expected value at line 1 column 1";
        let summary = safe_error_summary(&LibsyError::AlgorithmError {
            message: message.to_string(),
        });
        assert_eq!(summary, message);
    }

    #[test]
    fn an_extension_failure_keeps_its_label_but_not_its_source() {
        let error = LibsyError::external("loading extension", std::io::Error::other(SECRET));

        let summary = safe_error_summary(&error);

        assert_eq!(summary, "loading extension failed");
        assert!(error.to_string().contains(SECRET));
    }

    #[test]
    fn structural_variants_keep_their_detail() {
        let summary = safe_error_summary(&LibsyError::TargetNotFound {
            target: "strong".to_string(),
        });
        assert!(summary.contains("strong"), "{summary}");
    }
}
