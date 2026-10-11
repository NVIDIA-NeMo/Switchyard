// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy policy used to select a route's execution lane.

mod input;
mod semantic;

use serde_json::Value;
use strum_macros::{EnumString, IntoStaticStr};
use switchyard_protocol::{LlmClientError, Request, WireFormat};

const EXTERNAL_RESTRICTION_KEY: &str = "switchyard.internal.external_privacy_restriction";
use input::validate_input;
pub(crate) use semantic::SemanticPrivacyClassifier;
pub(crate) const SELECTED_LANE_KEY: &str = "switchyard.internal.privacy_lane";
const RESPONSES_STATE_FIELDS: [&str; 2] = ["previous_response_id", "conversation"];

pub(crate) struct PrivacyPolicy {
    accept_external_signal: bool,
    classifier: Option<SemanticPrivacyClassifier>,
}

/// Target set allowed to serve one request.
#[derive(Clone, Copy, EnumString, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PrivacyLane {
    Standard,
    Restricted,
}

impl PrivacyLane {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }
}

/// Records the lane after routing so target metadata can resolve duplicate model IDs.
pub(crate) fn record_selected_lane(request: &mut Request, lane: PrivacyLane) {
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(SELECTED_LANE_KEY.to_string(), lane.as_str().to_string());
}

/// Reads the runner-owned lane marker from a completed routing request.
pub(crate) fn selected_lane(request: &Request) -> Option<PrivacyLane> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|metadata| metadata.get(SELECTED_LANE_KEY))
        .and_then(|lane| lane.parse().ok())
}

/// Bounded explanation of the privacy lane selected for one request.
pub(crate) struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
    reason_code: &'static str,
    clear_score: Option<f64>,
    clear_threshold: Option<f64>,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PrivacySource {
    Policy,
    ExternalSignal,
    SemanticClassifier,
}

impl PrivacySource {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }
}

impl PrivacyDecision {
    /// Returns the selected execution lane.
    pub(crate) fn lane(&self) -> &'static str {
        self.lane.as_str()
    }

    /// Returns the policy component that selected the lane.
    pub(crate) fn source(&self) -> &'static str {
        self.source.as_str()
    }

    /// Returns a bounded reason for the decision.
    pub(crate) const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    /// Returns the classifier probability assigned to clear content, when available.
    pub(crate) const fn clear_score(&self) -> Option<f64> {
        self.clear_score
    }

    /// Returns the minimum clear score required for the standard lane, when applicable.
    pub(crate) const fn clear_threshold(&self) -> Option<f64> {
        self.clear_threshold
    }

    pub(crate) const fn selected_lane(&self) -> PrivacyLane {
        self.lane
    }

    const fn new(lane: PrivacyLane, source: PrivacySource, reason_code: &'static str) -> Self {
        Self {
            lane,
            source,
            reason_code,
            clear_score: None,
            clear_threshold: None,
        }
    }

    const fn semantic(
        lane: PrivacyLane,
        reason_code: &'static str,
        clear_score: Option<f64>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            lane,
            source: PrivacySource::SemanticClassifier,
            reason_code,
            clear_score,
            clear_threshold: Some(clear_threshold),
        }
    }

    const fn all_clear() -> Self {
        Self::new(PrivacyLane::Standard, PrivacySource::Policy, "all_clear")
    }
}

/// Marks a request as requiring a route that accepts external privacy signals.
/// Execution fails if the selected route has not enabled `accept_external_signal`.
/// This marker is available only to trusted in-process hosts, not HTTP clients.
pub fn mark_privacy_restricted(request: &mut Request) {
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(EXTERNAL_RESTRICTION_KEY.to_string(), String::new());
}

pub(crate) fn has_external_restriction(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(EXTERNAL_RESTRICTION_KEY))
}

pub(crate) fn external_signal_not_accepted() -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: "route does not accept external privacy signals".to_string(),
    }
}

impl PrivacyPolicy {
    pub(crate) fn new(
        accept_external_signal: bool,
        classifier: Option<SemanticPrivacyClassifier>,
    ) -> Self {
        Self {
            accept_external_signal,
            classifier,
        }
    }

    pub(crate) async fn decide(
        &self,
        request: &Request,
    ) -> Result<PrivacyDecision, LlmClientError> {
        if has_external_restriction(request) && !self.accept_external_signal {
            return Err(external_signal_not_accepted());
        }
        self.assess(request).await
    }

    async fn assess(&self, request: &Request) -> Result<PrivacyDecision, LlmClientError> {
        if has_external_restriction(request) {
            return Ok(PrivacyDecision::new(
                PrivacyLane::Restricted,
                PrivacySource::ExternalSignal,
                "restricted",
            ));
        }
        Ok(match &self.classifier {
            Some(classifier) => {
                if let Err(reason_code) = validate_input(request) {
                    return Ok(PrivacyDecision::new(
                        PrivacyLane::Restricted,
                        PrivacySource::Policy,
                        reason_code,
                    ));
                }
                classifier.assess(request).await
            }
            None => PrivacyDecision::all_clear(),
        })
    }
}

pub(crate) fn validate_mixed_request(request: &Request) -> Result<(), LlmClientError> {
    let extensions = &request.llm_request.extensions.fields;
    let preserved = &request.llm_request.preservation.requests;
    let inspect_extensions = match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.wire_format)
    {
        Some(format) => format == WireFormat::OpenAiResponses,
        None => preserved.is_empty(),
    };
    let has_state = inspect_extensions
        && RESPONSES_STATE_FIELDS
            .iter()
            .any(|field| extensions.get(*field).is_some_and(|value| !value.is_null()))
        || preserved.iter().any(|(format, body)| {
            format.as_str() == WireFormat::OpenAiResponses.as_str() && has_responses_state(body)
        });
    if has_state {
        return Err(LlmClientError::InvalidRequest {
            message:
                "mixed-traffic privacy routes do not support OpenAI Responses continuation state"
                    .to_string(),
        });
    }
    Ok(())
}

fn has_responses_state(body: &Value) -> bool {
    RESPONSES_STATE_FIELDS
        .iter()
        .any(|field| body.get(*field).is_some_and(|value| !value.is_null()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{ContentBlock, Message, Role};

    #[tokio::test]
    async fn external_restriction_requires_route_opt_in() {
        let mut request = Request::default();
        mark_privacy_restricted(&mut request);

        assert!(
            PrivacyPolicy::new(false, None)
                .decide(&request)
                .await
                .is_err()
        );
        assert!(matches!(
            PrivacyPolicy::new(true, None).decide(&request).await,
            Ok(PrivacyDecision {
                lane: PrivacyLane::Restricted,
                ..
            })
        ));

        let mut opaque = Request::default();
        opaque.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: Value::Null,
            }],
        });
        let decision = PrivacyPolicy::new(true, None)
            .decide(&opaque)
            .await
            .expect("unmarked request should remain valid");
        assert!(matches!(decision.lane, PrivacyLane::Standard));
    }

    #[test]
    fn responses_continuations_are_rejected_from_normalized_and_preserved_state() {
        for field in RESPONSES_STATE_FIELDS {
            for preserved in [false, true] {
                let mut request = Request::default();
                if preserved {
                    request.llm_request.preservation.requests.insert(
                        "openai_responses".into(),
                        serde_json::json!({field: "state_123"}),
                    );
                } else {
                    request.metadata.get_or_insert_default().wire_format =
                        Some(WireFormat::OpenAiResponses);
                    request
                        .llm_request
                        .extensions
                        .fields
                        .insert(field.to_string(), Value::String("state_123".to_string()));
                }

                assert!(matches!(
                    validate_mixed_request(&request),
                    Err(LlmClientError::InvalidRequest { .. })
                ));
            }
        }

        let mut chat = Request::default();
        chat.metadata.get_or_insert_default().wire_format = Some(WireFormat::OpenAiChat);
        chat.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            serde_json::json!({"conversation": "chat extension"}),
        );
        chat.llm_request.extensions.fields.insert(
            "conversation".to_string(),
            Value::String("chat extension".to_string()),
        );
        assert!(validate_mixed_request(&chat).is_ok());

        let mut untagged = Request::default();
        untagged.llm_request.extensions.fields.insert(
            "conversation".to_string(),
            Value::String("responses state".to_string()),
        );
        assert!(validate_mixed_request(&untagged).is_err());
    }
}
