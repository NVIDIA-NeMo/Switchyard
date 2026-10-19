// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy assessment contributing a fixed private-target requirement.

mod input;
mod semantic;

use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request};

use crate::eligibility::Requirements;

const EXTERNAL_RESTRICTION_KEY: &str = "switchyard.internal.external_privacy_restriction";
use input::validate_input;
pub(crate) use semantic::SemanticPrivacyClassifier;

pub(crate) struct PrivacyPolicy {
    classifier: Option<SemanticPrivacyClassifier>,
    classifier_labels: Requirements,
}

/// Target set allowed to serve one request.
#[derive(Clone, Copy, IntoStaticStr)]
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

/// Bounded privacy assessment, separate from other target requirements.
pub(crate) struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
    reason_code: &'static str,
    clear_score: Option<f64>,
    clear_threshold: Option<f64>,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum PrivacySource {
    Policy,
    ExternalSignal,
    SemanticClassifier,
}

impl PrivacySource {
    fn as_str(self) -> &'static str {
        self.into()
    }
}

impl PrivacyDecision {
    /// Returns whether privacy permits standard targets or requires private targets.
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

impl PrivacyPolicy {
    pub(crate) fn new(
        classifier: Option<SemanticPrivacyClassifier>,
        classifier_labels: Requirements,
    ) -> Self {
        Self {
            classifier,
            classifier_labels,
        }
    }

    pub(crate) async fn assess(
        &self,
        request: &Request,
        requirements: &Requirements,
        route: &str,
        algorithm: &str,
    ) -> Result<PrivacyDecision, LlmClientError> {
        if requirements.contains("private") {
            return Ok(PrivacyDecision::new(
                PrivacyLane::Restricted,
                PrivacySource::ExternalSignal,
                "restricted",
            ));
        }
        let Some(classifier) = &self.classifier else {
            return Ok(PrivacyDecision::all_clear());
        };
        // Approval covers known constraints before any request content leaves the host.
        if !self.classifier_labels.is_superset(requirements) {
            return Err(LlmClientError::InvalidRequest {
                message: "privacy classifier target does not satisfy target requirements".into(),
            });
        }
        if let Err(reason_code) = validate_input(request) {
            return Ok(PrivacyDecision::new(
                PrivacyLane::Restricted,
                PrivacySource::Policy,
                reason_code,
            ));
        }
        Ok(classifier.assess(request, route, algorithm).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn known_private_requirements_skip_assessment() {
        let policy = PrivacyPolicy::new(None, Requirements::new());
        let requirements = Requirements::from(["private".into()]);
        for external in [false, true] {
            let mut request = Request::default();
            if external {
                crate::eligibility::require_target_labels(&mut request, ["private"])
                    .expect("trusted requirement");
            }
            let decision = policy
                .assess(&request, &requirements, "test/route", "test_algorithm")
                .await
                .expect("private requirement");
            assert_eq!(decision.lane(), "restricted");
            assert_eq!(decision.source(), "external_signal");
        }
    }
}
