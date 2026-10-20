// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy helpers and observations over trusted target requirements.

use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request};

use crate::eligibility::{Requirements, external_requirements};

const EXTERNAL_RESTRICTION_KEY: &str = "switchyard.internal.external_privacy_restriction";

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

/// Privacy selection derived from trusted request and task requirements.
pub struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
    reason_code: &'static str,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum PrivacySource {
    Policy,
    ExternalSignal,
    TaskRetention,
}

impl PrivacySource {
    fn as_str(self) -> &'static str {
        self.into()
    }
}

impl PrivacyDecision {
    /// Returns whether privacy permits standard targets or requires private targets.
    pub fn lane(&self) -> &'static str {
        self.lane.as_str()
    }

    /// Returns the policy component that selected the lane.
    pub fn source(&self) -> &'static str {
        self.source.as_str()
    }

    /// Returns a bounded reason for the decision.
    pub const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    const fn new(lane: PrivacyLane, source: PrivacySource, reason_code: &'static str) -> Self {
        Self {
            lane,
            source,
            reason_code,
        }
    }

    const fn all_clear() -> Self {
        Self::new(PrivacyLane::Standard, PrivacySource::Policy, "all_clear")
    }

    pub(crate) const fn task_restriction(reason_code: &'static str) -> Self {
        Self::new(
            PrivacyLane::Restricted,
            PrivacySource::TaskRetention,
            reason_code,
        )
    }

    pub(crate) fn from_requirements(
        request: &Request,
        requirements: &Requirements,
    ) -> Result<Self, LlmClientError> {
        if requirements.contains("private") {
            return Ok(if external_requirements(request)?.contains("private") {
                Self::new(
                    PrivacyLane::Restricted,
                    PrivacySource::ExternalSignal,
                    "restricted",
                )
            } else {
                Self::task_restriction("retained_restriction")
            });
        }
        Ok(Self::all_clear())
    }
}

/// Marks a request as requiring targets with the `private` label.
/// Execution fails if the selected route has not configured that eligibility requirement set.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_private_requirements_report_their_source() {
        let requirements = Requirements::from(["private".into()]);
        for external in [false, true] {
            let mut request = Request::default();
            if external {
                crate::eligibility::require_target_labels(&mut request, ["private"])
                    .expect("trusted requirement");
            }
            let decision = PrivacyDecision::from_requirements(&request, &requirements)
                .expect("private requirement");
            assert_eq!(decision.lane(), "restricted");
            assert_eq!(
                decision.source(),
                if external {
                    "external_signal"
                } else {
                    "task_retention"
                }
            );
        }
    }
}
