// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure commit-or-escalate policy for verification-gated routing.

use super::{Branch, Capabilities, ToolErrorCount, select_branch};

const POLICY_VERSION: &str = "1.0.0";
const READOUT: f64 = 0.9;
const DELIBERATION: f64 = 0.5;

/// Tier selected for the terminal response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    /// Serve the verified local attempt.
    Local,
    /// Escalate to the capable tier.
    Cloud,
}

/// Definite verifier answer or unusable output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tri {
    /// Verifier affirmed.
    Yes,
    /// Verifier rejected.
    No,
    /// Verifier failed to return a usable answer.
    Unknown,
}

/// Bounded evidence gathered for one local attempt.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Signals {
    /// Cheap probability that the attempt is correct.
    pub readout: Option<f64>,
    /// Considered probability from the same verifier.
    pub deliberation: Option<f64>,
    /// Strict local evidence such as an observed successful test run.
    pub strict_evidence: Option<Tri>,
    /// Capable-tier confirmation when the coding dial is used.
    pub cloud_judge: Option<Tri>,
}

/// Auditable result of applying the frozen policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// Selected terminal tier.
    pub route: Route,
    /// Verification regime used.
    pub branch: Branch,
    /// Evidence supplied to the policy.
    pub signals: Signals,
    /// Frozen policy identity.
    pub policy_version: &'static str,
}

/// Applies the frozen VGR policy. Missing or contradictory evidence escalates.
pub fn decide_from_signals(caps: &Capabilities, signals: &Signals) -> Decision {
    let branch = select_branch(caps);
    let has_reported_error = matches!(
        caps.tool_errors,
        Some(ToolErrorCount::Host(count) | ToolErrorCount::Untrusted(count)) if count > 0
    );
    let commits = !has_reported_error
        && match branch {
            Branch::Coding => coding_commits(signals),
            Branch::Chat => {
                clears(signals.deliberation, DELIBERATION) || clears(signals.readout, 0.3)
            }
            Branch::Agentic => {
                matches!(caps.tool_errors, Some(ToolErrorCount::Host(0)))
                    && (clears(signals.deliberation, DELIBERATION) || clears(signals.readout, 0.2))
            }
            Branch::DefaultVerified => {
                clears(signals.deliberation, DELIBERATION) || clears(signals.readout, 0.7)
            }
            Branch::Unknown => false,
        };
    Decision {
        route: if commits { Route::Local } else { Route::Cloud },
        branch,
        signals: *signals,
        policy_version: POLICY_VERSION,
    }
}

fn coding_commits(signals: &Signals) -> bool {
    let confident = clears(signals.readout, READOUT);
    if confident && matches!(signals.cloud_judge, Some(Tri::No | Tri::Unknown)) {
        return false;
    }
    let strict =
        signals.strict_evidence == Some(Tri::Yes) && clears(signals.deliberation, DELIBERATION);
    let dial = clears(signals.readout, 0.2) && signals.cloud_judge == Some(Tri::Yes);
    strict || dial || confident
}

fn clears(score: Option<f64>, threshold: f64) -> bool {
    score.is_some_and(|score| score >= threshold)
}
