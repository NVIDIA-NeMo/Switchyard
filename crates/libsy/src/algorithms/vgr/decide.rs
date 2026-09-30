// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frozen commit rules: pure, and indeterminate evidence never commits.
//!
//! A signal is `None` when its verifier was never consulted and `Some(Unknown)`
//! when it was consulted and gave no usable verdict.

use super::{Branch, Capabilities};

const READOUT: f64 = 0.9;
const DELIBERATION: f64 = 0.5;
/// Most tool results a run may contain and still recover without the capable tier.
const SHORT_RUN_MAX_TOOL_RESULTS: i32 = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Route {
    Local,
    Cloud,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Tri {
    Yes,
    No,
    Unknown,
}

/// Evidence gathered for one decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Signals {
    /// Local logprob score that the attempt is correct.
    pub(super) readout: Option<f64>,
    /// Local deliberating verdict, mapped to 1.0 or 0.0.
    pub(super) deliberation: Option<f64>,
    /// Capable-tier evidence verdict.
    pub(super) cloud_judge: Option<Tri>,
    /// Capable-tier answer verdict after `cloud_judge` affirms.
    pub(super) evidence_confirm: Option<Tri>,
    /// Capable-tier verdicts on the answer branch.
    pub(super) answer_verifier: Option<Tri>,
    pub(super) evidence_verifier: Option<Tri>,
}

/// How an agentic run's tool record lets it spend verification rungs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AgenticRun {
    /// Earlier tool errors veto the commit; no rung can change it.
    Vetoed,
    Clean,
    /// A bounded run that failed and ended clean: local rungs only.
    ShortRecovered,
    /// A long run with a clean tail: local rungs, then the capable-tier judge.
    ConfirmedRecovery,
}

pub(super) fn agentic_run(
    caps: &Capabilities,
    confirmed_min_clean_tail: Option<i32>,
) -> AgenticRun {
    let Some(tools) = caps.tools else {
        return AgenticRun::Vetoed;
    };
    if tools.errors == 0 {
        AgenticRun::Clean
    } else if tools.results <= SHORT_RUN_MAX_TOOL_RESULTS && tools.tail_clean {
        AgenticRun::ShortRecovered
    } else if confirmed_min_clean_tail.is_some_and(|min| tools.clean_tail >= min) {
        AgenticRun::ConfirmedRecovery
    } else {
        AgenticRun::Vetoed
    }
}

/// The readout bar that licenses a commit on each branch.
fn dial(branch: Branch) -> Option<f64> {
    match branch {
        Branch::Answer | Branch::DefaultVerified => Some(0.7),
        Branch::Chat => Some(0.3),
        Branch::Agentic => Some(0.2),
        Branch::Coding | Branch::Unknown => None,
    }
}

fn clears(score: Option<f64>, bar: f64) -> bool {
    score.is_some_and(|score| score >= bar)
}

fn affirms(verdict: Option<Tri>) -> bool {
    verdict == Some(Tri::Yes)
}

/// The local rungs affirm: the readout at the branch dial, or deliberation.
pub(super) fn locally_verified(branch: Branch, signals: &Signals) -> bool {
    let bar = dial(branch).map_or(READOUT, |dial| dial.min(READOUT));
    clears(signals.readout, bar) || clears(signals.deliberation, DELIBERATION)
}

/// Commits the local attempt or escalates.
///
/// Coding has no sandboxed checker here, so it never commits.
pub(super) fn decide(
    caps: &Capabilities,
    signals: &Signals,
    confirmed_min_clean_tail: Option<i32>,
) -> Route {
    let branch = caps.branch;
    let clean = caps.tools.is_none_or(|tools| tools.errors == 0);
    let commit = match branch {
        Branch::Coding | Branch::Unknown => false,
        Branch::Answer => {
            affirms(signals.answer_verifier)
                || affirms(signals.evidence_verifier)
                || clears(signals.readout, 0.7)
        }
        Branch::Chat => {
            clean
                && (clears(signals.deliberation, DELIBERATION)
                    || clears(signals.readout, 0.3)
                    || (affirms(signals.cloud_judge) && affirms(signals.evidence_confirm)))
        }
        Branch::Agentic => {
            locally_verified(branch, signals)
                && match agentic_run(caps, confirmed_min_clean_tail) {
                    AgenticRun::Clean | AgenticRun::ShortRecovered => true,
                    AgenticRun::ConfirmedRecovery => affirms(signals.cloud_judge),
                    AgenticRun::Vetoed => false,
                }
        }
        Branch::DefaultVerified => clean && locally_verified(branch, signals),
    };
    if commit { Route::Local } else { Route::Cloud }
}
