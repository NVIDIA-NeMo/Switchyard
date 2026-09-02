// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frozen decision constants: commit thresholds and the per-branch offload dial.
//!
//! Every number the decision rules consult lives here, so a policy revision is a
//! change to this file rather than an edit scattered across the rules. The
//! constants are frozen ahead of the measurements that justify them, which is why
//! they are data rather than tuning knobs exposed to callers.

use super::Branch;

/// Commit thresholds shared by the decision rules.
///
/// `readout` is the cheap logprob score a verifier produces in a few tokens;
/// `deliberation` is the same verifier's considered score, which costs more and
/// is therefore held to a lower bar. `prior` applies only to an operator-declared
/// agentic surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Bar the cheap readout must clear to commit on its own.
    pub readout: f64,
    /// Floor of the uncertain readout band, below which only local evidence commits.
    pub readout_band_low: f64,
    /// Bar the deliberating readout must clear.
    pub deliberation: f64,
    /// Bar an operator-declared prior must clear.
    pub prior: f64,
}

/// The per-branch offload dial: an extra readout bar that licenses a commit.
///
/// Each entry adds one disjunct to its branch's rule — commit if the branch's
/// base rule commits **or** the readout clears the dial. A `None` entry disables
/// the arm for that branch, which reproduces the pre-dial policy exactly. The
/// dial only ever adds commits; it can never turn a commit into an escalation,
/// and every veto that binds a branch also binds its dial arm.
///
/// Lowering a dial admits more local commits, which raises the share of traffic
/// served locally. The values are an empirical selection, not a derivation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OffloadDial {
    /// Bar for the coding-without-checks branch.
    pub coding: Option<f64>,
    /// Bar for the conversational branch.
    pub chat: Option<f64>,
    /// Bar for the typed-answer branch.
    pub answer: Option<f64>,
    /// Bar for the untyped default branch.
    pub default_verified: Option<f64>,
    /// Bar for the evidence-verified agentic branch.
    pub agentic: Option<f64>,
}

impl OffloadDial {
    /// The dial bar for a branch, or `None` when the branch has no dial arm.
    ///
    /// Branches absent from the dial — checks and the operator-prior agentic
    /// branch — never gain a readout arm, because neither consults a readout.
    pub fn for_branch(&self, branch: Branch) -> Option<f64> {
        match branch {
            Branch::CodingNoChecks => self.coding,
            Branch::Chat => self.chat,
            Branch::Answer => self.answer,
            Branch::DefaultVerified => self.default_verified,
            Branch::AgenticVerified => self.agentic,
            Branch::Checks | Branch::AgenticRecognized | Branch::Unknown => None,
        }
    }
}

/// The complete constant set a decision is made under.
///
/// Carried on every [`Decision`](super::Decision) as `policy_version`, so a
/// recorded decision can be replayed against the constants that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// Identifier of this constant set, recorded on every decision.
    pub version: &'static str,
    /// Commit thresholds.
    pub thresholds: Thresholds,
    /// The per-branch offload dial.
    pub offload_dial: OffloadDial,
    /// Whether the typed-answer branch may commit on the universal transcript
    /// judge alone.
    ///
    /// Disabled: local judges were measured unable to verify evidence-light
    /// answers, so an answer commits on independent verification, on typed
    /// agreement over an operator-declared structured surface, or at the dial
    /// bar — never on the judge alone.
    pub answer_judge_arms: bool,
}

impl Policy {
    /// The current constant set.
    pub const CURRENT: Self = Self {
        version: "2.10.0-dev",
        thresholds: Thresholds {
            readout: 0.9,
            readout_band_low: 0.5,
            deliberation: 0.5,
            prior: 0.5,
        },
        offload_dial: OffloadDial {
            coding: Some(0.2),
            chat: Some(0.3),
            answer: Some(0.7),
            default_verified: Some(0.7),
            // The deliberation arm already saturates this branch; a readout arm
            // added nothing at any bar.
            agentic: None,
        },
        answer_judge_arms: false,
    };
}

impl Default for Policy {
    fn default() -> Self {
        Self::CURRENT
    }
}
