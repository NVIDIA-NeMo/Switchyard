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
/// Each entry adds a readout arm to its branch's rule. The coding arm also
/// requires its grading judge to affirm. A `None` entry disables the arm for
/// that branch, which reproduces the pre-dial policy exactly. The dial only ever
/// adds commits; it can never turn a commit into an escalation, and every veto
/// that binds a branch also binds its dial arm.
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

/// Shape that lets a short agentic run recover from earlier tool errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortRecoveredRun {
    /// Whether the recovery arm is part of this policy.
    pub enabled: bool,
    /// Least number of earlier tool errors needed to call the run recovered.
    pub min_errors: i32,
    /// Most tool results a recovered run may contain.
    pub max_tool_results: i32,
    /// Whether the final tool result must be clean.
    pub tail_clean: bool,
}

/// Frozen controls for judging proposed tool-bearing assistant turns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnVerification {
    /// Whether proposed tool-bearing turns are judged.
    pub enabled: bool,
    /// Consecutive escalation votes required to latch a session.
    pub confirmations: u32,
    /// Probability at or above which a judgment is an escalation vote.
    pub escalate_at: f64,
    /// Number of recent conversation messages retained.
    pub recent_messages: usize,
    /// Per-message character budget in the recent window.
    pub message_chars: usize,
    /// Character budget for the system anchor.
    pub system_chars: usize,
    /// Character budget for the opening user-task anchor.
    pub first_user_chars: usize,
    /// Backstop for the complete rendered trajectory.
    pub max_chars: usize,
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
    /// Whether the coding dial arm requires a strict cloud-judge affirmation.
    pub coding_dial_requires_judge: bool,
    /// Host-attested short-run recovery from earlier tool errors.
    pub short_recovered_run: ShortRecoveredRun,
    /// In-flight verification of proposed tool-bearing turns.
    pub turn_verification: TurnVerification,
}

impl Policy {
    /// The current constant set.
    ///
    /// Switchyard's first Rust policy release was assigned identity `1.0.0` in
    /// MR !3. Its behavior is synchronized with the reference POLICY 2.11
    /// shipment, but that upstream development label does not replace the Rust
    /// release identity recorded in telemetry and replay data.
    pub const CURRENT: Self = Self {
        version: "1.0.0",
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
            agentic: Some(0.2),
        },
        answer_judge_arms: false,
        coding_dial_requires_judge: true,
        short_recovered_run: ShortRecoveredRun {
            enabled: true,
            min_errors: 1,
            max_tool_results: 15,
            tail_clean: true,
        },
        turn_verification: TurnVerification {
            enabled: true,
            confirmations: 2,
            escalate_at: 0.5,
            recent_messages: 28,
            message_chars: 500,
            system_chars: 1_000,
            first_user_chars: 2_000,
            max_chars: 18_000,
        },
    };
}

impl Default for Policy {
    fn default() -> Self {
        Self::CURRENT
    }
}
