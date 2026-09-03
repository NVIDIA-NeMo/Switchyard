// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The record of what one verification-gated turn did, and what it cost.
//!
//! [`Record`] is what an operator reads to answer "why did this turn route the
//! way it did, and what did deciding it cost". It carries the decision itself,
//! the per-rung timings and token counts behind it, and the flags that say how
//! much of the evidence gathering actually completed.
//!
//! # Why this is separate from `Decision`
//!
//! [`Decision`](super::decide::Decision) is the output of a pure function: the
//! same capabilities and signals always produce the same one. Timings, token
//! counts and transport failures are none of those things — folding them in
//! would make the decision core's output depend on when it ran. So the pure
//! record stays pure and this one wraps it.
//!
//! # What may appear here
//!
//! Everything on this type is safe to log. Nothing derived from request or
//! response *content* is recorded: errors are reduced to their class by
//! [`safe_error_summary`](crate::algorithms::util::robustness::safe_error_summary)
//! before they arrive, stage names are static labels, and the task type is a
//! closed enum. There is deliberately no field for the attempt, the judged view,
//! a verifier's reply, or an upstream error body.

use std::time::Duration;

use super::decide::{Decision, Route};
use super::TaskType;

/// Identifies the prompt set the verifiers were called with.
///
/// A decision can only be compared against another decision made by the same
/// questions, so the questions are versioned alongside the policy constants. The
/// reference derives this by hashing its own source text, which has no Rust
/// analogue; a hand-maintained version is the honest equivalent — it must be
/// bumped when a prompt in [`rungs`](super::rungs) changes.
pub(super) const PROMPT_VERSION: &str = "1";

/// Which rung a timing belongs to.
///
/// A static label rather than a string so a stage name can never carry
/// request-derived text into a log line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Stage {
    /// Producing the local attempt being judged.
    Attempt,
    /// Typing the request so derivation can select a regime.
    Typing,
    /// Running the sandboxed checker.
    Checker,
    /// Producing an independent witness answer for the agreement rung.
    Witness,
    /// The cheap probability-scored readout.
    Readout,
    /// The deliberating readout.
    Deliberation,
    /// A cloud confirmation call.
    CloudJudge,
}

impl Stage {
    /// The label this stage is recorded under.
    pub(super) fn label(self) -> &'static str {
        match self {
            Stage::Attempt => "attempt",
            Stage::Typing => "typing",
            Stage::Checker => "checker",
            Stage::Witness => "witness",
            Stage::Readout => "readout",
            Stage::Deliberation => "deliberation",
            Stage::CloudJudge => "cloud_judge",
        }
    }
}

/// Why a turn produced no usable evidence from a rung.
///
/// Distinguishes the ways a rung can fail to answer, which the decision rules
/// deliberately collapse into "indeterminate" but an operator needs apart: a
/// verifier that was never configured is a deployment gap, one that timed out is
/// a capacity problem, and one that hedged is a prompt problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Unknown {
    /// The rung ran out of decision budget before it could start.
    DeadlineExhausted,
    /// The call was made and did not return in the remaining budget.
    TimedOut,
    /// The call failed. The class is recorded separately in `errors`.
    CallFailed,
    /// The verifier replied, but not with a bare verdict.
    Unparsable,
}

impl Unknown {
    /// The label this reason is recorded under.
    fn label(self) -> &'static str {
        match self {
            Unknown::DeadlineExhausted => "deadline_exhausted",
            Unknown::TimedOut => "timed_out",
            Unknown::CallFailed => "call_failed",
            Unknown::Unparsable => "unparsable",
        }
    }
}

/// What one verification-gated turn decided, and what deciding it cost.
#[derive(Clone, Debug, Default)]
pub(super) struct Record {
    /// The decision, once one was made.
    ///
    /// `None` when the turn short-circuited before deciding — the kill switch,
    /// the open circuit breaker, or a local tier that produced no attempt.
    pub(super) decision: Option<Decision>,
    /// The route actually served, which the serving mode may hold back.
    pub(super) served_route: Option<Route>,
    /// Why the turn short-circuited, when it did.
    pub(super) short_circuit: Option<&'static str>,
    /// Wall-clock time from the start of the turn to the decision.
    pub(super) elapsed: Duration,
    /// Per-rung timings, in the order the rungs ran.
    pub(super) stages: Vec<(Stage, Duration)>,
    /// Reasons rungs returned no usable evidence, by stage.
    pub(super) unknowns: Vec<(Stage, Unknown)>,
    /// Redacted failure classes. Never carries a message body.
    pub(super) errors: Vec<String>,
    /// Tokens billed to the local tier, across the attempt and the local rungs.
    pub(super) local_tokens: u64,
    /// Tokens billed to the cloud tier, across the confirmation rungs.
    pub(super) cloud_tokens: u64,
    /// Whether any rung was skipped or cut short because the budget ran out.
    pub(super) deadline_exhausted: bool,
    /// Whether the request carried content the router cannot faithfully judge.
    pub(super) unsupported_content: bool,
    /// Spans the baseline scrub replaced before any text was sent to a verifier.
    pub(super) redaction_events: u32,
    /// The router's own typing of the request, when it typed it.
    pub(super) task_type: Option<TaskType>,
    /// Provenance of the tool-error count the veto acted on, when there was one.
    pub(super) tool_errors_source: Option<&'static str>,
    /// The prompt set the verifiers were called with.
    pub(super) prompt_version: &'static str,
}

impl Record {
    /// A record for a turn that is starting now.
    pub(super) fn new() -> Self {
        Self {
            prompt_version: PROMPT_VERSION,
            ..Self::default()
        }
    }

    /// Records how long a rung took.
    pub(super) fn stage(&mut self, stage: Stage, elapsed: Duration) {
        self.stages.push((stage, elapsed));
    }

    /// Records that a rung produced no usable evidence, and why.
    ///
    /// Budget-related reasons also set [`Record::deadline_exhausted`], so a
    /// reader does not have to scan the list to learn the turn was cut short.
    pub(super) fn unknown(&mut self, stage: Stage, reason: Unknown) {
        if matches!(reason, Unknown::DeadlineExhausted | Unknown::TimedOut) {
            self.deadline_exhausted = true;
        }
        self.unknowns.push((stage, reason));
    }

    /// Records a failure by class, discarding anything it carried.
    pub(super) fn error(&mut self, stage: Stage, error: &crate::LibsyError) {
        let summary = crate::algorithms::util::robustness::safe_error_summary(error);
        self.errors.push(format!("{}: {summary}", stage.label()));
    }

    /// Emits the record as one structured event.
    ///
    /// Sequence fields are rendered rather than nested because the tracing field
    /// set is flat; every rendered value is a static label or a number.
    pub(super) fn emit(&self) {
        let stages = self
            .stages
            .iter()
            .map(|(stage, elapsed)| format!("{}={}", stage.label(), elapsed.as_millis()))
            .collect::<Vec<_>>()
            .join(",");
        let unknowns = self
            .unknowns
            .iter()
            .map(|(stage, reason)| format!("{}={}", stage.label(), reason.label()))
            .collect::<Vec<_>>()
            .join(",");
        tracing::info!(
            target: "libsy",
            branch = ?self.decision.map(|decision| decision.branch),
            route = ?self.decision.map(|decision| decision.route),
            effective = ?self.decision.map(|decision| decision.effective_route),
            gate = ?self.decision.and_then(|decision| decision.readiness_gate),
            served = ?self.served_route,
            short_circuit = self.short_circuit,
            policy_version = self.decision.map(|decision| decision.policy_version),
            prompt_version = self.prompt_version,
            task_type = ?self.task_type,
            tool_errors_source = ?self.tool_errors_source,
            elapsed_ms = self.elapsed.as_millis() as u64,
            stages = %stages,
            unknowns = %unknowns,
            errors = ?self.errors,
            local_tokens = self.local_tokens,
            cloud_tokens = self.cloud_tokens,
            deadline_exhausted = self.deadline_exhausted,
            unsupported_content = self.unsupported_content,
            redaction_events = self.redaction_events,
            "vgr decision"
        );
    }
}

/// Counts the spans the baseline scrub replaced in the judged material.
///
/// Counts placeholders in the rendered view rather than instrumenting the
/// redactor: the view is what actually leaves for a verifier, so this measures
/// the thing that matters and needs no plumbing through the pure render path.
pub(super) fn count_redactions(judged: &str) -> u32 {
    judged.matches("[REDACTED]").count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_reason_marks_the_turn_as_cut_short() {
        // An operator reading `deadline_exhausted` must not have to scan the
        // per-stage reasons to discover the turn ran out of budget.
        let mut record = Record::new();
        record.unknown(Stage::Readout, Unknown::TimedOut);
        assert!(record.deadline_exhausted);

        let mut other = Record::new();
        other.unknown(Stage::Readout, Unknown::Unparsable);
        assert!(!other.deadline_exhausted);
    }

    #[test]
    fn an_error_is_reduced_to_its_class_before_it_is_recorded() {
        // The record is logged verbatim, so an unconstrained error message must
        // never survive into it.
        const SECRET: &str = "patient name is Jane Doe";
        let mut record = Record::new();
        record.error(
            Stage::Readout,
            &crate::LibsyError::AlgorithmError {
                message: format!("judge reply did not parse: {SECRET}"),
            },
        );
        assert_eq!(record.errors, vec!["readout: algorithm error".to_string()]);
    }

    #[test]
    fn redaction_events_count_the_scrubbed_spans() {
        assert_eq!(count_redactions("nothing to scrub"), 0);
        assert_eq!(count_redactions("key [REDACTED] and mail [REDACTED]"), 2);
    }
}
