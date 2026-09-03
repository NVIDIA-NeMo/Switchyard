// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The per-branch commit rules and the evidence vocabulary they read.
//!
//! One rule per [`Branch`](super::Branch), each a pure predicate over the
//! evidence that branch's verification regime licenses. A rule answers exactly
//! one question: may the locally produced attempt be committed, or must the
//! request escalate?
//!
//! # Indeterminate evidence never commits
//!
//! Every signal is optional and tri-state, because a verifier that timed out,
//! returned unparseable output, or was never consulted is not the same as one
//! that answered no. Both escalate, but they escalate for different reasons, and
//! only one of them is a fault. No rule commits on anything but a definite
//! affirmative.
//!
//! # Absence is not indeterminacy
//!
//! For the two cloud confirmations the coding and chat rules consult, whether a
//! signal was *produced at all* changes the rule, not just its outcome. A
//! confirmation that was never requested leaves the rule at its unconfirmed
//! semantics; a confirmation that was requested and came back indeterminate
//! withholds the commit it was requested to authorize. `Option<Tri>` carries
//! that distinction: `None` is absent, `Some(Tri::Unknown)` is indeterminate.

use super::policy::{Policy, Thresholds};

/// A verifier verdict, or the absence of one.
///
/// [`Tri::Unknown`] is a verdict that was sought and not obtained — a timeout, an
/// unparseable response, an unavailable verifier. It never authorizes a commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tri {
    /// The verifier affirmed.
    Yes,
    /// The verifier refuted.
    No,
    /// The verifier was consulted and produced no usable verdict.
    Unknown,
}

impl Tri {
    /// Whether this is a definite affirmative. Nothing else authorizes a commit.
    fn affirms(self) -> bool {
        self == Tri::Yes
    }
}

/// Whether an optional verdict is a definite affirmative.
fn affirms(verdict: Option<Tri>) -> bool {
    verdict.is_some_and(Tri::affirms)
}

/// The runtime's tool-error evidence as the veto resolves it.
///
/// Distinct from a raw count because two different situations both decline to
/// authorize a commit while meaning opposite things: no evidence was available,
/// versus evidence was available and is not trustworthy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolErrors {
    /// No usable error evidence. Does not veto, and does not authorize.
    NoInformation,
    /// Evidence exists but is not a usable count — sources disagreed, or the
    /// signal came back indeterminate. Vetoes.
    Indeterminate,
    /// An authoritative error count.
    Count(i32),
}

impl ToolErrors {
    /// A clean, authoritative count of zero — the only value that authorizes.
    fn is_clean(self) -> bool {
        self == ToolErrors::Count(0)
    }
}

/// A snapshot of the evidence gathered for one decision.
///
/// Every field is `None` when the corresponding verifier was never consulted,
/// which for the cloud confirmations is meaningfully different from a consulted
/// verifier that returned nothing — see the module documentation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Signals {
    /// Cheap logprob score that the attempt is correct. `None` when absent or
    /// indeterminate; the rules treat those identically, since no rule branches
    /// on whether a readout was attempted.
    pub readout: Option<f64>,
    /// The same verifier's considered score, costlier and held to a lower bar.
    pub deliberation: Option<f64>,
    /// Whether the judged view contains programmatic test evidence.
    pub evidence_strict: Option<Tri>,
    /// The grading judge's verdict over the judged view.
    pub cloud_judge: Option<Tri>,
    /// The evidence verifier's verdict over the same view.
    pub evidence_confirm: Option<Tri>,
    /// Whether the attempt's answer agrees with an independently produced one.
    pub agreement: Option<Tri>,
    /// The answer verifier's verdict, which judges the answer on its own terms.
    pub answer_verifier: Option<Tri>,
    /// The evidence verifier's verdict on the answer branch.
    pub evidence_verifier: Option<Tri>,
    /// The sandboxed checker's result.
    pub tests_pass: Option<Tri>,
    /// A tool-error count reported alongside the decision, which the host's own
    /// count overrides on disagreement.
    pub tool_errors: ToolErrorSignal,
    /// Total tool results in a host-attested recovered run.
    pub tool_results: Option<i32>,
    /// Whether the final tool result in a host-attested recovered run was clean.
    pub tool_tail_clean: Option<bool>,
    /// Whether the host-attested short-recovered-run arm was active.
    pub recovered_run: bool,
}

/// A tool-error count as reported in the signal snapshot.
///
/// Four states rather than `Option<i32>`, because all four are behaviorally
/// distinct once the host's own count is compared against them:
///
/// - [`Absent`](ToolErrorSignal::Absent) never contradicts the host.
/// - [`NoCount`](ToolErrorSignal::NoCount) *does* contradict a host count, since
///   the reporter was asked and stated it had no errors to report. It is
///   nonetheless veto-neutral when the host has no count either.
/// - [`Indeterminate`](ToolErrorSignal::Indeterminate) vetoes outright: the
///   reporter was asked and failed to answer.
/// - [`Count`](ToolErrorSignal::Count) contradicts the host only on a different
///   number.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolErrorSignal {
    /// No entry was reported.
    #[default]
    Absent,
    /// An entry was reported carrying no count.
    NoCount,
    /// An entry was reported but could not be determined.
    Indeterminate,
    /// A reported count.
    Count(i32),
}

/// Whether a score is present and clears `bar`.
fn clears(score: Option<f64>, bar: f64) -> bool {
    score.is_some_and(|value| value >= bar)
}

/// Whether the readout clears the branch's offload dial, when it has one.
fn clears_dial(readout: Option<f64>, dial: Option<f64>) -> bool {
    dial.is_some_and(|bar| clears(readout, bar))
}

/// Commits only on a definite checker pass.
///
/// The checker is ground truth, so nothing weaker than a pass substitutes for
/// one: an indeterminate result is a checker that failed to run, not a checker
/// that approved.
pub fn rule_checks(tests_pass: Option<Tri>) -> bool {
    affirms(tests_pass)
}

/// Commits coding work that has no sandboxed checker to appeal to.
///
/// Stratified by how confident the cheap readout is, because the readout's
/// reliability differs sharply across its range:
///
/// - **Confident** (readout at or above the bar): the grading judge, when it was
///   consulted, is decisive in both directions. A refutation here escalates even
///   against otherwise sufficient local evidence — a confident readout and an
///   explicit refutation are contradicting instruments, and contradiction
///   escalates.
/// - **Uncertain band** (readout between the band floor and the bar): local test
///   evidence paired with deliberation still commits. A readout-only commit
///   needs *both* cloud confirmations, since either one alone was measured to
///   admit attempts the capable tier had to rescue.
/// - **Below the band, or no readout**: local test evidence paired with
///   deliberation is the only path.
///
/// A confirmation that was never consulted leaves the rule at its unconfirmed
/// semantics, so a deployment that consults no cloud verifier is unaffected.
/// The coding dial is the judged family: clearing its bar also requires a
/// definite grading-judge affirmation.
pub fn rule_coding_no_checks(
    sig: &Signals,
    thr: &Thresholds,
    dial: Option<f64>,
    dial_requires_judge: bool,
) -> bool {
    let confident = clears(sig.readout, thr.readout);
    let deliberated = clears(sig.deliberation, thr.deliberation);
    let mut local_evidence = affirms(sig.evidence_strict) && deliberated;
    let mut readout_alone = confident;

    // A consulted judge is decisive at confidence, and its refutation is
    // terminal: it withdraws the local-evidence arm as well as the readout arm.
    if confident && let Some(judge) = sig.cloud_judge {
        readout_alone = judge.affirms();
        local_evidence = local_evidence && judge.affirms();
    }

    let in_band = sig
        .readout
        .is_some_and(|score| score >= thr.readout_band_low && score < thr.readout);
    let band_confirmed = in_band
        && sig.cloud_judge.is_some()
        && sig.evidence_confirm.is_some()
        && affirms(sig.cloud_judge)
        && affirms(sig.evidence_confirm);

    let dial_commit =
        clears_dial(sig.readout, dial) && (!dial_requires_judge || affirms(sig.cloud_judge));
    readout_alone || band_confirmed || local_evidence || dial_commit
}

/// Commits a typed answer on verification of the answer itself.
///
/// Free-text agreement never commits on its own: two models can agree on a wrong
/// answer, so agreement counts only where the operator has declared the surface
/// enforces a schema-validated terse format, which makes agreement checkable
/// rather than merely plausible. Everything else routes through a verifier that
/// judges the answer independently.
pub fn rule_answer(sig: &Signals, structured_answer: bool, policy: &Policy) -> bool {
    if affirms(sig.answer_verifier) || affirms(sig.evidence_verifier) {
        return true;
    }
    // The universal transcript judge is held off this branch by default: local
    // judges were measured unable to verify evidence-light answers.
    if policy.answer_judge_arms && rule_default_verified(sig, &policy.thresholds, None) {
        return true;
    }
    if clears_dial(sig.readout, policy.offload_dial.answer) {
        return true;
    }
    affirms(sig.agreement) && structured_answer
}

/// Commits conversational traffic on deliberation or dual cloud confirmation.
///
/// The cheap readout was measured inert on conversation at the confident bar, so
/// this branch has no readout pre-filter of its own beyond the dial. The dual
/// arm requires both cloud verifiers to have been consulted and both to affirm.
pub fn rule_chat(sig: &Signals, thr: &Thresholds, dial: Option<f64>) -> bool {
    if clears(sig.deliberation, thr.deliberation) {
        return true;
    }
    if clears_dial(sig.readout, dial) {
        return true;
    }
    sig.cloud_judge.is_some()
        && sig.evidence_confirm.is_some()
        && affirms(sig.cloud_judge)
        && affirms(sig.evidence_confirm)
}

/// Commits an operator-declared agentic surface on the operator's own prior.
///
/// Reachable only from operator route configuration: capability derivation never
/// produces a prior, so no request can select this rule.
pub fn rule_agentic_recognized(prior_local: Option<f64>, thr: &Thresholds) -> bool {
    clears(prior_local, thr.prior)
}

/// Commits tool-using traffic verified from evidence alone.
///
/// A strict tightening of the default judge: the runtime's own error count must
/// be clean *and* the judge must affirm in either of its forms. The count is the
/// veto, so an indeterminate or unavailable count never passes.
pub fn rule_agentic_verified(
    tool_errors: ToolErrors,
    sig: &Signals,
    thr: &Thresholds,
    dial: Option<f64>,
) -> bool {
    (tool_errors.is_clean() || sig.recovered_run) && rule_default_verified(sig, thr, dial)
}

/// Commits derived traffic that carries no confident type.
///
/// The universal transcript judge in both its forms: the cheap readout at the
/// confident bar, or the deliberating readout at its lower one. The dial, when
/// set, lowers the readout bar rather than adding a separate arm — the same
/// commits, reached one comparison earlier.
pub fn rule_default_verified(sig: &Signals, thr: &Thresholds, dial: Option<f64>) -> bool {
    let readout_bar = dial.map_or(thr.readout, |bar| thr.readout.min(bar));
    clears(sig.readout, readout_bar) || clears(sig.deliberation, thr.deliberation)
}
