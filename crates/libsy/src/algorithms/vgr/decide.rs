// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single decision function, and the readiness gates applied on top of it.
//!
//! [`decide_from_signals`] is where a branch, its evidence, and the host's own
//! tool-error record combine into one commit-or-escalate verdict. It is pure: the
//! same capabilities and signals always produce the same decision.
//!
//! # Two routes, not one
//!
//! A decision carries both what the rules concluded and what a deployment may act
//! on. [`Decision::route`] is the rules' own verdict. [`Decision::effective_route`]
//! applies the per-branch readiness gates, which force escalation on any branch
//! whose safety precondition is unmet regardless of how strong its evidence was.
//! Keeping both means a deployment can measure what the rules would have decided
//! while serving only what it is ready to serve.

use super::policy::Policy;
use super::rules::{self, Signals, ToolErrorSignal, ToolErrors};
use super::{Branch, Capabilities, ToolErrorCount, select_branch};

/// Where a request is served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    /// Commit the locally produced attempt.
    Local,
    /// Escalate to the capable tier.
    Cloud,
}

/// Deployment state the readiness gates consult.
///
/// Separate from [`Capabilities`] because these are facts about the route's
/// configuration, not about the request: no derivation over a request can
/// establish them, and no request may influence them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Readiness {
    /// An operator-configured sandboxed checker, with a pinned immutable-tests
    /// manifest, was validated for this route.
    pub checker_validated: bool,
}

/// The safety precondition a branch must meet before its local commits count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadinessGate {
    /// The branch judges code, so it needs an operator-configured sandboxed
    /// checker with a pinned test manifest. Coding without one is never
    /// effectively local, whatever its evidence said.
    SecureCheckerMissing,
    /// The branch vetoes on tool errors, so the count must come from the
    /// evidence source the routing host has chosen to trust.
    ToolEvidenceNotHostAttested,
}

/// The full record of one decision.
///
/// Carries the evidence it was made from so a decision can be audited without
/// re-running the verifiers that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// What the rules concluded from the evidence.
    pub route: Route,
    /// What a deployment may act on, after the readiness gates.
    pub effective_route: Route,
    /// The gate that forced escalation, when one did.
    pub readiness_gate: Option<ReadinessGate>,
    /// The verification regime the capabilities selected.
    pub branch: Branch,
    /// The evidence the decision was made from.
    pub signals: Signals,
    /// The constant set the decision was made under.
    pub policy_version: &'static str,
}

/// Resolves the tool-error evidence the veto acts on.
///
/// The host's own count is authoritative: a reported count can never override it,
/// and a reporter that disagrees with it makes the evidence indeterminate rather
/// than picking a winner. Provenance is read **only** when a count is actually
/// present, so a host-provenance marker on an absent count never stands in for
/// evidence that a trusted path produced something.
///
/// A clean count of untrusted provenance is discarded rather than believed. Such
/// a count may still veto — a reported failure escalates whoever reported it —
/// but it can never authorize a commit.
fn resolve_tool_errors(caps: &Capabilities, sig: &Signals) -> ToolErrors {
    let Some(reported) = caps.tool_errors else {
        // A reported failure may still veto, but a reported clean count has no
        // provenance and therefore cannot authorize an agentic commit.
        return match sig.tool_errors {
            ToolErrorSignal::Absent | ToolErrorSignal::NoCount => ToolErrors::NoInformation,
            ToolErrorSignal::Indeterminate => ToolErrors::Indeterminate,
            ToolErrorSignal::Count(0) => ToolErrors::NoInformation,
            ToolErrorSignal::Count(count) => ToolErrors::Count(count),
        };
    };
    let reported_count = reported.count();

    let contradicts_host = match sig.tool_errors {
        ToolErrorSignal::Absent => false,
        ToolErrorSignal::NoCount | ToolErrorSignal::Indeterminate => true,
        ToolErrorSignal::Count(count) => count != reported_count,
    };
    if contradicts_host {
        return ToolErrors::Indeterminate;
    }
    if reported_count == 0 && !reported.is_host() {
        return ToolErrors::NoInformation;
    }
    ToolErrors::Count(reported_count)
}

/// Whether host evidence describes the policy's bounded recovered-run shape.
fn recovered_run_ok(caps: &Capabilities, tool_errors: ToolErrors, policy: &Policy) -> bool {
    let config = policy.short_recovered_run;
    if !config.enabled || !caps.tool_errors.is_some_and(ToolErrorCount::is_host) {
        return false;
    }
    let ToolErrors::Count(errors) = tool_errors else {
        return false;
    };
    let Some(results) = caps.tool_results else {
        return false;
    };
    errors >= config.min_errors
        && results >= 0
        && results <= config.max_tool_results
        && (!config.tail_clean || caps.tool_tail_clean == Some(true))
}

/// Whether the agentic branch can profitably spend local verification rungs.
pub(super) fn agentic_can_gather(
    caps: &Capabilities,
    sig: &Signals,
    policy: &Policy,
) -> (bool, bool) {
    let tool_errors = resolve_tool_errors(caps, sig);
    let recovered = recovered_run_ok(caps, tool_errors, policy);
    (
        matches!(tool_errors, ToolErrors::Count(0)) || recovered,
        recovered,
    )
}

/// Decides whether to commit the local attempt or escalate.
///
/// Picks the branch the capabilities license, resolves the host tool-error veto,
/// and applies that branch's rule. The veto is enforced here rather than inside
/// the rules so that it binds every attempt-judging branch uniformly, and can
/// never be sidestepped by a branch forgetting to consult it.
///
/// Two branches are exempt, for reasons specific to what their evidence is:
/// checks, because a sandboxed test run is ground truth that supersedes
/// trajectory noise, and answer, because its verifiers judge the output
/// independently of how the attempt reached it.
pub fn decide_from_signals(
    caps: &Capabilities,
    sig: &Signals,
    policy: &Policy,
    readiness: &Readiness,
) -> Decision {
    let branch = select_branch(caps);
    let thr = &policy.thresholds;
    let dial = policy.offload_dial.for_branch(branch);
    let tool_errors = resolve_tool_errors(caps, sig);
    // No evidence of errors does not veto; anything short of a clean, trusted
    // count does.
    let veto_ok = matches!(
        tool_errors,
        ToolErrors::NoInformation | ToolErrors::Count(0)
    );

    let recovered =
        branch == Branch::AgenticVerified && recovered_run_ok(caps, tool_errors, policy);
    let mut resolved_signals = *sig;
    resolved_signals.recovered_run = recovered;
    if recovered {
        resolved_signals.tool_results = caps.tool_results;
        resolved_signals.tool_tail_clean = caps.tool_tail_clean;
    }

    let commit = match branch {
        Branch::Checks => rules::rule_checks(sig.tests_pass),
        Branch::CodingNoChecks => {
            rules::rule_coding_no_checks(
                &resolved_signals,
                thr,
                dial,
                policy.coding_dial_requires_judge,
            ) && veto_ok
        }
        Branch::Answer => rules::rule_answer(sig, caps.structured_answer, policy),
        Branch::Chat => rules::rule_chat(sig, thr, dial) && veto_ok,
        // An operator's prior never outranks the host's own error count.
        Branch::AgenticRecognized => {
            rules::rule_agentic_recognized(caps.prior_local, thr) && veto_ok
        }
        // The veto is this branch's own rule, so it is not applied twice.
        Branch::AgenticVerified => {
            rules::rule_agentic_verified(tool_errors, &resolved_signals, thr, dial)
        }
        Branch::DefaultVerified => rules::rule_default_verified(sig, thr, dial) && veto_ok,
        Branch::Unknown => false,
    };

    let route = if commit { Route::Local } else { Route::Cloud };
    let (effective_route, readiness_gate) = readiness_effective(branch, route, caps, readiness);
    Decision {
        route,
        effective_route,
        readiness_gate,
        branch,
        signals: resolved_signals,
        policy_version: policy.version,
    }
}

/// Applies the per-branch readiness gates to a decided route.
///
/// A gate is a precondition on the *deployment*, not on the request: it asks
/// whether the machinery a branch's local commits depend on actually exists yet.
/// Until it does, that branch escalates even when its evidence was conclusive.
/// Escalation is never gated, since there is nothing to make safe.
pub fn readiness_effective(
    branch: Branch,
    route: Route,
    caps: &Capabilities,
    readiness: &Readiness,
) -> (Route, Option<ReadinessGate>) {
    if route != Route::Local {
        return (Route::Cloud, None);
    }
    match branch {
        // Both coding branches judge code, and neither is trustworthy without a
        // validated checker: the branch that has one must have had it validated,
        // and the branch that has none is never effectively local.
        Branch::Checks | Branch::CodingNoChecks if !readiness.checker_validated => {
            (Route::Cloud, Some(ReadinessGate::SecureCheckerMissing))
        }
        // Provenance is the whole gate here: these branches commit on the
        // absence of tool errors, which only a host-trusted source can attest.
        Branch::AgenticVerified | Branch::AgenticRecognized
            if !matches!(caps.tool_errors, Some(ToolErrorCount::Host(_))) =>
        {
            (
                Route::Cloud,
                Some(ReadinessGate::ToolEvidenceNotHostAttested),
            )
        }
        _ => (Route::Local, None),
    }
}
