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

use opentelemetry::KeyValue;
use switchyard_protocol::Request;

use super::Branch;
use super::TaskType;
use super::decide::{Decision, ReadinessGate, Route};
use crate::observability::meter;

const VGR_DECISIONS_METRIC: &str = "switchyard.vgr.decisions";

/// Identifies the prompt set the verifiers were called with.
///
/// A decision can only be compared against another decision made by the same
/// questions, so the questions are versioned alongside the policy constants. The
/// reference derives this by hashing its own source text, which has no Rust
/// analogue; a hand-maintained version is the honest equivalent — it must be
/// bumped when a prompt in [`rungs`](super::rungs) changes.
pub(super) const PROMPT_VERSION: &str = "3";

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
    /// Judging a proposed tool-bearing assistant turn in flight.
    TurnVerification,
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
            Stage::TurnVerification => "turn_verification",
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
    /// Stable public identity of the checker contract used for this decision.
    pub(super) checker_manifest: Option<String>,
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

    /// Carries the bounded decision labels to the terminal response for durable logging.
    pub(super) fn annotate(&self, request: &mut Request) {
        let extra = request
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default();
        for (key, value) in [
            (
                "switchyard.vgr.predicted",
                route_label(self.decision.map(|d| d.route)),
            ),
            (
                "switchyard.vgr.effective",
                route_label(self.decision.map(|d| d.effective_route)),
            ),
            ("switchyard.vgr.served", route_label(self.served_route)),
            (
                "switchyard.vgr.branch",
                branch_label(self.decision.map(|d| d.branch)),
            ),
            (
                "switchyard.vgr.readiness_gate",
                readiness_gate_label(self.decision.and_then(|d| d.readiness_gate)),
            ),
            (
                "switchyard.vgr.short_circuit",
                self.short_circuit.unwrap_or("none"),
            ),
        ] {
            extra.insert(key.to_string(), value.to_string());
        }
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
        meter().u64_counter(VGR_DECISIONS_METRIC).build().add(
            1,
            &[
                KeyValue::new("predicted", route_label(self.decision.map(|d| d.route))),
                KeyValue::new(
                    "effective",
                    route_label(self.decision.map(|d| d.effective_route)),
                ),
                KeyValue::new("served", route_label(self.served_route)),
                KeyValue::new("branch", branch_label(self.decision.map(|d| d.branch))),
                KeyValue::new(
                    "readiness_gate",
                    readiness_gate_label(self.decision.and_then(|d| d.readiness_gate)),
                ),
                KeyValue::new("short_circuit", self.short_circuit.unwrap_or("none")),
            ],
        );
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
            checker_manifest = self.checker_manifest.as_deref(),
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

fn route_label(route: Option<Route>) -> &'static str {
    match route {
        Some(Route::Local) => "local",
        Some(Route::Cloud) => "cloud",
        None => "none",
    }
}

fn branch_label(branch: Option<Branch>) -> &'static str {
    match branch {
        Some(Branch::Checks) => "checks",
        Some(Branch::CodingNoChecks) => "coding_no_checks",
        Some(Branch::Answer) => "answer",
        Some(Branch::Chat) => "chat",
        Some(Branch::AgenticRecognized) => "agentic_recognized",
        Some(Branch::AgenticVerified) => "agentic_verified",
        Some(Branch::DefaultVerified) => "default_verified",
        Some(Branch::Unknown) => "unknown",
        None => "none",
    }
}

fn readiness_gate_label(gate: Option<ReadinessGate>) -> &'static str {
    match gate {
        Some(ReadinessGate::SecureCheckerMissing) => "secure_checker_missing",
        Some(ReadinessGate::ToolEvidenceNotHostAttested) => "tool_evidence_not_host_attested",
        None => "none",
    }
}

/// Labels a route retained by a processor that bypassed the VGR classifier.
pub(super) fn annotate_retained_route(request: &mut Request, route: Route) {
    let extra = request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default();
    if extra.contains_key("switchyard.vgr.served") {
        return;
    }
    let route = route_label(Some(route));
    for (key, value) in [
        ("switchyard.vgr.predicted", route),
        ("switchyard.vgr.effective", route),
        ("switchyard.vgr.served", route),
        ("switchyard.vgr.branch", "affinity"),
        ("switchyard.vgr.readiness_gate", "none"),
        ("switchyard.vgr.short_circuit", "user_turn_affinity"),
    ] {
        extra.insert(key.to_string(), value.to_string());
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
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::{Event, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt as _};
    use tracing_subscriber::{Layer, Registry};

    use super::*;

    #[derive(Default)]
    struct FieldVisitor(BTreeMap<String, String>);

    impl Visit for FieldVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    struct CaptureLayer(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber,
    {
        fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
            let mut visitor = FieldVisitor::default();
            event.record(&mut visitor);
            self.0.lock().push(visitor.0);
        }
    }

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

    #[test]
    fn decision_labels_are_attached_without_request_content() {
        let mut request = Request::default();
        let mut record = Record::new();
        record.served_route = Some(Route::Cloud);
        record.short_circuit = Some("local_unavailable");

        record.annotate(&mut request);

        let labels = request
            .metadata
            .and_then(|metadata| metadata.extra_metadata)
            .expect("VGR labels");
        assert_eq!(labels["switchyard.vgr.predicted"], "none");
        assert_eq!(labels["switchyard.vgr.effective"], "none");
        assert_eq!(labels["switchyard.vgr.served"], "cloud");
        assert_eq!(labels["switchyard.vgr.branch"], "none");
        assert_eq!(labels["switchyard.vgr.readiness_gate"], "none");
        assert_eq!(labels["switchyard.vgr.short_circuit"], "local_unavailable");
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn retained_route_gets_complete_static_evidence() {
        let mut request = Request::default();

        annotate_retained_route(&mut request, Route::Local);

        let labels = request
            .metadata
            .and_then(|metadata| metadata.extra_metadata)
            .expect("VGR labels");
        assert_eq!(labels["switchyard.vgr.predicted"], "local");
        assert_eq!(labels["switchyard.vgr.effective"], "local");
        assert_eq!(labels["switchyard.vgr.served"], "local");
        assert_eq!(labels["switchyard.vgr.branch"], "affinity");
        assert_eq!(labels["switchyard.vgr.short_circuit"], "user_turn_affinity");
    }

    #[test]
    fn checker_manifest_identity_is_emitted_with_the_decision() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Registry::default().with(CaptureLayer(Arc::clone(&captured)));
        let mut record = Record::new();
        record.checker_manifest = Some("stable-checker-digest".to_string());

        tracing::subscriber::with_default(subscriber, || record.emit());

        let fields = captured.lock();
        assert_eq!(
            fields
                .iter()
                .find(|event| event.contains_key("checker_manifest"))
                .and_then(|event| event.get("checker_manifest"))
                .map(String::as_str),
            Some("stable-checker-digest")
        );
    }
}
