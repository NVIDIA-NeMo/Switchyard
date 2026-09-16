// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ring::digest::{SHA256, digest};
use serde::Deserialize;

use super::decide::{Readiness, ReadinessGate, Route, decide_from_signals};
use super::policy::Policy;
use super::rules::{Signals, ToolErrorSignal, Tri};
use super::{Branch, Capabilities, ToolErrorCount};

const CORPUS: &[u8] = include_bytes!("testdata/policy_2_11.json");
const CORPUS_SHA256: &str = "80b223ceb0362f64b9331248072584f6ccbca924adf9b32597cab13e298a2084";

#[derive(Deserialize)]
struct Corpus {
    corpus: String,
    policy: String,
    implementation_policy_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    capabilities: CapabilitySpec,
    #[serde(default)]
    signals: SignalSpec,
    #[serde(default)]
    readiness: ReadinessSpec,
    expected: Expected,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct CapabilitySpec {
    task_text: Option<String>,
    final_answer: Option<String>,
    transcript: Option<String>,
    has_checks: bool,
    is_coding: bool,
    is_chat: bool,
    prior_local: Option<f64>,
    default_verified: bool,
    structured_answer: bool,
    is_agentic: bool,
    tool_error_count: Option<i32>,
    tool_error_source: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct SignalSpec {
    readout: Option<f64>,
    deliberation: Option<f64>,
    evidence_strict: Option<String>,
    cloud_judge: Option<String>,
    evidence_confirm: Option<String>,
    agreement: Option<String>,
    answer_verifier: Option<String>,
    evidence_verifier: Option<String>,
    tests_pass: Option<String>,
    tool_errors: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct ReadinessSpec {
    checker_validated: bool,
}

#[derive(Deserialize)]
struct Expected {
    branch: String,
    predicted: String,
    effective: String,
    readiness_gate: String,
}

#[test]
fn pinned_policy_2_11_corpus_has_zero_mismatches() {
    assert_eq!(
        hex(digest(&SHA256, CORPUS).as_ref()),
        CORPUS_SHA256,
        "the corpus changed; review it and explicitly update the pinned digest"
    );
    let corpus: Corpus = serde_json::from_slice(CORPUS).expect("valid conformance corpus");
    assert_eq!(corpus.corpus, "switchyard-vgr-policy-2.11");
    assert_eq!(corpus.policy, "POLICY 2.11");
    assert_eq!(
        corpus.implementation_policy_version,
        Policy::CURRENT.version
    );

    let mut mismatches = Vec::new();
    for case in corpus.cases {
        let decision = decide_from_signals(
            &case.capabilities.into_capabilities(),
            &case.signals.into_signals(),
            &Policy::CURRENT,
            &Readiness {
                checker_validated: case.readiness.checker_validated,
            },
        );
        let actual = Expected {
            branch: branch_label(decision.branch).to_string(),
            predicted: route_label(decision.route).to_string(),
            effective: route_label(decision.effective_route).to_string(),
            readiness_gate: gate_label(decision.readiness_gate).to_string(),
        };
        if actual.branch != case.expected.branch
            || actual.predicted != case.expected.predicted
            || actual.effective != case.expected.effective
            || actual.readiness_gate != case.expected.readiness_gate
        {
            mismatches.push(format!(
                "{}: expected {}/{}/{}/{}, got {}/{}/{}/{}",
                case.name,
                case.expected.branch,
                case.expected.predicted,
                case.expected.effective,
                case.expected.readiness_gate,
                actual.branch,
                actual.predicted,
                actual.effective,
                actual.readiness_gate,
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} policy mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

impl CapabilitySpec {
    fn into_capabilities(self) -> Capabilities {
        let tool_errors = match (self.tool_error_source.as_deref(), self.tool_error_count) {
            (Some("host"), Some(count)) => Some(ToolErrorCount::Host(count)),
            (Some("untrusted"), Some(count)) => Some(ToolErrorCount::Untrusted(count)),
            (None, None) => None,
            other => panic!("invalid tool-error capability: {other:?}"),
        };
        Capabilities {
            task_text: self.task_text,
            final_answer: self.final_answer,
            transcript: self.transcript,
            has_checks: self.has_checks,
            is_coding: self.is_coding,
            is_chat: self.is_chat,
            prior_local: self.prior_local,
            default_verified: self.default_verified,
            structured_answer: self.structured_answer,
            is_agentic: self.is_agentic,
            tool_errors,
            ..Default::default()
        }
    }
}

impl SignalSpec {
    fn into_signals(self) -> Signals {
        Signals {
            readout: self.readout,
            deliberation: self.deliberation,
            evidence_strict: tri(self.evidence_strict.as_deref()),
            cloud_judge: tri(self.cloud_judge.as_deref()),
            evidence_confirm: tri(self.evidence_confirm.as_deref()),
            agreement: tri(self.agreement.as_deref()),
            answer_verifier: tri(self.answer_verifier.as_deref()),
            evidence_verifier: tri(self.evidence_verifier.as_deref()),
            tests_pass: tri(self.tests_pass.as_deref()),
            tool_errors: tool_error_signal(self.tool_errors.as_deref()),
            ..Default::default()
        }
    }
}

fn tri(value: Option<&str>) -> Option<Tri> {
    value.map(|value| match value {
        "yes" => Tri::Yes,
        "no" => Tri::No,
        "unknown" => Tri::Unknown,
        other => panic!("invalid tri-state value: {other}"),
    })
}

fn tool_error_signal(value: Option<&str>) -> ToolErrorSignal {
    match value {
        None | Some("absent") => ToolErrorSignal::Absent,
        Some("no_count") => ToolErrorSignal::NoCount,
        Some("indeterminate") => ToolErrorSignal::Indeterminate,
        Some(value) if value.starts_with("count:") => ToolErrorSignal::Count(
            value["count:".len()..]
                .parse()
                .expect("integer tool-error count"),
        ),
        Some(other) => panic!("invalid tool-error signal: {other}"),
    }
}

fn route_label(route: Route) -> &'static str {
    match route {
        Route::Local => "local",
        Route::Cloud => "cloud",
    }
}

fn branch_label(branch: Branch) -> &'static str {
    match branch {
        Branch::Checks => "checks",
        Branch::CodingNoChecks => "coding_no_checks",
        Branch::Answer => "answer",
        Branch::Chat => "chat",
        Branch::AgenticRecognized => "agentic_recognized",
        Branch::AgenticVerified => "agentic_verified",
        Branch::DefaultVerified => "default_verified",
        Branch::Unknown => "unknown",
    }
}

fn gate_label(gate: Option<ReadinessGate>) -> &'static str {
    match gate {
        Some(ReadinessGate::SecureCheckerMissing) => "secure_checker_missing",
        Some(ReadinessGate::ToolEvidenceNotHostAttested) => "tool_evidence_not_host_attested",
        None => "none",
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").expect("write to string");
        output
    })
}
