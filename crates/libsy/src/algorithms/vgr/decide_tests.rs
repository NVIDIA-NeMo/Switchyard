// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral tests for the commit rules, the decision function, and the
//! readiness gates.
//!
//! Two things are pinned here: the invariants a verification-gated router cannot
//! violate without becoming unsafe, and the tunable constants in [`policy`], so
//! that changing one shows up as a test that fails with a readable name rather
//! than as a silent shift in routing behavior. Tests name the reference
//! implementation's pytest case they derive from, where one exists.
//!
//! Exhaustive rule-by-rule conformance is checked out of tree, by replaying the
//! reference's frozen corpus; it is not duplicated here.

use super::decide::{
    Decision, Readiness, ReadinessGate, Route, agentic_can_gather, decide_from_signals,
};
use super::policy::{OffloadDial, Policy};
use super::rules::{Signals, ToolErrorSignal, Tri};
use super::{Branch, Capabilities, ToolErrorCount};

// ─── fixtures ─────────────────────────────────────────────────────────────────

/// Capabilities selecting the coding-without-checks branch.
fn coding() -> Capabilities {
    Capabilities {
        task_text: Some("fix the build".into()),
        transcript: Some("view".into()),
        attempt: Some("```py\nfix()\n```".into()),
        is_coding: true,
        ..Default::default()
    }
}

/// Capabilities selecting the conversational branch.
fn chat() -> Capabilities {
    Capabilities {
        task_text: Some("explain the tradeoffs".into()),
        transcript: Some("view".into()),
        attempt: Some("a detailed comparison".into()),
        is_chat: true,
        ..Default::default()
    }
}

/// Capabilities selecting the typed-answer branch.
fn answer(structured_answer: bool) -> Capabilities {
    Capabilities {
        task_text: Some("what bird is it?".into()),
        final_answer: Some("a penguin".into()),
        transcript: Some("view".into()),
        attempt: Some("it is a penguin".into()),
        structured_answer,
        ..Default::default()
    }
}

/// Capabilities selecting the evidence-verified agentic branch.
fn agentic(tool_errors: Option<ToolErrorCount>) -> Capabilities {
    Capabilities {
        task_text: Some("move my meeting".into()),
        transcript: Some("view".into()),
        attempt: Some("[tool result] ok".into()),
        is_agentic: true,
        tool_errors,
        ..Default::default()
    }
}

fn recovered_agentic(
    tool_errors: ToolErrorCount,
    tool_results: Option<i32>,
    tool_tail_clean: Option<bool>,
) -> Capabilities {
    Capabilities {
        tool_results,
        tool_tail_clean,
        ..agentic(Some(tool_errors))
    }
}

/// Capabilities selecting the untyped default branch.
fn default_verified() -> Capabilities {
    Capabilities {
        task_text: Some("q".into()),
        transcript: Some("view".into()),
        attempt: Some("a fine plain answer".into()),
        default_verified: true,
        ..Default::default()
    }
}

/// The same capabilities, carrying a host-attested tool-error count.
fn with_host_errors(caps: Capabilities, count: i32) -> Capabilities {
    Capabilities {
        tool_errors: Some(ToolErrorCount::Host(count)),
        ..caps
    }
}

/// Signals carrying only a readout score.
fn readout(score: f64) -> Signals {
    Signals {
        readout: Some(score),
        ..Default::default()
    }
}

/// The decision under the current policy, with no readiness machinery deployed.
fn decide(caps: &Capabilities, sig: &Signals) -> Decision {
    decide_from_signals(caps, sig, &Policy::CURRENT, &Readiness::default())
}

/// The route the rules decide, under the current policy.
fn route(caps: &Capabilities, sig: &Signals) -> Route {
    decide(caps, sig).route
}

/// The route the rules decide with every offload-dial arm disabled.
fn route_no_dial(caps: &Capabilities, sig: &Signals) -> Route {
    let policy = Policy {
        offload_dial: OffloadDial::default(),
        ..Policy::CURRENT
    };
    decide_from_signals(caps, sig, &policy, &Readiness::default()).route
}

// ─── core invariants ──────────────────────────────────────────────────────────

/// From `test_unknown_signals_never_commit` and
/// `test_checker_only_literal_true_commits`.
#[test]
fn indeterminate_evidence_never_commits_on_any_branch() {
    // The defining property of the router: a verifier that timed out, returned
    // unparseable output, or was never consulted must escalate, never commit.
    let nothing_usable = Signals {
        agreement: Some(Tri::Unknown),
        answer_verifier: Some(Tri::Unknown),
        evidence_verifier: Some(Tri::Unknown),
        cloud_judge: Some(Tri::Unknown),
        evidence_confirm: Some(Tri::Unknown),
        evidence_strict: Some(Tri::Unknown),
        tests_pass: Some(Tri::Unknown),
        ..Default::default()
    };
    for caps in [
        coding(),
        chat(),
        answer(true),
        default_verified(),
        agentic(Some(ToolErrorCount::Host(0))),
        Capabilities {
            has_checks: true,
            ..Default::default()
        },
        // Nothing to verify at all.
        Capabilities::default(),
    ] {
        assert_eq!(route(&caps, &nothing_usable), Route::Cloud, "{caps:?}");
        assert_eq!(route(&caps, &Signals::default()), Route::Cloud, "{caps:?}");
    }
}

/// From `test_default_verified_rule`.
#[test]
fn the_transcript_judge_commits_on_either_of_its_two_forms() {
    // The universal judge: a cheap readout at the confident bar, or a
    // deliberating readout at its lower one. Both bars are tunable constants.
    let caps = default_verified();
    let thr = Policy::CURRENT.thresholds;

    assert_eq!(route_no_dial(&caps, &readout(thr.readout)), Route::Local);
    let deliberated = Signals {
        readout: Some(0.0),
        deliberation: Some(thr.deliberation),
        ..Default::default()
    };
    assert_eq!(route_no_dial(&caps, &deliberated), Route::Local);

    // Just under either bar commits on neither.
    let under = Signals {
        readout: Some(thr.readout - 0.01),
        deliberation: Some(thr.deliberation - 0.01),
        ..Default::default()
    };
    assert_eq!(route_no_dial(&caps, &under), Route::Cloud);
}

/// From `test_coding_readout_arm_requires_cloud_confirmation` and
/// `test_coding_band_arm_requires_dual_confirmation`.
#[test]
fn the_coding_readout_strata_require_escalating_cloud_confirmation() {
    // The readout's reliability differs sharply across its range, so what it
    // takes to commit on one differs by stratum.
    let caps = coding();
    let confident = 0.95;
    let in_band = 0.7;

    // Confident: one consulted judge is enough, and is decisive.
    for (judge, expected) in [
        (Some(Tri::Yes), Route::Local),
        (Some(Tri::No), Route::Cloud),
        (Some(Tri::Unknown), Route::Cloud),
        // Never consulted: the arm keeps its unconfirmed semantics, so a
        // deployment with no cloud verifier is unaffected.
        (None, Route::Local),
    ] {
        let sig = Signals {
            readout: Some(confident),
            cloud_judge: judge,
            ..Default::default()
        };
        assert_eq!(route_no_dial(&caps, &sig), expected, "confident, {judge:?}");
    }

    // In the band: a readout-only commit needs both confirmations.
    let dual = Signals {
        readout: Some(in_band),
        cloud_judge: Some(Tri::Yes),
        evidence_confirm: Some(Tri::Yes),
        ..Default::default()
    };
    assert_eq!(route_no_dial(&caps, &dual), Route::Local);
    for weakened in [
        Signals {
            evidence_confirm: Some(Tri::No),
            ..dual
        },
        Signals {
            cloud_judge: Some(Tri::No),
            ..dual
        },
        // Either confirmation absent means the band arm does not exist.
        Signals {
            evidence_confirm: None,
            ..dual
        },
    ] {
        assert_eq!(
            route_no_dial(&caps, &weakened),
            Route::Cloud,
            "{weakened:?}"
        );
    }

    // Below the band, neither confirmation helps.
    let below = Signals {
        readout: Some(0.49),
        ..dual
    };
    assert_eq!(route_no_dial(&caps, &below), Route::Cloud);
}

/// From `test_coding_band_arm_requires_dual_confirmation`.
#[test]
fn a_judge_refutation_at_confidence_overrides_sufficient_local_evidence() {
    // A confident readout and an explicit refutation are contradicting
    // instruments, and contradiction escalates — even against local evidence
    // that would otherwise have committed on its own.
    let caps = coding();
    let local_evidence = Signals {
        evidence_strict: Some(Tri::Yes),
        deliberation: Some(1.0),
        ..Default::default()
    };
    let confident = Signals {
        readout: Some(0.95),
        cloud_judge: Some(Tri::No),
        ..local_evidence
    };
    assert_eq!(route_no_dial(&caps, &confident), Route::Cloud);

    // The refutation is terminal only at confidence: in the band the local
    // evidence arm stands.
    let banded = Signals {
        readout: Some(0.7),
        ..confident
    };
    assert_eq!(route_no_dial(&caps, &banded), Route::Local);
}

// ─── the host tool-error veto: the trust boundary ─────────────────────────────

/// From `test_host_veto_is_canonical_not_text_gated`.
#[test]
fn a_host_error_count_vetoes_every_attempt_judging_branch() {
    // Enforced centrally rather than inside each rule, so a branch cannot
    // sidestep it by forgetting to look.
    let overwhelming = Signals {
        readout: Some(1.0),
        deliberation: Some(1.0),
        cloud_judge: Some(Tri::Yes),
        evidence_confirm: Some(Tri::Yes),
        ..Default::default()
    };
    for caps in [
        with_host_errors(coding(), 2),
        with_host_errors(chat(), 2),
        with_host_errors(default_verified(), 2),
        with_host_errors(agentic(Some(ToolErrorCount::Host(2))), 2),
    ] {
        assert_eq!(route(&caps, &overwhelming), Route::Cloud, "{caps:?}");
    }

    // Two documented exemptions. The answer branch verifies the output on its
    // own terms, and the checks branch has sandbox ground truth; neither is
    // judging the trajectory the errors occurred in.
    let verified = Signals {
        answer_verifier: Some(Tri::Yes),
        ..Default::default()
    };
    let answer_caps = with_host_errors(answer(false), 2);
    assert_eq!(decide(&answer_caps, &verified).branch, Branch::Answer);
    assert_eq!(route(&answer_caps, &verified), Route::Local);

    let checks_caps = with_host_errors(
        Capabilities {
            has_checks: true,
            ..Default::default()
        },
        2,
    );
    let passed = Signals {
        tests_pass: Some(Tri::Yes),
        ..Default::default()
    };
    assert_eq!(route(&checks_caps, &passed), Route::Local);
}

/// From `test_host_count_not_overridable_by_signals`.
#[test]
fn only_a_clean_host_attested_count_authorizes_a_commit() {
    // The host's own count is authoritative. Anything that disagrees with it
    // makes the evidence indeterminate rather than picking a winner, and a
    // clean count nobody trustworthy attested to is treated as no evidence at
    // all — it may veto, but it can never authorize.
    let confident = Signals {
        readout: Some(1.0),
        cloud_judge: Some(Tri::Yes),
        ..Default::default()
    };

    // A reported count cannot talk the host's count down.
    let host_saw_errors = with_host_errors(coding(), 3);
    let claims_clean = Signals {
        tool_errors: ToolErrorSignal::Count(0),
        ..confident
    };
    assert_eq!(route(&host_saw_errors, &claims_clean), Route::Cloud);

    // Agreement with the host commits normally.
    let host_clean = with_host_errors(coding(), 0);
    assert_eq!(route(&host_clean, &claims_clean), Route::Local);

    // A reported entry carrying no count contradicts a host count of zero,
    // where an absent entry does not. These are different states.
    assert_eq!(
        route(
            &host_clean,
            &Signals {
                tool_errors: ToolErrorSignal::NoCount,
                ..confident
            }
        ),
        Route::Cloud,
    );
    assert_eq!(route(&host_clean, &confident), Route::Local);

    // A clean count of untrusted provenance authorizes nothing.
    let untrusted = agentic(Some(ToolErrorCount::Untrusted(0)));
    let deliberated = Signals {
        deliberation: Some(1.0),
        ..Default::default()
    };
    assert_eq!(route(&untrusted, &deliberated), Route::Cloud);
}

// ─── the answer branch ────────────────────────────────────────────────────────

/// From `test_agreement_never_commits_alone_unstructured`.
#[test]
fn an_answer_commits_on_verification_or_on_a_structured_surface_only() {
    // Two models can agree on a wrong answer, so free-text agreement is not
    // evidence. It counts only where the operator has declared the surface
    // enforces a schema-validated terse format, which makes agreement
    // checkable rather than merely plausible.
    let agreed = Signals {
        agreement: Some(Tri::Yes),
        ..Default::default()
    };
    assert_eq!(route_no_dial(&answer(false), &agreed), Route::Cloud);
    assert_eq!(route_no_dial(&answer(true), &agreed), Route::Local);

    for verified in [
        Signals {
            answer_verifier: Some(Tri::Yes),
            ..Default::default()
        },
        Signals {
            evidence_verifier: Some(Tri::Yes),
            ..Default::default()
        },
    ] {
        assert_eq!(route_no_dial(&answer(false), &verified), Route::Local);
    }

    // The universal transcript judge is held off this branch: local judges were
    // measured unable to verify evidence-light answers. Flipping
    // `answer_judge_arms` re-enables it.
    let deliberated = Signals {
        deliberation: Some(1.0),
        ..Default::default()
    };
    assert_eq!(route_no_dial(&answer(false), &deliberated), Route::Cloud);
    let with_arms = Policy {
        answer_judge_arms: true,
        offload_dial: OffloadDial::default(),
        ..Policy::CURRENT
    };
    let readiness = Readiness::default();
    let decision = decide_from_signals(&answer(false), &deliberated, &with_arms, &readiness);
    assert_eq!(decision.route, Route::Local);
}

// ─── the offload dial: the operator-facing tuning knob ────────────────────────

#[test]
fn the_policy_version_starts_at_one() {
    assert_eq!(Policy::CURRENT.version, "1.0.0");
}

/// From `test_dial_point_is_the_exp98_selection`.
#[test]
fn the_policy_parameters_match_the_authoritative_reference() {
    // Changing any of these changes how much traffic is served locally. This
    // test exists so that a retune is a deliberate, visible edit.
    let policy = Policy::CURRENT;
    assert_eq!(policy.thresholds.readout, 0.9);
    assert_eq!(policy.thresholds.readout_band_low, 0.5);
    assert_eq!(policy.thresholds.deliberation, 0.5);
    assert_eq!(policy.thresholds.prior, 0.5);
    assert!(!policy.answer_judge_arms);

    let dial = policy.offload_dial;
    assert_eq!(dial.for_branch(Branch::CodingNoChecks), Some(0.2));
    assert_eq!(dial.for_branch(Branch::Chat), Some(0.3));
    assert_eq!(dial.for_branch(Branch::Answer), Some(0.7));
    assert_eq!(dial.for_branch(Branch::DefaultVerified), Some(0.7));
    assert_eq!(dial.for_branch(Branch::AgenticVerified), Some(0.2));
    // Neither of these consults a readout at all.
    assert_eq!(dial.for_branch(Branch::Checks), None);
    assert_eq!(dial.for_branch(Branch::AgenticRecognized), None);
}

/// From `test_judged_family_requires_judge_yes`.
#[test]
fn the_coding_dial_requires_a_strict_judge_affirmation() {
    let caps = coding();
    for (judge, expected) in [
        (Some(Tri::Yes), Route::Local),
        (Some(Tri::No), Route::Cloud),
        (Some(Tri::Unknown), Route::Cloud),
        (None, Route::Cloud),
    ] {
        let sig = Signals {
            readout: Some(0.25),
            cloud_judge: judge,
            ..Default::default()
        };
        assert_eq!(route(&caps, &sig), expected, "{judge:?}");
    }
}

/// From `test_coding_dial_commits_and_unknown_never` and
/// `test_host_veto_binds_the_dial_arm`.
#[test]
fn the_dial_adds_a_readout_arm_that_still_answers_to_the_veto() {
    // A readout clearing the coding dial commits where the base rule would not
    // once the judged family receives its strict confirmation.
    let confirmed_low_readout = Signals {
        readout: Some(0.25),
        evidence_strict: Some(Tri::No),
        deliberation: Some(0.0),
        cloud_judge: Some(Tri::Yes),
        ..Default::default()
    };
    assert_eq!(route(&coding(), &confirmed_low_readout), Route::Local);
    assert_eq!(
        route_no_dial(&coding(), &confirmed_low_readout),
        Route::Cloud
    );

    // Below the dial, and with no readout at all, it does not.
    let under = Signals {
        readout: Some(0.15),
        ..confirmed_low_readout
    };
    assert_eq!(route(&coding(), &under), Route::Cloud);
    let none = Signals {
        readout: None,
        ..confirmed_low_readout
    };
    assert_eq!(route(&coding(), &none), Route::Cloud);

    // The host veto binds the dial arm like every other arm.
    let vetoed = with_host_errors(coding(), 2);
    assert_eq!(route(&vetoed, &confirmed_low_readout), Route::Cloud);

    // Chat gains an arm at its own, higher bar.
    let chat_over = Signals {
        readout: Some(0.35),
        deliberation: Some(0.0),
        ..Default::default()
    };
    assert_eq!(route(&chat(), &chat_over), Route::Local);
    assert_eq!(route_no_dial(&chat(), &chat_over), Route::Cloud);

    // The latest reference point also gives the agentic branch a readout arm.
    let agentic_caps = agentic(Some(ToolErrorCount::Host(0)));
    let agentic_over = readout(0.25);
    assert_eq!(route(&agentic_caps, &agentic_over), Route::Local);
    assert_eq!(route_no_dial(&agentic_caps, &agentic_over), Route::Cloud);
}

#[test]
fn enabling_the_dial_only_ever_adds_commits() {
    // The dial is defined as an additional arm, so it must never turn a commit
    // into an escalation. Retuning a dial value is safe only while this holds;
    // it is also what lets the reference's frozen pre-dial corpus stay a valid
    // check on the current rules.
    let scores = [
        None,
        Some(0.0),
        Some(0.25),
        Some(0.55),
        Some(0.75),
        Some(0.95),
    ];
    let verdicts = [None, Some(Tri::Yes), Some(Tri::No), Some(Tri::Unknown)];
    let branches = [
        coding(),
        chat(),
        answer(false),
        answer(true),
        default_verified(),
        agentic(Some(ToolErrorCount::Host(0))),
    ];
    for caps in &branches {
        for readout in scores {
            for deliberation in scores {
                for verdict in verdicts {
                    let sig = Signals {
                        readout,
                        deliberation,
                        cloud_judge: verdict,
                        evidence_confirm: verdict,
                        evidence_strict: verdict,
                        ..Default::default()
                    };
                    if route_no_dial(caps, &sig) == Route::Local {
                        assert_eq!(
                            route(caps, &sig),
                            Route::Local,
                            "dial withdrew a commit: {caps:?} {sig:?}"
                        );
                    }
                }
            }
        }
    }
}

// ─── policy 2.11 short recovered run ─────────────────────────────────────────

#[test]
fn current_policy_keeps_switchyard_identity_and_ports_recovery_constants() {
    let policy = Policy::default();
    assert_eq!(policy.version, "1.0.0");
    assert!(policy.coding_dial_requires_judge);
    assert_eq!(
        policy.short_recovered_run,
        super::policy::ShortRecoveredRun {
            enabled: true,
            min_errors: 1,
            max_tool_results: 15,
            tail_clean: true,
        }
    );
}

#[test]
fn recovered_run_shape_and_shipped_bars_bind() {
    let strong = readout(0.95);
    for (errors, results, tail, expected) in [
        (1, Some(15), Some(true), Route::Local),
        (3, Some(15), Some(true), Route::Local),
        (1, Some(16), Some(true), Route::Cloud),
        (1, Some(15), Some(false), Route::Cloud),
        (1, Some(15), None, Route::Cloud),
        (46, Some(51), Some(true), Route::Cloud),
        (0, Some(5), Some(true), Route::Local),
        (1, Some(-1), Some(true), Route::Cloud),
        (1, None, Some(true), Route::Cloud),
    ] {
        let caps = recovered_agentic(ToolErrorCount::Host(errors), results, tail);
        assert_eq!(
            route(&caps, &strong),
            expected,
            "errors={errors} results={results:?} tail={tail:?}"
        );
    }

    let caps = recovered_agentic(ToolErrorCount::Host(1), Some(5), Some(true));
    assert_eq!(route(&caps, &readout(0.2)), Route::Local);
    assert_eq!(
        route(
            &caps,
            &Signals {
                readout: Some(0.19),
                deliberation: Some(0.49),
                ..Default::default()
            }
        ),
        Route::Cloud
    );
    assert_eq!(
        route(
            &caps,
            &Signals {
                readout: Some(0.1),
                deliberation: Some(0.5),
                ..Default::default()
            }
        ),
        Route::Local
    );
}

#[test]
fn only_matching_host_provenance_can_relax_the_agentic_error_veto() {
    let strong = readout(0.99);
    let untrusted = recovered_agentic(ToolErrorCount::Untrusted(1), Some(5), Some(true));
    assert_eq!(route(&untrusted, &strong), Route::Cloud);
    let absent = Capabilities {
        tool_results: Some(5),
        tool_tail_clean: Some(true),
        ..agentic(None)
    };
    assert_eq!(route(&absent, &strong), Route::Cloud);

    let host = recovered_agentic(ToolErrorCount::Host(1), Some(5), Some(true));
    let disagreement = Signals {
        tool_errors: ToolErrorSignal::Count(0),
        ..strong
    };
    assert_eq!(route(&host, &disagreement), Route::Cloud);
    let agreement = Signals {
        tool_errors: ToolErrorSignal::Count(1),
        ..strong
    };
    let decision = decide(&host, &agreement);
    assert_eq!(decision.route, Route::Local);
    assert!(decision.signals.recovered_run);
    assert_eq!(decision.signals.tool_results, Some(5));
    assert_eq!(decision.signals.tool_tail_clean, Some(true));

    // A caller cannot assert recovery directly in Signals; it is recomputed
    // solely from typed host capabilities.
    let spoofed = Signals {
        recovered_run: true,
        ..strong
    };
    assert_eq!(
        route(
            &recovered_agentic(ToolErrorCount::Untrusted(1), Some(5), Some(true)),
            &spoofed
        ),
        Route::Cloud
    );
}

#[test]
fn disabling_recovery_preserves_the_error_veto() {
    let mut policy = Policy::CURRENT;
    policy.short_recovered_run.enabled = false;
    let decision = decide_from_signals(
        &recovered_agentic(ToolErrorCount::Host(1), Some(5), Some(true)),
        &readout(0.95),
        &policy,
        &Readiness::default(),
    );
    assert_eq!(decision.route, Route::Cloud);
}

#[test]
fn runtime_rungs_are_available_only_to_clean_or_recovered_agentic_runs() {
    let reported = ToolErrorSignal::Count(1);
    let signals = Signals {
        tool_errors: reported,
        ..Default::default()
    };
    assert_eq!(
        agentic_can_gather(
            &recovered_agentic(ToolErrorCount::Host(1), Some(5), Some(true)),
            &signals,
            &Policy::CURRENT,
        ),
        (true, true)
    );
    for caps in [
        recovered_agentic(ToolErrorCount::Host(1), Some(16), Some(true)),
        recovered_agentic(ToolErrorCount::Host(1), Some(5), Some(false)),
        recovered_agentic(ToolErrorCount::Untrusted(1), Some(5), Some(true)),
    ] {
        assert_eq!(
            agentic_can_gather(&caps, &signals, &Policy::CURRENT),
            (false, false),
            "{caps:?}"
        );
    }
}

// ─── readiness gates: the operator-facing safety controls ─────────────────────

#[test]
fn readiness_gates_force_escalation_until_their_machinery_is_deployed() {
    // A gate asks whether what a branch's local commits depend on actually
    // exists yet. Until it does, that branch escalates however strong its
    // evidence was — while `route` still records what the rules concluded, so a
    // deployment can measure the decision it is not yet ready to serve.
    let ready = Readiness {
        checker_validated: true,
    };
    let policy = Policy::CURRENT;

    // Both coding branches need a validated sandboxed checker.
    let confident = Signals {
        readout: Some(0.95),
        cloud_judge: Some(Tri::Yes),
        tests_pass: Some(Tri::Yes),
        ..Default::default()
    };
    for caps in [
        coding(),
        Capabilities {
            has_checks: true,
            ..Default::default()
        },
    ] {
        let gated = decide(&caps, &confident);
        assert_eq!(gated.route, Route::Local, "{caps:?}");
        assert_eq!(gated.effective_route, Route::Cloud, "{caps:?}");
        assert_eq!(
            gated.readiness_gate,
            Some(ReadinessGate::SecureCheckerMissing),
            "{caps:?}"
        );
        let deployed = decide_from_signals(&caps, &confident, &policy, &ready);
        assert_eq!(deployed.effective_route, Route::Local, "{caps:?}");
        assert_eq!(deployed.readiness_gate, None, "{caps:?}");
    }

    // Both agentic branches need host-attested tool evidence.
    let recognized = Capabilities {
        prior_local: Some(0.9),
        ..agentic(None)
    };
    let decision = decide(&recognized, &Signals::default());
    assert_eq!(decision.branch, Branch::AgenticRecognized);
    assert_eq!(decision.route, Route::Local);
    assert_eq!(decision.effective_route, Route::Cloud);
    assert_eq!(
        decision.readiness_gate,
        Some(ReadinessGate::ToolEvidenceNotHostAttested)
    );
    let attested = agentic(Some(ToolErrorCount::Host(0)));
    let deliberated = Signals {
        deliberation: Some(1.0),
        ..Default::default()
    };
    assert_eq!(
        decide(&attested, &deliberated).effective_route,
        Route::Local
    );

    // An escalation is never gated: there is nothing to make safe.
    let escalated = decide(&coding(), &Signals::default());
    assert_eq!(escalated.route, Route::Cloud);
    assert_eq!(escalated.readiness_gate, None);
}
