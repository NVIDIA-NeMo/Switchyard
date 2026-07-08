# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import benchmark.trialqa_local_gate as gate


def _metrics(
    *,
    treatment_tokens: tuple[int, ...],
    treatment_calls: tuple[int, ...],
    treatment_scores: tuple[float, ...] | None = None,
    baseline_scores: tuple[float, ...] | None = None,
    pair_ids: tuple[str, ...] | None = None,
) -> tuple[gate.TaskMetrics, ...]:
    values: list[gate.TaskMetrics] = []
    for index, (tokens, calls) in enumerate(zip(treatment_tokens, treatment_calls, strict=True)):
        pair_id = pair_ids[index] if pair_ids is not None else f"pair-{index}"
        baseline_score = (
            baseline_scores[index]
            if baseline_scores is not None
            else 1.0
            if treatment_scores is not None
            else None
        )
        treatment_score = treatment_scores[index] if treatment_scores is not None else None
        values.extend(
            [
                gate.TaskMetrics(
                    task_id=f"{pair_id}-baseline",
                    pair_id=pair_id,
                    condition="baseline",
                    total_tokens=100,
                    operational_calls=10,
                    successful_operational_calls=10,
                    skill_load_calls=1,
                    successful_skill_load_calls=1,
                    model_turns=5,
                    score=baseline_score,
                ),
                gate.TaskMetrics(
                    task_id=f"{pair_id}-treatment",
                    pair_id=pair_id,
                    condition="treatment",
                    total_tokens=tokens,
                    operational_calls=calls,
                    successful_operational_calls=calls,
                    skill_load_calls=1,
                    successful_skill_load_calls=1,
                    model_turns=4,
                    score=treatment_score,
                ),
            ]
        )
    return tuple(values)


def test_operational_gate_promotes_robust_efficiency_win() -> None:
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70, 75, 80, 80, 82, 84, 85, 85),
            treatment_calls=(5, 6, 6, 7, 7, 7, 8, 8),
        ),
        gate="operational",
    )

    assert report["decision"] == "promote_to_score"
    assert report["performance_eligible"] is False
    assert report["benefit"]["token_reduction_fraction"] >= 0.15
    assert report["benefit"]["operational_call_reduction_fraction"] >= 0.20
    assert all(item["passed"] for item in report["criteria"])


def test_treatment_retry_sensitivity_is_charged_against_token_benefit() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(80, 80, 80, 80),
            treatment_calls=(5, 5, 5, 5),
        )
    )
    for index in range(1, len(metrics), 2):
        metrics[index] = replace(
            metrics[index],
            physical_attempts=metrics[index].model_turns + 1,
            null_eof_retries=1,
            retry_usage_charges=1,
            retry_token_sensitivity=10,
        )

    report = gate.build_gate_report(metrics, gate="operational")

    assert report["decision"] == "kill"
    assert report["benefit"]["raw_token_reduction_fraction"] == pytest.approx(0.20)
    assert report["benefit"]["token_reduction_fraction"] == pytest.approx(0.10)
    assert report["benefit"]["treatment_retry_token_sensitivity"] == 40
    assert report["paired"]["raw_token_deltas"] == [-20] * 4
    assert report["paired"]["token_deltas"] == [-10] * 4


def test_baseline_retry_sensitivity_cannot_inflate_treatment_benefit() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(85, 85, 85, 85),
            treatment_calls=(5, 5, 5, 5),
        )
    )
    for index in range(0, len(metrics), 2):
        metrics[index] = replace(
            metrics[index],
            physical_attempts=metrics[index].model_turns + 1,
            null_eof_retries=1,
            retry_usage_charges=1,
            retry_token_sensitivity=100,
        )

    report = gate.build_gate_report(metrics, gate="operational")

    assert report["decision"] == "kill"
    assert report["benefit"]["token_reduction_fraction"] == pytest.approx(0.15)
    assert report["benefit"]["raw_token_reduction_fraction"] == pytest.approx(0.15)
    assert report["benefit"]["baseline_retry_token_sensitivity_ignored_for_benefit"] == 400


def test_unpriced_null_eof_retry_kills_plumbing() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        )
    )
    metrics[1] = replace(
        metrics[1],
        physical_attempts=metrics[1].model_turns + 1,
        null_eof_retries=1,
        unpriced_null_eof_retries=1,
    )

    report = gate.build_gate_report(metrics, gate="plumbing")

    assert report["decision"] == "kill"
    criterion = next(
        item for item in report["criteria"] if item["name"] == "no_unpriced_null_eof_retries"
    )
    assert criterion["passed"] is False


def test_even_priced_null_eof_retry_makes_performance_capture_nonpromotable() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        )
    )
    metrics[1] = replace(
        metrics[1],
        physical_attempts=metrics[1].model_turns + 1,
        null_eof_retries=1,
        retry_usage_charges=1,
        retry_token_sensitivity=5,
    )

    report = gate.build_gate_report(metrics, gate="plumbing")

    assert report["decision"] == "kill"
    criterion = next(
        item
        for item in report["criteria"]
        if item["name"] == "no_null_eof_retries_in_performance_capture"
    )
    assert criterion["passed"] is False


def test_development_gate_is_always_labeled_historical_and_nonperformance() -> None:
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        ),
        gate="operational",
        manifest_kind="development",
        manifest_task_count=120,
        control_design="historical-cached-donor",
        baseline_manifest_id="trialqa-donor-historical",
    )

    assert report["decision"] == "promote_to_score"
    assert report["control_design"] == "historical-cached-donor"
    assert report["baseline_manifest_id"] == "trialqa-donor-historical"
    assert report["performance_eligible"] is False


def test_any_generation_timeout_kills_the_gate() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        )
    )
    metrics[1] = replace(
        metrics[1],
        terminal=True,
        timed_out=True,
        generation_timeout_seconds=600,
        generation_timeout_policy=gate.batch.DEVELOPMENT_GENERATION_TIMEOUT_POLICY,
    )

    report = gate.build_gate_report(metrics, gate="operational")

    assert report["decision"] == "kill"
    assert report["generation_timeout"]["timed_out_task_ids"] == ["pair-0-treatment"]
    timeout_criterion = next(
        item for item in report["criteria"] if item["name"] == "no_generation_timeouts"
    )
    assert timeout_criterion["passed"] is False
    assert report["generation_timeout"]["conditions"]["treatment"] == {
        "seconds": [600, 1800],
        "policies": ["development-terminal-v1", "protocol-default-v1"],
    }


def test_stale_source_attestation_kills_the_gate() -> None:
    metrics = [
        gate.TaskMetrics(
            task_id="pair-0-baseline",
            pair_id="pair-0",
            condition="baseline",
            total_tokens=100,
            model_turns=2,
            operational_calls=2,
            successful_operational_calls=2,
            skill_load_calls=0,
            successful_skill_load_calls=0,
            terminal=False,
            score=None,
            score_bound=False,
        ),
        gate.TaskMetrics(
            task_id="pair-0-treatment",
            pair_id="pair-0",
            condition="treatment",
            total_tokens=70,
            model_turns=2,
            operational_calls=1,
            successful_operational_calls=1,
            skill_load_calls=1,
            successful_skill_load_calls=1,
            terminal=False,
            score=None,
            score_bound=False,
        ),
    ]

    report = gate.build_gate_report(
        manifest_id="manifest",
        manifest_kind="development",
        manifest_task_count=1,
        metrics=metrics,
        gate="plumbing",
        candidate=None,
        source_attestation_current=False,
    )

    assert report["decision"] == "kill"
    source_criterion = next(
        item for item in report["criteria"] if item["name"] == "source_attestation_current"
    )
    assert source_criterion["passed"] is False


def test_plumbing_gate_reports_compliant_duplicate_skill_loads() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        )
    )
    metrics[1] = replace(
        metrics[1],
        skill_load_calls=2,
        successful_skill_load_calls=2,
    )

    report = gate.build_gate_report(metrics, gate="plumbing")

    assert report["decision"] == "promote_to_operational"
    assert report["compliance"]["gating"] is False
    criterion = next(
        item
        for item in report["compliance"]["diagnostics"]
        if item["name"] == "at_least_one_successful_skill_load_per_treatment"
    )
    assert criterion["passed"] is True
    assert criterion["compliance_rate"] == 1.0
    assert criterion["noncompliant_task_ids"] == []


def test_plumbing_gate_keeps_tool_and_skill_compliance_diagnostic_only() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70, 75, 80, 80),
            treatment_calls=(5, 6, 7, 8),
        )
    )
    metrics[1] = replace(
        metrics[1],
        operational_calls=0,
        successful_operational_calls=0,
        skill_load_calls=1,
        successful_skill_load_calls=0,
    )

    report = gate.build_gate_report(metrics, gate="plumbing")

    assert report["decision"] == "promote_to_operational"
    assert report["compliance"]["gating"] is False
    diagnostics = {item["name"]: item for item in report["compliance"]["diagnostics"]}
    assert diagnostics["at_least_one_successful_skill_load_per_treatment"] == {
        "name": "at_least_one_successful_skill_load_per_treatment",
        "value": False,
        "operator": "is",
        "threshold": True,
        "passed": False,
        "gating": False,
        "scope": "treatment",
        "task_count": 4,
        "compliant_task_count": 3,
        "compliance_rate": 0.75,
        "noncompliant_task_ids": ["pair-0-treatment"],
    }
    assert diagnostics["at_least_one_operational_call_per_task"] == {
        "name": "at_least_one_operational_call_per_task",
        "value": False,
        "operator": "is",
        "threshold": True,
        "passed": False,
        "gating": False,
        "scope": "all",
        "task_count": 8,
        "compliant_task_count": 7,
        "compliance_rate": 0.875,
        "noncompliant_task_ids": ["pair-0-treatment"],
    }
    assert not {
        "at_least_one_successful_skill_load_per_treatment",
        "at_least_one_operational_call_per_task",
    } & {item["name"] for item in report["criteria"]}


def test_timeout_evidence_survives_a_successful_retry(tmp_path: Path) -> None:
    manifest = {
        "manifest_id": "trialqa-development-timeout-retry",
        "tasks": [{"task_id": "pair-treatment"}],
    }
    ledger = gate.demo.ResumableLedger(tmp_path / "ledger.jsonl", manifest)
    ledger.append("pair-treatment", "generation_started")
    ledger.append(
        "pair-treatment",
        "failed",
        {
            "stage": "generation",
            "timed_out": True,
            "retry_permitted": True,
        },
    )
    ledger.append("pair-treatment", "generation_started")
    ledger.append("pair-treatment", "generation_completed")

    assert gate._task_timed_out(ledger, "pair-treatment") is True


def test_operational_gate_kills_headline_win_driven_by_one_outlier() -> None:
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(1, 105, 105, 105, 105, 105, 105, 105),
            treatment_calls=(1, 11, 11, 11, 11, 11, 11, 11),
        ),
        gate="operational",
    )

    assert report["decision"] == "kill"
    assert report["paired"]["median_token_delta"] > 0
    assert report["paired"]["token_cheaper_pairs"] == {
        "baseline": 7,
        "treatment": 1,
        "equal": 0,
    }


def _primary_scope() -> dict[str, object]:
    return {
        "question_start": 88,
        "question_count": 8,
        "repeat_count": 5,
        "task_count": 80,
        "question_group_keys_sha256": "sha256:" + "1" * 64,
    }


def test_early_primary_operational_gate_continues_when_final_thresholds_are_noisy() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 92))
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(95,) * 4,
            treatment_calls=(9,) * 4,
            pair_ids=pair_ids,
        ),
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "promote_to_score"
    assert report["efficiency_checkpoint"]["mode"] == "early_primary_futility"
    assert report["efficiency_checkpoint"]["final_thresholds_enforced"] is False
    assert report["efficiency_checkpoint"]["visible_aggregate_benefit_required"] is True
    assert report["efficiency_checkpoint"]["visible_aggregate_benefit_passed"] is True
    assert report["efficiency_checkpoint"]["visible_paired_majority_benefit_required"] is True
    assert report["efficiency_checkpoint"]["visible_paired_majority_benefit_passed"] is True
    assert report["efficiency_checkpoint"]["visible_benefit_passed"] is True
    assert report["benefit"]["token_reduction_fraction"] > 0
    assert report["benefit"]["operational_call_reduction_fraction"] > 0
    assert report["benefit"]["token_reduction_fraction"] < gate.TOKEN_REDUCTION_MIN
    assert (
        report["benefit"]["operational_call_reduction_fraction"]
        < gate.OPERATIONAL_CALL_REDUCTION_MIN
    )


def test_prospective_primary_canary_scope_is_manifest_bound() -> None:
    groups = [f"trialqa-{index:04d}-prospective" for index in range(8)]
    primary_scope = {
        "question_start": 0,
        "question_count": 8,
        "repeat_count": 5,
        "task_count": 80,
        "question_group_keys_sha256": gate.demo._sha256_bytes(
            gate.demo._canonical_json(groups)
        ),
    }
    manifest = {
        "manifest_id": "trialqa-full-prospective",
        "kind": "full",
        "dataset": {
            "official_labbench2": False,
            "test_count": 8,
            "heldout_ordering": {
                "question_count": 8,
                "question_group_keys": groups,
                "question_group_keys_sha256": primary_scope["question_group_keys_sha256"],
            },
        },
        "protocol": {
            "primary_evaluation_scope": primary_scope,
            "performance_eligible": True,
        },
    }
    tasks = [
        {
            "task_id": f"{group}-r{repeat:03d}-{condition}",
            "pair_id": f"{group}-r{repeat:03d}",
            "question_group_key": group,
            "repeat_index": repeat,
            "condition": condition,
            "partition": "test",
            "phase": "evaluation",
        }
        for group in groups
        for repeat in range(1, 6)
        for condition in ("baseline", "treatment")
    ]

    scope = gate.batch._build_manifest_task_scope(
        manifest,
        tasks,
        limit=None,
        question_start=0,
        question_limit=4,
        repeat_limit=1,
        condition="both",
    )

    assert len(scope.tasks) == 8
    assert scope.selected_question_groups == tuple(groups[:4])
    assert scope.selected_repeat_indices == (1,)
    assert scope.heldout_quarantine_questions == 0
    assert scope.metadata(manifest["manifest_id"])["question_end_exclusive"] == 4

    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70, 75, 80, 82),
            treatment_calls=(5, 6, 7, 8),
            pair_ids=tuple(f"{group}-r001" for group in groups[:4]),
        ),
        gate="operational",
        manifest_id=manifest["manifest_id"],
        manifest_kind="full",
        manifest_task_count=len(tasks),
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
        scope_attestation=scope.metadata(manifest["manifest_id"]),
    )

    assert report["decision"] == "promote_to_score"
    assert report["scope"]["confirmatory_scope_complete"] is False
    assert report["scope"]["selection_attestation"]["selected_question_count"] == 4
    assert report["scope"]["selection_attestation"]["selected_repeat_indices"] == [1]


def test_early_primary_operational_gate_kills_lopsided_aggregate_win() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 92))
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(50, 120, 120, 50),
            treatment_calls=(1, 11, 11, 9),
            pair_ids=pair_ids,
        ),
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "kill"
    assert report["benefit"]["token_reduction_fraction"] > 0
    assert report["benefit"]["operational_call_reduction_fraction"] > 0
    assert report["efficiency_checkpoint"]["visible_aggregate_benefit_passed"] is True
    assert report["efficiency_checkpoint"]["visible_paired_majority_benefit_passed"] is False
    assert report["efficiency_checkpoint"]["visible_benefit_passed"] is False
    assert report["paired"]["token_cheaper_pairs"] == {
        "baseline": 2,
        "treatment": 2,
        "equal": 0,
    }
    assert report["paired"]["operational_call_cheaper_pairs"] == {
        "baseline": 2,
        "treatment": 2,
        "equal": 0,
    }
    failed = {item["name"] for item in report["criteria"] if not item["passed"]}
    assert failed == {
        "early_treatment_token_cheaper_pairs_majority",
        "early_treatment_call_cheaper_pairs_majority",
    }


def test_early_primary_operational_gate_kills_without_visible_benefit() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 92))
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(100, 100, 100, 100),
            treatment_calls=(10, 10, 10, 10),
            pair_ids=pair_ids,
        ),
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "kill"
    assert report["efficiency_checkpoint"]["mode"] == "early_primary_futility"
    assert report["efficiency_checkpoint"]["final_thresholds_enforced"] is False
    assert report["efficiency_checkpoint"]["futile"] is False
    assert report["efficiency_checkpoint"]["visible_aggregate_benefit_required"] is True
    assert report["efficiency_checkpoint"]["visible_aggregate_benefit_passed"] is False
    assert report["efficiency_checkpoint"]["visible_paired_majority_benefit_required"] is True
    assert report["efficiency_checkpoint"]["visible_paired_majority_benefit_passed"] is False
    assert report["efficiency_checkpoint"]["visible_benefit_passed"] is False
    failed = {item["name"] for item in report["criteria"] if not item["passed"]}
    assert failed == {
        "early_aggregate_token_reduction_positive",
        "early_aggregate_operational_call_reduction_positive",
        "early_treatment_token_cheaper_pairs_majority",
        "early_treatment_call_cheaper_pairs_majority",
    }


def test_early_primary_operational_gate_kills_when_both_benefits_are_ruled_out() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 92))
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(110,) * 4,
            treatment_calls=(12,) * 4,
            pair_ids=pair_ids,
        ),
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "kill"
    assert report["efficiency_checkpoint"]["futile"] is True
    assert report["efficiency_checkpoint"]["token_delta_optimistic_lower_bound"] > 0
    assert report["efficiency_checkpoint"]["operational_call_delta_optimistic_lower_bound"] > 0


def test_complete_question_sweep_enforces_final_efficiency_thresholds() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 96))
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(95,) * 8,
            treatment_calls=(9,) * 8,
            pair_ids=pair_ids,
        ),
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "kill"
    assert report["efficiency_checkpoint"]["mode"] == "full_v3_enforcement"
    assert report["efficiency_checkpoint"]["final_thresholds_enforced"] is True


def test_promotion_gate_waits_for_scores_then_accepts_quality_parity() -> None:
    unscored = _metrics(
        treatment_tokens=(70, 75, 80, 80),
        treatment_calls=(5, 6, 7, 8),
    )
    assert gate.build_gate_report(unscored, gate="promotion")["decision"] == "incomplete"

    scored = _metrics(
        treatment_tokens=(70, 75, 80, 80),
        treatment_calls=(5, 6, 7, 8),
        treatment_scores=(1.0, 1.0, 1.0, 1.0),
    )
    report = gate.build_gate_report(scored, gate="promotion")

    assert report["decision"] == "promote_to_next_cohort"
    assert report["benefit"]["mean_score_delta"] == 0.0
    assert report["schema_version"] == "switchyard.trialqa_gate_report.v3"
    assert report["policy"]["name"] == "ultra-efficiency-v3"
    assert report["quality"]["mode"] == "interim_harm_screen"
    assert report["quality"]["method"] == "paired-binary-bonferroni-clopper-pearson"


def test_equal_terminal_failures_are_zero_score_itt_not_plumbing_failures() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70,) * 16,
            treatment_calls=(5,) * 16,
            treatment_scores=(1.0,) * 16,
        )
    )
    for pair_index in range(5):
        for metric_index in (pair_index * 2, pair_index * 2 + 1):
            metrics[metric_index] = replace(
                metrics[metric_index],
                operational_calls=0,
                successful_operational_calls=0,
                skill_load_calls=0,
                successful_skill_load_calls=0,
                score=0.0,
                terminal=True,
            )

    report = gate.build_gate_report(metrics, gate="promotion")

    assert report["decision"] == "promote_to_next_cohort"
    assert report["conditions"]["baseline"]["mean_score"] == 11 / 16
    assert report["conditions"]["treatment"]["mean_score"] == 11 / 16
    assert report["conditions"]["baseline"]["terminal_tasks"] == 5
    assert report["conditions"]["treatment"]["terminal_tasks"] == 5
    assert report["conditions"]["baseline"]["terminal_rate"] == 5 / 16
    assert report["conditions"]["treatment"]["terminal_rate"] == 5 / 16
    assert report["benefit"]["mean_score_delta"] == 0.0
    assert report["benefit"]["terminal_rate_delta"] == 0.0
    policy_metadata = {
        "analysis_population": "intention-to-treat",
        "completed_draw_policy": "terminal-no-retry-or-replacement",
        "empty_answer_policy": "score-zero",
        "operational_call_compliance_policy": "diagnostic-only",
        "skill_load_compliance_policy": "diagnostic-only",
    }
    assert {key: report["policy"][key] for key in policy_metadata} == policy_metadata
    assert all(diagnostic["passed"] is False for diagnostic in report["compliance"]["diagnostics"])


def test_asymmetric_terminal_rate_remains_an_operational_veto() -> None:
    metrics = list(
        _metrics(
            treatment_tokens=(70,) * 8,
            treatment_calls=(5,) * 8,
            treatment_scores=(1.0,) * 8,
        )
    )
    metrics[1] = replace(
        metrics[1],
        operational_calls=0,
        successful_operational_calls=0,
        skill_load_calls=0,
        successful_skill_load_calls=0,
        score=0.0,
        terminal=True,
    )

    report = gate.build_gate_report(metrics, gate="promotion")

    assert report["decision"] == "kill"
    assert report["quality"]["upper_bound"] >= gate.QUALITY_DELTA_MIN
    failed = [item["name"] for item in report["criteria"] if not item["passed"]]
    assert failed == ["treatment_terminal_rate_delta"]


def test_early_primary_checkpoint_still_enforces_terminal_rate_delta() -> None:
    pair_ids = tuple(f"trialqa-{question:04d}-hash-r001" for question in range(88, 92))
    metrics = list(
        _metrics(
            treatment_tokens=(70,) * 4,
            treatment_calls=(5,) * 4,
            pair_ids=pair_ids,
        )
    )
    metrics[1] = replace(metrics[1], terminal=True)

    report = gate.build_gate_report(
        metrics,
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=_primary_scope(),
        manifest_performance_eligible=True,
    )

    assert report["decision"] == "kill"
    terminal_criterion = next(
        item for item in report["criteria"] if item["name"] == "treatment_terminal_rate_delta"
    )
    assert terminal_criterion["passed"] is False


def test_interim_quality_screen_continues_after_one_loss_at_n8() -> None:
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70,) * 8,
            treatment_calls=(5,) * 8,
            treatment_scores=(1.0,) * 7 + (0.0,),
        ),
        gate="promotion",
    )

    assert report["decision"] == "promote_to_next_cohort"
    assert report["quality"]["mode"] == gate.INTERIM_QUALITY_MODE
    assert report["quality"]["baseline_only_wins"] == 1
    assert report["quality"]["treatment_only_wins"] == 0
    assert report["quality"]["point_delta"] == -0.125
    assert report["quality"]["upper_bound"] >= gate.QUALITY_DELTA_MIN
    criterion = next(
        item for item in report["criteria"] if item["name"] == "paired_quality_upper_bound"
    )
    assert criterion["passed"] is True


def test_interim_quality_screen_kills_repeated_severe_losses() -> None:
    report = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70,) * 8,
            treatment_calls=(5,) * 8,
            treatment_scores=(0.0,) * 8,
        ),
        gate="promotion",
    )

    assert report["decision"] == "kill"
    assert report["quality"]["baseline_only_wins"] == 8
    assert report["quality"]["treatment_only_wins"] == 0
    assert report["quality"]["upper_bound"] < gate.QUALITY_DELTA_MIN
    criterion = next(
        item for item in report["criteria"] if item["name"] == "paired_quality_upper_bound"
    )
    assert criterion["passed"] is False


def _full_protocol_metrics(*, lost_questions: int) -> tuple[gate.TaskMetrics, ...]:
    pair_ids: list[str] = []
    treatment_scores: list[float] = []
    for question in range(gate.demo.SERGEI_TEST_COUNT):
        for repeat in range(1, gate.demo.FULL_REPEATS + 1):
            pair_ids.append(f"question-{question:03d}-r{repeat:03d}")
            treatment_scores.append(0.0 if question < lost_questions else 1.0)
    pair_count = len(pair_ids)
    return _metrics(
        treatment_tokens=(70,) * pair_count,
        treatment_calls=(5,) * pair_count,
        treatment_scores=tuple(treatment_scores),
        pair_ids=tuple(pair_ids),
    )


def test_confirmatory_scope_requires_lower_bound_to_meet_margin() -> None:
    parity_metrics = _full_protocol_metrics(lost_questions=0)
    parity = gate.build_gate_report(
        parity_metrics,
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=len(parity_metrics),
    )

    assert parity["decision"] == "promote_to_next_cohort"
    assert parity["performance_eligible"] is True
    assert parity["quality"]["mode"] == gate.CONFIRMATORY_QUALITY_MODE
    assert parity["quality"]["method"] == ("question-clustered-normal-small-sample-corrected")
    assert parity["quality"]["question_cluster_count"] == gate.demo.SERGEI_TEST_COUNT
    assert parity["quality"]["lower_bound"] >= gate.QUALITY_DELTA_MIN

    loss_metrics = _full_protocol_metrics(lost_questions=3)
    losses = gate.build_gate_report(
        loss_metrics,
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=len(loss_metrics),
    )

    assert losses["benefit"]["mean_score_delta"] > gate.QUALITY_DELTA_MIN
    assert losses["quality"]["lower_bound"] < gate.QUALITY_DELTA_MIN
    assert losses["quality"]["baseline_only_wins"] == 3 * gate.demo.FULL_REPEATS
    assert losses["decision"] == "kill"
    assert losses["performance_eligible"] is False
    criterion = next(
        item for item in losses["criteria"] if item["name"] == "paired_quality_lower_bound"
    )
    assert criterion["passed"] is False


def test_explicit_quarantined_primary_scope_controls_confirmatory_mode() -> None:
    primary_scope = {
        "question_start": 8,
        "question_count": 4,
        "repeat_count": 2,
        "task_count": 16,
        "question_group_keys_sha256": "sha256:" + "2" * 64,
    }
    pair_ids = tuple(
        f"trialqa-{question:04d}-hash-r{repeat:03d}"
        for question in range(8, 12)
        for repeat in range(1, 3)
    )
    metrics = _metrics(
        treatment_tokens=(70,) * 8,
        treatment_calls=(5,) * 8,
        treatment_scores=(1.0,) * 8,
        pair_ids=pair_ids,
    )

    interim = gate.build_gate_report(
        metrics[:8],
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=16,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )
    assert interim["quality"]["mode"] == gate.INTERIM_QUALITY_MODE
    assert interim["scope"]["confirmatory_scope_complete"] is False

    confirmatory = gate.build_gate_report(
        metrics,
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=16,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )
    assert confirmatory["quality"]["mode"] == gate.CONFIRMATORY_QUALITY_MODE
    assert confirmatory["performance_eligible"] is True
    assert confirmatory["scope"]["confirmatory_scope_complete"] is True
    assert confirmatory["scope"]["primary_evaluation_scope"] == primary_scope


def test_prospective_q0_q7_r5_scope_is_first_confirmatory_completion() -> None:
    groups = [f"trialqa-{index:04d}-prospective" for index in range(8)]
    primary_scope = {
        "question_start": 0,
        "question_count": 8,
        "repeat_count": 5,
        "task_count": 80,
        "question_group_keys_sha256": gate.demo._sha256_bytes(
            gate.demo._canonical_json(groups)
        ),
    }
    partial_pair_ids = tuple(
        f"{group}-r{repeat:03d}" for group in groups for repeat in range(1, 4)
    )
    complete_pair_ids = tuple(
        f"{group}-r{repeat:03d}" for group in groups for repeat in range(1, 6)
    )

    partial = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70,) * len(partial_pair_ids),
            treatment_calls=(5,) * len(partial_pair_ids),
            treatment_scores=(1.0,) * len(partial_pair_ids),
            pair_ids=partial_pair_ids,
        ),
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )
    assert partial["decision"] == "promote_to_next_cohort"
    assert partial["quality"]["mode"] == gate.INTERIM_QUALITY_MODE
    assert partial["scope"]["confirmatory_scope_complete"] is False
    assert partial["quality"]["pair_count"] == 24
    assert partial["quality"]["question_cluster_count"] == 8

    complete = gate.build_gate_report(
        _metrics(
            treatment_tokens=(70,) * len(complete_pair_ids),
            treatment_calls=(5,) * len(complete_pair_ids),
            treatment_scores=(1.0,) * len(complete_pair_ids),
            pair_ids=complete_pair_ids,
        ),
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )

    assert complete["decision"] == "promote_to_next_cohort"
    assert complete["quality"]["mode"] == gate.CONFIRMATORY_QUALITY_MODE
    assert complete["scope"]["confirmatory_scope_complete"] is True
    assert complete["quality"]["pair_count"] == 40
    assert complete["quality"]["question_cluster_count"] == 8
    assert complete["efficiency_checkpoint"]["final_thresholds_enforced"] is True
    assert complete["performance_eligible"] is True


def test_explicit_primary_scope_disagreement_fails_closed() -> None:
    metrics = _metrics(
        treatment_tokens=(70, 70, 70, 70),
        treatment_calls=(5, 5, 5, 5),
        treatment_scores=(1.0, 1.0, 1.0, 1.0),
    )
    primary_scope = {
        "question_start": 8,
        "question_count": 4,
        "repeat_count": 1,
        "task_count": 8,
        "question_group_keys_sha256": "sha256:" + "3" * 64,
    }

    with pytest.raises(gate.TrialQAGateError, match="disagrees with manifest"):
        gate.build_gate_report(
            metrics,
            gate="promotion",
            manifest_kind="full",
            manifest_task_count=10,
            primary_evaluation_scope=primary_scope,
            manifest_performance_eligible=True,
        )
    malformed = {**primary_scope, "task_count": 10}
    with pytest.raises(gate.TrialQAGateError, match=r"question_count \* repeat_count \* 2"):
        gate.build_gate_report(
            metrics,
            gate="promotion",
            manifest_kind="full",
            manifest_task_count=10,
            primary_evaluation_scope=malformed,
            manifest_performance_eligible=True,
        )

    wrong_repeats = _metrics(
        treatment_tokens=(70, 70, 70, 70),
        treatment_calls=(5, 5, 5, 5),
        treatment_scores=(1.0, 1.0, 1.0, 1.0),
        pair_ids=tuple(f"trialqa-0008-hash-r{repeat:03d}" for repeat in range(1, 5)),
    )
    with pytest.raises(gate.TrialQAGateError, match="outside primary evaluation"):
        gate.build_gate_report(
            wrong_repeats,
            gate="promotion",
            manifest_kind="full",
            manifest_task_count=8,
            primary_evaluation_scope=primary_scope,
            manifest_performance_eligible=True,
        )


def test_explicit_descriptive_full_manifest_cannot_become_confirmatory() -> None:
    metrics = _full_protocol_metrics(lost_questions=0)

    report = gate.build_gate_report(
        metrics,
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=len(metrics),
        manifest_performance_eligible=False,
    )

    assert report["quality"]["mode"] == gate.INTERIM_QUALITY_MODE
    assert report["scope"]["confirmatory_scope_complete"] is False
    assert report["scope"]["manifest_performance_eligible"] is False
    assert report["performance_eligible"] is False


def test_quality_bounds_reject_nonbinary_scores() -> None:
    metrics = _metrics(
        treatment_tokens=(70, 70, 70, 70),
        treatment_calls=(5, 5, 5, 5),
        treatment_scores=(1.0, 1.0, 1.0, 0.5),
    )

    with pytest.raises(gate.TrialQAGateError, match="paired binary scores"):
        gate.build_gate_report(metrics, gate="promotion")


def test_gate_rejects_incomplete_pair() -> None:
    metrics = _metrics(treatment_tokens=(70,), treatment_calls=(5,))

    with pytest.raises(gate.TrialQAGateError, match="incomplete pair"):
        gate.build_gate_report(metrics[:-1], gate="plumbing")
