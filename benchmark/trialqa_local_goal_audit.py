# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goal-level audit for the local TrialQA skill-distillation demo.

This command aggregates the no-spend evidence that says the staged workflow is
ready for the next guarded spend boundary, while explicitly marking the live
generation, quality-parity, and efficiency-benefit evidence that is still
missing. It is intentionally read-only and never authorizes spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_gate as gate  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_goal_audit.v1"
JsonObject = dict[str, Any]
RequirementStatus = str
SETUP_EVIDENCE_REQUIREMENT_IDS = {
    "prospective_manifest_bound",
    "reference_workflow_alignment_bound",
    "local_switchyard_trialqa_transfer_runtime_bound",
    "switchyard_only_skill_distillation_ab_invariant_bound",
    "frozen_promotion_kill_policy_bound",
    "human_spend_review_packet_ready",
}


class TrialQAGoalAuditError(RuntimeError):
    """Goal-audit inputs are malformed or stale."""


@dataclass(frozen=True)
class GoalAuditConfig:
    manifest: Path
    reference_targets: Path
    reference_alignment: Path
    ladder_rehearsal: Path
    preflight: Path
    protocol_audit: Path
    spend_review: Path
    operational_gate: Path | None = None
    promotion_gate: Path | None = None


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQAGoalAuditError(f"{label} must be an object")
    return value


def _require_schema(report: Mapping[str, object], schema: str, label: str) -> None:
    if report.get("schema_version") != schema:
        raise TrialQAGoalAuditError(f"{label} has invalid schema_version")


def _requirement(
    requirement_id: str,
    status: RequirementStatus,
    evidence: str,
    *,
    required_for_completion: bool = True,
) -> JsonObject:
    return {
        "id": requirement_id,
        "status": status,
        "evidence": evidence,
        "required_for_completion": required_for_completion,
    }


def _manifest_requirement(manifest: Mapping[str, object]) -> JsonObject:
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    primary = _require_mapping(
        protocol.get("primary_evaluation_scope"),
        "manifest primary_evaluation_scope",
    )
    ok = (
        dataset.get("official_labbench2") is False
        and primary.get("question_start") == 0
        and primary.get("question_count") == 8
        and primary.get("repeat_count") == 5
        and primary.get("task_count") == 80
    )
    return _requirement(
        "prospective_manifest_bound",
        "proved" if ok else "failed",
        f"manifest primary scope is {dict(primary)} and official_labbench2={dataset.get('official_labbench2')!r}",
    )


def _reference_requirement(reference: Mapping[str, object]) -> JsonObject:
    population = _require_mapping(reference.get("population"), "reference population")
    super_targets = _require_mapping(reference.get("super"), "reference super targets")
    super_r1 = _require_mapping(super_targets.get("r1"), "reference super r1 targets")
    ok = (
        population.get("trials") == 480
        and population.get("heldout_questions") == 96
        and population.get("repeats_per_question") == 5
        and isinstance(super_r1.get("accuracy"), (int, float))
        and isinstance(super_r1.get("token_reduction"), (int, float))
        and isinstance(super_r1.get("operational_call_reduction"), (int, float))
    )
    return _requirement(
        "reference_targets_bound",
        "proved" if ok else "failed",
        f"reference population is {dict(population)} and Super R1 is {dict(super_r1)}",
    )


def _reference_alignment_requirement(alignment: Mapping[str, object]) -> JsonObject:
    requirements = alignment.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQAGoalAuditError("reference alignment has invalid requirements")
    current_scope = alignment.get("current_scope")
    if not isinstance(current_scope, Mapping):
        current_scope = {}
    reference_scope = alignment.get("reference_scope")
    if not isinstance(reference_scope, Mapping):
        reference_scope = {}
    official_requirement = None
    workflow_requirement = None
    for item in requirements:
        if isinstance(item, Mapping) and item.get("id") == "official_96_question_reproduction_bound":
            official_requirement = item
        if isinstance(item, Mapping) and item.get("id") == "reference_workflow_source_evidence_bound":
            workflow_requirement = item
            break
    current_paired_tasks = current_scope.get("paired_tasks")
    reference_paired_tasks = reference_scope.get("paired_tasks")
    ok = (
        alignment.get("canary_alignment_status") == "proved"
        and alignment.get("claim_scope")
        in {"prospective_transfer_canary", "official_labbench2_reproduction"}
        and current_paired_tasks == 80
        and reference_paired_tasks == 960
        and isinstance(official_requirement, Mapping)
        and official_requirement.get("required_for_official_reproduction") is True
        and official_requirement.get("required_for_canary") is False
        and isinstance(workflow_requirement, Mapping)
        and workflow_requirement.get("status") == "proved"
    )
    return _requirement(
        "reference_workflow_alignment_bound",
        "proved" if ok else "failed",
        (
            f"canary_alignment={alignment.get('canary_alignment_status')!r}, "
            f"official_reproduction={alignment.get('official_reproduction_status')!r}, "
            f"claim_scope={alignment.get('claim_scope')!r}, "
            f"current_paired_tasks={current_paired_tasks!r}, "
            f"official_paired_tasks={reference_paired_tasks!r}, "
            f"reference_workflow_source={workflow_requirement.get('status') if isinstance(workflow_requirement, Mapping) else None!r}"
        ),
    )


def _rehearsal_requirement(rehearsal: Mapping[str, object]) -> JsonObject:
    budget = rehearsal.get("ladder_budget")
    first_boundary = None
    budget_ok = False
    max_model_calls = None
    max_judge_calls = None
    max_total_calls = None
    if isinstance(budget, Mapping):
        first_boundary = budget.get("first_spend_boundary")
        max_model_calls = budget.get("max_model_calls_before_directional_completion")
        max_judge_calls = budget.get("max_judge_calls_before_directional_completion")
        max_total_calls = budget.get("max_total_live_calls_before_directional_completion")
        budget_ok = (
            isinstance(first_boundary, Mapping)
            and first_boundary.get("stage") == "generation"
            and first_boundary.get("expected_model_calls") == 8
            and first_boundary.get("expected_judge_calls") == 0
            and max_model_calls == 80
            and max_judge_calls == 80
            and max_total_calls == 160
        )
    ok = (
        rehearsal.get("status") == "passed"
        and rehearsal.get("scenario_count") == 8
        and rehearsal.get("failed_scenario_count") == 0
        and rehearsal.get("model_calls") == 0
        and rehearsal.get("judge_calls") == 0
        and budget_ok
    )
    return _requirement(
        "staged_ladder_rehearsed",
        "proved" if ok else "failed",
        (
            f"rehearsal status={rehearsal.get('status')!r}, "
            f"scenarios={rehearsal.get('scenario_count')!r}, "
            f"failed={rehearsal.get('failed_scenario_count')!r}, "
            f"model_calls={rehearsal.get('model_calls')!r}, "
            f"judge_calls={rehearsal.get('judge_calls')!r}, "
            f"budget={{'first_model_calls': "
            f"{first_boundary.get('expected_model_calls') if isinstance(first_boundary, Mapping) else None!r}, "
            f"'first_judge_calls': "
            f"{first_boundary.get('expected_judge_calls') if isinstance(first_boundary, Mapping) else None!r}, "
            f"'max_model_calls': {max_model_calls!r}, "
            f"'max_judge_calls': {max_judge_calls!r}, "
            f"'max_total_calls': {max_total_calls!r}}}"
        ),
    )


def _spend_review_stage(spend_review: Mapping[str, object]) -> str:
    scope_value = spend_review.get("guarded_spend_scope")
    if not isinstance(scope_value, Mapping):
        return "generation"
    scope = _require_mapping(scope_value, "guarded_spend_scope")
    stage = scope.get("stage")
    return str(stage) if stage in {"generation", "score"} else "generation"


def _preflight_requirement(preflight: Mapping[str, object], *, spend_stage: str) -> JsonObject:
    next_command = _require_mapping(preflight.get("next_command"), "preflight next_command")
    if spend_stage == "generation":
        expected_schema = "switchyard.trialqa_no_spend_preflight.v1"
        expected_bundle_state = "awaiting_generation_canary_spend_authorization"
        expected_command_kind = "guarded_generation_canary"
        requirement_id = "generation_spend_boundary_preflight_passed"
    elif spend_stage == "score":
        expected_schema = "switchyard.trialqa_no_spend_score_preflight.v1"
        expected_bundle_state = "awaiting_score_canary_spend_authorization"
        expected_command_kind = "guarded_score_canary"
        requirement_id = "score_spend_boundary_preflight_passed"
    else:  # pragma: no cover - guarded by _spend_review_stage.
        raise TrialQAGoalAuditError(f"unsupported spend stage {spend_stage!r}")
    ok = (
        preflight.get("schema_version") == expected_schema
        and preflight.get("status") == "passed"
        and preflight.get("spend_authorized") is False
        and preflight.get("bundle_state") == expected_bundle_state
        and next_command.get("kind") == expected_command_kind
        and next_command.get("requires_yes_spend") is True
    )
    return _requirement(
        requirement_id,
        "proved" if ok else "failed",
        (
            f"spend_stage={spend_stage!r}, "
            f"preflight_schema={preflight.get('schema_version')!r}, "
            f"preflight status={preflight.get('status')!r}, "
            f"bundle_state={preflight.get('bundle_state')!r}, "
            f"next_command={next_command.get('kind')!r}"
        ),
    )


def _gate_policy_requirement(primary_scope: Mapping[str, object]) -> JsonObject:
    expected_policy = {
        "name": "ultra-efficiency-v3",
        "token_reduction_min": 0.15,
        "operational_call_reduction_min": 0.20,
        "quality_delta_min": -0.05,
        "quality_confidence_level": 0.95,
        "futility_confidence_level": 0.95,
        "interim_quality_mode": "interim_harm_screen",
        "confirmatory_quality_mode": "confirmatory_noninferiority",
    }
    observed_policy = {
        "name": gate.POLICY_NAME,
        "token_reduction_min": gate.TOKEN_REDUCTION_MIN,
        "operational_call_reduction_min": gate.OPERATIONAL_CALL_REDUCTION_MIN,
        "quality_delta_min": gate.QUALITY_DELTA_MIN,
        "quality_confidence_level": gate.QUALITY_CONFIDENCE_LEVEL,
        "futility_confidence_level": gate.FUTILITY_CONFIDENCE_LEVEL,
        "interim_quality_mode": gate.INTERIM_QUALITY_MODE,
        "confirmatory_quality_mode": gate.CONFIRMATORY_QUALITY_MODE,
    }

    early_metrics = _synthetic_gate_metrics(
        pair_ids=tuple(f"trialqa-{index:04d}-prospective-r001" for index in range(4)),
        treatment_tokens=100,
        treatment_calls=10,
        scored=False,
    )
    early_report = gate.build_gate_report(
        early_metrics,
        gate="operational",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )
    complete_pair_ids = tuple(
        f"trialqa-{question:04d}-prospective-r{repeat:03d}"
        for question in range(8)
        for repeat in range(1, 6)
    )
    complete_report = gate.build_gate_report(
        _synthetic_gate_metrics(
            pair_ids=complete_pair_ids,
            treatment_tokens=70,
            treatment_calls=5,
            scored=True,
        ),
        gate="promotion",
        manifest_kind="full",
        manifest_task_count=80,
        primary_evaluation_scope=primary_scope,
        manifest_performance_eligible=True,
    )
    early_failed = {str(item["name"]) for item in early_report["criteria"] if not item["passed"]}
    ok = (
        observed_policy == expected_policy
        and early_report.get("decision") == "kill"
        and early_report.get("performance_eligible") is False
        and {
            "early_aggregate_token_reduction_positive",
            "early_aggregate_operational_call_reduction_positive",
        }.issubset(early_failed)
        and complete_report.get("decision") == "promote_to_next_cohort"
        and complete_report.get("performance_eligible") is True
        and _require_mapping(complete_report.get("quality"), "complete synthetic quality").get(
            "mode"
        )
        == gate.CONFIRMATORY_QUALITY_MODE
        and _require_mapping(
            complete_report.get("efficiency_checkpoint"),
            "complete synthetic efficiency checkpoint",
        ).get("final_thresholds_enforced")
        is True
    )
    return _requirement(
        "frozen_promotion_kill_policy_bound",
        "proved" if ok else "failed",
        (
            f"policy={observed_policy}, early_no_benefit_decision="
            f"{early_report.get('decision')!r}, complete_scope_decision="
            f"{complete_report.get('decision')!r}"
        ),
    )


def _synthetic_gate_metrics(
    *,
    pair_ids: Sequence[str],
    treatment_tokens: int,
    treatment_calls: int,
    scored: bool,
) -> tuple[gate.TaskMetrics, ...]:
    metrics: list[gate.TaskMetrics] = []
    for pair_id in pair_ids:
        metrics.extend(
            [
                gate.TaskMetrics(
                    task_id=f"{pair_id}-baseline",
                    pair_id=pair_id,
                    condition="baseline",
                    total_tokens=100,
                    operational_calls=10,
                    successful_operational_calls=10,
                    skill_load_calls=0,
                    successful_skill_load_calls=0,
                    model_turns=5,
                    score=1.0 if scored else None,
                ),
                gate.TaskMetrics(
                    task_id=f"{pair_id}-treatment",
                    pair_id=pair_id,
                    condition="treatment",
                    total_tokens=treatment_tokens,
                    operational_calls=treatment_calls,
                    successful_operational_calls=treatment_calls,
                    skill_load_calls=1,
                    successful_skill_load_calls=1,
                    model_turns=4,
                    score=1.0 if scored else None,
                ),
            ]
        )
    return tuple(metrics)


def _protocol_skill_invariant_requirement(
    protocol_audit: Mapping[str, object],
    *,
    spend_stage: str,
) -> JsonObject:
    requirements = protocol_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQAGoalAuditError("protocol audit has invalid requirements")
    invariant = None
    for item in requirements:
        if isinstance(item, Mapping) and item.get("id") == "skill_distillation_ab_invariant_bound":
            invariant = item
            break
    invariant_evidence = str(invariant.get("evidence")) if isinstance(invariant, Mapping) else ""
    expected_state = (
        "awaiting_generation_canary_spend_authorization"
        if spend_stage == "generation"
        else "awaiting_score_canary_spend_authorization"
    )
    ok = (
        protocol_audit.get("completion_state") == expected_state
        and isinstance(invariant, Mapping)
        and invariant.get("status") == "proved"
        and invariant.get("required_for_spend") is True
        and "concurrent-paired-same-executor-skill-only" in invariant_evidence
        and "nvidia/nvidia/nemotron-3-ultra" in invariant_evidence
        and "'skill_loaded': False" in invariant_evidence
        and "'skill_loaded': True" in invariant_evidence
    )
    return _requirement(
        "switchyard_only_skill_distillation_ab_invariant_bound",
        "proved" if ok else "failed",
        (
            f"protocol_state={protocol_audit.get('completion_state')!r}, "
            f"expected_state={expected_state!r}, "
            f"invariant_status={invariant.get('status') if isinstance(invariant, Mapping) else None!r}, "
            "requires same Switchyard Ultra executor with baseline skill_loaded=False "
            "and treatment skill_loaded=True"
        ),
    )


def _protocol_local_runtime_requirement(protocol_audit: Mapping[str, object]) -> JsonObject:
    requirements = protocol_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQAGoalAuditError("protocol audit has invalid requirements")
    runtime = None
    for item in requirements:
        if (
            isinstance(item, Mapping)
            and item.get("id") == "local_switchyard_trialqa_transfer_runtime_bound"
        ):
            runtime = item
            break
    runtime_evidence = str(runtime.get("evidence")) if isinstance(runtime, Mapping) else ""
    ok = (
        isinstance(runtime, Mapping)
        and runtime.get("status") == "proved"
        and runtime.get("required_for_spend") is True
        and "container-free local Switchyard transfer workflow" in runtime_evidence
        and "trialqa-compatible-prospective" in runtime_evidence
        and "clinicaltrials-gov" in runtime_evidence
        and "benchmark/trialqa_local_batch.py" in runtime_evidence
    )
    return _requirement(
        "local_switchyard_trialqa_transfer_runtime_bound",
        "proved" if ok else "failed",
        (
            f"runtime_status={runtime.get('status') if isinstance(runtime, Mapping) else None!r}, "
            "requires local Switchyard TrialQA-compatible prospective parquet, "
            "not Docker or a second Hugging Face runtime repository"
        ),
    )


def _spend_review_requirement(spend_review: Mapping[str, object]) -> JsonObject:
    guarded = _require_mapping(spend_review.get("guarded_spend_command"), "guarded spend command")
    safe = _require_mapping(spend_review.get("safe_no_spend_command"), "safe no-spend command")
    verification = _require_mapping(
        spend_review.get("bundle_verification"),
        "spend-review bundle_verification",
    )
    monitor_value = spend_review.get("progress_monitor_command")
    monitor = monitor_value if isinstance(monitor_value, Mapping) else {}
    recovery_value = spend_review.get("guarded_recovery_command")
    recovery = recovery_value if isinstance(recovery_value, Mapping) else {}
    checkpoint_value = spend_review.get("post_spend_checkpoint_command")
    checkpoint = checkpoint_value if isinstance(checkpoint_value, Mapping) else {}
    acceptance_value = spend_review.get("post_spend_acceptance_criteria")
    acceptance = acceptance_value if isinstance(acceptance_value, Mapping) else {}
    policy_value = spend_review.get("decision_policy")
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    current_progress_value = spend_review.get("current_progress_verification")
    current_progress = (
        current_progress_value if isinstance(current_progress_value, Mapping) else {}
    )
    scope = spend_review.get("guarded_spend_scope")
    guarded_command = guarded.get("command")
    safe_command = safe.get("command")
    monitor_command = monitor.get("command")
    recovery_command = recovery.get("command")
    checkpoint_command = checkpoint.get("command")
    source_file_check_count = verification.get("source_file_check_count")
    scope_ok = False
    monitor_ok = False
    recovery_ok = False
    checkpoint_ok = False
    acceptance_ok = False
    policy_ok = False
    current_progress_ok = False
    scope_stage = None
    scope_task_count = None
    expected_model_calls = None
    expected_judge_calls = None
    if isinstance(scope, Mapping):
        scope_stage = scope.get("stage")
        scope_task_count = scope.get("task_count")
        expected_model_calls = scope.get("expected_model_calls")
        expected_judge_calls = scope.get("expected_judge_calls")
        scope_ok = (
            scope_stage in {"generation", "score"}
            and isinstance(scope.get("question_start"), int)
            and isinstance(scope.get("question_limit"), int)
            and scope.get("question_limit", 0) > 0
            and isinstance(scope.get("repeat_limit"), int)
            and scope.get("repeat_limit", 0) > 0
            and isinstance(scope_task_count, int)
            and scope_task_count > 0
            and isinstance(expected_model_calls, int)
            and isinstance(expected_judge_calls, int)
            and (
                (
                    scope_stage == "generation"
                    and expected_model_calls == scope_task_count
                    and expected_judge_calls == 0
                )
                or (
                    scope_stage == "score"
                    and expected_model_calls == 0
                    and expected_judge_calls == scope_task_count
                )
            )
        )
        expected_checkpoint_kind = (
            "post_generation_checkpoint" if scope_stage == "generation" else "post_score_checkpoint"
        )
        expected_post_spend_gate = "operational" if scope_stage == "generation" else "promotion"
        expected_promote_decision = (
            "promote_to_score" if scope_stage == "generation" else "promote_to_next_cohort"
        )
        expected_judge_deferred = scope_stage == "generation"
        expected_next_boundary = (
            "score_spend_review"
            if scope_stage == "generation"
            else "generation_expansion_spend_review_or_complete"
        )
        if isinstance(monitor_command, list):
            monitor_ok = (
                monitor.get("stage") == scope_stage
                and monitor.get("requires_spend") is False
                and monitor.get("contains_yes_spend") is False
                and "--yes-spend" not in monitor_command
            )
        if isinstance(recovery_command, list):
            recovery_ok = (
                recovery.get("stage") == scope_stage
                and recovery.get("requires_yes_spend") is True
                and recovery.get("authorized_by_packet") is False
                and recovery_command[-2:] == ["--recover-interrupted", "--yes-spend"]
            )
        if isinstance(checkpoint_command, list):
            checkpoint_ok = (
                checkpoint.get("kind") == expected_checkpoint_kind
                and checkpoint.get("requires_spend") is False
                and checkpoint.get("contains_yes_spend") is False
                and "--yes-spend" not in checkpoint_command
            )
        required_gate_artifact = acceptance.get("required_gate_artifact")
        acceptance_ok = (
            acceptance.get("stage") == scope_stage
            and acceptance.get("required_gate") == expected_post_spend_gate
            and isinstance(required_gate_artifact, str)
            and bool(required_gate_artifact)
            and acceptance.get("required_gate_schema_version")
            == "switchyard.trialqa_gate_report.v3"
            and acceptance.get("promote_decision") == expected_promote_decision
            and acceptance.get("kill_decision") == "kill"
            and acceptance.get("next_no_spend_checkpoint_kind") == expected_checkpoint_kind
            and acceptance.get("checkpoint_command_available") is True
            and acceptance.get("must_run_checkpoint_before_more_spend") is True
            and acceptance.get("next_boundary_if_promoted") == expected_next_boundary
            and (
                (
                    scope_stage == "generation"
                    and acceptance.get("judge_spend_before_checkpoint_allowed") is False
                )
                or (
                    scope_stage == "score"
                    and acceptance.get("model_spend_before_checkpoint_allowed") is False
                )
            )
        )
        thresholds_value = policy.get("thresholds")
        thresholds = thresholds_value if isinstance(thresholds_value, Mapping) else {}
        boundary_value = policy.get("current_boundary")
        boundary = boundary_value if isinstance(boundary_value, Mapping) else {}
        policy_ok = (
            policy.get("name") == gate.POLICY_NAME
            and thresholds.get("token_reduction_min") == gate.TOKEN_REDUCTION_MIN
            and thresholds.get("operational_call_reduction_min")
            == gate.OPERATIONAL_CALL_REDUCTION_MIN
            and thresholds.get("quality_delta_min") == gate.QUALITY_DELTA_MIN
            and boundary.get("stage") == scope_stage
            and boundary.get("post_spend_gate") == expected_post_spend_gate
            and boundary.get("promote_decision") == expected_promote_decision
            and boundary.get("kill_decision") == "kill"
            and boundary.get("judge_spend_deferred") is expected_judge_deferred
        )
        current_progress_ok = (
            current_progress.get("status") == "matched"
            and current_progress.get("stage") == scope_stage
            and current_progress.get("requires_spend") is True
            and current_progress.get("selected_task_count") == scope_task_count
            and isinstance(current_progress.get("done_task_count"), int)
            and isinstance(current_progress.get("remaining_task_count"), int)
        )
    ok = (
        spend_review.get("status") == "ready_for_user_spend_decision"
        and spend_review.get("authorized_by_packet") is False
        and isinstance(guarded_command, list)
        and guarded_command[-1:] == ["--yes-spend"]
        and guarded.get("requires_yes_spend") is True
        and isinstance(safe_command, list)
        and "--yes-spend" not in safe_command
        and isinstance(source_file_check_count, int)
        and source_file_check_count >= 20
        and scope_ok
        and monitor_ok
        and recovery_ok
        and checkpoint_ok
        and acceptance_ok
        and policy_ok
        and current_progress_ok
    )
    return _requirement(
        "human_spend_review_packet_ready",
        "proved" if ok else "failed",
        (
            f"packet status={spend_review.get('status')!r}, "
            f"authorized={spend_review.get('authorized_by_packet')!r}, "
            f"source_checks={source_file_check_count!r}, "
            f"spend_scope={{'stage': {scope_stage!r}, 'task_count': {scope_task_count!r}, "
            f"'expected_model_calls': {expected_model_calls!r}, "
            f"'expected_judge_calls': {expected_judge_calls!r}}}, "
            f"monitor_ok={monitor_ok!r}, recovery_ok={recovery_ok!r}, "
            f"checkpoint_ok={checkpoint_ok!r}, acceptance_ok={acceptance_ok!r}, "
            f"policy_ok={policy_ok!r}, "
            f"current_progress_ok={current_progress_ok!r}"
        ),
    )


def _final_primary_scope_complete(
    gate: Mapping[str, object],
    *,
    gate_name: str,
    primary_scope: Mapping[str, object],
) -> tuple[bool, str]:
    scope = _require_mapping(gate.get("scope"), f"{gate_name} gate scope")
    attestation = _require_mapping(
        scope.get("selection_attestation"),
        f"{gate_name} gate selection_attestation",
    )
    primary_question_start = primary_scope.get("question_start")
    primary_question_count = primary_scope.get("question_count")
    primary_repeat_count = primary_scope.get("repeat_count")
    primary_task_count = primary_scope.get("task_count")
    repeats = attestation.get("selected_repeat_indices")
    expected_repeats = (
        list(range(1, primary_repeat_count + 1))
        if isinstance(primary_repeat_count, int)
        else None
    )
    selected_task_count = attestation.get("selected_task_count")
    complete_by_attestation = (
        attestation.get("question_start") == primary_question_start
        and attestation.get("question_limit") == primary_question_count
        and attestation.get("selected_question_count") == primary_question_count
        and repeats == expected_repeats
        and selected_task_count == primary_task_count
    )
    complete_by_scope = (
        scope.get("confirmatory_scope_complete") is True
        and (gate_name != "promotion" or gate.get("performance_eligible") is True)
    )
    evidence = (
        f"selection_attestation={dict(attestation)}, "
        f"confirmatory_scope_complete={scope.get('confirmatory_scope_complete')!r}, "
        f"performance_eligible={gate.get('performance_eligible')!r}, "
        f"primary_scope={dict(primary_scope)}"
    )
    return bool(complete_by_attestation and complete_by_scope), evidence


def _optional_gate_requirement(
    *,
    gate_path: Path | None,
    gate_name: str,
    requirement_id: str,
    missing_evidence: str,
    manifest_id: str,
    primary_scope: Mapping[str, object] | None = None,
) -> JsonObject:
    if gate_path is None:
        return _requirement(requirement_id, "missing", missing_evidence)
    gate = demo._read_json_object(gate_path, f"{gate_name} gate")
    _require_schema(gate, "switchyard.trialqa_gate_report.v3", f"{gate_name} gate")
    if gate.get("gate") != gate_name:
        return _requirement(
            requirement_id,
            "failed",
            f"expected {gate_name!r} gate, got {gate.get('gate')!r}",
        )
    if gate.get("manifest_id") != manifest_id:
        return _requirement(
            requirement_id,
            "failed",
            f"{gate_name} gate belongs to manifest {gate.get('manifest_id')!r}",
        )
    decision = gate.get("decision")
    expected = "promote_to_score" if gate_name == "operational" else "promote_to_next_cohort"
    if decision != expected:
        return _requirement(
            requirement_id,
            "failed",
            f"{gate_name} gate decision is {decision!r}",
        )
    if primary_scope is not None:
        final_scope_complete, scope_evidence = _final_primary_scope_complete(
            gate,
            gate_name=gate_name,
            primary_scope=primary_scope,
        )
        if gate_name == "promotion":
            return _requirement(
                requirement_id,
                "proved" if final_scope_complete else "missing",
                (
                    "promotion gate proved full primary quality/efficiency evidence; "
                    if final_scope_complete
                    else "promotion gate has not yet covered the full primary scope; "
                )
                + scope_evidence,
            )
        return _requirement(
            requirement_id,
            "proved" if final_scope_complete else "missing",
            (
                "operational gate proved full primary generation evidence; "
                if final_scope_complete
                else "operational gate has not yet covered the full primary scope; "
            )
            + scope_evidence,
        )
    if gate_name == "promotion":
        if primary_scope is None:
            raise TrialQAGoalAuditError("promotion gate requirement needs primary scope")
    return _requirement(
        requirement_id,
        "proved",
        f"{gate_name} gate decision is {decision!r}",
    )


def _status(requirements: Sequence[Mapping[str, object]], *, spend_stage: str) -> str:
    required = [item for item in requirements if item.get("required_for_completion") is True]
    if any(item.get("status") == "failed" for item in required):
        return "not_ready_for_spend"
    if any(item.get("status") == "missing" for item in required):
        if spend_stage == "score":
            return "ready_for_score_spend_decision"
        return "ready_for_generation_spend_decision"
    return "complete"


def _requirement_status_counts(requirements: Sequence[Mapping[str, object]]) -> JsonObject:
    counts: JsonObject = {}
    for item in requirements:
        status = item.get("status")
        key = status if isinstance(status, str) else "unknown"
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _requirement_evidence(
    requirements: Sequence[Mapping[str, object]],
    *,
    status: str,
    selected_ids: set[str] | None = None,
) -> list[JsonObject]:
    evidence: list[JsonObject] = []
    for item in requirements:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            raise TrialQAGoalAuditError("goal requirement has invalid id")
        if selected_ids is not None and item_id not in selected_ids:
            continue
        if item.get("status") == status:
            evidence.append({"id": item_id, "evidence": item.get("evidence")})
    return evidence


def _required_ids_by_status(
    requirements: Sequence[Mapping[str, object]],
    *,
    status: str,
) -> list[str]:
    ids: list[str] = []
    for item in requirements:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            raise TrialQAGoalAuditError("goal requirement has invalid id")
        if item.get("required_for_completion") is True and item.get("status") == status:
            ids.append(item_id)
    return ids


def _next_required_action(status: str, *, spend_stage: str) -> JsonObject:
    if status == "complete":
        return {
            "action": "none",
            "requires_spend": False,
            "instruction": "All audited requirements are proved.",
        }
    if status == "not_ready_for_spend":
        return {
            "action": "repair_failed_no_spend_requirements",
            "requires_spend": False,
            "instruction": (
                "Do not run a guarded canary; repair the failed no-spend "
                "requirements and regenerate the packet."
            ),
        }
    if spend_stage == "score":
        return {
            "action": "request_explicit_score_canary_spend_approval",
            "requires_spend": True,
            "instruction": (
                "Review the current packet, then only run the guarded score "
                "canary after explicit approval for --yes-spend."
            ),
        }
    return {
        "action": "request_explicit_generation_canary_spend_approval",
        "requires_spend": True,
        "instruction": (
            "Review the current packet, then only run the guarded generation "
            "canary after explicit approval for --yes-spend."
        ),
    }


def build_goal_audit(config: GoalAuditConfig) -> JsonObject:
    """Build a read-only requirement audit for the active TrialQA demo goal."""

    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    _require_schema(manifest, "switchyard.trialqa_experiment_manifest.v1", "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str):
        raise TrialQAGoalAuditError("manifest has no manifest_id")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    primary_scope = _require_mapping(
        protocol.get("primary_evaluation_scope"),
        "manifest primary_evaluation_scope",
    )
    reference = demo._read_json_object(config.reference_targets, "reference targets")
    _require_schema(reference, "switchyard.trialqa_reference_targets.v1", "reference targets")
    alignment = demo._read_json_object(config.reference_alignment, "reference alignment")
    _require_schema(
        alignment,
        "switchyard.trialqa_reference_alignment.v1",
        "reference alignment",
    )
    rehearsal = demo._read_json_object(config.ladder_rehearsal, "ladder rehearsal")
    _require_schema(rehearsal, "switchyard.trialqa_ladder_rehearsal.v1", "ladder rehearsal")
    spend_review = demo._read_json_object(config.spend_review, "spend review")
    _require_schema(spend_review, "switchyard.trialqa_spend_review_packet.v1", "spend review")
    spend_stage = _spend_review_stage(spend_review)
    preflight = demo._read_json_object(config.preflight, "no-spend preflight")
    if preflight.get("schema_version") not in {
        "switchyard.trialqa_no_spend_preflight.v1",
        "switchyard.trialqa_no_spend_score_preflight.v1",
    }:
        raise TrialQAGoalAuditError("no-spend preflight has invalid schema_version")
    protocol_audit = demo._read_json_object(config.protocol_audit, "protocol audit")
    _require_schema(
        protocol_audit,
        "switchyard.trialqa_protocol_audit.v1",
        "protocol audit",
    )
    requirements = [
        _manifest_requirement(manifest),
        _reference_requirement(reference),
        _reference_alignment_requirement(alignment),
        _rehearsal_requirement(rehearsal),
        _preflight_requirement(preflight, spend_stage=spend_stage),
        _gate_policy_requirement(primary_scope),
        _protocol_skill_invariant_requirement(protocol_audit, spend_stage=spend_stage),
        _protocol_local_runtime_requirement(protocol_audit),
        _spend_review_requirement(spend_review),
        _optional_gate_requirement(
            gate_path=config.operational_gate,
            gate_name="operational",
            requirement_id="live_generation_operational_gate_passed",
            missing_evidence="no live operational gate exists; generation has not been bought",
            manifest_id=manifest_id,
            primary_scope=primary_scope,
        ),
        _optional_gate_requirement(
            gate_path=config.promotion_gate,
            gate_name="promotion",
            requirement_id="quality_parity_and_efficiency_gate_passed",
            missing_evidence="no live promotion gate exists; quality parity and efficiency benefit are unproven",
            manifest_id=manifest_id,
            primary_scope=primary_scope,
        ),
    ]
    status = _status(requirements, spend_stage=spend_stage)
    guarded = _require_mapping(spend_review.get("guarded_spend_command"), "guarded spend command")
    requirement_summary = {
        "total": len(requirements),
        "status_counts": _requirement_status_counts(requirements),
        "required_missing_ids": _required_ids_by_status(requirements, status="missing"),
        "required_failed_ids": _required_ids_by_status(requirements, status="failed"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "goal_status": status,
        "goal_complete": status == "complete",
        "spend_authorized": False,
        "manifest_id": manifest_id,
        "requirement_summary": requirement_summary,
        "requirements": requirements,
        "proved_setup_evidence": _requirement_evidence(
            requirements,
            status="proved",
            selected_ids=SETUP_EVIDENCE_REQUIREMENT_IDS,
        ),
        "missing_goal_evidence": _requirement_evidence(requirements, status="missing"),
        "failed_goal_evidence": _requirement_evidence(requirements, status="failed"),
        "next_required_action": _next_required_action(status, spend_stage=spend_stage),
        "next_boundary": {
            "bundle_state": spend_review.get("bundle_state"),
            "guarded_command_kind": _require_mapping(
                spend_review.get("preflight"),
                "spend-review preflight",
            ).get("next_command_kind"),
            "requires_yes_spend": guarded.get("requires_yes_spend") is True,
            "authorized_by_packet": spend_review.get("authorized_by_packet") is True,
        },
        "completion_note": (
            "No-spend workflow evidence is ready for the next guarded boundary, "
            "but the objective is incomplete until live generation and final-"
            "primary-scope scored promotion evidence prove quality parity and "
            "efficiency benefit."
            if status != "complete"
            else "All audited requirements are proved."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--reference-alignment", type=Path, required=True)
    parser.add_argument("--ladder-rehearsal", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--spend-review", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument("--promotion-gate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> GoalAuditConfig:
    return GoalAuditConfig(
        manifest=args.manifest,
        reference_targets=args.reference_targets,
        reference_alignment=args.reference_alignment,
        ladder_rehearsal=args.ladder_rehearsal,
        preflight=args.preflight,
        protocol_audit=args.protocol_audit,
        spend_review=args.spend_review,
        operational_gate=args.operational_gate,
        promotion_gate=args.promotion_gate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_goal_audit(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "not_ready_for_spend" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
