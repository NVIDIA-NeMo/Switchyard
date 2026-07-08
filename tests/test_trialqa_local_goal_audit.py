# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_goal_audit as goal_audit


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    row_ids = [f"row-{index}" for index in range(8)]
    groups = [
        f"trialqa-{index:04d}-{hashlib.sha256(row_id.encode()).hexdigest()[:12]}"
        for index, row_id in enumerate(row_ids)
    ]
    digest = demo._sha256_bytes(demo._canonical_json(groups))
    tasks = [
        {
            "task_id": f"{group}-r{repeat:03d}-{condition}",
            "pair_id": f"{group}-r{repeat:03d}",
            "row_id": row_ids[index],
            "dataset_row_index": index,
            "question_group_key": group,
            "partition": "test",
            "phase": "evaluation",
            "condition": condition,
            "arm": condition,
            "repeat_index": repeat,
            "n_repeats": 5,
        }
        for index, group in enumerate(groups)
        for repeat in range(1, 6)
        for condition in ("baseline", "treatment")
    ]
    manifest = {
        "schema_version": "switchyard.trialqa_experiment_manifest.v1",
        "kind": "full",
        "dataset": {
            "official_labbench2": False,
            "test_count": 8,
            "heldout_ordering": {
                "question_count": 8,
                "question_group_keys": groups,
                "question_group_keys_sha256": digest,
            },
        },
        "protocol": {
            "performance_eligible": True,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": 8,
                "repeat_count": 5,
                "task_count": 80,
                "question_group_keys_sha256": digest,
            },
            "heldout_quarantine": {
                "question_start": 0,
                "question_count": 0,
                "disposition": "none-new-prospective-population",
                "question_group_keys_sha256": demo._sha256_bytes(demo._canonical_json([])),
            },
            "max_generation_concurrency": 4,
        },
        "tasks": tasks,
    }
    manifest = {
        "manifest_id": f"trialqa-full-{hashlib.sha256(demo._canonical_json(manifest)).hexdigest()[:20]}",
        **manifest,
    }
    return _write_json(path, manifest)


def _reference(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_targets.v1",
            "population": {
                "trials": 480,
                "heldout_questions": 96,
                "repeats_per_question": 5,
            },
            "super": {
                "r1": {
                    "accuracy": 0.738,
                    "token_reduction": 0.3,
                    "operational_call_reduction": 0.45,
                }
            },
        },
    )


def _reference_alignment(path: Path, *, canary_status: str = "proved") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_alignment.v1",
            "canary_alignment_status": canary_status,
            "official_reproduction_status": "missing",
            "claim_scope": "prospective_transfer_canary",
            "current_scope": {
                "questions": 8,
                "repeats_per_question": 5,
                "paired_tasks": 80,
            },
            "reference_scope": {
                "questions": 96,
                "repeats_per_question": 5,
                "unpaired_trials": 480,
                "paired_tasks": 960,
            },
            "requirements": [
                {
                    "id": "official_96_question_reproduction_bound",
                    "status": "missing",
                    "required_for_canary": False,
                    "required_for_official_reproduction": True,
                    "evidence": "fixture",
                },
                {
                    "id": "reference_workflow_source_evidence_bound",
                    "status": "proved",
                    "required_for_canary": True,
                    "required_for_official_reproduction": True,
                    "evidence": "fixture",
                }
            ],
        },
    )


def _rehearsal(path: Path, *, status: str = "passed") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_ladder_rehearsal.v1",
            "status": status,
            "scenario_count": 8,
            "failed_scenario_count": 0 if status == "passed" else 1,
            "model_calls": 0,
            "judge_calls": 0,
            "ladder_budget": {
                "first_spend_boundary": {
                    "stage": "generation",
                    "expected_model_calls": 8,
                    "expected_judge_calls": 0,
                },
                "max_model_calls_before_directional_completion": 80,
                "max_judge_calls_before_directional_completion": 80,
                "max_total_live_calls_before_directional_completion": 160,
            },
        },
    )


def _preflight(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_no_spend_preflight.v1",
            "status": "passed",
            "spend_authorized": False,
            "bundle_state": "awaiting_generation_canary_spend_authorization",
            "next_command": {
                "kind": "guarded_generation_canary",
                "requires_yes_spend": True,
            },
        },
    )


def _score_preflight(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_no_spend_score_preflight.v1",
            "status": "passed",
            "spend_authorized": False,
            "bundle_state": "awaiting_score_canary_spend_authorization",
            "next_command": {
                "kind": "guarded_score_canary",
                "requires_yes_spend": True,
            },
        },
    )


def _protocol_audit(path: Path, *, invariant_status: str = "proved") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "completion_state": "awaiting_generation_canary_spend_authorization",
            "requirements": [
                {
                    "id": "skill_distillation_ab_invariant_bound",
                    "status": invariant_status,
                    "required_for_spend": True,
                    "evidence": (
                        "comparison invariant is {'design': "
                        "'concurrent-paired-same-executor-skill-only', "
                        "'shared_executor': {'model': "
                        "'nvidia/nvidia/nemotron-3-ultra'}, "
                        "'baseline_arm': {'skill_loaded': False}, "
                        "'treatment_arm': {'skill_loaded': True}}"
                    ),
                },
                {
                    "id": "local_switchyard_trialqa_transfer_runtime_bound",
                    "status": "proved",
                    "required_for_spend": True,
                    "evidence": (
                        "manifest binds container-free local Switchyard transfer "
                        "workflow with dataset={'id': "
                        "'trialqa-compatible-prospective', 'config': "
                        "'clinicaltrials-gov'} and protocol_subset="
                        "{'batch_driver': 'benchmark/trialqa_local_batch.py'}"
                    ),
                },
            ],
        },
    )


def _score_protocol_audit(path: Path, *, invariant_status: str = "proved") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "completion_state": "awaiting_score_canary_spend_authorization",
            "requirements": [
                {
                    "id": "skill_distillation_ab_invariant_bound",
                    "status": invariant_status,
                    "required_for_spend": True,
                    "evidence": (
                        "comparison invariant is {'design': "
                        "'concurrent-paired-same-executor-skill-only', "
                        "'shared_executor': {'model': "
                        "'nvidia/nvidia/nemotron-3-ultra'}, "
                        "'baseline_arm': {'skill_loaded': False}, "
                        "'treatment_arm': {'skill_loaded': True}}"
                    ),
                },
                {
                    "id": "local_switchyard_trialqa_transfer_runtime_bound",
                    "status": "proved",
                    "required_for_spend": True,
                    "evidence": (
                        "manifest binds container-free local Switchyard transfer "
                        "workflow with dataset={'id': "
                        "'trialqa-compatible-prospective', 'config': "
                        "'clinicaltrials-gov'} and protocol_subset="
                        "{'batch_driver': 'benchmark/trialqa_local_batch.py'}"
                    ),
                },
            ],
        },
    )


def _spend_review(path: Path, *, source_file_check_count: int = 20) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "bundle_state": "awaiting_generation_canary_spend_authorization",
            "preflight": {"next_command_kind": "guarded_generation_canary"},
            "bundle_verification": {
                "status": "passed",
                "source_file_check_count": source_file_check_count,
            },
            "guarded_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "requires_yes_spend": True,
            },
            "progress_monitor_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_progress",
                    "--stage",
                    "generation",
                ],
                "stage": "generation",
                "requires_spend": False,
                "contains_yes_spend": False,
            },
            "guarded_recovery_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary",
                    "--recover-interrupted",
                    "--yes-spend",
                ],
                "stage": "generation",
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "safe_no_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
            },
            "post_spend_checkpoint_command": {
                "kind": "post_generation_checkpoint",
                "command": ["python", "-m", "benchmark.trialqa_local_generation_checkpoint"],
                "requires_spend": False,
                "contains_yes_spend": False,
            },
            "post_spend_acceptance_criteria": {
                "stage": "generation",
                "required_gate": "operational",
                "required_gate_artifact": "gate-operational.json",
                "required_gate_schema_version": "switchyard.trialqa_gate_report.v3",
                "promote_decision": "promote_to_score",
                "kill_decision": "kill",
                "next_no_spend_checkpoint_kind": "post_generation_checkpoint",
                "checkpoint_command_available": True,
                "must_run_checkpoint_before_more_spend": True,
                "next_boundary_if_promoted": "score_spend_review",
                "judge_spend_before_checkpoint_allowed": False,
            },
            "decision_policy": {
                "name": "ultra-efficiency-v3",
                "thresholds": {
                    "token_reduction_min": 0.15,
                    "operational_call_reduction_min": 0.2,
                    "quality_delta_min": -0.05,
                },
                "current_boundary": {
                    "stage": "generation",
                    "post_spend_gate": "operational",
                    "promote_decision": "promote_to_score",
                    "kill_decision": "kill",
                    "judge_spend_deferred": True,
                },
            },
            "current_progress_verification": {
                "status": "matched",
                "stage": "generation",
                "requires_spend": True,
                "selected_task_count": 8,
                "done_task_count": 0,
                "remaining_task_count": 8,
            },
            "guarded_spend_scope": {
                "stage": "generation",
                "question_start": 0,
                "question_limit": 4,
                "question_stop_inclusive": 3,
                "repeat_limit": 1,
                "paired_draw_count": 4,
                "arm_count": 2,
                "task_count": 8,
                "expected_model_calls": 8,
                "expected_judge_calls": 0,
                "configured_worker_limit": 4,
                "configured_max_generation_attempts": 1,
                "maximum_generation_attempts": 8,
                "scope_label": "q0-q3, 1 repeat(s), 2 arms, 8 task(s)",
            },
        },
    )


def _score_spend_review(path: Path, *, source_file_check_count: int = 20) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "bundle_state": "awaiting_score_canary_spend_authorization",
            "preflight": {"next_command_kind": "guarded_score_canary"},
            "bundle_verification": {
                "status": "passed",
                "source_file_check_count": source_file_check_count,
            },
            "guarded_spend_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary_score",
                    "--yes-spend",
                ],
                "requires_yes_spend": True,
            },
            "progress_monitor_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_progress",
                    "--stage",
                    "score",
                ],
                "stage": "score",
                "requires_spend": False,
                "contains_yes_spend": False,
            },
            "guarded_recovery_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary_score",
                    "--recover-interrupted",
                    "--yes-spend",
                ],
                "stage": "score",
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "safe_no_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_score_preflight"],
            },
            "post_spend_checkpoint_command": {
                "kind": "post_score_checkpoint",
                "command": ["python", "-m", "benchmark.trialqa_local_score_checkpoint"],
                "requires_spend": False,
                "contains_yes_spend": False,
            },
            "post_spend_acceptance_criteria": {
                "stage": "score",
                "required_gate": "promotion",
                "required_gate_artifact": "gate-promotion.json",
                "required_gate_schema_version": "switchyard.trialqa_gate_report.v3",
                "promote_decision": "promote_to_next_cohort",
                "kill_decision": "kill",
                "next_no_spend_checkpoint_kind": "post_score_checkpoint",
                "checkpoint_command_available": True,
                "must_run_checkpoint_before_more_spend": True,
                "next_boundary_if_promoted": "generation_expansion_spend_review_or_complete",
                "model_spend_before_checkpoint_allowed": False,
            },
            "decision_policy": {
                "name": "ultra-efficiency-v3",
                "thresholds": {
                    "token_reduction_min": 0.15,
                    "operational_call_reduction_min": 0.2,
                    "quality_delta_min": -0.05,
                },
                "current_boundary": {
                    "stage": "score",
                    "post_spend_gate": "promotion",
                    "promote_decision": "promote_to_next_cohort",
                    "kill_decision": "kill",
                    "judge_spend_deferred": False,
                },
            },
            "current_progress_verification": {
                "status": "matched",
                "stage": "score",
                "requires_spend": True,
                "selected_task_count": 8,
                "done_task_count": 0,
                "remaining_task_count": 8,
            },
            "guarded_spend_scope": {
                "stage": "score",
                "question_start": 0,
                "question_limit": 4,
                "question_stop_inclusive": 3,
                "repeat_limit": 1,
                "paired_draw_count": 4,
                "arm_count": 2,
                "task_count": 8,
                "expected_model_calls": 0,
                "expected_judge_calls": 8,
                "configured_worker_limit": 4,
                "configured_max_generation_attempts": 1,
                "maximum_generation_attempts": 8,
                "scope_label": "q0-q3, 1 repeat(s), 2 arms, 8 task(s)",
            },
        },
    )


def _manifest_id(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_id = payload["manifest_id"]
    assert isinstance(manifest_id, str)
    return manifest_id


def _gate(
    path: Path,
    *,
    gate: str,
    decision: str,
    manifest_id: str,
    question_limit: int = 4,
    selected_repeat_indices: list[int] | None = None,
    confirmatory_scope_complete: bool = False,
    performance_eligible: bool = False,
) -> Path:
    repeat_indices = selected_repeat_indices or [1]
    task_count = question_limit * len(repeat_indices) * 2
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_gate_report.v3",
            "manifest_id": manifest_id,
            "gate": gate,
            "decision": decision,
            "performance_eligible": performance_eligible,
            "scope": {
                "task_count": task_count,
                "pair_count": task_count // 2,
                "confirmatory_scope_complete": confirmatory_scope_complete,
                "selection_attestation": {
                    "question_start": 0,
                    "question_limit": question_limit,
                    "selected_repeat_indices": repeat_indices,
                    "selected_question_count": question_limit,
                    "selected_task_count": task_count,
                },
            },
        },
    )


def _config(tmp_path: Path) -> goal_audit.GoalAuditConfig:
    return goal_audit.GoalAuditConfig(
        manifest=_manifest(tmp_path / "manifest.json"),
        reference_targets=_reference(tmp_path / "reference.json"),
        reference_alignment=_reference_alignment(tmp_path / "reference-alignment.json"),
        ladder_rehearsal=_rehearsal(tmp_path / "rehearsal.json"),
        preflight=_preflight(tmp_path / "preflight.json"),
        protocol_audit=_protocol_audit(tmp_path / "protocol-audit.json"),
        spend_review=_spend_review(tmp_path / "spend-review.json"),
    )


def _requirements(report: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = report["requirements"]
    assert isinstance(raw, list)
    return {str(item["id"]): item for item in raw if isinstance(item, dict)}


def test_goal_audit_marks_no_spend_ready_but_not_complete(tmp_path: Path) -> None:
    report = goal_audit.build_goal_audit(_config(tmp_path))
    requirements = _requirements(report)

    assert report["schema_version"] == goal_audit.SCHEMA_VERSION
    assert report["status"] == "ready_for_generation_spend_decision"
    assert report["goal_status"] == "ready_for_generation_spend_decision"
    assert report["goal_complete"] is False
    assert report["spend_authorized"] is False
    assert report["requirement_summary"] == {
        "total": 11,
        "status_counts": {"proved": 9, "missing": 2},
        "required_missing_ids": [
            "live_generation_operational_gate_passed",
            "quality_parity_and_efficiency_gate_passed",
        ],
        "required_failed_ids": [],
    }
    assert [item["id"] for item in report["proved_setup_evidence"]] == [
        "prospective_manifest_bound",
        "reference_workflow_alignment_bound",
        "frozen_promotion_kill_policy_bound",
        "switchyard_only_skill_distillation_ab_invariant_bound",
        "local_switchyard_trialqa_transfer_runtime_bound",
        "human_spend_review_packet_ready",
    ]
    assert [item["id"] for item in report["missing_goal_evidence"]] == [
        "live_generation_operational_gate_passed",
        "quality_parity_and_efficiency_gate_passed",
    ]
    assert report["failed_goal_evidence"] == []
    assert report["next_required_action"] == {
        "action": "request_explicit_generation_canary_spend_approval",
        "requires_spend": True,
        "instruction": (
            "Review the current packet, then only run the guarded generation "
            "canary after explicit approval for --yes-spend."
        ),
    }
    assert report["next_boundary"] == {
        "bundle_state": "awaiting_generation_canary_spend_authorization",
        "guarded_command_kind": "guarded_generation_canary",
        "requires_yes_spend": True,
        "authorized_by_packet": False,
    }
    assert requirements["prospective_manifest_bound"]["status"] == "proved"
    assert requirements["reference_targets_bound"]["status"] == "proved"
    assert requirements["reference_workflow_alignment_bound"]["status"] == "proved"
    assert "current_paired_tasks=80" in str(
        requirements["reference_workflow_alignment_bound"]["evidence"]
    )
    assert "official_paired_tasks=960" in str(
        requirements["reference_workflow_alignment_bound"]["evidence"]
    )
    assert requirements["staged_ladder_rehearsed"]["status"] == "proved"
    assert "max_total_calls" in str(requirements["staged_ladder_rehearsed"]["evidence"])
    assert requirements["generation_spend_boundary_preflight_passed"]["status"] == "proved"
    assert requirements["frozen_promotion_kill_policy_bound"]["status"] == "proved"
    assert "early_no_benefit_decision='kill'" in str(
        requirements["frozen_promotion_kill_policy_bound"]["evidence"]
    )
    assert (
        requirements["switchyard_only_skill_distillation_ab_invariant_bound"]["status"]
        == "proved"
    )
    assert (
        requirements["local_switchyard_trialqa_transfer_runtime_bound"]["status"]
        == "proved"
    )
    assert "baseline skill_loaded=False" in str(
        requirements["switchyard_only_skill_distillation_ab_invariant_bound"]["evidence"]
    )
    assert requirements["human_spend_review_packet_ready"]["status"] == "proved"
    assert "expected_model_calls" in str(
        requirements["human_spend_review_packet_ready"]["evidence"]
    )
    assert "acceptance_ok=True" in str(
        requirements["human_spend_review_packet_ready"]["evidence"]
    )
    assert requirements["live_generation_operational_gate_passed"]["status"] == "missing"
    assert requirements["quality_parity_and_efficiency_gate_passed"]["status"] == "missing"


def test_goal_audit_marks_score_spend_boundary_ready(tmp_path: Path) -> None:
    base = _config(tmp_path)
    manifest_id = _manifest_id(base.manifest)
    config = replace(
        base,
        preflight=_score_preflight(tmp_path / "score-preflight.json"),
        protocol_audit=_score_protocol_audit(tmp_path / "score-protocol-audit.json"),
        spend_review=_score_spend_review(tmp_path / "score-spend-review.json"),
        operational_gate=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "ready_for_score_spend_decision"
    assert report["goal_status"] == "ready_for_score_spend_decision"
    assert report["goal_complete"] is False
    assert report["next_required_action"]["action"] == (
        "request_explicit_score_canary_spend_approval"
    )
    assert report["next_boundary"] == {
        "bundle_state": "awaiting_score_canary_spend_authorization",
        "guarded_command_kind": "guarded_score_canary",
        "requires_yes_spend": True,
        "authorized_by_packet": False,
    }
    assert requirements["score_spend_boundary_preflight_passed"]["status"] == "proved"
    assert (
        requirements["switchyard_only_skill_distillation_ab_invariant_bound"]["status"]
        == "proved"
    )
    assert (
        requirements["local_switchyard_trialqa_transfer_runtime_bound"]["status"]
        == "proved"
    )
    assert "expected_state='awaiting_score_canary_spend_authorization'" in str(
        requirements["switchyard_only_skill_distillation_ab_invariant_bound"]["evidence"]
    )
    assert requirements["human_spend_review_packet_ready"]["status"] == "proved"
    assert "expected_judge_calls" in str(
        requirements["human_spend_review_packet_ready"]["evidence"]
    )
    assert requirements["live_generation_operational_gate_passed"]["status"] == "missing"
    assert requirements["quality_parity_and_efficiency_gate_passed"]["status"] == "missing"


def test_goal_audit_keeps_early_promoted_gate_incomplete(tmp_path: Path) -> None:
    base = _config(tmp_path)
    manifest_id = _manifest_id(base.manifest)
    config = replace(
        base,
        operational_gate=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
        ),
        promotion_gate=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "ready_for_generation_spend_decision"
    assert report["goal_complete"] is False
    assert requirements["live_generation_operational_gate_passed"]["status"] == "missing"
    assert requirements["quality_parity_and_efficiency_gate_passed"]["status"] == "missing"


def test_goal_audit_rejects_final_promotion_without_final_operational_scope(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    manifest_id = _manifest_id(base.manifest)
    config = replace(
        base,
        operational_gate=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
        ),
        promotion_gate=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
            confirmatory_scope_complete=True,
            performance_eligible=True,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "ready_for_generation_spend_decision"
    assert report["goal_complete"] is False
    assert requirements["live_generation_operational_gate_passed"]["status"] == "missing"
    assert requirements["quality_parity_and_efficiency_gate_passed"]["status"] == "proved"


def test_goal_audit_marks_complete_when_final_primary_scope_is_promoted(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    manifest_id = _manifest_id(base.manifest)
    config = replace(
        base,
        operational_gate=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
            confirmatory_scope_complete=True,
            performance_eligible=True,
        ),
        promotion_gate=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
            confirmatory_scope_complete=True,
            performance_eligible=True,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "complete"
    assert report["goal_status"] == "complete"
    assert report["goal_complete"] is True
    assert report["missing_goal_evidence"] == []
    assert report["failed_goal_evidence"] == []
    assert report["next_required_action"] == {
        "action": "none",
        "requires_spend": False,
        "instruction": "All audited requirements are proved.",
    }
    assert requirements["quality_parity_and_efficiency_gate_passed"]["status"] == "proved"


def test_goal_audit_fails_when_rehearsal_failed(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        ladder_rehearsal=_rehearsal(tmp_path / "failed-rehearsal.json", status="failed"),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert report["goal_complete"] is False
    assert report["requirement_summary"]["required_failed_ids"] == [
        "staged_ladder_rehearsed"
    ]
    assert report["next_required_action"]["action"] == "repair_failed_no_spend_requirements"
    assert requirements["staged_ladder_rehearsed"]["status"] == "failed"


def test_goal_audit_fails_when_ladder_budget_is_missing(tmp_path: Path) -> None:
    rehearsal = json.loads(_rehearsal(tmp_path / "rehearsal.json").read_text(encoding="utf-8"))
    del rehearsal["ladder_budget"]
    config = replace(
        _config(tmp_path),
        ladder_rehearsal=_write_json(tmp_path / "rehearsal-without-budget.json", rehearsal),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["staged_ladder_rehearsed"]["status"] == "failed"


def test_goal_audit_fails_when_source_hash_binding_is_too_weak(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        spend_review=_spend_review(tmp_path / "weak-spend-review.json", source_file_check_count=0),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"


def test_goal_audit_fails_when_spend_scope_is_missing(tmp_path: Path) -> None:
    spend_review = json.loads(
        _spend_review(tmp_path / "spend-review.json").read_text(encoding="utf-8")
    )
    del spend_review["guarded_spend_scope"]
    config = replace(
        _config(tmp_path),
        spend_review=_write_json(tmp_path / "spend-review-without-scope.json", spend_review),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"


def test_goal_audit_fails_when_current_progress_verification_is_missing(
    tmp_path: Path,
) -> None:
    spend_review = json.loads(
        _spend_review(tmp_path / "spend-review.json").read_text(encoding="utf-8")
    )
    del spend_review["current_progress_verification"]
    config = replace(
        _config(tmp_path),
        spend_review=_write_json(tmp_path / "spend-review-without-progress.json", spend_review),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"


def test_goal_audit_fails_when_spend_review_progress_is_stale(tmp_path: Path) -> None:
    spend_review = json.loads(
        _spend_review(tmp_path / "spend-review.json").read_text(encoding="utf-8")
    )
    spend_review["current_progress_verification"]["status"] = "stale"
    spend_review["current_progress_verification"]["done_task_count"] = 8
    spend_review["current_progress_verification"]["remaining_task_count"] = 0
    config = replace(
        _config(tmp_path),
        spend_review=_write_json(tmp_path / "spend-review-with-stale-progress.json", spend_review),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"
    assert "current_progress_ok=False" in str(
        requirements["human_spend_review_packet_ready"]["evidence"]
    )


def test_goal_audit_fails_when_spend_review_policy_is_missing(tmp_path: Path) -> None:
    spend_review = json.loads(
        _spend_review(tmp_path / "spend-review.json").read_text(encoding="utf-8")
    )
    del spend_review["decision_policy"]
    config = replace(
        _config(tmp_path),
        spend_review=_write_json(tmp_path / "spend-review-without-policy.json", spend_review),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"


def test_goal_audit_fails_when_post_spend_acceptance_is_missing(tmp_path: Path) -> None:
    spend_review = json.loads(
        _spend_review(tmp_path / "spend-review.json").read_text(encoding="utf-8")
    )
    del spend_review["post_spend_acceptance_criteria"]
    config = replace(
        _config(tmp_path),
        spend_review=_write_json(
            tmp_path / "spend-review-without-acceptance.json",
            spend_review,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["human_spend_review_packet_ready"]["status"] == "failed"
    assert "acceptance_ok=False" in str(
        requirements["human_spend_review_packet_ready"]["evidence"]
    )


def test_goal_audit_fails_when_reference_alignment_is_not_canary_ready(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        reference_alignment=_reference_alignment(
            tmp_path / "bad-reference-alignment.json",
            canary_status="failed",
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["reference_workflow_alignment_bound"]["status"] == "failed"


def test_goal_audit_fails_when_reference_alignment_scope_delta_is_missing(
    tmp_path: Path,
) -> None:
    reference_alignment = json.loads(
        _reference_alignment(tmp_path / "reference-alignment.json").read_text()
    )
    del reference_alignment["reference_scope"]
    config = replace(
        _config(tmp_path),
        reference_alignment=_write_json(
            tmp_path / "bad-reference-alignment.json",
            reference_alignment,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["reference_workflow_alignment_bound"]["status"] == "failed"


def test_goal_audit_fails_when_reference_workflow_source_evidence_is_missing(
    tmp_path: Path,
) -> None:
    reference_alignment = json.loads(
        _reference_alignment(tmp_path / "reference-alignment.json").read_text()
    )
    reference_alignment["requirements"] = [
        item
        for item in reference_alignment["requirements"]
        if item["id"] != "reference_workflow_source_evidence_bound"
    ]
    config = replace(
        _config(tmp_path),
        reference_alignment=_write_json(
            tmp_path / "bad-reference-alignment.json",
            reference_alignment,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["reference_workflow_alignment_bound"]["status"] == "failed"


def test_goal_audit_fails_when_switchyard_skill_invariant_is_missing(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        protocol_audit=_protocol_audit(
            tmp_path / "bad-protocol-audit.json",
            invariant_status="missing",
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert (
        requirements["switchyard_only_skill_distillation_ab_invariant_bound"]["status"]
        == "failed"
    )


def test_goal_audit_fails_when_local_runtime_proof_is_missing(
    tmp_path: Path,
) -> None:
    protocol_audit = json.loads(
        _protocol_audit(tmp_path / "protocol-audit.json").read_text(encoding="utf-8")
    )
    protocol_audit["requirements"] = [
        item
        for item in protocol_audit["requirements"]
        if item["id"] != "local_switchyard_trialqa_transfer_runtime_bound"
    ]
    config = replace(
        _config(tmp_path),
        protocol_audit=_write_json(
            tmp_path / "protocol-audit-without-local-runtime.json",
            protocol_audit,
        ),
    )

    report = goal_audit.build_goal_audit(config)
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert (
        requirements["local_switchyard_trialqa_transfer_runtime_bound"]["status"]
        == "failed"
    )


def test_goal_audit_fails_when_frozen_gate_policy_threshold_drifts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(goal_audit.gate, "TOKEN_REDUCTION_MIN", 0.10)

    report = goal_audit.build_goal_audit(_config(tmp_path))
    requirements = _requirements(report)

    assert report["status"] == "not_ready_for_spend"
    assert requirements["frozen_promotion_kill_policy_bound"]["status"] == "failed"
