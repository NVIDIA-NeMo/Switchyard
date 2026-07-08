# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_decision_summary as decision_summary


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _spend_review(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "manifest_id": "trialqa-full-test",
            "preflight": {"next_command_kind": "guarded_generation_canary"},
            "bundle_verification": {
                "bundle_sha256": "sha256:" + "1" * 64,
                "source_file_check_count": 23,
            },
            "guarded_spend_scope": {
                "stage": "generation",
                "scope_label": "q0-q3, 1 repeat(s), 2 arms, 8 task(s)",
                "task_count": 8,
                "expected_model_calls": 8,
                "expected_judge_calls": 0,
            },
            "guarded_spend_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary",
                    "--gate-output",
                    "gate-operational.json",
                    "--spend-review",
                    str(path),
                    "--yes-spend",
                ],
                "shell_command": (
                    "python -m benchmark.trialqa_local_canary "
                    f"--gate-output gate-operational.json --spend-review {path} --yes-spend"
                ),
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "progress_monitor_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_progress",
                ],
                "shell_command": "python -m benchmark.trialqa_local_progress",
                "requires_spend": False,
                "contains_yes_spend": False,
            },
            "safe_no_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
                "shell_command": "python -m benchmark.trialqa_local_preflight",
                "contains_yes_spend": False,
            },
            "post_spend_checkpoint_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_generation_checkpoint",
                ],
                "shell_command": "python -m benchmark.trialqa_local_generation_checkpoint",
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
            "current_progress_verification": {
                "status": "matched",
                "done_task_count": 0,
                "remaining_task_count": 8,
                "ledger_record_count": 0,
                "batch_lock_state": "missing",
            },
        },
    )


def _goal_audit(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_goal_audit.v1",
            "status": "ready_for_generation_spend_decision",
            "goal_complete": False,
            "spend_authorized": False,
            "requirement_summary": {
                "total": 5,
                "status_counts": {"proved": 3, "missing": 2},
                "required_missing_ids": [
                    "live_generation_operational_gate_passed",
                    "quality_parity_and_efficiency_gate_passed",
                ],
                "required_failed_ids": [],
            },
            "next_required_action": {
                "action": "request_explicit_generation_canary_spend_approval",
                "requires_spend": True,
                "instruction": (
                    "Review the current packet, then only run the guarded generation "
                    "canary after explicit approval for --yes-spend."
                ),
            },
            "completion_note": "fixture is not complete",
            "requirements": [
                {
                    "id": "prospective_manifest_bound",
                    "status": "proved",
                    "evidence": "manifest primary scope is prospective",
                },
                {
                    "id": "local_switchyard_trialqa_transfer_runtime_bound",
                    "status": "proved",
                    "evidence": (
                        "requires local Switchyard TrialQA-compatible prospective parquet, "
                        "not Docker or a second Hugging Face runtime repository"
                    ),
                },
                {
                    "id": "switchyard_only_skill_distillation_ab_invariant_bound",
                    "status": "proved",
                    "evidence": "same Ultra executor, baseline skill_loaded=False, treatment skill_loaded=True",
                },
                {
                    "id": "live_generation_operational_gate_passed",
                    "status": "missing",
                    "evidence": "no live operational gate exists",
                },
                {
                    "id": "quality_parity_and_efficiency_gate_passed",
                    "status": "missing",
                    "evidence": "no live promotion gate exists",
                },
            ],
        },
    )


def _config(tmp_path: Path) -> decision_summary.DecisionSummaryConfig:
    return decision_summary.DecisionSummaryConfig(
        spend_review=_spend_review(tmp_path / "spend-review.json"),
        goal_audit=_goal_audit(tmp_path / "goal-audit.json"),
        decision_summary_output=tmp_path / "decision-summary.json",
    )


def test_decision_summary_distills_next_boundary_without_authorizing_spend(
    tmp_path: Path,
) -> None:
    report = decision_summary.build_decision_summary(_config(tmp_path))

    assert report["schema_version"] == decision_summary.SCHEMA_VERSION
    assert report["status"] == "awaiting_explicit_generation_spend_authorization"
    assert report["spend_authorized"] is False
    assert report["goal_requirement_summary"] == {
        "total": 5,
        "status_counts": {"proved": 3, "missing": 2},
        "required_missing_ids": [
            "live_generation_operational_gate_passed",
            "quality_parity_and_efficiency_gate_passed",
        ],
        "required_failed_ids": [],
    }
    assert report["next_required_action"] == {
        "action": "request_explicit_generation_canary_spend_approval",
        "requires_spend": True,
        "instruction": (
            "Review the current packet, then only run the guarded generation "
            "canary after explicit approval for --yes-spend."
        ),
    }
    assert report["failed_goal_evidence"] == []
    assert report["goal_completion_note"] == "fixture is not complete"
    assert report["next_boundary"] == {
        "stage": "generation",
        "guarded_command_kind": "guarded_generation_canary",
        "requires_yes_spend": True,
        "authorized_by_packet": False,
        "scope_label": "q0-q3, 1 repeat(s), 2 arms, 8 task(s)",
        "task_count": 8,
        "expected_model_calls": 8,
        "expected_judge_calls": 0,
    }
    assert report["commands"]["guarded_spend"]["command"][-1] == "--yes-spend"
    assert report["commands"]["guarded_spend"]["authorized_by_packet"] is False
    assert "--yes-spend" not in report["commands"]["progress_monitor"]["command"]
    assert "--yes-spend" not in report["commands"]["safe_preflight_refresh"]["command"]
    assert report["commands"]["pre_spend_guard_check"] == {
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_spend_guard",
            "--spend-review",
            str(tmp_path / "spend-review.json"),
            "--output",
            str(tmp_path / "spend-guard-check-spend-review.json"),
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_spend_guard --spend-review "
            f"{tmp_path / 'spend-review.json'} --output "
            f"{tmp_path / 'spend-guard-check-spend-review.json'}"
        ),
        "contains_yes_spend": False,
        "review_note": (
            "Run immediately before approving spend; this rechecks the reviewed "
            "guarded command, current hash-bound bundle, and selected ledger/lock "
            "progress without making model or judge calls."
        ),
    }
    assert report["commands"]["post_spend_gate_inspection"] == {
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_gate_inspect",
            "--gate",
            "gate-operational.json",
            "--decision-summary",
            str(tmp_path / "decision-summary.json"),
            "--output",
            "gate-inspection-gate-operational.json",
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_gate_inspect "
            "--gate gate-operational.json --decision-summary "
            f"{tmp_path / 'decision-summary.json'} "
            "--output gate-inspection-gate-operational.json"
        ),
        "command_available": True,
        "contains_yes_spend": False,
        "review_note": (
            "Run after the guarded canary writes the required gate; this "
            "inspects the gate and still does not authorize next-stage spend."
        ),
    }
    assert "--yes-spend" not in report["commands"]["post_spend_checkpoint"]["command"]
    assert report["post_spend_acceptance_criteria"]["promote_decision"] == "promote_to_score"
    assert report["post_spend_acceptance_criteria"]["judge_spend_before_checkpoint_allowed"] is False
    assert [item["id"] for item in report["operator_checklist"]] == [
        "review_packet",
        "validate_spend_guard",
        "run_guarded_canary_if_approved",
        "monitor_without_spend",
        "inspect_post_spend_gate",
        "promote_or_kill",
        "checkpoint_before_more_spend",
    ]
    assert [item["id"] for item in report["proved_setup_evidence"]] == [
        "prospective_manifest_bound",
        "local_switchyard_trialqa_transfer_runtime_bound",
        "switchyard_only_skill_distillation_ab_invariant_bound",
    ]
    assert "not Docker" in str(report["proved_setup_evidence"][1]["evidence"])
    assert len(report["missing_goal_evidence"]) == 2


def test_decision_summary_rejects_safe_command_with_yes_spend(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.spend_review.read_text(encoding="utf-8"))
    payload["safe_no_spend_command"]["command"].append("--yes-spend")
    _write_json(config.spend_review, payload)

    with pytest.raises(decision_summary.TrialQADecisionSummaryError, match="safe"):
        decision_summary.build_decision_summary(config)


def test_decision_summary_rejects_guarded_command_without_spend_review(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.spend_review.read_text(encoding="utf-8"))
    command = payload["guarded_spend_command"]["command"]
    index = command.index("--spend-review")
    del command[index : index + 2]
    payload["guarded_spend_command"]["shell_command"] = " ".join(command)
    _write_json(config.spend_review, payload)

    with pytest.raises(decision_summary.TrialQADecisionSummaryError, match="spend-review"):
        decision_summary.build_decision_summary(config)


def test_decision_summary_rejects_stale_progress(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.spend_review.read_text(encoding="utf-8"))
    payload["current_progress_verification"]["status"] = "stale"
    _write_json(config.spend_review, payload)

    with pytest.raises(decision_summary.TrialQADecisionSummaryError, match="progress"):
        decision_summary.build_decision_summary(config)


def test_decision_summary_rejects_goal_audit_without_requirement_summary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.goal_audit.read_text(encoding="utf-8"))
    del payload["requirement_summary"]
    _write_json(config.goal_audit, payload)

    with pytest.raises(
        decision_summary.TrialQADecisionSummaryError,
        match="requirement_summary",
    ):
        decision_summary.build_decision_summary(config)


def test_decision_summary_rejects_goal_audit_with_failed_required_ids(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.goal_audit.read_text(encoding="utf-8"))
    payload["requirement_summary"]["required_failed_ids"] = [
        "quality_parity_and_efficiency_gate_passed"
    ]
    _write_json(config.goal_audit, payload)

    with pytest.raises(
        decision_summary.TrialQADecisionSummaryError,
        match="failed required",
    ):
        decision_summary.build_decision_summary(config)


def test_decision_summary_rejects_goal_audit_status_stage_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.goal_audit.read_text(encoding="utf-8"))
    payload["status"] = "ready_for_score_spend_decision"
    _write_json(config.goal_audit, payload)

    with pytest.raises(
        decision_summary.TrialQADecisionSummaryError,
        match="status",
    ):
        decision_summary.build_decision_summary(config)
