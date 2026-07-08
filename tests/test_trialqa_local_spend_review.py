# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_spend_review as review


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _guarded_command(kind: str = "guarded_generation_canary") -> list[str]:
    module = (
        "benchmark.trialqa_local_canary"
        if kind == "guarded_generation_canary"
        else "benchmark.trialqa_local_canary_score"
    )
    command = [
        "python",
        "-m",
        module,
        "--manifest",
        "manifest.json",
        "--experiment-root",
        "experiments",
        "--question-start",
        "0",
        "--question-limit",
        "4",
        "--repeat-limit",
        "1",
        "--workers",
        "4",
        "--max-generation-attempts",
        "1",
    ]
    if kind == "guarded_generation_canary":
        command.extend(["--gate-output", "gate-operational.json"])
    else:
        command.extend(["--promotion-gate-output", "gate-promotion.json"])
    command.append("--yes-spend")
    return command


def _preflight(path: Path, *, kind: str = "guarded_generation_canary") -> Path:
    command = _guarded_command(kind)
    checkpoint_kind = (
        "post_generation_checkpoint"
        if kind == "guarded_generation_canary"
        else "post_score_checkpoint"
    )
    checkpoint_module = (
        "benchmark.trialqa_local_generation_checkpoint"
        if kind == "guarded_generation_canary"
        else "benchmark.trialqa_local_score_checkpoint"
    )
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_no_spend_preflight.v1",
            "status": "passed",
            "spend_authorized": False,
            "manifest_id": "trialqa-full-test",
            "next_command": {
                "kind": kind,
                "command": command,
                "shell_command": " ".join(command),
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
            "post_spend_checkpoint_command": {
                "kind": checkpoint_kind,
                "command": [
                    "python",
                    "-m",
                    checkpoint_module,
                ],
                "shell_command": f"python -m {checkpoint_module}",
                "requires_spend": False,
                "contains_yes_spend": False,
            },
        },
    )


def _verification(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_pre_spend_audit_bundle_verification.v1",
            "status": "passed",
            "bundle": {
                "manifest_id": "trialqa-full-test",
                "bundle_state": "awaiting_generation_canary_spend_authorization",
                "sha256": "sha256:" + "1" * 64,
            },
            "artifact_checks": [{"name": "manifest", "status": "matched"}],
            "source_file_checks": [{"path": "guardrail.py", "status": "matched"}],
        },
    )


def _next_step(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "run_guarded_generation_canary",
            "safe_next_command": {
                "kind": "generation_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
                "shell_command": "python -m benchmark.trialqa_local_preflight",
            },
        },
    )


def _progress(
    path: Path,
    *,
    stage: str = "generation",
    action: str = "run_guarded_generation_canary_after_spend_review",
    requires_spend: bool = True,
    category_counts: dict[str, int] | None = None,
    done_task_count: int = 0,
    remaining_task_count: int = 8,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_progress.v1",
            "manifest_id": "trialqa-full-test",
            "stage": stage,
            "scope": {
                "question_start": 0,
                "question_limit": 4,
                "selected_repeat_indices": [1],
                "selected_task_count": 8,
                "condition": "both",
            },
            "progress": {
                "selected_task_count": 8,
                "done_task_count": done_task_count,
                "remaining_task_count": remaining_task_count,
                "category_counts": category_counts or {"not_started": 8},
            },
            "ledger": {
                "record_count": 0,
            },
            "batch_lock": {
                "state": "missing",
            },
            "recommendation": {
                "action": action,
                "requires_spend": requires_spend,
            },
        },
    )


def test_spend_review_packet_exposes_guarded_command_without_authorizing_spend(
    tmp_path: Path,
) -> None:
    spend_review_path = tmp_path / "spend-review.json"
    report = review.build_spend_review_packet(
        preflight_path=_preflight(tmp_path / "preflight.json"),
        bundle_verification_path=_verification(tmp_path / "verification.json"),
        next_step_path=_next_step(tmp_path / "next-step.json"),
        spend_review_path=spend_review_path,
    )

    assert report["schema_version"] == review.SCHEMA_VERSION
    assert report["status"] == "ready_for_user_spend_decision"
    assert report["authorized_by_packet"] is False
    assert report["safe_no_spend_command"]["contains_yes_spend"] is False
    assert report["guarded_spend_command"]["requires_yes_spend"] is True
    assert report["guarded_spend_command"]["authorized_by_packet"] is False
    assert report["guarded_spend_command"]["requires_spend_review_guard"] is True
    assert report["guarded_spend_command"]["command"][-3:] == [
        "--spend-review",
        str(spend_review_path),
        "--yes-spend",
    ]
    assert report["guarded_recovery_command"] == {
        "command": [
            *report["guarded_spend_command"]["command"][:-1],
            "--recover-interrupted",
            "--yes-spend",
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_canary --manifest manifest.json "
            "--experiment-root experiments --question-start 0 --question-limit 4 "
            "--repeat-limit 1 --workers 4 --max-generation-attempts 1 "
            f"--gate-output gate-operational.json --spend-review {spend_review_path} "
            "--recover-interrupted --yes-spend"
        ),
        "requires_yes_spend": True,
        "authorized_by_audit": False,
        "authorized_by_packet": False,
        "recovery_flag": "--recover-interrupted",
        "stage": "generation",
        "review_note": (
            "Use only if the read-only progress monitor reports an interrupted "
            "generation canary; still requires explicit spend approval."
        ),
    }
    assert report["progress_monitor_command"] == {
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_progress",
            "--manifest",
            "manifest.json",
            "--experiment-root",
            "experiments",
            "--stage",
            "generation",
            "--question-start",
            "0",
            "--question-limit",
            "4",
            "--repeat-limit",
            "1",
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_progress --manifest manifest.json "
            "--experiment-root experiments --stage generation --question-start 0 "
            "--question-limit 4 --repeat-limit 1"
        ),
        "contains_yes_spend": False,
        "requires_spend": False,
        "stage": "generation",
        "review_note": (
            "Read-only ledger progress monitor; safe to run before, during, "
            "or after the guarded spend command."
        ),
    }
    assert report["post_spend_checkpoint_command"] == {
        "kind": "post_generation_checkpoint",
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_generation_checkpoint",
        ],
        "shell_command": "python -m benchmark.trialqa_local_generation_checkpoint",
        "requires_spend": False,
        "contains_yes_spend": False,
    }
    assert report["post_spend_acceptance_criteria"] == {
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
        "review_note": (
            "After the guarded generation canary finishes, inspect the "
            "operational gate at the listed artifact path. Proceed only "
            "if its decision is 'promote_to_score'; otherwise stop on "
            "the kill decision. Run the no-spend checkpoint before any "
            "additional spend."
        ),
    }
    assert report["bundle_verification"]["source_file_check_count"] == 1
    assert report["guarded_spend_scope"] == {
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
    }
    assert report["decision_policy"] == {
        "name": "ultra-efficiency-v3",
        "thresholds": {
            "token_reduction_min": 0.15,
            "operational_call_reduction_min": 0.2,
            "quality_delta_min": -0.05,
            "quality_confidence_level": 0.95,
            "futility_confidence_level": 0.95,
        },
        "quality_modes": {
            "interim": "interim_harm_screen",
            "confirmatory": "confirmatory_noninferiority",
        },
        "population_and_retry_policy": {
            "analysis_population": "intention-to-treat",
            "completed_draw_policy": "terminal-no-retry-or-replacement",
            "empty_answer_policy": "score-zero",
            "performance_retry_policy": "zero-null-eof-retries",
        },
        "current_boundary": {
            "stage": "generation",
            "post_spend_gate": "operational",
            "promote_decision": "promote_to_score",
            "kill_decision": "kill",
            "judge_spend_deferred": True,
        },
        "review_note": (
            "Frozen gate policy that will interpret the post-spend checkpoint; "
            "do not revise thresholds after seeing TrialQA outcomes."
        ),
    }


def test_spend_review_verifies_current_progress_when_provided(tmp_path: Path) -> None:
    report = review.build_spend_review_packet(
        preflight_path=_preflight(tmp_path / "preflight.json"),
        bundle_verification_path=_verification(tmp_path / "verification.json"),
        next_step_path=_next_step(tmp_path / "next-step.json"),
        progress_path=_progress(tmp_path / "progress.json"),
    )

    assert report["current_progress_verification"] == {
        "path": str(tmp_path / "progress.json"),
        "status": "matched",
        "stage": "generation",
        "action": "run_guarded_generation_canary_after_spend_review",
        "requires_spend": True,
        "selected_task_count": 8,
        "done_task_count": 0,
        "remaining_task_count": 8,
        "category_counts": {"not_started": 8},
        "ledger_record_count": 0,
        "batch_lock_state": "missing",
        "review_note": (
            "Current read-only progress matched the reviewed scope and is still "
            "at a clean spend boundary."
        ),
    }


def test_spend_review_rejects_stale_current_progress(tmp_path: Path) -> None:
    with pytest.raises(review.TrialQASpendReviewError, match="clean spend boundary"):
        review.build_spend_review_packet(
            preflight_path=_preflight(tmp_path / "preflight.json"),
            bundle_verification_path=_verification(tmp_path / "verification.json"),
            next_step_path=_next_step(tmp_path / "next-step.json"),
            progress_path=_progress(
                tmp_path / "progress.json",
                action="run_generation_checkpoint_after_operational_gate",
                requires_spend=False,
                category_counts={"generated": 8},
                done_task_count=8,
                remaining_task_count=0,
            ),
        )


def test_spend_review_accepts_generation_expansion_boundary(tmp_path: Path) -> None:
    next_step = _write_json(
        tmp_path / "next-step.json",
        {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "expand_generation_scope",
            "safe_next_command": {
                "kind": "generation_expansion_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
                "shell_command": "python -m benchmark.trialqa_local_preflight",
            },
        },
    )

    report = review.build_spend_review_packet(
        preflight_path=_preflight(tmp_path / "preflight.json"),
        bundle_verification_path=_verification(tmp_path / "verification.json"),
        next_step_path=next_step,
    )

    assert report["status"] == "ready_for_user_spend_decision"
    assert report["next_step"]["action"] == "expand_generation_scope"
    assert report["next_step"]["safe_command_kind"] == "generation_expansion_preflight"
    assert report["guarded_spend_command"]["command"][-1] == "--yes-spend"
    assert report["guarded_spend_scope"]["expected_model_calls"] == 8


def test_spend_review_quantifies_score_boundary_as_judge_spend(tmp_path: Path) -> None:
    verification = json.loads(
        _verification(tmp_path / "verification.json").read_text(encoding="utf-8")
    )
    verification["bundle"]["bundle_state"] = "awaiting_score_canary_spend_authorization"
    _write_json(tmp_path / "verification.json", verification)
    next_step = _write_json(
        tmp_path / "next-step.json",
        {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "run_guarded_score_canary",
            "safe_next_command": {
                "kind": "score_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_score_preflight"],
                "shell_command": "python -m benchmark.trialqa_local_score_preflight",
            },
        },
    )

    report = review.build_spend_review_packet(
        preflight_path=_preflight(
            tmp_path / "preflight.json",
            kind="guarded_score_canary",
        ),
        bundle_verification_path=tmp_path / "verification.json",
        next_step_path=next_step,
        spend_review_path=tmp_path / "spend-review-score.json",
    )

    assert report["status"] == "ready_for_user_spend_decision"
    assert report["guarded_spend_scope"]["stage"] == "score"
    assert report["guarded_spend_scope"]["task_count"] == 8
    assert report["guarded_spend_scope"]["expected_model_calls"] == 0
    assert report["guarded_spend_scope"]["expected_judge_calls"] == 8
    assert report["guarded_spend_scope"]["maximum_generation_attempts"] == 0
    assert report["guarded_spend_command"]["command"][-3:] == [
        "--spend-review",
        str(tmp_path / "spend-review-score.json"),
        "--yes-spend",
    ]
    assert report["decision_policy"]["name"] == "ultra-efficiency-v3"
    assert report["decision_policy"]["current_boundary"] == {
        "stage": "score",
        "post_spend_gate": "promotion",
        "promote_decision": "promote_to_next_cohort",
        "kill_decision": "kill",
        "judge_spend_deferred": False,
    }
    assert report["decision_policy"]["thresholds"]["quality_delta_min"] == -0.05
    assert report["progress_monitor_command"]["stage"] == "score"
    assert "--yes-spend" not in report["progress_monitor_command"]["command"]
    assert report["guarded_recovery_command"]["stage"] == "score"
    assert report["guarded_recovery_command"]["authorized_by_packet"] is False
    assert report["guarded_recovery_command"]["command"][-2:] == [
        "--recover-interrupted",
        "--yes-spend",
    ]
    assert report["guarded_recovery_command"]["command"].count("--recover-interrupted") == 1
    assert report["post_spend_checkpoint_command"]["kind"] == "post_score_checkpoint"
    assert report["post_spend_checkpoint_command"]["command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_score_checkpoint",
    ]
    assert report["post_spend_acceptance_criteria"] == {
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
        "review_note": (
            "After the guarded score canary finishes, inspect the "
            "promotion gate at the listed artifact path. Proceed only "
            "if its decision is 'promote_to_next_cohort'; otherwise stop on "
            "the kill decision. Run the no-spend checkpoint before any "
            "additional spend."
        ),
    }


def test_spend_review_rejects_safe_command_with_yes_spend(tmp_path: Path) -> None:
    next_step = json.loads(_next_step(tmp_path / "next-step.json").read_text(encoding="utf-8"))
    next_step["safe_next_command"]["command"].append("--yes-spend")
    _write_json(tmp_path / "next-step.json", next_step)

    with pytest.raises(review.TrialQASpendReviewError, match="safe next command"):
        review.build_spend_review_packet(
            preflight_path=_preflight(tmp_path / "preflight.json"),
            bundle_verification_path=_verification(tmp_path / "verification.json"),
            next_step_path=tmp_path / "next-step.json",
        )


def test_spend_review_rejects_missing_source_file_checks(tmp_path: Path) -> None:
    verification = json.loads(
        _verification(tmp_path / "verification.json").read_text(encoding="utf-8")
    )
    verification["source_file_checks"] = []
    _write_json(tmp_path / "verification.json", verification)

    with pytest.raises(review.TrialQASpendReviewError, match="source-file"):
        review.build_spend_review_packet(
            preflight_path=_preflight(tmp_path / "preflight.json"),
            bundle_verification_path=tmp_path / "verification.json",
            next_step_path=_next_step(tmp_path / "next-step.json"),
        )


def test_spend_review_rejects_checkpoint_command_with_yes_spend(tmp_path: Path) -> None:
    preflight = json.loads(_preflight(tmp_path / "preflight.json").read_text(encoding="utf-8"))
    preflight["post_spend_checkpoint_command"]["command"].append("--yes-spend")
    _write_json(tmp_path / "preflight.json", preflight)

    with pytest.raises(review.TrialQASpendReviewError, match="post_spend_checkpoint"):
        review.build_spend_review_packet(
            preflight_path=tmp_path / "preflight.json",
            bundle_verification_path=_verification(tmp_path / "verification.json"),
            next_step_path=_next_step(tmp_path / "next-step.json"),
        )


def test_spend_review_rejects_checkpoint_kind_that_does_not_match_stage(
    tmp_path: Path,
) -> None:
    preflight = json.loads(
        _preflight(
            tmp_path / "preflight.json",
            kind="guarded_score_canary",
        ).read_text(encoding="utf-8")
    )
    preflight["post_spend_checkpoint_command"]["kind"] = "post_generation_checkpoint"
    _write_json(tmp_path / "preflight.json", preflight)
    verification = json.loads(
        _verification(tmp_path / "verification.json").read_text(encoding="utf-8")
    )
    verification["bundle"]["bundle_state"] = "awaiting_score_canary_spend_authorization"
    _write_json(tmp_path / "verification.json", verification)
    next_step = _write_json(
        tmp_path / "next-step.json",
        {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "run_guarded_score_canary",
            "safe_next_command": {
                "kind": "score_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_score_preflight"],
            },
        },
    )

    with pytest.raises(review.TrialQASpendReviewError, match="post_score_checkpoint"):
        review.build_spend_review_packet(
            preflight_path=tmp_path / "preflight.json",
            bundle_verification_path=tmp_path / "verification.json",
            next_step_path=next_step,
        )
