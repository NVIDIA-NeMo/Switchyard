# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

import benchmark.trialqa_local_current_packet as current_packet


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _decision_summary(
    artifact_dir: Path,
    *,
    version: int,
    stage: str = "generation",
    scope: str = "q0-q3-r1",
    stem: str = "ctgov-prospective-v1",
) -> Path:
    path = artifact_dir / f"decision-summary-{stem}-compact-v{version}-{scope}.json"
    spend_review = artifact_dir / f"spend-review-{stem}-compact-v{version}-{scope}.json"
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_decision_summary.v1",
            "status": f"awaiting_explicit_{stage}_spend_authorization",
            "goal_status": f"ready_for_{stage}_spend_decision",
            "goal_complete": False,
            "spend_authorized": False,
            "bundle_sha256": "sha256:" + str(version % 10) * 64,
            "next_boundary": {
                "stage": stage,
                "guarded_command_kind": f"guarded_{stage}_canary",
                "requires_yes_spend": True,
                "authorized_by_packet": False,
                "scope_label": "q0-q3, 1 repeat(s), 2 arms, 8 task(s)",
                "task_count": 8,
                "expected_model_calls": 8 if stage == "generation" else 0,
                "expected_judge_calls": 0 if stage == "generation" else 8,
            },
            "commands": {
                "guarded_spend": {
                    "command": [
                        "python",
                        "-m",
                        "benchmark.trialqa_local_canary",
                        "--spend-review",
                        str(spend_review),
                        "--yes-spend",
                    ],
                    "shell_command": (
                        "python -m benchmark.trialqa_local_canary "
                        f"--spend-review {spend_review} --yes-spend"
                    ),
                },
                "pre_spend_guard_check": {
                    "command": [
                        "python",
                        "-m",
                        "benchmark.trialqa_local_spend_guard",
                        "--spend-review",
                        str(spend_review),
                    ],
                    "shell_command": (
                        "python -m benchmark.trialqa_local_spend_guard "
                        f"--spend-review {spend_review}"
                    ),
                },
                "progress_monitor": {"shell_command": "python -m progress"},
                "post_spend_gate_inspection": {"shell_command": "python -m inspect"},
                "post_spend_checkpoint": {"shell_command": "python -m checkpoint"},
            },
            "proved_setup_evidence": [
                {
                    "id": "local_switchyard_trialqa_transfer_runtime_bound",
                    "evidence": "not Docker",
                }
            ],
            "goal_requirement_summary": {
                "total": 3,
                "status_counts": {"proved": 2, "missing": 1},
                "required_missing_ids": ["live_generation_operational_gate_passed"],
                "required_failed_ids": [],
            },
            "next_required_action": {
                "action": f"request_explicit_{stage}_canary_spend_approval",
                "requires_spend": True,
                "instruction": "approve before --yes-spend",
            },
            "missing_goal_evidence": [
                {"id": "live_generation_operational_gate_passed", "evidence": "missing"}
            ],
            "failed_goal_evidence": [],
            "goal_completion_note": "fixture incomplete",
            "blocked_actions": [
                {
                    "id": "no_full_benchmark_before_canaries",
                    "instruction": "do not expand early",
                },
                {
                    "id": "no_next_stage_spend_before_checkpoint",
                    "instruction": "checkpoint first",
                },
            ],
            "operator_checklist": [
                {"id": "review_packet", "requires_spend": False},
                {"id": "validate_spend_guard", "requires_spend": False},
                {"id": "run_guarded_canary_if_approved", "requires_spend": True},
                {"id": "monitor_without_spend", "requires_spend": False},
                {"id": "inspect_post_spend_gate", "requires_spend": False},
                {"id": "promote_or_kill", "requires_spend": False},
                {"id": "checkpoint_before_more_spend", "requires_spend": False},
            ],
            "post_spend_acceptance_criteria": {
                "stage": stage,
                "required_gate": "operational" if stage == "generation" else "promotion",
                "promote_decision": (
                    "promote_to_score"
                    if stage == "generation"
                    else "promote_to_next_cohort"
                ),
                "kill_decision": "kill",
                "must_run_checkpoint_before_more_spend": True,
            },
        },
    )


def _fake_guard_report(bundle_sha256: str) -> dict[str, object]:
    spend_review = "spend-review-ctgov-prospective-v1-compact-v10-q0-q3-r1.json"
    command = [
        "python",
        "-m",
        "benchmark.trialqa_local_canary",
        "--spend-review",
        spend_review,
        "--yes-spend",
    ]
    return {
        "schema_version": "switchyard.trialqa_spend_guard_check.v1",
        "status": "passed",
        "stage": "generation",
        "spend_authorized": False,
        "next_stage_spend_authorized": False,
        "guard": {
            "bundle": {"bundle_sha256": bundle_sha256},
            "current_progress": {
                "status": "matched",
                "selected_task_count": 8,
                "remaining_task_count": 8,
            },
        },
        "reviewed_command": {
            "command": command,
            "shell_command": " ".join(command),
            "requires_yes_spend": True,
            "contains_yes_spend": True,
            "contains_spend_review_guard": True,
        },
    }


def test_current_packet_selects_highest_generation_packet_without_spend(
    tmp_path: Path,
) -> None:
    _decision_summary(tmp_path, version=8)
    _decision_summary(tmp_path, version=10)

    report = current_packet.build_current_packet_report(
        artifact_dir=tmp_path,
        verify_guard=False,
    )

    assert report["schema_version"] == current_packet.SCHEMA_VERSION
    assert report["status"] == "selected_without_fresh_spend_guard"
    assert report["spend_authorized"] is False
    assert report["fresh_spend_guard_required_before_spend"] is True
    assert report["goal_status"] == "ready_for_generation_spend_decision"
    assert report["goal_complete"] is False
    assert report["selected"]["version"] == 10
    assert report["older_candidate_count"] == 1
    assert report["next_boundary"]["expected_model_calls"] == 8
    assert report["next_boundary"]["expected_judge_calls"] == 0
    assert report["goal_requirement_summary"] == {
        "total": 3,
        "status_counts": {"proved": 2, "missing": 1},
        "required_missing_ids": ["live_generation_operational_gate_passed"],
        "required_failed_ids": [],
    }
    assert report["next_required_action"] == {
        "action": "request_explicit_generation_canary_spend_approval",
        "requires_spend": True,
        "instruction": "approve before --yes-spend",
    }
    assert report["failed_goal_evidence"] == []
    assert report["goal_completion_note"] == "fixture incomplete"
    assert [item["id"] for item in report["blocked_actions"]] == [
        "no_full_benchmark_before_canaries",
        "no_next_stage_spend_before_checkpoint",
    ]
    assert any(
        item["id"] == "promote_or_kill" for item in report["operator_checklist"]
    )
    assert report["post_spend_acceptance_criteria"]["kill_decision"] == "kill"
    assert report["commands"]["guarded_spend"] is None
    assert report["fresh_spend_guard"] is None


def test_current_packet_filters_by_stage_and_scope(tmp_path: Path) -> None:
    _decision_summary(tmp_path, version=11, stage="score")
    _decision_summary(tmp_path, version=10, stage="generation", scope="q0-q3-r1")
    _decision_summary(tmp_path, version=12, stage="generation", scope="q0-q7-r1")

    report = current_packet.build_current_packet_report(
        artifact_dir=tmp_path,
        stage="generation",
        scope="q0-q3-r1",
        verify_guard=False,
    )

    assert report["selected"]["version"] == 10
    assert report["selected"]["scope"] == "q0-q3-r1"
    assert report["selected"]["stage"] == "generation"


def test_current_packet_can_rerun_no_spend_guard(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    bundle_sha = str(payload["bundle_sha256"])
    guarded = payload["commands"]["guarded_spend"]
    fake_reviewed = {
        "command": guarded["command"],
        "shell_command": guarded["shell_command"],
        "requires_yes_spend": True,
        "contains_yes_spend": True,
        "contains_spend_review_guard": True,
    }
    calls: list[Path] = []

    def fake_guard(*, spend_review: Path, recover_interrupted: bool = False) -> dict[str, object]:
        calls.append(spend_review)
        assert recover_interrupted is False
        report = _fake_guard_report(bundle_sha)
        report["reviewed_command"] = fake_reviewed
        return report

    monkeypatch.setattr(
        current_packet.spend_guard,
        "build_spend_guard_check",
        fake_guard,
    )

    report = current_packet.build_current_packet_report(
        artifact_dir=tmp_path,
        verify_guard=True,
    )

    assert calls == [tmp_path / "spend-review-ctgov-prospective-v1-compact-v10-q0-q3-r1.json"]
    assert report["status"] == "ready_for_user_spend_decision"
    assert report["fresh_spend_guard_required_before_spend"] is False
    assert report["commands"]["guarded_spend"] == fake_reviewed["shell_command"]
    assert report["fresh_spend_guard"]["status"] == "passed"
    assert report["fresh_spend_guard"]["bundle_sha256"] == bundle_sha
    assert report["fresh_spend_guard"]["reviewed_command"] == fake_reviewed


def test_current_packet_rejects_missing_packet(tmp_path: Path) -> None:
    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="no matching decision-summary",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_complete_goal_with_spend_boundary(tmp_path: Path) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["goal_complete"] = True
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="goal_complete",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_missing_goal_complete(tmp_path: Path) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    del payload["goal_complete"]
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="goal_complete",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_status_that_does_not_match_boundary(
    tmp_path: Path,
) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "awaiting_explicit_score_spend_authorization"
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="status",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_goal_status_that_does_not_match_boundary(
    tmp_path: Path,
) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["goal_status"] = "ready_for_score_spend_decision"
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="goal_status",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_missing_operator_handoff(tmp_path: Path) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["operator_checklist"] = [
        item
        for item in payload["operator_checklist"]
        if item["id"] != "promote_or_kill"
    ]
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="operator_checklist",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_missing_goal_state_handoff(tmp_path: Path) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    del payload["goal_requirement_summary"]
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="goal_requirement_summary",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_failed_goal_requirements(tmp_path: Path) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["goal_requirement_summary"]["required_failed_ids"] = [
        "quality_parity_and_efficiency_gate_passed"
    ]
    payload["failed_goal_evidence"] = [
        {
            "id": "quality_parity_and_efficiency_gate_passed",
            "evidence": "failed",
        }
    ]
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="failed required goal requirements",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_next_action_that_does_not_match_stage(
    tmp_path: Path,
) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["next_required_action"]["action"] = "request_explicit_score_canary_spend_approval"
    _write_json(summary, payload)

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="next_required_action",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=False,
        )


def test_current_packet_rejects_guarded_command_drift(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    summary = _decision_summary(tmp_path, version=10)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    bundle_sha = str(payload["bundle_sha256"])
    report = _fake_guard_report(bundle_sha)

    monkeypatch.setattr(
        current_packet.spend_guard,
        "build_spend_guard_check",
        lambda **_kwargs: report,
    )

    with pytest.raises(
        current_packet.TrialQACurrentPacketError,
        match="guarded command differs",
    ):
        current_packet.build_current_packet_report(
            artifact_dir=tmp_path,
            verify_guard=True,
        )
