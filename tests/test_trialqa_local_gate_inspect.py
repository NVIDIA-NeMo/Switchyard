# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_gate as gate_module
import benchmark.trialqa_local_gate_inspect as gate_inspect


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate(
    path: Path,
    *,
    gate: str = "operational",
    decision: str = "promote_to_score",
    manifest_id: str = "trialqa-full-test",
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": gate_module.SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "gate": gate,
            "decision": decision,
            "benefit": {
                "token_reduction_fraction": 0.25,
                "operational_call_reduction_fraction": 0.4,
                "mean_score_delta": None,
                "terminal_rate_delta": 0.0,
            },
            "quality": {
                "available": False,
                "mode": "interim_harm_screen",
                "point_delta": None,
                "decision_bound": None,
                "decision_threshold": -0.05,
            },
            "scope": {
                "pair_count": 4,
                "task_count": 8,
                "confirmatory_scope_complete": False,
                "selection_attestation": {
                    "question_start": 0,
                    "question_limit": 4,
                    "selected_repeat_indices": [1],
                },
            },
            "criteria": [
                {
                    "name": "token_reduction_fraction",
                    "passed": decision == "promote_to_score",
                    "value": 0.25 if decision == "promote_to_score" else 0.01,
                    "operator": ">=",
                    "threshold": 0.15,
                }
            ],
        },
    )


def _decision_summary(
    path: Path,
    *,
    gate_path: Path,
    manifest_id: str = "trialqa-full-test",
) -> Path:
    gate_inspection_output = gate_path.with_name(f"gate-inspection-{gate_path.name}")
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_decision_summary.v1",
            "status": "awaiting_explicit_generation_spend_authorization",
            "goal_status": "ready_for_generation_spend_decision",
            "goal_complete": False,
            "spend_authorized": False,
            "manifest_id": manifest_id,
            "next_boundary": {
                "stage": "generation",
                "guarded_command_kind": "guarded_generation_canary",
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "goal_requirement_summary": {
                "total": 11,
                "status_counts": {"proved": 9, "missing": 2},
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
            "failed_goal_evidence": [],
            "goal_completion_note": "fixture incomplete",
            "commands": {
                "post_spend_gate_inspection": {
                    "command": [
                        "python",
                        "-m",
                        "benchmark.trialqa_local_gate_inspect",
                        "--gate",
                        str(gate_path),
                        "--decision-summary",
                        str(path),
                        "--output",
                        str(gate_inspection_output),
                    ],
                    "shell_command": (
                        "python -m benchmark.trialqa_local_gate_inspect "
                        f"--gate {gate_path} --decision-summary {path} "
                        f"--output {gate_inspection_output}"
                    ),
                    "command_available": True,
                    "contains_yes_spend": False,
                },
                "post_spend_checkpoint": {
                    "command": [
                        "python",
                        "-m",
                        "benchmark.trialqa_local_generation_checkpoint",
                    ],
                    "shell_command": "python -m benchmark.trialqa_local_generation_checkpoint",
                    "contains_yes_spend": False,
                }
            },
            "post_spend_acceptance_criteria": {
                "stage": "generation",
                "required_gate": "operational",
                "required_gate_artifact": str(gate_path),
                "required_gate_schema_version": gate_module.SCHEMA_VERSION,
                "promote_decision": "promote_to_score",
                "kill_decision": "kill",
                "next_no_spend_checkpoint_kind": "post_generation_checkpoint",
                "checkpoint_command_available": True,
                "must_run_checkpoint_before_more_spend": True,
                "next_boundary_if_promoted": "score_spend_review",
                "judge_spend_before_checkpoint_allowed": False,
            },
        },
    )


def test_gate_inspection_promotes_to_checkpoint_without_authorizing_spend(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)

    report = gate_inspect.inspect_gate(
        gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
    )

    assert report["schema_version"] == gate_inspect.SCHEMA_VERSION
    assert report["status"] == "gate_promoted"
    assert report["spend_authorized"] is False
    assert report["next_stage_spend_authorized"] is False
    assert report["promoted"] is True
    assert report["next_action"] == "run_post_spend_checkpoint"
    assert report["must_run_checkpoint_before_more_spend"] is True
    assert "--yes-spend" not in report["validated_gate_inspection_command"]["command"]
    assert report["decision_summary_state"] == {
        "status": "awaiting_explicit_generation_spend_authorization",
        "goal_status": "ready_for_generation_spend_decision",
        "goal_complete": False,
        "goal_requirement_summary": {
            "total": 11,
            "status_counts": {"proved": 9, "missing": 2},
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
        "next_boundary": {
            "stage": "generation",
            "guarded_command_kind": "guarded_generation_canary",
            "requires_yes_spend": True,
            "authorized_by_packet": False,
        },
    }
    assert "--yes-spend" not in report["post_spend_checkpoint_command"]["command"]
    assert report["failed_criteria"] == []
    assert report["gate_summary"]["benefit"]["token_reduction_fraction"] == 0.25


def test_gate_inspection_kill_still_points_to_terminal_checkpoint(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate-operational.json", decision="kill")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)

    report = gate_inspect.inspect_gate(
        gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
    )

    assert report["status"] == "gate_killed"
    assert report["promoted"] is False
    assert report["next_action"] == "run_post_spend_checkpoint_to_record_terminal_decision"
    assert report["next_stage_spend_authorized"] is False
    assert report["failed_criteria"] == [
        {
            "name": "token_reduction_fraction",
            "value": 0.01,
            "operator": ">=",
            "threshold": 0.15,
        }
    ]


def test_gate_inspection_rejects_gate_path_mismatch(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(
        tmp_path / "decision-summary.json",
        gate_path=tmp_path / "other-gate.json",
    )

    with pytest.raises(gate_inspect.TrialQAGateInspectError, match="gate path"):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_checkpoint_with_yes_spend(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["commands"]["post_spend_checkpoint"]["command"].append("--yes-spend")
    _write_json(summary, payload)

    with pytest.raises(gate_inspect.TrialQAGateInspectError, match="checkpoint"):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_summary_status_stage_mismatch(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "awaiting_explicit_score_spend_authorization"
    _write_json(summary, payload)

    with pytest.raises(gate_inspect.TrialQAGateInspectError, match="status"):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_summary_with_failed_goal_requirements(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["goal_requirement_summary"]["required_failed_ids"] = [
        "quality_parity_and_efficiency_gate_passed"
    ]
    _write_json(summary, payload)

    with pytest.raises(
        gate_inspect.TrialQAGateInspectError,
        match="failed required",
    ):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_summary_without_explicit_spend_action(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["next_required_action"]["requires_spend"] = False
    _write_json(summary, payload)

    with pytest.raises(
        gate_inspect.TrialQAGateInspectError,
        match="next_required_action",
    ):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_stale_embedded_inspection_command(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["commands"]["post_spend_gate_inspection"]["command"][
        payload["commands"]["post_spend_gate_inspection"]["command"].index("--gate") + 1
    ] = str(tmp_path / "old-gate.json")
    _write_json(summary, payload)

    with pytest.raises(gate_inspect.TrialQAGateInspectError, match="--gate"):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )


def test_gate_inspection_rejects_inspection_shell_command_with_yes_spend(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate-operational.json")
    summary = _decision_summary(tmp_path / "decision-summary.json", gate_path=gate)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["commands"]["post_spend_gate_inspection"]["shell_command"] += " --yes-spend"
    _write_json(summary, payload)

    with pytest.raises(gate_inspect.TrialQAGateInspectError, match="gate inspection"):
        gate_inspect.inspect_gate(
            gate_inspect.GateInspectConfig(gate=gate, decision_summary=summary)
        )
