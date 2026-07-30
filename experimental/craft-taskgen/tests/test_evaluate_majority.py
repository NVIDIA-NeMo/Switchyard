# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

from scripts.run_evaluate_majority import (
    aggregate_votes,
    apply_majority_to_state,
    extract_trial_rows,
    prepare_trial_state,
)


def _state() -> dict:
    return {
        "created": "2026-01-01T00:00:00",
        "last_updated": "2026-01-01T00:00:00",
        "profile_data": {"llm_step_model": "test-model", "default_concurrency": 2},
        "tasks": {
            "teleport-abc123": {
                "task_id": "teleport-abc123",
                "repo": "teleport",
                "commit_sha": "abc123",
                "stage": "rejected",
                "eval_verdict": "reject",
                "eval_reason": "old",
                "eval_instruction_sketch": "",
                "eval_verifier_notes": "AL1",
                "iteration_log": [{"step": "evaluate", "verdict": "reject"}],
                "in_progress_step": "evaluate",
                "candidate_data": {
                    "source_task_id": "instance_1",
                    "source_metadata": {"instance_id": "instance_1"},
                },
            },
            "teleport-def456": {
                "task_id": "teleport-def456",
                "repo": "teleport",
                "commit_sha": "def456",
                "stage": "rejected",
                "eval_verdict": "reject",
                "eval_reason": "old",
                "candidate_data": {
                    "source_task_id": "instance_2",
                    "source_metadata": {"instance_id": "instance_2"},
                },
            },
        },
    }


def _args(**overrides):
    values = {
        "task_id": [],
        "repo": "",
        "instance_id": "",
        "limit": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _row(verdict: str, reason: str, run_number: int) -> dict:
    return {
        "run_number": run_number,
        "task_id": "teleport-abc123",
        "source_task_id": "instance_1",
        "repo": "teleport",
        "commit_sha": "abc123",
        "stage": "promising" if verdict == "accept" else "rejected",
        "eval_verdict": verdict,
        "eval_reason": reason,
        "eval_instruction_sketch": "sketch",
        "eval_verifier_notes": "",
    }


def test_prepare_trial_state_resets_selected_tasks_to_candidate(tmp_path) -> None:
    input_state = tmp_path / "state.json"
    output_state = tmp_path / "trial.json"
    input_state.write_text(json.dumps(_state()))

    selected = prepare_trial_state(
        input_state,
        output_state,
        _args(task_id=["teleport-abc123"]),
    )

    assert selected == ["teleport-abc123"]
    trial = json.loads(output_state.read_text())
    task = trial["tasks"]["teleport-abc123"]
    assert task["stage"] == "candidate"
    assert task["eval_verdict"] == ""
    assert task["eval_reason"] == ""
    assert task["iteration_log"] == []
    assert task["in_progress_step"] == ""
    assert trial["tasks"]["teleport-def456"]["stage"] == "rejected"


def test_extract_trial_rows_reads_evaluate_outputs(tmp_path) -> None:
    state = _state()
    task = state["tasks"]["teleport-abc123"]
    task["stage"] = "promising"
    task["eval_verdict"] = "accept"
    task["eval_reason"] = "looks good"
    task["llm_usage"] = {"evaluate": [{"tokens_in": 10, "tokens_out": 3}]}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))

    rows = extract_trial_rows(state_path, ["teleport-abc123"], 2)

    assert rows == [
        {
            "run_number": 2,
            "task_id": "teleport-abc123",
            "source_task_id": "instance_1",
            "repo": "teleport",
            "commit_sha": "abc123",
            "stage": "promising",
            "eval_verdict": "accept",
            "eval_reason": "looks good",
            "eval_instruction_sketch": "",
            "eval_verifier_notes": "AL1",
            "usage": {"tokens_in": 10, "tokens_out": 3},
        }
    ]


def test_aggregate_votes_requires_strict_majority_by_default() -> None:
    summaries = aggregate_votes(
        [
            [_row("accept", "yes", 1)],
            [_row("reject", "no", 2)],
            [_row("ERROR", "infra", 3)],
        ]
    )

    assert summaries[0]["winner"] == ""
    assert summaries[0]["strict_majority"] is False


def test_apply_majority_to_state_updates_eval_fields_and_stage(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()))
    summaries = aggregate_votes(
        [
            [_row("accept", "yes 1", 1)],
            [_row("reject", "no", 2)],
            [_row("accept", "yes 2", 3)],
        ]
    )

    applied = apply_majority_to_state(state_path, summaries, backup_path=tmp_path / "backup.json")

    assert applied == 1
    updated = json.loads(state_path.read_text())
    task = updated["tasks"]["teleport-abc123"]
    assert task["eval_verdict"] == "accept"
    assert task["eval_reason"] == "yes 1"
    assert task["eval_instruction_sketch"] == "sketch"
    assert task["stage"] == "promising"
    assert task["iteration_log"][-1]["step"] == "evaluate_majority"
    assert task["iteration_log"][-1]["counts"] == {"accept": 2, "reject": 1}
    assert (
        json.loads((tmp_path / "backup.json").read_text())["tasks"]["teleport-abc123"]["eval_verdict"]
        == "reject"
    )


def test_apply_majority_preserves_downstream_stage_without_force(tmp_path) -> None:
    state = _state()
    state["tasks"]["teleport-abc123"]["stage"] = "accepted"
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    summaries = aggregate_votes(
        [
            [_row("accept", "yes 1", 1)],
            [_row("accept", "yes 2", 2)],
            [_row("reject", "no", 3)],
        ]
    )

    applied = apply_majority_to_state(state_path, summaries)

    assert applied == 1
    updated = json.loads(state_path.read_text())
    assert updated["tasks"]["teleport-abc123"]["stage"] == "accepted"
    assert updated["tasks"]["teleport-abc123"]["eval_verdict"] == "accept"
