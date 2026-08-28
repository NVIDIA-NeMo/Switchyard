# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from scripts.run_alignment_majority import aggregate_votes, apply_majority_to_state


def _state(task_id: str = "teleport-abc123") -> dict:
    return {
        "created": "2026-01-01T00:00:00",
        "last_updated": "2026-01-01T00:00:00",
        "tasks": {
            task_id: {
                "task_id": task_id,
                "repo": "teleport",
                "commit_sha": "abc123",
                "stage": "promising",
                "alignment_verdict": "stale",
                "alignment_reason": "old",
                "alignment_attempts": [],
                "candidate_data": {
                    "source_task_id": "instance_1",
                    "source_metadata": {"instance_id": "instance_1"},
                },
            }
        },
    }


def _row(verdict: str, reason: str, *, run_status: str = "ok") -> dict:
    row = {
        "task_id": "teleport-abc123",
        "source_task_id": "instance_1",
        "repo": "teleport",
        "commit_sha": "abc123",
        "status": run_status,
    }
    if run_status == "ok":
        row.update({"verdict": verdict, "reason": reason, "tokens_in": 10, "tokens_out": 5})
    else:
        row["error"] = reason
    return row


def test_aggregate_votes_picks_majority_label() -> None:
    summaries = aggregate_votes(
        [
            [_row("ok", "first ok")],
            [_row("narrow_tests", "too narrow")],
            [_row("ok", "second ok")],
        ]
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["winner"] == "ok"
    assert summary["winner_count"] == 2
    assert summary["strict_majority"] is True
    assert summary["alignment_reason"] == "first ok"


def test_apply_majority_to_state_updates_alignment_fields_and_attempts(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()))
    summaries = aggregate_votes(
        [
            [_row("ok", "first ok")],
            [_row("leaked", "too much")],
            [_row("ok", "second ok")],
        ]
    )

    applied = apply_majority_to_state(state_path, summaries, backup_path=tmp_path / "backup.json")

    assert applied == 1
    updated = json.loads(state_path.read_text())
    task = updated["tasks"]["teleport-abc123"]
    assert task["alignment_verdict"] == "ok"
    assert task["alignment_reason"] == "first ok"
    assert [attempt["verdict"] for attempt in task["alignment_attempts"]] == ["ok", "leaked", "ok"]
    assert [attempt["selected_by_majority"] for attempt in task["alignment_attempts"]] == [
        True,
        False,
        True,
    ]
    assert (
        json.loads((tmp_path / "backup.json").read_text())["tasks"]["teleport-abc123"]["alignment_verdict"]
        == "stale"
    )


def test_tie_keep_current_does_not_apply(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()))
    summaries = aggregate_votes(
        [
            [_row("ok", "first ok")],
            [_row("leaked", "too much")],
        ],
        tie_policy="keep-current",
    )

    applied = apply_majority_to_state(state_path, summaries)

    assert applied == 0
    updated = json.loads(state_path.read_text())
    assert updated["tasks"]["teleport-abc123"]["alignment_verdict"] == "stale"
    assert summaries[0]["tie"] is True
    assert summaries[0]["winner"] == ""


def test_non_ok_status_participates_in_vote() -> None:
    summaries = aggregate_votes(
        [
            [_row("", "missing tests", run_status="skipped")],
            [_row("", "missing tests again", run_status="skipped")],
            [_row("ok", "fine")],
        ]
    )

    assert summaries[0]["winner"] == "skipped"
    assert summaries[0]["alignment_reason"] == "missing tests"
