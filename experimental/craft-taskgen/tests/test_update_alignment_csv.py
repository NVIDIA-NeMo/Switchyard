# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

from scripts.update_alignment_csv import update_alignment_csv


def _write_state(path):
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "teleport-abc123": {
                        "task_id": "teleport-abc123",
                        "commit_sha": "abc123",
                        "stage": "promising",
                        "alignment_verdict": "ok",
                        "alignment_reason": "looks aligned\nwith tests",
                        "candidate_data": {
                            "source_task_id": "instance_1",
                            "source_metadata": {"instance_id": "instance_1"},
                        },
                    }
                }
            }
        )
    )


def test_update_alignment_csv_adds_plain_columns_and_preserves_duplicate_headers(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    state_json = tmp_path / "state.json"
    output_csv = tmp_path / "output.csv"
    _write_state(state_json)

    with input_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "swebench_instance_id",
                "new_alignment_verdict",
                "new_alignment_reason",
                "stage",
                "new_alignment_verdict",
                "new_alignment_reason",
            ]
        )
        writer.writerow(["teleport-abc123", "instance_1", "old", "old reason", "candidate", "", ""])
        writer.writerow(["missing", "missing_instance", "old", "old reason", "candidate", "", ""])

    stats = update_alignment_csv(input_csv, state_json, output_csv)

    assert stats.rows == 2
    assert stats.matched == 1

    with output_csv.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header.count("new_alignment_verdict") == 2
    assert header.count("new_alignment_reason") == 2
    assert header[4:6] == ["alignment_verdict", "alignment_reason"]

    verdict_idx = header.index("alignment_verdict")
    reason_idx = header.index("alignment_reason")
    assert rows[0][verdict_idx] == "ok"
    assert rows[0][reason_idx] == "looks aligned with tests"
    assert rows[1][verdict_idx] == ""
    assert rows[1][reason_idx] == ""


def test_update_alignment_csv_updates_existing_plain_columns(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    state_json = tmp_path / "state.json"
    _write_state(state_json)

    with input_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "commit_sha", "alignment_verdict", "alignment_reason"])
        writer.writerow(["teleport-abc123", "abc123", "stale", "stale reason"])

    stats = update_alignment_csv(input_csv, state_json, input_csv)

    assert stats.matched == 1
    with input_csv.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["alignment_verdict"] == "ok"
    assert row["alignment_reason"] == "looks aligned with tests"
