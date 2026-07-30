# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

import pandas as pd

from scripts.agent_pass_rate_matrix import build_joined_df, build_matrix, pretty_summary


def _write_state(path) -> None:
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "task-1": {
                        "task_id": "task-1",
                        "stage": "promising",
                        "alignment_verdict": "ok",
                        "candidate_data": {
                            "source_task_id": "instance_1",
                            "source_metadata": {"instance_id": "instance_1"},
                        },
                    },
                    "task-2": {
                        "task_id": "task-2",
                        "stage": "rejected",
                        "alignment_verdict": "leaked",
                        "candidate_data": {
                            "source_task_id": "instance_2",
                            "source_metadata": {"instance_id": "instance_2"},
                        },
                    },
                }
            }
        )
    )


def _write_agent_runs(path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metadata.instance_id",
                "metadata.model_name",
                "metadata.resolved",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metadata.instance_id": "instance_1",
                "metadata.model_name": "Model A -- 10222025",
                "metadata.resolved": "true",
            }
        )
        writer.writerow(
            {
                "metadata.instance_id": "instance_1",
                "metadata.model_name": "Model B",
                "metadata.resolved": "false",
            }
        )
        writer.writerow(
            {
                "metadata.instance_id": "instance_2",
                "metadata.model_name": "Model A -- 10222025",
                "metadata.resolved": "false",
            }
        )
        writer.writerow(
            {
                "metadata.instance_id": "instance_2",
                "metadata.model_name": "Model B",
                "metadata.resolved": "true",
            }
        )


def test_build_matrix_groups_by_stage_alignment_and_model(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    runs_path = tmp_path / "runs.csv"
    _write_state(state_path)
    _write_agent_runs(runs_path)

    joined = build_joined_df(runs_path, state_path)
    matrix = build_matrix(joined)

    row = matrix[(matrix["stage"] == "promising") & (matrix["alignment_verdict"] == "ok")].iloc[0]
    assert row["matched_agent_runs"] == 2
    assert row["matched_instances"] == 1
    assert row["Model A pass_rate"] == 1.0
    assert row["Model A runs"] == 1
    assert row["Model B pass_rate"] == 0.0
    assert row["Model B runs"] == 1

    row = matrix[(matrix["stage"] == "rejected") & (matrix["alignment_verdict"] == "leaked")].iloc[0]
    assert row["Model A pass_rate"] == 0.0
    assert row["Model B pass_rate"] == 1.0


def test_build_matrix_can_emit_percent_without_counts() -> None:
    joined = pd.DataFrame(
        [
            {
                "instance_id": "instance_1",
                "model_label": "Model A",
                "resolved": True,
                "stage": "promising",
                "alignment_verdict": "ok",
            },
            {
                "instance_id": "instance_2",
                "model_label": "Model A",
                "resolved": False,
                "stage": "promising",
                "alignment_verdict": "ok",
            },
        ]
    )

    matrix = build_matrix(joined, include_counts=False, as_percent=True)

    row = matrix[(matrix["stage"] == "promising") & (matrix["alignment_verdict"] == "ok")].iloc[0]
    assert row["Model A pass_rate"] == 50.0
    assert "Model A runs" not in matrix.columns


def test_pretty_summary_prints_model_blocks() -> None:
    joined = pd.DataFrame(
        [
            {
                "instance_id": "instance_1",
                "model_label": "Claude Opus 4.5",
                "resolved": True,
                "stage": "accepted",
                "alignment_verdict": "ok",
            },
            {
                "instance_id": "instance_2",
                "model_label": "Claude Opus 4.5",
                "resolved": False,
                "stage": "rejected",
                "alignment_verdict": "narrow_tests",
            },
        ]
    )

    text = pretty_summary(joined)

    assert "-----Claude Opus 4.5-----" in text
    assert "overall: 50.0% (1/2)" in text
    assert "Rejected: 0.0% (0/1)" in text
    assert "Accepted: 100.0% (1/1)" in text
    assert "Narrow test: 0.0% (0/1)" in text
    assert "OK: 100.0% (1/1)" in text
