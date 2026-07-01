# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for joining interleaved routing events to Harbor trials."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

JOINER = Path(__file__).parents[1] / "benchmark" / "routing_trace_joiner.py"


def _load_joiner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("routing_trace_joiner", JOINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_trial(job_dir: Path, trial: str, task: str, reward: float, request_id: str) -> None:
    trial_dir = job_dir / trial
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "trial_name": trial,
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )
    _write_jsonl(trial_dir / "artifacts" / "request_map.jsonl", [{"request_id": request_id}])


def test_join_uses_trial_local_request_maps_for_interleaved_events(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    _write_trial(job_dir, "trial-a", "task-a", 1.0, "request-a")
    _write_trial(job_dir, "trial-b", "task-b", 0.0, "request-b")

    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"request_id": "request-b", "sequence": 0, "name": "router.input", "payload": {}},
            {"request_id": "request-a", "sequence": 0, "name": "router.input", "payload": {}},
            {
                "request_id": "request-b",
                "sequence": 1,
                "name": "router.decision",
                "payload": {"tier": "strong"},
            },
        ],
    )
    assert module.join_routing_traces(job_dir, trace_path) == 2
    outputs = sorted(job_dir.glob("*/artifacts/routing_trace.jsonl"))
    rows = [json.loads(path.read_text()) for path in outputs]
    by_task = {row["task_name"]: row for row in rows}
    assert by_task["task-a"]["request_id"] == "request-a"
    assert by_task["task-a"]["reward"] == 1.0
    assert [event["name"] for event in by_task["task-b"]["events"]] == [
        "router.input",
        "router.decision",
    ]


def test_join_skips_requests_without_a_trace(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    _write_trial(job_dir, "trial-a", "task-a", 1.0, "missing")
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text("")
    assert module.join_routing_traces(job_dir, trace_path) == 0
    output = job_dir / "trial-a" / "artifacts" / "routing_trace.jsonl"
    assert output.read_text() == ""
