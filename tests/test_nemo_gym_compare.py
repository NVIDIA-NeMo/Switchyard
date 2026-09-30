# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

BIG = "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"
SMALL = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"


@pytest.fixture
def comparator() -> ModuleType:
    """Load the offline comparator without the model-serving dependencies."""
    path = Path(__file__).resolve().parents[1] / "benchmark/nemo_gym/compare.py"
    spec = importlib.util.spec_from_file_location("switchyard_nemo_gym_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts() -> dict[str, dict[str, Any]]:
    """Build paired runs with one correct and one wrong answer per condition."""
    runs = {}
    for route in ("fixed", "routed"):
        inputs, rows, models = [], [], []
        for index in range(2):
            task = {
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "expected_answer": "B",
                "responses_create_params": {
                    "input": [{"role": "user", "content": f"Question {index}"}],
                    "temperature": 0,
                    "max_output_tokens": 8192,
                },
            }
            response_id = f"{route}-{index}"
            inputs.append(task)
            rows.append(
                {
                    **deepcopy(task),
                    "reward": 1 - index,
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "\\boxed{B}" if index == 0 else "\\boxed{A}",
                                    }
                                ],
                            }
                        ],
                    },
                    "ng_perf": {"total_latency_ms": 100 + 200 * index},
                    "ng_model_call_capture": {
                        "calls": [
                            {
                                "model_call_id": response_id,
                                "response_id": response_id,
                                "model": route,
                                "status_code": 200,
                                "error_category": None,
                                "response_status": "completed",
                                "tokens_in": 10 + 10 * index,
                                "tokens_out": 20,
                            }
                        ],
                    },
                }
            )
            models.append(
                {
                    "response_id": response_id,
                    "model": SMALL if route == "routed" and index == 0 else BIG,
                }
            )
        runs[route] = {
            "rollouts_materialized_inputs.jsonl": inputs,
            "rollouts.jsonl": rows,
            "rollouts_failures.jsonl": [],
            "models.jsonl": models,
        }
    return runs


def _write_runs(tmp_path: Path, artifacts: dict[str, dict[str, Any]]) -> list[str]:
    for route, files in artifacts.items():
        directory = tmp_path / route
        directory.mkdir()
        for name, rows in files.items():
            (directory / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
    return [str(tmp_path)]


def _assert_metric(output: str, metric: str, values: list[str]) -> None:
    line = next(line for line in output.splitlines() if line[:29].rstrip() == metric)
    assert line[29:].split() == values


def test_complete_reordered_pair(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pair by identity, not file order, and report Gym usage and serving models."""
    for rows in artifacts["routed"].values():
        rows.reverse()
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Pairing: matched=2, fixed-only=0, routed-only=0" in output.out
    assert f'fixed serving models: {{"{BIG}": 2}}' in output.out
    assert f"routed serving models: {json.dumps({SMALL: 1, BIG: 1}, sort_keys=True)}" in output.out
    for metric, values in {
        "Paired rollouts": ["2", "2"],
        "Incomplete responses": ["0", "0"],
        "Mean reward": ["0.500", "0.500"],
        "Captured input tokens": ["30", "30"],
        "Captured output tokens": ["40", "40"],
        "Mean rollout latency (ms)": ["200", "200"],
        "Captured calls": ["2", "2"],
        "Captured failed calls": ["0", "0"],
        "Calls with unknown usage": ["0", "0"],
    }.items():
        _assert_metric(output.out, metric, values)
    assert "WARNING" not in output.out
    assert "Gym-captured" in output.out
    assert "Gateway-reported" not in output.out


@pytest.mark.parametrize(
    ("problem", "expected_error"),
    [
        ("incomplete", "Incomplete runs"),
        ("mismatched", "Task inputs, verifier metadata, or generation settings differ"),
        ("capture", "incomplete model-call capture"),
    ],
)
def test_invalid_evidence_never_prints_averages(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    problem: str,
    expected_error: str,
) -> None:
    """Reject misleading comparisons with a useful diagnostic, not partial averages."""
    routed = artifacts["routed"]
    if problem == "incomplete":
        for run in artifacts.values():
            run["rollouts.jsonl"].pop()
    elif problem == "mismatched":
        routed["rollouts_materialized_inputs.jsonl"][0]["expected_answer"] = "A"
    else:
        routed["rollouts.jsonl"][0]["ng_model_call_capture"]["gaps"] = ["missing exchange"]
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 1
    output = capsys.readouterr()
    assert "Cannot compare:" in output.err
    assert expected_error in output.err
    assert "Mean reward" not in output.out


def test_incomplete_model_response_is_reported(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep a Gym-scored truncation in the comparison and make it visible."""
    row = artifacts["routed"]["rollouts.jsonl"][0]
    row["response"]["status"] = "incomplete"
    row["reward"] = 0
    row["ng_model_call_capture"]["calls"][0]["response_status"] = "incomplete"

    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr().out
    _assert_metric(output, "Incomplete responses", ["0", "1"])
    assert "WARNING: routed" in output
    assert "WARNING: fixed" not in output


def test_recovery_keeps_extra_work_and_terminal_attribution(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Count extra captured work without treating unreported failed usage as zero."""
    calls = artifacts["routed"]["rollouts.jsonl"][0]["ng_model_call_capture"]["calls"]
    calls.insert(
        0,
        {
            "model_call_id": "failed",
            "status_code": 503,
            "error_category": "upstream",
            "tokens_in": None,
            "tokens_out": None,
        },
    )
    calls.append(
        {
            **calls[1],
            "model_call_id": "extra",
            "response_id": "superseded",
            "tokens_in": 7,
            "tokens_out": 13,
        }
    )
    artifacts["routed"]["models.jsonl"].append({"response_id": "superseded", "model": BIG})
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr().out
    for metric, values in {
        "Captured input tokens": ["30", "37+unknown"],
        "Captured output tokens": ["40", "53+unknown"],
        "Captured calls": ["2", "4"],
        "Captured failed calls": ["0", "1"],
        "Calls with unknown usage": ["0", "1"],
    }.items():
        _assert_metric(output, metric, values)
    assert f"routed serving models: {json.dumps({SMALL: 1, BIG: 1}, sort_keys=True)}" in output
    assert "WARNING: routed" in output
    assert "WARNING: fixed" not in output
