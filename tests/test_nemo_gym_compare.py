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

BIG = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
SMALL = "nvidia_nim/openai/gpt-oss-20b"
FILES = {
    "inputs": "rollouts_materialized_inputs.jsonl",
    "rows": "rollouts.jsonl",
    "failures": "rollouts_failures.jsonl",
    "events": "litellm-calls.jsonl",
    "provenance": "run-provenance.json",
}


@pytest.fixture
def comparator() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "benchmark/nemo_gym/compare.py"
    spec = importlib.util.spec_from_file_location("switchyard_nemo_gym_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attempt(route: str, request_id: str, **fields: Any) -> list[dict[str, Any]]:
    common = {"route": route, "request_id": request_id, "instance_id": "fixture-proxy"}
    return [
        {**common, "event": "start"},
        {
            **common,
            "event": "finish",
            "status_code": 200,
            "error_type": None,
            "response_id": request_id,
            "response_status": "completed",
            "selected_model": BIG,
            "deployment_model": BIG,
            "tokens_total": 30,
            "routing_ms": 5,
            **fields,
        },
    ]


@pytest.fixture
def artifacts() -> dict[str, dict[str, Any]]:
    """Build paired runs with one correct and one wrong answer per condition."""
    runs = {}
    for route in ("fixed", "routed"):
        inputs, rows, events = [], [], []
        for index in range(2):
            task = {
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "expected_answer": "B",
                "agent_ref": {"name": "mmlu-redux_mcqa_simple_agent"},
                "responses_create_params": {
                    "input": [{"role": "user", "content": f"Question {index}"}],
                    "temperature": 0,
                    "max_output_tokens": 4096,
                },
            }
            inputs.append(task)
            response_id = f"{route}-{index}"
            tokens = 30 + 10 * index
            model = SMALL if route == "routed" and index == 0 else BIG
            rows.append(
                {
                    **deepcopy(task),
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "model": route,
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "\\boxed{B}" if index == 0 else "\\boxed{A}",
                                    }
                                ],
                            }
                        ],
                    },
                    "reward": 1 - index,
                    "ng_perf": {"total_latency_ms": 100 + 200 * index},
                    "ng_model_call_capture": {
                        "gaps": [],
                        "calls": [
                            {
                                "response_id": response_id,
                                "status_code": 200,
                                "error_category": None,
                                "response_status": "completed",
                                "model": route,
                                "tokens_total": tokens,
                            }
                        ],
                    },
                }
            )
            events.extend(
                _attempt(
                    route,
                    response_id,
                    tokens_total=tokens,
                    selected_model=model,
                    deployment_model=model,
                )
            )
        runs[route] = {
            "inputs": inputs,
            "rows": rows,
            "failures": [],
            "events": events,
            "provenance": {
                "gym_revision": "b" * 40,
                "switchyard_revision": "c" * 40,
                "runtime": {
                    "mode": "litellm_libsy",
                    "instance_id": "fixture-proxy",
                    "litellm_version": "1.97.0",
                    "switchyard_version": "0.2.0",
                    "routing_plugin": "switchyard_litellm.RandomRoutingPlugin",
                    "models": {"fixed": [BIG], "routed": [BIG, SMALL]},
                    "profile_sha256": "a" * 64,
                    "routing_sha256": "a" * 64,
                    "callback_sha256": "a" * 64,
                    "provider_base_sha256": "a" * 64,
                },
            },
        }
    return runs


def _write_runs(tmp_path: Path, artifacts: dict[str, dict[str, Any]]) -> list[str]:
    for route, run in artifacts.items():
        directory = tmp_path / route
        directory.mkdir()
        for name, filename in FILES.items():
            value = run[name]
            text = (
                "".join(json.dumps(row) + "\n" for row in value)
                if filename.endswith(".jsonl")
                else json.dumps(value)
            )
            (directory / filename).write_text(text, encoding="utf-8")
    return [str(tmp_path / route) for route in ("fixed", "routed")]


def test_complete_reordered_pair(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts["routed"]["rows"].reverse()
    artifacts["routed"]["inputs"].reverse()
    artifacts["routed"]["events"].reverse()
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Pairing: matched=2, fixed-only=0, routed-only=0" in output.out
    assert f'fixed selected models: {{"{BIG}": 2}}' in output.out
    assert f"routed selected models: {json.dumps({SMALL: 1, BIG: 1}, sort_keys=True)}" in output.out
    for metric, values in {
        "Mean reward": ["0.500", "0.500"],
        "Terminal-answer tokens": ["70", "70"],
        "Gateway-reported tokens": ["70", "70"],
        "Mean rollout latency (ms)": ["200", "200"],
        "Gateway requests": ["2", "2"],
        "Gateway errors": ["0", "0"],
    }.items():
        line = next(line for line in output.out.splitlines() if line[:29].rstrip() == metric)
        assert line[29:].split() == values
    assert "Classifier tokens: N/A" in output.out
    assert "not exhaustive provider-attempt" in output.out


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "input",
        "provenance",
        "capture",
        "usage",
        "failure",
        "ledger_gap",
        "instance",
        "selection",
        "gateway_response",
    ],
)
def test_invalid_evidence_never_prints_averages(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    problem: str,
) -> None:
    routed = artifacts["routed"]
    if problem == "missing":
        for run in artifacts.values():
            run["rows"].pop()
    elif problem == "duplicate":
        routed["rows"].append(deepcopy(routed["rows"][0]))
    elif problem == "input":
        routed["inputs"][0]["expected_answer"] = "A"
    elif problem == "provenance":
        routed["provenance"]["gym_revision"] = "d" * 40
    elif problem == "capture":
        routed["rows"][0]["ng_model_call_capture"]["gaps"] = ["missing exchange"]
    elif problem == "usage":
        routed["events"][1]["tokens_total"] = 29
    elif problem == "failure":
        routed["failures"] = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    elif problem == "ledger_gap":
        routed["events"].pop()
    elif problem == "instance":
        routed["events"][1]["instance_id"] = "another-proxy"
    elif problem == "selection":
        routed["events"][1]["deployment_model"] = BIG
    else:
        routed["events"][1]["response_id"] = "not-the-final-response"
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 1
    output = capsys.readouterr()
    assert "Cannot compare:" in output.err
    assert "Mean reward" not in output.out


def test_recovery_keeps_extra_work_and_terminal_attribution(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = artifacts["routed"]
    calls = run["rows"][0]["ng_model_call_capture"]["calls"]
    calls.insert(0, {"status_code": 503, "error_category": "upstream", "tokens_total": None})
    calls.append({**calls[1], "response_id": "superseded", "tokens_total": 20})
    run["events"].extend(
        _attempt(
            "routed",
            "failed",
            status_code=503,
            error_type="ServiceUnavailable",
            response_id=None,
            response_status=None,
            tokens_total=None,
            routing_ms=None,
        )
    )
    run["events"].extend(_attempt("routed", "superseded", tokens_total=20))
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr().out
    for metric, values in {
        "Terminal-answer tokens": ["70", "70"],
        "Gateway-reported tokens": ["70", "90"],
        "Gateway requests": ["2", "4"],
        "Gateway errors": ["0", "1"],
        "Gateway requests w/o usage": ["0", "1"],
        "Captured failed attempts": ["0", "1"],
    }.items():
        line = next(line for line in output.splitlines() if line[:29].rstrip() == metric)
        assert line[29:].split() == values
    assert "WARNING: routed" in output


@pytest.mark.parametrize(
    "problem", ["truncated", "gateway_truncated", "ambiguous", "missing_ledger"]
)
def test_missing_terminal_or_snapshot_has_actionable_error(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict[str, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    problem: str,
) -> None:
    row = artifacts["routed"]["rows"][0]
    if problem == "truncated":
        row["response"]["status"] = "incomplete"
    elif problem == "gateway_truncated":
        artifacts["routed"]["events"][1]["response_status"] = "incomplete"
    elif problem == "ambiguous":
        row["ng_model_call_capture"]["calls"] *= 2
    args = _write_runs(tmp_path, artifacts)
    if problem == "missing_ledger":
        (Path(args[1]) / FILES["events"]).unlink()
    assert comparator.main(args) == 1
    output = capsys.readouterr()
    assert "Mean reward" not in output.out
    if problem == "missing_ledger":
        assert "litellm.log" in output.err
