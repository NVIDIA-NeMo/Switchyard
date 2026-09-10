# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

MISSING = object()
BIG = "nvidia/nemotron-3-super-120b-a12b"
SMALL = "openai/gpt-oss-20b"
FILES = {
    "inputs": "rollouts_materialized_inputs.jsonl",
    "rows": "rollouts.jsonl",
    "failures": "rollouts_failures.jsonl",
    "condition": "switchyard-condition.json",
    "snapshot": "switchyard-stats.json",
    "commit": "gym-commit.txt",
}


@pytest.fixture
def comparator() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "benchmark/nemo_gym/compare.py"
    spec = importlib.util.spec_from_file_location("switchyard_nemo_gym_compare", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts() -> dict[str, dict[str, Any]]:
    runs = {}
    for route in ("fixed", "routed"):
        inputs, rows = [], []
        for index in range(2):
            task = {
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "expected_answer": "B",
                "grading_mode": "strict_single_letter_boxed",
                "agent_ref": {"name": "mcqa_simple_agent"},
                "responses_create_params": {
                    "input": [{"role": "user", "content": f"Question {index}"}],
                    "temperature": 0,
                    "max_output_tokens": 512,
                },
            }
            inputs.append(task)
            response = {
                "status": "completed",
                "model": SMALL if route == "routed" and index == 0 else BIG,
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
            }
            rows.append(
                {
                    **deepcopy(task),
                    "response": dict(deepcopy(response), model=route),
                    "reward": 1 - index,
                    "ng_perf": {"total_latency_ms": 100 + 200 * index},
                    "ng_model_call_capture": {
                        "gaps": [],
                        "calls": [
                            {
                                "status_code": 200,
                                "error_category": None,
                                "response_status": "completed",
                                "model": response["model"],
                                "tokens_total": 30 + 10 * index,
                            }
                        ],
                    },
                }
            )
        runs[route] = {
            "inputs": inputs,
            "rows": rows,
            "failures": [],
            "commit": "3a26c35fa90c243427378569511f7b06f503e0fd",
            "condition": {
                "route": route,
                "mode": "hosted",
                "nemo_switchyard_version": "0.2.0",
                "deployment_sha256": "a" * 64,
            },
            "snapshot": {
                "mode": "hosted",
                "scope": "this run (proxy hosted for exactly this run)",
                "stats": {
                    "total_errors": 0,
                    "total_requests": 2,
                    "total_tokens": {"total": 70},
                    "routing_overhead": {"avg_ms": 5},
                    "classifier": {
                        "total_errors": 0,
                        "total_requests": 2 if route == "routed" else 0,
                        "total_tokens": {"total": 11 if route == "routed" else 0},
                    },
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
            if filename.endswith(".jsonl"):
                text = "".join(json.dumps(row) + "\n" for row in value)
            else:
                text = value + "\n" if name == "commit" else json.dumps(value)
            (directory / filename).write_text(text, encoding="utf-8")
    return [str(tmp_path / route) for route in ("fixed", "routed")]


def test_reordered_pairing_uses_captured_models_and_separate_classifier_tokens(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts["routed"]["rows"].reverse()
    artifacts["routed"]["inputs"].reverse()
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Pairing: matched=2, fixed-only=0, routed-only=0" in output.out
    assert f'fixed selected models: {{"{BIG}": 2}}' in output.out
    assert f"routed selected models: {json.dumps({SMALL: 1, BIG: 1}, sort_keys=True)}" in output.out
    expected = {
        "Mean reward": ["0.500", "0.500"],
        "Selected-model tokens": ["70", "70"],
        "Classifier tokens": ["0", "11"],
        "Combined reported tokens": ["70", "81"],
        "Mean rollout latency (ms)": ["200", "200"],
        "Mean routing overhead (ms)": ["5", "5"],
        "Classifier requests": ["0", "2"],
    }
    for metric, values in expected.items():
        line = next(line for line in output.out.splitlines() if line.startswith(metric))
        assert line[len(metric) :].split() == values


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("routed.rows", []),
        (
            "routed.failures",
            [{"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "timeout"}],
        ),
        ("routed.rows.1._ng_task_index", 0),
        ("routed.rows.0._ng_task_index", -1),
        ("routed.rows.0._ng_task_index", True),
        ("routed.rows.0._ng_rollout_index", "0"),
        ("routed.rows.0._ng_rollout_index", MISSING),
        ("routed.rows.0._ng_task_index", 99),
        ("routed.inputs.1._ng_task_index", 0),
        ("routed.inputs.0.expected_answer", "A"),
        ("routed.inputs.0.responses_create_params.input.0.content", "Different question"),
        ("routed.inputs.0.responses_create_params.temperature", 1),
        ("routed.inputs.0.responses_create_params.max_output_tokens", 256),
        ("routed.condition.deployment_sha256", "b" * 64),
        ("routed.condition.deployment_sha256", MISSING),
        ("routed.condition.nemo_switchyard_version", "0.3.0"),
        ("routed.commit", "different-commit"),
        ("routed.condition.route", "fixed"),
        ("routed.condition.mode", "external"),
        ("routed.snapshot.mode", "external"),
        ("routed.snapshot.scope", "all runs"),
        ("routed.condition", None),
        ("routed.snapshot", []),
        ("routed.rows.0._ng_failure_class", "timeout"),
        ("routed.rows.0.reward", None),
        ("routed.rows.0.reward", 1.1),
        ("routed.rows.0.response", MISSING),
        ("routed.rows.0.response", None),
        ("routed.rows.0.response.status", "incomplete"),
        ("routed.rows.0.response.output", []),
        ("routed.rows.0.response.output", None),
        ("routed.rows.0.response.output", [None]),
        ("routed.rows.0.response.output.0.role", "user"),
        ("routed.rows.0.response.output.0.content", None),
        ("routed.rows.0.response.output.0.content", [None]),
        ("routed.rows.0.response.output.0.content.0.text", "  "),
        ("routed.rows.0.ng_model_call_capture", MISSING),
        ("routed.rows.0.ng_model_call_capture", None),
        ("routed.rows.0.ng_model_call_capture.gaps", ["missing call"]),
        ("routed.rows.0.ng_model_call_capture.calls", []),
        ("routed.rows.0.ng_model_call_capture.calls", [{}, {}]),
        ("routed.rows.0.ng_model_call_capture.calls", [None]),
        ("routed.rows.0.ng_model_call_capture.calls.0.status_code", 500),
        ("routed.rows.0.ng_model_call_capture.calls.0.error_category", "timeout"),
        ("routed.rows.0.ng_model_call_capture.calls.0.response_status", None),
        ("routed.rows.0.ng_model_call_capture.calls.0.response_status", "incomplete"),
        ("routed.rows.0.ng_model_call_capture.calls.0.model", "routed"),
        ("routed.rows.0.ng_model_call_capture.calls.0.model", MISSING),
        ("routed.rows.0.ng_model_call_capture.calls.0.tokens_total", None),
        ("routed.rows.0.ng_perf.total_latency_ms", None),
        ("routed.snapshot.stats.total_requests", 3),
        ("routed.snapshot.stats.total_tokens.total", 81),
        ("routed.snapshot.stats.classifier.total_requests", 0),
        ("fixed.snapshot.stats.classifier.total_requests", 2),
        ("routed.snapshot.stats.classifier.total_tokens.total", None),
    ],
)
def test_rejects_incompatible_or_malformed_artifacts(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    capsys: pytest.CaptureFixture[str],
    path: str,
    value: Any,
) -> None:
    keys = [int(key) if key.isdecimal() else key for key in path.split(".")]
    target = artifacts
    for key in keys[:-1]:
        target = target[key]
    if value is MISSING:
        del target[keys[-1]]
    else:
        target[keys[-1]] = value
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 1
    output = capsys.readouterr()
    assert "Cannot compare:" in output.err
    assert "Mean reward" not in output.out


@pytest.mark.parametrize("side", ["fixed", "routed"])
@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("value", [MISSING, None, 1])
def test_error_counts_must_be_present_and_zero(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    side: str,
    classifier: bool,
    value: Any,
) -> None:
    stats = artifacts[side]["snapshot"]["stats"]
    target = stats["classifier"] if classifier else stats
    if value is MISSING:
        del target["total_errors"]
    else:
        target["total_errors"] = value
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 1


@pytest.mark.parametrize("sides", [("fixed",), ("routed",), ("fixed", "routed")])
def test_missing_expected_row_rejects_even_when_both_sides_match(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    capsys: pytest.CaptureFixture[str],
    sides: tuple,
) -> None:
    for side in sides:
        artifacts[side]["rows"].pop()
    assert comparator.main(_write_runs(tmp_path, artifacts)) == 1
    output = capsys.readouterr()
    assert "missing=1" in output.out
    assert "Incomplete runs" in output.err
    assert "Mean reward" not in output.out


@pytest.mark.parametrize(
    ("filename", "hint"),
    [
        ("switchyard-condition.json", "when the model server starts"),
        ("switchyard-stats.json", "press Ctrl-C"),
    ],
)
def test_missing_server_artifacts_explain_the_required_step(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    capsys: pytest.CaptureFixture[str],
    filename: str,
    hint: str,
) -> None:
    args = _write_runs(tmp_path, artifacts)
    missing = Path(args[1]) / filename
    missing.unlink()
    assert comparator.main(args) == 1
    output = capsys.readouterr()
    assert str(missing) in output.err
    assert hint in output.err
    assert "Mean reward" not in output.out
    assert not missing.exists()


def test_misfiled_manifest_explains_route_directory_mismatch(
    tmp_path: Path,
    comparator: ModuleType,
    artifacts: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts["fixed"]["condition"]["route"] = "routed"
    args = _write_runs(tmp_path, artifacts)
    assert comparator.main(args) == 1
    output = capsys.readouterr()
    assert "fixed: manifest records route 'routed'" in output.err
    assert "fresh directory" in output.err
    assert "Mean reward" not in output.out
    manifest = json.loads((Path(args[0]) / "switchyard-condition.json").read_text())
    assert manifest["route"] == "routed"


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash is not installed")
@pytest.mark.parametrize("block_index", [0, 1])
def test_routed_commands_reset_a_stale_fixed_output_directory(
    tmp_path: Path,
    block_index: int,
) -> None:
    readme = Path(__file__).resolve().parents[1] / "benchmark/nemo_gym/README.md"
    section = readme.read_text(encoding="utf-8").split("## 4. Repeat with routing", 1)[1]
    section = section.split("## 5.", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)\n```", section, re.DOTALL)
    assert len(blocks) == 2
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = subprocess.run(
        [
            "bash",
            "-c",
            "gym() { printf '%s\\n' \"$@\"; }; git() { printf 'test-revision\\n'; };\n"
            + blocks[block_index],
        ],
        env={
            "PATH": os.defpath,
            "EXAMPLE": str(tmp_path / "example"),
            "WORK": str(tmp_path / "work"),
            "RUN_DIR": str(run_dir),
            "OUT": str(run_dir / "fixed"),
            "ROUTE": "fixed",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert f"++model_call_capture_dir={run_dir}/routed/model-calls" in args
    if block_index == 0:
        assert args[args.index("--model") + 1] == "routed"
        assert (
            f"++policy_model.responses_api_models.switchyard_model.condition_dir={run_dir}/routed"
            in args
        )
        assert (run_dir / "routed/gym-commit.txt").read_text() == "test-revision\n"
    else:
        assert args[args.index("--output") + 1] == str(run_dir / "routed/rollouts.jsonl")
    assert not (run_dir / "fixed").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash is not installed")
def test_readme_shell_blocks_have_valid_bash_syntax() -> None:
    readme = Path(__file__).resolve().parents[1] / "benchmark/nemo_gym/README.md"
    blocks = re.findall(r"```bash\n(.*?)\n```", readme.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "The tutorial must contain executable Bash examples"
    for index, block in enumerate(blocks, start=1):
        result = subprocess.run(
            ["bash", "-n"], input=block, text=True, capture_output=True, check=False, timeout=10
        )
        assert result.returncode == 0, f"Bash example {index}: {result.stderr}"
