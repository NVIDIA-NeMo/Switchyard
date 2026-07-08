# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import benchmark.trialqa_local_canary as canary


def _config(tmp_path: Path) -> canary.CanaryConfig:
    return canary.CanaryConfig(
        manifest=tmp_path / "manifest.json",
        dataset=tmp_path / "dataset.parquet",
        experiment_root=tmp_path / "experiments",
        doctor=tmp_path / "doctor.json",
        population_report=tmp_path / "population.json",
        candidate=tmp_path / "candidate",
        switchyard=tmp_path / "bin" / "switchyard",
        codex=tmp_path / "bin" / "codex",
        tooluniverse=tmp_path / "tooluniverse" / "bin" / "tooluniverse-smcp-stdio",
        profile=tmp_path / "profile.yaml",
        question_start=0,
        question_limit=4,
        repeat_limit=1,
        workers=4,
        max_generation_attempts=1,
        readiness_output=tmp_path / "readiness.json",
        gate_output=tmp_path / "gate.json",
    )


def _ready_report() -> dict[str, object]:
    return {
        "schema_version": "switchyard.trialqa_canary_readiness.v1",
        "status": "ready_for_generation",
        "first_generation_canary": {"task_count": 8, "pair_count": 4},
    }


def test_default_canary_driver_is_zero_spend_until_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(canary, "build_readiness", lambda _config: _ready_report())
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    report = canary.run_canary(config, yes_spend=False, run=fake_run, python="python")

    assert calls == []
    assert report["status"] == "awaiting_spend_authorization"
    assert report["spend_authorized"] is False
    assert report["readiness_status"] == "ready_for_generation"
    assert report["generation_command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_batch",
    ]
    assert "--stage" in report["generation_command"]
    assert "generation" in report["generation_command"]
    assert report["authorized_rerun_command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_canary",
    ]
    assert report["authorized_rerun_command"][-1] == "--yes-spend"


def test_default_canary_driver_accepts_cumulative_expansion_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        canary,
        "build_readiness",
        lambda _config: {
            "status": "ready_for_generation_expansion",
            "first_generation_canary": {"task_count": 16, "pair_count": 8},
        },
    )
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )

    report = canary.run_canary(config, yes_spend=False, python="python")

    assert report["status"] == "awaiting_spend_authorization"
    assert report["readiness_status"] == "ready_for_generation_expansion"


def test_canary_driver_persists_zero_spend_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    summary_output = tmp_path / "summary.json"
    monkeypatch.setattr(canary, "build_readiness", lambda _config: _ready_report())
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )

    report = canary.run_canary(
        config,
        yes_spend=False,
        python="python",
        summary_output=summary_output,
    )

    persisted = json.loads(summary_output.read_text(encoding="utf-8"))
    assert persisted == report
    assert persisted["status"] == "awaiting_spend_authorization"
    assert persisted["spend_authorized"] is False
    assert persisted["authorized_rerun_command"][-3:] == [
        "--summary-output",
        str(summary_output),
        "--yes-spend",
    ]


def test_authorized_canary_runs_generation_then_operational_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    spend_review = tmp_path / "spend-review.json"
    calls: list[list[str]] = []
    guard_calls: list[dict[str, object]] = []
    monkeypatch.setattr(canary, "build_readiness", lambda _config: _ready_report())
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )
    monkeypatch.setattr(
        canary.spend_guard,
        "validate_spend_review_for_command",
        lambda **kwargs: guard_calls.append(kwargs) or {"status": "matched"},
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    report = canary.run_canary(
        config,
        yes_spend=True,
        run=fake_run,
        python="python",
        spend_review=spend_review,
    )

    assert report["status"] == "operational_gate_completed"
    assert report["spend_review_guard"] == {"status": "matched"}
    assert guard_calls == [
        {
            "spend_review": spend_review,
            "expected_command": canary.guarded_canary_command(
                config,
                python="python",
                spend_review=spend_review,
                yes_spend=True,
            ),
            "expected_stage": "generation",
            "recover_interrupted": False,
        }
    ]
    assert calls == [
        canary.generation_command(config, python="python"),
        ["python", "-m", "benchmark.trialqa_local_gate"],
    ]
    assert report["generation_returncode"] == 0
    assert report["operational_gate_returncode"] == 0


def test_canary_recovery_flag_passes_through_to_generation_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    spend_review = tmp_path / "spend-review.json"
    calls: list[list[str]] = []
    monkeypatch.setattr(canary, "build_readiness", lambda _config: _ready_report())
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )
    monkeypatch.setattr(
        canary.spend_guard,
        "validate_spend_review_for_command",
        lambda **_kwargs: {"status": "matched"},
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    report = canary.run_canary(
        config,
        yes_spend=True,
        recover_interrupted=True,
        run=fake_run,
        python="python",
        spend_review=spend_review,
    )

    assert report["recover_interrupted"] is True
    assert calls[0] == canary.generation_command(
        config,
        python="python",
        recover_interrupted=True,
    )
    assert calls[0][-1] == "--recover-interrupted"
    assert report["authorized_rerun_command"][-4:] == [
        "--spend-review",
        str(spend_review),
        "--recover-interrupted",
        "--yes-spend",
    ]


def test_authorized_canary_refuses_to_spend_when_readiness_is_not_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        canary,
        "build_readiness",
        lambda _config: {"status": "has_existing_selected_task_state"},
    )

    with pytest.raises(canary.TrialQACanaryError, match="refusing to spend"):
        canary.run_canary(config, yes_spend=True, python="python")


def test_authorized_canary_refuses_without_spend_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(canary, "build_readiness", lambda _config: _ready_report())
    monkeypatch.setattr(
        canary,
        "operational_gate_command",
        lambda _config, *, python: [str(python), "-m", "benchmark.trialqa_local_gate"],
    )

    with pytest.raises(canary.TrialQACanaryError, match="--spend-review"):
        canary.run_canary(config, yes_spend=True, python="python")
