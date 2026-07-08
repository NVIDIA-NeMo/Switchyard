# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import benchmark.trialqa_local_canary_score as score_driver


def _config(tmp_path: Path) -> score_driver.ScoreConfig:
    return score_driver.ScoreConfig(
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
        operational_gate=tmp_path / "operational-gate.json",
        question_start=0,
        question_limit=4,
        repeat_limit=1,
        workers=4,
        max_generation_attempts=1,
        promotion_gate_output=tmp_path / "promotion-gate.json",
    )


def _write_manifest_and_gate(
    config: score_driver.ScoreConfig,
    *,
    decision: str = "promote_to_score",
) -> None:
    config.manifest.write_text(
        json.dumps({"manifest_id": "trialqa-full-prospective"}) + "\n",
        encoding="utf-8",
    )
    config.operational_gate.write_text(
        json.dumps(
            {
                "schema_version": "switchyard.trialqa_gate_report.v3",
                "gate": "operational",
                "manifest_id": "trialqa-full-prospective",
                "decision": decision,
                "scope": {
                    "selection_attestation": {
                        "question_start": 0,
                        "question_limit": 4,
                        "selected_repeat_indices": [1],
                        "selected_task_count": 8,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_score_driver_is_zero_spend_until_authorized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_manifest_and_gate(config)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    report = score_driver.run_score_canary(
        config,
        yes_spend=False,
        run=fake_run,
        python="python",
    )

    assert calls == []
    assert report["status"] == "awaiting_spend_authorization"
    assert report["operational_decision"] == "promote_to_score"
    assert report["score_command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_batch",
    ]
    assert "score" in report["score_command"]
    assert report["authorized_rerun_command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_canary_score",
    ]
    assert report["authorized_rerun_command"][-1] == "--yes-spend"


def test_score_driver_persists_zero_spend_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_manifest_and_gate(config)
    summary_output = tmp_path / "score-summary.json"

    report = score_driver.run_score_canary(
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


def test_authorized_score_driver_runs_score_then_promotion_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    spend_review = tmp_path / "spend-review.json"
    _write_manifest_and_gate(config)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    guard_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        score_driver.spend_guard,
        "validate_spend_review_for_command",
        lambda **kwargs: guard_calls.append(kwargs) or {"status": "matched"},
    )
    report = score_driver.run_score_canary(
        config,
        yes_spend=True,
        run=fake_run,
        python="python",
        spend_review=spend_review,
    )

    assert guard_calls == [
        {
            "spend_review": spend_review,
            "expected_command": score_driver.guarded_score_canary_command(
                config,
                python="python",
                spend_review=spend_review,
                yes_spend=True,
            ),
            "expected_stage": "score",
            "recover_interrupted": False,
        }
    ]
    assert report["status"] == "promotion_gate_completed"
    assert report["spend_review_guard"] == {"status": "matched"}
    assert calls == [
        score_driver.score_command(config, python="python"),
        score_driver.promotion_gate_command(config, python="python"),
    ]
    assert report["score_returncode"] == 0
    assert report["promotion_gate_returncode"] == 0


def test_score_recovery_flag_passes_through_to_score_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    spend_review = tmp_path / "spend-review.json"
    _write_manifest_and_gate(config)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        score_driver.spend_guard,
        "validate_spend_review_for_command",
        lambda **_kwargs: {"status": "matched"},
    )
    report = score_driver.run_score_canary(
        config,
        yes_spend=True,
        recover_interrupted=True,
        run=fake_run,
        python="python",
        spend_review=spend_review,
    )

    assert report["recover_interrupted"] is True
    assert calls[0] == score_driver.score_command(
        config,
        python="python",
        recover_interrupted=True,
    )
    assert calls[0][-2:] == ["--recover-interrupted", "--retry-failed"]
    assert report["authorized_rerun_command"][-4:] == [
        "--spend-review",
        str(spend_review),
        "--recover-interrupted",
        "--yes-spend",
    ]


def test_score_driver_refuses_when_operational_gate_kills(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_manifest_and_gate(config, decision="kill")

    with pytest.raises(score_driver.TrialQACanaryScoreError, match="did not promote"):
        score_driver.run_score_canary(config, yes_spend=True, python="python")


def test_authorized_score_driver_refuses_without_spend_review(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_manifest_and_gate(config)

    with pytest.raises(score_driver.TrialQACanaryScoreError, match="--spend-review"):
        score_driver.run_score_canary(config, yes_spend=True, python="python")
