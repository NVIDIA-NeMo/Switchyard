# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_generation_checkpoint as generation_checkpoint


def _config(tmp_path: Path) -> generation_checkpoint.GenerationCheckpointConfig:
    return generation_checkpoint.GenerationCheckpointConfig(
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
        readiness=tmp_path / "readiness.json",
        operational_gate=tmp_path / "operational-gate.json",
        question_start=0,
        question_limit=4,
        repeat_limit=1,
        workers=4,
        max_generation_attempts=1,
        reference_targets=tmp_path / "reference.json",
        runbook=tmp_path / "runbook.md",
        artifact_dir=tmp_path / "artifacts",
        artifact_stem="ctgov-prospective-v1-compact-v5",
        status_output=tmp_path / "status.json",
        next_step_output=tmp_path / "next-step.json",
        score_summary_output=tmp_path / "score-summary.json",
        promotion_gate_output=tmp_path / "promotion-gate.json",
        protocol_audit_output=tmp_path / "protocol-audit.json",
        reference_alignment_output=tmp_path / "reference-alignment.json",
        audit_bundle_output=tmp_path / "bundle.json",
        audit_bundle_verification_output=tmp_path / "bundle-verification.json",
        score_preflight_output=tmp_path / "score-preflight.json",
        score_progress_output=tmp_path / "score-progress.json",
        spend_review_output=tmp_path / "spend-review.json",
        skills_distillation_repo=tmp_path / "skills-distillation",
    )


def test_generation_checkpoint_propagates_custom_ladder_path(tmp_path: Path) -> None:
    ladder_rehearsal = tmp_path / "custom-ladder.json"
    config = replace(_config(tmp_path), ladder_rehearsal=ladder_rehearsal)

    next_config = generation_checkpoint._next_step_config(config)
    score_config = generation_checkpoint._score_preflight_config(config)

    assert next_config.ladder_rehearsal == ladder_rehearsal
    assert score_config.ladder_rehearsal == ladder_rehearsal


def test_generation_checkpoint_kill_path_does_not_run_score_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_build_status_report(**kwargs: object) -> dict[str, object]:
        calls.append("status")
        assert kwargs["operational_gate_path"] == config.operational_gate
        return {"schema_version": "switchyard.trialqa_protocol_status.v1"}

    def fake_build_next_step_plan(_config: object, *, python: str | Path) -> dict[str, object]:
        calls.append("next_step")
        assert python == "python"
        return {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": True,
            "action": "kill_candidate",
            "decision": "kill_candidate",
            "reason": "operational decision is 'kill_candidate'",
            "safe_next_command": None,
        }

    def unexpected_score_preflight(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("score preflight must not run after a terminal kill")

    def unexpected_spend_review(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("spend review must not run after a terminal kill")

    monkeypatch.setattr(
        generation_checkpoint.status,
        "build_status_report",
        fake_build_status_report,
    )
    monkeypatch.setattr(
        generation_checkpoint.next_step,
        "build_next_step_plan",
        fake_build_next_step_plan,
    )
    monkeypatch.setattr(
        generation_checkpoint.score_preflight,
        "run_score_preflight",
        unexpected_score_preflight,
    )
    monkeypatch.setattr(
        generation_checkpoint.spend_review,
        "build_spend_review_packet",
        unexpected_spend_review,
    )

    report = generation_checkpoint.run_generation_checkpoint(config, python="python")

    assert calls == ["status", "next_step"]
    assert report["schema_version"] == generation_checkpoint.SCHEMA_VERSION
    assert report["status"] == "terminal_no_score_spend_boundary"
    assert report["decision"] == "kill_candidate"
    assert report["spend_authorized"] is False
    assert json.loads(config.status_output.read_text(encoding="utf-8"))["schema_version"]
    assert json.loads(config.next_step_output.read_text(encoding="utf-8"))["schema_version"]
    assert not config.score_preflight_output.exists()
    assert not config.score_progress_output.exists()
    assert not config.spend_review_output.exists()


def test_generation_checkpoint_promote_path_runs_score_preflight_and_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_build_status_report(**kwargs: object) -> dict[str, object]:
        calls.append("status")
        assert kwargs["manifest_path"] == config.manifest
        assert kwargs["readiness_path"] == config.readiness
        assert kwargs["reference_targets_path"] == config.reference_targets
        assert kwargs["operational_gate_path"] == config.operational_gate
        return {
            "schema_version": "switchyard.trialqa_protocol_status.v1",
            "next_action": {"action": "run_guarded_score_canary"},
        }

    def fake_build_next_step_plan(
        next_config: Any,
        *,
        python: str | Path,
    ) -> dict[str, object]:
        calls.append("next_step")
        assert python == "python"
        assert next_config.status == config.status_output
        assert next_config.operational_gate == config.operational_gate
        assert next_config.skills_distillation_repo == config.skills_distillation_repo
        return {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "run_guarded_score_canary",
            "safe_next_command": {
                "kind": "score_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_score_preflight"],
            },
        }

    def fake_run_score_preflight(
        score_config: Any,
        *,
        python: str | Path,
    ) -> dict[str, object]:
        calls.append("score_preflight")
        assert python == "python"
        assert score_config.operational_gate == config.operational_gate
        assert score_config.readiness_output == config.readiness
        assert score_config.status_output == config.status_output
        assert score_config.reference_alignment_output == config.reference_alignment_output
        assert score_config.skills_distillation_repo == config.skills_distillation_repo
        demo._write_json_atomic(
            config.audit_bundle_verification_output,
            {
                "schema_version": (
                    "switchyard.trialqa_pre_spend_audit_bundle_verification.v1"
                ),
                "status": "passed",
            },
        )
        return {
            "schema_version": "switchyard.trialqa_no_spend_score_preflight.v1",
            "status": "passed",
            "spend_authorized": False,
        }

    def fake_build_spend_review_packet(
        *,
        preflight_path: Path,
        bundle_verification_path: Path,
        next_step_path: Path,
        progress_path: Path | None,
        spend_review_path: Path | None,
    ) -> dict[str, object]:
        calls.append("spend_review")
        assert preflight_path == config.score_preflight_output
        assert bundle_verification_path == config.audit_bundle_verification_output
        assert next_step_path == config.next_step_output
        assert progress_path == config.score_progress_output
        assert spend_review_path == config.spend_review_output
        assert json.loads(config.score_progress_output.read_text(encoding="utf-8"))[
            "schema_version"
        ] == "switchyard.trialqa_progress.v1"
        return {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "guarded_spend_command": {
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary_score",
                    "--yes-spend",
                ],
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "safe_no_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_score_preflight"],
                "contains_yes_spend": False,
            },
        }

    def fake_build_progress_report(**kwargs: object) -> dict[str, object]:
        calls.append("score_progress")
        assert kwargs["manifest_path"] == config.manifest
        assert kwargs["experiment_root"] == config.experiment_root
        assert kwargs["stage"] == "score"
        assert kwargs["question_start"] == config.question_start
        assert kwargs["question_limit"] == config.question_limit
        assert kwargs["repeat_limit"] == config.repeat_limit
        return {
            "schema_version": "switchyard.trialqa_progress.v1",
            "manifest_id": "manifest",
            "stage": "score",
            "progress": {"selected_task_count": 8},
            "recommendation": {
                "action": "run_guarded_score_canary_after_spend_review",
                "requires_spend": True,
            },
        }

    monkeypatch.setattr(
        generation_checkpoint.status,
        "build_status_report",
        fake_build_status_report,
    )
    monkeypatch.setattr(
        generation_checkpoint.next_step,
        "build_next_step_plan",
        fake_build_next_step_plan,
    )
    monkeypatch.setattr(
        generation_checkpoint.score_preflight,
        "run_score_preflight",
        fake_run_score_preflight,
    )
    monkeypatch.setattr(
        generation_checkpoint.progress,
        "build_progress_report",
        fake_build_progress_report,
    )
    monkeypatch.setattr(
        generation_checkpoint.spend_review,
        "build_spend_review_packet",
        fake_build_spend_review_packet,
    )

    report = generation_checkpoint.run_generation_checkpoint(config, python="python")

    assert calls == ["status", "next_step", "score_preflight", "score_progress", "spend_review"]
    assert report["schema_version"] == generation_checkpoint.SCHEMA_VERSION
    assert report["status"] == "awaiting_score_spend_authorization"
    assert report["spend_authorized"] is False
    assert report["next_action"] == "run_guarded_score_canary"
    assert report["score_preflight_status"] == "passed"
    assert report["spend_review_status"] == "ready_for_user_spend_decision"
    assert report["pre_spend_guard_check"] == {
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_spend_guard",
            "--spend-review",
            str(config.spend_review_output),
            "--output",
            str(config.spend_review_output.with_name("spend-guard-check-spend-review.json")),
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_spend_guard "
            f"--spend-review {config.spend_review_output} "
            "--output "
            f"{config.spend_review_output.with_name('spend-guard-check-spend-review.json')}"
        ),
        "contains_yes_spend": False,
        "review_note": (
            "Run immediately before approving score spend; this rechecks the "
            "reviewed guarded command, current hash-bound bundle, and selected "
            "ledger/lock progress without making model or judge calls."
        ),
    }
    assert report["guarded_spend_command"]["command"][-1] == "--yes-spend"
    assert json.loads(config.score_preflight_output.read_text(encoding="utf-8"))[
        "schema_version"
    ] == "switchyard.trialqa_no_spend_score_preflight.v1"
    assert json.loads(config.spend_review_output.read_text(encoding="utf-8"))[
        "schema_version"
    ] == "switchyard.trialqa_spend_review_packet.v1"


def test_generation_checkpoint_rejects_unexpected_nonterminal_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        generation_checkpoint.status,
        "build_status_report",
        lambda **_kwargs: {"schema_version": "switchyard.trialqa_protocol_status.v1"},
    )
    monkeypatch.setattr(
        generation_checkpoint.next_step,
        "build_next_step_plan",
        lambda *_args, **_kwargs: {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "expand_generation_scope",
        },
    )

    with pytest.raises(
        generation_checkpoint.TrialQAGenerationCheckpointError,
        match="expected score boundary",
    ):
        generation_checkpoint.run_generation_checkpoint(config, python="python")
