# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import benchmark.trialqa_local_score_checkpoint as score_checkpoint


def _config(tmp_path: Path) -> score_checkpoint.ScoreCheckpointConfig:
    return score_checkpoint.ScoreCheckpointConfig(
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
        current_readiness=tmp_path / "readiness-q0-q3-r1.json",
        operational_gate=tmp_path / "operational-q0-q3-r1.json",
        promotion_gate=tmp_path / "promotion-q0-q3-r1.json",
        workers=4,
        max_generation_attempts=1,
        reference_targets=tmp_path / "reference.json",
        runbook=tmp_path / "runbook.md",
        artifact_dir=tmp_path / "artifacts",
        artifact_stem="ctgov-prospective-v1-compact-v5",
        post_score_status_output=tmp_path / "status-after-score.json",
        next_step_output=tmp_path / "next-step.json",
        expansion_readiness_output=tmp_path / "readiness-q0-q7-r1.json",
        expansion_operational_gate_output=tmp_path / "operational-q0-q7-r1.json",
        generation_summary_output=tmp_path / "generation-summary-q0-q7-r1.json",
        expansion_status_output=tmp_path / "status-q0-q7-r1.json",
        protocol_audit_output=tmp_path / "protocol-audit-q0-q7-r1.json",
        reference_alignment_output=tmp_path / "reference-alignment-q0-q7-r1.json",
        audit_bundle_output=tmp_path / "bundle-q0-q7-r1.json",
        audit_bundle_verification_output=tmp_path / "bundle-verification-q0-q7-r1.json",
        generation_preflight_output=tmp_path / "preflight-q0-q7-r1.json",
        spend_review_output=tmp_path / "spend-review-q0-q7-r1.json",
        skills_distillation_repo=tmp_path / "skills-distillation",
    )


def test_score_checkpoint_propagates_custom_ladder_path(tmp_path: Path) -> None:
    ladder_rehearsal = tmp_path / "custom-ladder.json"
    config = replace(_config(tmp_path), ladder_rehearsal=ladder_rehearsal)

    next_config = score_checkpoint._next_step_config(config)
    preflight_config = score_checkpoint._preflight_config(
        config,
        question_start=0,
        question_limit=8,
        repeat_limit=1,
    )

    assert next_config.ladder_rehearsal == ladder_rehearsal
    assert preflight_config.ladder_rehearsal == ladder_rehearsal


def test_score_checkpoint_terminal_path_does_not_run_generation_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_build_status_report(**kwargs: object) -> dict[str, object]:
        calls.append("status")
        assert kwargs["operational_gate_path"] == config.operational_gate
        assert kwargs["promotion_gate_path"] == config.promotion_gate
        return {"schema_version": "switchyard.trialqa_protocol_status.v1"}

    def fake_build_next_step_plan(_config: Any, *, python: str | Path) -> dict[str, object]:
        calls.append("next_step")
        assert python == "python"
        assert _config.promotion_gate == config.promotion_gate
        return {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": True,
            "action": "kill_candidate",
            "decision": "kill_candidate",
            "reason": "promotion decision is 'kill'",
            "safe_next_command": None,
        }

    monkeypatch.setattr(
        score_checkpoint.status,
        "build_status_report",
        fake_build_status_report,
    )
    monkeypatch.setattr(
        score_checkpoint.next_step,
        "build_next_step_plan",
        fake_build_next_step_plan,
    )
    monkeypatch.setattr(
        score_checkpoint.generation_preflight,
        "run_preflight",
        lambda *_args, **_kwargs: pytest.fail("generation preflight must not run"),
    )
    monkeypatch.setattr(
        score_checkpoint.spend_review,
        "build_spend_review_packet",
        lambda **_kwargs: pytest.fail("spend review must not run"),
    )

    report = score_checkpoint.run_score_checkpoint(config, python="python")

    assert calls == ["status", "next_step"]
    assert report["schema_version"] == score_checkpoint.SCHEMA_VERSION
    assert report["status"] == "terminal_no_generation_expansion_boundary"
    assert report["decision"] == "kill_candidate"
    assert report["spend_authorized"] is False
    assert json.loads(config.post_score_status_output.read_text(encoding="utf-8"))[
        "schema_version"
    ]
    assert json.loads(config.next_step_output.read_text(encoding="utf-8"))["schema_version"]
    assert not config.generation_preflight_output.exists()
    assert not config.spend_review_output.exists()


def test_score_checkpoint_expansion_path_runs_preflight_and_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_build_status_report(**kwargs: object) -> dict[str, object]:
        calls.append("status")
        assert kwargs["readiness_path"] == config.current_readiness
        assert kwargs["operational_gate_path"] == config.operational_gate
        assert kwargs["promotion_gate_path"] == config.promotion_gate
        return {
            "schema_version": "switchyard.trialqa_protocol_status.v1",
            "next_action": {"action": "expand_generation_scope"},
        }

    def fake_build_next_step_plan(next_config: Any, *, python: str | Path) -> dict[str, object]:
        calls.append("next_step")
        assert python == "python"
        assert next_config.status == config.post_score_status_output
        assert next_config.operational_gate == config.operational_gate
        assert next_config.promotion_gate == config.promotion_gate
        assert next_config.skills_distillation_repo == config.skills_distillation_repo
        return {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "expand_generation_scope",
            "scope": {
                "question_start": 0,
                "question_limit": 8,
                "repeat_limit": 1,
                "suffix": "q0-q7-r1",
            },
            "safe_next_command": {
                "kind": "generation_expansion_preflight",
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
            },
        }

    def fake_run_preflight(preflight_config: Any, *, python: str | Path) -> dict[str, object]:
        calls.append("preflight")
        assert python == "python"
        assert preflight_config.question_start == 0
        assert preflight_config.question_limit == 8
        assert preflight_config.repeat_limit == 1
        assert preflight_config.operational_gate == config.operational_gate
        assert preflight_config.promotion_gate == config.promotion_gate
        assert preflight_config.reference_alignment_output == config.reference_alignment_output
        assert preflight_config.skills_distillation_repo == config.skills_distillation_repo
        return {
            "schema_version": "switchyard.trialqa_no_spend_preflight.v1",
            "status": "passed",
            "spend_authorized": False,
        }

    def fake_build_spend_review_packet(
        *,
        preflight_path: Path,
        bundle_verification_path: Path,
        next_step_path: Path,
        spend_review_path: Path | None,
    ) -> dict[str, object]:
        calls.append("spend_review")
        assert preflight_path == config.generation_preflight_output
        assert bundle_verification_path == config.audit_bundle_verification_output
        assert next_step_path == config.next_step_output
        assert spend_review_path == config.spend_review_output
        return {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "guarded_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "requires_yes_spend": True,
            },
            "safe_no_spend_command": {
                "command": ["python", "-m", "benchmark.trialqa_local_preflight"],
                "contains_yes_spend": False,
            },
        }

    monkeypatch.setattr(
        score_checkpoint.status,
        "build_status_report",
        fake_build_status_report,
    )
    monkeypatch.setattr(
        score_checkpoint.next_step,
        "build_next_step_plan",
        fake_build_next_step_plan,
    )
    monkeypatch.setattr(
        score_checkpoint.generation_preflight,
        "run_preflight",
        fake_run_preflight,
    )
    monkeypatch.setattr(
        score_checkpoint.spend_review,
        "build_spend_review_packet",
        fake_build_spend_review_packet,
    )

    report = score_checkpoint.run_score_checkpoint(config, python="python")

    assert calls == ["status", "next_step", "preflight", "spend_review"]
    assert report["schema_version"] == score_checkpoint.SCHEMA_VERSION
    assert report["status"] == "awaiting_generation_expansion_spend_authorization"
    assert report["spend_authorized"] is False
    assert report["scope"] == {"question_start": 0, "question_limit": 8, "repeat_limit": 1}
    assert report["generation_preflight_status"] == "passed"
    assert report["spend_review_status"] == "ready_for_user_spend_decision"
    assert report["pre_spend_guard_check"] == {
        "command": [
            "python",
            "-m",
            "benchmark.trialqa_local_spend_guard",
            "--spend-review",
            str(config.spend_review_output),
            "--output",
            str(config.spend_review_output.with_name("spend-guard-check-q0-q7-r1.json")),
        ],
        "shell_command": (
            "python -m benchmark.trialqa_local_spend_guard "
            f"--spend-review {config.spend_review_output} "
            "--output "
            f"{config.spend_review_output.with_name('spend-guard-check-q0-q7-r1.json')}"
        ),
        "contains_yes_spend": False,
        "review_note": (
            "Run immediately before approving generation-expansion spend; this "
            "rechecks the reviewed guarded command, current hash-bound bundle, "
            "and selected ledger/lock progress without making model or judge calls."
        ),
    }
    assert report["guarded_spend_command"]["command"][-1] == "--yes-spend"
    assert json.loads(config.generation_preflight_output.read_text(encoding="utf-8"))[
        "schema_version"
    ] == "switchyard.trialqa_no_spend_preflight.v1"
    assert json.loads(config.spend_review_output.read_text(encoding="utf-8"))[
        "schema_version"
    ] == "switchyard.trialqa_spend_review_packet.v1"


def test_score_checkpoint_rejects_unexpected_nonterminal_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        score_checkpoint.status,
        "build_status_report",
        lambda **_kwargs: {"schema_version": "switchyard.trialqa_protocol_status.v1"},
    )
    monkeypatch.setattr(
        score_checkpoint.next_step,
        "build_next_step_plan",
        lambda *_args, **_kwargs: {
            "schema_version": "switchyard.trialqa_next_step_plan.v1",
            "terminal": False,
            "action": "run_guarded_score_canary",
        },
    )

    with pytest.raises(
        score_checkpoint.TrialQAScoreCheckpointError,
        match="expected expansion boundary",
    ):
        score_checkpoint.run_score_checkpoint(config, python="python")
