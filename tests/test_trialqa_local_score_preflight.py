# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_score_preflight as score_preflight


def _config(tmp_path: Path) -> score_preflight.ScorePreflightConfig:
    return score_preflight.ScorePreflightConfig(
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
        reference_targets=tmp_path / "reference.json",
        runbook=tmp_path / "runbook.md",
        skills_distillation_repo=tmp_path / "skills-distillation",
        readiness_output=tmp_path / "readiness.json",
        score_summary_output=tmp_path / "score-summary.json",
        promotion_gate_output=tmp_path / "promotion-gate.json",
        status_output=tmp_path / "status.json",
        protocol_audit_output=tmp_path / "protocol-audit.json",
        reference_alignment_output=tmp_path / "reference-alignment.json",
        audit_bundle_output=tmp_path / "bundle.json",
        audit_bundle_verification_output=tmp_path / "bundle-verification.json",
    )


def test_score_preflight_runs_no_spend_steps_in_dependency_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    order: list[str] = []
    demo._write_json_atomic(
        config.manifest,
        {
            "schema_version": "switchyard.trialqa_experiment_manifest.v1",
            "manifest_id": "trialqa-full-test",
            "protocol": {
                "primary_evaluation_scope": {
                    "question_start": 0,
                    "question_count": 8,
                    "repeat_count": 5,
                },
            },
        },
    )

    def fake_run_score_canary(
        _config: object,
        *,
        yes_spend: bool,
        python: str | Path,
        summary_output: Path | None,
    ) -> dict[str, object]:
        order.append("score_canary")
        assert yes_spend is False
        assert python == "python"
        assert summary_output == config.score_summary_output
        summary = {
            "schema_version": "switchyard.trialqa_canary_score_driver.v1",
            "status": "awaiting_spend_authorization",
            "spend_authorized": False,
            "operational_decision": "promote_to_score",
            "authorized_rerun_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_canary_score",
                "--yes-spend",
            ],
        }
        demo._write_json_atomic(config.score_summary_output, summary)
        return summary

    def fake_build_status_report(**_kwargs: object) -> dict[str, object]:
        order.append("status")
        return {"schema_version": "switchyard.trialqa_protocol_status.v1"}

    def fake_build_protocol_audit(**_kwargs: object) -> dict[str, object]:
        order.append("protocol_audit")
        return {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "completion_state": "awaiting_score_canary_spend_authorization",
            "next_command": {
                "kind": "guarded_score_canary",
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
        }

    def fake_build_audit_bundle(**_kwargs: object) -> dict[str, object]:
        order.append("audit_bundle")
        return {"schema_version": "switchyard.trialqa_pre_spend_audit_bundle.v1"}

    def fake_build_reference_alignment(_config: object) -> dict[str, object]:
        order.append("reference_alignment")
        assert (
            _config.skills_distillation_repo  # type: ignore[attr-defined]
            == config.skills_distillation_repo
        )
        return {
            "schema_version": "switchyard.trialqa_reference_alignment.v1",
            "canary_alignment_status": "proved",
        }

    def fake_verify_audit_bundle(*, bundle_path: Path) -> dict[str, object]:
        order.append("bundle_verify")
        assert bundle_path == config.audit_bundle_output
        return {
            "schema_version": "switchyard.trialqa_pre_spend_audit_bundle_verification.v1",
            "status": "passed",
            "bundle": {
                "manifest_id": "trialqa-full-test",
                "bundle_state": "awaiting_score_canary_spend_authorization",
            },
            "next_command": {
                "kind": "guarded_score_canary",
                "command": [
                    "python",
                    "-m",
                    "benchmark.trialqa_local_canary_score",
                    "--yes-spend",
                ],
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
        }

    monkeypatch.setattr(
        score_preflight.canary_score,
        "run_score_canary",
        fake_run_score_canary,
    )
    monkeypatch.setattr(
        score_preflight.status,
        "build_status_report",
        fake_build_status_report,
    )
    monkeypatch.setattr(
        score_preflight.protocol_audit,
        "build_protocol_audit",
        fake_build_protocol_audit,
    )
    monkeypatch.setattr(
        score_preflight.reference_alignment,
        "build_reference_alignment",
        fake_build_reference_alignment,
    )
    monkeypatch.setattr(
        score_preflight.audit_bundle,
        "build_audit_bundle",
        fake_build_audit_bundle,
    )
    monkeypatch.setattr(
        score_preflight.bundle_verify,
        "verify_audit_bundle",
        fake_verify_audit_bundle,
    )

    report = score_preflight.run_score_preflight(config, python="python")

    assert order == [
        "score_canary",
        "status",
        "protocol_audit",
        "reference_alignment",
        "audit_bundle",
        "bundle_verify",
    ]
    assert report["schema_version"] == score_preflight.SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["spend_authorized"] is False
    assert report["bundle_state"] == "awaiting_score_canary_spend_authorization"
    assert report["next_command"]["kind"] == "guarded_score_canary"
    assert report["next_command"]["authorized_by_audit"] is False
    checkpoint = report["post_spend_checkpoint_command"]
    assert checkpoint["kind"] == "post_score_checkpoint"
    assert checkpoint["requires_spend"] is False
    assert checkpoint["contains_yes_spend"] is False
    assert checkpoint["command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_score_checkpoint",
    ]
    assert "--yes-spend" not in checkpoint["command"]
    assert checkpoint["command"][checkpoint["command"].index("--promotion-gate") + 1] == str(
        config.promotion_gate_output
    )
    assert checkpoint["command"][checkpoint["command"].index("--skills-distillation-repo") + 1] == str(
        config.skills_distillation_repo
    )
    assert checkpoint["command"][checkpoint["command"].index("--expansion-readiness-output") + 1] == str(
        config.audit_bundle_output.parent / "readiness-status-q0-q7-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--ladder-rehearsal") + 1] == str(
        config.audit_bundle_output.parent / "ladder-rehearsal-status.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--goal-audit-output") + 1] == str(
        config.audit_bundle_output.parent / "goal-audit-status-q0-q7-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--decision-summary-output") + 1] == str(
        config.audit_bundle_output.parent / "decision-summary-status-q0-q7-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--output") + 1] == str(
        config.audit_bundle_output.parent / "score-checkpoint-status-q0-q3-r1.json"
    )
    for path in (
        config.score_summary_output,
        config.status_output,
        config.protocol_audit_output,
        config.reference_alignment_output,
        config.audit_bundle_output,
        config.audit_bundle_verification_output,
    ):
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"]


def test_score_preflight_rejects_unpromoted_score_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_run_score_canary(_config: object, **_kwargs: object) -> dict[str, object]:
        calls.append("score_canary")
        return {
            "schema_version": "switchyard.trialqa_canary_score_driver.v1",
            "status": "operational_gate_not_promoted",
            "spend_authorized": False,
            "operational_decision": "kill",
            "authorized_rerun_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_canary_score",
                "--yes-spend",
            ],
        }

    monkeypatch.setattr(
        score_preflight.canary_score,
        "run_score_canary",
        fake_run_score_canary,
    )

    with pytest.raises(score_preflight.TrialQAScorePreflightError, match="not ready"):
        score_preflight.run_score_preflight(config, python="python")

    assert calls == ["score_canary"]
    assert not config.status_output.exists()
