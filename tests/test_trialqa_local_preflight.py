# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_preflight as preflight


def _config(tmp_path: Path) -> preflight.PreflightConfig:
    return preflight.PreflightConfig(
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
        reference_targets=tmp_path / "reference.json",
        runbook=tmp_path / "runbook.md",
        skills_distillation_repo=tmp_path / "skills-distillation",
        readiness_output=tmp_path / "readiness.json",
        gate_output=tmp_path / "gate.json",
        generation_summary_output=tmp_path / "generation-summary.json",
        status_output=tmp_path / "status.json",
        protocol_audit_output=tmp_path / "protocol-audit.json",
        reference_alignment_output=tmp_path / "reference-alignment.json",
        audit_bundle_output=tmp_path / "bundle.json",
        audit_bundle_verification_output=tmp_path / "bundle-verification.json",
    )


def test_preflight_runs_no_spend_steps_in_dependency_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    order: list[str] = []

    def fake_run_canary(
        _config: object,
        *,
        yes_spend: bool,
        python: str | Path,
        summary_output: Path | None,
    ) -> dict[str, object]:
        order.append("canary")
        assert yes_spend is False
        assert python == "python"
        assert summary_output == config.generation_summary_output
        demo._write_json_atomic(
            config.readiness_output,
            {"schema_version": "switchyard.trialqa_canary_readiness.v1"},
        )
        summary = {
            "schema_version": "switchyard.trialqa_canary_driver.v1",
            "status": "awaiting_spend_authorization",
            "spend_authorized": False,
            "authorized_rerun_command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
        }
        demo._write_json_atomic(config.generation_summary_output, summary)
        return summary

    def fake_build_status_report(**_kwargs: object) -> dict[str, object]:
        order.append("status")
        return {"schema_version": "switchyard.trialqa_protocol_status.v1"}

    def fake_build_protocol_audit(**_kwargs: object) -> dict[str, object]:
        order.append("protocol_audit")
        return {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "completion_state": "awaiting_generation_canary_spend_authorization",
            "next_command": {
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
                "bundle_state": "awaiting_generation_canary_spend_authorization",
            },
            "next_command": {
                "kind": "guarded_generation_canary",
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
        }

    monkeypatch.setattr(preflight.canary, "run_canary", fake_run_canary)
    monkeypatch.setattr(preflight.status, "build_status_report", fake_build_status_report)
    monkeypatch.setattr(
        preflight.protocol_audit,
        "build_protocol_audit",
        fake_build_protocol_audit,
    )
    monkeypatch.setattr(
        preflight.reference_alignment,
        "build_reference_alignment",
        fake_build_reference_alignment,
    )
    monkeypatch.setattr(preflight.audit_bundle, "build_audit_bundle", fake_build_audit_bundle)
    monkeypatch.setattr(
        preflight.bundle_verify,
        "verify_audit_bundle",
        fake_verify_audit_bundle,
    )

    report = preflight.run_preflight(config, python="python")

    assert order == [
        "canary",
        "status",
        "protocol_audit",
        "reference_alignment",
        "audit_bundle",
        "bundle_verify",
    ]
    assert report["schema_version"] == preflight.SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["spend_authorized"] is False
    assert report["manifest_id"] == "trialqa-full-test"
    assert report["bundle_state"] == "awaiting_generation_canary_spend_authorization"
    assert report["next_command"]["authorized_by_audit"] is False
    checkpoint = report["post_spend_checkpoint_command"]
    assert checkpoint["kind"] == "post_generation_checkpoint"
    assert checkpoint["requires_spend"] is False
    assert checkpoint["contains_yes_spend"] is False
    assert checkpoint["command"][:3] == [
        "python",
        "-m",
        "benchmark.trialqa_local_generation_checkpoint",
    ]
    assert "--yes-spend" not in checkpoint["command"]
    assert checkpoint["command"][checkpoint["command"].index("--operational-gate") + 1] == str(
        config.gate_output
    )
    assert checkpoint["command"][checkpoint["command"].index("--skills-distillation-repo") + 1] == str(
        config.skills_distillation_repo
    )
    assert checkpoint["command"][checkpoint["command"].index("--status-output") + 1] == str(
        config.audit_bundle_output.parent / "status-after-generation-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--status-output") + 1] != str(
        config.status_output
    )
    assert checkpoint["command"][checkpoint["command"].index("--protocol-audit-output") + 1] == str(
        config.audit_bundle_output.parent / "protocol-audit-score-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--protocol-audit-output") + 1] != str(
        config.protocol_audit_output
    )
    assert checkpoint["command"][checkpoint["command"].index("--reference-alignment-output") + 1] == str(
        config.audit_bundle_output.parent / "reference-alignment-score-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--reference-alignment-output") + 1] != str(
        config.reference_alignment_output
    )
    assert checkpoint["command"][checkpoint["command"].index("--score-preflight-output") + 1] == str(
        config.audit_bundle_output.parent / "no-spend-score-preflight-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--score-progress-output") + 1] == str(
        config.audit_bundle_output.parent / "progress-score-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--ladder-rehearsal") + 1] == str(
        config.audit_bundle_output.parent / "ladder-rehearsal-status.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--goal-audit-output") + 1] == str(
        config.audit_bundle_output.parent / "goal-audit-score-status-q0-q3-r1.json"
    )
    assert checkpoint["command"][checkpoint["command"].index("--decision-summary-output") + 1] == str(
        config.audit_bundle_output.parent / "decision-summary-score-status-q0-q3-r1.json"
    )
    for path in (
        config.readiness_output,
        config.generation_summary_output,
        config.status_output,
        config.protocol_audit_output,
        config.reference_alignment_output,
        config.audit_bundle_output,
        config.audit_bundle_verification_output,
    ):
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"]


def test_preflight_rejects_not_ready_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_run_canary(_config: object, **_kwargs: object) -> dict[str, object]:
        calls.append("canary")
        return {
            "schema_version": "switchyard.trialqa_canary_driver.v1",
            "status": "readiness_not_clean",
            "spend_authorized": False,
            "authorized_rerun_command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
        }

    monkeypatch.setattr(preflight.canary, "run_canary", fake_run_canary)

    with pytest.raises(preflight.TrialQAPreflightError, match="not ready"):
        preflight.run_preflight(config, python="python")

    assert calls == ["canary"]
    assert not config.status_output.exists()


def test_preflight_preserves_gate_context_for_generation_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        operational_gate=tmp_path / "operational.json",
        promotion_gate=tmp_path / "promotion.json",
    )
    seen: dict[str, object] = {}

    def fake_run_canary(
        _config: object,
        *,
        yes_spend: bool,
        python: str | Path,
        summary_output: Path | None,
    ) -> dict[str, object]:
        assert yes_spend is False
        assert python == "python"
        assert summary_output == config.generation_summary_output
        summary = {
            "schema_version": "switchyard.trialqa_canary_driver.v1",
            "status": "awaiting_spend_authorization",
            "spend_authorized": False,
            "authorized_rerun_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_canary",
                "--yes-spend",
            ],
        }
        demo._write_json_atomic(config.generation_summary_output, summary)
        return summary

    def fake_build_status_report(**kwargs: object) -> dict[str, object]:
        seen["status_kwargs"] = kwargs
        return {
            "schema_version": "switchyard.trialqa_protocol_status.v1",
            "next_action": {"action": "expand_generation_scope"},
        }

    def fake_build_protocol_audit(**kwargs: object) -> dict[str, object]:
        seen["audit_kwargs"] = kwargs
        return {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "completion_state": "awaiting_generation_canary_spend_authorization",
            "next_command": {
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
        }

    monkeypatch.setattr(preflight.canary, "run_canary", fake_run_canary)
    monkeypatch.setattr(preflight.status, "build_status_report", fake_build_status_report)
    monkeypatch.setattr(
        preflight.protocol_audit,
        "build_protocol_audit",
        fake_build_protocol_audit,
    )
    monkeypatch.setattr(
        preflight.reference_alignment,
        "build_reference_alignment",
        lambda _config: {
            "schema_version": "switchyard.trialqa_reference_alignment.v1",
            "canary_alignment_status": "proved",
        },
    )
    monkeypatch.setattr(
        preflight.audit_bundle,
        "build_audit_bundle",
        lambda **_kwargs: {"schema_version": "switchyard.trialqa_pre_spend_audit_bundle.v1"},
    )
    monkeypatch.setattr(
        preflight.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: {
            "schema_version": "switchyard.trialqa_pre_spend_audit_bundle_verification.v1",
            "status": "passed",
            "bundle": {
                "manifest_id": "trialqa-full-test",
                "bundle_state": "awaiting_generation_canary_spend_authorization",
            },
            "next_command": {
                "kind": "guarded_generation_canary",
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "requires_yes_spend": True,
                "authorized_by_audit": False,
            },
        },
    )

    report = preflight.run_preflight(config, python="python")

    status_kwargs = seen["status_kwargs"]
    audit_kwargs = seen["audit_kwargs"]
    assert status_kwargs["operational_gate_path"] == config.operational_gate
    assert status_kwargs["promotion_gate_path"] == config.promotion_gate
    assert audit_kwargs["operational_gate_path"] == config.operational_gate
    assert report["status"] == "passed"
