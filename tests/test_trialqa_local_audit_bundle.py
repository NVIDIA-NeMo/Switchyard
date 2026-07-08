# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_audit_bundle as bundle
import benchmark.trialqa_local_demo as demo


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, schema: str) -> Path:
    return _write_json(path, {"schema_version": schema})


def _reference_targets(path: Path, source_document: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_targets.v1",
            "source": {
                "document": source_document.name,
                "document_sha256": demo._sha256_file(source_document),
            },
        },
    )


def _protocol_audit(path: Path, *, required_status: str = "proved") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "manifest_id": "trialqa-full-test",
            "completion_state": "awaiting_generation_canary_spend_authorization",
            "requirements": [
                {
                    "id": "guarded_generation_dry_run_persisted",
                    "status": required_status,
                    "required_for_spend": True,
                    "evidence": "fixture",
                },
                {
                    "id": "quality_parity_evidence",
                    "status": "missing",
                    "required_for_spend": False,
                    "evidence": "fixture",
                },
            ],
            "next_command": {
                "kind": "guarded_generation_canary",
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "shell_command": "python -m benchmark.trialqa_local_canary --yes-spend",
                "requires_yes_spend": True,
                "authorized_by_audit": False,
                "source": "summary.json",
            },
        },
    )


def _score_protocol_audit(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_audit.v1",
            "manifest_id": "trialqa-full-test",
            "completion_state": "awaiting_score_canary_spend_authorization",
            "requirements": [
                {
                    "id": "guarded_score_dry_run_persisted",
                    "status": "proved",
                    "required_for_spend": True,
                    "evidence": "fixture",
                },
                {
                    "id": "operational_generation_gate_completed",
                    "status": "proved",
                    "required_for_spend": False,
                    "evidence": "fixture",
                },
            ],
            "next_command": {
                "kind": "guarded_score_canary",
                "command": ["python", "-m", "benchmark.trialqa_local_canary_score", "--yes-spend"],
                "shell_command": "python -m benchmark.trialqa_local_canary_score --yes-spend",
                "requires_yes_spend": True,
                "authorized_by_audit": False,
                "source": "score-summary.json",
            },
        },
    )


def test_audit_bundle_hashes_artifacts_and_preserves_next_command(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _protocol_audit(tmp_path / "protocol-audit.json")
    reference_document = tmp_path / "reference.pdf"
    reference_document.write_bytes(b"reference slide")
    reference = _reference_targets(tmp_path / "reference.json", reference_document)
    reference_alignment = _artifact(
        tmp_path / "reference-alignment.json",
        "switchyard.trialqa_reference_alignment.v1",
    )
    summary = _artifact(tmp_path / "summary.json", "switchyard.trialqa_canary_driver.v1")
    runbook = tmp_path / "runbook.md"
    runbook.write_text("runbook\n", encoding="utf-8")

    report = bundle.build_audit_bundle(
        manifest_path=manifest,
        readiness_path=readiness,
        status_path=status,
        protocol_audit_path=protocol_audit,
        reference_targets_path=reference,
        reference_alignment_path=reference_alignment,
        generation_canary_summary_path=summary,
        runbook_path=runbook,
    )

    assert report["schema_version"] == bundle.SCHEMA_VERSION
    assert report["bundle_state"] == "awaiting_generation_canary_spend_authorization"
    assert report["manifest_id"] == "trialqa-full-test"
    assert report["next_command"]["requires_yes_spend"] is True
    assert report["next_command"]["authorized_by_audit"] is False
    assert report["artifacts"]["manifest"]["sha256"] == demo._sha256_file(manifest)
    assert report["artifacts"]["reference_alignment"]["sha256"] == demo._sha256_file(
        reference_alignment
    )
    assert report["artifacts"]["reference_source_document"] == {
        "path": str(reference_document),
        "sha256": demo._sha256_file(reference_document),
        "schema_version": None,
        "source_field": "reference_targets.source.document",
    }
    assert report["artifacts"]["generation_canary_summary"]["sha256"] == demo._sha256_file(
        summary
    )
    assert report["artifacts"]["runbook"]["sha256"] == demo._sha256_file(runbook)
    bundled_sources = {Path(str(item["path"])).name for item in report["source_files"]}
    assert {
        "trialqa_local_canary.py",
        "trialqa_local_gate.py",
        "trialqa_local_goal_audit.py",
        "trialqa_local_generation_checkpoint.py",
        "trialqa_local_progress.py",
        "trialqa_local_score_checkpoint.py",
        "trialqa_local_spend_review.py",
    } <= bundled_sources
    assert report["required_pre_spend_requirements"] == [
        {
            "id": "guarded_generation_dry_run_persisted",
            "status": "proved",
            "evidence": "fixture",
        }
    ]


def test_default_source_files_cover_all_local_trialqa_helpers() -> None:
    repo_root = Path(bundle.__file__).resolve().parents[1]
    expected = {
        path.resolve()
        for path in (repo_root / "benchmark").glob("trialqa_local_*.py")
    }
    expected.add((repo_root / "benchmark" / "trialqa_tooluniverse_mcp.py").resolve())
    actual = {path.resolve() for path in bundle.default_source_file_paths()}

    assert expected <= actual


def test_audit_bundle_can_override_source_file_hashes(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _protocol_audit(tmp_path / "protocol-audit.json")
    reference = _artifact(tmp_path / "reference.json", "switchyard.trialqa_reference_targets.v1")
    summary = _artifact(tmp_path / "summary.json", "switchyard.trialqa_canary_driver.v1")
    source = tmp_path / "guardrail.py"
    source.write_text("print('guardrail')\n", encoding="utf-8")

    report = bundle.build_audit_bundle(
        manifest_path=manifest,
        readiness_path=readiness,
        status_path=status,
        protocol_audit_path=protocol_audit,
        reference_targets_path=reference,
        generation_canary_summary_path=summary,
        source_file_paths=[source],
    )

    assert report["source_files"] == [
        {
            "path": str(source),
            "sha256": demo._sha256_file(source),
        }
    ]


def test_audit_bundle_rejects_stale_reference_source_document_hash(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _protocol_audit(tmp_path / "protocol-audit.json")
    reference_document = tmp_path / "reference.pdf"
    reference_document.write_bytes(b"reference slide")
    reference = _reference_targets(tmp_path / "reference.json", reference_document)
    reference_document.write_bytes(b"mutated reference slide")
    summary = _artifact(tmp_path / "summary.json", "switchyard.trialqa_canary_driver.v1")

    with pytest.raises(bundle.TrialQAAuditBundleError, match="reference source document hash"):
        bundle.build_audit_bundle(
            manifest_path=manifest,
            readiness_path=readiness,
            status_path=status,
            protocol_audit_path=protocol_audit,
            reference_targets_path=reference,
            generation_canary_summary_path=summary,
        )


def test_audit_bundle_hashes_score_dry_run_at_score_boundary(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _score_protocol_audit(tmp_path / "protocol-audit.json")
    reference = _artifact(tmp_path / "reference.json", "switchyard.trialqa_reference_targets.v1")
    score_summary = _artifact(
        tmp_path / "score-summary.json",
        "switchyard.trialqa_canary_score_driver.v1",
    )

    report = bundle.build_audit_bundle(
        manifest_path=manifest,
        readiness_path=readiness,
        status_path=status,
        protocol_audit_path=protocol_audit,
        reference_targets_path=reference,
        score_canary_summary_path=score_summary,
    )

    assert report["bundle_state"] == "awaiting_score_canary_spend_authorization"
    assert report["next_command"]["kind"] == "guarded_score_canary"
    assert report["artifacts"]["score_canary_summary"]["sha256"] == demo._sha256_file(
        score_summary
    )
    assert "generation_canary_summary" not in report["artifacts"]
    assert report["required_pre_spend_requirements"] == [
        {
            "id": "guarded_score_dry_run_persisted",
            "status": "proved",
            "evidence": "fixture",
        }
    ]


def test_audit_bundle_requires_matching_dry_run_for_score_boundary(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _score_protocol_audit(tmp_path / "protocol-audit.json")
    reference = _artifact(tmp_path / "reference.json", "switchyard.trialqa_reference_targets.v1")

    with pytest.raises(bundle.TrialQAAuditBundleError, match="score canary summary"):
        bundle.build_audit_bundle(
            manifest_path=manifest,
            readiness_path=readiness,
            status_path=status,
            protocol_audit_path=protocol_audit,
            reference_targets_path=reference,
        )


def test_audit_bundle_rejects_unproved_required_requirement(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    readiness = _artifact(tmp_path / "readiness.json", "switchyard.trialqa_canary_readiness.v1")
    status = _artifact(tmp_path / "status.json", "switchyard.trialqa_protocol_status.v1")
    protocol_audit = _protocol_audit(tmp_path / "protocol-audit.json", required_status="missing")
    reference = _artifact(tmp_path / "reference.json", "switchyard.trialqa_reference_targets.v1")

    with pytest.raises(bundle.TrialQAAuditBundleError, match="required audit item"):
        bundle.build_audit_bundle(
            manifest_path=manifest,
            readiness_path=readiness,
            status_path=status,
            protocol_audit_path=protocol_audit,
            reference_targets_path=reference,
        )
