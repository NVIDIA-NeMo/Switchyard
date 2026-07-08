# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_audit_bundle as bundle
import benchmark.trialqa_local_audit_bundle_verify as verify
import benchmark.trialqa_local_demo as demo


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, schema: str) -> Path:
    return _write_json(path, {"schema_version": schema})


def _bundle(
    path: Path,
    *,
    artifact: Path,
    bundle_state: str = "awaiting_generation_canary_spend_authorization",
    command_kind: str = "guarded_generation_canary",
    source_file: Path | None = None,
) -> Path:
    source_files = []
    if source_file is not None:
        source_files.append({"path": str(source_file), "sha256": demo._sha256_file(source_file)})
    return _write_json(
        path,
        {
            "schema_version": bundle.SCHEMA_VERSION,
            "bundle_state": bundle_state,
            "manifest_id": "trialqa-full-test",
            "artifacts": {
                "manifest": {
                    "path": str(artifact),
                    "schema_version": "switchyard.trialqa_experiment_manifest.v1",
                    "sha256": demo._sha256_file(artifact),
                }
            },
            "source_files": source_files,
            "next_command": {
                "kind": command_kind,
                "command": ["python", "-m", "benchmark.trialqa_local_canary", "--yes-spend"],
                "shell_command": "python -m benchmark.trialqa_local_canary --yes-spend",
                "requires_yes_spend": True,
                "authorized_by_audit": False,
                "source": "summary.json",
            },
        },
    )


def test_verify_audit_bundle_passes_when_artifacts_match(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    bundle_path = _bundle(tmp_path / "bundle.json", artifact=artifact)

    report = verify.verify_audit_bundle(bundle_path=bundle_path)

    assert report["schema_version"] == verify.SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["bundle"]["sha256"] == demo._sha256_file(bundle_path)
    assert report["summary"] == {
        "bundle_sha256": demo._sha256_file(bundle_path),
        "bundle_state": "awaiting_generation_canary_spend_authorization",
        "manifest_id": "trialqa-full-test",
        "artifact_count": 1,
        "source_file_count": 0,
        "artifact_status_counts": {"matched": 1},
        "source_file_status_counts": {},
        "all_artifacts_matched": True,
        "all_source_files_matched": True,
        "next_command_kind": "guarded_generation_canary",
        "next_command_requires_yes_spend": True,
        "next_command_authorized_by_audit": False,
    }
    assert report["artifact_checks"] == [
        {
            "name": "manifest",
            "path": str(artifact),
            "sha256": demo._sha256_file(artifact),
            "schema_version": "switchyard.trialqa_experiment_manifest.v1",
            "status": "matched",
        }
    ]
    assert report["next_command"]["authorized_by_audit"] is False
    assert report["source_file_checks"] == []


def test_verify_audit_bundle_accepts_score_spend_boundary(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    bundle_path = _bundle(
        tmp_path / "bundle.json",
        artifact=artifact,
        bundle_state="awaiting_score_canary_spend_authorization",
        command_kind="guarded_score_canary",
    )

    report = verify.verify_audit_bundle(bundle_path=bundle_path)

    assert report["status"] == "passed"
    assert report["bundle"]["bundle_state"] == "awaiting_score_canary_spend_authorization"
    assert report["next_command"]["kind"] == "guarded_score_canary"


def test_verify_audit_bundle_rejects_boundary_command_mismatch(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    bundle_path = _bundle(
        tmp_path / "bundle.json",
        artifact=artifact,
        bundle_state="awaiting_score_canary_spend_authorization",
        command_kind="guarded_generation_canary",
    )

    with pytest.raises(verify.TrialQAAuditBundleVerificationError, match="kind"):
        verify.verify_audit_bundle(bundle_path=bundle_path)


def test_verify_audit_bundle_rejects_stale_source_file_hash(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    source = tmp_path / "guardrail.py"
    source.write_text("before\n", encoding="utf-8")
    bundle_path = _bundle(tmp_path / "bundle.json", artifact=artifact, source_file=source)
    source.write_text("after\n", encoding="utf-8")

    with pytest.raises(verify.TrialQAAuditBundleVerificationError, match="source file hash"):
        verify.verify_audit_bundle(bundle_path=bundle_path)


def test_verify_audit_bundle_rejects_stale_artifact_hash(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "manifest.json", "switchyard.trialqa_experiment_manifest.v1")
    bundle_path = _bundle(tmp_path / "bundle.json", artifact=artifact)
    _write_json(
        artifact,
        {
            "schema_version": "switchyard.trialqa_experiment_manifest.v1",
            "changed": True,
        },
    )

    with pytest.raises(verify.TrialQAAuditBundleVerificationError, match="hash mismatch"):
        verify.verify_audit_bundle(bundle_path=bundle_path)
