# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a saved local TrialQA pre-spend audit bundle against current files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_audit_bundle as bundle  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_pre_spend_audit_bundle_verification.v1"
JsonObject = dict[str, Any]


class TrialQAAuditBundleVerificationError(RuntimeError):
    """A saved pre-spend audit bundle no longer matches current files."""


def _require_bundle(value: Mapping[str, object]) -> None:
    if value.get("schema_version") != bundle.SCHEMA_VERSION:
        raise TrialQAAuditBundleVerificationError("bundle has invalid schema_version")
    bundle_state = value.get("bundle_state")
    if not isinstance(bundle_state, str):
        raise TrialQAAuditBundleVerificationError("bundle has invalid bundle_state")
    expected_kind = bundle.SPEND_BOUNDARY_COMMAND_KINDS.get(bundle_state)
    if expected_kind is None:
        raise TrialQAAuditBundleVerificationError(
            f"bundle is not at a supported spend boundary: {bundle_state!r}"
        )
    next_command = value.get("next_command")
    if not isinstance(next_command, Mapping):
        raise TrialQAAuditBundleVerificationError("bundle has no next_command")
    if next_command.get("kind") != expected_kind:
        raise TrialQAAuditBundleVerificationError(
            f"bundle next_command kind must be {expected_kind!r}"
        )
    if next_command.get("requires_yes_spend") is not True:
        raise TrialQAAuditBundleVerificationError("bundle next_command does not require spend")
    if next_command.get("authorized_by_audit") is not False:
        raise TrialQAAuditBundleVerificationError("bundle must not authorize spend")
    command = next_command.get("command")
    if not isinstance(command, list) or not command or command[-1] != "--yes-spend":
        raise TrialQAAuditBundleVerificationError("bundle next_command must end in --yes-spend")


def _artifact_checks(bundle_root: Mapping[str, object]) -> list[JsonObject]:
    artifacts = bundle_root.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TrialQAAuditBundleVerificationError("bundle has invalid artifacts")
    checks: list[JsonObject] = []
    for name, raw in sorted(artifacts.items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise TrialQAAuditBundleVerificationError("bundle artifact entry is invalid")
        path_value = raw.get("path")
        expected_hash = raw.get("sha256")
        expected_schema = raw.get("schema_version")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise TrialQAAuditBundleVerificationError(f"bundle artifact {name} is incomplete")
        path = Path(path_value)
        actual_hash = demo._sha256_file(path)
        if actual_hash != expected_hash:
            raise TrialQAAuditBundleVerificationError(
                f"bundle artifact {name} hash mismatch: {actual_hash} != {expected_hash}"
            )
        actual_schema = None
        if expected_schema is not None:
            payload = demo._read_json_object(path, f"bundle artifact {name}")
            actual_schema = payload.get("schema_version")
            if actual_schema != expected_schema:
                raise TrialQAAuditBundleVerificationError(
                    f"bundle artifact {name} schema mismatch: {actual_schema!r}"
                )
        checks.append(
            {
                "name": name,
                "path": path_value,
                "sha256": actual_hash,
                "schema_version": actual_schema if expected_schema is not None else None,
                "status": "matched",
            }
        )
    return checks


def _source_file_checks(bundle_root: Mapping[str, object]) -> list[JsonObject]:
    source_files = bundle_root.get("source_files", [])
    if not isinstance(source_files, list):
        raise TrialQAAuditBundleVerificationError("bundle source_files must be a list")
    checks: list[JsonObject] = []
    for raw in source_files:
        if not isinstance(raw, Mapping):
            raise TrialQAAuditBundleVerificationError("bundle source file entry is invalid")
        path_value = raw.get("path")
        expected_hash = raw.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise TrialQAAuditBundleVerificationError("bundle source file entry is incomplete")
        path = Path(path_value)
        actual_hash = demo._sha256_file(path)
        if actual_hash != expected_hash:
            raise TrialQAAuditBundleVerificationError(
                f"bundle source file hash mismatch: {actual_hash} != {expected_hash}"
            )
        checks.append(
            {
                "path": path_value,
                "sha256": actual_hash,
                "status": "matched",
            }
        )
    return checks


def _status_counts(checks: Sequence[Mapping[str, object]]) -> JsonObject:
    counts: JsonObject = {}
    for check in checks:
        status = check.get("status")
        key = status if isinstance(status, str) else "unknown"
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _verification_summary(
    *,
    bundle_root: Mapping[str, object],
    bundle_hash: str,
    artifact_checks: Sequence[Mapping[str, object]],
    source_file_checks: Sequence[Mapping[str, object]],
) -> JsonObject:
    next_command = bundle_root.get("next_command")
    command_summary: Mapping[str, object] = (
        next_command if isinstance(next_command, Mapping) else {}
    )
    artifact_counts = _status_counts(artifact_checks)
    source_counts = _status_counts(source_file_checks)
    return {
        "bundle_sha256": bundle_hash,
        "bundle_state": bundle_root.get("bundle_state"),
        "manifest_id": bundle_root.get("manifest_id"),
        "artifact_count": len(artifact_checks),
        "source_file_count": len(source_file_checks),
        "artifact_status_counts": artifact_counts,
        "source_file_status_counts": source_counts,
        "all_artifacts_matched": all(
            check.get("status") == "matched" for check in artifact_checks
        ),
        "all_source_files_matched": all(
            check.get("status") == "matched" for check in source_file_checks
        ),
        "next_command_kind": command_summary.get("kind"),
        "next_command_requires_yes_spend": command_summary.get("requires_yes_spend"),
        "next_command_authorized_by_audit": command_summary.get("authorized_by_audit"),
    }


def verify_audit_bundle(*, bundle_path: Path) -> JsonObject:
    """Verify a saved pre-spend audit bundle against the current filesystem."""

    bundle_root = demo._read_json_object(bundle_path, "pre-spend audit bundle")
    _require_bundle(bundle_root)
    artifact_checks = _artifact_checks(bundle_root)
    source_file_checks = _source_file_checks(bundle_root)
    bundle_hash = demo._sha256_file(bundle_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle": {
            "path": str(bundle_path),
            "sha256": bundle_hash,
            "schema_version": bundle_root.get("schema_version"),
            "bundle_state": bundle_root.get("bundle_state"),
            "manifest_id": bundle_root.get("manifest_id"),
        },
        "summary": _verification_summary(
            bundle_root=bundle_root,
            bundle_hash=bundle_hash,
            artifact_checks=artifact_checks,
            source_file_checks=source_file_checks,
        ),
        "artifact_checks": artifact_checks,
        "source_file_checks": source_file_checks,
        "next_command": bundle_root.get("next_command"),
        "status": "passed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_audit_bundle(bundle_path=args.bundle)
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
