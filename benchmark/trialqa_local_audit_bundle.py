# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-bound pre-spend audit bundle for the local TrialQA validation ladder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_pre_spend_audit_bundle.v1"
JsonObject = dict[str, Any]
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILES = tuple(
    _REPO_ROOT / path
    for path in (
        "benchmark/trialqa_local_audit_bundle.py",
        "benchmark/trialqa_local_audit_bundle_verify.py",
        "benchmark/trialqa_local_batch.py",
        "benchmark/trialqa_local_canary.py",
        "benchmark/trialqa_local_canary_score.py",
        "benchmark/trialqa_local_candidate_repair.py",
        "benchmark/trialqa_local_current_packet.py",
        "benchmark/trialqa_local_dataset.py",
        "benchmark/trialqa_local_decision_summary.py",
        "benchmark/trialqa_local_demo.py",
        "benchmark/trialqa_local_distiller.py",
        "benchmark/trialqa_local_gate.py",
        "benchmark/trialqa_local_gate_inspect.py",
        "benchmark/trialqa_local_generation_checkpoint.py",
        "benchmark/trialqa_local_goal_audit.py",
        "benchmark/trialqa_local_identifier_terminal_repair.py",
        "benchmark/trialqa_local_ladder_rehearsal.py",
        "benchmark/trialqa_local_next_step.py",
        "benchmark/trialqa_local_population_audit.py",
        "benchmark/trialqa_local_preflight.py",
        "benchmark/trialqa_local_progress.py",
        "benchmark/trialqa_local_prospective_population.py",
        "benchmark/trialqa_local_protocol_audit.py",
        "benchmark/trialqa_local_readiness.py",
        "benchmark/trialqa_local_reference_alignment.py",
        "benchmark/trialqa_local_regression.py",
        "benchmark/trialqa_local_runner.py",
        "benchmark/trialqa_local_search_gate.py",
        "benchmark/trialqa_local_search_repair.py",
        "benchmark/trialqa_local_score_checkpoint.py",
        "benchmark/trialqa_local_score_preflight.py",
        "benchmark/trialqa_local_spend_guard.py",
        "benchmark/trialqa_local_spend_review.py",
        "benchmark/trialqa_local_status.py",
        "benchmark/trialqa_tooluniverse_mcp.py",
    )
)
SPEND_BOUNDARY_COMMAND_KINDS = {
    "awaiting_generation_canary_spend_authorization": "guarded_generation_canary",
    "awaiting_score_canary_spend_authorization": "guarded_score_canary",
}


class TrialQAAuditBundleError(RuntimeError):
    """Audit bundle inputs are missing, stale, or not ready for the next gate."""


def _artifact(path: Path, *, label: str, expected_schema: str | None = None) -> JsonObject:
    payload = demo._read_json_object(path, label)
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        raise TrialQAAuditBundleError(f"{label} has invalid schema_version")
    return {
        "path": str(path),
        "sha256": demo._sha256_file(path),
        "schema_version": payload.get("schema_version"),
    }


def _reference_source_document(path: Path) -> JsonObject | None:
    payload = demo._read_json_object(path, "reference targets")
    source = payload.get("source")
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise TrialQAAuditBundleError("reference targets source must be an object")
    document = source.get("document")
    expected_sha256 = source.get("document_sha256")
    if not isinstance(document, str) or not isinstance(expected_sha256, str):
        raise TrialQAAuditBundleError(
            "reference targets source must include document and document_sha256"
        )
    document_path = Path(document)
    if not document_path.is_absolute():
        candidates = (path.parent / document_path, _REPO_ROOT / document_path)
        document_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    actual_sha256 = demo._sha256_file(document_path)
    if actual_sha256 != expected_sha256:
        raise TrialQAAuditBundleError(
            f"reference source document hash mismatch: {actual_sha256} != {expected_sha256}"
        )
    return {
        "path": str(document_path),
        "sha256": actual_sha256,
        "schema_version": None,
        "source_field": "reference_targets.source.document",
    }


def default_source_file_paths() -> tuple[Path, ...]:
    """Return source files that must not drift between preflight and spend."""

    return DEFAULT_SOURCE_FILES


def _source_file(path: Path) -> JsonObject:
    return {
        "path": str(path),
        "sha256": demo._sha256_file(path),
    }


def _required_requirements(protocol_audit: Mapping[str, object]) -> list[Mapping[str, object]]:
    requirements = protocol_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQAAuditBundleError("protocol audit has invalid requirements")
    required = []
    for item in requirements:
        if not isinstance(item, Mapping):
            raise TrialQAAuditBundleError("protocol audit requirement is invalid")
        if item.get("required_for_spend") is True:
            required.append(item)
    return required


def _validate_protocol_audit(protocol_audit: Mapping[str, object]) -> str:
    if protocol_audit.get("schema_version") != "switchyard.trialqa_protocol_audit.v1":
        raise TrialQAAuditBundleError("protocol audit has invalid schema_version")
    completion_state = protocol_audit.get("completion_state")
    if not isinstance(completion_state, str):
        raise TrialQAAuditBundleError("protocol audit has invalid completion_state")
    expected_kind = SPEND_BOUNDARY_COMMAND_KINDS.get(completion_state)
    if expected_kind is None:
        raise TrialQAAuditBundleError(
            f"protocol audit is not at a supported spend boundary: {completion_state!r}"
        )
    next_command = protocol_audit.get("next_command")
    if not isinstance(next_command, Mapping):
        raise TrialQAAuditBundleError("protocol audit has no guarded next_command")
    if next_command.get("kind") != expected_kind:
        raise TrialQAAuditBundleError(
            f"protocol audit next_command kind must be {expected_kind!r}"
        )
    if next_command.get("requires_yes_spend") is not True:
        raise TrialQAAuditBundleError("protocol audit next_command does not require spend")
    if next_command.get("authorized_by_audit") is not False:
        raise TrialQAAuditBundleError("protocol audit must not authorize spend")
    command = next_command.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise TrialQAAuditBundleError("protocol audit next_command.command is invalid")
    if not command or command[-1] != "--yes-spend":
        raise TrialQAAuditBundleError("protocol audit next_command must end in --yes-spend")
    for item in _required_requirements(protocol_audit):
        if item.get("status") != "proved":
            raise TrialQAAuditBundleError(
                f"required audit item {item.get('id')!r} is {item.get('status')!r}"
            )
    return completion_state


def build_audit_bundle(
    *,
    manifest_path: Path,
    readiness_path: Path,
    status_path: Path,
    protocol_audit_path: Path,
    reference_targets_path: Path,
    reference_alignment_path: Path | None = None,
    generation_canary_summary_path: Path | None = None,
    score_canary_summary_path: Path | None = None,
    runbook_path: Path | None = None,
    source_file_paths: Sequence[Path] | None = None,
) -> JsonObject:
    """Build a single hash-bound pre-spend evidence bundle."""

    protocol_audit = demo._read_json_object(protocol_audit_path, "protocol audit")
    completion_state = _validate_protocol_audit(protocol_audit)
    if (
        completion_state == "awaiting_generation_canary_spend_authorization"
        and generation_canary_summary_path is None
    ):
        raise TrialQAAuditBundleError("generation boundary requires a generation canary summary")
    if (
        completion_state == "awaiting_score_canary_spend_authorization"
        and score_canary_summary_path is None
    ):
        raise TrialQAAuditBundleError("score boundary requires a score canary summary")
    next_command = cast(Mapping[str, object], protocol_audit["next_command"])
    artifacts: JsonObject = {
        "manifest": _artifact(
            manifest_path,
            label="experiment manifest",
            expected_schema="switchyard.trialqa_experiment_manifest.v1",
        ),
        "readiness": _artifact(
            readiness_path,
            label="readiness report",
            expected_schema="switchyard.trialqa_canary_readiness.v1",
        ),
        "status": _artifact(
            status_path,
            label="protocol status",
            expected_schema="switchyard.trialqa_protocol_status.v1",
        ),
        "protocol_audit": _artifact(
            protocol_audit_path,
            label="protocol audit",
            expected_schema="switchyard.trialqa_protocol_audit.v1",
        ),
        "reference_targets": _artifact(
            reference_targets_path,
            label="reference targets",
            expected_schema="switchyard.trialqa_reference_targets.v1",
        ),
    }
    reference_source = _reference_source_document(reference_targets_path)
    if reference_source is not None:
        artifacts["reference_source_document"] = reference_source
    if reference_alignment_path is not None:
        artifacts["reference_alignment"] = _artifact(
            reference_alignment_path,
            label="reference alignment",
            expected_schema="switchyard.trialqa_reference_alignment.v1",
        )
    if generation_canary_summary_path is not None:
        artifacts["generation_canary_summary"] = _artifact(
            generation_canary_summary_path,
            label="generation canary summary",
            expected_schema="switchyard.trialqa_canary_driver.v1",
        )
    if score_canary_summary_path is not None:
        artifacts["score_canary_summary"] = _artifact(
            score_canary_summary_path,
            label="score canary summary",
            expected_schema="switchyard.trialqa_canary_score_driver.v1",
        )
    if runbook_path is not None:
        artifacts["runbook"] = {
            "path": str(runbook_path),
            "sha256": demo._sha256_file(runbook_path),
            "schema_version": None,
        }
    source_files = [
        _source_file(path)
        for path in (default_source_file_paths() if source_file_paths is None else source_file_paths)
    ]
    required = [
        {
            "id": item.get("id"),
            "status": item.get("status"),
            "evidence": item.get("evidence"),
        }
        for item in _required_requirements(protocol_audit)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": protocol_audit.get("manifest_id"),
        "bundle_state": protocol_audit.get("completion_state"),
        "artifacts": artifacts,
        "source_files": source_files,
        "required_pre_spend_requirements": required,
        "next_command": {
            "kind": next_command.get("kind"),
            "command": next_command.get("command"),
            "shell_command": next_command.get("shell_command"),
            "requires_yes_spend": next_command.get("requires_yes_spend"),
            "authorized_by_audit": next_command.get("authorized_by_audit"),
            "source": next_command.get("source"),
        },
        "scope_note": (
            "This bundle is a read-only pre-spend snapshot. It preserves evidence "
            "for the next guarded gate but does not authorize model or judge spend."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--reference-alignment", type=Path)
    parser.add_argument("--generation-canary-summary", type=Path)
    parser.add_argument("--score-canary-summary", type=Path)
    parser.add_argument("--runbook", type=Path)
    parser.add_argument(
        "--source-file",
        action="append",
        dest="source_files",
        type=Path,
        help=(
            "Source file to hash-bind. Repeat to override the default TrialQA "
            "guardrail source list; pass /dev/null only for isolated tests."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_audit_bundle(
        manifest_path=args.manifest,
        readiness_path=args.readiness,
        status_path=args.status,
        protocol_audit_path=args.protocol_audit,
        reference_targets_path=args.reference_targets,
        reference_alignment_path=args.reference_alignment,
        generation_canary_summary_path=args.generation_canary_summary,
        score_canary_summary_path=args.score_canary_summary,
        runbook_path=args.runbook,
        source_file_paths=args.source_files,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
