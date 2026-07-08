# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command no-spend preflight for the local TrialQA validation ladder.

This command regenerates the first-generation readiness report, guarded dry-run
summary, staged status, protocol audit, hash-bound pre-spend bundle, and bundle
verification in sequence. It never runs live generation or judge scoring; the
only emitted next command remains explicitly gated behind ``--yes-spend``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_audit_bundle as audit_bundle  # noqa: E402
import benchmark.trialqa_local_audit_bundle_verify as bundle_verify  # noqa: E402
import benchmark.trialqa_local_canary as canary  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_protocol_audit as protocol_audit  # noqa: E402
import benchmark.trialqa_local_reference_alignment as reference_alignment  # noqa: E402
import benchmark.trialqa_local_status as status  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_no_spend_preflight.v1"
JsonObject = dict[str, Any]


class TrialQAPreflightError(RuntimeError):
    """The no-spend preflight could not prove the next guarded boundary."""


@dataclass(frozen=True)
class PreflightConfig:
    manifest: Path
    dataset: Path
    experiment_root: Path
    doctor: Path
    population_report: Path
    candidate: Path
    switchyard: Path
    codex: Path
    tooluniverse: Path
    profile: Path
    question_start: int
    question_limit: int
    repeat_limit: int
    workers: int
    max_generation_attempts: int
    reference_targets: Path
    runbook: Path
    readiness_output: Path
    gate_output: Path
    generation_summary_output: Path
    status_output: Path
    protocol_audit_output: Path
    reference_alignment_output: Path
    audit_bundle_output: Path
    audit_bundle_verification_output: Path
    operational_gate: Path | None = None
    promotion_gate: Path | None = None
    ladder_rehearsal: Path | None = None
    skills_distillation_repo: Path | None = Path("skills-distillation")


def _canary_config(config: PreflightConfig) -> canary.CanaryConfig:
    return canary.CanaryConfig(
        manifest=config.manifest,
        dataset=config.dataset,
        experiment_root=config.experiment_root,
        doctor=config.doctor,
        population_report=config.population_report,
        candidate=config.candidate,
        switchyard=config.switchyard,
        codex=config.codex,
        tooluniverse=config.tooluniverse,
        profile=config.profile,
        question_start=config.question_start,
        question_limit=config.question_limit,
        repeat_limit=config.repeat_limit,
        workers=config.workers,
        max_generation_attempts=config.max_generation_attempts,
        readiness_output=config.readiness_output,
        gate_output=config.gate_output,
    )


def _require_awaiting_spend(summary: JsonObject) -> None:
    if summary.get("status") != "awaiting_spend_authorization":
        raise TrialQAPreflightError(
            f"generation canary dry-run is not ready: {summary.get('status')!r}"
        )
    if summary.get("spend_authorized") is not False:
        raise TrialQAPreflightError("generation canary dry-run must not authorize spend")
    command = summary.get("authorized_rerun_command")
    if not isinstance(command, list) or not command or command[-1] != "--yes-spend":
        raise TrialQAPreflightError("generation canary dry-run has no guarded --yes-spend command")


def _require_generation_boundary(audit: JsonObject) -> None:
    if audit.get("completion_state") != "awaiting_generation_canary_spend_authorization":
        raise TrialQAPreflightError(
            f"protocol audit did not reach generation spend boundary: "
            f"{audit.get('completion_state')!r}"
        )
    next_command = audit.get("next_command")
    if not isinstance(next_command, dict):
        raise TrialQAPreflightError("protocol audit did not emit a guarded next_command")
    if next_command.get("requires_yes_spend") is not True:
        raise TrialQAPreflightError("protocol audit next_command is not spend-gated")
    if next_command.get("authorized_by_audit") is not False:
        raise TrialQAPreflightError("protocol audit must not authorize spend")


def _scope_suffix(config: PreflightConfig) -> str:
    question_stop = config.question_start + config.question_limit - 1
    return f"q{config.question_start}-q{question_stop}-r{config.repeat_limit}"


def _artifact_stem(config: PreflightConfig) -> str:
    scope_suffix = _scope_suffix(config)
    name = config.status_output.name
    expected_suffix = f"-{scope_suffix}.json"
    if name.startswith("status-") and name.endswith(expected_suffix):
        return name.removeprefix("status-")[: -len(expected_suffix)]
    audit_name = config.audit_bundle_output.name
    if audit_name.startswith("pre-spend-audit-bundle-") and audit_name.endswith(expected_suffix):
        return audit_name.removeprefix("pre-spend-audit-bundle-")[: -len(expected_suffix)]
    return config.status_output.stem.removeprefix("status-") or "trialqa-local"


def _skills_distillation_repo_args(config: PreflightConfig) -> list[str]:
    if config.skills_distillation_repo is None:
        return []
    return ["--skills-distillation-repo", str(config.skills_distillation_repo)]


def _ladder_rehearsal_path(config: PreflightConfig) -> Path:
    if config.ladder_rehearsal is not None:
        return config.ladder_rehearsal
    return config.audit_bundle_output.parent / f"ladder-rehearsal-{_artifact_stem(config)}.json"


def _post_generation_checkpoint_command(
    config: PreflightConfig,
    *,
    python: Path | str,
) -> JsonObject:
    artifact_dir = config.audit_bundle_output.parent
    stem = _artifact_stem(config)
    tail = f"{stem}-{_scope_suffix(config)}.json"
    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_generation_checkpoint",
        "--manifest",
        str(config.manifest),
        "--dataset",
        str(config.dataset),
        "--experiment-root",
        str(config.experiment_root),
        "--doctor",
        str(config.doctor),
        "--population-report",
        str(config.population_report),
        "--candidate",
        str(config.candidate),
        "--switchyard",
        str(config.switchyard),
        "--codex",
        str(config.codex),
        "--tooluniverse",
        str(config.tooluniverse),
        "--profile",
        str(config.profile),
        "--readiness",
        str(config.readiness_output),
        "--operational-gate",
        str(config.gate_output),
        "--question-start",
        str(config.question_start),
        "--question-limit",
        str(config.question_limit),
        "--repeat-limit",
        str(config.repeat_limit),
        "--workers",
        str(config.workers),
        "--max-generation-attempts",
        str(config.max_generation_attempts),
        "--reference-targets",
        str(config.reference_targets),
        "--runbook",
        str(config.runbook),
        *_skills_distillation_repo_args(config),
        "--artifact-dir",
        str(artifact_dir),
        "--artifact-stem",
        stem,
        "--status-output",
        str(artifact_dir / f"status-after-generation-{tail}"),
        "--next-step-output",
        str(artifact_dir / f"next-step-after-generation-{tail}"),
        "--score-summary-output",
        str(artifact_dir / f"canary-score-dryrun-{tail}"),
        "--promotion-gate-output",
        str(artifact_dir / f"gate-promotion-{tail}"),
        "--protocol-audit-output",
        str(artifact_dir / f"protocol-audit-score-{tail}"),
        "--reference-alignment-output",
        str(artifact_dir / f"reference-alignment-score-{tail}"),
        "--audit-bundle-output",
        str(artifact_dir / f"pre-score-spend-audit-bundle-{tail}"),
        "--audit-bundle-verification-output",
        str(artifact_dir / f"pre-score-spend-audit-bundle-verification-{tail}"),
        "--score-preflight-output",
        str(artifact_dir / f"no-spend-score-preflight-{tail}"),
        "--score-progress-output",
        str(artifact_dir / f"progress-score-{tail}"),
        "--spend-review-output",
        str(artifact_dir / f"spend-review-score-{tail}"),
        "--ladder-rehearsal",
        str(_ladder_rehearsal_path(config)),
        "--goal-audit-output",
        str(artifact_dir / f"goal-audit-score-{tail}"),
        "--decision-summary-output",
        str(artifact_dir / f"decision-summary-score-{tail}"),
        "--output",
        str(artifact_dir / f"generation-checkpoint-{tail}"),
    ]
    return {
        "kind": "post_generation_checkpoint",
        "command": command,
        "shell_command": " ".join(command),
        "contains_yes_spend": False,
        "requires_spend": False,
        "review_note": (
            "Run after the guarded generation canary finishes; this refreshes "
            "status and prepares either a terminal kill or the next score "
            "preflight/spend-review packet without judge spend."
        ),
    }


def run_preflight(config: PreflightConfig, *, python: Path | str = sys.executable) -> JsonObject:
    """Regenerate and verify the no-spend preflight ladder in dependency order."""

    canary_summary = canary.run_canary(
        _canary_config(config),
        yes_spend=False,
        python=python,
        summary_output=config.generation_summary_output,
    )
    _require_awaiting_spend(canary_summary)

    status_report = status.build_status_report(
        manifest_path=config.manifest,
        readiness_path=config.readiness_output,
        reference_targets_path=config.reference_targets,
        operational_gate_path=config.operational_gate,
        promotion_gate_path=config.promotion_gate,
    )
    demo._write_json_atomic(config.status_output, status_report)

    audit_report = protocol_audit.build_protocol_audit(
        manifest_path=config.manifest,
        status_path=config.status_output,
        generation_canary_summary_path=config.generation_summary_output,
        operational_gate_path=config.operational_gate,
    )
    _require_generation_boundary(audit_report)
    demo._write_json_atomic(config.protocol_audit_output, audit_report)

    reference_alignment_report = reference_alignment.build_reference_alignment(
        reference_alignment.ReferenceAlignmentConfig(
            manifest=config.manifest,
            reference_targets=config.reference_targets,
            skills_distillation_repo=config.skills_distillation_repo,
        )
    )
    if reference_alignment_report.get("canary_alignment_status") != "proved":
        raise TrialQAPreflightError(
            "reference alignment did not prove the canary comparison shape"
        )
    demo._write_json_atomic(config.reference_alignment_output, reference_alignment_report)

    bundle_report = audit_bundle.build_audit_bundle(
        manifest_path=config.manifest,
        readiness_path=config.readiness_output,
        status_path=config.status_output,
        protocol_audit_path=config.protocol_audit_output,
        reference_targets_path=config.reference_targets,
        reference_alignment_path=config.reference_alignment_output,
        generation_canary_summary_path=config.generation_summary_output,
        runbook_path=config.runbook,
    )
    demo._write_json_atomic(config.audit_bundle_output, bundle_report)

    verification_report = bundle_verify.verify_audit_bundle(
        bundle_path=config.audit_bundle_output,
    )
    if verification_report.get("status") != "passed":
        raise TrialQAPreflightError(
            f"pre-spend audit bundle verification failed: {verification_report.get('status')!r}"
        )
    demo._write_json_atomic(config.audit_bundle_verification_output, verification_report)

    bundle_section = cast(dict[str, Any], verification_report["bundle"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "spend_authorized": False,
        "manifest_id": bundle_section.get("manifest_id"),
        "bundle_state": bundle_section.get("bundle_state"),
        "next_command": verification_report.get("next_command"),
        "post_spend_checkpoint_command": _post_generation_checkpoint_command(
            config,
            python=python,
        ),
        "artifacts": {
            "readiness": str(config.readiness_output),
            "generation_canary_summary": str(config.generation_summary_output),
            "status": str(config.status_output),
            "protocol_audit": str(config.protocol_audit_output),
            "reference_alignment": str(config.reference_alignment_output),
            "audit_bundle": str(config.audit_bundle_output),
            "audit_bundle_verification": str(config.audit_bundle_verification_output),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--population-report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--switchyard", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--tooluniverse", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--question-start", type=int, required=True)
    parser.add_argument("--question-limit", type=int, required=True)
    parser.add_argument("--repeat-limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument(
        "--skills-distillation-repo",
        type=Path,
        default=Path("skills-distillation"),
    )
    parser.add_argument("--readiness-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--generation-summary-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--protocol-audit-output", type=Path, required=True)
    parser.add_argument("--reference-alignment-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-verification-output", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument("--promotion-gate", type=Path)
    parser.add_argument("--ladder-rehearsal", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> PreflightConfig:
    return PreflightConfig(
        manifest=args.manifest,
        dataset=args.dataset,
        experiment_root=args.experiment_root,
        doctor=args.doctor,
        population_report=args.population_report,
        candidate=args.candidate,
        switchyard=args.switchyard,
        codex=args.codex,
        tooluniverse=args.tooluniverse,
        profile=args.profile,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        reference_targets=args.reference_targets,
        runbook=args.runbook,
        skills_distillation_repo=args.skills_distillation_repo,
        readiness_output=args.readiness_output,
        gate_output=args.gate_output,
        generation_summary_output=args.generation_summary_output,
        status_output=args.status_output,
        protocol_audit_output=args.protocol_audit_output,
        reference_alignment_output=args.reference_alignment_output,
        audit_bundle_output=args.audit_bundle_output,
        audit_bundle_verification_output=args.audit_bundle_verification_output,
        operational_gate=args.operational_gate,
        promotion_gate=args.promotion_gate,
        ladder_rehearsal=args.ladder_rehearsal,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
