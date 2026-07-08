# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command no-spend score preflight for the local TrialQA ladder.

This command is the post-operational-gate companion to
``trialqa_local_preflight``. It validates that the operational gate promoted to
score, persists the guarded score dry-run summary, regenerates the staged
status and protocol audit, builds a hash-bound score-spend bundle, and verifies
that bundle. It never runs judge calls unless the separate guarded score driver
is explicitly rerun with ``--yes-spend``.
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
import benchmark.trialqa_local_canary_score as canary_score  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_protocol_audit as protocol_audit  # noqa: E402
import benchmark.trialqa_local_reference_alignment as reference_alignment  # noqa: E402
import benchmark.trialqa_local_status as status  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_no_spend_score_preflight.v1"
JsonObject = dict[str, Any]


class TrialQAScorePreflightError(RuntimeError):
    """The no-spend score preflight could not prove the judge-spend boundary."""


@dataclass(frozen=True)
class ScorePreflightConfig:
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
    operational_gate: Path
    question_start: int
    question_limit: int
    repeat_limit: int
    workers: int
    max_generation_attempts: int
    reference_targets: Path
    runbook: Path
    readiness_output: Path
    score_summary_output: Path
    promotion_gate_output: Path
    status_output: Path
    protocol_audit_output: Path
    reference_alignment_output: Path
    audit_bundle_output: Path
    audit_bundle_verification_output: Path
    ladder_rehearsal: Path | None = None
    skills_distillation_repo: Path | None = Path("skills-distillation")


def _score_config(config: ScorePreflightConfig) -> canary_score.ScoreConfig:
    return canary_score.ScoreConfig(
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
        operational_gate=config.operational_gate,
        question_start=config.question_start,
        question_limit=config.question_limit,
        repeat_limit=config.repeat_limit,
        workers=config.workers,
        max_generation_attempts=config.max_generation_attempts,
        promotion_gate_output=config.promotion_gate_output,
    )


def _require_awaiting_score_spend(summary: JsonObject) -> None:
    if summary.get("status") != "awaiting_spend_authorization":
        raise TrialQAScorePreflightError(
            f"score canary dry-run is not ready: {summary.get('status')!r}"
        )
    if summary.get("spend_authorized") is not False:
        raise TrialQAScorePreflightError("score canary dry-run must not authorize spend")
    if summary.get("operational_decision") != "promote_to_score":
        raise TrialQAScorePreflightError("score dry-run must be backed by a promoted gate")
    command = summary.get("authorized_rerun_command")
    if not isinstance(command, list) or not command or command[-1] != "--yes-spend":
        raise TrialQAScorePreflightError("score canary dry-run has no guarded --yes-spend command")


def _require_score_boundary(audit: JsonObject) -> None:
    if audit.get("completion_state") != "awaiting_score_canary_spend_authorization":
        raise TrialQAScorePreflightError(
            f"protocol audit did not reach score spend boundary: "
            f"{audit.get('completion_state')!r}"
        )
    next_command = audit.get("next_command")
    if not isinstance(next_command, dict):
        raise TrialQAScorePreflightError("protocol audit did not emit a guarded next_command")
    if next_command.get("kind") != "guarded_score_canary":
        raise TrialQAScorePreflightError("protocol audit next_command is not a score canary")
    if next_command.get("requires_yes_spend") is not True:
        raise TrialQAScorePreflightError("protocol audit next_command is not spend-gated")
    if next_command.get("authorized_by_audit") is not False:
        raise TrialQAScorePreflightError("protocol audit must not authorize spend")


def _scope_suffix(*, question_start: int, question_limit: int, repeat_limit: int) -> str:
    question_stop = question_start + question_limit - 1
    return f"q{question_start}-q{question_stop}-r{repeat_limit}"


def _artifact_stem(config: ScorePreflightConfig) -> str:
    scope_suffix = _scope_suffix(
        question_start=config.question_start,
        question_limit=config.question_limit,
        repeat_limit=config.repeat_limit,
    )
    name = config.status_output.name
    expected_suffix = f"-{scope_suffix}.json"
    if name.startswith("status-") and name.endswith(expected_suffix):
        return name.removeprefix("status-")[: -len(expected_suffix)]
    audit_name = config.audit_bundle_output.name
    if audit_name.startswith("pre-score-spend-audit-bundle-") and audit_name.endswith(
        expected_suffix
    ):
        return audit_name.removeprefix("pre-score-spend-audit-bundle-")[: -len(expected_suffix)]
    return config.status_output.stem.removeprefix("status-") or "trialqa-local"


def _primary_scope(config: ScorePreflightConfig) -> tuple[int, int, int]:
    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise TrialQAScorePreflightError("manifest has no protocol metadata")
    primary = protocol.get("primary_evaluation_scope")
    if not isinstance(primary, dict):
        raise TrialQAScorePreflightError("manifest has no primary_evaluation_scope")
    question_start = primary.get("question_start")
    question_count = primary.get("question_count")
    repeat_count = primary.get("repeat_count")
    if (
        isinstance(question_start, bool)
        or not isinstance(question_start, int)
        or isinstance(question_count, bool)
        or not isinstance(question_count, int)
        or isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
    ):
        raise TrialQAScorePreflightError("manifest primary scope is invalid")
    return question_start, question_count, repeat_count


def _next_generation_scope_after_score(config: ScorePreflightConfig) -> tuple[int, int, int]:
    primary_start, primary_questions, primary_repeats = _primary_scope(config)
    if config.question_limit < primary_questions:
        return primary_start, primary_questions, config.repeat_limit
    if config.repeat_limit < 3:
        return primary_start, primary_questions, 3
    if config.repeat_limit < primary_repeats:
        return primary_start, primary_questions, primary_repeats
    return config.question_start, config.question_limit, config.repeat_limit


def _skills_distillation_repo_args(config: ScorePreflightConfig) -> list[str]:
    if config.skills_distillation_repo is None:
        return []
    return ["--skills-distillation-repo", str(config.skills_distillation_repo)]


def _ladder_rehearsal_path(config: ScorePreflightConfig) -> Path:
    if config.ladder_rehearsal is not None:
        return config.ladder_rehearsal
    return config.audit_bundle_output.parent / f"ladder-rehearsal-{_artifact_stem(config)}.json"


def _post_score_checkpoint_command(
    config: ScorePreflightConfig,
    *,
    python: Path | str,
) -> JsonObject:
    artifact_dir = config.audit_bundle_output.parent
    stem = _artifact_stem(config)
    current_tail = (
        f"{stem}-"
        f"{_scope_suffix(question_start=config.question_start, question_limit=config.question_limit, repeat_limit=config.repeat_limit)}.json"
    )
    next_question_start, next_question_limit, next_repeat_limit = _next_generation_scope_after_score(
        config
    )
    next_tail = (
        f"{stem}-"
        f"{_scope_suffix(question_start=next_question_start, question_limit=next_question_limit, repeat_limit=next_repeat_limit)}.json"
    )
    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_score_checkpoint",
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
        "--current-readiness",
        str(config.readiness_output),
        "--operational-gate",
        str(config.operational_gate),
        "--promotion-gate",
        str(config.promotion_gate_output),
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
        "--post-score-status-output",
        str(artifact_dir / f"status-after-score-{current_tail}"),
        "--next-step-output",
        str(artifact_dir / f"next-step-after-score-{current_tail}"),
        "--expansion-readiness-output",
        str(artifact_dir / f"readiness-{next_tail}"),
        "--expansion-operational-gate-output",
        str(artifact_dir / f"gate-operational-{next_tail}"),
        "--generation-summary-output",
        str(artifact_dir / f"canary-generation-dryrun-{next_tail}"),
        "--expansion-status-output",
        str(artifact_dir / f"status-{next_tail}"),
        "--protocol-audit-output",
        str(artifact_dir / f"protocol-audit-{next_tail}"),
        "--reference-alignment-output",
        str(artifact_dir / f"reference-alignment-{next_tail}"),
        "--audit-bundle-output",
        str(artifact_dir / f"pre-spend-audit-bundle-{next_tail}"),
        "--audit-bundle-verification-output",
        str(artifact_dir / f"pre-spend-audit-bundle-verification-{next_tail}"),
        "--generation-preflight-output",
        str(artifact_dir / f"no-spend-preflight-{next_tail}"),
        "--spend-review-output",
        str(artifact_dir / f"spend-review-{next_tail}"),
        "--ladder-rehearsal",
        str(_ladder_rehearsal_path(config)),
        "--goal-audit-output",
        str(artifact_dir / f"goal-audit-{next_tail}"),
        "--decision-summary-output",
        str(artifact_dir / f"decision-summary-{next_tail}"),
        "--output",
        str(artifact_dir / f"score-checkpoint-{current_tail}"),
    ]
    return {
        "kind": "post_score_checkpoint",
        "command": command,
        "shell_command": " ".join(command),
        "contains_yes_spend": False,
        "requires_spend": False,
        "review_note": (
            "Run after the guarded score canary finishes; this refreshes status "
            "and prepares either a terminal kill/completion decision or the next "
            "generation-expansion preflight/spend-review packet without model spend."
        ),
    }


def run_score_preflight(
    config: ScorePreflightConfig,
    *,
    python: Path | str = sys.executable,
) -> JsonObject:
    """Regenerate and verify the no-spend score boundary in dependency order."""

    score_summary = canary_score.run_score_canary(
        _score_config(config),
        yes_spend=False,
        python=python,
        summary_output=config.score_summary_output,
    )
    _require_awaiting_score_spend(score_summary)

    status_report = status.build_status_report(
        manifest_path=config.manifest,
        readiness_path=config.readiness_output,
        reference_targets_path=config.reference_targets,
        operational_gate_path=config.operational_gate,
    )
    demo._write_json_atomic(config.status_output, status_report)

    audit_report = protocol_audit.build_protocol_audit(
        manifest_path=config.manifest,
        status_path=config.status_output,
        score_canary_summary_path=config.score_summary_output,
        operational_gate_path=config.operational_gate,
    )
    _require_score_boundary(audit_report)
    demo._write_json_atomic(config.protocol_audit_output, audit_report)

    reference_alignment_report = reference_alignment.build_reference_alignment(
        reference_alignment.ReferenceAlignmentConfig(
            manifest=config.manifest,
            reference_targets=config.reference_targets,
            skills_distillation_repo=config.skills_distillation_repo,
        )
    )
    if reference_alignment_report.get("canary_alignment_status") != "proved":
        raise TrialQAScorePreflightError(
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
        score_canary_summary_path=config.score_summary_output,
        runbook_path=config.runbook,
    )
    demo._write_json_atomic(config.audit_bundle_output, bundle_report)

    verification_report = bundle_verify.verify_audit_bundle(
        bundle_path=config.audit_bundle_output,
    )
    if verification_report.get("status") != "passed":
        raise TrialQAScorePreflightError(
            f"score audit bundle verification failed: {verification_report.get('status')!r}"
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
        "post_spend_checkpoint_command": _post_score_checkpoint_command(
            config,
            python=python,
        ),
        "artifacts": {
            "score_canary_summary": str(config.score_summary_output),
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
    parser.add_argument("--operational-gate", type=Path, required=True)
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
    parser.add_argument("--score-summary-output", type=Path, required=True)
    parser.add_argument("--promotion-gate-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--protocol-audit-output", type=Path, required=True)
    parser.add_argument("--reference-alignment-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-verification-output", type=Path, required=True)
    parser.add_argument("--ladder-rehearsal", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> ScorePreflightConfig:
    return ScorePreflightConfig(
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
        operational_gate=args.operational_gate,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        reference_targets=args.reference_targets,
        runbook=args.runbook,
        skills_distillation_repo=args.skills_distillation_repo,
        readiness_output=args.readiness_output,
        score_summary_output=args.score_summary_output,
        promotion_gate_output=args.promotion_gate_output,
        status_output=args.status_output,
        protocol_audit_output=args.protocol_audit_output,
        reference_alignment_output=args.reference_alignment_output,
        audit_bundle_output=args.audit_bundle_output,
        audit_bundle_verification_output=args.audit_bundle_verification_output,
        ladder_rehearsal=args.ladder_rehearsal,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_score_preflight(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
