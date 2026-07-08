# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-spend checkpoint after a guarded TrialQA score canary.

Run this after the live score canary has produced a promotion gate. The
checkpoint refreshes post-score status, asks the next-step planner whether the
candidate should be killed, declared complete, or expanded, and prepares the
next generation-expansion preflight plus spend-review packet when appropriate.
It never authorizes model or judge spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_decision_summary as decision_summary  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_goal_audit as goal_audit  # noqa: E402
import benchmark.trialqa_local_next_step as next_step  # noqa: E402
import benchmark.trialqa_local_preflight as generation_preflight  # noqa: E402
import benchmark.trialqa_local_spend_review as spend_review  # noqa: E402
import benchmark.trialqa_local_status as status  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_score_checkpoint.v1"
JsonObject = dict[str, Any]


class TrialQAScoreCheckpointError(RuntimeError):
    """The post-score checkpoint cannot safely choose the next boundary."""


@dataclass(frozen=True)
class ScoreCheckpointConfig:
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
    current_readiness: Path
    operational_gate: Path
    promotion_gate: Path
    workers: int
    max_generation_attempts: int
    reference_targets: Path
    runbook: Path
    artifact_dir: Path
    artifact_stem: str
    post_score_status_output: Path
    next_step_output: Path
    expansion_readiness_output: Path
    expansion_operational_gate_output: Path
    generation_summary_output: Path
    expansion_status_output: Path
    protocol_audit_output: Path
    reference_alignment_output: Path
    audit_bundle_output: Path
    audit_bundle_verification_output: Path
    generation_preflight_output: Path
    spend_review_output: Path
    ladder_rehearsal: Path | None = None
    goal_audit_output: Path | None = None
    decision_summary_output: Path | None = None
    skills_distillation_repo: Path | None = Path("skills-distillation")


def _next_step_config(config: ScoreCheckpointConfig) -> next_step.NextStepConfig:
    return next_step.NextStepConfig(
        status=config.post_score_status_output,
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
        reference_targets=config.reference_targets,
        runbook=config.runbook,
        artifact_dir=config.artifact_dir,
        artifact_stem=config.artifact_stem,
        workers=config.workers,
        max_generation_attempts=config.max_generation_attempts,
        operational_gate=config.operational_gate,
        promotion_gate=config.promotion_gate,
        ladder_rehearsal=config.ladder_rehearsal,
        skills_distillation_repo=config.skills_distillation_repo,
    )


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrialQAScoreCheckpointError(f"next-step {label} must be an integer")
    return value


def _scope_from_next_step(next_step_report: Mapping[str, object]) -> tuple[int, int, int]:
    scope = next_step_report.get("scope")
    if not isinstance(scope, Mapping):
        raise TrialQAScoreCheckpointError("next-step report has no expansion scope")
    return (
        _require_int(scope.get("question_start"), "question_start"),
        _require_int(scope.get("question_limit"), "question_limit"),
        _require_int(scope.get("repeat_limit"), "repeat_limit"),
    )


def _preflight_config(
    config: ScoreCheckpointConfig,
    *,
    question_start: int,
    question_limit: int,
    repeat_limit: int,
) -> generation_preflight.PreflightConfig:
    return generation_preflight.PreflightConfig(
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
        question_start=question_start,
        question_limit=question_limit,
        repeat_limit=repeat_limit,
        workers=config.workers,
        max_generation_attempts=config.max_generation_attempts,
        reference_targets=config.reference_targets,
        runbook=config.runbook,
        skills_distillation_repo=config.skills_distillation_repo,
        readiness_output=config.expansion_readiness_output,
        gate_output=config.expansion_operational_gate_output,
        generation_summary_output=config.generation_summary_output,
        status_output=config.expansion_status_output,
        protocol_audit_output=config.protocol_audit_output,
        reference_alignment_output=config.reference_alignment_output,
        audit_bundle_output=config.audit_bundle_output,
        audit_bundle_verification_output=config.audit_bundle_verification_output,
        operational_gate=config.operational_gate,
        promotion_gate=config.promotion_gate,
        ladder_rehearsal=config.ladder_rehearsal,
    )


def _spend_guard_check_output(spend_review_path: Path) -> Path:
    name = spend_review_path.name
    tail = name.removeprefix("spend-review-") if name.startswith("spend-review-") else name
    return spend_review_path.with_name(f"spend-guard-check-{tail}")


def _spend_guard_check_command(
    *,
    spend_review_path: Path,
    python: Path | str,
) -> JsonObject:
    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_spend_guard",
        "--spend-review",
        str(spend_review_path),
        "--output",
        str(_spend_guard_check_output(spend_review_path)),
    ]
    return {
        "command": command,
        "shell_command": " ".join(command),
        "contains_yes_spend": False,
        "review_note": (
            "Run immediately before approving generation-expansion spend; this "
            "rechecks the reviewed guarded command, current hash-bound bundle, "
            "and selected ledger/lock progress without making model or judge calls."
        ),
    }


def _optional_summary_outputs(
    *,
    config: ScoreCheckpointConfig,
) -> JsonObject | None:
    values = (
        config.ladder_rehearsal,
        config.goal_audit_output,
        config.decision_summary_output,
    )
    if all(value is None for value in values):
        return None
    if not all(value is not None for value in values):
        raise TrialQAScoreCheckpointError(
            "ladder_rehearsal, goal_audit_output, and decision_summary_output "
            "must be provided together"
        )
    assert config.ladder_rehearsal is not None
    assert config.goal_audit_output is not None
    assert config.decision_summary_output is not None
    goal_report = goal_audit.build_goal_audit(
        goal_audit.GoalAuditConfig(
            manifest=config.manifest,
            reference_targets=config.reference_targets,
            reference_alignment=config.reference_alignment_output,
            ladder_rehearsal=config.ladder_rehearsal,
            preflight=config.generation_preflight_output,
            protocol_audit=config.protocol_audit_output,
            spend_review=config.spend_review_output,
            operational_gate=config.operational_gate,
            promotion_gate=config.promotion_gate,
        )
    )
    demo._write_json_atomic(config.goal_audit_output, goal_report)
    summary_report = decision_summary.build_decision_summary(
        decision_summary.DecisionSummaryConfig(
            spend_review=config.spend_review_output,
            goal_audit=config.goal_audit_output,
            decision_summary_output=config.decision_summary_output,
        )
    )
    demo._write_json_atomic(config.decision_summary_output, summary_report)
    return {
        "goal_audit": str(config.goal_audit_output),
        "goal_audit_status": goal_report.get("status"),
        "decision_summary": str(config.decision_summary_output),
        "decision_summary_status": summary_report.get("status"),
    }


def _terminal_report(
    *,
    next_step_report: JsonObject,
    config: ScoreCheckpointConfig,
) -> JsonObject:
    decision = next_step_report.get("decision") or next_step_report.get("action")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "terminal_no_generation_expansion_boundary",
        "decision": decision,
        "reason": next_step_report.get("reason"),
        "spend_authorized": False,
        "next_action": next_step_report.get("action"),
        "artifacts": {
            "post_score_status": str(config.post_score_status_output),
            "next_step": str(config.next_step_output),
        },
        "review_note": (
            "The promotion gate did not produce a generation-expansion spend "
            "boundary; do not buy further generation for this candidate/scope."
        ),
    }


def run_score_checkpoint(
    config: ScoreCheckpointConfig,
    *,
    python: Path | str = sys.executable,
) -> JsonObject:
    """Refresh post-score status and prepare the next no-spend checkpoint."""

    status_report = status.build_status_report(
        manifest_path=config.manifest,
        readiness_path=config.current_readiness,
        reference_targets_path=config.reference_targets,
        operational_gate_path=config.operational_gate,
        promotion_gate_path=config.promotion_gate,
    )
    demo._write_json_atomic(config.post_score_status_output, status_report)

    next_step_report = next_step.build_next_step_plan(
        _next_step_config(config),
        python=python,
    )
    demo._write_json_atomic(config.next_step_output, next_step_report)

    if next_step_report.get("terminal") is True:
        return _terminal_report(next_step_report=next_step_report, config=config)

    action = next_step_report.get("action")
    if action != "expand_generation_scope":
        raise TrialQAScoreCheckpointError(
            f"post-score checkpoint expected expansion boundary, got {action!r}"
        )

    question_start, question_limit, repeat_limit = _scope_from_next_step(next_step_report)
    preflight_report = generation_preflight.run_preflight(
        _preflight_config(
            config,
            question_start=question_start,
            question_limit=question_limit,
            repeat_limit=repeat_limit,
        ),
        python=python,
    )
    demo._write_json_atomic(config.generation_preflight_output, preflight_report)

    review_report = spend_review.build_spend_review_packet(
        preflight_path=config.generation_preflight_output,
        bundle_verification_path=config.audit_bundle_verification_output,
        next_step_path=config.next_step_output,
        spend_review_path=config.spend_review_output,
    )
    demo._write_json_atomic(config.spend_review_output, review_report)
    next_summary = _optional_summary_outputs(config=config)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_generation_expansion_spend_authorization",
        "spend_authorized": False,
        "next_action": action,
        "scope": {
            "question_start": question_start,
            "question_limit": question_limit,
            "repeat_limit": repeat_limit,
        },
        "generation_preflight_status": preflight_report.get("status"),
        "spend_review_status": review_report.get("status"),
        "pre_spend_guard_check": _spend_guard_check_command(
            spend_review_path=config.spend_review_output,
            python=python,
        ),
        "guarded_spend_command": review_report.get("guarded_spend_command"),
        "safe_no_spend_command": review_report.get("safe_no_spend_command"),
        "artifacts": {
            "post_score_status": str(config.post_score_status_output),
            "next_step": str(config.next_step_output),
            "generation_preflight": str(config.generation_preflight_output),
            "readiness": str(config.expansion_readiness_output),
            "generation_canary_summary": str(config.generation_summary_output),
            "operational_gate": str(config.expansion_operational_gate_output),
            "expansion_status": str(config.expansion_status_output),
            "protocol_audit": str(config.protocol_audit_output),
            "reference_alignment": str(config.reference_alignment_output),
            "audit_bundle": str(config.audit_bundle_output),
            "audit_bundle_verification": str(config.audit_bundle_verification_output),
            "spend_review": str(config.spend_review_output),
        },
        "review_note": (
            "Expansion generation evidence is staged, but this checkpoint does "
            "not authorize model spend; inspect the spend-review packet before "
            "running --yes-spend."
        ),
    }
    if next_summary is not None:
        report["next_boundary_summary"] = next_summary
    return report


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
    parser.add_argument("--current-readiness", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument(
        "--skills-distillation-repo",
        type=Path,
        default=Path("skills-distillation"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--post-score-status-output", type=Path, required=True)
    parser.add_argument("--next-step-output", type=Path, required=True)
    parser.add_argument("--expansion-readiness-output", type=Path, required=True)
    parser.add_argument("--expansion-operational-gate-output", type=Path, required=True)
    parser.add_argument("--generation-summary-output", type=Path, required=True)
    parser.add_argument("--expansion-status-output", type=Path, required=True)
    parser.add_argument("--protocol-audit-output", type=Path, required=True)
    parser.add_argument("--reference-alignment-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-output", type=Path, required=True)
    parser.add_argument("--audit-bundle-verification-output", type=Path, required=True)
    parser.add_argument("--generation-preflight-output", type=Path, required=True)
    parser.add_argument("--spend-review-output", type=Path, required=True)
    parser.add_argument("--ladder-rehearsal", type=Path)
    parser.add_argument("--goal-audit-output", type=Path)
    parser.add_argument("--decision-summary-output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> ScoreCheckpointConfig:
    return ScoreCheckpointConfig(
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
        current_readiness=args.current_readiness,
        operational_gate=args.operational_gate,
        promotion_gate=args.promotion_gate,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        reference_targets=args.reference_targets,
        runbook=args.runbook,
        skills_distillation_repo=args.skills_distillation_repo,
        artifact_dir=args.artifact_dir,
        artifact_stem=args.artifact_stem,
        post_score_status_output=args.post_score_status_output,
        next_step_output=args.next_step_output,
        expansion_readiness_output=args.expansion_readiness_output,
        expansion_operational_gate_output=args.expansion_operational_gate_output,
        generation_summary_output=args.generation_summary_output,
        expansion_status_output=args.expansion_status_output,
        protocol_audit_output=args.protocol_audit_output,
        reference_alignment_output=args.reference_alignment_output,
        audit_bundle_output=args.audit_bundle_output,
        audit_bundle_verification_output=args.audit_bundle_verification_output,
        generation_preflight_output=args.generation_preflight_output,
        spend_review_output=args.spend_review_output,
        ladder_rehearsal=args.ladder_rehearsal,
        goal_audit_output=args.goal_audit_output,
        decision_summary_output=args.decision_summary_output,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_score_checkpoint(_config_from_args(args), python=args.python)
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
