# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-spend synthetic rehearsal of the local TrialQA promotion ladder.

This command does not run generation, scoring, judge calls, or network access.
It writes synthetic readiness/gate artifacts from the real manifest and asks the
status plus next-step planners to walk the intended ladder:

* first 4-question generation spend boundary;
* score spend boundary after an operational promotion;
* generation-expansion boundaries for 8 x 1, 8 x 3, and 8 x 5; and
* terminal kill / directional-complete outcomes.

The output is a compact audit artifact that catches state-machine drift before a
long benchmark is bought.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_next_step as next_step  # noqa: E402
import benchmark.trialqa_local_status as status  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_ladder_rehearsal.v1"
JsonObject = dict[str, Any]


class TrialQALadderRehearsalError(RuntimeError):
    """The synthetic promotion ladder cannot be rehearsed safely."""


@dataclass(frozen=True)
class RehearsalConfig:
    manifest: Path
    reference_targets: Path
    dataset: Path
    experiment_root: Path
    doctor: Path
    population_report: Path
    candidate: Path
    switchyard: Path
    codex: Path
    tooluniverse: Path
    profile: Path
    runbook: Path
    artifact_dir: Path
    artifact_stem: str
    rehearsal_dir: Path
    workers: int
    max_generation_attempts: int


@dataclass(frozen=True)
class Scope:
    question_start: int
    question_limit: int
    repeat_limit: int

    @property
    def repeat_indices(self) -> list[int]:
        return list(range(1, self.repeat_limit + 1))

    @property
    def suffix(self) -> str:
        return f"q{self.question_start}-q{self.question_start + self.question_limit - 1}-r{self.repeat_limit}"


@dataclass(frozen=True)
class ExpectedNextStep:
    action: str
    terminal: bool
    safe_kind: str | None = None
    decision: str | None = None
    scope: Scope | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    current_scope: Scope
    expected: ExpectedNextStep
    readiness_status: str = "ready_for_generation"
    operational_decision: str | None = None
    promotion_decision: str | None = None
    promotion_performance_eligible: bool = False


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQALadderRehearsalError(f"{label} must be an object")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrialQALadderRehearsalError(f"{label} must be an integer")
    return value


def _primary_scope(manifest: Mapping[str, object]) -> Scope:
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    primary = _require_mapping(
        protocol.get("primary_evaluation_scope"),
        "manifest primary_evaluation_scope",
    )
    scope = Scope(
        question_start=_require_int(primary.get("question_start"), "question_start"),
        question_limit=_require_int(primary.get("question_count"), "question_count"),
        repeat_limit=_require_int(primary.get("repeat_count"), "repeat_count"),
    )
    if scope.question_limit < 4:
        raise TrialQALadderRehearsalError("rehearsal requires at least four questions")
    if scope.repeat_limit < 5:
        raise TrialQALadderRehearsalError("rehearsal requires at least five repeats")
    return scope


def _manifest_id(manifest: Mapping[str, object]) -> str:
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str):
        raise TrialQALadderRehearsalError("manifest has no string manifest_id")
    return manifest_id


def _write_json(path: Path, payload: object) -> Path:
    demo._write_json_atomic(path, payload)
    return path


def _readiness_report(
    *,
    manifest_id: str,
    scope: Scope,
    status_value: str,
) -> JsonObject:
    states = {
        f"trialqa-rehearsal-q{question}-r{repeat:03d}-{condition}": "not_started"
        for question in range(scope.question_start, scope.question_start + scope.question_limit)
        for repeat in scope.repeat_indices
        for condition in ("baseline", "treatment")
    }
    if status_value == "ready_for_generation_expansion":
        first_key = next(iter(states))
        states[first_key] = "completed"
    task_count = scope.question_limit * len(scope.repeat_indices) * 2
    return {
        "schema_version": "switchyard.trialqa_canary_readiness.v1",
        "status": status_value,
        "manifest": {"manifest_id": manifest_id},
        "first_generation_canary": {
            "task_count": task_count,
            "pair_count": task_count // 2,
            "selected_task_states": states,
            "scope_attestation": {
                "question_start": scope.question_start,
                "question_limit": scope.question_limit,
                "selected_question_count": scope.question_limit,
                "selected_repeat_indices": scope.repeat_indices,
                "selected_task_count": task_count,
            },
        },
    }


def _gate_report(
    *,
    manifest_id: str,
    gate: str,
    decision: str,
    scope: Scope,
    performance_eligible: bool = False,
) -> JsonObject:
    task_count = scope.question_limit * len(scope.repeat_indices) * 2
    return {
        "schema_version": "switchyard.trialqa_gate_report.v3",
        "manifest_id": manifest_id,
        "gate": gate,
        "decision": decision,
        "performance_eligible": performance_eligible,
        "scope": {
            "task_count": task_count,
            "pair_count": task_count // 2,
            "confirmatory_scope_complete": scope.repeat_limit >= 5,
            "selection_attestation": {
                "question_start": scope.question_start,
                "question_limit": scope.question_limit,
                "selected_repeat_indices": scope.repeat_indices,
                "selected_question_count": scope.question_limit,
                "selected_task_count": task_count,
            },
        },
    }


def _scenario_paths(config: RehearsalConfig, scenario_id: str) -> dict[str, Path]:
    return {
        "readiness": config.rehearsal_dir / f"readiness-{scenario_id}.json",
        "operational_gate": config.rehearsal_dir / f"gate-operational-{scenario_id}.json",
        "promotion_gate": config.rehearsal_dir / f"gate-promotion-{scenario_id}.json",
        "status": config.rehearsal_dir / f"status-{scenario_id}.json",
        "next_step": config.rehearsal_dir / f"next-step-{scenario_id}.json",
    }


def _next_step_config(
    config: RehearsalConfig,
    *,
    status_path: Path,
    operational_gate_path: Path | None,
    promotion_gate_path: Path | None,
) -> next_step.NextStepConfig:
    return next_step.NextStepConfig(
        status=status_path,
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
        operational_gate=operational_gate_path,
        promotion_gate=promotion_gate_path,
    )


def _scope_dict(scope: Scope | None) -> JsonObject | None:
    if scope is None:
        return None
    return {
        "question_start": scope.question_start,
        "question_limit": scope.question_limit,
        "repeat_limit": scope.repeat_limit,
        "suffix": scope.suffix,
    }


def _task_count(scope: Scope) -> int:
    return scope.question_limit * scope.repeat_limit * 2


def _ladder_budget(primary: Scope) -> JsonObject:
    first = Scope(primary.question_start, 4, 1)
    full_r1 = Scope(primary.question_start, primary.question_limit, 1)
    full_r3 = Scope(primary.question_start, primary.question_limit, 3)
    full_r5 = Scope(primary.question_start, primary.question_limit, primary.repeat_limit)
    scopes = [first, full_r1, full_r3, full_r5]
    boundaries: list[JsonObject] = []
    generated_total = 0
    judged_total = 0
    previous_generated_scope_total = 0
    previous_judged_scope_total = 0
    for scope in scopes:
        scope_total = _task_count(scope)
        generation_increment = scope_total - previous_generated_scope_total
        if generation_increment < 0:
            raise TrialQALadderRehearsalError("ladder generation scope regressed")
        generated_total += generation_increment
        boundaries.append(
            {
                "stage": "generation",
                "scope": _scope_dict(scope),
                "incremental_model_calls": generation_increment,
                "incremental_judge_calls": 0,
                "cumulative_model_calls": generated_total,
                "cumulative_judge_calls": judged_total,
                "reason": (
                    "initial guarded generation canary"
                    if previous_generated_scope_total == 0
                    else "generation expansion after scored promotion"
                ),
            }
        )
        previous_generated_scope_total = scope_total
        score_increment = scope_total - previous_judged_scope_total
        if score_increment < 0:
            raise TrialQALadderRehearsalError("ladder score scope regressed")
        judged_total += score_increment
        boundaries.append(
            {
                "stage": "score",
                "scope": _scope_dict(scope),
                "incremental_model_calls": 0,
                "incremental_judge_calls": score_increment,
                "cumulative_model_calls": generated_total,
                "cumulative_judge_calls": judged_total,
                "reason": "score only after operational promotion",
            }
        )
        previous_judged_scope_total = scope_total
    return {
        "policy": "all-promote upper bound with resumable-ledger incremental work",
        "primary_scope": _scope_dict(primary),
        "first_spend_boundary": {
            "stage": "generation",
            "scope": _scope_dict(first),
            "expected_model_calls": _task_count(first),
            "expected_judge_calls": 0,
        },
        "all_promote_boundaries": boundaries,
        "max_model_calls_before_directional_completion": generated_total,
        "max_judge_calls_before_directional_completion": judged_total,
        "max_total_live_calls_before_directional_completion": generated_total + judged_total,
        "review_note": (
            "This budget is a no-spend upper bound for the prospective directional "
            "ladder. Every generation or score stage remains separately gated and "
            "can stop early on a predeclared kill."
        ),
    }


def _expectation_status(
    *,
    expected: ExpectedNextStep,
    report: Mapping[str, object],
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if report.get("action") != expected.action:
        failures.append(f"action {report.get('action')!r} != {expected.action!r}")
    if report.get("terminal") is not expected.terminal:
        failures.append(f"terminal {report.get('terminal')!r} != {expected.terminal!r}")
    if expected.decision is not None and report.get("decision") != expected.decision:
        failures.append(f"decision {report.get('decision')!r} != {expected.decision!r}")
    safe_command = report.get("safe_next_command")
    if expected.safe_kind is None:
        if safe_command is not None:
            failures.append("expected no safe_next_command")
    elif not isinstance(safe_command, Mapping):
        failures.append("safe_next_command is missing")
    elif safe_command.get("kind") != expected.safe_kind:
        failures.append(f"safe kind {safe_command.get('kind')!r} != {expected.safe_kind!r}")
    elif "--yes-spend" in cast(list[object], safe_command.get("command", [])):
        failures.append("safe command unexpectedly contains --yes-spend")
    if expected.scope is not None:
        observed_scope = report.get("scope")
        if not isinstance(observed_scope, Mapping):
            failures.append("expected scope is missing")
        else:
            expected_scope = _scope_dict(expected.scope)
            for key in ("question_start", "question_limit", "repeat_limit", "suffix"):
                if observed_scope.get(key) != cast(Mapping[str, object], expected_scope).get(key):
                    failures.append(
                        f"scope {key} {observed_scope.get(key)!r} != "
                        f"{cast(Mapping[str, object], expected_scope).get(key)!r}"
                    )
    return ("passed" if not failures else "failed", failures)


def _run_scenario(
    config: RehearsalConfig,
    *,
    manifest_id: str,
    scenario: Scenario,
    python: Path | str,
) -> JsonObject:
    paths = _scenario_paths(config, scenario.scenario_id)
    readiness_path = _write_json(
        paths["readiness"],
        _readiness_report(
            manifest_id=manifest_id,
            scope=scenario.current_scope,
            status_value=scenario.readiness_status,
        ),
    )
    operational_gate_path: Path | None = None
    if scenario.operational_decision is not None:
        operational_gate_path = _write_json(
            paths["operational_gate"],
            _gate_report(
                manifest_id=manifest_id,
                gate="operational",
                decision=scenario.operational_decision,
                scope=scenario.current_scope,
            ),
        )
    promotion_gate_path: Path | None = None
    if scenario.promotion_decision is not None:
        promotion_gate_path = _write_json(
            paths["promotion_gate"],
            _gate_report(
                manifest_id=manifest_id,
                gate="promotion",
                decision=scenario.promotion_decision,
                scope=scenario.current_scope,
                performance_eligible=scenario.promotion_performance_eligible,
            ),
        )
    status_report = status.build_status_report(
        manifest_path=config.manifest,
        readiness_path=readiness_path,
        reference_targets_path=config.reference_targets,
        operational_gate_path=operational_gate_path,
        promotion_gate_path=promotion_gate_path,
    )
    status_path = _write_json(paths["status"], status_report)
    next_step_report = next_step.build_next_step_plan(
        _next_step_config(
            config,
            status_path=status_path,
            operational_gate_path=operational_gate_path,
            promotion_gate_path=promotion_gate_path,
        ),
        python=python,
    )
    next_step_path = _write_json(paths["next_step"], next_step_report)
    check_status, failures = _expectation_status(
        expected=scenario.expected,
        report=next_step_report,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "status": check_status,
        "failures": failures,
        "current_scope": _scope_dict(scenario.current_scope),
        "expected": {
            "action": scenario.expected.action,
            "terminal": scenario.expected.terminal,
            "safe_kind": scenario.expected.safe_kind,
            "decision": scenario.expected.decision,
            "scope": _scope_dict(scenario.expected.scope),
        },
        "observed": {
            "action": next_step_report.get("action"),
            "terminal": next_step_report.get("terminal"),
            "safe_kind": _require_mapping(
                next_step_report.get("safe_next_command"),
                "safe_next_command",
            ).get("kind")
            if next_step_report.get("safe_next_command") is not None
            else None,
            "decision": next_step_report.get("decision"),
            "scope": next_step_report.get("scope"),
        },
        "artifacts": {
            "readiness": str(readiness_path),
            "operational_gate": str(operational_gate_path)
            if operational_gate_path is not None
            else None,
            "promotion_gate": str(promotion_gate_path)
            if promotion_gate_path is not None
            else None,
            "status": str(status_path),
            "next_step": str(next_step_path),
        },
    }


def _scenarios(primary: Scope) -> list[Scenario]:
    q4 = Scope(primary.question_start, 4, 1)
    q_full_r1 = Scope(primary.question_start, primary.question_limit, 1)
    q_full_r3 = Scope(primary.question_start, primary.question_limit, 3)
    q_full_r5 = Scope(primary.question_start, primary.question_limit, primary.repeat_limit)
    return [
        Scenario(
            "initial-generation",
            q4,
            ExpectedNextStep(
                "run_guarded_generation_canary",
                terminal=False,
                safe_kind="generation_preflight",
                scope=q4,
            ),
        ),
        Scenario(
            "post-generation-promote",
            q4,
            ExpectedNextStep(
                "run_guarded_score_canary",
                terminal=False,
                safe_kind="score_preflight",
                scope=q4,
            ),
            operational_decision="promote_to_score",
        ),
        Scenario(
            "post-generation-kill",
            q4,
            ExpectedNextStep(
                "kill_candidate",
                terminal=True,
                decision="kill_candidate",
            ),
            operational_decision="kill",
        ),
        Scenario(
            "post-score-q4-promote",
            q4,
            ExpectedNextStep(
                "expand_generation_scope",
                terminal=False,
                safe_kind="generation_expansion_preflight",
                scope=q_full_r1,
            ),
            operational_decision="promote_to_score",
            promotion_decision="promote_to_next_cohort",
        ),
        Scenario(
            "post-score-q8-r1-promote",
            q_full_r1,
            ExpectedNextStep(
                "expand_generation_scope",
                terminal=False,
                safe_kind="generation_expansion_preflight",
                scope=q_full_r3,
            ),
            readiness_status="ready_for_generation_expansion",
            operational_decision="promote_to_score",
            promotion_decision="promote_to_next_cohort",
        ),
        Scenario(
            "post-score-q8-r3-promote",
            q_full_r3,
            ExpectedNextStep(
                "expand_generation_scope",
                terminal=False,
                safe_kind="generation_expansion_preflight",
                scope=q_full_r5,
            ),
            readiness_status="ready_for_generation_expansion",
            operational_decision="promote_to_score",
            promotion_decision="promote_to_next_cohort",
        ),
        Scenario(
            "post-score-q8-r5-promote",
            q_full_r5,
            ExpectedNextStep(
                "prospective_directional_scope_complete",
                terminal=True,
                decision="prospective_directional_scope_complete",
            ),
            readiness_status="ready_for_generation_expansion",
            operational_decision="promote_to_score",
            promotion_decision="promote_to_next_cohort",
            promotion_performance_eligible=True,
        ),
        Scenario(
            "post-score-kill",
            q4,
            ExpectedNextStep(
                "kill_candidate",
                terminal=True,
                decision="kill_candidate",
            ),
            operational_decision="promote_to_score",
            promotion_decision="kill",
        ),
    ]


def run_ladder_rehearsal(
    config: RehearsalConfig,
    *,
    python: Path | str = sys.executable,
) -> JsonObject:
    """Write and verify synthetic ladder artifacts for the configured manifest."""

    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    manifest_id = _manifest_id(manifest)
    primary = _primary_scope(manifest)
    config.rehearsal_dir.mkdir(parents=True, exist_ok=True)
    scenario_reports = [
        _run_scenario(
            config,
            manifest_id=manifest_id,
            scenario=scenario,
            python=python,
        )
        for scenario in _scenarios(primary)
    ]
    failed = [
        report
        for report in scenario_reports
        if report.get("status") != "passed"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failed else "failed",
        "manifest_id": manifest_id,
        "spend_authorized": False,
        "model_calls": 0,
        "judge_calls": 0,
        "primary_scope": _scope_dict(primary),
        "ladder_budget": _ladder_budget(primary),
        "scenario_count": len(scenario_reports),
        "failed_scenario_count": len(failed),
        "scenarios": scenario_reports,
        "rehearsal_dir": str(config.rehearsal_dir),
        "review_note": (
            "This rehearsal uses synthetic gates to validate the staged control "
            "flow only; it does not prove live quality or efficiency outcomes."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--population-report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--switchyard", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--tooluniverse", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--rehearsal-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> RehearsalConfig:
    return RehearsalConfig(
        manifest=args.manifest,
        reference_targets=args.reference_targets,
        dataset=args.dataset,
        experiment_root=args.experiment_root,
        doctor=args.doctor,
        population_report=args.population_report,
        candidate=args.candidate,
        switchyard=args.switchyard,
        codex=args.codex,
        tooluniverse=args.tooluniverse,
        profile=args.profile,
        runbook=args.runbook,
        artifact_dir=args.artifact_dir,
        artifact_stem=args.artifact_stem,
        rehearsal_dir=args.rehearsal_dir,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_ladder_rehearsal(_config_from_args(args), python=args.python)
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
