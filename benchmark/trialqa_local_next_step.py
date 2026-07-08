# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan the next no-spend TrialQA command from a staged status artifact."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_next_step_plan.v1"
JsonObject = dict[str, Any]


class TrialQANextStepError(RuntimeError):
    """A status artifact cannot be converted into a safe next step."""


@dataclass(frozen=True)
class NextStepConfig:
    status: Path
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
    reference_targets: Path
    runbook: Path
    artifact_dir: Path
    artifact_stem: str
    workers: int
    max_generation_attempts: int
    operational_gate: Path | None = None
    promotion_gate: Path | None = None
    ladder_rehearsal: Path | None = None
    skills_distillation_repo: Path | None = Path("skills-distillation")


@dataclass(frozen=True)
class Scope:
    question_start: int
    question_limit: int
    repeat_limit: int


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQANextStepError(f"{label} must be an object")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrialQANextStepError(f"{label} must be an integer")
    return value


def _repeat_limit(value: object, label: str) -> int:
    if not isinstance(value, list) or not value:
        raise TrialQANextStepError(f"{label} must be a non-empty list")
    return max(_require_int(item, label) for item in value)


def _scope_suffix(scope: Scope) -> str:
    if scope.question_start < 0:
        raise TrialQANextStepError("question_start cannot be negative")
    if scope.question_limit < 1:
        raise TrialQANextStepError("question_limit must be positive")
    if scope.repeat_limit < 1:
        raise TrialQANextStepError("repeat_limit must be positive")
    end = scope.question_start + scope.question_limit - 1
    return f"q{scope.question_start}-q{end}-r{scope.repeat_limit}"


def _generation_scope_from_readiness(status_report: Mapping[str, object]) -> Scope:
    readiness = _require_mapping(status_report.get("readiness"), "status readiness")
    attestation = _require_mapping(
        readiness.get("scope_attestation"),
        "readiness scope_attestation",
    )
    return Scope(
        question_start=_require_int(attestation.get("question_start"), "question_start"),
        question_limit=_require_int(attestation.get("question_limit"), "question_limit"),
        repeat_limit=_repeat_limit(attestation.get("selected_repeat_indices"), "repeat_indices"),
    )


def _generation_scope_from_next_action(next_action: Mapping[str, object]) -> Scope:
    return Scope(
        question_start=_require_int(next_action.get("question_start"), "question_start"),
        question_limit=_require_int(next_action.get("question_limit"), "question_limit"),
        repeat_limit=_require_int(next_action.get("repeat_limit"), "repeat_limit"),
    )


def _score_scope_from_operational(status_report: Mapping[str, object]) -> Scope:
    operational = _require_mapping(status_report.get("operational_gate"), "operational gate")
    attestation = _require_mapping(
        operational.get("selection_attestation"),
        "operational selection_attestation",
    )
    return Scope(
        question_start=_require_int(attestation.get("question_start"), "question_start"),
        question_limit=_require_int(attestation.get("question_limit"), "question_limit"),
        repeat_limit=_repeat_limit(attestation.get("selected_repeat_indices"), "repeat_indices"),
    )


def _artifact_path(config: NextStepConfig, name: str, suffix: str) -> Path:
    return config.artifact_dir / f"{name}-{config.artifact_stem}-{suffix}.json"


def _common_path_args(config: NextStepConfig) -> list[str]:
    return [
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
    ]


def _skills_distillation_repo_args(config: NextStepConfig) -> list[str]:
    if config.skills_distillation_repo is None:
        return []
    return ["--skills-distillation-repo", str(config.skills_distillation_repo)]


def _ladder_rehearsal_path(config: NextStepConfig) -> Path:
    if config.ladder_rehearsal is not None:
        return config.ladder_rehearsal
    return config.artifact_dir / f"ladder-rehearsal-{config.artifact_stem}.json"


def _generation_preflight_command(
    config: NextStepConfig,
    scope: Scope,
    *,
    python: Path | str,
) -> list[str]:
    suffix = _scope_suffix(scope)
    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_preflight",
        *_common_path_args(config),
        "--question-start",
        str(scope.question_start),
        "--question-limit",
        str(scope.question_limit),
        "--repeat-limit",
        str(scope.repeat_limit),
        "--workers",
        str(config.workers),
        "--max-generation-attempts",
        str(config.max_generation_attempts),
        "--reference-targets",
        str(config.reference_targets),
        "--runbook",
        str(config.runbook),
        *_skills_distillation_repo_args(config),
        "--readiness-output",
        str(_artifact_path(config, "readiness", suffix)),
        "--gate-output",
        str(_artifact_path(config, "gate-operational", suffix)),
        "--generation-summary-output",
        str(_artifact_path(config, "canary-generation-dryrun", suffix)),
        "--status-output",
        str(_artifact_path(config, "status", suffix)),
        "--protocol-audit-output",
        str(_artifact_path(config, "protocol-audit", suffix)),
        "--reference-alignment-output",
        str(_artifact_path(config, "reference-alignment", suffix)),
        "--audit-bundle-output",
        str(_artifact_path(config, "pre-spend-audit-bundle", suffix)),
        "--audit-bundle-verification-output",
        str(_artifact_path(config, "pre-spend-audit-bundle-verification", suffix)),
        "--ladder-rehearsal",
        str(_ladder_rehearsal_path(config)),
        "--output",
        str(_artifact_path(config, "no-spend-preflight", suffix)),
    ]
    if config.operational_gate is not None:
        command.extend(["--operational-gate", str(config.operational_gate)])
    if config.promotion_gate is not None:
        command.extend(["--promotion-gate", str(config.promotion_gate)])
    return command


def _operational_gate_path(config: NextStepConfig, suffix: str) -> Path:
    return config.operational_gate or _artifact_path(config, "gate-operational", suffix)


def _score_preflight_command(
    config: NextStepConfig,
    scope: Scope,
    *,
    python: Path | str,
) -> list[str]:
    suffix = _scope_suffix(scope)
    return [
        str(python),
        "-m",
        "benchmark.trialqa_local_score_preflight",
        *_common_path_args(config),
        "--operational-gate",
        str(_operational_gate_path(config, suffix)),
        "--question-start",
        str(scope.question_start),
        "--question-limit",
        str(scope.question_limit),
        "--repeat-limit",
        str(scope.repeat_limit),
        "--workers",
        str(config.workers),
        "--max-generation-attempts",
        str(config.max_generation_attempts),
        "--reference-targets",
        str(config.reference_targets),
        "--runbook",
        str(config.runbook),
        *_skills_distillation_repo_args(config),
        "--readiness-output",
        str(_artifact_path(config, "readiness", suffix)),
        "--score-summary-output",
        str(_artifact_path(config, "canary-score-dryrun", suffix)),
        "--promotion-gate-output",
        str(_artifact_path(config, "gate-promotion", suffix)),
        "--status-output",
        str(_artifact_path(config, "status", suffix)),
        "--protocol-audit-output",
        str(_artifact_path(config, "protocol-audit", suffix)),
        "--reference-alignment-output",
        str(_artifact_path(config, "reference-alignment", suffix)),
        "--audit-bundle-output",
        str(_artifact_path(config, "pre-score-spend-audit-bundle", suffix)),
        "--audit-bundle-verification-output",
        str(_artifact_path(config, "pre-score-spend-audit-bundle-verification", suffix)),
        "--ladder-rehearsal",
        str(_ladder_rehearsal_path(config)),
        "--output",
        str(_artifact_path(config, "no-spend-score-preflight", suffix)),
    ]


def _command_entry(kind: str, command: Sequence[str]) -> JsonObject:
    return {
        "kind": kind,
        "command": list(command),
        "shell_command": shlex.join(command),
        "spend_authorized": False,
        "note": "safe next command is a no-spend preflight and does not include --yes-spend",
    }


def _manifest_id(manifest_path: Path) -> str:
    manifest = demo._read_json_object(manifest_path, "experiment manifest")
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str):
        raise TrialQANextStepError("manifest has no manifest_id")
    return manifest_id


def build_next_step_plan(
    config: NextStepConfig,
    *,
    python: Path | str = sys.executable,
) -> JsonObject:
    """Build the next safe no-spend command from the staged status report."""

    status_report = demo._read_json_object(config.status, "protocol status")
    if status_report.get("schema_version") != "switchyard.trialqa_protocol_status.v1":
        raise TrialQANextStepError("status report has invalid schema_version")
    status_manifest = _require_mapping(status_report.get("manifest"), "status manifest")
    manifest_id = _manifest_id(config.manifest)
    if status_manifest.get("manifest_id") != manifest_id:
        raise TrialQANextStepError("status report belongs to a different manifest")
    next_action = _require_mapping(status_report.get("next_action"), "status next_action")
    action = next_action.get("action")
    base: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status_path": str(config.status),
        "manifest_id": manifest_id,
        "action": action,
        "reason": next_action.get("reason"),
    }

    if action == "run_guarded_generation_canary":
        scope = _generation_scope_from_readiness(status_report)
        command = _generation_preflight_command(config, scope, python=python)
        return {
            **base,
            "terminal": False,
            "scope": {**scope.__dict__, "suffix": _scope_suffix(scope)},
            "safe_next_command": _command_entry("generation_preflight", command),
            "live_spend_boundary": "inspect preflight output before running its guarded --yes-spend command",
        }
    if action == "expand_generation_scope":
        scope = _generation_scope_from_next_action(next_action)
        command = _generation_preflight_command(config, scope, python=python)
        return {
            **base,
            "terminal": False,
            "scope": {**scope.__dict__, "suffix": _scope_suffix(scope)},
            "safe_next_command": _command_entry("generation_expansion_preflight", command),
            "live_spend_boundary": "inspect preflight output before running its guarded --yes-spend command",
        }
    if action == "run_guarded_score_canary":
        scope = _score_scope_from_operational(status_report)
        command = _score_preflight_command(config, scope, python=python)
        return {
            **base,
            "terminal": False,
            "scope": {**scope.__dict__, "suffix": _scope_suffix(scope)},
            "safe_next_command": _command_entry("score_preflight", command),
            "live_spend_boundary": "inspect score preflight output before running its guarded --yes-spend command",
        }
    if action == "kill_candidate":
        return {
            **base,
            "terminal": True,
            "safe_next_command": None,
            "decision": "kill_candidate",
        }
    if action == "prospective_directional_scope_complete":
        return {
            **base,
            "terminal": True,
            "safe_next_command": None,
            "decision": "prospective_directional_scope_complete",
        }
    if action == "fix_readiness_before_spend":
        return {
            **base,
            "terminal": True,
            "safe_next_command": None,
            "decision": "fix_readiness_before_spend",
        }
    raise TrialQANextStepError(f"unsupported next action {action!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
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
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument(
        "--skills-distillation-repo",
        type=Path,
        default=Path("skills-distillation"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument("--promotion-gate", type=Path)
    parser.add_argument("--ladder-rehearsal", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> NextStepConfig:
    return NextStepConfig(
        status=args.status,
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
        reference_targets=args.reference_targets,
        runbook=args.runbook,
        skills_distillation_repo=args.skills_distillation_repo,
        artifact_dir=args.artifact_dir,
        artifact_stem=args.artifact_stem,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        operational_gate=args.operational_gate,
        promotion_gate=args.promotion_gate,
        ladder_rehearsal=args.ladder_rehearsal,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_next_step_plan(_config_from_args(args), python=args.python)
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
