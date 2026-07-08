# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-alignment audit for the local TrialQA skill-distillation demo.

This command compares the current Switchyard manifest against the frozen
TrialQA reference targets. It deliberately separates two ideas that are easy to
blur during a long benchmark run:

* whether the current canary matches the reference comparison shape
  (paired skill-off/skill-on, same repeats, ToolUniverse-backed TrialQA);
* whether it is an official 96-question LABBench2 reproduction.

The prospective local canary should pass the first check and fail/mark missing
for the second. That is a feature: it prevents a quick proxy run from becoming
an accidental performance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_reference_alignment.v1"
JsonObject = dict[str, Any]


class TrialQAReferenceAlignmentError(RuntimeError):
    """Reference-alignment inputs are malformed."""


@dataclass(frozen=True)
class ReferenceAlignmentConfig:
    manifest: Path
    reference_targets: Path
    skills_distillation_repo: Path | None = None


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQAReferenceAlignmentError(f"{label} must be an object")
    return value


def _require_schema(report: Mapping[str, object], schema: str, label: str) -> None:
    if report.get("schema_version") != schema:
        raise TrialQAReferenceAlignmentError(f"{label} has invalid schema_version")


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrialQAReferenceAlignmentError(f"{label} must be an integer")
    return value


def _requirement(
    requirement_id: str,
    status: str,
    evidence: str,
    *,
    required_for_canary: bool,
    required_for_official_reproduction: bool,
) -> JsonObject:
    return {
        "id": requirement_id,
        "status": status,
        "evidence": evidence,
        "required_for_canary": required_for_canary,
        "required_for_official_reproduction": required_for_official_reproduction,
    }


def _reference_population(reference: Mapping[str, object]) -> Mapping[str, object]:
    population = _require_mapping(reference.get("population"), "reference population")
    ok = (
        population.get("dataset") == "LABBench2 TrialQA"
        and population.get("heldout_questions") == 96
        and population.get("repeats_per_question") == 5
        and population.get("trials") == 480
        and population.get("tool_provider") == "ToolUniverse MCP"
        and population.get("injected_context") is False
    )
    if not ok:
        raise TrialQAReferenceAlignmentError(
            f"reference population is not the expected TrialQA target: {dict(population)}"
        )
    return population


def _primary_scope(manifest: Mapping[str, object]) -> Mapping[str, object]:
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    return _require_mapping(
        protocol.get("primary_evaluation_scope"),
        "manifest primary_evaluation_scope",
    )


def _task_arms(manifest: Mapping[str, object]) -> set[str]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise TrialQAReferenceAlignmentError("manifest tasks must be a list")
    arms: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise TrialQAReferenceAlignmentError("manifest task must be an object")
        arm = task.get("arm")
        if not isinstance(arm, str):
            raise TrialQAReferenceAlignmentError("manifest task arm must be a string")
        arms.add(arm)
    return arms


def _tooluniverse_version(manifest: Mapping[str, object]) -> object:
    runtime = _require_mapping(manifest.get("runtime"), "manifest runtime")
    tooluniverse = _require_mapping(runtime.get("tooluniverse"), "manifest runtime.tooluniverse")
    return tooluniverse.get("version")


def _requirements(
    *,
    manifest: Mapping[str, object],
    reference_population: Mapping[str, object],
    workflow_evidence: Mapping[str, object] | None,
) -> list[JsonObject]:
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    routing = _require_mapping(manifest.get("routing"), "manifest routing")
    primary = _primary_scope(manifest)
    arms = _task_arms(manifest)
    reference_questions = _require_int(
        reference_population.get("heldout_questions"),
        "reference heldout_questions",
    )
    reference_repeats = _require_int(
        reference_population.get("repeats_per_question"),
        "reference repeats_per_question",
    )
    reference_trials = _require_int(reference_population.get("trials"), "reference trials")
    current_questions = _require_int(primary.get("question_count"), "current question_count")
    current_repeats = _require_int(primary.get("repeat_count"), "current repeat_count")
    current_tasks = _require_int(primary.get("task_count"), "current task_count")
    current_is_official = dataset.get("official_labbench2") is True
    current_is_proxy = dataset.get("official_labbench2") is False

    paired_shape_ok = (
        arms == {"baseline", "treatment"}
        and current_repeats == reference_repeats
        and current_tasks == current_questions * current_repeats * 2
    )
    official_reproduction_ok = (
        current_is_official
        and current_questions == reference_questions
        and current_repeats == reference_repeats
        and current_tasks == reference_trials * 2
    )
    prospective_scope_ok = (
        current_is_proxy
        and protocol.get("prospective_population_kind")
        == "trialqa-compatible-clinicaltrials-gov"
        and current_questions == 8
        and current_repeats == reference_repeats
        and current_tasks == 80
    )
    tooluniverse_ok = isinstance(_tooluniverse_version(manifest), str)
    ultra_ok = routing.get("executor_model") == "nvidia/nvidia/nemotron-3-ultra"
    switchyard_ok = protocol.get("batch_driver") == "benchmark/trialqa_local_batch.py"

    requirements = [
        _requirement(
            "reference_population_bound",
            "proved",
            f"reference population is {dict(reference_population)}",
            required_for_canary=True,
            required_for_official_reproduction=True,
        ),
        _requirement(
            "paired_off_on_shape_matches_reference",
            "proved" if paired_shape_ok else "failed",
            (
                f"arms={sorted(arms)}, questions={current_questions!r}, "
                f"repeats={current_repeats!r}, tasks={current_tasks!r}"
            ),
            required_for_canary=True,
            required_for_official_reproduction=True,
        ),
        _requirement(
            "nemotron_ultra_switchyard_runtime_bound",
            "proved" if ultra_ok and switchyard_ok else "failed",
            (
                f"executor_model={routing.get('executor_model')!r}, "
                f"batch_driver={protocol.get('batch_driver')!r}"
            ),
            required_for_canary=True,
            required_for_official_reproduction=True,
        ),
        _requirement(
            "tooluniverse_trialqa_interface_bound",
            "proved" if tooluniverse_ok else "failed",
            f"tooluniverse_version={_tooluniverse_version(manifest)!r}",
            required_for_canary=True,
            required_for_official_reproduction=True,
        ),
        _requirement(
            "prospective_proxy_scope_explicit",
            "proved" if prospective_scope_ok else "failed",
            (
                f"official_labbench2={dataset.get('official_labbench2')!r}, "
                f"prospective_population_kind={protocol.get('prospective_population_kind')!r}, "
                f"questions={current_questions!r}, repeats={current_repeats!r}, tasks={current_tasks!r}"
            ),
            required_for_canary=True,
            required_for_official_reproduction=False,
        ),
        _requirement(
            "official_96_question_reproduction_bound",
            "proved" if official_reproduction_ok else "missing",
            (
                f"official_labbench2={dataset.get('official_labbench2')!r}, "
                f"questions={current_questions!r}/{reference_questions!r}, "
                f"repeats={current_repeats!r}/{reference_repeats!r}, "
                f"current_tasks={current_tasks!r}; official paired reproduction would require "
                f"{reference_trials * 2} baseline+treatment tasks"
            ),
            required_for_canary=False,
            required_for_official_reproduction=True,
        ),
    ]
    if workflow_evidence is not None:
        status = workflow_evidence.get("status")
        requirements.append(
            _requirement(
                "reference_workflow_source_evidence_bound",
                "proved" if status == "proved" else "failed",
                (
                    f"skills_distillation_repo={workflow_evidence.get('repo')!r}, "
                    f"super_reference={workflow_evidence.get('super_reference_status')!r}, "
                    f"ultra_trialqa_reference={workflow_evidence.get('ultra_trialqa_reference_status')!r}, "
                    f"materialization={workflow_evidence.get('materialization_status')!r}, "
                    f"aggregate_metrics={workflow_evidence.get('aggregate_metrics_status')!r}"
                ),
                required_for_canary=True,
                required_for_official_reproduction=True,
            )
        )
    return requirements


def _contains_all(text: str, snippets: Sequence[str]) -> bool:
    return all(snippet in text for snippet in snippets)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TrialQAReferenceAlignmentError(f"missing reference workflow file {path}") from exc


def _skills_distillation_workflow_evidence(repo: Path) -> JsonObject:
    """Summarize the cloned Sergei workflow evidence used for claim scoping."""

    exp1 = repo / "docs" / "exp1.md"
    exp1_ultra = repo / "docs" / "exp1_ultra.md"
    config = repo / "configs" / "trialqa-opencode.harbor.yaml"
    aggregate = repo / "scripts" / "aggregate_trialqa_replicate_metrics.py"
    exp1_text = _read_text(exp1)
    exp1_ultra_text = _read_text(exp1_ultra)
    config_text = _read_text(config)
    aggregate_text = _read_text(aggregate)

    super_reference_ok = _contains_all(
        exp1_text,
        (
            "Trials: 480",
            "Mean: 0.610",
            "Mean: 0.738",
            "Test (distilled):",
            "trial_mean 0.737500",
            "Mean tokens / trial",
            "549,406",
            "384,654",
            "Operational tool calls / trial",
            "15.5",
            "8.6",
        ),
    )
    ultra_placeholder_ok = _contains_all(
        exp1_ultra_text,
        (
            "nvidia/nvidia/nvidia/nemotron-3-ultra",
            "<<PLACEHOLDER>>",
            "Success criterion: heldout accuracy improves or stays flat",
        ),
    )
    materialization_ok = _contains_all(
        config_text,
        (
            "dataset_config: trialqa",
            "train_fraction: 0.2",
            "split_seed: trace2skill-trialqa",
            "n_repeats: 5",
            "tooluniverse_mcp: true",
        ),
    )
    aggregate_metrics_ok = _contains_all(
        aggregate_text,
        (
            '"trial_mean"',
            '"question_macro_mean"',
            '"worst_case"',
            '"oracle"',
            '"token_metrics"',
            "aggregate_token_metrics",
        ),
    )
    status = (
        "proved"
        if super_reference_ok
        and ultra_placeholder_ok
        and materialization_ok
        and aggregate_metrics_ok
        else "failed"
    )
    return {
        "status": status,
        "repo": str(repo),
        "super_reference_status": "complete" if super_reference_ok else "failed",
        "ultra_trialqa_reference_status": (
            "placeholder_only" if ultra_placeholder_ok else "not_proved_missing"
        ),
        "materialization_status": "matched" if materialization_ok else "failed",
        "aggregate_metrics_status": "matched" if aggregate_metrics_ok else "failed",
        "files": {
            "super_exp1": {
                "path": str(exp1),
                "sha256": demo._sha256_file(exp1),
            },
            "ultra_exp1": {
                "path": str(exp1_ultra),
                "sha256": demo._sha256_file(exp1_ultra),
            },
            "trialqa_config": {
                "path": str(config),
                "sha256": demo._sha256_file(config),
            },
            "aggregate_script": {
                "path": str(aggregate),
                "sha256": demo._sha256_file(aggregate),
            },
        },
        "interpretation": (
            "The cloned reference repo contains a completed Super TrialQA result "
            "and an Ultra TrialQA workflow stub with placeholders. Therefore the "
            "Switchyard Ultra run may reproduce the method and test transfer, "
            "but must not claim a pre-existing published Ultra TrialQA score."
        ),
    }


def _status(requirements: Sequence[Mapping[str, object]], *, key: str) -> str:
    scoped = [item for item in requirements if item.get(key) is True]
    if any(item.get("status") == "failed" for item in scoped):
        return "failed"
    if any(item.get("status") == "missing" for item in scoped):
        return "missing"
    return "proved"


def build_reference_alignment(config: ReferenceAlignmentConfig) -> JsonObject:
    """Build a read-only comparison between current manifest and reference targets."""

    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    _require_schema(manifest, "switchyard.trialqa_experiment_manifest.v1", "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    reference = demo._read_json_object(config.reference_targets, "reference targets")
    _require_schema(reference, "switchyard.trialqa_reference_targets.v1", "reference targets")
    reference_population = _reference_population(reference)
    workflow_evidence = (
        _skills_distillation_workflow_evidence(config.skills_distillation_repo)
        if config.skills_distillation_repo is not None
        else None
    )
    requirements = _requirements(
        manifest=manifest,
        reference_population=reference_population,
        workflow_evidence=workflow_evidence,
    )
    canary_status = _status(requirements, key="required_for_canary")
    official_status = _status(requirements, key="required_for_official_reproduction")
    primary = _primary_scope(manifest)
    reference_trials = _require_int(reference_population.get("trials"), "reference trials")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id"),
        "canary_alignment_status": canary_status,
        "official_reproduction_status": official_status,
        "claim_scope": (
            "prospective_transfer_canary"
            if canary_status == "proved" and official_status != "proved"
            else "official_labbench2_reproduction"
            if official_status == "proved"
            else "not_aligned"
        ),
        "current_scope": {
            "questions": primary.get("question_count"),
            "repeats_per_question": primary.get("repeat_count"),
            "paired_tasks": primary.get("task_count"),
        },
        "reference_scope": {
            "questions": reference_population.get("heldout_questions"),
            "repeats_per_question": reference_population.get("repeats_per_question"),
            "unpaired_trials": reference_population.get("trials"),
            "paired_tasks": reference_trials * 2,
        },
        "requirements": requirements,
        "reference_workflow_evidence": workflow_evidence,
        "scope_note": (
            "The current Switchyard canary is aligned for fast paired ON/OFF validation, "
            "but it is not an official LABBench2 TrialQA reproduction unless "
            "official_96_question_reproduction_bound is proved."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument(
        "--skills-distillation-repo",
        type=Path,
        default=Path("skills-distillation"),
        help=(
            "Local clone of Sergei's skills-distillation repo used to bind the "
            "reference workflow and avoid overclaiming Ultra TrialQA results."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> ReferenceAlignmentConfig:
    return ReferenceAlignmentConfig(
        manifest=args.manifest,
        reference_targets=args.reference_targets,
        skills_distillation_repo=args.skills_distillation_repo,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_reference_alignment(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("canary_alignment_status") == "proved" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
