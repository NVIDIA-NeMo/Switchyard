# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-spend readiness check for the next local TrialQA canary checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the CLI itself.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
from benchmark.trialqa_local_runner import NAMESPACE, validate_candidate_skill  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_canary_readiness.v1"
JsonObject = dict[str, Any]
GENERATION_READY_STATUSES = frozenset(
    {
        "ready_for_generation",
        "ready_for_generation_expansion",
    }
)


class TrialQAReadinessError(RuntimeError):
    """Readiness inputs are stale, inconsistent, or not safe to spend against."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _task_conditions_by_pair(tasks: Sequence[Mapping[str, object]]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for task in tasks:
        pair_id = task.get("pair_id")
        condition = task.get("condition")
        if not isinstance(pair_id, str) or not pair_id:
            raise TrialQAReadinessError("selected task has invalid pair_id")
        if condition not in {"baseline", "treatment"}:
            raise TrialQAReadinessError("selected task has invalid condition")
        pairs.setdefault(pair_id, set()).add(condition)
    return pairs


def _selected_task_states(
    *,
    capture: Path,
    manifest: Mapping[str, object],
    scoped_tasks: Sequence[Mapping[str, object]],
) -> tuple[bool, dict[str, str]]:
    ledger_path = capture / "ledger.jsonl"
    if not ledger_path.exists():
        return False, {cast(str, task["task_id"]): "not_started" for task in scoped_tasks}
    ledger = demo.ResumableLedger(ledger_path, manifest)
    states = ledger.states()
    return True, {
        cast(str, task["task_id"]): states.get(cast(str, task["task_id"]), "not_started")
        for task in scoped_tasks
    }


def _generation_readiness_status(selected_states: Mapping[str, str]) -> str:
    values = set(selected_states.values())
    if values == {"not_started"}:
        return "ready_for_generation"
    if values == {"completed", "not_started"}:
        return "ready_for_generation_expansion"
    if values == {"completed"}:
        return "selected_scope_already_completed"
    return "has_nonterminal_selected_task_state"


def build_readiness_report(
    *,
    manifest_path: Path,
    dataset_path: Path,
    experiment_root: Path,
    doctor_report: Path,
    population_report: Path,
    candidate_root: Path,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
    routing_profile: Path,
    question_start: int,
    question_limit: int,
    repeat_limit: int,
) -> JsonObject:
    """Validate exact canary inputs and return a zero-spend readiness report."""

    manifest_path = manifest_path.absolute()
    dataset_path = dataset_path.absolute()
    experiment_root = experiment_root.absolute()
    doctor_report = doctor_report.absolute()
    population_report = population_report.absolute()
    candidate_root = candidate_root.absolute()
    switchyard_bin = switchyard_bin.absolute()
    tooluniverse_bin = tooluniverse_bin.absolute()
    routing_profile = routing_profile.absolute()

    manifest = demo._read_json_object(manifest_path, "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    dataset = demo.load_manifest_dataset(dataset_path, manifest)
    split = demo.create_manifest_split(dataset, manifest)
    candidate = validate_candidate_skill(candidate_root, NAMESPACE)
    expected = demo.build_reproducible_manifest_from_supplied(
        supplied=manifest,
        dataset=dataset,
        split_manifest=split,
        candidate=candidate,
        routing_profile=routing_profile,
        switchyard_bin=switchyard_bin,
        codex_bin=codex_bin,
        tooluniverse_bin=tooluniverse_bin,
        doctor_report=doctor_report,
        population_report=population_report,
    )
    if expected != manifest:
        raise TrialQAReadinessError("manifest does not reproduce from current readiness inputs")

    tasks = [dict(item) for item in cast(list[dict[str, object]], manifest["tasks"])]
    scope = batch._build_manifest_task_scope(
        manifest,
        tasks,
        limit=None,
        question_start=question_start,
        question_limit=question_limit,
        repeat_limit=repeat_limit,
        condition="both",
    )
    scoped_tasks = list(scope.tasks)
    pairs = _task_conditions_by_pair(scoped_tasks)
    if not all(conditions == {"baseline", "treatment"} for conditions in pairs.values()):
        raise TrialQAReadinessError("selected canary scope is not exactly paired")

    capture = experiment_root / cast(str, manifest["manifest_id"])
    ledger_exists, selected_states = _selected_task_states(
        capture=capture,
        manifest=manifest,
        scoped_tasks=scoped_tasks,
    )
    status = _generation_readiness_status(selected_states)
    selected_state_counts = {
        state: sum(value == state for value in selected_states.values())
        for state in sorted(set(selected_states.values()))
    }
    population_sha256 = _sha256_file(population_report)
    doctor_sha256 = _sha256_file(doctor_report)
    candidate_document = cast(Mapping[str, object], manifest["candidate"])
    manifest_dataset = cast(Mapping[str, object], manifest["dataset"])
    preflight = cast(Mapping[str, object], manifest["preflight"])
    if population_sha256 != manifest_dataset.get("population_report_sha256"):
        raise TrialQAReadinessError("population report hash differs from manifest")
    if doctor_sha256 != preflight.get("doctor_report_sha256"):
        raise TrialQAReadinessError("doctor report hash differs from manifest")
    if candidate.sha256 != candidate_document.get("skill_sha256"):
        raise TrialQAReadinessError("candidate skill hash differs from manifest")
    routing = cast(Mapping[str, object], manifest["routing"])
    protocol = cast(Mapping[str, object], manifest["protocol"])
    comparison_invariant = {
        "status": "proved",
        "design": "concurrent-paired-same-executor-skill-only",
        "control_design": protocol.get("control_design"),
        "conditions": list(cast(Sequence[str], protocol.get("conditions"))),
        "shared_executor": {
            "route": routing.get("executor_route"),
            "model": routing.get("executor_model"),
            "routing_profile_sha256": routing.get("profile_sha256"),
        },
        "baseline_arm": {
            "condition": "baseline",
            "skill_loaded": False,
            "candidate_id": None,
            "candidate_manifest_sha256": None,
            "candidate_skill_sha256": None,
        },
        "treatment_arm": {
            "condition": "treatment",
            "skill_loaded": True,
            "candidate_id": candidate_document.get("candidate_id"),
            "candidate_manifest_sha256": candidate_document.get("manifest_sha256"),
            "candidate_skill_sha256": candidate_document.get("skill_sha256"),
        },
        "runtime_enforcement": [
            "manifest requires exact baseline/treatment pairs",
            "generation launch context binds executor route/model per task",
            "session proof requires Ultra-only served models",
            "session proof binds active-skill evidence per turn",
            "baseline workspace must expose no project skill",
            "treatment workspace must expose only the candidate skill",
        ],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "spend_status": "zero_model_calls_in_this_readiness_check",
        "manifest": {
            "path": str(manifest_path),
            "manifest_id": manifest["manifest_id"],
            "kind": manifest["kind"],
            "task_count": len(tasks),
            "official_labbench2": manifest_dataset.get("official_labbench2"),
            "performance_eligible": cast(Mapping[str, object], manifest["protocol"]).get(
                "performance_eligible"
            ),
            "max_generation_concurrency": cast(
                Mapping[str, object], manifest["protocol"]
            ).get("max_generation_concurrency"),
        },
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(dataset.rows),
            "parquet_sha256": dataset.parquet_sha256,
            "population_report_sha256": population_sha256,
            "manifest_population_report_sha256": manifest_dataset.get(
                "population_report_sha256"
            ),
        },
        "doctor": {
            "path": str(doctor_report),
            "sha256": doctor_sha256,
            "manifest_doctor_report_sha256": preflight.get("doctor_report_sha256"),
        },
        "candidate": {
            "root": str(candidate_root),
            "candidate_id": candidate_document.get("candidate_id"),
            "skill_sha256": candidate.sha256,
            "manifest_skill_sha256": candidate_document.get("skill_sha256"),
        },
        "routing": dict(routing),
        "comparison_invariant": comparison_invariant,
        "first_generation_canary": {
            "question_start": question_start,
            "question_limit": question_limit,
            "repeat_limit": repeat_limit,
            "task_count": len(scoped_tasks),
            "pair_count": len(pairs),
            "selected_question_groups": list(scope.selected_question_groups),
            "selected_repeat_indices": list(scope.selected_repeat_indices),
            "task_ids": [cast(str, task["task_id"]) for task in scoped_tasks],
            "ledger_exists": ledger_exists,
            "selected_task_states": selected_states,
            "selected_task_state_counts": selected_state_counts,
            "scope_attestation": scope.metadata(manifest["manifest_id"]),
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
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_readiness_report(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        experiment_root=args.experiment_root,
        doctor_report=args.doctor,
        population_report=args.population_report,
        candidate_root=args.candidate,
        switchyard_bin=args.switchyard,
        codex_bin=args.codex,
        tooluniverse_bin=args.tooluniverse,
        routing_profile=args.profile,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
