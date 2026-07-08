# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-backed completion audit for the local TrialQA validation ladder."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_readiness as readiness_module  # noqa: E402
from benchmark.trialqa_local_runner import EXECUTOR_MODEL, EXECUTOR_ROUTE  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_protocol_audit.v1"
RequirementStatus = Literal["proved", "missing", "failed"]
JsonObject = dict[str, Any]


class TrialQAProtocolAuditError(RuntimeError):
    """Protocol audit inputs are stale, malformed, or inconsistent."""


def _require_schema(report: Mapping[str, object], schema: str, label: str) -> None:
    if report.get("schema_version") != schema:
        raise TrialQAProtocolAuditError(f"{label} has invalid schema_version")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQAProtocolAuditError(f"{label} must be an object")
    return value


def _requirement(
    requirement_id: str,
    status: RequirementStatus,
    evidence: str,
    *,
    required_for_spend: bool = True,
) -> JsonObject:
    return {
        "id": requirement_id,
        "status": status,
        "evidence": evidence,
        "required_for_spend": required_for_spend,
    }


def _task_pair_summary(manifest: Mapping[str, object]) -> JsonObject:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise TrialQAProtocolAuditError("manifest tasks must be a list")
    pair_conditions: dict[str, set[str]] = {}
    conditions: set[str] = set()
    for item in tasks:
        task = _require_mapping(item, "manifest task")
        pair_id = task.get("pair_id")
        condition = task.get("condition")
        if not isinstance(pair_id, str) or not isinstance(condition, str):
            raise TrialQAProtocolAuditError("manifest task lacks pair_id or condition")
        pair_conditions.setdefault(pair_id, set()).add(condition)
        conditions.add(condition)
    complete_pair_count = sum(
        values == {"baseline", "treatment"} for values in pair_conditions.values()
    )
    return {
        "task_count": len(tasks),
        "condition_values": sorted(conditions),
        "pair_count": len(pair_conditions),
        "complete_baseline_treatment_pair_count": complete_pair_count,
        "all_pairs_complete": complete_pair_count == len(pair_conditions),
    }


def _command_option(command: Sequence[object], flag: str) -> str | None:
    for index, item in enumerate(command):
        if item != flag:
            continue
        if index + 1 >= len(command) or not isinstance(command[index + 1], str):
            return None
        return cast(str, command[index + 1])
    return None


def _same_path(left: str | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return Path(left).expanduser().resolve() == right.expanduser().resolve()


def _generation_dry_run_requirement(
    path: Path | None,
    *,
    manifest_path: Path,
) -> tuple[JsonObject, JsonObject | None]:
    if path is None:
        return (
            _requirement(
                "guarded_generation_dry_run_persisted",
                "missing",
                "no generation canary summary artifact was provided",
            ),
            None,
        )
    summary = demo._read_json_object(path, "generation canary summary")
    _require_schema(summary, "switchyard.trialqa_canary_driver.v1", "generation canary summary")
    checks = {
        "status": summary.get("status") == "awaiting_spend_authorization",
        "spend_authorized": summary.get("spend_authorized") is False,
        "readiness_status": summary.get("readiness_status")
        in readiness_module.GENERATION_READY_STATUSES,
        "generation_command": isinstance(summary.get("generation_command"), list)
        and "--stage" in cast(list[object], summary["generation_command"])
        and "generation" in cast(list[object], summary["generation_command"]),
        "operational_gate_command": isinstance(summary.get("operational_gate_command"), list)
        and "benchmark.trialqa_local_gate"
        in {str(item) for item in cast(list[object], summary["operational_gate_command"])},
        "authorized_rerun_command": isinstance(summary.get("authorized_rerun_command"), list)
        and "benchmark.trialqa_local_canary"
        in {str(item) for item in cast(list[object], summary["authorized_rerun_command"])}
        and cast(list[object], summary["authorized_rerun_command"])[-1] == "--yes-spend",
    }
    command = summary.get("authorized_rerun_command")
    if isinstance(command, list):
        checks["authorized_rerun_manifest"] = _same_path(
            _command_option(command, "--manifest"),
            manifest_path,
        )
        checks["authorized_rerun_summary_output"] = _same_path(
            _command_option(command, "--summary-output"),
            path,
        )
    if all(checks.values()):
        return (
            _requirement(
                "guarded_generation_dry_run_persisted",
                "proved",
                f"{path} records awaiting_spend_authorization with exact generation/gate commands",
            ),
            summary,
        )
    failed = ", ".join(key for key, ok in checks.items() if not ok)
    return (
        _requirement(
            "guarded_generation_dry_run_persisted",
            "failed",
            f"{path} failed dry-run checks: {failed}",
        ),
        summary,
    )


def _score_dry_run_requirement(
    path: Path | None,
    *,
    manifest_path: Path,
    operational_gate_path: Path | None,
) -> tuple[JsonObject, JsonObject | None]:
    if path is None:
        return (
            _requirement(
                "guarded_score_dry_run_persisted",
                "missing",
                "no score canary summary artifact was provided",
            ),
            None,
        )
    summary = demo._read_json_object(path, "score canary summary")
    _require_schema(
        summary,
        "switchyard.trialqa_canary_score_driver.v1",
        "score canary summary",
    )
    checks = {
        "status": summary.get("status") == "awaiting_spend_authorization",
        "spend_authorized": summary.get("spend_authorized") is False,
        "operational_decision": summary.get("operational_decision") == "promote_to_score",
        "score_command": isinstance(summary.get("score_command"), list)
        and "--stage" in cast(list[object], summary["score_command"])
        and "score" in cast(list[object], summary["score_command"]),
        "promotion_gate_command": isinstance(summary.get("promotion_gate_command"), list)
        and "benchmark.trialqa_local_gate"
        in {str(item) for item in cast(list[object], summary["promotion_gate_command"])}
        and "promotion" in cast(list[object], summary["promotion_gate_command"]),
        "authorized_rerun_command": isinstance(summary.get("authorized_rerun_command"), list)
        and "benchmark.trialqa_local_canary_score"
        in {str(item) for item in cast(list[object], summary["authorized_rerun_command"])}
        and cast(list[object], summary["authorized_rerun_command"])[-1] == "--yes-spend",
    }
    command = summary.get("authorized_rerun_command")
    if isinstance(command, list):
        checks["authorized_rerun_manifest"] = _same_path(
            _command_option(command, "--manifest"),
            manifest_path,
        )
        checks["authorized_rerun_summary_output"] = _same_path(
            _command_option(command, "--summary-output"),
            path,
        )
        if operational_gate_path is not None:
            checks["authorized_rerun_operational_gate"] = _same_path(
                _command_option(command, "--operational-gate"),
                operational_gate_path,
            )
        else:
            checks["authorized_rerun_operational_gate"] = (
                _command_option(command, "--operational-gate") is not None
            )
    if all(checks.values()):
        return (
            _requirement(
                "guarded_score_dry_run_persisted",
                "proved",
                f"{path} records awaiting_spend_authorization with exact score/gate commands",
            ),
            summary,
        )
    failed = ", ".join(key for key, ok in checks.items() if not ok)
    return (
        _requirement(
            "guarded_score_dry_run_persisted",
            "failed",
            f"{path} failed dry-run checks: {failed}",
        ),
        summary,
    )


def _live_evidence_requirements(status_report: Mapping[str, object]) -> list[JsonObject]:
    operational = status_report.get("operational_gate")
    promotion = status_report.get("promotion_gate")
    requirements: list[JsonObject] = []
    if operational is None:
        requirements.append(
            _requirement(
                "operational_generation_gate_completed",
                "missing",
                "no operational gate report exists; live generation has not been bought yet",
                required_for_spend=False,
            )
        )
    else:
        operational_report = _require_mapping(operational, "operational gate summary")
        decision = operational_report.get("decision")
        requirements.append(
            _requirement(
                "operational_generation_gate_completed",
                "proved" if decision == "promote_to_score" else "failed",
                f"operational gate decision is {decision!r}",
                required_for_spend=False,
            )
        )
    if promotion is None:
        requirements.extend(
            [
                _requirement(
                    "quality_parity_evidence",
                    "missing",
                    "no promotion gate report exists; judged quality parity is unproven",
                    required_for_spend=False,
                ),
                _requirement(
                    "efficiency_benefit_evidence",
                    "missing",
                    "no promotion gate report exists; scored efficiency benefit is unproven",
                    required_for_spend=False,
                ),
            ]
        )
    else:
        promotion_report = _require_mapping(promotion, "promotion gate summary")
        decision = promotion_report.get("decision")
        final_scope_proved = (
            decision == "promote_to_next_cohort"
            and promotion_report.get("confirmatory_scope_complete") is True
            and promotion_report.get("performance_eligible") is True
        )
        if decision == "promote_to_next_cohort" and not final_scope_proved:
            evidence_status: RequirementStatus = "missing"
            evidence = (
                "promotion gate is an interim promotion; final primary-scope "
                f"quality and efficiency are not proved: {dict(promotion_report)}"
            )
        else:
            evidence_status = "proved" if final_scope_proved else "failed"
            evidence = f"promotion gate decision is {decision!r}"
        requirements.extend(
            [
                _requirement(
                    "quality_parity_evidence",
                    evidence_status,
                    evidence,
                    required_for_spend=False,
                ),
                _requirement(
                    "efficiency_benefit_evidence",
                    evidence_status,
                    evidence,
                    required_for_spend=False,
                ),
            ]
        )
    return requirements


def _readiness_requirement(readiness: Mapping[str, object]) -> JsonObject:
    states = readiness.get("selected_task_state_values")
    state_values = {str(value) for value in states} if isinstance(states, list) else set()
    task_count = readiness.get("task_count")
    pair_count = readiness.get("pair_count")
    status_ok = readiness.get("status") in readiness_module.GENERATION_READY_STATUSES
    state_ok = state_values in ({"not_started"}, {"completed"}, {"completed", "not_started"})
    pairing_ok = isinstance(task_count, int) and isinstance(pair_count, int) and task_count == pair_count * 2
    return _requirement(
        "generation_scope_readiness_clean",
        "proved" if status_ok and state_ok and pairing_ok else "failed",
        f"readiness summary is {readiness}",
    )


def _comparison_invariant_requirement(
    readiness: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> JsonObject:
    invariant = readiness.get("comparison_invariant")
    if not isinstance(invariant, Mapping):
        return _requirement(
            "skill_distillation_ab_invariant_bound",
            "failed",
            "readiness summary lacks comparison_invariant",
        )
    routing = _require_mapping(manifest.get("routing"), "manifest routing")
    candidate = _require_mapping(manifest.get("candidate"), "manifest candidate")
    shared = _require_mapping(
        invariant.get("shared_executor"),
        "readiness comparison_invariant.shared_executor",
    )
    baseline = _require_mapping(
        invariant.get("baseline_arm"),
        "readiness comparison_invariant.baseline_arm",
    )
    treatment = _require_mapping(
        invariant.get("treatment_arm"),
        "readiness comparison_invariant.treatment_arm",
    )
    runtime = invariant.get("runtime_enforcement")
    treatment_skill = treatment.get("candidate_skill_sha256")
    ok = (
        invariant.get("status") == "proved"
        and invariant.get("design") == "concurrent-paired-same-executor-skill-only"
        and invariant.get("control_design") == "concurrent-paired"
        and invariant.get("conditions") == ["baseline", "treatment"]
        and shared.get("route") == EXECUTOR_ROUTE
        and shared.get("route") == routing.get("executor_route")
        and shared.get("model") == EXECUTOR_MODEL
        and shared.get("model") == routing.get("executor_model")
        and shared.get("routing_profile_sha256") == routing.get("profile_sha256")
        and baseline.get("condition") == "baseline"
        and baseline.get("skill_loaded") is False
        and baseline.get("candidate_id") is None
        and baseline.get("candidate_manifest_sha256") is None
        and baseline.get("candidate_skill_sha256") is None
        and treatment.get("condition") == "treatment"
        and treatment.get("skill_loaded") is True
        and treatment.get("candidate_id") == candidate.get("candidate_id")
        and treatment.get("candidate_manifest_sha256") == candidate.get("manifest_sha256")
        and treatment_skill == candidate.get("skill_sha256")
        and isinstance(treatment_skill, str)
        and treatment_skill.startswith("sha256:")
        and isinstance(runtime, list)
        and "session proof binds active-skill evidence per turn" in runtime
        and "session proof requires Ultra-only served models" in runtime
    )
    return _requirement(
        "skill_distillation_ab_invariant_bound",
        "proved" if ok else "failed",
        f"comparison invariant is {dict(invariant)}",
    )


def _local_trialqa_transfer_runtime_requirement(
    *,
    dataset: Mapping[str, object],
    protocol: Mapping[str, object],
) -> JsonObject:
    """Prove the current fast path is the local Switchyard transfer workflow.

    The reference material is allowed to inform targets and population audits, but
    the spend boundary should not silently drift into a Docker/HF-repo reproduction
    path. The live canary is a local Switchyard run over the generated
    ClinicalTrials.gov TrialQA-compatible parquet.
    """

    primary_scope = _require_mapping(
        protocol.get("primary_evaluation_scope"),
        "manifest primary_evaluation_scope",
    )
    checks = {
        "official_labbench2_false": dataset.get("official_labbench2") is False,
        "prospective_dataset_id": dataset.get("id") == "trialqa-compatible-prospective",
        "clinicaltrials_config": dataset.get("config") == "clinicaltrials-gov",
        "prospective_split": dataset.get("split") == "prospective",
        "prospective_revision": isinstance(dataset.get("revision"), str)
        and str(dataset["revision"]).startswith("clinicaltrials-gov-prospective"),
        "row_count_matches_primary_questions": dataset.get("row_count")
        == primary_scope.get("question_count"),
        "local_batch_driver": protocol.get("batch_driver")
        == "benchmark/trialqa_local_batch.py",
        "prospective_population_kind": protocol.get("prospective_population_kind")
        == "trialqa-compatible-clinicaltrials-gov",
        "no_gold_in_manifest": protocol.get("gold_in_manifest") is False,
        "performance_eligible": protocol.get("performance_eligible") is True,
    }
    status: RequirementStatus = "proved" if all(checks.values()) else "failed"
    failed = [key for key, ok in checks.items() if not ok]
    evidence = (
        "manifest binds container-free local Switchyard transfer workflow "
        f"with dataset={dict(dataset)} and protocol_subset="
        f"{ {key: protocol.get(key) for key in ('batch_driver', 'prospective_population_kind', 'gold_in_manifest', 'performance_eligible', 'max_generation_concurrency')} }"
    )
    if failed:
        evidence += f"; failed checks: {', '.join(failed)}"
    return _requirement(
        "local_switchyard_trialqa_transfer_runtime_bound",
        status,
        evidence,
    )


def _completion_state(requirements: Sequence[Mapping[str, object]], next_action: Mapping[str, object]) -> str:
    spend_required = [item for item in requirements if item.get("required_for_spend") is True]
    if any(item.get("status") == "failed" for item in spend_required):
        return "not_ready_for_spend"
    if any(item.get("status") == "missing" for item in spend_required):
        return "missing_no_spend_evidence"
    action = next_action.get("action")
    if action == "kill_candidate":
        return "candidate_killed"
    if action == "expand_generation_scope":
        return "awaiting_generation_canary_spend_authorization"
    if next_action.get("requires_yes_spend") is True:
        return f"awaiting_{str(action).replace('run_guarded_', '')}_spend_authorization"
    if action == "prospective_directional_scope_complete":
        return "prospective_directional_scope_complete"
    return "incomplete"


def _next_command(
    *,
    completion_state: str,
    generation_dry_run_summary: Mapping[str, object] | None,
    generation_canary_summary_path: Path | None,
    score_dry_run_summary: Mapping[str, object] | None,
    score_canary_summary_path: Path | None,
) -> JsonObject | None:
    if completion_state == "awaiting_generation_canary_spend_authorization":
        dry_run_summary = generation_dry_run_summary
        source = generation_canary_summary_path
        kind = "guarded_generation_canary"
    elif completion_state == "awaiting_score_canary_spend_authorization":
        dry_run_summary = score_dry_run_summary
        source = score_canary_summary_path
        kind = "guarded_score_canary"
    else:
        return None
    if dry_run_summary is None:
        return None
    command = dry_run_summary.get("authorized_rerun_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    return {
        "kind": kind,
        "command": command,
        "shell_command": shlex.join(command),
        "source": str(source) if source is not None else None,
        "requires_yes_spend": bool(command) and command[-1] == "--yes-spend",
        "authorized_by_audit": False,
    }


def build_protocol_audit(
    *,
    manifest_path: Path,
    status_path: Path,
    generation_canary_summary_path: Path | None = None,
    score_canary_summary_path: Path | None = None,
    operational_gate_path: Path | None = None,
) -> JsonObject:
    """Build an audit report that distinguishes proven setup from missing live evidence."""

    manifest = demo._read_json_object(manifest_path, "experiment manifest")
    _require_schema(manifest, "switchyard.trialqa_experiment_manifest.v1", "experiment manifest")
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str):
        raise TrialQAProtocolAuditError("manifest_id must be a string")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    primary_scope = _require_mapping(
        protocol.get("primary_evaluation_scope"), "manifest primary_evaluation_scope"
    )
    pairs = _task_pair_summary(manifest)

    status_report = demo._read_json_object(status_path, "protocol status report")
    _require_schema(
        status_report, "switchyard.trialqa_protocol_status.v1", "protocol status report"
    )
    status_manifest = _require_mapping(status_report.get("manifest"), "status manifest")
    if status_manifest.get("manifest_id") != manifest_id:
        raise TrialQAProtocolAuditError("status report belongs to a different manifest")
    readiness = _require_mapping(status_report.get("readiness"), "status readiness")
    next_action = _require_mapping(status_report.get("next_action"), "status next_action")
    reference = _require_mapping(status_report.get("reference_targets"), "status reference targets")
    generation_dry_run_summary: JsonObject | None = None
    score_dry_run_summary: JsonObject | None = None
    action = next_action.get("action")
    dry_run_requirement: JsonObject | None = None
    if action in {"run_guarded_generation_canary", "expand_generation_scope"}:
        dry_run_requirement, generation_dry_run_summary = _generation_dry_run_requirement(
            generation_canary_summary_path,
            manifest_path=manifest_path,
        )
    elif action == "run_guarded_score_canary":
        dry_run_requirement, score_dry_run_summary = _score_dry_run_requirement(
            score_canary_summary_path,
            manifest_path=manifest_path,
            operational_gate_path=operational_gate_path,
        )

    official = dataset.get("official_labbench2")
    requirements = [
        _requirement(
            "reference_targets_bound",
            "proved"
            if reference.get("trials") == 480
            and reference.get("heldout_questions") == 96
            and reference.get("repeats_per_question") == 5
            else "failed",
            f"status binds reference targets {reference}",
        ),
        _requirement(
            "prospective_manifest_bound",
            "proved"
            if official is False
            and primary_scope.get("question_count") == 8
            and primary_scope.get("repeat_count") == 5
            and primary_scope.get("task_count") == 80
            else "failed",
            "manifest declares non-official prospective 8-question x 5-repeat x 2-arm scope",
        ),
        _requirement(
            "paired_skill_off_on_scope",
            "proved"
            if pairs["condition_values"] == ["baseline", "treatment"] and pairs["all_pairs_complete"]
            else "failed",
            f"manifest task pairing summary is {pairs}",
        ),
        _readiness_requirement(readiness),
        _comparison_invariant_requirement(readiness, manifest=manifest),
        _local_trialqa_transfer_runtime_requirement(dataset=dataset, protocol=protocol),
        *_live_evidence_requirements(status_report),
    ]
    if dry_run_requirement is not None:
        requirements.insert(5, dry_run_requirement)
    completion_state = _completion_state(requirements, next_action)
    next_command = _next_command(
        completion_state=completion_state,
        generation_dry_run_summary=generation_dry_run_summary,
        generation_canary_summary_path=generation_canary_summary_path,
        score_dry_run_summary=score_dry_run_summary,
        score_canary_summary_path=score_canary_summary_path,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "completion_state": completion_state,
        "next_action": dict(next_action),
        "next_command": next_command,
        "spend_boundary": {
            "requires_yes_spend": next_action.get("requires_yes_spend") is True,
            "authorized_by_audit": False,
            "reason": "this audit is read-only and never authorizes model spend",
        },
        "requirements": requirements,
        "dry_run_summary": {
            "path": str(generation_canary_summary_path)
            if action == "run_guarded_generation_canary"
            else str(score_canary_summary_path)
            if action == "run_guarded_score_canary"
            and score_canary_summary_path is not None
            else None,
            "status": generation_dry_run_summary.get("status")
            if generation_dry_run_summary is not None
            else score_dry_run_summary.get("status")
            if score_dry_run_summary is not None
            else None,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--generation-canary-summary", type=Path)
    parser.add_argument("--score-canary-summary", type=Path)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_protocol_audit(
        manifest_path=args.manifest,
        status_path=args.status,
        generation_canary_summary_path=args.generation_canary_summary,
        score_canary_summary_path=args.score_canary_summary,
        operational_gate_path=args.operational_gate,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
