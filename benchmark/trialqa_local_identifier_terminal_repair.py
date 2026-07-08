# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one deterministic zero-call TrialQA identifier-terminal repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_candidate_repair as base  # noqa: E402
import benchmark.trialqa_local_dataset as dataset_module  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_distiller as distiller  # noqa: E402
import benchmark.trialqa_local_regression as regression  # noqa: E402
import benchmark.trialqa_local_search_gate as search_gate  # noqa: E402
import benchmark.trialqa_local_search_repair as prior  # noqa: E402
import switchyard.lib.skill_distillation_store as store_module  # noqa: E402
from benchmark.trialqa_local_dataset import TrialQADataset  # noqa: E402
from switchyard.lib.skill_distillation_store import SkillDistillationStore  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_candidate_identifier_terminal_repair.v1"
ZERO_CALLS = {"model": 0, "judge": 0, "evidence_import": 0, "network": 0}
EXPECTED_PARENT_EVIDENCE_COUNT = 122
EXPECTED_LOGICAL_REQUESTS = 10
EXPECTED_TOTAL_TOKENS = 145_057
EXPECTED_OPERATIONAL_CALLS = 8
EXPECTED_SEARCH_KILL_REASONS = [
    "at_most_three_searches",
    "no_search_after_first_resolution",
    "next_call_is_expected_evidence_getter",
]

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class IdentifierTerminalPlan:
    run_id: str
    run_path: Path
    parent: base.ParentCandidate
    manifest: JsonObject
    catalog: JsonObject
    skill: str
    candidate_id: str
    validation: JsonObject
    dataset_path: Path
    descriptive_manifest_path: Path
    primary_manifest_path: Path
    capture_path: Path
    semantic_report_path: Path
    search_gate_report_path: Path


@dataclass(frozen=True)
class IdentifierTerminalResult:
    run_id: str
    candidate_id: str
    candidate_path: Path
    report_path: Path
    model_call_count: int = 0


def _error(message: str) -> distiller.TrialQADistillationError:
    return distiller.TrialQADistillationError(message)


def _load_parent(
    *, project_dir: Path, store_dir: Path, candidate_id: str, work_dir: Path
) -> base.ParentCandidate:
    return base._load_parent(
        project_dir=project_dir,
        store_dir=store_dir,
        candidate_id=candidate_id,
        work_dir=work_dir,
        expected_mode=distiller.SEARCH_DISCIPLINE_REPAIR_MODE,
        expected_stage="search_discipline_repair",
        expected_new_calls=ZERO_CALLS,
        expected_catalog_mode_field="search_discipline_repair_mode",
        expected_evidence_count=EXPECTED_PARENT_EVIDENCE_COUNT,
    )


def _validate_report_identity(
    *,
    report: Mapping[str, Any],
    descriptive: Mapping[str, Any],
    descriptive_binding: Mapping[str, Any],
    dataset: TrialQADataset,
    label: str,
) -> None:
    prior._validate_report_identity(
        report=report,
        descriptive=descriptive,
        descriptive_binding=descriptive_binding,
        dataset=dataset,
        label=label,
    )


def _load_search_gate_report(
    *,
    path: Path,
    semantic_report: Mapping[str, Any],
    semantic_result: Mapping[str, Any],
    descriptive: Mapping[str, Any],
    descriptive_binding: Mapping[str, Any],
    dataset: TrialQADataset,
    capture: Path,
    task: Mapping[str, Any],
) -> tuple[Path, JsonObject, JsonObject]:
    report_path = distiller._real_file(path, "v10 q2 search gate report")
    report = distiller._read_json_file(report_path, "v10 q2 search gate report")
    supplied = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    policy = distiller._mapping(report.get("policy"), "v10 q2 search gate policy")
    expected_policy = {
        "name": search_gate.SEARCH_GATE_POLICY,
        "performance_eligible": False,
        "condition": "treatment",
        "max_searches": search_gate.MAX_SEARCHES,
        "model_calls": 0,
        "judge_calls": 0,
        "evidence_imports": 0,
        "network_calls": 0,
    }
    if supplied != base._canonical_sha256(unsigned) or policy != expected_policy:
        raise _error("v10 q2 search gate hash or zero-call policy is invalid")
    _validate_report_identity(
        report=report,
        descriptive=descriptive,
        descriptive_binding=descriptive_binding,
        dataset=dataset,
        label="v10 q2 search gate report",
    )
    task_id = distiller._required_text(task.get("task_id"), "q2 task id")
    try:
        recomputed = search_gate.build_search_gate_report(
            manifest=descriptive,
            dataset=dataset,
            capture=capture,
            task_ids=[task_id],
        )
    except (
        search_gate.TrialQASearchGateError,
        regression.TrialQARegressionError,
        demo.TrialQADemoError,
        OSError,
    ) as exc:
        raise _error("v10 q2 search gate evidence failed independent replay") from exc
    results = [
        distiller._mapping(item, "v10 q2 search gate result")
        for item in distiller._list(report.get("results"), "v10 q2 search gate results")
    ]
    summary = distiller._mapping(report.get("summary"), "v10 q2 search gate summary")
    expected_checks = {
        "semantic_replay_passed": True,
        "search_arguments_valid": True,
        "at_most_three_searches": False,
        "unique_canonical_arguments": True,
        "unique_normalized_queries": True,
        "unique_title_resolution_found": True,
        "no_search_after_first_resolution": False,
        "next_call_is_expected_evidence_getter": False,
    }
    if (
        recomputed != report
        or report.get("schema_version") != search_gate.SEARCH_GATE_SCHEMA_VERSION
        or report.get("semantic_report_sha256") != semantic_report.get("report_sha256")
        or summary != {"checked_tasks": 1, "passed_tasks": 0, "killed_tasks": 1, "decision": "kill"}
        or len(results) != 1
        or results[0].get("task_id") != task_id
        or results[0].get("question_ordinal") != 2
        or results[0].get("decision") != "kill"
        or results[0].get("semantic_result") != dict(semantic_result)
        or results[0].get("checks") != expected_checks
        or results[0].get("kill_reasons") != EXPECTED_SEARCH_KILL_REASONS
        or results[0].get("search_count") != 7
        or results[0].get("successful_execute_tool_count") != 8
        or results[0].get("resolution_index") != 0
        or results[0].get("post_resolution_search_count") != 6
        or results[0].get("repeated_argument_count") != 0
        or results[0].get("repeated_normalized_query_count") != 0
        or results[0].get("next_operation") != search_gate.SEARCH_OPERATION
    ):
        raise _error("v10 q2 search gate is not the exact reviewed seven-search kill")
    return report_path, report, results[0]


def _assert_identifier_leaf_only(parent: Mapping[str, Any], repaired: Mapping[str, Any]) -> None:
    if repaired.get("identifier_terminal_repair_mode") != distiller.IDENTIFIER_TERMINAL_REPAIR_MODE:
        raise _error("identifier-terminal repair mode marker is missing")
    parent_without_tools = dict(parent)
    repaired_without_tools = dict(repaired)
    parent_tools = parent_without_tools.pop("tool_rules", None)
    repaired_tools = repaired_without_tools.pop("tool_rules", None)
    repaired_without_tools.pop("identifier_terminal_repair_mode", None)
    if parent_without_tools != repaired_without_tools:
        raise _error("identifier-terminal repair changed content outside the search leaf")
    if not isinstance(parent_tools, list) or not isinstance(repaired_tools, list):
        raise _error("identifier-terminal tool rules are invalid")
    if len(parent_tools) != len(repaired_tools):
        raise _error("identifier-terminal repair changed the tool group count")
    changed = 0
    for parent_group, repaired_group in zip(parent_tools, repaired_tools, strict=True):
        if not isinstance(parent_group, Mapping) or not isinstance(repaired_group, Mapping):
            raise _error("identifier-terminal tool group is invalid")
        if parent_group.get("tool_name") != "trialqa_search":
            if parent_group != repaired_group:
                raise _error("identifier-terminal repair changed a non-search tool leaf")
            continue
        parent_header = {key: value for key, value in parent_group.items() if key != "rules"}
        repaired_header = {key: value for key, value in repaired_group.items() if key != "rules"}
        if parent_header != repaired_header or parent_group.get("rules") == repaired_group.get(
            "rules"
        ):
            raise _error("identifier-terminal repair did not make one leaf-only change")
        changed += 1
    if changed != 1 or repaired.get("workflow_rules") != parent.get("workflow_rules"):
        raise _error("identifier-terminal repair must preserve every workflow rule")


def build_identifier_terminal_plan(
    *,
    parent_project_dir: Path,
    parent_store_dir: Path,
    parent_candidate_id: str,
    dataset_path: Path,
    descriptive_manifest: Path,
    primary_manifest: Path,
    capture_dir: Path,
    semantic_report: Path,
    search_gate_report: Path,
    work_dir: Path,
) -> IdentifierTerminalPlan:
    """Re-attest the v10 failure and build one read-only deterministic v11 plan."""

    work = work_dir.expanduser().absolute()
    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise _error("identifier-terminal work path must be a real directory or absent")
    parent = _load_parent(
        project_dir=parent_project_dir,
        store_dir=parent_store_dir,
        candidate_id=parent_candidate_id,
        work_dir=work,
    )
    dataset, dataset_binding = base._load_dataset(dataset_path)
    capture = distiller._real_directory(capture_dir, "v10 descriptive capture")
    descriptive, _primary, manifest_bindings, groups, tasks = base._load_manifests(
        descriptive_manifest, primary_manifest, parent=parent, capture=capture
    )
    q2_task = base._expected_task(tasks, groups[2], repeat=1)
    q2_task_id = distiller._required_text(q2_task.get("task_id"), "q2 task id")
    ledger_binding = base._validate_ledger_scope(
        descriptive=descriptive,
        capture=capture,
        expected_task_ids=[q2_task_id],
    )
    semantic_path, semantic, semantic_result = prior._load_semantic_report(
        path=semantic_report,
        descriptive=descriptive,
        descriptive_binding=cast(JsonObject, manifest_bindings["descriptive"]),
        dataset=dataset,
        capture=capture,
        task=q2_task,
    )
    gate_path, gate, _gate_result = _load_search_gate_report(
        path=search_gate_report,
        semantic_report=semantic,
        semantic_result=semantic_result,
        descriptive=descriptive,
        descriptive_binding=cast(JsonObject, manifest_bindings["descriptive"]),
        dataset=dataset,
        capture=capture,
        task=q2_task,
    )
    if semantic_path.parent != capture or gate_path.parent != capture:
        raise _error("v10 q2 reports must be inside the descriptive capture")
    generation_binding, sensitive_literals = prior._validate_generation(
        descriptive=descriptive,
        capture=capture,
        task=q2_task,
        semantic_result=semantic_result,
        parent=parent,
        expected_logical_requests=EXPECTED_LOGICAL_REQUESTS,
        expected_total_tokens=EXPECTED_TOTAL_TOKENS,
        expected_turn_count=EXPECTED_LOGICAL_REQUESTS,
        expected_operational_calls=EXPECTED_OPERATIONAL_CALLS,
    )

    catalog = distiller.layer_exposed_identifier_terminal_repair_catalog(parent.catalog)
    _assert_identifier_leaf_only(parent.catalog, catalog)
    skill = distiller.render_skill_markdown(catalog, tool_contract="compact")
    metrics = distiller.validate_compact_skill(catalog, skill, tool_contract="compact")
    distiller._assert_no_sensitive(skill, sensitive_literals, "identifier-terminal repair skill")

    input_bindings: JsonObject = {
        "parent_candidate": parent.binding,
        "parent_catalog": parent.catalog_binding,
        "dataset": dataset_binding,
        "manifests": manifest_bindings,
        "descriptive_ledger": ledger_binding,
        "q2_corrected_semantic_report": prior._report_binding(semantic_path, semantic),
        "q2_search_gate_report": prior._report_binding(gate_path, gate),
        "q2_generation": generation_binding,
    }
    seed: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "mode": distiller.IDENTIFIER_TERMINAL_REPAIR_MODE,
        "source_sha256": {
            "identifier_terminal_repair": (
                f"sha256:{distiller._file_sha256(Path(__file__).resolve())}"
            ),
            "search_repair": f"sha256:{distiller._file_sha256(Path(prior.__file__).resolve())}",
            "candidate_repair": (f"sha256:{distiller._file_sha256(Path(base.__file__).resolve())}"),
            "search_gate": (
                f"sha256:{distiller._file_sha256(Path(search_gate.__file__).resolve())}"
            ),
            "distiller": f"sha256:{distiller._file_sha256(Path(distiller.__file__).resolve())}",
            "regression": (f"sha256:{distiller._file_sha256(Path(regression.__file__).resolve())}"),
            "demo": f"sha256:{distiller._file_sha256(Path(demo.__file__).resolve())}",
            "dataset": (
                f"sha256:{distiller._file_sha256(Path(dataset_module.__file__).resolve())}"
            ),
            "batch": f"sha256:{distiller._file_sha256(Path(batch.__file__).resolve())}",
            "skill_distillation_store": (
                f"sha256:{distiller._file_sha256(Path(store_module.__file__).resolve())}"
            ),
        },
        "input_bindings": input_bindings,
        "call_budget": dict(ZERO_CALLS),
    }
    run_id = f"trialqa-identifier-terminal-{distiller._digest(seed)[:32]}"
    skill_sha = hashlib.sha256(skill.encode()).hexdigest()
    candidate_seed = {
        "run_id": run_id,
        "mode": distiller.IDENTIFIER_TERMINAL_REPAIR_MODE,
        "parent_candidate": parent.binding,
        "skill_path": distiller.SKILL_PATH,
        "skill_sha256": skill_sha,
        "source_evidence_ids": list(parent.evidence_ids),
    }
    candidate_id = f"trialqa-{distiller._digest(candidate_seed)[:32]}"
    checks = {
        "v10_parent_mode_catalog_and_hash_bound": True,
        "parent_122_evidence_ids_inherited_exactly": len(parent.evidence_ids) == 122,
        "descriptive_and_primary_manifest_pair_bound": True,
        "primary_88_capture_absent": True,
        "ledger_exactly_one_q2_started_and_completed": ledger_binding.get("record_count") == 2,
        "q2_semantic_exact_recompute_pass": True,
        "q2_search_gate_exact_recompute_kill": True,
        "q2_search_count_7_resolution_0_post_resolution_6_no_repeats": True,
        "q2_10_logical_10_physical_zero_retry": (
            generation_binding.get("logical_requests")
            == generation_binding.get("physical_attempts")
            == EXPECTED_LOGICAL_REQUESTS
        ),
        "q2_total_tokens_145057": (generation_binding.get("total_tokens") == EXPECTED_TOTAL_TOKENS),
        "only_search_leaf_changed": True,
        "all_workflow_rules_preserved_byte_for_byte": True,
        "generic_repair_has_no_task_literals": True,
        "zero_new_model_judge_import_network_calls": True,
        "candidate_remains_inactive": True,
        "compact_size": metrics["size_bytes"] <= distiller.COMPACT_SKILL_MAX_BYTES,
        "compact_words": metrics["word_count"] <= distiller.COMPACT_SKILL_MAX_WORDS,
        "compact_rules": metrics["rule_count"] <= distiller.COMPACT_SKILL_MAX_RULES,
    }
    if not all(checks.values()):
        raise _error("identifier-terminal validation contains a failed check")
    validation: JsonObject = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "train-base-plus-exposed-identifier-terminal-repair-primary88-only",
        "distillation_mode": distiller.IDENTIFIER_TERMINAL_REPAIR_MODE,
        "performance_validated": False,
        "performance_eligible": True,
        "full_96_performance_eligible": False,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent.candidate_id,
        "tool_contract": "compact",
        "source_evidence_ids": list(parent.evidence_ids),
        "new_calls": dict(ZERO_CALLS),
        "input_bindings": input_bindings,
        "checks": checks,
        "routing": {"attested_call_count": 0, "attestations": []},
        "artifacts": {"skill_sha256": f"sha256:{skill_sha}", **metrics},
    }
    manifest = {"run_id": run_id, **seed, "candidate_id": candidate_id}
    return IdentifierTerminalPlan(
        run_id=run_id,
        run_path=work / run_id,
        parent=parent,
        manifest=manifest,
        catalog=catalog,
        skill=skill,
        candidate_id=candidate_id,
        validation=validation,
        dataset_path=dataset_path.expanduser().absolute(),
        descriptive_manifest_path=descriptive_manifest.expanduser().absolute(),
        primary_manifest_path=primary_manifest.expanduser().absolute(),
        capture_path=capture,
        semantic_report_path=semantic_path,
        search_gate_report_path=gate_path,
    )


def _assert_primary_capture_absent(plan: IdentifierTerminalPlan) -> None:
    manifests = distiller._mapping(
        plan.manifest["input_bindings"].get("manifests"), "repair manifest bindings"
    )
    primary = distiller._mapping(manifests.get("primary_untouched"), "primary manifest binding")
    primary_id = distiller._safe_component(
        distiller._required_text(primary.get("content_id"), "primary manifest id"),
        "primary manifest id",
    )
    primary_capture = plan.capture_path.parent / primary_id
    if primary_capture.exists() or primary_capture.is_symlink():
        raise _error("primary 88-question capture started before v11 candidate save")


def execute_identifier_terminal(plan: IdentifierTerminalPlan) -> IdentifierTerminalResult:
    """Materialize the immutable candidate under one store lock without activation."""

    rebuilt = build_identifier_terminal_plan(
        parent_project_dir=plan.parent.project_dir,
        parent_store_dir=plan.parent.store_dir,
        parent_candidate_id=plan.parent.candidate_id,
        dataset_path=plan.dataset_path,
        descriptive_manifest=plan.descriptive_manifest_path,
        primary_manifest=plan.primary_manifest_path,
        capture_dir=plan.capture_path,
        semantic_report=plan.semantic_report_path,
        search_gate_report=plan.search_gate_report_path,
        work_dir=plan.run_path.parent,
    )
    if rebuilt != plan:
        raise _error("identifier-terminal plan differs from immediately re-attested inputs")
    if plan.run_path.is_symlink():
        raise _error("identifier-terminal run path cannot be a symlink")
    plan.run_path.mkdir(parents=True, exist_ok=True)
    distiller._write_json_atomic(plan.run_path / "run_manifest.json", plan.manifest)
    catalog_path = plan.run_path / "final_catalog.json"
    distiller._write_stage_artifact(
        catalog_path,
        {
            "schema_version": distiller.SCHEMA_VERSION,
            "stage": "identifier_terminal_repair",
            "key": plan.run_id,
            "input_sha256": distiller._digest(plan.manifest["input_bindings"]),
            "output": plan.catalog,
            "provenance": {
                "mode": distiller.IDENTIFIER_TERMINAL_REPAIR_MODE,
                "parent_candidate_id": plan.parent.candidate_id,
                "input_bindings": plan.manifest["input_bindings"],
                "new_calls": dict(ZERO_CALLS),
            },
        },
    )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan.run_id,
        "mode": distiller.IDENTIFIER_TERMINAL_REPAIR_MODE,
        "candidate_id": plan.candidate_id,
        "new_calls": dict(ZERO_CALLS),
        "stage_artifacts": [
            {
                "path": catalog_path.name,
                "sha256": f"sha256:{distiller._file_sha256(catalog_path)}",
                "size_bytes": catalog_path.stat().st_size,
            }
        ],
    }
    completion_path = plan.run_path / "completion_manifest.json"
    distiller._write_json_atomic(completion_path, completion)
    validation = json.loads(json.dumps(plan.validation))
    validation["artifacts"].update(
        {
            "catalog_sha256": f"sha256:{distiller._file_sha256(catalog_path)}",
            "completion_manifest_sha256": f"sha256:{distiller._file_sha256(completion_path)}",
        }
    )
    distiller._write_text_atomic(plan.run_path / "candidate" / distiller.SKILL_PATH, plan.skill)
    report_path = plan.run_path / "candidate_validation.json"

    store = SkillDistillationStore(distiller.NAMESPACE, plan.parent.project_dir)
    if store.store_path.resolve(strict=True) != plan.parent.store_dir:
        raise _error("v11 candidate save store changed after planning")
    index = (
        "# TrialQA identifier-terminal repair bundle\n\n"
        f"The executable skill is [`{distiller.SKILL_PATH}`]({distiller.SKILL_PATH}).\n"
    )
    with store.exclusive_lock():
        active_before = base._active_binding(store)
        if active_before is not None and active_before.get("candidate_id") == plan.candidate_id:
            raise _error("v11 identifier-terminal candidate is already active")
        _assert_primary_capture_absent(plan)
        candidate_path = store._save_candidate(
            candidate_id=plan.candidate_id,
            skills={"SKILL.md": index, distiller.SKILL_PATH: plan.skill},
            generator=(
                f"deterministic {distiller.IDENTIFIER_TERMINAL_REPAIR_MODE} "
                f"parent={plan.parent.candidate_id}"
            ),
            evidence_ids=list(plan.parent.evidence_ids),
            validation=validation,
            created_at=None,
        )
        saved_manifest_path = distiller._real_file(
            candidate_path / "manifest.json", "saved v11 candidate manifest"
        )
        saved_manifest = distiller._read_json_file(
            saved_manifest_path, "saved v11 candidate manifest"
        )
        saved_provenance = distiller._mapping(
            saved_manifest.get("provenance"), "saved v11 provenance"
        )
        if (
            saved_manifest.get("candidate_id") != plan.candidate_id
            or saved_manifest.get("validation") != validation
            or saved_provenance.get("source_evidence_ids") != list(plan.parent.evidence_ids)
            or distiller._file_sha256(candidate_path / distiller.SKILL_PATH)
            != hashlib.sha256(plan.skill.encode()).hexdigest()
        ):
            raise _error("saved v11 candidate differs from its immutable plan")
        if base._active_binding(store) != active_before:
            raise _error("saving v11 changed the active candidate")
        _assert_primary_capture_absent(plan)
    distiller._write_json_atomic(report_path, validation)
    return IdentifierTerminalResult(
        run_id=plan.run_id,
        candidate_id=plan.candidate_id,
        candidate_path=candidate_path,
        report_path=report_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"))
    parser.add_argument("--parent-project-dir", type=Path, required=True)
    parser.add_argument("--parent-store-dir", type=Path, required=True)
    parser.add_argument("--parent-candidate-id", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--descriptive-manifest", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--search-gate-report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_identifier_terminal_plan(
            parent_project_dir=args.parent_project_dir,
            parent_store_dir=args.parent_store_dir,
            parent_candidate_id=args.parent_candidate_id,
            dataset_path=args.dataset,
            descriptive_manifest=args.descriptive_manifest,
            primary_manifest=args.primary_manifest,
            capture_dir=args.capture_dir,
            semantic_report=args.semantic_report,
            search_gate_report=args.search_gate_report,
            work_dir=args.work_dir,
        )
        if args.command == "plan":
            output: Mapping[str, Any] = plan.manifest
        else:
            result = execute_identifier_terminal(plan)
            output = {
                "run_id": result.run_id,
                "candidate_id": result.candidate_id,
                "candidate_path": str(result.candidate_path),
                "validation_report": str(result.report_path),
                "activated": False,
                "model_call_count": result.model_call_count,
                "judge_call_count": 0,
                "evidence_import_count": 0,
                "network_call_count": 0,
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (
        distiller.TrialQADistillationError,
        search_gate.TrialQASearchGateError,
        OSError,
        ValueError,
    ) as exc:
        print(f"trialqa_local_identifier_terminal_repair: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
