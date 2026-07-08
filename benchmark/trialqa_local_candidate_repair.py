# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create one deterministic, zero-call repair of an exposed TrialQA candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_dataset as dataset_module  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_distiller as distiller  # noqa: E402
import benchmark.trialqa_local_regression as regression  # noqa: E402
import switchyard.lib.skill_distillation_store as store_module  # noqa: E402
from benchmark.trialqa_local_dataset import (  # noqa: E402
    TRIALQA_DATASET_ID,
    TrialQADataset,
    load_pinned_trialqa_parquet,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_candidate_mechanism_repair.v1"
EXPECTED_Q2_CHECKS = {
    "normalized_answer": True,
    "no_unsupported_quantitative_value": True,
    "direct_successful_operation": False,
    "supporting_payload": False,
}
EXPECTED_OPERATIONS = {
    "ClinicalTrials_search_studies": 6,
    "ClinicalTrials_get_study": 1,
    "get_clinical_trial_eligibility_criteria": 0,
}
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NATIVE_ID = re.compile(r"native-[0-9a-f]{32}\Z")

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ParentCandidate:
    project_dir: Path
    store_dir: Path
    candidate_id: str
    candidate_path: Path
    binding: JsonObject
    catalog_binding: JsonObject
    catalog: JsonObject
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRepairPlan:
    run_id: str
    run_path: Path
    parent: ParentCandidate
    manifest: JsonObject
    catalog: JsonObject
    skill: str
    candidate_id: str
    validation: JsonObject
    dataset_path: Path
    descriptive_manifest_path: Path
    primary_manifest_path: Path
    capture_path: Path
    q7_report_path: Path
    q2_report_path: Path


@dataclass(frozen=True)
class CandidateRepairResult:
    run_id: str
    candidate_id: str
    candidate_path: Path
    report_path: Path
    model_call_count: int = 0


def _error(message: str) -> distiller.TrialQADistillationError:
    return distiller.TrialQADistillationError(message)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _binding(path: Path, document: Mapping[str, Any], *, content_id: str) -> JsonObject:
    return {
        "name": path.name,
        "content_id": content_id,
        "file_sha256": f"sha256:{distiller._file_sha256(path)}",
        "canonical_sha256": _canonical_sha256(document),
        "size_bytes": path.stat().st_size,
    }


def _validate_content_id(document: Mapping[str, Any]) -> str:
    kind = document.get("kind")
    content_id = document.get("manifest_id")
    if kind != "full" or not isinstance(content_id, str):
        raise _error("repair inputs must be full TrialQA manifests")
    seed = {key: value for key, value in document.items() if key != "manifest_id"}
    expected = f"trialqa-{kind}-{hashlib.sha256(demo._canonical_json(seed)).hexdigest()[:20]}"
    if content_id != expected:
        raise _error("TrialQA manifest ID is not content-addressed")
    return content_id


def _load_parent(
    *,
    project_dir: Path,
    store_dir: Path,
    candidate_id: str,
    work_dir: Path,
    expected_mode: str = distiller.DEVELOPMENT_LAYER_MODE,
    expected_stage: str = "development_layer",
    expected_new_calls: Mapping[str, int] | None = None,
    expected_catalog_mode_field: str | None = None,
    expected_evidence_count: int | None = None,
) -> ParentCandidate:
    """Load an immutable compact candidate and its content-bound source catalog.

    The defaults retain the original development-layer contract. Later zero-call
    repair stages pass their exact mode, stage, call attestation, and evidence
    cardinality rather than reimplementing the candidate/store validation.
    """

    project = distiller._real_directory(project_dir, "parent project directory")
    store = distiller._real_directory(store_dir, "parent skill store")
    expected_store = (
        project / ".switchyard" / "skill-distillation" / distiller.NAMESPACE
    ).absolute()
    if store != expected_store.resolve(strict=True):
        raise _error("explicit parent store does not belong to the parent project/namespace")
    candidate_id = distiller._safe_component(candidate_id, "parent candidate id")
    candidate_path = distiller._real_directory(
        store / "candidates" / candidate_id, "parent candidate directory"
    )
    manifest_path = distiller._real_file(
        candidate_path / "manifest.json", "parent candidate manifest"
    )
    manifest = distiller._read_json_file(manifest_path, "parent candidate manifest")
    validation = distiller._mapping(manifest.get("validation"), "parent validation")
    checks = distiller._mapping(validation.get("checks"), "parent validation checks")
    provenance = distiller._mapping(manifest.get("provenance"), "parent provenance")
    evidence_ids = tuple(
        cast(list[str], distiller._list(provenance.get("source_evidence_ids"), "parent evidence"))
    )
    calls_are_valid = (
        validation.get("new_model_call_count") == 0
        if expected_new_calls is None
        else validation.get("new_calls") == dict(expected_new_calls)
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("namespace") != distiller.NAMESPACE
        or manifest.get("candidate_id") != candidate_id
        or validation.get("status") != "passed"
        or validation.get("candidate_id") != candidate_id
        or validation.get("distillation_mode") != expected_mode
        or validation.get("tool_contract") != "compact"
        or validation.get("performance_eligible") is not True
        or not calls_are_valid
        or not checks
        or any(value is not True for value in checks.values())
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or any(_NATIVE_ID.fullmatch(value) is None for value in evidence_ids)
        or validation.get("source_evidence_ids") != list(evidence_ids)
        or (expected_evidence_count is not None and len(evidence_ids) != expected_evidence_count)
    ):
        raise _error("parent is not one passed compact candidate in the required mode")

    skill_path = distiller._real_file(
        candidate_path / distiller.SKILL_PATH, "parent executable skill"
    )
    skill_sha = distiller._file_sha256(skill_path)
    skills = [
        distiller._mapping(item, "parent skill entry")
        for item in distiller._list(manifest.get("skills"), "parent skills")
    ]
    executable = [item for item in skills if item.get("path") == distiller.SKILL_PATH]
    artifacts = distiller._mapping(validation.get("artifacts"), "parent artifacts")
    if (
        len(executable) != 1
        or executable[0].get("sha256") != skill_sha
        or artifacts.get("skill_sha256") != f"sha256:{skill_sha}"
    ):
        raise _error("parent executable skill hash binding is invalid")

    run_id = distiller._safe_component(
        distiller._required_text(validation.get("run_id"), "parent run id"), "parent run id"
    )
    run_path = distiller._real_directory(work_dir / run_id, "parent repair-source run")
    report = distiller._read_json_file(
        distiller._real_file(run_path / "candidate_validation.json", "parent run report"),
        "parent run report",
    )
    if report != validation:
        raise _error("parent candidate validation differs from its immutable run report")
    catalog_path = distiller._real_file(run_path / "final_catalog.json", "parent catalog")
    distiller._validate_stage_integrity(catalog_path)
    catalog_document = distiller._read_json_file(catalog_path, "parent catalog")
    catalog = distiller._mapping(catalog_document.get("output"), "parent catalog output")
    if (
        catalog_document.get("key") != run_id
        or catalog_document.get("stage") != expected_stage
        or artifacts.get("catalog_sha256") != f"sha256:{distiller._file_sha256(catalog_path)}"
        or (
            expected_catalog_mode_field is not None
            and catalog.get(expected_catalog_mode_field) != expected_mode
        )
        or distiller.render_skill_markdown(catalog, tool_contract="compact")
        != skill_path.read_text(encoding="utf-8")
    ):
        raise _error("parent catalog does not reproduce the bound executable skill")
    distiller.validate_compact_skill(catalog, skill_path.read_text(), tool_contract="compact")
    manifest_sha = distiller._file_sha256(manifest_path)
    return ParentCandidate(
        project_dir=project,
        store_dir=store,
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        binding=distiller._candidate_binding(candidate_id, manifest_sha, skill_sha),
        catalog_binding={
            "run_id": run_id,
            "sha256": f"sha256:{distiller._file_sha256(catalog_path)}",
            "integrity_sha256": (
                f"sha256:{distiller._file_sha256(distiller._integrity_path(catalog_path))}"
            ),
            "size_bytes": catalog_path.stat().st_size,
        },
        catalog=dict(catalog),
        evidence_ids=evidence_ids,
    )


def _task_groups(manifest: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, JsonObject]]:
    tasks = [
        distiller._mapping(item, "manifest task")
        for item in distiller._list(manifest.get("tasks"), "manifest tasks")
    ]
    by_id: dict[str, JsonObject] = {}
    groups: list[str] = []
    pairs: Counter[tuple[str, int, str]] = Counter()
    for task in tasks:
        task_id = distiller._required_text(task.get("task_id"), "task id")
        group = distiller._required_text(task.get("question_group_key"), "question group")
        repeat = distiller._required_int(task.get("repeat_index"), "repeat index", minimum=1)
        condition = task.get("condition")
        if task_id in by_id or condition not in {"baseline", "treatment"}:
            raise _error("manifest tasks are duplicated or have an invalid condition")
        by_id[task_id] = task
        if group not in groups:
            groups.append(group)
        pairs[(group, repeat, cast(str, condition))] += 1
    expected = {
        (group, repeat, condition)
        for group in groups
        for repeat in range(1, 6)
        for condition in ("baseline", "treatment")
    }
    if set(pairs) != expected or any(count != 1 for count in pairs.values()):
        raise _error("manifest is not a complete five-repeat paired matrix")
    return tuple(groups), by_id


def _load_manifests(
    descriptive_path: Path, primary_path: Path, *, parent: ParentCandidate, capture: Path
) -> tuple[JsonObject, JsonObject, JsonObject, tuple[str, ...], dict[str, JsonObject]]:
    descriptive_path = distiller._real_file(descriptive_path, "descriptive manifest")
    primary_path = distiller._real_file(primary_path, "primary manifest")
    descriptive = distiller._read_json_file(descriptive_path, "descriptive manifest")
    primary = distiller._read_json_file(primary_path, "primary manifest")
    try:
        demo.validate_manifest_pairing(descriptive)
        demo.validate_manifest_pairing(primary)
    except demo.TrialQADemoError as exc:
        raise _error("descriptive/primary manifest pairing is invalid") from exc
    descriptive_id = _validate_content_id(descriptive)
    primary_id = _validate_content_id(primary)
    descriptive_groups, descriptive_tasks = _task_groups(descriptive)
    primary_groups, _primary_tasks = _task_groups(primary)
    descriptive_protocol = distiller._mapping(descriptive.get("protocol"), "descriptive protocol")
    primary_protocol = distiller._mapping(primary.get("protocol"), "primary protocol")
    scope = distiller._mapping(primary_protocol.get("primary_evaluation_scope"), "primary scope")
    quarantine = distiller._mapping(
        primary_protocol.get("heldout_quarantine"), "primary quarantine"
    )
    common_top_level = (
        "schema_version",
        "kind",
        "dataset",
        "candidate",
        "routing",
        "implementation",
        "preflight",
        "runtime",
    )
    protocol_exceptions = {
        "performance_eligible",
        "primary_evaluation_scope",
        "heldout_quarantine",
    }
    descriptive_common_protocol = {
        key: value for key, value in descriptive_protocol.items() if key not in protocol_exceptions
    }
    primary_common_protocol = {
        key: value for key, value in primary_protocol.items() if key not in protocol_exceptions
    }
    descriptive_task_list = cast(list[JsonObject], descriptive["tasks"])
    primary_task_list = cast(list[JsonObject], primary["tasks"])
    expected_primary_tasks = [
        task
        for task in descriptive_task_list
        if task.get("question_group_key") in set(descriptive_groups[8:])
    ]
    if (
        descriptive.get("candidate") != parent.binding
        or primary.get("candidate") != parent.binding
        or any(descriptive.get(key) != primary.get(key) for key in common_top_level)
        or descriptive_common_protocol != primary_common_protocol
        or descriptive.get("dataset") != primary.get("dataset")
        or len(descriptive_groups) != 96
        or len(descriptive_tasks) != 960
        or descriptive_protocol.get("performance_eligible") is not False
        or descriptive_protocol.get("primary_evaluation_scope") is not None
        or len(primary_groups) != 88
        or len(_primary_tasks) != 880
        or primary_groups != descriptive_groups[8:]
        or primary_task_list != expected_primary_tasks
        or primary_protocol.get("performance_eligible") is not True
        or scope
        != {"question_start": 8, "question_count": 88, "repeat_count": 5, "task_count": 880}
        or quarantine
        != {
            "question_start": 0,
            "question_count": 8,
            "disposition": "excluded-exposed-heldout",
            "question_group_keys_sha256": _canonical_sha256(list(descriptive_groups[:8])),
        }
    ):
        raise _error("descriptive/primary manifests do not encode the reviewed 8/88 split")
    capture = distiller._real_directory(capture, "descriptive capture")
    if descriptive_path.parent != primary_path.parent or capture.parent != descriptive_path.parent:
        raise _error("manifests and descriptive capture must share one experiment root")
    if capture.name != descriptive_id:
        raise _error("descriptive capture directory does not match its manifest ID")
    primary_capture = capture.parent / primary_id
    if primary_capture.exists() or primary_capture.is_symlink():
        raise _error("primary 88-question capture has already been started")
    bindings = {
        "descriptive": _binding(descriptive_path, descriptive, content_id=descriptive_id),
        "primary_untouched": _binding(primary_path, primary, content_id=primary_id),
        "primary_capture_started": False,
    }
    return descriptive, primary, bindings, descriptive_groups, descriptive_tasks


def _load_dataset(path: Path) -> tuple[TrialQADataset, JsonObject]:
    dataset_path = distiller._real_file(path, "pinned TrialQA parquet")
    dataset = load_pinned_trialqa_parquet(dataset_path)
    file_sha256 = distiller._file_sha256(dataset_path)
    if file_sha256 != dataset.parquet_sha256:
        raise _error("pinned TrialQA parquet hash changed while loading")
    return dataset, {
        "name": dataset_path.name,
        "dataset_id": TRIALQA_DATASET_ID,
        "revision": dataset.revision,
        "file_sha256": f"sha256:{file_sha256}",
        "size_bytes": dataset_path.stat().st_size,
    }


def _load_report(path: Path, label: str) -> tuple[Path, JsonObject]:
    path = distiller._real_file(path, label)
    report = distiller._read_json_file(path, label)
    supplied = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise _error(f"{label} self-hash is invalid")
    policy = distiller._mapping(report.get("policy"), f"{label} policy")
    if policy != {
        "name": regression.REGRESSION_POLICY,
        "performance_eligible": False,
        "allowed_question_ordinals": [2, 5, 7],
        "condition": "treatment",
        "model_calls": 0,
        "judge_calls": 0,
        "evidence_imports": 0,
    }:
        raise _error(f"{label} is not a zero-call exposed mechanism report")
    return path, report


def _validate_ledger_scope(
    *,
    descriptive: Mapping[str, Any],
    capture: Path,
    expected_task_ids: Sequence[str],
) -> JsonObject:
    ledger_path = distiller._real_file(capture / "ledger.jsonl", "descriptive ledger")
    ledger = demo.ResumableLedger(ledger_path, descriptive)
    try:
        states = ledger.states()
        records = ledger.records()
    except demo.TrialQADemoError as exc:
        raise _error("descriptive ledger integrity is invalid") from exc
    expected = dict.fromkeys(expected_task_ids, "generation_completed")
    event_sequences = {
        task_id: [record.get("event") for record in records if record.get("task_id") == task_id]
        for task_id in expected
    }
    if (
        not expected
        or len(expected) != len(expected_task_ids)
        or states != expected
        or len(records) != 2 * len(expected)
        or any(
            sequence != ["generation_started", "generation_completed"]
            for sequence in event_sequences.values()
        )
    ):
        raise _error("descriptive ledger does not contain exactly the requested completions")
    return {
        "name": ledger_path.name,
        "file_sha256": f"sha256:{distiller._file_sha256(ledger_path)}",
        "size_bytes": ledger_path.stat().st_size,
        "record_count": len(records),
        "final_record_sha256": records[-1]["record_sha256"],
        "terminal_states": dict(sorted(states.items())),
    }


def _expected_task(tasks: Mapping[str, JsonObject], group: str, *, repeat: int) -> JsonObject:
    matches = [
        task
        for task in tasks.values()
        if task.get("question_group_key") == group
        and task.get("repeat_index") == repeat
        and task.get("condition") == "treatment"
    ]
    if len(matches) != 1:
        raise _error("manifest lacks one expected exposed treatment task")
    return matches[0]


def _validate_reports(
    *,
    q7_path: Path,
    q2_path: Path,
    descriptive: Mapping[str, Any],
    descriptive_binding: Mapping[str, Any],
    dataset: TrialQADataset,
    capture: Path,
    groups: Sequence[str],
    tasks: Mapping[str, JsonObject],
) -> tuple[JsonObject, JsonObject, JsonObject, list[JsonObject]]:
    q7_file, q7 = _load_report(q7_path, "q7 pass report")
    q2_file, q2 = _load_report(q2_path, "q2 kill report")
    manifest_id = descriptive.get("manifest_id")
    manifest_sha = cast(str, descriptive_binding["canonical_sha256"])
    for label, report in (("q7", q7), ("q2", q2)):
        if (
            report.get("schema_version") != regression.REGRESSION_SCHEMA_VERSION
            or report.get("manifest_id") != manifest_id
            or report.get("manifest_sha256") != manifest_sha
            or report.get("dataset")
            != {
                key: distiller._mapping(descriptive.get("dataset"), "dataset").get(key)
                for key in ("id", "revision", "parquet_sha256")
            }
        ):
            raise _error(f"{label} report is not bound to the descriptive manifest")

    q7_results = [
        distiller._mapping(item, "q7 result")
        for item in distiller._list(q7.get("results"), "q7 results")
    ]
    q7_expected = [_expected_task(tasks, groups[7], repeat=repeat) for repeat in range(1, 6)]
    q7_summary = distiller._mapping(q7.get("summary"), "q7 summary")
    if (
        q7_summary != {"checked_tasks": 5, "passed_tasks": 5, "killed_tasks": 0, "decision": "pass"}
        or len(q7_results) != 5
    ):
        raise _error("q7 report is not the required five-of-five pass")
    for result, task in zip(q7_results, q7_expected, strict=True):
        bindings = distiller._mapping(result.get("bindings"), "q7 result bindings")
        if (
            result.get("task_id") != task.get("task_id")
            or result.get("question_ordinal") != 7
            or result.get("repeat_index") != task.get("repeat_index")
            or result.get("condition") != "treatment"
            or result.get("decision") != "pass"
            or result.get("checks") != dict.fromkeys(EXPECTED_Q2_CHECKS, True)
            or result.get("kill_reasons") != []
            or bindings.get("row_id") != task.get("row_id")
            or bindings.get("dataset_row_index") != task.get("dataset_row_index")
            or any(
                _SHA256.fullmatch(str(bindings.get(field))) is None
                for field in ("generation_sha256", "codex_events_sha256")
            )
        ):
            raise _error("q7 report result does not match the pinned five-repeat task")

    q2_results = [
        distiller._mapping(item, "q2 result")
        for item in distiller._list(q2.get("results"), "q2 results")
    ]
    q2_task = _expected_task(tasks, groups[2], repeat=1)
    q2_summary = distiller._mapping(q2.get("summary"), "q2 summary")
    if (
        q2_summary != {"checked_tasks": 1, "passed_tasks": 0, "killed_tasks": 1, "decision": "kill"}
        or len(q2_results) != 1
    ):
        raise _error("q2 report is not the required one-draw mechanism kill")
    q2_result = q2_results[0]
    q2_bindings = distiller._mapping(q2_result.get("bindings"), "q2 bindings")
    if (
        q2_result.get("task_id") != q2_task.get("task_id")
        or q2_result.get("question_ordinal") != 2
        or q2_result.get("repeat_index") != 1
        or q2_result.get("condition") != "treatment"
        or q2_result.get("decision") != "kill"
        or q2_result.get("checks") != EXPECTED_Q2_CHECKS
        or q2_result.get("kill_reasons") != ["direct_successful_operation", "supporting_payload"]
        or q2_result.get("expected_operation") != "get_clinical_trial_eligibility_criteria"
        or q2_result.get("supporting_payload_sha256") != []
        or q2_result.get("operational_calls") != 7
        or q2_bindings.get("row_id") != q2_task.get("row_id")
        or q2_bindings.get("dataset_row_index") != q2_task.get("dataset_row_index")
    ):
        raise _error("q2 kill report does not attest the reviewed mechanism failure")
    try:
        recomputed_q7 = regression.check_treatment_tasks(
            manifest=descriptive,
            dataset=dataset,
            capture=capture,
            task_ids=[cast(str, task["task_id"]) for task in q7_expected],
        )
        recomputed_q2 = regression.check_treatment_tasks(
            manifest=descriptive,
            dataset=dataset,
            capture=capture,
            task_ids=[cast(str, q2_task["task_id"])],
        )
    except (regression.TrialQARegressionError, demo.TrialQADemoError, OSError) as exc:
        raise _error("mechanism report source evidence failed independent replay") from exc
    if recomputed_q7 != q7:
        raise _error("q7 report differs from independently recomputed five-repeat evidence")
    if recomputed_q2 != q2:
        raise _error("q2 report differs from independently recomputed mechanism evidence")
    report_bindings = {
        "q7_pass": _binding(q7_file, q7, content_id=cast(str, q7["report_sha256"])),
        "q2_kill": _binding(q2_file, q2, content_id=cast(str, q2["report_sha256"])),
    }
    return q2_task, q2_result, report_bindings, q7_results


def _successful_operation_counts(events: Sequence[Mapping[str, Any]]) -> Counter[str]:
    operations: Counter[str] = Counter()
    for event in events:
        item = event.get("item")
        if (
            event.get("type") != "item.completed"
            or not isinstance(item, Mapping)
            or item.get("type") != "mcp_tool_call"
            or item.get("server") != "tooluniverse"
            or item.get("tool") != "execute_tool"
            or item.get("status") != "completed"
            or item.get("error") is not None
            or not regression._successful_child_payloads(item)
        ):
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, Mapping) or not isinstance(arguments.get("tool_name"), str):
            raise _error("q2 successful execute_tool event has invalid arguments")
        operations[cast(str, arguments["tool_name"])] += 1
    return operations


def _zero_retry_accounting(generation: demo.GenerationResult, *, label: str) -> JsonObject:
    stats = distiller._read_json_file(generation.stats_path, f"{label} session stats")
    if stats != generation.stats:
        raise _error(f"{label} generation stats differ from the immutable session stats")
    transport = distiller._mapping(stats.get("openai_transport"), f"{label} transport stats")
    retry_sensitivity = distiller._mapping(
        transport.get("retry_token_sensitivity"), f"{label} retry sensitivity"
    )
    models = distiller._mapping(stats.get("models"), f"{label} model stats")
    model = distiller._mapping(models.get(distiller.EXECUTOR_MODEL), f"{label} Ultra stats")
    logical_requests = stats.get("total_requests")
    if (
        not isinstance(logical_requests, int)
        or isinstance(logical_requests, bool)
        or logical_requests < 1
        or set(models) != {distiller.EXECUTOR_MODEL}
        or stats.get("total_errors") != 0
        or model.get("calls") != logical_requests
        or model.get("errors") != 0
        or transport.get("physical_attempts") != logical_requests
        or transport.get("null_eof_retries") != 0
        or transport.get("retry_usage_charges") != 0
        or transport.get("unpriced_null_eof_retries") != 0
        or not retry_sensitivity
        or any(value != 0 for value in retry_sensitivity.values())
    ):
        raise _error(f"{label} is not zero-retry one-physical-attempt-per-logical-request")
    return {
        "logical_requests": logical_requests,
        "physical_attempts": transport["physical_attempts"],
        "null_eof_retries": 0,
        "retry_usage_charges": 0,
        "unpriced_null_eof_retries": 0,
    }


def _generation_literals(
    generation: demo.GenerationResult, events: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    literals: set[str] = {
        generation.answer,
        generation.task_id,
        generation.pair_id,
        generation.row_id,
    }
    literals.update(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:weeks?|days?|months?|years?|mg|copies/mL)\b",
            generation.answer,
            flags=re.IGNORECASE,
        )
    )
    for event in events:
        item = event.get("item")
        if not isinstance(item, Mapping) or not isinstance(item.get("arguments"), Mapping):
            continue
        encoded = cast(Mapping[str, Any], item["arguments"]).get("arguments_json")
        if not isinstance(encoded, str):
            continue
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            literals.update(
                value.strip()
                for value in parsed.values()
                if isinstance(value, str) and len(value.strip()) >= 5
            )
    return tuple(sorted(literals, key=lambda value: (-len(value), value)))


def _validate_q7_generations(
    *,
    capture: Path,
    manifest_id: str,
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonObject], tuple[str, ...]]:
    bindings: list[JsonObject] = []
    literals: set[str] = set()
    for task, result in zip(tasks, results, strict=True):
        pair_id = distiller._required_text(task.get("pair_id"), "q7 pair id")
        generation_path = distiller._real_file(
            capture
            / "trialqa-local"
            / pair_id
            / "arms"
            / "treatment"
            / "outputs"
            / "generation.json",
            "q7 generation",
        )
        generation = demo.load_generation_result(generation_path)
        identity = {
            "manifest_id": manifest_id,
            "task_id": task.get("task_id"),
            "pair_id": pair_id,
            "row_id": task.get("row_id"),
            "dataset_row_index": task.get("dataset_row_index"),
            "condition": "treatment",
            "repeat_index": task.get("repeat_index"),
            "n_repeats": 5,
        }
        if any(getattr(generation, key) != value for key, value in identity.items()):
            raise _error("q7 generation identity differs from its manifest task")
        recomputed = regression._evaluate_generation(generation, regression.MECHANISM_SPECS[7])
        if recomputed != dict(result):
            raise _error("q7 result differs from its independently recomputed generation")
        accounting = _zero_retry_accounting(
            generation, label=f"q7 repeat {generation.repeat_index}"
        )
        events = demo.read_codex_events(generation.codex_events_path)
        literals.update(_generation_literals(generation, events))
        bindings.append(
            {
                "task_id": generation.task_id,
                "repeat_index": generation.repeat_index,
                "generation_sha256": demo._sha256_file(generation_path),
                "codex_events_sha256": demo._sha256_file(generation.codex_events_path),
                "stats_sha256": demo._sha256_file(generation.stats_path),
                "session_sha256": demo._sha256_file(generation.session_dir / "session.json"),
                **accounting,
            }
        )
    return bindings, tuple(sorted(literals, key=lambda value: (-len(value), value)))


def _validate_q2_generation(
    *,
    capture: Path,
    manifest_id: str,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    parent: ParentCandidate,
) -> tuple[JsonObject, tuple[str, ...]]:
    pair_id = distiller._required_text(task.get("pair_id"), "q2 pair id")
    generation_path = distiller._real_file(
        capture / "trialqa-local" / pair_id / "arms" / "treatment" / "outputs" / "generation.json",
        "q2 generation",
    )
    generation = demo.load_generation_result(generation_path)
    identity = {
        "manifest_id": manifest_id,
        "task_id": task.get("task_id"),
        "pair_id": pair_id,
        "row_id": task.get("row_id"),
        "dataset_row_index": task.get("dataset_row_index"),
        "condition": "treatment",
        "repeat_index": 1,
        "n_repeats": 5,
    }
    if any(getattr(generation, key) != value for key, value in identity.items()):
        raise _error("q2 generation identity differs from its manifest task")
    result_bindings = distiller._mapping(result.get("bindings"), "q2 result bindings")
    if result_bindings.get("generation_sha256") != demo._sha256_file(generation_path):
        raise _error("q2 report generation hash changed")
    for path in (
        generation.session_dir,
        generation.stats_path,
        generation.trajectory_path,
        generation.codex_events_path,
        generation.final_output_path,
        generation.generation_path,
    ):
        if not path.resolve(strict=True).is_relative_to(capture):
            raise _error("q2 generation artifact escapes the descriptive capture")
    if result_bindings.get("codex_events_sha256") != demo._sha256_file(
        generation.codex_events_path
    ):
        raise _error("q2 report event hash changed")
    accounting = _zero_retry_accounting(generation, label="q2")
    if accounting["logical_requests"] != 9:
        raise _error("q2 draw is not 9 logical/9 physical attempts with zero retry")

    try:
        demo._parse_codex_events(
            generation.codex_events_path, require_skill_load=True, enforce_tool_policy=True
        )
        metrics = demo.codex_tool_metrics(generation.codex_events_path)
    except demo.TrialQADemoError as exc:
        raise _error("q2 Codex events fail the benchmark policy") from exc
    events = demo.read_codex_events(generation.codex_events_path)
    recomputed = regression._evaluate_generation(generation, regression.MECHANISM_SPECS[2])
    if recomputed != dict(result):
        raise _error("q2 result differs from its independently recomputed generation")
    operation_counts = _successful_operation_counts(events)
    if (
        metrics.get("operational_calls") != 7
        or metrics.get("successful_operational_calls") != 7
        or metrics.get("skill_load_calls") != 1
        or operation_counts
        != Counter({key: value for key, value in EXPECTED_OPERATIONS.items() if value})
    ):
        raise _error("q2 operations are not exactly 6 search / 1 full-study / 0 eligibility")

    session = distiller._read_json_file(
        distiller._real_file(generation.session_dir / "session.json", "q2 session manifest"),
        "q2 session manifest",
    )
    active = distiller._mapping(session.get("active_skill"), "q2 active skill")
    context = distiller._mapping(session.get("run_context"), "q2 run context")
    if (
        session.get("status") != "completed"
        or session.get("exit_code") != 0
        or session.get("turn_count") != 9
        or active.get("loaded") is not True
        or {key: active.get(key) for key in parent.binding} != parent.binding
        or context.get("task_id") != task.get("task_id")
        or context.get("candidate_id") != parent.candidate_id
        or context.get("candidate_manifest_sha256") != parent.binding["manifest_sha256"]
        or context.get("candidate_skill_sha256") != parent.binding["skill_sha256"]
    ):
        raise _error("q2 session is not bound to the reviewed parent candidate/task")

    artifact_hashes = generation.artifact_sha256
    answer_path = generation_path.parent.parent / "answer.txt"
    artifact_paths = {
        "answer": answer_path,
        "codex_events": generation.codex_events_path,
        "final_output": generation.final_output_path,
        "stats": generation.stats_path,
        "trajectory": generation.trajectory_path,
    }
    if set(artifact_hashes) != set(artifact_paths) or any(
        artifact_hashes[name] != demo._sha256_file(distiller._real_file(path, name))
        for name, path in artifact_paths.items()
    ):
        raise _error("q2 generation artifact hash set changed")

    evidence_binding = {
        "task_id": generation.task_id,
        "generation_sha256": demo._sha256_file(generation_path),
        "codex_events_sha256": demo._sha256_file(generation.codex_events_path),
        "stats_sha256": demo._sha256_file(generation.stats_path),
        "session_sha256": demo._sha256_file(generation.session_dir / "session.json"),
        **accounting,
        "operational_calls": 7,
        "operation_counts": dict(sorted({**EXPECTED_OPERATIONS}.items())),
    }
    return evidence_binding, _generation_literals(generation, events)


def build_candidate_repair_plan(
    *,
    parent_project_dir: Path,
    parent_store_dir: Path,
    parent_candidate_id: str,
    dataset_path: Path,
    descriptive_manifest: Path,
    primary_manifest: Path,
    capture_dir: Path,
    q7_pass_report: Path,
    q2_kill_report: Path,
    work_dir: Path,
) -> CandidateRepairPlan:
    """Re-attest all local inputs and build a deterministic, zero-call repair plan."""

    work = work_dir.expanduser().absolute()
    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise _error("repair work path must be a real directory or absent")
    parent = _load_parent(
        project_dir=parent_project_dir,
        store_dir=parent_store_dir,
        candidate_id=parent_candidate_id,
        work_dir=work,
    )
    dataset, dataset_binding = _load_dataset(dataset_path)
    capture = distiller._real_directory(capture_dir, "descriptive capture")
    descriptive, _primary, manifest_bindings, groups, tasks = _load_manifests(
        descriptive_manifest, primary_manifest, parent=parent, capture=capture
    )
    q2_task, q2_result, report_bindings, q7_results = _validate_reports(
        q7_path=q7_pass_report,
        q2_path=q2_kill_report,
        descriptive=descriptive,
        descriptive_binding=cast(JsonObject, manifest_bindings["descriptive"]),
        dataset=dataset,
        capture=capture,
        groups=groups,
        tasks=tasks,
    )
    q7_tasks = [_expected_task(tasks, groups[7], repeat=repeat) for repeat in range(1, 6)]
    ledger_binding = _validate_ledger_scope(
        descriptive=descriptive,
        capture=capture,
        expected_task_ids=[
            *(cast(str, task["task_id"]) for task in q7_tasks),
            cast(str, q2_task["task_id"]),
        ],
    )
    q7_evidence, q7_literals = _validate_q7_generations(
        capture=capture,
        manifest_id=cast(str, descriptive["manifest_id"]),
        tasks=q7_tasks,
        results=q7_results,
    )
    q2_evidence, sensitive_literals = _validate_q2_generation(
        capture=capture,
        manifest_id=cast(str, descriptive["manifest_id"]),
        task=q2_task,
        result=q2_result,
        parent=parent,
    )
    catalog = distiller.layer_exposed_mechanism_repair_catalog(parent.catalog)
    skill = distiller.render_skill_markdown(catalog, tool_contract="compact")
    metrics = distiller.validate_compact_skill(catalog, skill, tool_contract="compact")
    distiller._assert_no_sensitive(
        skill, (*sensitive_literals, *q7_literals), "mechanism-repair skill"
    )
    parent_q7 = [
        item
        for item in cast(list[JsonObject], parent.catalog["workflow_rules"])
        if "trialqa_extract_adverse_events" in str(item.get("rule"))
    ]
    repaired_q7 = [
        item
        for item in cast(list[JsonObject], catalog["workflow_rules"])
        if "trialqa_extract_adverse_events" in str(item.get("rule"))
    ]
    if parent_q7 != repaired_q7 or len(parent_q7) != 1:
        raise _error("mechanism repair changed the attested q7 fallback rule")

    input_bindings: JsonObject = {
        "parent_candidate": parent.binding,
        "parent_catalog": parent.catalog_binding,
        "dataset": dataset_binding,
        "manifests": manifest_bindings,
        "mechanism_reports": report_bindings,
        "descriptive_ledger": ledger_binding,
        "q7_generations": q7_evidence,
        "q2_generation": q2_evidence,
    }
    seed: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "mode": distiller.MECHANISM_REPAIR_MODE,
        "source_sha256": {
            "candidate_repair": f"sha256:{distiller._file_sha256(Path(__file__).resolve())}",
            "distiller": f"sha256:{distiller._file_sha256(Path(distiller.__file__).resolve())}",
            "regression": f"sha256:{distiller._file_sha256(Path(regression.__file__).resolve())}",
            "demo": f"sha256:{distiller._file_sha256(Path(demo.__file__).resolve())}",
            "dataset": f"sha256:{distiller._file_sha256(Path(dataset_module.__file__).resolve())}",
            "batch": f"sha256:{distiller._file_sha256(Path(batch.__file__).resolve())}",
            "skill_distillation_store": (
                f"sha256:{distiller._file_sha256(Path(store_module.__file__).resolve())}"
            ),
        },
        "input_bindings": input_bindings,
        "call_budget": {"model": 0, "judge": 0, "evidence_import": 0, "network": 0},
    }
    run_id = f"trialqa-mechanism-repair-{distiller._digest(seed)[:32]}"
    skill_sha = hashlib.sha256(skill.encode()).hexdigest()
    candidate_seed = {
        "run_id": run_id,
        "mode": distiller.MECHANISM_REPAIR_MODE,
        "parent_candidate": parent.binding,
        "skill_path": distiller.SKILL_PATH,
        "skill_sha256": skill_sha,
        "source_evidence_ids": list(parent.evidence_ids),
    }
    candidate_id = f"trialqa-{distiller._digest(candidate_seed)[:32]}"
    validation: JsonObject = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "train-base-plus-exposed-development-and-mechanism-repair-primary88-only",
        "distillation_mode": distiller.MECHANISM_REPAIR_MODE,
        "performance_validated": False,
        "performance_eligible": True,
        "full_96_performance_eligible": False,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent.candidate_id,
        "tool_contract": "compact",
        "source_evidence_ids": list(parent.evidence_ids),
        "new_calls": {"model": 0, "judge": 0, "evidence_import": 0, "network": 0},
        "input_bindings": input_bindings,
        "checks": {
            "parent_candidate_and_catalog_hash_bound": True,
            "parent_evidence_ids_inherited_exactly": True,
            "descriptive_manifest_ineligible": True,
            "primary_manifest_eligible_and_untouched": True,
            "q7_five_of_five_pass_bound": True,
            "q2_kill_checks_bound": True,
            "q2_actual_operations_recounted": True,
            "q2_nine_logical_nine_physical_zero_retry": True,
            "q7_rule_preserved_byte_for_byte": True,
            "generic_repair_has_no_task_literals": True,
            "zero_new_model_judge_import_network_calls": True,
            "candidate_remains_inactive": True,
            "compact_size": metrics["size_bytes"] <= distiller.COMPACT_SKILL_MAX_BYTES,
            "compact_words": metrics["word_count"] <= distiller.COMPACT_SKILL_MAX_WORDS,
            "compact_rules": metrics["rule_count"] <= distiller.COMPACT_SKILL_MAX_RULES,
        },
        "routing": {"attested_call_count": 0, "attestations": []},
        "artifacts": {
            "skill_sha256": f"sha256:{skill_sha}",
            **metrics,
        },
    }
    if not all(cast(dict[str, bool], validation["checks"]).values()):
        raise _error("candidate repair validation contains a failed check")
    manifest = {"run_id": run_id, **seed, "candidate_id": candidate_id}
    return CandidateRepairPlan(
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
        q7_report_path=q7_pass_report.expanduser().absolute(),
        q2_report_path=q2_kill_report.expanduser().absolute(),
    )


def _active_binding(store: SkillDistillationStore) -> JsonObject | None:
    manifest_path = store.active_path / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = distiller._read_json_file(manifest_path, "active candidate manifest")
    return {
        "candidate_id": manifest.get("candidate_id"),
        "manifest_sha256": f"sha256:{distiller._file_sha256(manifest_path)}",
    }


def _assert_primary_capture_absent(plan: CandidateRepairPlan) -> None:
    manifests = distiller._mapping(
        plan.manifest["input_bindings"].get("manifests"), "repair manifest bindings"
    )
    primary = distiller._mapping(manifests.get("primary_untouched"), "primary binding")
    primary_id = distiller._safe_component(
        distiller._required_text(primary.get("content_id"), "primary manifest id"),
        "primary manifest id",
    )
    primary_capture = plan.capture_path.parent / primary_id
    if primary_capture.exists() or primary_capture.is_symlink():
        raise _error("primary 88-question capture started before candidate save")


def execute_candidate_repair(plan: CandidateRepairPlan) -> CandidateRepairResult:
    """Materialize the content-addressed repair without activating it."""

    rebuilt = build_candidate_repair_plan(
        parent_project_dir=plan.parent.project_dir,
        parent_store_dir=plan.parent.store_dir,
        parent_candidate_id=plan.parent.candidate_id,
        dataset_path=plan.dataset_path,
        descriptive_manifest=plan.descriptive_manifest_path,
        primary_manifest=plan.primary_manifest_path,
        capture_dir=plan.capture_path,
        q7_pass_report=plan.q7_report_path,
        q2_kill_report=plan.q2_report_path,
        work_dir=plan.run_path.parent,
    )
    if rebuilt != plan:
        raise _error("repair plan differs from its immediately re-attested inputs")
    if plan.run_path.is_symlink():
        raise _error("repair run path cannot be a symlink")
    plan.run_path.mkdir(parents=True, exist_ok=True)
    distiller._write_json_atomic(plan.run_path / "run_manifest.json", plan.manifest)
    catalog_path = plan.run_path / "final_catalog.json"
    distiller._write_stage_artifact(
        catalog_path,
        {
            "schema_version": distiller.SCHEMA_VERSION,
            "stage": "mechanism_repair",
            "key": plan.run_id,
            "input_sha256": distiller._digest(plan.manifest["input_bindings"]),
            "output": plan.catalog,
            "provenance": {
                "mode": distiller.MECHANISM_REPAIR_MODE,
                "parent_candidate_id": plan.parent.candidate_id,
                "input_bindings": plan.manifest["input_bindings"],
                "new_calls": {"model": 0, "judge": 0, "evidence_import": 0, "network": 0},
            },
        },
    )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan.run_id,
        "mode": distiller.MECHANISM_REPAIR_MODE,
        "candidate_id": plan.candidate_id,
        "new_calls": {"model": 0, "judge": 0, "evidence_import": 0, "network": 0},
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
    report_path = plan.run_path / "candidate_validation.json"
    distiller._write_text_atomic(plan.run_path / "candidate" / distiller.SKILL_PATH, plan.skill)

    store = SkillDistillationStore(distiller.NAMESPACE, plan.parent.project_dir)
    if store.store_path.resolve(strict=True) != plan.parent.store_dir:
        raise _error("candidate save store changed after planning")
    index = (
        "# TrialQA exposed-mechanism repair bundle\n\n"
        f"The executable skill is [`{distiller.SKILL_PATH}`]({distiller.SKILL_PATH}).\n"
    )
    with store.exclusive_lock():
        active_before = _active_binding(store)
        if active_before is not None and active_before.get("candidate_id") == plan.candidate_id:
            raise _error("repair candidate is already active")
        _assert_primary_capture_absent(plan)
        candidate_path = store._save_candidate(
            candidate_id=plan.candidate_id,
            skills={"SKILL.md": index, distiller.SKILL_PATH: plan.skill},
            generator=(
                f"deterministic {distiller.MECHANISM_REPAIR_MODE} parent={plan.parent.candidate_id}"
            ),
            evidence_ids=list(plan.parent.evidence_ids),
            validation=validation,
            created_at=None,
        )
        if _active_binding(store) != active_before:
            raise _error("saving the repair candidate changed the active candidate")
        _assert_primary_capture_absent(plan)
    distiller._write_json_atomic(report_path, validation)
    return CandidateRepairResult(
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
    parser.add_argument("--q7-pass-report", type=Path, required=True)
    parser.add_argument("--q2-kill-report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_candidate_repair_plan(
            parent_project_dir=args.parent_project_dir,
            parent_store_dir=args.parent_store_dir,
            parent_candidate_id=args.parent_candidate_id,
            dataset_path=args.dataset,
            descriptive_manifest=args.descriptive_manifest,
            primary_manifest=args.primary_manifest,
            capture_dir=args.capture_dir,
            q7_pass_report=args.q7_pass_report,
            q2_kill_report=args.q2_kill_report,
            work_dir=args.work_dir,
        )
        if args.command == "plan":
            output: Mapping[str, Any] = plan.manifest
        else:
            result = execute_candidate_repair(plan)
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
    except (distiller.TrialQADistillationError, OSError, ValueError) as exc:
        print(f"trialqa_local_candidate_repair: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
