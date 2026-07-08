# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-model exposed-heldout mechanism checks for TrialQA treatment draws.

This checker is intentionally narrower than semantic judging.  It accepts only
the three reviewed, already-exposed held-out questions and proves that a
ledger-bound treatment generation used the expected direct evidence operation
and returned the pinned mechanism answer.  It never scores or imports evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
from benchmark.trialqa_local_dataset import (  # noqa: E402
    TRIALQA_DATASET_CONFIG,
    TRIALQA_DATASET_ID,
    TRIALQA_DATASET_REVISION,
    TrialQADataset,
    TrialQARow,
    create_split_manifest,
    load_pinned_trialqa_parquet,
    question_group_key,
)

REGRESSION_SCHEMA_VERSION = "switchyard.trialqa_exposed_mechanism_regression.v1"
REGRESSION_POLICY = "exposed-heldout-treatment-mechanism-v1"


class TrialQARegressionError(RuntimeError):
    """The requested task or its immutable evidence cannot be trusted."""


@dataclass(frozen=True)
class MechanismSpec:
    ordinal: int
    ideal: str
    nct_id: str
    operation: str
    answer_check: Callable[[str], tuple[bool, bool]]
    payload_check: Callable[[str], bool]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().replace("≥", ">=").split())


_CONTRADICTION = re.compile(
    r"\b(?:not|never|unknown|unclear|uncertain|maybe|possibly)\b"
    r"|\b(?:is|was|were|are|did|does|could|can)(?:n't|n['’]t)\b"
    r"|\b(?:rather\s+than|instead\s+of)\b"
)


def _has_contradiction(value: str) -> bool:
    return _CONTRADICTION.search(_normalized_text(value)) is not None


def _q7_answer(answer: str) -> tuple[bool, bool]:
    normalized = _normalized_text(answer)
    doses = re.findall(
        r"(?<![\w.-])(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\b",
        normalized,
    )
    dose_ok = bool(doses) and all(float(value) == 10.0 for value in doses)
    once_daily = re.search(
        r"\b(?:once\s+(?:a\s+day|daily)|q\.?d\.?|1\s*(?:time|times|x)\s*(?:a|per)?\s*day)\b",
        normalized,
    )
    unsupported_frequency = re.search(
        r"\b(?:twice\s+(?:a\s+day|daily)|b\.?i\.?d\.?|[2-9]\s*(?:times|x)\s*(?:a|per)?\s*day)\b",
        normalized,
    )
    no_unsupported_quantity = (
        dose_ok and unsupported_frequency is None and not _has_contradiction(normalized)
    )
    return dose_ok and once_daily is not None and no_unsupported_quantity, no_unsupported_quantity


_Q2_WEEK = re.compile(r"\b(\d+)\s*(?:weeks?|wks?)\b")
_Q2_RNA = re.compile(r"\b(?:hiv(?:\s+rna)?|rna|viral\s+(?:suppression|load))\b")
_Q2_REGIMEN = re.compile(
    r"\b(?:stable\s+(?:(?:drug(?:/dose)?|dose(?:/drug)?|antiretroviral)\s+)?regimen|"
    r"drug(?:/|\s+)dose\s+regimen)\b"
)
_Q2_CLAUSE_BOUNDARY = re.compile(r"\s*(?:[;\n]|\.(?=\s|$)|\b(?:while|whereas)\b)\s*")
_Q2_INVALID_UPPER_BOUND = re.compile(
    r"(?:<=|<|\bless\s+than\b|\bat\s+most\b|\bno\s+more\s+than\b)\s*"
    r"(?:4|12)\s*(?:weeks?|wks?)\b"
)


def _nearest_q2_week_values(clause: str, marker: re.Match[str]) -> set[int]:
    weeks = list(_Q2_WEEK.finditer(clause))
    if not weeks:
        return set()

    def score(week: re.Match[str]) -> tuple[int, int, int]:
        if week.end() <= marker.start():
            gap = clause[week.end() : marker.start()]
            direction = 1
        elif marker.end() <= week.start():
            gap = clause[marker.end() : week.start()]
            direction = 0
        else:
            gap = ""
            direction = 0
        relation = re.search(r"(?:\b(?:for|of|is|was|were|requires?|required)\b|[:=])", gap)
        return (0 if relation else 1, len(gap), direction)

    nearest = min(score(week) for week in weeks)
    return {int(week.group(1)) for week in weeks if score(week) == nearest}


def _q2_duration_disjunction(normalized: str, left: re.Match[str], right: re.Match[str]) -> bool:
    bridge = normalized[left.end() : right.start()]
    after_left = re.match(r"\s*[\(\[\{,;:/-]*\s*\bor\b", bridge)
    before_right = re.search(
        r"\bor\b\s*(?:[\)\]\},;:/-]\s*)*"
        r"(?:(?:at\s+(?:least|most)|no\s+more\s+than|less\s+than|[<>]=?)\s*)?\Z",
        bridge,
    )
    return after_left is not None or before_right is not None


def _q2_answer(answer: str) -> tuple[bool, bool]:
    normalized = _normalized_text(answer)
    week_matches = list(_Q2_WEEK.finditer(normalized))
    explicit_week_values = [int(match.group(1)) for match in week_matches]
    exact_week_multiset = Counter(explicit_week_values) == Counter({12: 1, 4: 1})
    disjunctive_assignment = any(
        _q2_duration_disjunction(normalized, left, right)
        for left, right in zip(week_matches, week_matches[1:], strict=False)
    )
    invalid_upper_bound = _Q2_INVALID_UPPER_BOUND.search(normalized) is not None

    has_rna_label = _Q2_RNA.search(normalized) is not None
    has_regimen_label = _Q2_REGIMEN.search(normalized) is not None
    rna_values: set[int] = set()
    regimen_values: set[int] = set()
    for clause in _Q2_CLAUSE_BOUNDARY.split(normalized):
        for marker in _Q2_RNA.finditer(clause):
            rna_values.update(_nearest_q2_week_values(clause, marker))
        for marker in _Q2_REGIMEN.finditer(clause):
            regimen_values.update(_nearest_q2_week_values(clause, marker))

    if has_rna_label or has_regimen_label:
        assignment_ok = (
            has_rna_label and has_regimen_label and rna_values == {12} and regimen_values == {4}
        )
        reversed_field_assignment = bool((rna_values - {12}) or (regimen_values - {4}))
    else:
        # Preserve the reviewed terse answer contract: without field labels,
        # the two values must retain the dataset's declared order.
        assignment_ok = (
            explicit_week_values == [12, 4]
            and re.search(r"\brespectively\b", normalized) is not None
        )
        reversed_field_assignment = explicit_week_values == [4, 12]

    no_unsupported_quantity = (
        exact_week_multiset
        and not invalid_upper_bound
        and not disjunctive_assignment
        and not reversed_field_assignment
        and not _has_contradiction(normalized)
    )
    return assignment_ok and no_unsupported_quantity, no_unsupported_quantity


def _q5_answer(answer: str) -> tuple[bool, bool]:
    normalized = _normalized_text(answer)
    dose_values = [int(value) for value in re.findall(r"\bdose\s*(\d+)\b", normalized)]
    expected = 6 in dose_values
    # Mentioning treatment dose 4 as context is supported by the question.
    no_unsupported_quantity = (
        bool(dose_values) and not (set(dose_values) - {4, 6}) and not _has_contradiction(normalized)
    )
    return expected and no_unsupported_quantity, no_unsupported_quantity


def _q7_payload(payload: str) -> bool:
    normalized = _normalized_text(payload)
    # A safety response legitimately contains many dose-escalation groups.  The
    # mechanism proof needs one self-contained group that affirmatively states
    # the reviewed starting regimen; quantities in sibling groups are not a
    # contradiction.
    groups = re.findall(r"\{[^{}]{1,2000}\}", normalized)
    candidates = groups or [normalized]
    return any(all(_q7_answer(candidate)) for candidate in candidates)


def _q2_payload(payload: str) -> bool:
    normalized = _normalized_text(payload)
    rna_marker = re.search(r"(?:hiv(?:\s+rna)?|rna|viral\s+suppression)", normalized)
    regimen_marker = re.search(
        r"(?:stable\s+(?:drug|dose|regimen)|drug(?:/|\s+)dose\s+regimen|"
        r"stable\s+antiretroviral\s+regimen)",
        normalized,
    )
    if rna_marker is None or regimen_marker is None or rna_marker.start() >= regimen_marker.start():
        return False
    rna_segment = normalized[rna_marker.start() : regimen_marker.start()]
    regimen_end = min(len(normalized), regimen_marker.start() + 240)
    for delimiter in (";", ".", r"\n", '"'):
        position = normalized.find(delimiter, regimen_marker.end())
        if position != -1:
            regimen_end = min(regimen_end, position)
    regimen_segment = normalized[regimen_marker.start() : regimen_end]
    rna_values = {int(value) for value in re.findall(r"(\d+)\s*(?:weeks?|wks?)\b", rna_segment)}
    regimen_values = {
        int(value) for value in re.findall(r"(\d+)\s*(?:weeks?|wks?)\b", regimen_segment)
    }
    return bool(
        not _has_contradiction(rna_segment)
        and not _has_contradiction(regimen_segment)
        and rna_values == {12}
        and regimen_values == {4}
        and re.search(
            r"(?:hiv(?:\s+rna)?|rna|viral\s+suppression).{0,240}?"
            r"(?:>=|at\s+least|minimum(?:\s+of)?)?\s*12\s*(?:weeks?|wks?)\b",
            normalized,
        )
        and re.search(
            r"(?:stable\s+(?:drug|dose|regimen)|drug(?:/|\s+)dose\s+regimen|"
            r"stable\s+antiretroviral\s+regimen).{0,240}?"
            r"(?:>=|at\s+least|minimum(?:\s+of)?)?\s*4\s*(?:weeks?|wks?)\b",
            normalized,
        )
    )


def _q5_payload(payload: str) -> bool:
    normalized = _normalized_text(payload)
    if not re.search(r"\b(?:anti[- ]?drug\s+antibod(?:y|ies)|ada)\b", normalized):
        return False
    explicit_doses = {int(value) for value in re.findall(r"\bdose\s*(\d+)\b", normalized)}
    if explicit_doses - {1, 2, 4, 6}:
        return False
    support = re.search(r"\bdose\s*6\b", normalized) or re.search(
        r"\beven[- ]numbered\s+doses?\s+after\s+(?:dose\s*)?d?2\b",
        normalized,
    )
    if support is None:
        return False
    local_support = normalized[
        max(0, support.start() - 80) : min(len(normalized), support.end() + 80)
    ]
    return not _has_contradiction(local_support)


MECHANISM_SPECS = {
    2: MechanismSpec(
        ordinal=2,
        ideal="12, 4",
        nct_id="NCT03249792",
        operation="get_clinical_trial_eligibility_criteria",
        answer_check=_q2_answer,
        payload_check=_q2_payload,
    ),
    5: MechanismSpec(
        ordinal=5,
        ideal="Dose 6",
        nct_id="NCT01693562",
        operation="get_clinical_trial_outcome_measures",
        answer_check=_q5_answer,
        payload_check=_q5_payload,
    ),
    7: MechanismSpec(
        ordinal=7,
        ideal="10, 1",
        nct_id="NCT01970865",
        operation="extract_clinical_trial_adverse_events",
        answer_check=_q7_answer,
        payload_check=_q7_payload,
    ),
}


def _successful_child_payloads(
    item: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    result = item.get("result")
    if not isinstance(result, Mapping):
        return ()
    candidates: list[object] = []
    if result.get("structured_content") is not None:
        candidates.append(result["structured_content"])
    content = result.get("content")
    if isinstance(content, list):
        for raw in content:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("text"), str):
                continue
            text = cast(str, raw["text"])
            try:
                candidates.append(json.loads(text))
            except json.JSONDecodeError:
                # Codex can retain a large ToolUniverse result with an explicit
                # truncation suffix.  The immutable MCP item still proves
                # completion; accept the raw payload only when its child status
                # is unambiguously the first JSON field.
                if re.match(r'^\s*\{\s*"status"\s*:\s*"success"\s*,', text):
                    candidates.append({"status": "success", "truncated_text": text})
    if result.get("status") is not None:
        candidates.append(result)
    return tuple(
        cast(Mapping[str, object], candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("status") == "success"
    )


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold()) if isinstance(value, str) else ""


def _declared_nct_ids(value: Mapping[str, object]) -> set[str]:
    return {
        item
        for key, item in value.items()
        if _normalized_key(key) == "nctid" and isinstance(item, str)
    }


def _records_for_nct(value: object, nct_id: str) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            declared = _declared_nct_ids(item)
            if declared:
                if declared == {nct_id}:
                    records.append(item)
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(records)


def _walk_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    values: list[Mapping[str, object]] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            values.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(values)


def _relevant_record_texts(record: Mapping[str, object], spec: MechanismSpec) -> tuple[str, ...]:
    texts: list[str] = []
    if spec.ordinal == 2:
        for mapping in _walk_mappings(record):
            for key, value in mapping.items():
                if _normalized_key(key) == "eligibilitycriteria":
                    texts.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    elif spec.ordinal == 5:
        for mapping in _walk_mappings(record):
            reviewed_fields = {
                key: value
                for key, value in mapping.items()
                if _normalized_key(key) in {"measure", "title", "timeframe"}
            }
            serialized = json.dumps(reviewed_fields, sort_keys=True, ensure_ascii=False)
            if reviewed_fields and re.search(
                r"\b(?:anti[- ]?drug\s+antibod(?:y|ies)|ada)\b",
                _normalized_text(serialized),
            ):
                texts.append(serialized)
    else:
        for mapping in _walk_mappings(record):
            for key, value in mapping.items():
                if _normalized_key(key) in {"groups", "eventgroups"}:
                    if isinstance(value, list):
                        texts.extend(
                            json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value
                        )
                    else:
                        texts.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return tuple(texts)


def _truncated_relevant_text(payload: Mapping[str, object], spec: MechanismSpec) -> str | None:
    raw = payload.get("truncated_text")
    if not isinstance(raw, str):
        return None
    declared_ids = set(
        re.findall(
            r'"(?:NCT[ _-]?ID|nct_id|nctId)"\s*:\s*"(NCT\d{8})"',
            raw,
            flags=re.IGNORECASE,
        )
    )
    if declared_ids != {spec.nct_id}:
        return None
    field_pattern = {
        2: r'"eligibility_criteria"\s*:',
        5: r'"(?:primary_outcomes|secondary_outcomes|outcomes)"\s*:',
        7: r'"(?:groups|eventGroups)"\s*:',
    }[spec.ordinal]
    field = re.search(field_pattern, raw, flags=re.IGNORECASE)
    if field is None:
        return None
    return raw[field.start() :]


def _payload_supports(payload: Mapping[str, object], spec: MechanismSpec) -> bool:
    truncated = _truncated_relevant_text(payload, spec)
    if truncated is not None:
        return spec.payload_check(truncated)
    return any(
        spec.payload_check(text)
        for record in _records_for_nct(payload, spec.nct_id)
        for text in _relevant_record_texts(record, spec)
    )


def _arguments_target_expected_trial(arguments: Mapping[str, object], spec: MechanismSpec) -> bool:
    encoded = arguments.get("arguments_json")
    if not isinstance(encoded, str):
        return False
    try:
        parsed: object = json.loads(encoded)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, Mapping) and parsed.get("nct_ids") == [spec.nct_id]


def _direct_support(
    events: Sequence[Mapping[str, object]], spec: MechanismSpec
) -> tuple[bool, bool, tuple[str, ...]]:
    direct_observed = False
    payload_supported = False
    payload_hashes: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments")
        if (
            item.get("type") != "mcp_tool_call"
            or item.get("server") != "tooluniverse"
            or item.get("tool") != "execute_tool"
            or item.get("status") != "completed"
            or item.get("error") is not None
            or not isinstance(arguments, Mapping)
            or arguments.get("tool_name") != spec.operation
            or not _arguments_target_expected_trial(arguments, spec)
        ):
            continue
        payloads = _successful_child_payloads(item)
        if not payloads:
            continue
        direct_observed = True
        for payload in payloads:
            serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            payload_hashes.append(f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}")
            payload_supported = payload_supported or _payload_supports(payload, spec)
    return direct_observed, payload_supported, tuple(sorted(set(payload_hashes)))


def _validate_content_addressed_manifest_id(manifest: Mapping[str, object]) -> str:
    kind = manifest.get("kind")
    supplied_manifest_id = manifest.get("manifest_id")
    seed = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected_manifest_id = (
        f"trialqa-{kind}-{hashlib.sha256(demo._canonical_json(seed)).hexdigest()[:20]}"
    )
    if supplied_manifest_id != expected_manifest_id:
        raise TrialQARegressionError("regression manifest ID is not content-addressed")
    return expected_manifest_id


def _validate_manifest_and_dataset(
    manifest: Mapping[str, object], dataset: TrialQADataset
) -> tuple[tuple[TrialQARow, ...], Mapping[str, Mapping[str, object]]]:
    _validate_content_addressed_manifest_id(manifest)
    kind = manifest.get("kind")
    try:
        demo.validate_manifest_pairing(manifest)
    except demo.TrialQADemoError as exc:
        raise TrialQARegressionError("regression manifest pairing is invalid") from exc
    protocol = manifest.get("protocol")
    if (
        kind != "full"
        or not isinstance(protocol, Mapping)
        or protocol.get("primary_evaluation_scope") is not None
        or protocol.get("performance_eligible") is not False
    ):
        raise TrialQARegressionError(
            "regression checks require a descriptive nonperformance full manifest"
        )
    candidate = manifest.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != {"candidate_id", "manifest_sha256", "skill_sha256"}
        or not isinstance(candidate.get("candidate_id"), str)
        or not candidate["candidate_id"]
        or not all(
            isinstance(candidate.get(field), str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", cast(str, candidate[field]))
            for field in ("manifest_sha256", "skill_sha256")
        )
    ):
        raise TrialQARegressionError("regression manifest candidate attestation is invalid")
    dataset_binding = manifest.get("dataset")
    split = create_split_manifest(dataset)
    if (
        not isinstance(dataset_binding, Mapping)
        or dataset_binding.get("id") != TRIALQA_DATASET_ID
        or dataset_binding.get("config") != TRIALQA_DATASET_CONFIG
        or dataset_binding.get("revision") != TRIALQA_DATASET_REVISION
        or dataset_binding.get("parquet_sha256") != dataset.parquet_sha256
        or dataset_binding.get("split_manifest_sha256")
        != demo._sha256_bytes(demo._canonical_json(split))
    ):
        raise TrialQARegressionError("regression manifest dataset binding is invalid")
    assignments = {
        cast(str, row["row_id"]): cast(str, row["partition"])
        for row in cast(list[dict[str, object]], split["rows"])
    }
    heldout = tuple(row for row in dataset.rows if assignments[row.id] == "test")
    tasks_raw = manifest.get("tasks")
    assert isinstance(tasks_raw, list)  # Established by validate_manifest_pairing.
    tasks = [cast(Mapping[str, object], raw) for raw in tasks_raw]
    group_order = tuple(dict.fromkeys(cast(str, task["question_group_key"]) for task in tasks))
    if group_order != tuple(question_group_key(row) for row in heldout):
        raise TrialQARegressionError("regression manifest held-out order differs from pinned data")
    by_id = {cast(str, task["task_id"]): task for task in tasks}
    return heldout, by_id


def _validate_generation_identity(
    generation: demo.GenerationResult,
    task: Mapping[str, object],
    *,
    manifest_id: str,
    row: TrialQARow,
) -> None:
    expected = {
        "manifest_id": manifest_id,
        "task_id": task.get("task_id"),
        "pair_id": task.get("pair_id"),
        "row_id": row.id,
        "dataset_row_index": row.dataset_row_index,
        "partition": "test",
        "condition": "treatment",
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
    }
    mismatches = [field for field, value in expected.items() if getattr(generation, field) != value]
    if mismatches:
        raise TrialQARegressionError(
            f"generation identity differs from manifest/pinned row at {mismatches[0]}"
        )


def _validate_manifest_session_attestation(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    capture: Path,
    generation: demo.GenerationResult,
) -> None:
    try:
        context, active, launch_sha256 = batch._launch_metadata(manifest, task, capture)
        batch._validate_session_proof(
            session_dir=generation.session_dir,
            expected_context=context,
            expected_active=active,
            launch_sha256=launch_sha256,
        )
    except (batch.SessionProofError, demo.TrialQADemoError, OSError) as exc:
        raise TrialQARegressionError(
            f"generation candidate/session attestation is invalid for {generation.task_id}"
        ) from exc


def _mechanism_spec_for_task(
    task: Mapping[str, object], group_order: Sequence[str]
) -> MechanismSpec:
    task_id = str(task.get("task_id"))
    group = task.get("question_group_key")
    if not isinstance(group, str):
        raise TrialQARegressionError(f"task has no pinned held-out ordinal: {task_id}")
    try:
        ordinal = group_order.index(group)
    except ValueError as exc:
        raise TrialQARegressionError(f"task has no pinned held-out ordinal: {task_id}") from exc
    spec = MECHANISM_SPECS.get(ordinal)
    if spec is None or task.get("condition") != "treatment":
        raise TrialQARegressionError(f"only q2/q5/q7 treatment tasks may be checked: {task_id}")
    return spec


def _evaluate_generation(
    generation: demo.GenerationResult,
    spec: MechanismSpec,
) -> dict[str, object]:
    try:
        demo._parse_codex_events(
            generation.codex_events_path,
            require_skill_load=True,
            enforce_tool_policy=True,
        )
        metrics = demo.codex_tool_metrics(generation.codex_events_path)
        events = demo.read_codex_events(generation.codex_events_path)
    except demo.TrialQADemoError as exc:
        raise TrialQARegressionError("Codex event evidence is invalid") from exc
    answer_ok, no_unsupported_quantity = spec.answer_check(generation.answer)
    direct_observed, payload_supported, payload_hashes = _direct_support(events, spec)
    checks = {
        "normalized_answer": answer_ok,
        "no_unsupported_quantitative_value": no_unsupported_quantity,
        "direct_successful_operation": direct_observed,
        "supporting_payload": payload_supported,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "task_id": generation.task_id,
        "question_ordinal": spec.ordinal,
        "repeat_index": generation.repeat_index,
        "condition": generation.condition,
        "decision": "pass" if not reasons else "kill",
        "checks": checks,
        "kill_reasons": reasons,
        "expected_operation": spec.operation,
        "supporting_payload_sha256": list(payload_hashes),
        "operational_calls": metrics["operational_calls"],
        "bindings": {
            "generation_sha256": demo._sha256_file(generation.generation_path),
            "codex_events_sha256": demo._sha256_file(generation.codex_events_path),
            "row_id": generation.row_id,
            "dataset_row_index": generation.dataset_row_index,
        },
    }


def check_treatment_tasks(
    *,
    manifest: Mapping[str, Any],
    dataset: TrialQADataset,
    capture: Path,
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Validate and check explicit q2/q5/q7 treatment generation tasks."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise TrialQARegressionError("regression task IDs must be nonempty and unique")
    heldout, tasks = _validate_manifest_and_dataset(manifest, dataset)
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise TrialQARegressionError("regression manifest ID is invalid")
    capture = capture.resolve(strict=True)
    ledger = demo.ResumableLedger(capture / "ledger.jsonl", manifest)
    results: list[dict[str, object]] = []
    group_order = tuple(question_group_key(row) for row in heldout)
    for task_id in task_ids:
        task = tasks.get(task_id)
        if task is None:
            raise TrialQARegressionError(f"task is absent from the manifest: {task_id}")
        spec = _mechanism_spec_for_task(task, group_order)
        row = heldout[spec.ordinal]
        if row.ideal != spec.ideal or task.get("row_id") != row.id:
            raise TrialQARegressionError(
                f"task differs from pinned q{spec.ordinal} gold: {task_id}"
            )
        try:
            generation = batch._load_completed_generation(ledger, task_id)
            demo.validate_generation_for_import(generation, project_dir=capture)
        except (RuntimeError, demo.TrialQADemoError) as exc:
            raise TrialQARegressionError(
                f"immutable generation evidence is invalid for {task_id}"
            ) from exc
        _validate_generation_identity(
            generation,
            task,
            manifest_id=manifest_id,
            row=row,
        )
        for artifact in (
            generation.generation_path,
            generation.codex_events_path,
            generation.final_output_path,
            generation.stats_path,
            generation.trajectory_path,
            generation.session_dir,
        ):
            if not artifact.resolve(strict=True).is_relative_to(capture):
                raise TrialQARegressionError(
                    f"generation artifact escapes the capture for {task_id}"
                )
        _validate_manifest_session_attestation(
            manifest=manifest,
            task=task,
            capture=capture,
            generation=generation,
        )
        results.append(_evaluate_generation(generation, spec))

    passed = sum(result["decision"] == "pass" for result in results)
    report: dict[str, object] = {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "policy": {
            "name": REGRESSION_POLICY,
            "performance_eligible": False,
            "allowed_question_ordinals": sorted(MECHANISM_SPECS),
            "condition": "treatment",
            "model_calls": 0,
            "judge_calls": 0,
            "evidence_imports": 0,
        },
        "manifest_id": manifest_id,
        "manifest_sha256": _canonical_sha256(manifest),
        "dataset": {
            "id": TRIALQA_DATASET_ID,
            "revision": dataset.revision,
            "parquet_sha256": dataset.parquet_sha256,
        },
        "results": results,
        "summary": {
            "checked_tasks": len(results),
            "passed_tasks": passed,
            "killed_tasks": len(results) - passed,
            "decision": "pass" if passed == len(results) else "kill",
        },
    }
    return {**report, "report_sha256": _canonical_sha256(report)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = demo._read_json_object(args.manifest.resolve(strict=True), "experiment manifest")
    dataset = load_pinned_trialqa_parquet(args.dataset.resolve(strict=True))
    report = check_treatment_tasks(
        manifest=manifest,
        dataset=dataset,
        capture=args.capture,
        task_ids=args.task_id,
    )
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    demo._write_json_atomic(output, report, exclusive=True)
    print(json.dumps(report, sort_keys=True))
    if cast(Mapping[str, object], report["summary"])["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
