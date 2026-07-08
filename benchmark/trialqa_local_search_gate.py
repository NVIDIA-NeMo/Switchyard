# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-call TrialQA search-resolution and immediate-evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_regression as regression  # noqa: E402
from benchmark.trialqa_local_dataset import (  # noqa: E402
    TRIALQA_DATASET_ID,
    TrialQADataset,
    load_pinned_trialqa_parquet,
)

SEARCH_GATE_SCHEMA_VERSION = "switchyard.trialqa_search_gate.v1"
SEARCH_GATE_POLICY = "bounded-unique-search-then-evidence-v1"
SEARCH_OPERATION = "ClinicalTrials_search_studies"
MAX_SEARCHES = 3
_NCT_ID = re.compile(r"NCT\d{8}\Z", re.IGNORECASE)
_QUERY_FIELDS = ("query_term", "query_cond", "query_intr")
_RESOLUTION_QUERY_PREFERENCE = ("query_term", "query_intr", "query_cond")

JsonObject = dict[str, Any]


class TrialQASearchGateError(RuntimeError):
    """The requested search gate or its immutable evidence is invalid."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _normalized_query_identity(decoded: Mapping[str, object]) -> str | None:
    values = [
        (field, normalized)
        for field in _QUERY_FIELDS
        if isinstance((value := decoded.get(field)), str)
        and (normalized := _normalize_search_text(value))
    ]
    return json.dumps(values, separators=(",", ":")) if values else None


def _resolution_query(decoded: Mapping[str, object]) -> str | None:
    for field in _RESOLUTION_QUERY_PREFERENCE:
        value = decoded.get(field)
        if isinstance(value, str) and (normalized := _normalize_search_text(value)):
            return normalized
    return None


def _decode_arguments(item: Mapping[str, object]) -> tuple[JsonObject, str] | None:
    arguments = item.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    encoded = arguments.get("arguments_json")
    if not isinstance(encoded, str):
        return None
    try:
        decoded: object = json.loads(encoded)
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    return cast(JsonObject, decoded), canonical


def _successful_execute_calls(
    events: Sequence[Mapping[str, object]],
) -> list[tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]]:
    calls: list[tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if (
            not isinstance(item, Mapping)
            or item.get("type") != "mcp_tool_call"
            or item.get("server") != "tooluniverse"
            or item.get("tool") != "execute_tool"
            or item.get("status") != "completed"
            or item.get("error") is not None
        ):
            continue
        payloads = regression._successful_child_payloads(item)
        if payloads:
            calls.append((item, payloads))
    return calls


def _study_title(study: Mapping[str, object]) -> str | None:
    titles = [
        value.strip()
        for key in ("brief_title", "official_title", "title")
        if isinstance((value := study.get(key)), str) and value.strip()
    ]
    return titles[0] if titles else None


def _search_resolution(
    decoded: Mapping[str, object], payloads: Sequence[Mapping[str, object]]
) -> JsonObject | None:
    normalized_query = _resolution_query(decoded)
    if normalized_query is None:
        return None
    candidates: dict[tuple[str, str], JsonObject] = {}
    for payload in payloads:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            continue
        studies = data.get("studies")
        total_count = data.get("total_count")
        if (
            not isinstance(total_count, int)
            or isinstance(total_count, bool)
            or total_count != 1
            or data.get("next_page_token") is not None
            or not isinstance(studies, list)
            or len(studies) != 1
            or not isinstance(studies[0], Mapping)
        ):
            continue
        study = cast(Mapping[str, object], studies[0])
        nct_id = study.get("nct_id")
        title = _study_title(study)
        if (
            not isinstance(nct_id, str)
            or _NCT_ID.fullmatch(nct_id) is None
            or title is None
            or normalized_query not in _normalize_search_text(title)
        ):
            continue
        normalized_nct = nct_id.upper()
        candidates[(normalized_nct, title)] = {
            "nct_id": normalized_nct,
            "title": title,
            "normalized_query": normalized_query,
        }
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _targets_resolved_nct(item: Mapping[str, object], resolved_nct: str) -> bool:
    decoded = _decode_arguments(item)
    return decoded is not None and decoded[0].get("nct_ids") == [resolved_nct]


def evaluate_search_gate(
    *,
    events: Sequence[Mapping[str, object]],
    semantic_result: Mapping[str, object],
) -> JsonObject:
    """Evaluate ordered successful ToolUniverse calls without any external calls."""

    ordinal = semantic_result.get("question_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise TrialQASearchGateError("semantic result has no valid question ordinal")
    spec = regression.MECHANISM_SPECS.get(ordinal)
    if spec is None:
        raise TrialQASearchGateError("semantic result is outside q2/q5/q7")

    calls = _successful_execute_calls(events)
    searches: list[JsonObject] = []
    for call_index, (item, payloads) in enumerate(calls):
        arguments = item.get("arguments")
        tool_name = arguments.get("tool_name") if isinstance(arguments, Mapping) else None
        if tool_name != SEARCH_OPERATION:
            continue
        decoded = _decode_arguments(item)
        normalized_query = _normalized_query_identity(decoded[0]) if decoded is not None else None
        searches.append(
            {
                "call_index": call_index,
                "search_index": len(searches),
                "canonical_arguments": decoded[1] if decoded is not None else None,
                "normalized_query": normalized_query or None,
                "resolution": (
                    _search_resolution(decoded[0], payloads) if decoded is not None else None
                ),
            }
        )

    canonical_counts = Counter(
        cast(str, search["canonical_arguments"])
        for search in searches
        if isinstance(search.get("canonical_arguments"), str)
    )
    query_counts = Counter(
        cast(str, search["normalized_query"])
        for search in searches
        if isinstance(search.get("normalized_query"), str)
    )
    repeated_arguments = [
        {"canonical_arguments": value, "count": count}
        for value, count in sorted(canonical_counts.items())
        if count > 1
    ]
    repeated_queries = [
        {"normalized_query": value, "count": count}
        for value, count in sorted(query_counts.items())
        if count > 1
    ]
    resolution_search = next(
        (search for search in searches if search.get("resolution") is not None), None
    )
    resolution = (
        cast(JsonObject, resolution_search["resolution"]) if resolution_search is not None else None
    )
    resolution_index = (
        cast(int, resolution_search["search_index"]) if resolution_search is not None else None
    )
    resolution_call_index = (
        cast(int, resolution_search["call_index"]) if resolution_search is not None else None
    )
    post_resolution_search_count = (
        sum(cast(int, search["search_index"]) > resolution_index for search in searches)
        if resolution_index is not None
        else 0
    )
    next_call = (
        calls[resolution_call_index + 1][0]
        if resolution_call_index is not None and resolution_call_index + 1 < len(calls)
        else None
    )
    next_arguments = next_call.get("arguments") if next_call is not None else None
    next_operation = (
        next_arguments.get("tool_name") if isinstance(next_arguments, Mapping) else None
    )
    next_evidence_ok = bool(
        resolution is not None
        and next_call is not None
        and next_operation == spec.operation
        and _targets_resolved_nct(next_call, cast(str, resolution["nct_id"]))
    )
    search_arguments_valid = sum(canonical_counts.values()) == len(searches) and sum(
        query_counts.values()
    ) == len(searches)
    checks = {
        "semantic_replay_passed": semantic_result.get("decision") == "pass",
        "search_arguments_valid": search_arguments_valid,
        "at_most_three_searches": len(searches) <= MAX_SEARCHES,
        "unique_canonical_arguments": not repeated_arguments,
        "unique_normalized_queries": not repeated_queries,
        "unique_title_resolution_found": resolution is not None,
        "no_search_after_first_resolution": post_resolution_search_count == 0,
        "next_call_is_expected_evidence_getter": next_evidence_ok,
    }
    kill_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "task_id": semantic_result.get("task_id"),
        "question_ordinal": ordinal,
        "decision": "pass" if not kill_reasons else "kill",
        "checks": checks,
        "kill_reasons": kill_reasons,
        "semantic_result": dict(semantic_result),
        "expected_operation": spec.operation,
        "successful_execute_tool_count": len(calls),
        "search_count": len(searches),
        "resolution_index": resolution_index,
        "resolution": resolution,
        "repeated_argument_count": sum(
            count - 1 for count in canonical_counts.values() if count > 1
        ),
        "repeated_arguments": repeated_arguments,
        "repeated_normalized_query_count": sum(
            count - 1 for count in query_counts.values() if count > 1
        ),
        "repeated_normalized_queries": repeated_queries,
        "post_resolution_search_count": post_resolution_search_count,
        "next_operation": next_operation,
    }


def build_search_gate_report(
    *,
    manifest: Mapping[str, Any],
    dataset: TrialQADataset,
    capture: Path,
    task_ids: Sequence[str],
) -> JsonObject:
    """Replay semantic and search-policy evidence for explicit treatment tasks."""

    semantic = regression.check_treatment_tasks(
        manifest=manifest,
        dataset=dataset,
        capture=capture,
        task_ids=task_ids,
    )
    semantic_results = cast(list[JsonObject], semantic["results"])
    ledger = demo.ResumableLedger(capture / "ledger.jsonl", manifest)
    results: list[JsonObject] = []
    for task_id, semantic_result in zip(task_ids, semantic_results, strict=True):
        try:
            generation = batch._load_completed_generation(ledger, task_id)
            events = demo.read_codex_events(generation.codex_events_path)
        except (RuntimeError, demo.TrialQADemoError, OSError) as exc:
            raise TrialQASearchGateError(
                f"immutable generation evidence is invalid for {task_id}"
            ) from exc
        results.append(evaluate_search_gate(events=events, semantic_result=semantic_result))

    passed = sum(result["decision"] == "pass" for result in results)
    report: JsonObject = {
        "schema_version": SEARCH_GATE_SCHEMA_VERSION,
        "policy": {
            "name": SEARCH_GATE_POLICY,
            "performance_eligible": False,
            "condition": "treatment",
            "max_searches": MAX_SEARCHES,
            "model_calls": 0,
            "judge_calls": 0,
            "evidence_imports": 0,
            "network_calls": 0,
        },
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": regression._canonical_sha256(manifest),
        "dataset": {
            "id": TRIALQA_DATASET_ID,
            "revision": dataset.revision,
            "parquet_sha256": dataset.parquet_sha256,
        },
        "semantic_report_sha256": semantic["report_sha256"],
        "results": results,
        "summary": {
            "checked_tasks": len(results),
            "passed_tasks": passed,
            "killed_tasks": len(results) - passed,
            "decision": "pass" if passed == len(results) else "kill",
        },
    }
    return {**report, "report_sha256": _canonical_sha256(report)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = demo._read_json_object(args.manifest.resolve(strict=True), "experiment manifest")
        dataset = load_pinned_trialqa_parquet(args.dataset.resolve(strict=True))
        report = build_search_gate_report(
            manifest=manifest,
            dataset=dataset,
            capture=args.capture.resolve(strict=True),
            task_ids=args.task_id,
        )
        output = args.output.absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        demo._write_json_atomic(output, report, exclusive=True)
        print(json.dumps(report, sort_keys=True))
        return 0 if cast(Mapping[str, object], report["summary"])["decision"] == "pass" else 1
    except (
        TrialQASearchGateError,
        regression.TrialQARegressionError,
        demo.TrialQADemoError,
        OSError,
        ValueError,
    ) as exc:
        print(f"trialqa_local_search_gate: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
