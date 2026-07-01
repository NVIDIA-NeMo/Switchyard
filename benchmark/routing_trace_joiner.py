# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join per-request Switchyard routing traces to Harbor trial results."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple, cast

SCHEMA_VERSION = 1
DEFAULT_JOINED_NAME = "routing_trace_joined.jsonl"
DEFAULT_REPORT_NAME = "routing_trace_completeness.json"

JsonObject = dict[str, Any]


class LoadContext(NamedTuple):
    job_dir: Path
    malformed_rows: list[JsonObject]
    input_errors: list[JsonObject]


class TrialRecord(NamedTuple):
    trial: JsonObject
    task: JsonObject
    outcome: JsonObject
    request_map_path: Path


class MappingRecord(NamedTuple):
    request_id: str
    mapping: JsonObject
    trial_record: TrialRecord
    path: str
    line: int


class TraceRecord(NamedTuple):
    event: JsonObject
    line: int


def _display_path(path: Path, job_dir: Path) -> str:
    try:
        return path.resolve().relative_to(job_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _issue(source: str, path: str, error: str, *, line: int | None = None) -> JsonObject:
    issue: JsonObject = {"source": source, "path": path, "error": error}
    if line is not None:
        issue["line"] = line
    return issue


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _trial_sort_key(record: TrialRecord) -> tuple[str, str, str, str]:
    return (
        str(record.task.get("source", "")),
        str(record.task["name"]),
        str(record.trial["name"]),
        str(record.trial.get("id", "")),
    )


def _missing_map(record: TrialRecord, path: str, reason: str) -> JsonObject:
    return {"trial": record.trial, "task": record.task, "path": path, "reason": reason}


def _trial_outcome(result: JsonObject) -> JsonObject:
    """Extract the task-level ability signal without copying error content."""
    verifier_result = result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    try:
        numeric_reward = (
            float(reward)
            if isinstance(reward, int | float) and not isinstance(reward, bool)
            else None
        )
    except OverflowError:
        numeric_reward = None
    if numeric_reward is not None and not math.isfinite(numeric_reward):
        numeric_reward = None
    if result.get("exception_info") is not None:
        status = "error"
    elif numeric_reward is None:
        status = "unknown"
    elif numeric_reward == 1.0:
        status = "pass"
    else:
        status = "fail"
    return {"status": status, "reward": numeric_reward}


def _load_trials(context: LoadContext) -> tuple[list[TrialRecord], int]:
    job_dir = context.job_dir
    result_paths = sorted(job_dir.glob("*/result.json"))
    trials: list[TrialRecord] = []

    for result_path in result_paths:
        display_path = _display_path(result_path, job_dir)
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("result must be a JSON object")
            result = cast(JsonObject, raw)
            trial_name = _nonempty_string(result.get("trial_name"), "trial_name")
            task_name = _nonempty_string(result.get("task_name"), "task_name")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            context.input_errors.append(_issue("result", display_path, str(exc)))
            continue

        trial: JsonObject = {"name": trial_name}
        if result.get("id") is not None:
            trial["id"] = result["id"]
        if result.get("trial_uri") is not None:
            trial["uri"] = result["trial_uri"]

        task: JsonObject = {"name": task_name}
        if result.get("task_id") is not None:
            task["id"] = result["task_id"]
        if result.get("source") is not None:
            task["source"] = result["source"]

        trials.append(
            TrialRecord(
                trial=trial,
                task=task,
                outcome=_trial_outcome(result),
                request_map_path=result_path.parent / "artifacts" / "request_map.jsonl",
            )
        )

    trials.sort(key=_trial_sort_key)
    return trials, len(result_paths)


def _read_jsonl(
    path: Path,
    source: str,
    context: LoadContext,
) -> Iterator[tuple[int, JsonObject]]:
    display_path = _display_path(path, context.job_dir)
    try:
        source_file = path.open(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        context.input_errors.append(_issue(source, display_path, str(exc)))
        return

    with source_file:
        try:
            rows = enumerate(source_file, start=1)
            for line_number, line in rows:
                if not line.strip():
                    context.malformed_rows.append(
                        _issue(source, display_path, "blank row", line=line_number)
                    )
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    context.malformed_rows.append(
                        _issue(source, display_path, "invalid JSON", line=line_number)
                    )
                    continue
                if not isinstance(raw, dict):
                    context.malformed_rows.append(
                        _issue(
                            source,
                            display_path,
                            "row must be a JSON object",
                            line=line_number,
                        )
                    )
                    continue
                yield line_number, cast(JsonObject, raw)
        except (OSError, UnicodeError) as exc:
            context.input_errors.append(_issue(source, display_path, str(exc)))


def _load_mappings(
    trials: list[TrialRecord],
    context: LoadContext,
) -> tuple[list[MappingRecord], list[JsonObject]]:
    mappings: list[MappingRecord] = []
    missing_maps: list[JsonObject] = []

    for trial in trials:
        map_path = trial.request_map_path
        display_path = _display_path(map_path, context.job_dir)
        if not map_path.is_file():
            missing_maps.append(_missing_map(trial, display_path, "not_found"))
            continue

        valid_for_trial = 0
        for line_number, row in _read_jsonl(map_path, "request_map", context):
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                context.malformed_rows.append(
                    _issue(
                        "request_map",
                        display_path,
                        "request_id must be a non-empty string",
                        line=line_number,
                    )
                )
                continue
            mappings.append(
                MappingRecord(
                    request_id=request_id,
                    mapping=row,
                    trial_record=trial,
                    path=display_path,
                    line=line_number,
                )
            )
            valid_for_trial += 1

        if valid_for_trial == 0:
            missing_maps.append(_missing_map(trial, display_path, "no_valid_rows"))

    mappings.sort(key=_mapping_sort_key)
    return mappings, missing_maps


def _load_traces(
    trace_path: Path,
    context: LoadContext,
) -> tuple[dict[str, list[TraceRecord]], int, set[str]]:
    traces: dict[str, list[TraceRecord]] = {}
    invalid_request_ids: set[str] = set()
    if not trace_path.is_file():
        context.input_errors.append(
            _issue("routing_trace", _display_path(trace_path, context.job_dir), "file not found")
        )
        return traces, 0, invalid_request_ids

    valid_rows = 0
    display_path = _display_path(trace_path, context.job_dir)
    for line_number, row in _read_jsonl(trace_path, "routing_trace", context):
        request_id = row.get("request_id")
        schema_version = row.get("schema_version")
        event = row.get("event")
        sequence = event.get("sequence") if isinstance(event, dict) else None
        error: str | None = None
        if not isinstance(request_id, str) or not request_id.strip():
            error = "request_id must be a non-empty string"
        elif isinstance(schema_version, bool) or not isinstance(schema_version, int):
            error = "schema_version must be an integer"
        elif schema_version != SCHEMA_VERSION:
            error = f"unsupported schema_version {schema_version}"
        elif not isinstance(event, dict):
            error = "event must be a JSON object"
        elif isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            error = "event.sequence must be a non-negative integer"

        if error is not None:
            if isinstance(request_id, str) and request_id.strip():
                invalid_request_ids.add(request_id)
            context.malformed_rows.append(
                _issue("routing_trace", display_path, error, line=line_number)
            )
            continue

        assert isinstance(request_id, str)
        assert isinstance(event, dict)
        traces.setdefault(request_id, []).append(
            TraceRecord(
                event=cast(JsonObject, event),
                line=line_number,
            )
        )
        valid_rows += 1

    for records in traces.values():
        records.sort(key=lambda record: (record.event["sequence"], record.line))
    return traces, valid_rows, invalid_request_ids


def _mapping_sort_key(record: MappingRecord) -> tuple[str, str, str, str, str, int]:
    return (*_trial_sort_key(record.trial_record), record.request_id, record.line)


def _mapping_reference(record: MappingRecord) -> JsonObject:
    return {
        "trial": record.trial_record.trial,
        "task": record.trial_record.task,
        "outcome": record.trial_record.outcome,
        "path": record.path,
        "line": record.line,
    }


def _mapping_group_issue(request_id: str, records: list[MappingRecord]) -> JsonObject:
    return {
        "request_id": request_id,
        "mappings": [_mapping_reference(record) for record in records],
    }


def _write_private_text(path: Path, content: str) -> None:
    """Write a private artifact that may contain benchmark request content."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700)
    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)
    path.chmod(0o600)


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    _write_private_text(path, content)


def join_routing_traces(
    job_dir: Path,
    routing_trace_path: Path,
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> JsonObject:
    """Join Harbor request mappings to routing events and write completeness metadata."""
    job_dir = job_dir.resolve()
    if not job_dir.is_dir():
        raise NotADirectoryError(f"Harbor job directory does not exist: {job_dir}")
    routing_trace_path = routing_trace_path.resolve()
    output_path = (output_path or job_dir / DEFAULT_JOINED_NAME).resolve()
    report_path = (report_path or job_dir / DEFAULT_REPORT_NAME).resolve()
    if len({routing_trace_path, output_path, report_path}) != 3:
        raise ValueError("routing trace input, joined output, and report paths must be distinct")

    context = LoadContext(job_dir=job_dir, malformed_rows=[], input_errors=[])
    trials, result_file_count = _load_trials(context)
    if result_file_count == 0:
        context.input_errors.append(
            _issue("job", str(job_dir), "no per-trial */result.json files found")
        )

    mappings, missing_maps = _load_mappings(trials, context)
    traces, trace_row_count, malformed_trace_request_ids = _load_traces(routing_trace_path, context)

    mappings_by_id: dict[str, list[MappingRecord]] = {}
    for mapping in mappings:
        mappings_by_id.setdefault(mapping.request_id, []).append(mapping)

    duplicate_mapped_ids = [
        _mapping_group_issue(request_id, records)
        for request_id, records in sorted(mappings_by_id.items())
        if len(records) > 1
    ]
    missing_traces = [
        _mapping_group_issue(request_id, records)
        for request_id, records in sorted(mappings_by_id.items())
        if request_id not in traces
    ]
    orphan_trace_ids = sorted(set(traces) - set(mappings_by_id))
    invalid_trace_sequences = [
        {
            "request_id": request_id,
            "sequences": [record.event["sequence"] for record in records],
        }
        for request_id, records in sorted(traces.items())
        if [record.event["sequence"] for record in records] != list(range(len(records)))
    ]
    invalid_trace_sequence_ids = {str(entry["request_id"]) for entry in invalid_trace_sequences}
    invalid_trace_request_id_set = malformed_trace_request_ids | invalid_trace_sequence_ids
    invalid_trace_request_ids = sorted(invalid_trace_request_id_set)
    missing_outcomes = [
        {
            "trial": trial.trial,
            "task": trial.task,
        }
        for trial in trials
        if trial.outcome["status"] == "unknown"
    ]

    joined_rows: list[JsonObject] = []
    for mapping in mappings:
        request_id = mapping.request_id
        records = mappings_by_id[request_id]
        trace_records = traces.get(request_id)
        if len(records) != 1 or not trace_records or request_id in invalid_trace_request_id_set:
            continue
        trial = mapping.trial_record
        joined_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "trial": trial.trial,
                "task": trial.task,
                "outcome": trial.outcome,
                "request_map": mapping.mapping,
                "routing_trace": {
                    "schema_versions": [SCHEMA_VERSION],
                    "events": [record.event for record in trace_records],
                },
            }
        )

    context.malformed_rows.sort(
        key=lambda row: (str(row["source"]), str(row["path"]), int(row["line"]))
    )
    context.input_errors.sort(key=lambda row: (str(row["source"]), str(row["path"])))

    issues: JsonObject = {
        "malformed_rows": context.malformed_rows,
        "input_errors": context.input_errors,
        "missing_maps": missing_maps,
        "missing_traces": missing_traces,
        "duplicate_mapped_ids": duplicate_mapped_ids,
        "orphan_trace_ids": orphan_trace_ids,
        "invalid_trace_sequences": invalid_trace_sequences,
        "invalid_trace_request_ids": invalid_trace_request_ids,
        "missing_outcomes": missing_outcomes,
    }
    outcome_counts = Counter(str(trial.outcome["status"]) for trial in trials)
    report: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete" if any(issues.values()) else "complete",
        "inputs": {
            "job_dir": str(job_dir),
            "routing_trace": str(routing_trace_path),
        },
        "counts": {
            "result_files": result_file_count,
            "trials": len(trials),
            "mapped_rows": len(mappings),
            "mapped_request_ids": len(mappings_by_id),
            "trace_rows": trace_row_count,
            "trace_request_ids": len(traces),
            "joined_requests": len(joined_rows),
            **{name: len(entries) for name, entries in issues.items()},
            **{
                f"outcome_{status}": outcome_counts[status]
                for status in ("pass", "fail", "error", "unknown")
            },
        },
        **issues,
    }

    _write_jsonl(output_path, joined_rows)
    _write_private_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join Switchyard routing events to Harbor trial request mappings."
    )
    parser.add_argument("--job-dir", type=Path, required=True, help="Harbor job directory.")
    parser.add_argument(
        "--routing-trace",
        type=Path,
        required=True,
        help="Switchyard event-level routing_trace.jsonl.",
    )
    for name, label, default in (
        ("output", "Joined JSONL output", DEFAULT_JOINED_NAME),
        ("report", "Completeness report", DEFAULT_REPORT_NAME),
    ):
        parser.add_argument(f"--{name}", type=Path, help=f"{label} (default: <job-dir>/{default}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the routing trace joiner CLI."""
    args = _parse_args(argv)
    output_path = args.output or args.job_dir / DEFAULT_JOINED_NAME
    report_path = args.report or args.job_dir / DEFAULT_REPORT_NAME
    report = join_routing_traces(
        args.job_dir,
        args.routing_trace,
        output_path=output_path,
        report_path=report_path,
    )
    print(
        f"{report['status']}: joined {report['counts']['joined_requests']} requests; "
        f"report={report_path}"
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
