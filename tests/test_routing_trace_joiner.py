# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
JOINER = REPO / "benchmark" / "routing_trace_joiner.py"


def _load_joiner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "switchyard_benchmark_routing_trace_joiner", JOINER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_trial(
    job_dir: Path,
    directory_name: str,
    *,
    trial_name: str,
    task_name: str,
    request_rows: list[object] | None,
) -> Path:
    trial_dir = job_dir / directory_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": f"id-{trial_name}",
                "trial_name": trial_name,
                "trial_uri": f"trial://{trial_name}",
                "task_name": task_name,
                "task_id": {"name": task_name},
                "source": "terminal-bench",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    if request_rows is not None:
        _write_jsonl(trial_dir / "artifacts" / "request_map.jsonl", request_rows)
    return trial_dir


def _trace_row(request_id: str, sequence: int, name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "event": {
            "sequence": sequence,
            "timestamp_ms": 1000 + sequence,
            "kind": "decision",
            "producer": "test",
            "name": name,
            "selection": {"tier": "strong"},
            "source": "test",
        },
    }


def test_trial_outcome_extracts_ability_without_error_content() -> None:
    module = _load_joiner()

    assert module._trial_outcome({"verifier_result": {"rewards": {"reward": 1.0}}}) == {
        "status": "pass",
        "reward": 1.0,
    }
    assert module._trial_outcome({"verifier_result": {"rewards": {"reward": 0.25}}}) == {
        "status": "fail",
        "reward": 0.25,
    }
    assert module._trial_outcome({"exception_info": "sensitive traceback"}) == {
        "status": "error",
        "reward": None,
    }
    assert module._trial_outcome({"verifier_result": {"rewards": {"reward": 10**10_000}}}) == {
        "status": "unknown",
        "reward": None,
    }
    assert module._trial_outcome({}) == {"status": "unknown", "reward": None}


def test_joiner_uses_result_identity_and_orders_rows_and_events(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    _write_trial(
        job_dir,
        "directory-name-must-not-be-used",
        trial_name="trial-from-result",
        task_name="task-from-result",
        request_rows=[
            {"request_id": "request-b", "turn": 2},
            {"request_id": "request-a", "turn": 1},
        ],
    )
    trace_path = tmp_path / "routing_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            _trace_row("request-b", 2, "last"),
            _trace_row("request-a", 0, "only"),
            _trace_row("request-b", 0, "first"),
            _trace_row("request-b", 1, "middle"),
        ],
    )
    output_path = tmp_path / "joined.jsonl"
    report_path = tmp_path / "report.json"

    report = module.join_routing_traces(
        job_dir,
        trace_path,
        output_path=output_path,
        report_path=report_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["request_id"] for row in rows] == ["request-a", "request-b"]
    assert rows[0]["trial"] == {
        "id": "id-trial-from-result",
        "name": "trial-from-result",
        "uri": "trial://trial-from-result",
    }
    assert rows[0]["task"] == {
        "id": {"name": "task-from-result"},
        "name": "task-from-result",
        "source": "terminal-bench",
    }
    assert rows[0]["outcome"] == {"status": "pass", "reward": 1.0}
    assert [event["name"] for event in rows[1]["routing_trace"]["events"]] == [
        "first",
        "middle",
        "last",
    ]
    assert rows[1]["routing_trace"]["schema_versions"] == [1]
    assert report["status"] == "complete"
    assert report["counts"] == {
        "result_files": 1,
        "trials": 1,
        "mapped_rows": 2,
        "mapped_request_ids": 2,
        "trace_rows": 4,
        "trace_request_ids": 2,
        "joined_requests": 2,
        "malformed_rows": 0,
        "input_errors": 0,
        "missing_maps": 0,
        "missing_traces": 0,
        "duplicate_mapped_ids": 0,
        "orphan_trace_ids": 0,
        "invalid_trace_sequences": 0,
        "invalid_trace_request_ids": 0,
        "missing_outcomes": 0,
        "outcome_pass": 1,
        "outcome_fail": 0,
        "outcome_error": 0,
        "outcome_unknown": 0,
    }
    assert json.loads(report_path.read_text()) == report
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600

    first_output = output_path.read_bytes()
    first_report = report_path.read_bytes()
    module.join_routing_traces(
        job_dir,
        trace_path,
        output_path=output_path,
        report_path=report_path,
    )
    assert output_path.read_bytes() == first_output
    assert report_path.read_bytes() == first_report


def test_joiner_reports_all_completeness_failures(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    _write_trial(
        job_dir,
        "missing-map-directory",
        trial_name="trial-missing-map",
        task_name="task-missing-map",
        request_rows=None,
    )
    mapped_trial = _write_trial(
        job_dir,
        "mapped-directory",
        trial_name="trial-mapped",
        task_name="task-mapped",
        request_rows=[],
    )
    map_path = mapped_trial / "artifacts" / "request_map.jsonl"
    map_path.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "duplicate", "turn": 1}),
                "not-json",
                json.dumps({"request_id": "duplicate", "turn": 2}),
                json.dumps({"request_id": "missing-trace"}),
                json.dumps({"request_id": "bad-sequence"}),
                json.dumps({"request_id": "tainted-trace"}),
                json.dumps({"turn": 5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "routing_trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(_trace_row("duplicate", 0, "duplicate-event")),
                "not-json",
                json.dumps(_trace_row("orphan", 0, "orphan-event")),
                json.dumps(_trace_row("bad-sequence", 1, "gap-event")),
                json.dumps({**_trace_row("unsupported", 0, "future-event"), "schema_version": 2}),
                json.dumps(_trace_row("tainted-trace", 0, "partial-event")),
                json.dumps(
                    {
                        **_trace_row("tainted-trace", 1, "malformed-event"),
                        "schema_version": -1,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "request_id": "bad-sequence",
                        "event": {"sequence": -1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "joined.jsonl"
    report_path = tmp_path / "report.json"

    report = module.join_routing_traces(
        job_dir,
        trace_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert report["status"] == "incomplete"
    assert report["counts"]["malformed_rows"] == 6
    assert [row["source"] for row in report["malformed_rows"]] == [
        "request_map",
        "request_map",
        "routing_trace",
        "routing_trace",
        "routing_trace",
        "routing_trace",
    ]
    assert report["missing_maps"] == [
        {
            "trial": {
                "id": "id-trial-missing-map",
                "name": "trial-missing-map",
                "uri": "trial://trial-missing-map",
            },
            "task": {
                "id": {"name": "task-missing-map"},
                "name": "task-missing-map",
                "source": "terminal-bench",
            },
            "path": "missing-map-directory/artifacts/request_map.jsonl",
            "reason": "not_found",
        }
    ]
    assert [entry["request_id"] for entry in report["missing_traces"]] == ["missing-trace"]
    assert [entry["request_id"] for entry in report["duplicate_mapped_ids"]] == ["duplicate"]
    assert report["orphan_trace_ids"] == ["orphan"]
    assert report["invalid_trace_sequences"] == [{"request_id": "bad-sequence", "sequences": [1]}]
    assert report["invalid_trace_request_ids"] == [
        "bad-sequence",
        "tainted-trace",
        "unsupported",
    ]
    assert report["missing_outcomes"] == []
    assert output_path.read_text() == ""


def test_joiner_reports_invalid_utf8_and_rejects_output_collisions(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    _write_trial(
        job_dir,
        "trial",
        trial_name="trial",
        task_name="task",
        request_rows=[{"request_id": "request-1"}],
    )
    trace_path = tmp_path / "routing_trace.jsonl"
    trace_path.write_bytes(b"\xff\n")

    report = module.join_routing_traces(job_dir, trace_path)

    assert report["status"] == "incomplete"
    assert report["counts"]["input_errors"] == 1
    with pytest.raises(ValueError, match="paths must be distinct"):
        module.join_routing_traces(job_dir, trace_path, output_path=trace_path)


def test_joiner_does_not_fall_back_to_directory_identity(tmp_path: Path) -> None:
    module = _load_joiner()
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "tempting-task-and-trial-name"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-from-result",
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        trial_dir / "artifacts" / "request_map.jsonl",
        [{"request_id": "request-1"}],
    )
    trace_path = tmp_path / "routing_trace.jsonl"
    _write_jsonl(trace_path, [_trace_row("request-1", 0, "decision")])

    rc = module.main(
        [
            "--job-dir",
            str(job_dir),
            "--routing-trace",
            str(trace_path),
        ]
    )

    report = json.loads((job_dir / module.DEFAULT_REPORT_NAME).read_text())
    assert rc == 1
    assert report["status"] == "incomplete"
    assert report["counts"]["trials"] == 0
    assert report["input_errors"] == [
        {
            "source": "result",
            "path": "tempting-task-and-trial-name/result.json",
            "error": "trial_name must be a non-empty string",
        }
    ]
    assert report["orphan_trace_ids"] == ["request-1"]
    assert (job_dir / module.DEFAULT_JOINED_NAME).read_text() == ""
