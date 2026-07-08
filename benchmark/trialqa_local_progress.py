# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only progress report for a running local TrialQA canary.

This is intentionally not a gate. It gives fast feedback from the manifest-bound
append-only ledger while a guarded generation or score canary is still running,
or after an interrupted run, without touching model, judge, or tool services.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_batch as batch  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_progress.v1"
JsonObject = dict[str, Any]
Stage = Literal["generation", "score"]


class TrialQAProgressError(RuntimeError):
    """The progress report cannot safely summarize the selected scope."""


def _latest_records_by_task(records: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for record in records:
        task_id = record.get("task_id")
        if isinstance(task_id, str):
            latest[task_id] = record
    return latest


def _failure_stage(record: Mapping[str, object] | None) -> str | None:
    if record is None or record.get("event") != "failed":
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    stage = payload.get("stage")
    return stage if isinstance(stage, str) else None


def _generation_category(state: str, latest: Mapping[str, object] | None) -> str:
    if state == "not_started":
        return "not_started"
    if state == "generation_started":
        return "in_progress"
    if state in {"generation_completed", "scored", "evidence_imported", "completed"}:
        return "generated"
    if state == "failed":
        failure_stage = _failure_stage(latest)
        return "generation_failed" if failure_stage == "generation" else "failed_other_stage"
    return "unexpected"


def _score_category(state: str, latest: Mapping[str, object] | None) -> str:
    if state == "completed":
        return "scored"
    if state == "generation_completed":
        return "ready_to_score"
    if state in {"score_retry_started", "scored", "evidence_imported"}:
        return "in_progress"
    if state == "failed":
        failure_stage = _failure_stage(latest)
        if failure_stage == "score-import":
            return "score_failed"
        if failure_stage == "generation":
            return "generation_failed"
        return "failed_other_stage"
    if state == "generation_started":
        return "generation_in_progress"
    if state == "not_started":
        return "not_generated"
    return "unexpected"


def _batch_lock_state(capture: Path) -> JsonObject:
    lock_path = capture / "batch.lock"
    if not lock_path.exists():
        return {
            "path": str(lock_path),
            "exists": False,
            "held": False,
            "state": "missing",
        }

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        return {
            "path": str(lock_path),
            "exists": True,
            "held": None,
            "state": "unknown",
            "error": str(exc),
        }
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "path": str(lock_path),
                "exists": True,
                "held": True,
                "state": "held",
            }
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return {
            "path": str(lock_path),
            "exists": True,
            "held": False,
            "state": "unheld",
        }
    finally:
        os.close(descriptor)


def _recommendation(
    *,
    stage: Stage,
    categories: Mapping[str, int],
    total: int,
    batch_lock_held: object,
) -> JsonObject:
    if stage == "generation":
        generated = categories.get("generated", 0)
        failed = categories.get("generation_failed", 0) + categories.get("failed_other_stage", 0)
        in_progress = categories.get("in_progress", 0)
        not_started = categories.get("not_started", 0)
        if failed:
            return {
                "action": "inspect_or_recover_generation_failure",
                "requires_spend": False,
                "reason": f"{failed} selected task(s) are failed in the ledger",
            }
        if in_progress:
            if batch_lock_held is False:
                return {
                    "action": "recover_interrupted_generation",
                    "requires_spend": True,
                    "reason": (
                        f"{in_progress} selected task(s) are generation_started, "
                        "but no batch driver holds the lock"
                    ),
                    "recovery_flag": "--recover-interrupted",
                }
            return {
                "action": "wait_or_monitor_generation",
                "requires_spend": False,
                "reason": f"{in_progress} selected task(s) are currently generation_started",
            }
        if generated == total:
            return {
                "action": "run_generation_checkpoint_after_operational_gate",
                "requires_spend": False,
                "reason": "all selected tasks have generation evidence",
            }
        if generated and not_started:
            return {
                "action": "resume_guarded_generation_if_still_authorized",
                "requires_spend": True,
                "reason": f"{generated} generated and {not_started} not started",
            }
        return {
            "action": "run_guarded_generation_canary_after_spend_review",
            "requires_spend": True,
            "reason": "no selected generation work has started",
        }
    scored = categories.get("scored", 0)
    failed = (
        categories.get("score_failed", 0)
        + categories.get("generation_failed", 0)
        + categories.get("failed_other_stage", 0)
    )
    in_progress = categories.get("in_progress", 0)
    ready = categories.get("ready_to_score", 0)
    upstream = categories.get("not_generated", 0) + categories.get("generation_in_progress", 0)
    if failed:
        return {
            "action": "inspect_or_recover_score_failure",
            "requires_spend": False,
            "reason": f"{failed} selected task(s) are failed in the ledger",
        }
    if in_progress:
        if batch_lock_held is False:
            return {
                "action": "recover_interrupted_score",
                "requires_spend": True,
                "reason": (
                    f"{in_progress} selected task(s) are in a partial score state, "
                    "but no batch driver holds the lock"
                ),
                "recovery_flag": "--recover-interrupted",
            }
        return {
            "action": "wait_or_monitor_score",
            "requires_spend": False,
            "reason": f"{in_progress} selected task(s) are in a partial score state",
        }
    if scored == total:
        return {
            "action": "run_score_checkpoint_after_promotion_gate",
            "requires_spend": False,
            "reason": "all selected tasks have completed score evidence",
        }
    if ready:
        return {
            "action": "run_guarded_score_canary_after_spend_review",
            "requires_spend": True,
            "reason": f"{ready} selected task(s) have generation evidence and await scoring",
        }
    return {
        "action": "finish_generation_before_scoring",
        "requires_spend": bool(upstream),
        "reason": "selected scope does not yet have generation evidence for scoring",
    }


def build_progress_report(
    *,
    manifest_path: Path,
    experiment_root: Path,
    stage: Stage,
    question_start: int,
    question_limit: int | None,
    repeat_limit: int | None,
    condition: batch.ConditionScope = "both",
) -> JsonObject:
    """Return a read-only progress summary for the selected TrialQA scope."""

    manifest = demo._read_json_object(manifest_path, "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    manifest_id = cast(str, manifest["manifest_id"])
    tasks = [dict(item) for item in cast(list[dict[str, object]], manifest["tasks"])]
    scope = batch._build_manifest_task_scope(
        manifest,
        tasks,
        limit=None,
        question_start=question_start,
        question_limit=question_limit,
        repeat_limit=repeat_limit,
        condition=condition,
    )
    scoped = list(scope.tasks)
    capture = experiment_root / manifest_id
    batch_lock = _batch_lock_state(capture)
    ledger_path = capture / "ledger.jsonl"
    ledger = demo.ResumableLedger(ledger_path, manifest)
    records = ledger.records()
    states = ledger.states()
    latest = _latest_records_by_task(records)
    category_for = _generation_category if stage == "generation" else _score_category

    task_states: dict[str, JsonObject] = {}
    category_counts: Counter[str] = Counter()
    for task in scoped:
        task_id = cast(str, task["task_id"])
        state = states.get(task_id, "not_started")
        record = latest.get(task_id)
        category = category_for(state, record)
        category_counts[category] += 1
        task_states[task_id] = {
            "state": state,
            "category": category,
            "latest_event": record.get("event") if record is not None else None,
            "latest_sequence": record.get("sequence") if record is not None else None,
            "failure_stage": _failure_stage(record),
        }

    total = len(scoped)
    if total == 0:
        raise TrialQAProgressError("selected scope is empty")
    done_category = "generated" if stage == "generation" else "scored"
    done = category_counts.get(done_category, 0)
    latest_record = records[-1] if records else None
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "stage": stage,
        "capture": str(capture),
        "ledger": {
            "path": str(ledger_path),
            "exists": ledger_path.exists(),
            "record_count": len(records),
            "latest": (
                {
                    "sequence": latest_record.get("sequence"),
                    "task_id": latest_record.get("task_id"),
                    "event": latest_record.get("event"),
                    "recorded_at": latest_record.get("recorded_at"),
                    "record_sha256": latest_record.get("record_sha256"),
                }
                if latest_record is not None
                else None
            ),
        },
        "batch_lock": batch_lock,
        "scope": scope.metadata(manifest_id),
        "progress": {
            "selected_task_count": total,
            "done_task_count": done,
            "remaining_task_count": total - done,
            "done_fraction": done / total,
            "category_counts": dict(sorted(category_counts.items())),
        },
        "task_states": task_states,
        "recommendation": _recommendation(
            stage=stage,
            categories=category_counts,
            total=total,
            batch_lock_held=batch_lock["held"],
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["generation", "score"], required=True)
    parser.add_argument("--question-start", type=int, required=True)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--repeat-limit", type=int)
    parser.add_argument("--condition", choices=["both", "baseline", "treatment"], default="both")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_progress_report(
        manifest_path=args.manifest,
        experiment_root=args.experiment_root,
        stage=cast(Stage, args.stage),
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        condition=cast(batch.ConditionScope, args.condition),
    )
    if args.output:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
