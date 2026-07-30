#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join an exported task CSV with eval results from another pipeline state.json."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def _load_state(path: Path, retries: int = 8, delay_s: float = 0.4) -> dict:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with path.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(delay_s)
    raise RuntimeError(f"Could not read a stable JSON snapshot from {path}: {last_error}") from last_error


def _clean_text(value: object, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _build_indexes(tasks: dict[str, dict], *, max_text: int) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    by_instance_id: dict[str, dict] = {}
    by_task_id: dict[str, dict] = {}
    by_commit_sha: dict[str, dict] = {}

    for task_id, task in tasks.items():
        candidate = task.get("candidate_data") or {}
        source_meta = candidate.get("source_metadata") or {}
        instance_id = (
            source_meta.get("instance_id")
            or candidate.get("source_task_id")
            or ""
        )
        commit_sha = task.get("commit_sha") or candidate.get("sha") or ""

        payload = {
            "new_task_id": task.get("task_id") or task_id,
            "new_stage": task.get("stage", ""),
            "new_eval_verdict": task.get("eval_verdict", ""),
            "new_eval_reason": _clean_text(task.get("eval_reason", ""), max_text),
            "new_hardness_band": task.get("hardness_band", ""),
            "new_hardness_verdict": task.get("hardness_verdict", ""),
            "new_needs_human_review": task.get("needs_human_review", False),
            "new_human_review_reason": _clean_text(task.get("human_review_reason", ""), max_text),
            "new_alignment_verdict": task.get("alignment_verdict", ""),
            "new_alignment_reason": _clean_text(task.get("alignment_reason", ""), max_text),
            "new_swebench_instance_id": instance_id,
            "new_commit_sha": commit_sha,
        }

        by_task_id[str(task_id)] = payload
        if commit_sha:
            by_commit_sha[str(commit_sha)] = payload
        if instance_id:
            by_instance_id[str(instance_id)] = payload

    return by_instance_id, by_task_id, by_commit_sha


def _match_row(
    row: dict[str, str],
    by_instance_id: dict[str, dict],
    by_task_id: dict[str, dict],
    by_commit_sha: dict[str, dict],
) -> tuple[dict[str, object], str]:
    keys = [
        ("swebench_instance_id", row.get("swebench_instance_id", "")),
        ("swebench_source_task_id", row.get("swebench_source_task_id", "")),
        ("task_id", row.get("task_id", "")),
        ("commit_sha", row.get("commit_sha", "")),
    ]

    for key_name, value in keys:
        if not value:
            continue
        if key_name in {"swebench_instance_id", "swebench_source_task_id"} and value in by_instance_id:
            return by_instance_id[value], key_name
        if key_name == "task_id" and value in by_task_id:
            return by_task_id[value], key_name
        if key_name == "commit_sha" and value in by_commit_sha:
            return by_commit_sha[value], key_name

    return {}, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append eval columns from a new state.json onto an exported task CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Original exported CSV")
    parser.add_argument("new_state", type=Path, help="New pipeline state.json to join from")
    parser.add_argument("output_csv", type=Path, help="Destination comparison CSV")
    parser.add_argument(
        "--max-text",
        type=int,
        default=500,
        help="Maximum length for free-text fields like reasons",
    )
    args = parser.parse_args()

    state = _load_state(args.new_state)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{args.new_state}: expected top-level 'tasks' dict")

    by_instance_id, by_task_id, by_commit_sha = _build_indexes(tasks, max_text=args.max_text)

    with args.input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{args.input_csv}: CSV has no header row")
        rows = list(reader)
        original_fields = list(reader.fieldnames)

    extra_fields = [
        "match_key_used",
        "new_task_id",
        "new_stage",
        "new_eval_verdict",
        "new_eval_reason",
        "new_hardness_band",
        "new_hardness_verdict",
        "new_alignment_verdict",
        "new_alignment_reason",
        "new_needs_human_review",
        "new_human_review_reason",
        "new_swebench_instance_id",
        "new_commit_sha",
    ]
    fieldnames = original_fields + [f for f in extra_fields if f not in original_fields]

    matched = 0
    for row in rows:
        matched_payload, match_key = _match_row(row, by_instance_id, by_task_id, by_commit_sha)
        row["match_key_used"] = match_key
        if matched_payload:
            matched += 1
            for key, value in matched_payload.items():
                row[key] = value
        else:
            for key in extra_fields:
                row.setdefault(key, "")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows to {args.output_csv} "
        f"({matched} matched against {args.new_state})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
