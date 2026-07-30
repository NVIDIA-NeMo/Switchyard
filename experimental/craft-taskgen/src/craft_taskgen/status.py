# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact state.json summary for monitoring a long-running pipeline.

Run `craft-taskgen-status path/to/state.json` to see stage counts, in-progress tasks,
last-update time, and NEEDS_FIX reasons at a glance.

Designed to run in under 1 second and print a single screen of useful info.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Display order — roughly matches pipeline stages, terminal states last
_STAGE_ORDER = [
    "candidate",
    "evaluated",
    "promising",
    "built",
    "alignment_checked",
    "f2p_p2p_classified",
    "docker_validated",
    "oracle_checked",
    "opus_smoke_tested",
    "opus_triaged",
    "accepted",
    "needs_fix",
    "rejected",
]


def _fmt_age(iso_ts: str) -> str:
    try:
        then = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return "?"
    delta = datetime.now().astimezone(then.tzinfo) if then.tzinfo else datetime.now()
    seconds = (delta - then).total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def _latest_timestamp(task: dict) -> str | None:
    logs = task.get("iteration_log") or []
    for entry in reversed(logs):
        ts = entry.get("timestamp")
        if ts:
            return ts
    return None


def summarize(state_path: str, show_tracebacks: bool = False) -> int:
    path = Path(state_path)
    if not path.is_file():
        print(f"ERROR: state file not found: {state_path}", file=sys.stderr)
        return 1

    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: could not read state.json: {e}", file=sys.stderr)
        return 1

    tasks = state.get("tasks", {})
    n_total = len(tasks)
    run_info = state.get("run_info") or {}

    if n_total == 0:
        print(f"Pipeline: {path}")
        if run_info:
            host = run_info.get("hostname", "?")
            n_files = len(run_info.get("candidate_files", []))
            print(f"  Host:                 {host}")
            print(f"  Candidate files:      {n_files}")
        print("  (no tasks in state)")
        return 0

    stage_counts: Counter[str] = Counter()
    in_progress: list[tuple[str, str]] = []  # (task_id, step)
    needs_fix: list[tuple[str, str]] = []  # (task_id, reason)
    accepted: list[tuple[str, str]] = []  # (task_id, opus_score)
    latest_ts: str | None = None

    for tid, task in tasks.items():
        stage = task.get("stage", "?")
        stage_counts[stage] += 1
        ip = task.get("in_progress_step") or ""
        if ip:
            in_progress.append((tid, ip))
        if stage == "needs_fix":
            reason = task.get("human_review_reason") or "(no reason)"
            needs_fix.append((tid, reason[:80]))
        if stage == "accepted":
            accepted.append((tid, str(task.get("opus_score", "?"))))
        ts = _latest_timestamp(task)
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    # File mtime as fallback
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

    print(f"Pipeline: {path}")
    print(f"  Tasks: {n_total}")
    if latest_ts:
        print(f"  Latest iteration log: {latest_ts[:19]} ({_fmt_age(latest_ts)})")
    print(f"  State file mtime:     {mtime[:19]} ({_fmt_age(mtime)})")

    if run_info:
        host = run_info.get("hostname", "?")
        n_files = len(run_info.get("candidate_files", []))
        patterns = run_info.get("candidate_patterns", [])
        print(f"  Host:                 {host}")
        if n_files:
            print(f"  Candidate files:      {n_files} ({', '.join(patterns) if patterns else ''})")
    print()

    print("Stage breakdown:")
    seen: set[str] = set()
    for stage in _STAGE_ORDER:
        if stage in stage_counts:
            print(f"  {stage:24s} {stage_counts[stage]:4d}")
            seen.add(stage)
    # Any unknown stages (forward-compat)
    for stage, count in stage_counts.items():
        if stage not in seen:
            print(f"  {stage:24s} {count:4d}")

    if in_progress:
        print()
        print(f"In progress ({len(in_progress)}):")
        for tid, step in in_progress[:10]:
            print(f"  {tid:20s} {step}")
        if len(in_progress) > 10:
            print(f"  ...and {len(in_progress) - 10} more")

    if accepted:
        print()
        print(f"Accepted ({len(accepted)}):")
        for tid, opus in accepted[:10]:
            print(f"  {tid:20s} Opus={opus}")
        if len(accepted) > 10:
            print(f"  ...and {len(accepted) - 10} more")

    if needs_fix:
        print()
        print(f"NEEDS_FIX ({len(needs_fix)}):")
        for tid, reason in needs_fix[:15]:
            print(f"  {tid:20s} {reason}")
        if len(needs_fix) > 15:
            print(f"  ...and {len(needs_fix) - 15} more")

    # Pipeline health hint
    print()
    if in_progress:
        print(f"Status: ACTIVE ({len(in_progress)} task(s) in progress)")
    elif (
        stage_counts.get("accepted", 0) + stage_counts.get("rejected", 0) + stage_counts.get("needs_fix", 0)
        == n_total
    ):
        print("Status: COMPLETE (all tasks in terminal state)")
    else:
        print("Status: IDLE (no tasks in progress; may be between steps or pipeline exited unexpectedly)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact state.json summary for monitoring",
    )
    parser.add_argument("state_file", help="Path to pipeline state.json")
    args = parser.parse_args()
    return summarize(args.state_file)


if __name__ == "__main__":
    sys.exit(main())
