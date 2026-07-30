#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize evaluate verdicts and their resulting stages from a pipeline state file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show eval verdict counts and verdict->stage cross-tab from state.json."
    )
    parser.add_argument("state_file", type=Path, help="Path to pipeline state.json")
    args = parser.parse_args()

    with args.state_file.open() as f:
        state = json.load(f)

    tasks = state.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"{args.state_file}: expected top-level 'tasks' dict")

    verdict_counts: Counter[str] = Counter()
    verdict_stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    verdict_reason_samples: dict[str, str] = {}

    for task in tasks.values():
        verdict = (task.get("eval_verdict") or "").strip()
        if not verdict:
            continue
        stage = task.get("stage", "?")
        reason = (task.get("eval_reason") or "").strip()
        verdict_counts[verdict] += 1
        verdict_stage_counts[verdict][stage] += 1
        if reason and verdict not in verdict_reason_samples:
            verdict_reason_samples[verdict] = reason

    print(f"Pipeline: {args.state_file}")
    print()

    if not verdict_counts:
        print("No tasks with eval_verdict set.")
        return 0

    print("Eval verdicts:")
    for verdict, count in verdict_counts.most_common():
        print(f"  {verdict:12s} {count:4d}")

    print()
    print("Verdict -> stage:")
    for verdict, _count in verdict_counts.most_common():
        parts = [f"{stage}={count}" for stage, count in sorted(verdict_stage_counts[verdict].items())]
        print(f"  {verdict:12s} {'  '.join(parts)}")

    print()
    print("Sample reasons:")
    for verdict, _count in verdict_counts.most_common():
        reason = verdict_reason_samples.get(verdict, "")
        if len(reason) > 180:
            reason = reason[:180] + "..."
        print(f"  {verdict:12s} {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
