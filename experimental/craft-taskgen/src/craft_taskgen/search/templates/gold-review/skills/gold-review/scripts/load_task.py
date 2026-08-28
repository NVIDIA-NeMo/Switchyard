#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and display a CRAFT Search task with gold answer and agent consensus.

Usage:
    uv run python .claude/skills/gold-review/scripts/load_task.py craft-httpx-9e2db24f
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def load_task(task_id: str) -> None:
    review_path = ROOT / "tools" / "search" / "review_data.json"
    if not review_path.exists():
        print("ERROR: Run 'uv run python tools/search/build_review_data.py' first", file=sys.stderr)
        sys.exit(1)

    with open(review_path) as f:
        data = json.load(f)

    task = next((t for t in data["tasks"] if t["task_id"] == task_id), None)
    if not task:
        print(f"ERROR: Task {task_id} not found in review_data.json", file=sys.stderr)
        sys.exit(1)

    # Also load the raw task JSON for the explanation
    repo = task["repo"]
    accepted_dir = ROOT / "tasks" / "accepted" / "search" / repo
    raw_task = None
    for f in accepted_dir.glob("*.json"):
        if f.stem[:8] == task_id.split("-")[-1][:8]:
            with open(f) as fh:
                raw_task = json.load(fh)
            break

    gold = task["gold"]
    tiers = task["tier_results"]
    n_tiers = len(tiers)

    # Header
    print(f"{'=' * 80}")
    print(f"Task: {task_id}")
    print(f"Repo: {repo}  Difficulty: {task['difficulty']}  Classification: {task['classification']}")
    print(f"Score range: {task['score_range']:.2f}  Monotonicity: {task['monotonicity']:.2f}")
    if task["audit_flags"]:
        print(f"Audit flags: {', '.join(task['audit_flags'])}")
    print(f"{'=' * 80}")

    # Instruction
    print(f"\n{'─' * 40} INSTRUCTION {'─' * 40}")
    print(task["instruction"])

    # Gold answer
    print(f"\n{'─' * 40} GOLD FILES {'─' * 40}")
    for f in gold["files"]:
        print(f"  {f}")

    print(f"\n{'─' * 40} GOLD FUNCTIONS {'─' * 40}")
    for f in gold["functions"]:
        print(f"  {f}")

    if gold.get("alt_files") or gold.get("alt_functions"):
        print(f"\n{'─' * 40} ALT {'─' * 40}")
        for f in gold.get("alt_files", []):
            print(f"  file: {f}")
        for f in gold.get("alt_functions", []):
            print(f"  func: {f}")

    print(f"\n{'─' * 40} ASSERTIONS {'─' * 40}")
    for i, a in enumerate(gold["assertions"]):
        print(f"  {i}: {a}")

    # Explanation
    if raw_task:
        explanation = raw_task["task"]["gold_answer"].get("explanation", "")
        print(f"\n{'─' * 40} EXPLANATION {'─' * 40}")
        print(explanation)

    # Agent consensus
    print(f"\n{'─' * 40} TIER RESULTS {'─' * 40}")
    for tier_name, tr in tiers.items():
        print(
            f"  {tier_name}: reward={tr['reward']:.2f} file_r={tr['file_recall']:.2f}"
            f" func_r={tr['function_recall']:.2f} assert={tr['assertion_coverage']:.2f}"
        )

    # Consensus counts
    file_counts = Counter()
    func_counts = Counter()
    all_agent_funcs = Counter()

    gold_func_set = set(gold["functions"])
    alt_func_set = set(gold.get("alt_functions", []))

    for tr in tiers.values():
        for f in tr.get("agent_files", []):
            file_counts[f] += 1
        for f in tr.get("agent_functions", []):
            all_agent_funcs[f] += 1
            if f in gold_func_set:
                func_counts[f] += 1

    print(f"\n{'─' * 40} CONSENSUS: GOLD FILES {'─' * 40}")
    for f in gold["files"]:
        c = file_counts.get(f, 0)
        flag = " *** 0 agents!" if c == 0 else ""
        print(f"  {c}/{n_tiers}: {f}{flag}")

    print(f"\n{'─' * 40} CONSENSUS: GOLD FUNCTIONS {'─' * 40}")
    for f in sorted(gold["functions"], key=lambda x: -func_counts.get(x, 0)):
        c = func_counts.get(f, 0)
        flag = " *** 0 agents!" if c == 0 else ""
        print(f"  {c}/{n_tiers}: {f}{flag}")

    print(f"\n{'─' * 40} CONSENSUS: ALT FUNCTIONS {'─' * 40}")
    for f in gold.get("alt_functions", []):
        c = sum(1 for tr in tiers.values() if f in tr.get("agent_functions", []))
        print(f"  {c}/{n_tiers}: {f}")

    # Extra functions found by agents but not in gold/alt
    extras = {
        f: c
        for f, c in all_agent_funcs.items()
        if f not in gold_func_set and f not in alt_func_set and c >= 2
    }
    if extras:
        print(f"\n{'─' * 40} EXTRA (not in gold/alt, 2+ agents) {'─' * 40}")
        for f, c in sorted(extras.items(), key=lambda x: -x[1]):
            print(f"  {c}/{n_tiers}: {f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: load_task.py <craft-task-id>", file=sys.stderr)
        sys.exit(1)
    load_task(sys.argv[1])


if __name__ == "__main__":
    main()
