#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a pipeline state file with EVALUATED tasks reset back to CANDIDATE."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _reset_task(task: dict) -> bool:
    """Reset an evaluated task to the pre-evaluate candidate state."""
    if task.get("stage") != "evaluated":
        return False

    task["stage"] = "candidate"
    task["eval_verdict"] = ""
    task["eval_reason"] = ""
    task["eval_instruction_sketch"] = ""
    task["eval_verifier_notes"] = ""
    task["hardness_band"] = ""
    task["in_progress_step"] = ""

    logs = task.get("iteration_log") or []
    task["iteration_log"] = [entry for entry in logs if entry.get("step") != "evaluate"]
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a new state.json with evaluated tasks reset back to candidate."
    )
    parser.add_argument("input_state", type=Path, help="Source pipeline state.json")
    parser.add_argument("output_state", type=Path, help="Destination state.json")
    args = parser.parse_args()

    with args.input_state.open() as f:
        state = json.load(f)

    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{args.input_state}: expected top-level 'tasks' dict")

    reset_count = 0
    for task in tasks.values():
        if _reset_task(task):
            reset_count += 1

    state["last_updated"] = datetime.now().isoformat()

    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    with args.output_state.open("w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")

    print(
        f"Wrote {args.output_state} with {reset_count} task(s) reset "
        f"from evaluated -> candidate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
