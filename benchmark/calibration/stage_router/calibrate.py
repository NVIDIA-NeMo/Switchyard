# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate stage_router confidence threshold from Harbor run outputs.

Pass the Harbor run output directories for the pure-strong and pure-weak arms:

  python calibrate.py --strong-run-dir /tmp/runs/strong --weak-run-dir /tmp/runs/weak
  python sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_extractor import replay_trajectory, task_summary  # noqa: E402

OUT = Path(__file__).parent

_RANK = {"pass": 2, "fail": 1, "err": 0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate stage_router confidence threshold from Harbor run outputs."
    )
    parser.add_argument(
        "--strong-run-dir",
        type=Path,
        required=True,
        help="Harbor output directory for the pure-strong run.",
    )
    parser.add_argument(
        "--weak-run-dir",
        type=Path,
        required=True,
        help="Harbor output directory for the pure-weak probe run.",
    )
    return parser.parse_args(argv)


def _outcome(result: dict) -> str:
    if result.get("exception_info") is not None:
        return "err"
    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    return "pass" if reward == 1.0 else "fail"


def read_arm(arm_dir: Path):
    if not arm_dir.is_dir():
        raise FileNotFoundError(f"{arm_dir} is not a directory")

    outcomes: dict[str, str] = {}
    feats: dict[str, dict] = {}
    per_turn: dict[str, list] = {}

    for td in sorted(arm_dir.iterdir()):
        if not td.is_dir():
            continue
        rp = td / "result.json"
        if not rp.exists():
            continue
        result = json.loads(rp.read_text())
        task = result.get("task_name") or td.name
        outcome = _outcome(result)
        trajs = list(td.glob("agent/sessions/projects/-app/*.jsonl"))
        signals = replay_trajectory(trajs[0]) if trajs else []

        if _RANK[outcome] > _RANK.get(outcomes.get(task, ""), -1):
            outcomes[task] = outcome
            feats[task] = task_summary(signals)
            per_turn[task] = [
                {
                    "task": task,
                    "turn": i,
                    "severity": s.severity,
                    "write_count": s.write_count,
                    "edit_count": s.edit_count,
                    "read_count": s.read_count,
                    "turn_depth": s.turn_depth,
                    "pure_bash_streak": s.pure_bash_streak,
                    "tests_passed": s.tests_passed,
                }
                for i, s in enumerate(signals)
            ]
    return outcomes, feats, per_turn


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    arms = {
        "strong": args.strong_run_dir,
        "weak": args.weak_run_dir,
    }
    all_outcomes: dict[str, dict] = {}
    all_feats: dict[str, dict] = {}
    all_turns: dict[str, dict] = {}

    for arm, d in arms.items():
        outcomes, feats, turns = read_arm(d)
        all_outcomes[arm] = outcomes
        all_feats[arm] = feats
        all_turns[arm] = turns
        passes = sum(1 for v in outcomes.values() if v == "pass")
        print(f"{arm}: {len(outcomes)} tasks  pass={passes}")

    arm_names = list(arms)
    if len(arm_names) >= 2:
        s_out = all_outcomes[arm_names[0]]
        w_out = all_outcomes[arm_names[1]]
        buckets: dict[str, int] = {"RESCUE": 0, "LOSS": 0, "SAFE": 0, "HARD": 0}
        for t in set(s_out) | set(w_out):
            s, w = s_out.get(t), w_out.get(t)
            if s == "pass" and w == "pass":
                buckets["SAFE"] += 1
            elif s == "fail" and w == "pass":
                buckets["RESCUE"] += 1
            elif s == "pass" and w != "pass":
                buckets["LOSS"] += 1
            elif s == "fail" and w != "pass":
                buckets["HARD"] += 1
        print()
        for b, n in buckets.items():
            print(f"  {b}: {n}")

    with (OUT / "per_task.jsonl").open("w") as f:
        for arm in arm_names:
            for task, outcome in all_outcomes[arm].items():
                f.write(json.dumps({"task": task, "arm": arm, "outcome": outcome,
                                    **all_feats[arm].get(task, {})}) + "\n")

    with (OUT / "per_turn.jsonl").open("w") as f:
        for arm in arm_names:
            for _task, turn_list in all_turns[arm].items():
                for rec in turn_list:
                    f.write(json.dumps({**rec, "arm": arm}) + "\n")

    print(f"\nWrote per_task.jsonl + per_turn.jsonl → {OUT}")
    print("Run sweep.py to score escalation policies.")


if __name__ == "__main__":
    main()
