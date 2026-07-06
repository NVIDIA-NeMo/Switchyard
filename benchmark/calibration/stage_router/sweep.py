# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score escalation policies against per_turn.jsonl + per_task.jsonl.

Run calibrate.py first to generate the input files.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent


def load():
    by: dict = defaultdict(list)
    with (OUT / "per_turn.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            by[(r["task"], r["arm"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["turn"])

    outcomes: dict = defaultdict(dict)
    rank = {"pass": 2, "fail": 1, "err": 0}
    with (OUT / "per_task.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            cur = outcomes[r["task"]].get(r["arm"])
            if cur is None or rank[r["outcome"]] > rank[cur]:
                outcomes[r["task"]][r["arm"]] = r["outcome"]
    return by, outcomes


def score(by, outcomes, picker_fn, strong="strong", weak="weak"):
    P = F = E = esc = 0
    for task, arms in outcomes.items():
        turns = by.get((task, strong), [])
        escalate = picker_fn(turns) if turns else False
        o = arms.get(weak if escalate else strong, "err")
        if escalate:
            esc += 1
        if o == "pass":
            P += 1
        elif o == "fail":
            F += 1
        else:
            E += 1
    total = P + F
    return {
        "pass_count": P,
        "pct": P / max(1, total) * 100,
        "esc_rate": esc / max(1, len(outcomes)),
    }


# --- Escalation policies ---

def always_stay(turns):   return False
def always_escalate(turns): return True


def no_write_by_turn(N: int):
    def p(turns):
        for t in turns:
            if t["turn_depth"] >= N:
                return t["write_count"] == 0 and t["edit_count"] == 0
        return False
    return p


def silent_stall(td: int, max_reads: int = 4):
    def p(turns):
        return any(
            t["turn_depth"] >= td and t["write_count"] == 0
            and t["edit_count"] == 0 and t["read_count"] <= max_reads
            for t in turns
        )
    return p


def combo(turns):
    """Escalate on stall or critical error; stay if tests passed."""
    for t in turns:
        if t["tests_passed"]:
            return False
        if t["severity"] >= 1.0:
            return True
        if t["turn_depth"] >= 8 and t["write_count"] == 0 and t["edit_count"] == 0:
            return True
        if t["turn_depth"] >= 7 and t["severity"] >= 0.7 and t["write_count"] == 0:
            return True
        if t["pure_bash_streak"] >= 4 and t["write_count"] == 0:
            return True
    return False


def main():
    by, outcomes = load()
    policies = [
        ("always_stay",              always_stay),
        ("always_escalate",          always_escalate),
        ("no_write_by_turn=8",       no_write_by_turn(8)),
        ("no_write_by_turn=12",      no_write_by_turn(12)),
        ("no_write_by_turn=15",      no_write_by_turn(15)),
        ("silent_stall td>=8 R<=2",  silent_stall(8, 2)),
        ("silent_stall td>=8 R<=4",  silent_stall(8, 4)),
        ("silent_stall td>=12 R<=6", silent_stall(12, 6)),
        ("combo",                    combo),
    ]
    print(f"{'policy':40}  pass%   esc%")
    print("-" * 55)
    for name, p in policies:
        r = score(by, outcomes, p)
        print(f"  {name:38}  {r['pct']:5.1f}%  {r['esc_rate']*100:4.0f}%")


if __name__ == "__main__":
    main()
