# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare accuracy and token cost across run-baseline.sh run directories.

Usage::

    uv run --no-sync python benchmark/compare_runs.py \
        label1=benchmark/tb_runs/<run-dir> label2=benchmark/tb_runs/<run-dir> ...

Reads each run's Harbor ``result.json`` (accuracy) and Switchyard
``routing_stats_final.json`` (per-tier tokens, judge/classifier overhead,
cost estimate) and prints a side-by-side table plus per-task outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any]:
    results = sorted(run_dir.glob("jobs/*/result.json"))
    if not results:
        raise FileNotFoundError(f"no jobs/*/result.json under {run_dir}")
    harbor = json.loads(results[0].read_text())
    stats_path = run_dir / "routing_stats_final.json"
    routing = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    ev = next(iter(harbor["stats"]["evals"].values()))
    rewards: dict[str, float] = {}
    for value, trials in (ev.get("reward_stats", {}).get("reward", {}) or {}).items():
        for trial in trials:
            # trial names look like <task>-<suffix>; strip the trial suffix.
            rewards[trial.rsplit("-", 1)[0]] = float(value)

    return {
        "harbor": harbor["stats"],
        "eval": ev,
        "rewards": rewards,
        "routing": routing,
    }


def _fmt_cost(routing: dict[str, Any]) -> tuple[float, float, float]:
    ce = routing.get("cost_estimate", {}) or {}
    return (
        ce.get("total_cost", 0.0),
        ce.get("backend_cost", 0.0),
        ce.get("classifier_cost", 0.0),
    )


def main(argv: list[str]) -> int:
    runs: dict[str, dict[str, Any]] = {}
    for arg in argv:
        label, _, path = arg.partition("=")
        if not path:
            print(f"ERROR: expected label=path, got {arg!r}")
            return 2
        runs[label] = _load_run(Path(path))

    header = f"{'metric':<28}" + "".join(f"{label:>18}" for label in runs)
    print(header)
    print("-" * len(header))

    def row(name: str, values: list[str]) -> None:
        print(f"{name:<28}" + "".join(f"{v:>18}" for v in values))

    row("completed trials", [str(r["harbor"]["n_completed_trials"]) for r in runs.values()])
    row("errored trials", [str(r["harbor"]["n_errored_trials"]) for r in runs.values()])
    row("mean reward", [f"{r['eval']['metrics'][0].get('mean', 0.0):.3f}" for r in runs.values()])
    row(
        "solved / total",
        [
            f"{sum(1 for v in r['rewards'].values() if v >= 1.0)}/{len(r['rewards'])}"
            for r in runs.values()
        ],
    )
    row(
        "input tokens",
        [f"{r['harbor'].get('n_input_tokens', 0):,}" for r in runs.values()],
    )
    row(
        "output tokens",
        [f"{r['harbor'].get('n_output_tokens', 0):,}" for r in runs.values()],
    )
    for label_idx, name in ((0, "total cost $"), (1, "backend cost $"), (2, "judge/clf cost $")):
        row(name, [f"{_fmt_cost(r['routing'])[label_idx]:.3f}" for r in runs.values()])

    # Tier split (escalation / routed runs only).
    row(
        "tier split (calls)",
        [
            " ".join(
                f"{v.get('tier') or m.rsplit('/', 1)[-1]}:{v.get('calls', 0)}"
                for m, v in (r["routing"].get("models", {}) or {}).items()
            )
            or "-"
            for r in runs.values()
        ],
    )
    clf_rows = []
    for r in runs.values():
        clf = r["routing"].get("classifier", {}) or {}
        n, e = clf.get("total_requests", 0), clf.get("total_errors", 0)
        clf_rows.append(f"{n} ({e} err)" if n or e else "-")
    row("judge calls", clf_rows)

    # Per-task outcome grid.
    tasks = sorted({t for r in runs.values() for t in r["rewards"]})
    print()
    print(f"{'task':<44}" + "".join(f"{label:>14}" for label in runs))
    for task in tasks:
        marks = []
        for r in runs.values():
            v = r["rewards"].get(task)
            marks.append("-" if v is None else ("PASS" if v >= 1.0 else "fail"))
        print(f"{task:<44}" + "".join(f"{m:>14}" for m in marks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
