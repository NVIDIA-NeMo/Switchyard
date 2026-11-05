# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.4"]
# ///
"""Evaluate frozen policies on complete, paired, held-out task results."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def validate(plan, rows):
    """Reject unmatched tasks, incomplete spend, and changed experiment identities."""
    tasks = plan["task_ids"]
    if len(tasks) < 200 or len(tasks) != len(set(tasks)):
        raise ValueError("need at least 200 distinct canonical task IDs")
    if set(tasks) & set(plan["development_task_ids"]):
        raise ValueError("development tasks overlap the test set")
    for key in ("protocol_sha256", "pricing_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", plan[key]):
            raise ValueError(f"invalid {key}")
    arms = plan["arm_sha256"]
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in arms.values()):
        raise ValueError("arm fingerprints must be SHA256 digests")
    candidate = plan["candidate"]
    if candidate not in arms:
        raise ValueError("candidate is missing")
    baselines = plan["baselines"]
    if not baselines or len({b["name"] for b in baselines}) != len(baselines):
        raise ValueError("need named, distinct, preregistered baselines")
    for baseline in baselines:
        mixture = baseline["mixture"]
        if not mixture or candidate in mixture or not set(mixture) <= arms.keys():
            raise ValueError("invalid baseline mixture")
        values = list(mixture.values())
        if any(not np.isfinite(v) or v < 0 for v in values) or not np.isclose(sum(values), 1):
            raise ValueError("mixture probabilities must sum to one")
    if not 0 <= plan["cost_tolerance"] <= 0.02:
        raise ValueError("cost tolerance must be between zero and 2 percent")
    repeats = plan["replicates"]
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("need at least two repeats per task and arm")
    expected = {(task, repeat, arm) for task in tasks for repeat in range(repeats) for arm in arms}
    indexed = {}
    for row in rows:
        key = row["task_id"], row["replicate"], row["arm"]
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate trial {key}")
        if row["split"] != "test" or row["terminal"] is not True:
            raise ValueError("only terminal test outcomes are allowed")
        if any(row[k] != plan[k] for k in ("protocol_sha256", "pricing_sha256")):
            raise ValueError("protocol or pricing differs")
        if row["arm_sha256"] != arms[row["arm"]]:
            raise ValueError("arm changed after preregistration")
        cost = row["total_cost_usd"]
        if row["cost_complete"] is not True or not isinstance(cost, (int, float)):
            raise ValueError("complete total costs are required")
        if not np.isfinite(cost) or cost < 0 or row["reward"] not in (0, 1):
            raise ValueError("invalid cost or binary outcome")
        indexed[key] = row
    if indexed.keys() != expected:
        raise ValueError(f"missing {len(expected - indexed.keys())} trials")
    return indexed


def evaluate(plan, rows, draws=40_000, seed=731):
    """Bootstrap tasks with all their repeats, paired across frozen policies."""
    indexed = validate(plan, rows)
    tasks, repeats = plan["task_ids"], plan["replicates"]
    means = {}
    for arm in plan["arm_sha256"]:
        means[arm] = np.array([
            [[indexed[task, r, arm][k] for k in ("reward", "total_cost_usd")]
             for r in range(repeats)] for task in tasks
        ]).mean(axis=1)
    candidate = means[plan["candidate"]]
    baselines = plan["baselines"]
    # Two one-sided bounds per baseline share a 5 percent error allowance.
    tail = 0.05 / (2 * len(baselines))
    if draws * tail < 100:
        raise ValueError("increase bootstrap draws to resolve the adjusted tail")
    deltas = []
    for baseline in baselines:
        control = sum(means[arm] * weight for arm, weight in baseline["mixture"].items())
        delta = candidate - control
        delta[:, 1] -= plan["cost_tolerance"] * control[:, 1]
        deltas.append(delta)
    deltas = np.array(deltas)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(baselines), 2))
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        sample = rng.integers(0, len(tasks), size=(size, len(tasks)))
        boot[start:start + size] = deltas[:, sample, :].mean(axis=2).transpose(1, 0, 2)
    comparisons = []
    for i, baseline in enumerate(baselines):
        lower_success = float(np.quantile(boot[:, i, 0], tail))
        upper_cost = float(np.quantile(boot[:, i, 1], 1 - tail))
        comparisons.append({
            "baseline": baseline["name"], "mixture": baseline["mixture"],
            "success_difference": float(deltas[i, :, 0].mean()),
            "adjusted_success_lower_bound": lower_success,
            "adjusted_cost_excess_upper_bound_usd": upper_cost,
            "passes": lower_success > 0 and upper_cost <= 0,
        })
    return {
        "status": "criteria_met" if all(c["passes"] for c in comparisons) else "not_established",
        "scope": "only the preregistered baselines and test task population",
        "method": "approximate paired task cluster percentile bootstrap with Bonferroni bounds",
        "independent_task_clusters": len(tasks), "replicates_per_task": repeats,
        "bootstrap_draws": draws, "seed": seed, "one_sided_tail": tail,
        "cost_tolerance": plan["cost_tolerance"],
        "candidate_mean_success": float(candidate[:, 0].mean()),
        "candidate_mean_cost_usd": float(candidate[:, 1].mean()),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--draws", type=int, default=40_000)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    result_bytes = args.results.read_bytes()
    result = evaluate(json.loads(plan_bytes), [json.loads(line) for line in result_bytes.splitlines()], args.draws)
    result["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    result["results_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
