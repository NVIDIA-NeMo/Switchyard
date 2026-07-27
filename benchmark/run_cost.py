#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-model cost breakdown for a benchmark run's routing log.

Aggregates ``routing_requests.jsonl`` by served model and prices each bucket with
:func:`switchyard.lib.cost_estimator.estimate_model_cost`, which splits input into
base / cache-read / cache-write tiers. That split matters: agentic runs are
cache-dominated (90%+ of prompt tokens are cache reads is normal), so a flat
input rate overstates cost several-fold.

On a run that routes sub-agent work to a different tier, the served-model column
doubles as the parent/child attribution key — the routing log records no
``agent_id``, so which model answered is the only role signal available.

Models absent from the price table contribute $0 and are listed separately, so an
unpriced model reads as a gap rather than as a free one.

Usage:
    uv run python benchmark/run_cost.py --run benchmark/tb_runs/<run-name>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from switchyard.lib.cost_estimator import MODEL_PRICING, estimate_model_cost

_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_creation_tokens",
)


def _aggregate(routing_log: Path) -> dict[str, collections.Counter[str]]:
    """Sum request counts and token fields per served model."""
    totals: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for line in routing_log.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        bucket = totals[record.get("model") or "<unknown>"]
        bucket["reqs"] += 1
        for field in _TOKEN_FIELDS:
            bucket[field] += record.get(field) or 0
    return totals


def _mean_agg_score(run: Path) -> tuple[float, int] | None:
    """Mean rubric ``agg_score`` across the run's scored tasks, if any."""
    scores = [
        json.loads(path.read_text())["agg_score"]
        for path in run.glob("jobs/*/task-*/verifier/evaluation_results.json")
    ]
    return (sum(scores) / len(scores), len(scores)) if scores else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Run directory under tb_runs/")
    args = parser.parse_args()

    totals = _aggregate(args.run / "routing_requests.jsonl")
    if not totals:
        print("no routing records found")
        return

    unpriced = [model for model in totals if model not in MODEL_PRICING]
    total_cost = 0.0
    header = f"{'model':46s} {'reqs':>5s} {'in':>11s} {'out':>9s} {'cached':>11s} {'USD':>9s}"
    print(header)
    for model, bucket in sorted(totals.items(), key=lambda item: -item[1]["reqs"]):
        cost = estimate_model_cost(
            model,
            bucket["prompt_tokens"],
            bucket["completion_tokens"],
            bucket["cached_tokens"],
            bucket["cache_creation_tokens"],
        )["total_cost"]
        total_cost += cost
        print(
            f"{model:46s} {bucket['reqs']:5d} {bucket['prompt_tokens']:11,} "
            f"{bucket['completion_tokens']:9,} {bucket['cached_tokens']:11,} {cost:9.3f}"
        )

    requests = sum(bucket["reqs"] for bucket in totals.values())
    print(f"\nTOTAL: ${total_cost:.2f} over {requests} requests")
    if unpriced:
        # Priced at zero by estimate_model_cost — surface it so the total is not
        # mistaken for complete.
        print(f"NOTE: no price entry for {', '.join(sorted(unpriced))} — excluded from the total")

    scored = _mean_agg_score(args.run)
    if scored:
        mean, count = scored
        print(f"mean agg_score: {mean:.3f} over {count} tasks  → ${total_cost / count:.2f}/task")


if __name__ == "__main__":
    main()
