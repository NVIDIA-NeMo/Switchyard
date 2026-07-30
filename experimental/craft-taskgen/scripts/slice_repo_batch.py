#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically shuffle candidates.csv and emit a batch slice as a miner-ready CSV.

Usage:
    uv run python scripts/slice_repo_batch.py <candidates.csv> --batch 0 [--batch-size 100] [--seed 42]
    uv run python scripts/slice_repo_batch.py <candidates.csv> --batch 1 [--batch-size 100] [--seed 42]

Output CSV has columns: short_name, github_repo
Pass directly to the miner:
    craft-taskgen-mine --repos-csv batch_0.csv --out candidates/ --top 50
"""

from __future__ import annotations

import argparse
import csv
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_csv", help="Path to candidates.csv (e.g. from craft-bench)")
    parser.add_argument("--batch", type=int, required=True, help="Batch index (0-based)")
    parser.add_argument("--batch-size", type=int, default=100, help="Repos per batch (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible shuffle (default 42)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: batch_<N>.csv)")
    args = parser.parse_args()

    with open(args.candidates_csv) as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    start = args.batch * args.batch_size
    end = start + args.batch_size
    batch = rows[start:end]

    if not batch:
        total_batches = len(rows) // args.batch_size
        print(f"ERROR: batch {args.batch} is out of range (max batch index: {total_batches - 1})")
        raise SystemExit(1)

    out_path = args.out or f"batch_{args.batch}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["short_name", "github_repo"])
        writer.writeheader()
        for row in batch:
            full_name = row["full_name"]
            short_name = full_name.split("/")[-1]
            writer.writerow({"short_name": short_name, "github_repo": full_name})

    total = len(rows)
    print(f"Batch {args.batch}: repos {start + 1}–{min(end, total)} of {total} (seed={args.seed})")
    print(f"Wrote {len(batch)} repos to {out_path}")
    print(f"\nNext: craft-taskgen-mine --repos-csv {out_path} --out candidates/ --top 50")


if __name__ == "__main__":
    main()
