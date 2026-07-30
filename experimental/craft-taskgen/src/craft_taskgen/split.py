# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balance candidate files into N shards for running across multiple machines.

Sharding is balanced by total candidate count (not file count) using a greedy
longest-processing-time heuristic — the file with the most candidates goes to
the currently lightest shard, and so on.

Usage:
    craft-taskgen-split <N> 'candidates/*.json'
    craft-taskgen-split 3 'candidates/*.json' --output shards.json

Default output is one line per shard, whitespace-separated file paths, ready
to paste into `scripts/run-pipeline.sh`. Use `--output PATH` to write a JSON
shard map for programmatic consumption.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys


def balance_shards(files_with_counts: list[tuple[int, str]], n_shards: int) -> list[tuple[int, list[str]]]:
    """Greedy LPT: assign largest files first to the currently-smallest shard."""
    sized = sorted(files_with_counts, reverse=True)
    shards: list[tuple[int, list[str]]] = [(0, []) for _ in range(n_shards)]
    for n, fpath in sized:
        idx = min(range(n_shards), key=lambda i: shards[i][0])
        total, paths = shards[idx]
        shards[idx] = (total + n, paths + [fpath])
    return shards


def count_candidates(fpath: str) -> int:
    try:
        with open(fpath) as f:
            data = json.load(f)
        return len(data.get("candidates", []))
    except (OSError, json.JSONDecodeError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Balance candidate files into N shards for multi-machine runs.",
    )
    parser.add_argument("n_shards", type=int, help="Number of shards")
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Candidate file glob(s), e.g. 'candidates/*.json'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path (shard map). Default: human-readable stdout.",
    )
    args = parser.parse_args()

    if args.n_shards < 1:
        parser.error(f"n_shards must be >= 1, got {args.n_shards}")

    files: list[str] = []
    for pattern in args.patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        print(f"ERROR: no files matched: {args.patterns}", file=sys.stderr)
        return 1

    sized = [(count_candidates(f), f) for f in files]
    shards = balance_shards(sized, args.n_shards)

    if args.output:
        payload = [
            {"shard": i + 1, "total_candidates": total, "n_files": len(paths), "files": paths}
            for i, (total, paths) in enumerate(shards)
        ]
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output} ({args.n_shards} shards, {len(files)} files)")
    else:
        for i, (total, paths) in enumerate(shards, 1):
            print(f"shard {i} ({total} cands, {len(paths)} files): {' '.join(paths)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
