#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export tasks labeled leaked where the agent did not succeed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_CSV = Path("docs/analyses/data/swebench-pro/swebench-pro-craft-analysis.csv")
DEFAULT_OUTPUT = Path("docs/analyses/data/swebench-pro/findings/leaked_agent_failures.csv")
ALIGNMENT_COL = "new_alignment_verdict"
SUCCESS_COL = "agent_success"


def populated_column_index(rows: list[list[str]], header: list[str], name: str) -> int:
    candidates = [idx for idx, col_name in enumerate(header) if col_name == name]
    if not candidates:
        raise ValueError(f"Missing required column: {name}")
    return max(
        candidates,
        key=lambda idx: sum(1 for row in rows[1:] if len(row) > idx and row[idx].strip()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.csv.open(newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {args.csv}")

    header = rows[0]
    alignment_idx = populated_column_index(rows, header, ALIGNMENT_COL)
    success_idx = header.index(SUCCESS_COL)

    leaked_failures = [
        row
        for row in rows[1:]
        if len(row) > max(alignment_idx, success_idx)
        and row[alignment_idx].strip().lower() == "leaked"
        and row[success_idx].strip().upper() != "TRUE"
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(leaked_failures)

    print(f"Leaked agent failures: {len(leaked_failures)}")
    print(f"Alignment column index used: {alignment_idx}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
