#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter a craft-repos-style CSV to only include permissive/approved licenses.

Usage:
    python3 scripts/filter_by_license.py references/repo_list_v2.csv
    python3 scripts/filter_by_license.py references/repo_list_v2.csv --out references/filtered.csv
    python3 scripts/filter_by_license.py references/repo_list_v2.csv --list-excluded
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FIELDS = ["short_name", "github_repo", "github_url", "stars", "license", "domain", "description"]

ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",  # Apache License — only version in use since 2004; GitHub always returns this SPDX ID
    "BSD-2-Clause",
    "BSD-2",  # alternate form seen in manually-entered rows
    "BSD-3-Clause",
    "BSD-3",  # alternate form seen in manually-entered rows
    "ISC",
    "Zlib",
    "CC0-1.0",  # CC0 only ever has one version (1.0); GitHub returns CC0-1.0
    "CC0",  # defensive fallback for manually-entered rows
    "LGPL-2.1",
    "LGPL-3.0",
    "MPL-2.0",
    "EPL-2.0",
    "BSL-1.0",
    "GPL-3.0",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter repo CSV to approved licenses only.")
    parser.add_argument("csv_file", type=Path, help="Input CSV (craft-repos-style)")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV (default: <stem>-filtered.csv)")
    parser.add_argument("--list-excluded", action="store_true", help="Print excluded repos to stderr")
    args = parser.parse_args()

    if not args.csv_file.exists():
        sys.exit(f"ERROR: input file not found: {args.csv_file}")

    with args.csv_file.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit(f"ERROR: {args.csv_file} is empty or contains no data rows.")

    included = [r for r in rows if r["license"] in ALLOWED_LICENSES]
    excluded = [r for r in rows if r["license"] not in ALLOWED_LICENSES]

    if args.list_excluded and excluded:
        print("Excluded repos:", file=sys.stderr)
        for r in excluded:
            print(f"  {r['short_name']}: {r['license']}", file=sys.stderr)

    out_file = args.out or args.csv_file.with_stem(args.csv_file.stem + "-filtered")
    tmp = out_file.with_suffix(".tmp")
    try:
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(included)
        tmp.rename(out_file)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"ERROR: failed to write {out_file}: {e}")

    print(f"Kept {len(included)}/{len(rows)} repos → {out_file}")
    excluded_licenses = sorted({r["license"] for r in excluded})
    print(f"Excluded licenses ({len(excluded)} repos): {excluded_licenses if excluded_licenses else 'none'}")


if __name__ == "__main__":
    main()
