#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze whether proxy fields confound alignment-judge verdicts."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

DEFAULT_INPUT = Path("docs/analyses/data/swebench-pro/findings/swebench_pro_difficulty_proxies_enriched.csv")
DEFAULT_REPORT = Path("docs/analyses/data/swebench-pro/findings/alignment_confounders_report.md")


SCALAR_METRICS = [
    "fail_to_pass_total",
    "pass_to_pass_total",
    "required_total",
    "selected_test_files_count",
    "issue_specificity_count",
    "issue_categories_count",
    "problem_statement_words",
    "requirements_words",
    "requirements_bullet_count",
    "interface_words",
    "patch_files",
    "patch_changed",
    "test_patch_files",
    "test_patch_changed",
]

BUCKET_FIELDS = [
    "requirements_words_bucket",
    "test_patch_changed_bucket",
    "patch_changed_bucket",
    "f2p_bucket",
    "selected_test_files_bucket",
    "patch_files_bucket",
]


def pct(num: float, den: float) -> str:
    if den == 0:
        return ""
    return f"{num / den * 100:.1f}%"


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def correlation_rows(rows: list[dict[str, str]], target: str) -> list[dict[str, object]]:
    y = [1.0 if row["alignment_verdict"] == target else 0.0 for row in rows]
    out = []
    for metric in SCALAR_METRICS:
        xs = [float(row[metric]) for row in rows]
        out.append(
            {
                "metric": metric,
                f"corr_with_{target}": f"{pearson_corr(xs, y):.3f}",
            }
        )
    return sorted(out, key=lambda row: abs(float(row[f"corr_with_{target}"])), reverse=True)


def bucket_rows(rows: list[dict[str, str]], field: str) -> list[dict[str, object]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[row[field]][row["alignment_verdict"]] += 1
    out = []
    for value, counts in sorted(groups.items()):
        total = sum(counts.values())
        out.append(
            {
                field: value,
                "n": total,
                "leaked": f"{counts['leaked']}/{total} ({pct(counts['leaked'], total)})",
                "narrow_tests": f"{counts['narrow_tests']}/{total} ({pct(counts['narrow_tests'], total)})",
                "ok": f"{counts['ok']}/{total} ({pct(counts['ok'], total)})",
            }
        )
    return out


def make_report(rows: list[dict[str, str]], report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    verdict_counts = Counter(row["alignment_verdict"] for row in rows)
    total = len(rows)

    lines = [
        "# Alignment-Judge Confounder Analysis",
        "",
        "This report asks whether SWE-bench Pro metadata/diff/test-size fields are associated with alignment verdicts.",
        "",
        "## What The Aligner Sees",
        "",
        "- For normal generated tasks, the alignment prompt sees only `instruction.md`, reference test bodies, and PR diff.",
        "- For this imported SWE-bench run, `swebench_alignment.py` can append `source_metadata.requirements` and `source_metadata.interface` to the problem statement. The stored leakage evidence confirms that happened for at least these labels, because leaked reasons quote `File: ...`, `Function: ...`, and explicit method signatures from metadata.",
        "- The aligner does not see agent trajectories or pass/fail outcomes.",
        "",
        "## Verdict Mix",
        "",
        "| verdict | n | share |",
        "| --- | --- | --- |",
    ]
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"| {verdict} | {count} | {pct(count, total)} |")

    lines.extend(["", "## Correlation With `leaked` Verdict", ""])
    lines.extend(markdown_table(correlation_rows(rows, "leaked"), ["metric", "corr_with_leaked"]))
    lines.extend(["", "## Correlation With `narrow_tests` Verdict", ""])
    lines.extend(markdown_table(correlation_rows(rows, "narrow_tests"), ["metric", "corr_with_narrow_tests"]))
    lines.extend(["", "## Bucketed Verdict Mix", ""])
    for field in BUCKET_FIELDS:
        lines.extend(["", f"### {field}", ""])
        lines.extend(markdown_table(bucket_rows(rows, field), [field, "n", "leaked", "narrow_tests", "ok"]))

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The aligner is directly sensitive to metadata content when `requirements`/`interface` are included, because those strings become part of the `<instruction>` block it audits for leakage.",
            "- The strongest associations with `leaked` are larger test patches, longer problem statements, longer requirements, and larger true F2P/required-test sets. These are weak-to-moderate correlations, not determinative labels.",
            "- Requirements length has a clear bucket effect: leaked share rises from 10.5% in `0-100` words to 35.6% in `251-500` words.",
            "- Test patch size has a similar effect: leaked share rises from 14.7% in `0-20` changed test lines to about 40% above 100 changed test lines.",
            "- Meaningful interface presence itself is not the main driver: earlier interface analysis found leaked share 25.5% for meaningful interface vs 28.0% for placeholder.",
            "- Mechanism: big/long metadata tends to enumerate APIs, files, private helpers, and exact behaviors. The prompt is explicitly looking for those as leakage, so the aligner can be confounded by metadata verbosity and API extraction style.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    make_report(rows, args.report)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.report}")


if __name__ == "__main__":
    main()
