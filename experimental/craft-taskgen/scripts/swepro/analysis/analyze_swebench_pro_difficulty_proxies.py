#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze SWE-bench Pro JSONL fields as task-difficulty proxies."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

DEFAULT_SWEBENCH_PRO = Path("swebench_pro.jsonl")
DEFAULT_ENRICHED = Path("docs/analyses/data/swebench-pro/findings/task_outcomes_enriched.csv")
DEFAULT_OUTPUT_CSV = Path(
    "docs/analyses/data/swebench-pro/findings/swebench_pro_difficulty_proxies_enriched.csv"
)
DEFAULT_REPORT = Path("docs/analyses/data/swebench-pro/findings/swebench_pro_difficulty_proxies_report.md")

NO_INTERFACE_RE = re.compile(
    r"\b(no|not)\b.*\bnew\b.*\b(public\s+)?interfaces?\b|"
    r"\bthere are no new interfaces?\b",
    re.IGNORECASE,
)


def parse_serialized_list(raw: object) -> list[str]:
    if not raw:
        return []
    if not isinstance(raw, str):
        return list(raw or [])
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def clean_text(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip().strip("\"'").strip()


def word_count(raw: object) -> int:
    return len(re.findall(r"\w+", clean_text(raw)))


def diff_metrics(diff_text: str) -> dict[str, int]:
    files = 0
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {
        "files": files,
        "added": added,
        "removed": removed,
        "changed": added + removed,
        "lines": len(diff_text.splitlines()),
    }


def bucket(value: int, cuts: list[tuple[int, str]], final_label: str) -> str:
    for limit, label in cuts:
        if value <= limit:
            return label
    return final_label


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def pct(num: float, den: float) -> str:
    if den == 0:
        return ""
    return f"{num / den * 100:.1f}%"


def number(value: float) -> str:
    return f"{value:.1f}"


def numeric_summary(values: list[int]) -> dict[str, str]:
    if not values:
        return {"mean": "", "median": "", "max": ""}
    return {
        "mean": number(mean(values)),
        "median": number(median(values)),
        "max": str(max(values)),
    }


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


def group_rate(rows: Iterable[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    output = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        passed = sum(1 for row in group if row["agent_success_bool"])
        required_total = numeric_summary([int(row["required_total"]) for row in group])
        f2p_total = numeric_summary([int(row["fail_to_pass_total"]) for row in group])
        output.append(
            {
                key: value,
                "n": len(group),
                "passed": passed,
                "pass_rate": pct(passed, len(group)),
                "avg_required_total": required_total["mean"],
                "avg_f2p_total": f2p_total["mean"],
            }
        )
    return output


def load_swebench_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            fail_to_pass = parse_serialized_list(obj.get("fail_to_pass"))
            pass_to_pass = parse_serialized_list(obj.get("pass_to_pass"))
            selected_test_files = parse_serialized_list(obj.get("selected_test_files_to_run"))
            issue_specificity = parse_serialized_list(obj.get("issue_specificity"))
            issue_categories = parse_serialized_list(obj.get("issue_categories"))
            patch = diff_metrics(obj.get("patch", ""))
            test_patch = diff_metrics(obj.get("test_patch", ""))
            interface = clean_text(obj.get("interface"))
            requirements = clean_text(obj.get("requirements"))
            problem_statement = clean_text(obj.get("problem_statement"))

            records[obj["instance_id"]] = {
                "swebench_repo": obj.get("repo", ""),
                "repo_language": obj.get("repo_language", ""),
                "fail_to_pass_total": len(fail_to_pass),
                "pass_to_pass_total": len(pass_to_pass),
                "required_total": len(fail_to_pass) + len(pass_to_pass),
                "selected_test_files_count": len(selected_test_files),
                "issue_specificity_count": len(issue_specificity),
                "issue_categories_count": len(issue_categories),
                "issue_specificity": ";".join(issue_specificity),
                "issue_categories": ";".join(issue_categories),
                "problem_statement_words": word_count(problem_statement),
                "requirements_words": word_count(requirements),
                "requirements_bullet_count": len(re.findall(r"(?m)^\s*[-*]\s+", requirements)),
                "interface_words": word_count(interface),
                "interface_meaningful": bool(interface and not NO_INTERFACE_RE.search(interface)),
                "patch_files": patch["files"],
                "patch_added": patch["added"],
                "patch_removed": patch["removed"],
                "patch_changed": patch["changed"],
                "test_patch_files": test_patch["files"],
                "test_patch_added": test_patch["added"],
                "test_patch_removed": test_patch["removed"],
                "test_patch_changed": test_patch["changed"],
            }
    return records


def enrich_rows(enriched_path: Path, records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with enriched_path.open() as f:
        for row in csv.DictReader(f):
            instance_id = row["swebench_instance_id"]
            if instance_id not in records:
                raise ValueError(f"{instance_id} missing from swebench_pro records")
            proxy = records[instance_id]
            merged: dict[str, object] = {**row, **proxy}
            merged["agent_success_bool"] = bool_value(row["agent_success"])
            merged["f2p_bucket"] = bucket(
                int(proxy["fail_to_pass_total"]),
                [(2, "01_1-2"), (5, "02_3-5"), (10, "03_6-10"), (25, "04_11-25")],
                "05_26+",
            )
            merged["p2p_bucket"] = bucket(
                int(proxy["pass_to_pass_total"]),
                [(10, "01_0-10"), (50, "02_11-50"), (100, "03_51-100")],
                "04_101+",
            )
            merged["selected_test_files_bucket"] = bucket(
                int(proxy["selected_test_files_count"]),
                [(1, "01_1"), (2, "02_2"), (5, "03_3-5")],
                "04_6+",
            )
            merged["patch_files_bucket"] = bucket(
                int(proxy["patch_files"]),
                [(1, "01_1"), (3, "02_2-3"), (10, "03_4-10")],
                "04_11+",
            )
            merged["patch_changed_bucket"] = bucket(
                int(proxy["patch_changed"]),
                [(50, "01_0-50"), (200, "02_51-200"), (500, "03_201-500")],
                "04_501+",
            )
            merged["test_patch_changed_bucket"] = bucket(
                int(proxy["test_patch_changed"]),
                [(20, "01_0-20"), (100, "02_21-100"), (250, "03_101-250")],
                "04_251+",
            )
            merged["requirements_words_bucket"] = bucket(
                int(proxy["requirements_words"]),
                [(100, "01_0-100"), (250, "02_101-250"), (500, "03_251-500")],
                "04_501+",
            )
            rows.append(merged)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "repo",
        "alignment_verdict",
        "new_eval_verdict",
        "agent_success",
        "fail_to_pass_total",
        "pass_to_pass_total",
        "required_total",
        "selected_test_files_count",
        "issue_specificity_count",
        "issue_categories_count",
        "issue_specificity",
        "issue_categories",
        "problem_statement_words",
        "requirements_words",
        "requirements_bullet_count",
        "interface_words",
        "interface_meaningful",
        "patch_files",
        "patch_changed",
        "test_patch_files",
        "test_patch_changed",
        "f2p_bucket",
        "p2p_bucket",
        "selected_test_files_bucket",
        "patch_files_bucket",
        "patch_changed_bucket",
        "test_patch_changed_bucket",
        "requirements_words_bucket",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def aggregate_by_alignment(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["alignment_verdict"])].append(row)
    metrics = [
        "fail_to_pass_total",
        "pass_to_pass_total",
        "required_total",
        "selected_test_files_count",
        "patch_files",
        "patch_changed",
        "test_patch_files",
        "test_patch_changed",
        "requirements_words",
        "issue_categories_count",
    ]
    output = []
    for verdict, group in sorted(groups.items()):
        passed = sum(1 for row in group if row["agent_success_bool"])
        row = {
            "alignment_verdict": verdict,
            "n": len(group),
            "passed": passed,
            "pass_rate": pct(passed, len(group)),
        }
        for metric in metrics:
            row[f"avg_{metric}"] = numeric_summary([int(item[metric]) for item in group])["mean"]
        output.append(row)
    return output


def category_rows(rows: list[dict[str, object]], column: str) -> list[dict[str, object]]:
    counts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        values = str(row[column]).split(";") if row[column] else []
        for value in values:
            counts[value].append(row)
    output = []
    for value, group in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))[:20]:
        passed = sum(1 for row in group if row["agent_success_bool"])
        output.append(
            {
                column: value,
                "n": len(group),
                "passed": passed,
                "pass_rate": pct(passed, len(group)),
            }
        )
    return output


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


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
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
    successes = [1.0 if row["agent_success_bool"] else 0.0 for row in rows]
    output = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        corr = pearson_corr(values, successes)
        output.append(
            {
                "metric": metric,
                "corr_with_success": f"{corr:.3f}",
                "direction": "harder_when_larger" if corr < 0 else "easier_when_larger",
            }
        )
    return sorted(output, key=lambda row: float(row["corr_with_success"]))


def make_report(rows: list[dict[str, object]], report_path: Path, output_csv: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SWE-bench Pro Difficulty Proxy Analysis",
        "",
        f"Enriched CSV: `{output_csv}`",
        "",
        "This joins the 263 selected runs to fields available directly in `swebench_pro.jsonl`.",
        "",
        "## Available Proxy Fields",
        "",
        "- `fail_to_pass`: true F2P test count.",
        "- `pass_to_pass`: regression/P2P test count.",
        "- `selected_test_files_to_run`: breadth of test files invoked.",
        "- `patch` and `test_patch`: source/test diff size and file count.",
        "- `requirements`, `interface`, `problem_statement`: instruction/metadata length and interface presence.",
        "- `issue_specificity` and `issue_categories`: human/task labels that can approximate bug type and domain breadth.",
        "",
        "## Alignment-Level Averages",
        "",
    ]
    alignment_columns = [
        "alignment_verdict",
        "n",
        "passed",
        "pass_rate",
        "avg_fail_to_pass_total",
        "avg_pass_to_pass_total",
        "avg_required_total",
        "avg_selected_test_files_count",
        "avg_patch_files",
        "avg_patch_changed",
        "avg_test_patch_files",
        "avg_test_patch_changed",
        "avg_requirements_words",
        "avg_issue_categories_count",
    ]
    lines.extend(markdown_table(aggregate_by_alignment(rows), alignment_columns))
    lines.extend(
        [
            "",
            "## Scalar Correlations With Agent Success",
            "",
            "Pearson correlation over the 263 selected runs. Negative means larger values are associated with lower pass rate.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            correlation_rows(rows),
            ["metric", "corr_with_success", "direction"],
        )
    )

    bucket_sections = [
        ("True F2P Count", "f2p_bucket"),
        ("P2P Count", "p2p_bucket"),
        ("Selected Test File Count", "selected_test_files_bucket"),
        ("Source Patch File Count", "patch_files_bucket"),
        ("Source Patch Changed Lines", "patch_changed_bucket"),
        ("Test Patch Changed Lines", "test_patch_changed_bucket"),
        ("Requirements Word Count", "requirements_words_bucket"),
    ]
    for title, column in bucket_sections:
        lines.extend(["", f"## By {title}", ""])
        lines.extend(
            markdown_table(
                group_rate(rows, column),
                [column, "n", "passed", "pass_rate", "avg_required_total", "avg_f2p_total"],
            )
        )

    lines.extend(["", "## Issue Specificity Labels", ""])
    lines.extend(
        markdown_table(
            category_rows(rows, "issue_specificity"),
            ["issue_specificity", "n", "passed", "pass_rate"],
        )
    )
    lines.extend(["", "## Issue Category Labels", ""])
    lines.extend(
        markdown_table(
            category_rows(rows, "issue_categories"),
            ["issue_categories", "n", "passed", "pass_rate"],
        )
    )
    lines.extend(
        [
            "",
            "## Initial Read",
            "",
            "- In this selected 263-task slice, the clearest scalar proxy is requirements length: more requirements words correlate with lower pass rate (`corr=-0.271`), and the 251-500 word bucket passes at 41.1%.",
            "- Patch/test-patch size also behaves like a difficulty proxy: source patches over 500 changed lines pass at 33.3%, and test patches over 100 changed lines pass around 41-43%.",
            "- True F2P count is directionally useful but weaker than expected here (`corr=-0.105`) and not monotonic across buckets.",
            "- P2P count and selected test-file count are not strong scalar proxies by themselves; they mix broad regression suites with relatively easy compatibility checks.",
            "- Issue label counts and categories are better as stratification variables than scalar difficulty scores.",
            "- Interface length has a small negative association with success (`corr=-0.103`), which is consistent with API-heavy tasks being brittle, but it is not strong enough alone.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swebench-pro", type=Path, default=DEFAULT_SWEBENCH_PRO)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = load_swebench_records(args.swebench_pro)
    rows = enrich_rows(args.enriched, records)
    write_csv(rows, args.output_csv)
    make_report(rows, args.report, args.output_csv)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.report}")


if __name__ == "__main__":
    main()
