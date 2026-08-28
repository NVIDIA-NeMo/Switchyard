#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze whether meaningful candidate interfaces confound pass rates."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

DEFAULT_STATE = Path("harbor-tasks/craft-tools-v3a/runs/new_pipeline_0427/state.json")
DEFAULT_ENRICHED = Path("docs/analyses/data/swebench-pro/findings/task_outcomes_enriched.csv")
DEFAULT_SWEBENCH_PRO = Path("swebench_pro.jsonl")
DEFAULT_OUTPUT_CSV = Path("docs/analyses/data/swebench-pro/findings/interface_confounders_enriched.csv")
DEFAULT_REPORT = Path("docs/analyses/data/swebench-pro/findings/interface_confounders_report.md")

NO_INTERFACE_RE = re.compile(
    r"\b(no|not)\b.*\bnew\b.*\b(public\s+)?interfaces?\b|"
    r"\bthere are no new interfaces?\b",
    re.IGNORECASE,
)

NO_REQUIREMENTS_RE = re.compile(
    r"^(no\s+(new\s+)?requirements?\.?|"
    r"there are no\s+(new\s+)?requirements?\.?|"
    r"no explicit requirements?\.?|not applicable\.?|n/?a|none|null)$",
    re.IGNORECASE,
)


def text_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip("\"'").strip()


def raw_non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(text_value(value))
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def is_interface_placeholder(value: object) -> bool:
    return bool(NO_INTERFACE_RE.search(text_value(value)))


def is_requirement_placeholder(value: object) -> bool:
    return bool(NO_REQUIREMENTS_RE.match(text_value(value)))


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def int_value(value: str) -> int:
    if value.strip() == "":
        return 0
    return int(float(value))


def pct(num: float, den: float) -> str:
    if den == 0:
        return ""
    return f"{num / den * 100:.1f}%"


def number(value: float) -> str:
    return f"{value:.1f}"


def summary(values: list[int]) -> dict[str, str]:
    if not values:
        return {"mean": "", "median": "", "p75": "", "max": ""}
    ordered = sorted(values)
    p75_idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.75)))
    return {
        "mean": number(mean(values)),
        "median": number(median(values)),
        "p75": number(ordered[p75_idx]),
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


def group_rows(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    result: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        total = len(group)
        passed = sum(1 for row in group if row["agent_success_bool"])
        required_total = summary([int(row["agent_required_tests_total"]) for row in group])
        f2p_total = summary([int(row["swebench_fail_to_pass_total"]) for row in group])
        result.append(
            {
                **dict(zip(keys, key)),
                "n": total,
                "passed": passed,
                "pass_rate": pct(passed, total),
                "avg_required_total": required_total["mean"],
                "avg_swebench_f2p_total": f2p_total["mean"],
            }
        )
    return result


def matched_interface_adjustment(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare interface groups inside matched repo/eval/alignment/F2P cells."""

    cell_keys = ("repo", "new_eval_verdict", "alignment_verdict", "required_total_bucket")
    by_cell: dict[tuple[object, ...], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_cell[tuple(row[key] for key in cell_keys)][row["interface_kind"]].append(row)

    matched_rows: list[dict[str, object]] = []
    meaningful_actual_passes = 0
    meaningful_total = 0
    meaningful_expected_at_placeholder_rates = 0.0

    for cell, by_kind in sorted(by_cell.items()):
        if "meaningful" not in by_kind or "placeholder_no_new" not in by_kind:
            continue
        placeholder = by_kind["placeholder_no_new"]
        meaningful = by_kind["meaningful"]
        placeholder_n = len(placeholder)
        placeholder_pass = sum(1 for row in placeholder if row["agent_success_bool"])
        meaningful_n = len(meaningful)
        meaningful_pass = sum(1 for row in meaningful if row["agent_success_bool"])
        placeholder_rate = placeholder_pass / placeholder_n

        meaningful_total += meaningful_n
        meaningful_actual_passes += meaningful_pass
        meaningful_expected_at_placeholder_rates += placeholder_rate * meaningful_n

        matched_rows.append(
            {
                "repo": cell[0],
                "eval": cell[1],
                "alignment": cell[2],
                "f2p_bucket": cell[3],
                "placeholder": f"{placeholder_pass}/{placeholder_n} ({pct(placeholder_pass, placeholder_n)})",
                "meaningful": f"{meaningful_pass}/{meaningful_n} ({pct(meaningful_pass, meaningful_n)})",
            }
        )

    summary_row = {
        "matched_meaningful_total": meaningful_total,
        "matched_meaningful_actual": meaningful_actual_passes,
        "matched_meaningful_actual_rate": pct(meaningful_actual_passes, meaningful_total),
        "expected_passes_at_placeholder_cell_rates": number(meaningful_expected_at_placeholder_rates),
        "expected_rate_at_placeholder_cell_rates": pct(
            meaningful_expected_at_placeholder_rates, meaningful_total
        ),
    }
    return matched_rows, summary_row


def load_interface_metadata(state_path: Path) -> dict[str, dict[str, object]]:
    with state_path.open() as f:
        state = json.load(f)

    metadata: dict[str, dict[str, object]] = {}
    for task_id, task in state["tasks"].items():
        source_metadata = task.get("candidate_data", {}).get("source_metadata", {}) or {}
        interface = source_metadata.get("interface")
        requirements = source_metadata.get("requirements")

        interface_raw = raw_non_empty(interface)
        interface_placeholder = interface_raw and is_interface_placeholder(interface)
        requirement_raw = raw_non_empty(requirements)
        requirement_placeholder = requirement_raw and is_requirement_placeholder(requirements)

        metadata[task_id] = {
            "interface_raw_non_empty": interface_raw,
            "interface_placeholder_no_new": interface_placeholder,
            "interface_meaningful": interface_raw and not interface_placeholder,
            "interface_kind": "meaningful"
            if interface_raw and not interface_placeholder
            else "placeholder_no_new",
            "requirements_raw_non_empty": requirement_raw,
            "requirements_placeholder": requirement_placeholder,
            "requirements_meaningful": requirement_raw and not requirement_placeholder,
            "interface_char_count": len(text_value(interface)),
            "requirements_char_count": len(text_value(requirements)),
        }
    return metadata


def parse_serialized_list(raw: object) -> list[str]:
    if not raw:
        return []
    if not isinstance(raw, str):
        return list(raw or [])
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def load_swebench_counts(swebench_pro_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    with swebench_pro_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            counts[obj["instance_id"]] = {
                "fail_to_pass_total": len(parse_serialized_list(obj.get("fail_to_pass"))),
                "pass_to_pass_total": len(parse_serialized_list(obj.get("pass_to_pass"))),
            }
    return counts


def load_rows(
    enriched_path: Path,
    interface_metadata: dict[str, dict[str, object]],
    swebench_counts: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    with enriched_path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            task_id = row["task_id"]
            if task_id not in interface_metadata:
                raise ValueError(f"{task_id} is missing from state metadata")
            instance_id = row["swebench_instance_id"]
            if instance_id not in swebench_counts:
                raise ValueError(f"{instance_id} is missing from swebench_pro counts")
            merged: dict[str, object] = {**row, **interface_metadata[task_id]}
            merged["agent_success_bool"] = bool_value(row["agent_success"])
            merged["agent_required_tests_total"] = int_value(row["agent_f2p_tests_total"])
            merged["agent_required_tests_passed"] = int_value(row["agent_f2p_tests_passed"])
            merged["required_total_bucket"] = row["f2p_total_bucket"]
            merged["swebench_fail_to_pass_total"] = swebench_counts[instance_id]["fail_to_pass_total"]
            merged["swebench_pass_to_pass_total"] = swebench_counts[instance_id]["pass_to_pass_total"]
            merged["required_total_matches_swebench"] = (
                merged["agent_required_tests_total"]
                == merged["swebench_fail_to_pass_total"] + merged["swebench_pass_to_pass_total"]
            )
            for column in (
                "source_lines_changed",
                "test_lines_changed",
                "source_file_count",
                "test_file_count",
                "edit_count",
                "edited_file_count",
            ):
                merged[column] = int_value(row[column])
            rows.append(merged)
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "repo",
        "new_eval_verdict",
        "alignment_verdict",
        "agent_success",
        "agent_required_tests_passed",
        "agent_required_tests_total",
        "swebench_fail_to_pass_total",
        "swebench_pass_to_pass_total",
        "required_total_matches_swebench",
        "required_total_bucket",
        "interface_kind",
        "interface_meaningful",
        "interface_placeholder_no_new",
        "requirements_meaningful",
        "requirements_placeholder",
        "interface_char_count",
        "requirements_char_count",
        "source_lines_changed",
        "test_lines_changed",
        "source_file_count",
        "test_file_count",
        "is_multi_file",
        "is_multi_package",
        "is_refactoring",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_report(
    rows: list[dict[str, object]],
    matched_rows: list[dict[str, object]],
    matched_summary: dict[str, object],
    report_path: Path,
    output_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    meaningful = [row for row in rows if row["interface_kind"] == "meaningful"]
    placeholder = [row for row in rows if row["interface_kind"] == "placeholder_no_new"]

    def pass_summary(group: list[dict[str, object]]) -> tuple[int, int, str]:
        passed = sum(1 for row in group if row["agent_success_bool"])
        return passed, len(group), pct(passed, len(group))

    summary_rows = []
    for label, group in (("meaningful", meaningful), ("placeholder_no_new", placeholder)):
        passed, total, rate = pass_summary(group)
        required_total = summary([int(row["agent_required_tests_total"]) for row in group])
        f2p_total = summary([int(row["swebench_fail_to_pass_total"]) for row in group])
        source_lines = summary([int(row["source_lines_changed"]) for row in group])
        test_lines = summary([int(row["test_lines_changed"]) for row in group])
        leaked_count = sum(1 for row in group if row["alignment_verdict"] == "leaked")
        ok_count = sum(1 for row in group if row["alignment_verdict"] == "ok")
        summary_rows.append(
            {
                "interface_kind": label,
                "n": total,
                "passed": passed,
                "pass_rate": rate,
                "leaked_share": pct(leaked_count, total),
                "ok_share": pct(ok_count, total),
                "avg_required_total": required_total["mean"],
                "avg_swebench_f2p_total": f2p_total["mean"],
                "avg_source_lines": source_lines["mean"],
                "avg_test_lines": test_lines["mean"],
            }
        )

    requirement_placeholder_count = sum(1 for row in rows if row["requirements_placeholder"])
    requirement_meaningful_count = sum(1 for row in rows if row["requirements_meaningful"])

    top_repo_rows = group_rows(rows, ("repo", "interface_kind"))
    by_alignment_rows = group_rows(rows, ("alignment_verdict", "interface_kind"))
    by_eval_rows = group_rows(rows, ("new_eval_verdict", "interface_kind"))
    by_f2p_bucket_rows = group_rows(rows, ("f2p_total_bucket", "interface_kind"))
    by_alignment_f2p_rows = group_rows(rows, ("alignment_verdict", "interface_kind", "f2p_total_bucket"))

    lines = [
        "# Interface Field Confounder Analysis",
        "",
        f"Enriched CSV: `{output_csv}`",
        "",
        "Raw `interface` is non-empty for every task, so this report uses the semantic split:",
        "`meaningful` vs `placeholder_no_new`.",
        "",
        "## Headline",
        "",
    ]
    lines.extend(
        markdown_table(
            summary_rows,
            [
                "interface_kind",
                "n",
                "passed",
                "pass_rate",
                "leaked_share",
                "ok_share",
                "avg_required_total",
                "avg_swebench_f2p_total",
                "avg_source_lines",
                "avg_test_lines",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Requirements are not useful as a splitter in this state file: "
            f"{requirement_meaningful_count}/{len(rows)} are semantically meaningful and "
            f"{requirement_placeholder_count}/{len(rows)} are placeholder-like.",
            "",
            "## By Alignment Verdict",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            by_alignment_rows,
            [
                "alignment_verdict",
                "interface_kind",
                "n",
                "passed",
                "pass_rate",
                "avg_required_total",
                "avg_swebench_f2p_total",
            ],
        )
    )
    lines.extend(["", "## By Required Test Count Bucket", ""])
    lines.extend(
        markdown_table(
            by_f2p_bucket_rows,
            [
                "f2p_total_bucket",
                "interface_kind",
                "n",
                "passed",
                "pass_rate",
                "avg_required_total",
                "avg_swebench_f2p_total",
            ],
        )
    )
    lines.extend(["", "## By Alignment Verdict And Required Test Bucket", ""])
    lines.extend(
        markdown_table(
            by_alignment_f2p_rows,
            [
                "alignment_verdict",
                "interface_kind",
                "f2p_total_bucket",
                "n",
                "passed",
                "pass_rate",
                "avg_required_total",
                "avg_swebench_f2p_total",
            ],
        )
    )
    lines.extend(["", "## By Repo", ""])
    lines.extend(
        markdown_table(
            top_repo_rows,
            [
                "repo",
                "interface_kind",
                "n",
                "passed",
                "pass_rate",
                "avg_required_total",
                "avg_swebench_f2p_total",
            ],
        )
    )
    lines.extend(["", "## By Eval Verdict", ""])
    lines.extend(
        markdown_table(
            by_eval_rows,
            [
                "new_eval_verdict",
                "interface_kind",
                "n",
                "passed",
                "pass_rate",
                "avg_required_total",
                "avg_swebench_f2p_total",
            ],
        )
    )
    lines.extend(["", "## Matched Cell Check", ""])
    lines.extend(
        [
            "Matched cells use `(repo, eval verdict, alignment verdict, required test-count bucket)`. "
            "This asks whether meaningful-interface tasks still underperform after comparing "
            "only against placeholder-interface tasks in the same broad strata.",
            "",
            "| metric | value |",
            "| --- | --- |",
        ]
    )
    for key, value in matched_summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "Matched cells:"])
    lines.extend(
        markdown_table(
            matched_rows,
            ["repo", "eval", "alignment", "f2p_bucket", "placeholder", "meaningful"],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Raw pass rate barely changes by interface kind: meaningful is 57.9% and placeholder-no-new is 56.8%. So meaningful interface metadata is not an overall negative predictor by itself.",
            "- The interesting interaction is with alignment verdict: leaked + meaningful interface is 43.2%, while leaked + placeholder-no-new is 63.6%; OK + meaningful is 65.2%. The leaked underperformance is concentrated in the meaningful-interface slice.",
            "- Required test count is a plausible confounder inside that slice: leaked + meaningful averages 112 required tests, while OK + meaningful averages 53. But required count alone is not sufficient, since leaked + placeholder-no-new averages 123 required tests and still passes at 63.6%.",
            "- The matched cell check is noisy because many strata are small, but it does not support a stable overall penalty for meaningful interfaces. Matched meaningful-interface tasks pass at 60.5%, versus 51.8% expected if they followed placeholder rates in the same cells.",
            "- Requirements do not provide a usable confounder split here because every candidate has meaningful requirements text.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--swebench-pro", type=Path, default=DEFAULT_SWEBENCH_PRO)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    interface_metadata = load_interface_metadata(args.state)
    swebench_counts = load_swebench_counts(args.swebench_pro)
    rows = load_rows(args.enriched, interface_metadata, swebench_counts)
    matched_rows, matched_summary = matched_interface_adjustment(rows)
    write_csv(rows, args.output_csv)
    make_report(rows, matched_rows, matched_summary, args.report, args.output_csv)

    print(f"rows: {len(rows)}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.report}")


if __name__ == "__main__":
    main()
