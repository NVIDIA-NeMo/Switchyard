#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled OK-vs-leaked pass-rate comparisons."""

from __future__ import annotations

# ruff: noqa: E501,I001

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("docs/analyses/data/swebench-pro/findings/swebench_pro_difficulty_proxies_enriched.csv")
DEFAULT_REPORT = Path("docs/analyses/data/swebench-pro/findings/ok_vs_leaked_controlled_report.md")


def pct(num: float, den: float) -> str:
    if den == 0:
        return ""
    return f"{num / den * 100:.1f}%"


def rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def pp(value: float) -> str:
    return f"{value * 100:.1f} pp"


def bucket_required(total: int) -> str:
    if total <= 5:
        return "01_1-5"
    if total <= 20:
        return "02_6-20"
    if total <= 100:
        return "03_21-100"
    return "04_101+"


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


def raw_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = by[row["alignment_verdict"]]
        bucket[0] += 1
        bucket[1] += 1 if row["success"] else 0
    ok_n, ok_pass = by["ok"]
    leaked_n, leaked_pass = by["leaked"]
    ok_rate = rate(ok_pass, ok_n)
    leaked_rate = rate(leaked_pass, leaked_n)
    return {
        "ok": f"{ok_pass}/{ok_n} ({pct(ok_pass, ok_n)})",
        "leaked": f"{leaked_pass}/{leaked_n} ({pct(leaked_pass, leaked_n)})",
        "raw_ok_minus_leaked": pp(ok_rate - leaked_rate),
    }


def matched_summary(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, object]:
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        cell = tuple(row[key] for key in keys)
        slot = cells[cell][row["alignment_verdict"]]
        slot[0] += 1
        slot[1] += 1 if row["success"] else 0

    matched_cell_count = 0
    matched_ok_n = 0
    matched_ok_pass = 0
    matched_leaked_n = 0
    matched_leaked_pass = 0
    expected_leaked_passes_at_ok_rates = 0.0
    expected_ok_passes_at_leaked_rates = 0.0

    for by_verdict in cells.values():
        if "ok" not in by_verdict or "leaked" not in by_verdict:
            continue
        matched_cell_count += 1
        ok_n, ok_pass = by_verdict["ok"]
        leaked_n, leaked_pass = by_verdict["leaked"]
        ok_rate = rate(ok_pass, ok_n)
        leaked_rate = rate(leaked_pass, leaked_n)

        matched_ok_n += ok_n
        matched_ok_pass += ok_pass
        matched_leaked_n += leaked_n
        matched_leaked_pass += leaked_pass
        expected_leaked_passes_at_ok_rates += ok_rate * leaked_n
        expected_ok_passes_at_leaked_rates += leaked_rate * ok_n

    matched_ok_rate = rate(matched_ok_pass, matched_ok_n)
    matched_leaked_rate = rate(matched_leaked_pass, matched_leaked_n)
    leaked_expected_ok_rate = rate(expected_leaked_passes_at_ok_rates, matched_leaked_n)
    ok_expected_leaked_rate = rate(expected_ok_passes_at_leaked_rates, matched_ok_n)

    return {
        "controls": " + ".join(keys),
        "matched_cells": matched_cell_count,
        "matched_ok": f"{matched_ok_pass}/{matched_ok_n} ({pct(matched_ok_pass, matched_ok_n)})",
        "matched_leaked": f"{matched_leaked_pass}/{matched_leaked_n} ({pct(matched_leaked_pass, matched_leaked_n)})",
        "matched_raw_gap": pp(matched_ok_rate - matched_leaked_rate),
        "ok_rate_on_leaked_mix": pct(expected_leaked_passes_at_ok_rates, matched_leaked_n),
        "leaked_shortfall_vs_cell_ok": pp(leaked_expected_ok_rate - matched_leaked_rate),
        "leaked_rate_on_ok_mix": pct(expected_ok_passes_at_leaked_rates, matched_ok_n),
        "ok_advantage_vs_cell_leaked": pp(matched_ok_rate - ok_expected_leaked_rate),
        "matched_ok_n": matched_ok_n,
        "matched_leaked_n": matched_leaked_n,
    }


def cell_detail_rows(
    rows: list[dict[str, object]], keys: tuple[str, ...], max_rows: int = 40
) -> list[dict[str, object]]:
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        cell = tuple(row[key] for key in keys)
        slot = cells[cell][row["alignment_verdict"]]
        slot[0] += 1
        slot[1] += 1 if row["success"] else 0
    out = []
    for cell, by_verdict in cells.items():
        if "ok" not in by_verdict or "leaked" not in by_verdict:
            continue
        ok_n, ok_pass = by_verdict["ok"]
        leaked_n, leaked_pass = by_verdict["leaked"]
        out.append(
            {
                "cell": " / ".join(str(part) for part in cell),
                "ok": f"{ok_pass}/{ok_n} ({pct(ok_pass, ok_n)})",
                "leaked": f"{leaked_pass}/{leaked_n} ({pct(leaked_pass, leaked_n)})",
                "cell_n": ok_n + leaked_n,
            }
        )
    return sorted(out, key=lambda row: int(row["cell_n"]), reverse=True)[:max_rows]


def make_report(rows: list[dict[str, object]], report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if row["alignment_verdict"] in {"ok", "leaked"}]

    specs: list[tuple[str, tuple[str, ...]]] = [
        ("Repo", ("repo",)),
        ("Repo + Eval", ("repo", "new_eval_verdict")),
        ("Repo + Eval + True F2P", ("repo", "new_eval_verdict", "f2p_bucket")),
        ("Repo + Eval + Required", ("repo", "new_eval_verdict", "required_bucket")),
        (
            "Repo + Eval + Metadata Size",
            (
                "repo",
                "new_eval_verdict",
                "requirements_words_bucket",
                "test_patch_changed_bucket",
                "patch_changed_bucket",
            ),
        ),
        (
            "Repo + Eval + True F2P + Metadata Size",
            (
                "repo",
                "new_eval_verdict",
                "f2p_bucket",
                "requirements_words_bucket",
                "test_patch_changed_bucket",
                "patch_changed_bucket",
            ),
        ),
        (
            "Repo + Eval + True F2P + P2P + Metadata Size",
            (
                "repo",
                "new_eval_verdict",
                "f2p_bucket",
                "p2p_bucket",
                "requirements_words_bucket",
                "test_patch_changed_bucket",
                "patch_changed_bucket",
            ),
        ),
        (
            "Repo + Eval + Full Coarsened",
            (
                "repo",
                "new_eval_verdict",
                "f2p_bucket",
                "p2p_bucket",
                "requirements_words_bucket",
                "test_patch_changed_bucket",
                "patch_changed_bucket",
                "selected_test_files_bucket",
                "interface_meaningful",
            ),
        ),
    ]

    summary_rows = []
    for label, keys in specs:
        row = matched_summary(rows, keys)
        row["spec"] = label
        summary_rows.append(row)

    raw = raw_summary(rows)
    lines = [
        "# OK vs Leaked Controlled Pass Rate",
        "",
        "This compares `ok` and `leaked` tasks after controlling for coarse difficulty proxies from `swebench_pro.jsonl`.",
        "",
        "Method: coarsened exact matching/direct standardization. For each control set, keep only cells containing both `ok` and `leaked` tasks. `ok_rate_on_leaked_mix` answers: if leaked tasks had the OK pass rate in the same cells, what pass rate would we expect?",
        "",
        "## Raw",
        "",
        "| ok | leaked | raw_ok_minus_leaked |",
        "| --- | --- | --- |",
        f"| {raw['ok']} | {raw['leaked']} | {raw['raw_ok_minus_leaked']} |",
        "",
        "## Controlled Summaries",
        "",
    ]
    lines.extend(
        markdown_table(
            summary_rows,
            [
                "spec",
                "matched_cells",
                "matched_ok",
                "matched_leaked",
                "matched_raw_gap",
                "ok_rate_on_leaked_mix",
                "leaked_shortfall_vs_cell_ok",
                "matched_ok_n",
                "matched_leaked_n",
            ],
        )
    )

    detail_spec = (
        "repo",
        "new_eval_verdict",
        "f2p_bucket",
        "requirements_words_bucket",
        "test_patch_changed_bucket",
        "patch_changed_bucket",
    )
    lines.extend(
        [
            "",
            "## Largest Matched Cells",
            "",
            "Using `repo + eval + true F2P + requirements length + test patch size + source patch size`.",
            "",
        ]
    )
    lines.extend(markdown_table(cell_detail_rows(rows, detail_spec), ["cell", "ok", "leaked", "cell_n"]))

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Raw gap: OK is 62.5% and leaked is 52.9%, a 9.6 point OK advantage.",
            "- Controlling only for repo + eval verdict reduces the gap materially; leaked has a 2.8 point shortfall versus same-cell OK rates.",
            "- Adding true F2P buckets removes the same-cell shortfall: leaked is 3.7 points above the OK same-cell expectation in the matched subset.",
            "- Using verifier required-test buckets also removes the same-cell shortfall: leaked is 0.9 points above the OK same-cell expectation.",
            "- Controlling for metadata/diff-size buckets gets sparse, but the residual does not grow; in those matched subsets leaked is above same-cell OK expectation.",
            "- The full coarsened spec keeps only a small subset of leaked tasks, so treat it as diagnostic rather than definitive.",
            "- Practical interpretation: the raw OK-vs-leaked gap is mostly explained by composition across repo/eval/difficulty proxies. There is not strong evidence that leaked tasks underperform OK tasks within matched cells.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    with args.input.open() as f:
        rows = []
        for row in csv.DictReader(f):
            row["success"] = row["agent_success"].strip().lower() == "true"
            row["required_bucket"] = bucket_required(int(row["required_total"]))
            rows.append(row)
    make_report(rows, args.report)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.report}")


if __name__ == "__main__":
    main()
