#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretty-print model pass rates by pipeline stage and alignment verdict."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

STAGE_ORDER = ["rejected", "accepted", "promising"]
ALIGNMENT_ORDER = ["narrow_tests", "leaked", "ok", "vague", "misaligned", "skipped"]
MISSING_LABEL = "(missing)"
DISPLAY_LABELS = {
    "accepted": "Accepted",
    "alignment_checked": "Alignment checked",
    "built": "Built",
    "candidate": "Candidate",
    "dockerfile_built": "Dockerfile built",
    "evaluated": "Evaluated",
    "f2p_p2p_classified": "F2P/P2P classified",
    "leaked": "Leaked",
    "misaligned": "Misaligned",
    "narrow_tests": "Narrow test",
    "needs_fix": "Needs fix",
    "ok": "OK",
    "opus_smoke_tested": "Opus smoke tested",
    "opus_triaged": "Opus triaged",
    "oracle_checked": "Oracle checked",
    "promising": "Promising",
    "rejected": "Rejected",
    "skipped": "Skipped",
    "tests_discovered": "Tests discovered",
    "vague": "Vague",
}


def shorten_model_name(name: str) -> str:
    cleaned = (
        name.replace(" - paper", "")
        .replace(" - 10132025", "")
        .replace(" -- 10222025", "")
        .replace(" -- debug-oct22", " debug")
        .replace(" -- ", " ")
        .strip()
    )
    route = cleaned.split("/")[-1]
    claude_match = re.fullmatch(r"claude-(opus|sonnet)-(\d+)-(\d+)", route)
    if claude_match:
        family, major, minor = claude_match.groups()
        return f"Claude {family.title()} {major}.{minor}"
    gpt_match = re.fullmatch(r"gpt-(\d+(?:\.\d+)?)", route)
    if gpt_match:
        return f"GPT-{gpt_match.group(1)}"
    return cleaned


def _required_columns() -> list[str]:
    return [
        "metadata.instance_id",
        "metadata.model_name",
        "metadata.resolved",
    ]


def load_agent_runs_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in _required_columns() if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")

    out = df[_required_columns()].rename(
        columns={
            "metadata.instance_id": "instance_id",
            "metadata.model_name": "model_name",
            "metadata.resolved": "resolved",
        }
    )
    out["instance_id"] = out["instance_id"].astype(str).str.strip()
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["resolved"] = out["resolved"].astype(str).str.lower().isin({"true", "1", "yes"})
    out["model_label"] = out["model_name"].map(shorten_model_name)
    return out[(out["instance_id"] != "") & (out["model_name"] != "")]


def _model_name_from_harbor_eval_key(eval_key: str) -> str:
    parts = eval_key.split("__")
    if len(parts) >= 2:
        return parts[1]
    return eval_key


def _task_name_to_instance_id(task_name: object) -> str:
    return str(task_name or "").rstrip("/").split("/")[-1]


def _model_name_from_trial_result(trial_result: dict[str, Any]) -> str:
    model_info = ((trial_result.get("agent_info") or {}).get("model_info") or {})
    provider = str(model_info.get("provider") or "").strip()
    name = str(model_info.get("name") or "").strip()
    if provider and name and not name.startswith(f"{provider}/"):
        return f"{provider}/{name}"
    if name:
        return name

    config_agent = ((trial_result.get("config") or {}).get("agent") or {})
    return str(config_agent.get("model_name") or "").strip()


def _load_harbor_trial_metadata(job_dir: Path, trial_name: str) -> dict[str, str]:
    trial_result_path = job_dir / trial_name / "result.json"
    if not trial_result_path.exists():
        return {}
    try:
        trial_result = json.loads(trial_result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    instance_id = _task_name_to_instance_id(trial_result.get("task_name"))
    if not instance_id:
        task_id = trial_result.get("task_id") or {}
        if isinstance(task_id, dict):
            instance_id = _task_name_to_instance_id(task_id.get("path"))

    model_name = _model_name_from_trial_result(trial_result)
    out: dict[str, str] = {}
    if instance_id:
        out["instance_id"] = instance_id
    if model_name:
        out["model_name"] = model_name
    return out


def load_harbor_result_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    evals = ((data.get("stats") or {}).get("evals") or {})
    if not isinstance(evals, dict) or not evals:
        raise ValueError(f"{path}: expected Harbor result JSON with stats.evals")

    rows: list[dict[str, object]] = []
    for eval_key, eval_data in evals.items():
        if not isinstance(eval_data, dict):
            continue
        model_name = _model_name_from_harbor_eval_key(str(eval_key))
        reward_stats = ((eval_data.get("reward_stats") or {}).get("reward") or {})
        if not isinstance(reward_stats, dict):
            continue
        for reward_value, trial_names in reward_stats.items():
            try:
                resolved = float(reward_value) > 0.0
            except (TypeError, ValueError):
                continue
            if not isinstance(trial_names, list):
                continue
            for trial_name in trial_names:
                if not trial_name:
                    continue
                trial_metadata = _load_harbor_trial_metadata(path.parent, str(trial_name))
                rows.append(
                    {
                        "instance_id": trial_metadata.get("instance_id", str(trial_name)),
                        "model_name": trial_metadata.get("model_name", model_name),
                        "resolved": resolved,
                    }
                )

    if not rows:
        raise ValueError(f"{path}: no trial rows found in Harbor reward_stats")

    df = pd.DataFrame(rows)
    df["model_label"] = df["model_name"].map(shorten_model_name)
    return df.dropna(subset=["instance_id", "model_name"])


def load_agent_runs(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return load_harbor_result_json(path)
    return load_agent_runs_csv(path)


def _task_instance_id(task: dict[str, Any]) -> str:
    candidate = task.get("candidate_data") or {}
    source_meta = candidate.get("source_metadata") or {}
    return str(source_meta.get("instance_id") or candidate.get("source_task_id") or "").strip()


def _label(value: object) -> str:
    text = str(value or "").strip()
    return text or MISSING_LABEL


def load_state_labels(path: Path) -> pd.DataFrame:
    state = json.loads(path.read_text())
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{path}: expected top-level 'tasks' dict")

    rows: list[dict[str, str]] = []
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue
        instance_id = _task_instance_id(task)
        if not instance_id:
            continue
        rows.append(
            {
                "task_id": str(task.get("task_id") or task_id),
                "instance_id": instance_id,
                "stage": _label(task.get("stage")),
                "alignment_verdict": _label(task.get("alignment_verdict")),
            }
        )
    return pd.DataFrame(rows)


def _ordered_unique(values: Iterable[str], preferred: list[str]) -> list[str]:
    seen = {value for value in values if value}
    ordered = [value for value in preferred if value in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def build_joined_df(agent_runs: Path, state_json: Path) -> pd.DataFrame:
    runs = load_agent_runs(agent_runs)
    labels = load_state_labels(state_json)
    joined = runs.merge(labels, on="instance_id", how="inner")
    if joined.empty:
        raise RuntimeError("No rows matched between the agent-runs input and pipeline state.json")
    return joined


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def _format_rate(resolved: pd.Series, *, include_counts: bool) -> str:
    total = int(resolved.size)
    if total == 0:
        return "n/a"
    solved = int(resolved.sum())
    text = f"{resolved.mean() * 100:.1f}%"
    if include_counts:
        text += f" ({solved}/{total})"
    return text


def _ordered_labels(values: pd.Series, preferred: list[str]) -> list[str]:
    return _ordered_unique(values.astype(str), preferred)


def pretty_summary(joined: pd.DataFrame, *, include_counts: bool = True) -> str:
    models = (
        joined.groupby("model_label", observed=True)["resolved"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    stages = _ordered_labels(joined["stage"], STAGE_ORDER)
    alignment_verdicts = _ordered_labels(joined["alignment_verdict"], ALIGNMENT_ORDER)

    blocks: list[str] = []
    for model in models:
        model_df = joined[joined["model_label"].astype(str) == model]
        lines = [
            f"-----{model}-----",
            f"overall: {_format_rate(model_df['resolved'], include_counts=include_counts)}",
        ]
        for stage in stages:
            stage_df = model_df[model_df["stage"].astype(str) == stage]
            if not stage_df.empty:
                rate = _format_rate(stage_df["resolved"], include_counts=include_counts)
                lines.append(f"{_display_label(stage)}: {rate}")
        for verdict in alignment_verdicts:
            verdict_df = model_df[model_df["alignment_verdict"].astype(str) == verdict]
            if not verdict_df.empty:
                rate = _format_rate(verdict_df["resolved"], include_counts=include_counts)
                lines.append(f"{_display_label(verdict)}: {rate}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def write_pretty_summary(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")


def build_matrix(
    joined: pd.DataFrame,
    *,
    include_counts: bool = True,
    as_percent: bool = False,
) -> pd.DataFrame:
    grouped = (
        joined.groupby(["stage", "alignment_verdict", "model_label"], observed=True)["resolved"]
        .agg(pass_rate="mean", runs="size", solved="sum")
        .reset_index()
    )
    if as_percent:
        grouped["pass_rate"] = grouped["pass_rate"] * 100.0

    stages = _ordered_unique(joined["stage"].astype(str), STAGE_ORDER)
    alignment_verdicts = _ordered_unique(joined["alignment_verdict"].astype(str), ALIGNMENT_ORDER)
    models = (
        joined.groupby("model_label", observed=True)["resolved"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )

    row_totals = (
        joined.groupby(["stage", "alignment_verdict"], observed=True)
        .agg(matched_agent_runs=("resolved", "size"), matched_instances=("instance_id", "nunique"))
        .reset_index()
    )

    pass_rates = grouped.pivot(
        index=["stage", "alignment_verdict"],
        columns="model_label",
        values="pass_rate",
    )
    runs = grouped.pivot(index=["stage", "alignment_verdict"], columns="model_label", values="runs")

    index = pd.MultiIndex.from_product([stages, alignment_verdicts], names=["stage", "alignment_verdict"])
    matrix = row_totals.set_index(["stage", "alignment_verdict"]).reindex(index)
    matrix["matched_agent_runs"] = matrix["matched_agent_runs"].fillna(0).astype(int)
    matrix["matched_instances"] = matrix["matched_instances"].fillna(0).astype(int)

    pass_rates = pass_rates.reindex(index)
    runs = runs.reindex(index)
    for model in models:
        matrix[f"{model} pass_rate"] = pass_rates.get(model)
        if include_counts:
            matrix[f"{model} runs"] = runs.get(model).fillna(0).astype(int)

    return matrix.reset_index()


def write_matrix(matrix: pd.DataFrame, output: Path | None) -> None:
    if output is None:
        matrix.to_csv(sys.stdout, index=False, quoting=csv.QUOTE_MINIMAL)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_runs", type=Path, help="Agent-runs CSV or Harbor aggregate result.json")
    parser.add_argument("state_json", type=Path, help="Pipeline state.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path; default stdout")
    parser.add_argument(
        "--format",
        choices=["pretty", "csv"],
        default="pretty",
        help="Pretty-print per-model rates by default, or emit the old CSV matrix.",
    )
    parser.add_argument(
        "--no-counts",
        action="store_true",
        help="In pretty mode, omit solved/total counts. In CSV mode, omit per-model run-count columns.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="CSV mode only: emit pass rates as 0-100 percentages instead of 0-1 fractions.",
    )
    args = parser.parse_args()

    try:
        joined = build_joined_df(args.agent_runs, args.state_json)
        if args.format == "pretty":
            text = pretty_summary(joined, include_counts=not args.no_counts)
            write_pretty_summary(text, args.output)
        else:
            matrix = build_matrix(joined, include_counts=not args.no_counts, as_percent=args.percent)
            write_matrix(matrix, args.output)
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if args.output and args.format == "csv":
        print(
            f"Wrote {len(matrix)} matrix rows to {args.output} "
            f"({len(joined)} matched agent-run rows, {joined['instance_id'].nunique()} instances)"
        )
    elif args.output:
        print(
            f"Wrote summary to {args.output} "
            f"({len(joined)} matched agent-run rows, {joined['instance_id'].nunique()} instances)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
