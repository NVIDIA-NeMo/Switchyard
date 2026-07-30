#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize pipeline stage labels against SWE-bench agent run outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STAGE_ORDER = ["promising", "rejected"]
STAGE_COLORS = {
    "promising": "#2E8B57",
    "rejected": "#D95F02",
}
ALIGNMENT_ORDER = ["ok", "narrow_tests", "leaked"]
ALIGNMENT_COLORS = {
    "ok": "#2E8B57",
    "narrow_tests": "#D95F02",
    "leaked": "#7570B3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "agent_runs",
        type=Path,
        help="CSV with agent run metadata, or Harbor aggregate result.json",
    )
    parser.add_argument("pipeline_state_json", type=Path, help="Pipeline state.json file")
    parser.add_argument(
        "--label-field",
        choices=["stage", "alignment_verdict"],
        default="stage",
        help="State field to compare against agent outcomes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/pipeline-agent-comparison"),
        help="Directory for generated figures",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#24292F",
            "text.color": "#24292F",
            "xtick.color": "#57606A",
            "ytick.color": "#57606A",
            "grid.color": "#D8DEE4",
            "grid.alpha": 0.5,
        }
    )


def _label_config(label_field: str) -> tuple[list[str], dict[str, str], str]:
    if label_field == "alignment_verdict":
        return ALIGNMENT_ORDER, ALIGNMENT_COLORS, "alignment_label"
    return STAGE_ORDER, STAGE_COLORS, "pipeline_stage"


def load_pipeline_df(path: Path, label_field: str) -> pd.DataFrame:
    label_order, _label_colors, output_col = _label_config(label_field)
    state = json.loads(path.read_text())
    rows: list[dict[str, str]] = []
    for task_id, task in state["tasks"].items():
        source_metadata = ((task.get("candidate_data") or {}).get("source_metadata") or {})
        instance_id = source_metadata.get("instance_id")
        label = task.get(label_field)
        if not instance_id or label not in label_order:
            continue
        rows.append(
            {
                "task_id": task_id,
                "instance_id": instance_id,
                output_col: label,
            }
        )
    return pd.DataFrame(rows)


def shorten_model_name(name: str) -> str:
    cleaned = (
        name.replace(" -- ", " ")
        .replace(" - paper", "")
        .replace(" - 10132025", "")
        .replace(" -- 10222025", "")
        .replace(" -- debug-oct22", " debug")
        .strip()
    )
    return cleaned


def load_agent_runs_csv_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        [
            "metadata.instance_id",
            "metadata.model_name",
            "metadata.resolved",
            "metadata.turns",
        ]
    ].rename(
        columns={
            "metadata.instance_id": "instance_id",
            "metadata.model_name": "model_name",
            "metadata.resolved": "resolved",
            "metadata.turns": "turns",
        }
    )
    df["resolved"] = df["resolved"].astype(str).str.lower().eq("true")
    df["turns"] = pd.to_numeric(df["turns"], errors="coerce")
    df["model_label"] = df["model_name"].map(shorten_model_name)
    return df.dropna(subset=["instance_id", "model_name"])


def _model_name_from_harbor_eval_key(eval_key: str) -> str:
    parts = eval_key.split("__")
    if len(parts) >= 2:
        return parts[1]
    return eval_key


def _task_name_to_instance_id(task_name: object) -> str:
    return str(task_name or "").rstrip("/").split("/")[-1]


def _model_name_from_trial_result(trial_result: dict) -> str:
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


def load_harbor_result_df(path: Path) -> pd.DataFrame:
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
        for reward_value, instance_ids in reward_stats.items():
            try:
                resolved = float(reward_value) > 0.0
            except (TypeError, ValueError):
                continue
            if not isinstance(instance_ids, list):
                continue
            for trial_name in instance_ids:
                if trial_name:
                    trial_metadata = _load_harbor_trial_metadata(path.parent, str(trial_name))
                    rows.append(
                        {
                            "instance_id": trial_metadata.get("instance_id", str(trial_name)),
                            "model_name": trial_metadata.get("model_name", model_name),
                            "resolved": resolved,
                            "turns": pd.NA,
                        }
                    )

    if not rows:
        raise ValueError(f"{path}: no trial rows found in Harbor reward_stats")

    df = pd.DataFrame(rows)
    df["turns"] = pd.to_numeric(df["turns"], errors="coerce")
    df["model_label"] = df["model_name"].map(shorten_model_name)
    return df


def load_agent_runs_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return load_harbor_result_df(path)
    return load_agent_runs_csv_df(path)


def build_joined_df(agent_runs: Path, pipeline_state_json: Path, label_field: str) -> pd.DataFrame:
    label_order, _label_colors, output_col = _label_config(label_field)
    runs_df = load_agent_runs_df(agent_runs)
    pipeline_df = load_pipeline_df(pipeline_state_json, label_field)
    joined = runs_df.merge(pipeline_df, on="instance_id", how="inner")
    if joined.empty:
        raise RuntimeError("No rows matched between the agent runs input and pipeline state.json")

    model_order = (
        joined.groupby("model_label")["resolved"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    joined[output_col] = pd.Categorical(
        joined[output_col], categories=label_order, ordered=True
    )
    joined["model_label"] = pd.Categorical(joined["model_label"], categories=model_order, ordered=True)
    return joined


def annotate_bar_percentages(ax: plt.Axes) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if pd.isna(height) else f"{height * 100:.0f}%")
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, color="#24292F")


def plot_solve_rate(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    label_order, label_colors, output_col = _label_config(label_field)
    summary = (
        df.groupby(["model_label", output_col], observed=True)["resolved"]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.scatterplot(
        data=summary,
        x="model_label",
        y="mean",
        hue=output_col,
        hue_order=label_order,
        palette=label_colors,
        style=output_col,
        s=180,
        ax=ax,
    )
    for _, row in summary.iterrows():
        ax.text(
            row["model_label"],
            row["mean"] + 0.02,
            f"{row['mean'] * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#24292F",
        )
    ax.set_title(f"Solve Rate by Model and {label_field}")
    ax.set_xlabel("")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.legend(title="", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_solved_turns(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    label_order, label_colors, output_col = _label_config(label_field)
    solved = df[df["resolved"] & df["turns"].notna()].copy()
    if solved.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(
        data=solved,
        x="model_label",
        y="turns",
        hue=output_col,
        hue_order=label_order,
        palette=label_colors,
        ax=ax,
        showfliers=False,
    )
    ax.set_title(f"Turns on Solved Tasks by Model and {label_field}")
    ax.set_xlabel("")
    ax.set_ylabel("Agent turns")
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.legend(title="", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stage_delta_heatmap(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    if label_field != "stage":
        return
    summary = (
        df.groupby(["model_label", "pipeline_stage"], observed=True)["resolved"]
        .mean()
        .unstack("pipeline_stage")
    )
    if summary.empty:
        return

    heatmap_df = pd.DataFrame(
        {
            "promising - rejected": summary["promising"] - summary["rejected"],
        }
    ).sort_values("promising - rejected", ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, max(5, len(heatmap_df) * 0.42)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Solve-rate delta"},
        ax=ax,
    )
    ax.set_title("Solve-Rate Delta\nPromising Minus Rejected")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def build_instance_entropy_df(df: pd.DataFrame, label_field: str) -> pd.DataFrame:
    _label_order, _label_colors, output_col = _label_config(label_field)
    summary = (
        df.groupby(["instance_id", output_col], observed=True)["resolved"]
        .agg(models_evaluated="size", solve_rate="mean", solved_models="sum")
        .reset_index()
    )
    summary["binary_entropy_bits"] = summary["solve_rate"].map(binary_entropy)
    summary["normalized_entropy"] = summary["binary_entropy_bits"]
    return summary.sort_values("binary_entropy_bits", ascending=False)


def plot_instance_entropy_histogram(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    label_order, label_colors, output_col = _label_config(label_field)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="binary_entropy_bits",
        hue=output_col,
        hue_order=label_order,
        palette=label_colors,
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.2,
        ax=ax,
    )
    ax.set_title(f"Instance Discrimination via Binary Entropy by {label_field}")
    ax.set_xlabel("Binary entropy of solve outcomes across models (bits)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_instance_entropy_strip(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    label_order, label_colors, output_col = _label_config(label_field)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(
        data=df,
        x=output_col,
        y="binary_entropy_bits",
        order=label_order,
        hue=output_col,
        hue_order=label_order,
        palette=label_colors,
        dodge=False,
        jitter=0.28,
        alpha=0.7,
        size=6,
        ax=ax,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title(f"Instance Entropy by {label_field}")
    ax.set_xlabel("")
    ax.set_ylabel("Binary entropy of solve outcomes across models (bits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_instance_entropy_violin(df: pd.DataFrame, out_path: Path, label_field: str) -> None:
    label_order, label_colors, output_col = _label_config(label_field)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x=output_col,
        y="binary_entropy_bits",
        order=label_order,
        hue=output_col,
        hue_order=label_order,
        palette=label_colors,
        dodge=False,
        inner="box",
        cut=0,
        ax=ax,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title(f"Instance Entropy by {label_field}")
    ax.set_xlabel("")
    ax.set_ylabel("Binary entropy of solve outcomes across models (bits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    output_col = "alignment_label" if "alignment_label" in df.columns else "pipeline_stage"
    summary = (
        df.groupby(["model_name", "model_label", output_col], observed=True)
        .agg(
            runs=("resolved", "size"),
            solve_rate=("resolved", "mean"),
            median_turns=("turns", "median"),
            solved_runs=("resolved", "sum"),
            median_turns_solved=("turns", lambda s: s[df.loc[s.index, "resolved"]].median()),
        )
        .reset_index()
        .sort_values(["model_label", output_col])
    )
    summary.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joined = build_joined_df(args.agent_runs, args.pipeline_state_json, args.label_field)
    instance_entropy = build_instance_entropy_df(joined, args.label_field)
    write_summary(joined, args.output_dir / "summary.csv")
    instance_entropy.to_csv(args.output_dir / "instance_entropy.csv", index=False)
    plot_solve_rate(joined, args.output_dir / "solve_rate_by_model_and_stage.png", args.label_field)
    plot_solved_turns(
        joined,
        args.output_dir / "turns_on_solved_tasks_by_model_and_stage.png",
        args.label_field,
    )
    plot_stage_delta_heatmap(joined, args.output_dir / "solve_rate_stage_heatmap.png", args.label_field)
    plot_instance_entropy_histogram(
        instance_entropy, args.output_dir / "instance_entropy_histogram_by_stage.png", args.label_field
    )
    plot_instance_entropy_strip(
        instance_entropy, args.output_dir / "instance_entropy_strip_by_stage.png", args.label_field
    )
    plot_instance_entropy_violin(
        instance_entropy, args.output_dir / "instance_entropy_violin_by_stage.png", args.label_field
    )

    print(f"Matched rows: {len(joined)}")
    print(f"Unique matched instances: {joined['instance_id'].nunique()}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
