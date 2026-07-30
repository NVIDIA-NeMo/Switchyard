#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze source-file extensions for selected tasks in a pipeline state file."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("deep")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#D0D7DE",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#24292F",
            "text.color": "#24292F",
            "xtick.color": "#57606A",
            "ytick.color": "#57606A",
            "grid.color": "#D8DEE4",
            "grid.alpha": 0.6,
        }
    )


def add_bar_labels(ax: plt.Axes, *, fmt: str = "{:.0f}", padding: float = 3.0) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if height <= 0 else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=padding, fontsize=10, color="#24292F")


def load_tasks(state_path: Path) -> list[dict[str, Any]]:
    data = json.loads(state_path.read_text())
    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"{state_path}: expected tasks to be a dict")
    return list(tasks.values())


def extension_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1]
    return ext.lower() if ext else "<no_ext>"


def task_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        candidate = task.get("candidate_data", {})
        source_files = candidate.get("source_files", []) or []
        extensions = [extension_for_path(path) for path in source_files if path]
        extension_set = sorted(set(extensions))
        has_py = ".py" in extension_set
        rows.append(
            {
                "repo": task.get("repo") or candidate.get("repo") or "<missing>",
                "task_id": candidate.get("source_task_id") or task.get("task_id") or "",
                "score": candidate.get("score", 0.0),
                "source_file_count": len(source_files),
                "extension_list": extension_set,
                "has_py_file": has_py,
            }
        )
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_repo_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    repo = (
        df.groupby("repo")
        .agg(
            total_tasks=("task_id", "count"),
            no_py_tasks=("has_py_file", lambda s: int((~s).sum())),
        )
        .sort_values("total_tasks", ascending=False)
    )

    plot_df = repo.reset_index().melt(
        id_vars="repo",
        value_vars=["total_tasks", "no_py_tasks"],
        var_name="metric",
        value_name="count",
    )
    label_map = {
        "total_tasks": "Selected",
        "no_py_tasks": "No .py files",
    }
    plot_df["metric"] = plot_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=plot_df,
        x="repo",
        y="count",
        hue="metric",
        order=repo.index.tolist(),
        hue_order=["Selected", "No .py files"],
        palette=["#4C78A8", "#F58518"],
        ax=ax,
    )
    ax.set_ylabel("Task count")
    ax.set_xlabel("")
    ax.set_title("Selected tasks by repo", pad=18)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="", frameon=False, loc="upper right")
    add_bar_labels(ax)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_extensions(df: pd.DataFrame, out_path: Path, *, by: str, title: str) -> None:
    counter: Counter[str] = Counter()
    for ext_list in df["extension_list"]:
        if by == "task":
            counter.update(set(ext_list))
        elif by == "file":
            counter.update(ext_list)
        else:
            raise ValueError(by)

    top_items = counter.most_common(15)
    labels = [item[0] for item in top_items]
    values = [item[1] for item in top_items]

    plot_df = pd.DataFrame({"label": labels[::-1], "value": values[::-1]})

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    sns.barplot(
        data=plot_df,
        x="value",
        y="label",
        hue="label",
        dodge=False,
        palette="blend:#9ecae1,#08519c",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title(title, pad=18)
    for patch, value in zip(ax.patches, values[::-1]):
        ax.text(value + max(values) * 0.01, patch.get_y() + patch.get_height() / 2, str(value), va="center", fontsize=10)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_python_presence(df: pd.DataFrame, out_path: Path) -> None:
    total = len(df)
    counts = {
        "Has .py file": int(df["has_py_file"].sum()),
        "No .py files": int((~df["has_py_file"]).sum()),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_df = pd.DataFrame({"label": list(counts.keys()), "value": list(counts.values())})
    sns.barplot(
        data=plot_df,
        x="label",
        y="value",
        hue="label",
        dodge=False,
        palette=["#2E8B57", "#D95F02", "#756BB1"],
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Task count")
    ax.set_xlabel("")
    ax.set_title(f"Python presence in selected tasks (n={total})", pad=18)
    ax.set_ylim(0, max(counts.values()) * 1.12)
    for patch, value in zip(ax.patches, counts.values()):
        ax.text(patch.get_x() + patch.get_width() / 2, value + max(total * 0.01, 1), f"{int(value)}", ha="center", fontsize=11)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_report(df: pd.DataFrame, out_path: Path) -> None:
    total = len(df)
    no_py = int((~df["has_py_file"]).sum())

    repo = (
        df.groupby("repo")
        .agg(
            selected_tasks=("task_id", "count"),
            no_py_tasks=("has_py_file", lambda s: int((~s).sum())),
        )
        .sort_values("selected_tasks", ascending=False)
    )

    task_ext_counter: Counter[str] = Counter()
    file_ext_counter: Counter[str] = Counter()
    for ext_list in df["extension_list"]:
        task_ext_counter.update(set(ext_list))
        file_ext_counter.update(ext_list)

    lines = [
        "# Selected Task Extension Report",
        "",
        f"- Total selected tasks: `{total}`",
        f"- Tasks with no `.py` source files: `{no_py}` (`{no_py / total:.1%}`)",
        "",
        "## By Repo",
        "",
        "| Repo | Selected | No `.py` |",
        "| --- | ---: | ---: |",
    ]
    for repo_name, row in repo.iterrows():
        lines.append(f"| {repo_name} | {int(row['selected_tasks'])} | {int(row['no_py_tasks'])} |")

    lines.extend(
        [
            "",
            "## Top Extensions By Task Coverage",
            "",
            "| Extension | Tasks |",
            "| --- | ---: |",
        ]
    )
    for ext, count in task_ext_counter.most_common(15):
        lines.append(f"| {ext} | {count} |")

    lines.extend(
        [
            "",
            "## Top Extensions By Source File Count",
            "",
            "| Extension | Files |",
            "| --- | ---: |",
        ]
    )
    for ext, count in file_ext_counter.most_common(15):
        lines.append(f"| {ext} | {count} |")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze selected-task source file extensions from a state.json.")
    parser.add_argument("--state", type=Path, required=True, help="Path to run state.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for reports and plots.")
    args = parser.parse_args()

    tasks = load_tasks(args.state)
    rows = task_rows(tasks)
    df = pd.DataFrame(rows)
    configure_plot_style()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_selected_tasks": int(len(df)),
        "tasks_with_no_py_files": int((~df["has_py_file"]).sum()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    repo_df = (
        df.groupby("repo")
        .agg(
            selected_tasks=("task_id", "count"),
            no_py_tasks=("has_py_file", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values("selected_tasks", ascending=False)
    )
    repo_df.to_csv(args.out_dir / "repo_stats.csv", index=False)

    task_ext_counter: Counter[str] = Counter()
    file_ext_counter: Counter[str] = Counter()
    for ext_list in df["extension_list"]:
        task_ext_counter.update(set(ext_list))
        file_ext_counter.update(ext_list)

    ext_rows = []
    for ext in sorted(set(task_ext_counter) | set(file_ext_counter)):
        ext_rows.append([ext, task_ext_counter.get(ext, 0), file_ext_counter.get(ext, 0)])
    write_csv(args.out_dir / "extension_stats.csv", ["extension", "tasks", "files"], ext_rows)

    plot_repo_breakdown(df, args.out_dir / "repo_breakdown.png")
    plot_top_extensions(df, args.out_dir / "top_extensions_by_task.png", by="task", title="Top source-file extensions by task coverage")
    plot_top_extensions(df, args.out_dir / "top_extensions_by_file.png", by="file", title="Top source-file extensions by file count")
    plot_python_presence(df, args.out_dir / "python_presence.png")
    build_report(df, args.out_dir / "report.md")

    print(f"Wrote analysis to {args.out_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
