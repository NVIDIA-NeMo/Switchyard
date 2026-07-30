#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create quick-look charts from a pipeline state.json file."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VERDICT_ORDER = ["PROMISING", "MAYBE", "REJECT", "ERROR"]
VERDICT_COLORS = {
    "PROMISING": "#2E8B57",
    "MAYBE": "#D4A017",
    "REJECT": "#D95F02",
    "ERROR": "#B42318",
}

STAGE_ORDER = [
    "candidate",
    "evaluated",
    "promising",
    "built",
    "hardness_checked",
    "tests_discovered",
    "dockerfile_built",
    "f2p_p2p_classified",
    "oracle_checked",
    "opus_smoke_tested",
    "opus_triaged",
    "haiku_smoke_tested",
    "accepted",
    "needs_fix",
    "rejected",
]


def load_state(path: Path, retries: int = 8, delay_s: float = 0.4) -> dict:
    """Retry reads because state.json may be updated while the pipeline runs."""
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with path.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(delay_s)
    raise RuntimeError(f"Could not read a stable JSON snapshot from {path}: {last_error}") from last_error


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
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


def task_repo(task: dict) -> str:
    return task.get("repo") or task.get("candidate_data", {}).get("repo") or "<missing>"


def top_repos(tasks: dict[str, dict], *, limit: int) -> list[str]:
    counts = Counter(task_repo(task) for task in tasks.values())
    return [repo for repo, _ in counts.most_common(limit)]


def add_bar_labels(ax: plt.Axes, *, fmt: str = "{:.0f}", padding: float = 3.0) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            width = bar.get_width()
            labels.append("" if width <= 0 else fmt.format(width))
        ax.bar_label(container, labels=labels, padding=padding, fontsize=9, color="#24292F")


def build_repo_verdict_df(tasks: dict[str, dict], repos: list[str]) -> pd.DataFrame:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks.values():
        repo = task_repo(task)
        verdict = (task.get("eval_verdict") or "").strip()
        if repo in repos and verdict:
            counts[repo][verdict] += 1

    rows = []
    for repo in repos:
        for verdict in VERDICT_ORDER:
            rows.append({"repo": repo, "verdict": verdict, "count": counts[repo][verdict]})
    return pd.DataFrame(rows)


def plot_repo_verdicts(tasks: dict[str, dict], repos: list[str], out_path: Path) -> None:
    df = build_repo_verdict_df(tasks, repos)
    if df.empty:
        return

    pivot = df.pivot(index="repo", columns="verdict", values="count").fillna(0)
    pivot = pivot[[v for v in VERDICT_ORDER if v in pivot.columns]]

    fig, ax = plt.subplots(figsize=(13, max(7, len(repos) * 0.5)))
    left = [0] * len(pivot.index)
    y = range(len(pivot.index))
    for verdict in pivot.columns:
        values = pivot[verdict].tolist()
        ax.barh(
            y,
            values,
            left=left,
            label=verdict,
            color=VERDICT_COLORS[verdict],
            edgecolor="white",
        )
        left = [l + v for l, v in zip(left, values)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Task count")
    ax.set_ylabel("")
    ax.set_title("Eval Verdicts by Repo")
    ax.legend(title="", frameon=False, ncol=4, loc="lower right")
    add_bar_labels(ax)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_repo_stage_df(tasks: dict[str, dict], repos: list[str]) -> pd.DataFrame:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks.values():
        repo = task_repo(task)
        stage = task.get("stage", "?")
        if repo in repos:
            counts[repo][stage] += 1

    rows = []
    for repo in repos:
        for stage in STAGE_ORDER:
            rows.append({"repo": repo, "stage": stage, "count": counts[repo][stage]})
    return pd.DataFrame(rows)


def plot_repo_stage_breakdown(tasks: dict[str, dict], repos: list[str], out_path: Path) -> None:
    df = build_repo_stage_df(tasks, repos)
    if df.empty:
        return

    pivot = df.pivot(index="repo", columns="stage", values="count").fillna(0)
    pivot = pivot[[s for s in STAGE_ORDER if s in pivot.columns and pivot[s].sum() > 0]]

    palette = sns.color_palette("Spectral", n_colors=max(len(pivot.columns), 3))
    stage_colors = {stage: palette[i] for i, stage in enumerate(pivot.columns)}

    fig, ax = plt.subplots(figsize=(13, max(7, len(repos) * 0.5)))
    left = [0] * len(pivot.index)
    y = range(len(pivot.index))
    for stage in pivot.columns:
        values = pivot[stage].tolist()
        ax.barh(
            y,
            values,
            left=left,
            label=stage,
            color=stage_colors[stage],
            edgecolor="white",
        )
        left = [l + v for l, v in zip(left, values)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Task count")
    ax.set_ylabel("")
    ax.set_title("Current Stage Breakdown by Repo")
    ax.legend(title="", frameon=False, ncol=3, loc="lower right", fontsize=9)
    add_bar_labels(ax)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_promising_rate_df(tasks: dict[str, dict], repos: list[str]) -> pd.DataFrame:
    attempted: Counter[str] = Counter()
    promising_stage: Counter[str] = Counter()
    for task in tasks.values():
        repo = task_repo(task)
        if repo not in repos:
            continue
        if task.get("eval_verdict"):
            attempted[repo] += 1
        if task.get("stage") == "promising":
            promising_stage[repo] += 1

    rows = []
    for repo in repos:
        if attempted[repo] == 0:
            continue
        rows.append(
            {
                "repo": repo,
                "attempted_eval": attempted[repo],
                "promising_stage": promising_stage[repo],
                "promising_rate_pct": 100.0 * promising_stage[repo] / attempted[repo],
            }
        )
    return pd.DataFrame(rows).sort_values("promising_rate_pct", ascending=False)


def plot_promising_rate(tasks: dict[str, dict], repos: list[str], out_path: Path) -> None:
    df = build_promising_rate_df(tasks, repos)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(6.5, len(df) * 0.45)))
    sns.barplot(
        data=df,
        x="promising_rate_pct",
        y="repo",
        hue="repo",
        dodge=False,
        palette="blend:#9ecae1,#08519c",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Percent of eval-attempted tasks currently in stage=promising")
    ax.set_ylabel("")
    ax.set_title("Promising Rate by Repo")
    for patch, value in zip(ax.patches, df["promising_rate_pct"]):
        ax.text(value + 0.6, patch.get_y() + patch.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_csv(tasks: dict[str, dict], repos: list[str], out_path: Path) -> None:
    rows = []
    for repo in repos:
        repo_tasks = [task for task in tasks.values() if task_repo(task) == repo]
        verdicts = Counter((task.get("eval_verdict") or "").strip() for task in repo_tasks if task.get("eval_verdict"))
        stages = Counter(task.get("stage", "?") for task in repo_tasks)
        attempted = sum(verdicts.values())
        rows.append(
            {
                "repo": repo,
                "tasks": len(repo_tasks),
                "attempted_eval": attempted,
                "verdict_promising": verdicts.get("PROMISING", 0),
                "verdict_maybe": verdicts.get("MAYBE", 0),
                "verdict_reject": verdicts.get("REJECT", 0),
                "verdict_error": verdicts.get("ERROR", 0),
                "stage_promising": stages.get("promising", 0),
                "stage_rejected": stages.get("rejected", 0),
                "stage_evaluated": stages.get("evaluated", 0),
                "promising_rate_pct": round((100.0 * stages.get("promising", 0) / attempted), 1) if attempted else 0.0,
            }
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["repo"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create charts from a craft-taskgen state.json file.")
    parser.add_argument("state_file", type=Path, help="Path to pipeline state.json")
    parser.add_argument("--out-dir", type=Path, default=Path("state-viz"), help="Output directory for charts")
    parser.add_argument("--top-repos", type=int, default=20, help="Limit charts to the top N repos by task count")
    args = parser.parse_args()

    configure_plot_style()
    state = load_state(args.state_file)
    tasks = state.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"{args.state_file}: expected top-level 'tasks' dict")

    repos = top_repos(tasks, limit=args.top_repos)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_repo_verdicts(tasks, repos, args.out_dir / "repo_eval_verdicts.png")
    plot_repo_stage_breakdown(tasks, repos, args.out_dir / "repo_stage_breakdown.png")
    plot_promising_rate(tasks, repos, args.out_dir / "repo_promising_rate.png")
    write_summary_csv(tasks, repos, args.out_dir / "repo_summary.csv")

    print(f"Wrote charts to {args.out_dir}")
    print(f"  - {args.out_dir / 'repo_eval_verdicts.png'}")
    print(f"  - {args.out_dir / 'repo_stage_breakdown.png'}")
    print(f"  - {args.out_dir / 'repo_promising_rate.png'}")
    print(f"  - {args.out_dir / 'repo_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
