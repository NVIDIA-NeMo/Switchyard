# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""analyze_search_vs_e2e.py — search baseline vs end-to-end correlation analysis.

Loads the per-trial search CSV (from `summarize_search_baseline.py --csv`) and
the per-trial v2b end-to-end CSV (from `summarize_baseline.py --csv` or hand-
prepared). Joins them on `search.parent_task_id == e2e.task` and reports
correlations across multiple unit-of-analysis framings. Emits markdown to
stdout and per-cut paired-observation CSVs to `--out-dir`.

Framings tried (all reported, no single "right" one):
  A. per-(model, parent)     — average across iters within model;
                               headline correlation, ~196 points.
  B. per-iter-pair             — search iter i ↔ e2e iter i (mod min iters);
                               preserves variance, max 49 × 4 × min_iters.
  C. per-parent (model-mean)   — collapse models too; "is this task
                               intrinsically hard at both?", ~49 points.
  D. per-trial-pair            — every search trial × every same-model e2e
                               trial of the same task; lots of points but
                               non-independent.

Search-aggregation (when multiple search variants share a parent T2 task):
  - mean across variants  (default — clean, what the headline reports)
  - max across variants   (robust to "did the model find it at all in any
                          variant?")
Both reported side-by-side.

Models: codex55 / opus47 / haiku45 / qwen36 (minimax dropped — no e2e baseline).

Stats: Pearson + Spearman + point-biserial (where applicable). p-values, n.
For binary `resolved`, Pearson collapses to point-biserial automatically.

Usage:
  uv run python scripts/analyze_search_vs_e2e.py \\
      --search-csv /tmp/search-baseline.csv \\
      --e2e-csv ~/Downloads/v2b-e2e-baselines-partial.csv \\
      --out-dir /tmp/analysis-out
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Model canonicalization
# ---------------------------------------------------------------------------
#
# Both CSVs identify models with platform-specific strings. Map both to a small
# canonical label set so we can join cleanly.
#
# Search side examples:
#   "codex / openai/openai/gpt-5.5 / effort=xhigh"       -> codex55
#   "claude-code / aws/anthropic/bedrock-claude-opus-4-7 / effort=xhigh" -> opus47
#   "opencode / nvidia/azure/anthropic/claude-haiku-4-5 / effort=medium" -> haiku45
#   "opencode / qwenai/qwen3-36 (vllm)"                  -> qwen36
#   "opencode / minimaxai/minimax-m2.7 (vllm)"           -> minimax (DROPPED)
#
# E2E side examples (iter_label):
#   "codex55-iter0", "opus47-iter3", "haiku45-iter2", "qwen36-iter1"
# Stripped to the base model.

CANONICAL_MODELS = ("codex55", "opus47", "haiku45", "qwen36")


def canonical_search_model(agent: str, model: str) -> str | None:
    """Map a search-side (agent, model) pair to a canonical short label.

    Returns None for minimax (dropped) or anything unrecognized.
    """
    haystack = f"{agent} / {model}".lower()
    if "minimax" in haystack:
        return None
    if "gpt-5.5" in haystack or "gpt-55" in haystack or "gpt5.5" in haystack:
        return "codex55"
    if "opus-4-7" in haystack or "opus47" in haystack or "claude-opus-4-7" in haystack:
        return "opus47"
    if "haiku-4-5" in haystack or "haiku45" in haystack or "claude-haiku-4-5" in haystack:
        return "haiku45"
    if "qwen" in haystack:
        return "qwen36"
    return None


def canonical_e2e_model(iter_label: str) -> str | None:
    """Strip the `-iterN` suffix from an e2e iter_label."""
    m = re.match(r"^([a-z0-9]+)-iter\d+$", iter_label.lower())
    if not m:
        return None
    base = m.group(1)
    if base in CANONICAL_MODELS:
        return base
    return None


def e2e_iter_index(iter_label: str) -> int | None:
    m = re.match(r"^[a-z0-9]+-iter(\d+)$", iter_label.lower())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float | None, float | None, int]:
    """Return (r, p, n) for Pearson/Spearman, dropping NaN pairs. None on degenerate input."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3:
        return (None, None, n)
    if df.x.nunique() < 2 or df.y.nunique() < 2:
        # Constant column — correlation undefined.
        return (None, None, n)
    if method == "pearson":
        r, p = stats.pearsonr(df.x, df.y)
    elif method == "spearman":
        r, p = stats.spearmanr(df.x, df.y)
    else:
        raise ValueError(f"unknown method {method!r}")
    return (float(r), float(p), n)


def _fisher_ci(r: float | None, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Fisher-z 95% CI for Pearson r. None when undefined."""
    if r is None or n < 4 or abs(r) >= 1.0:
        return (None, None)
    import math

    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return (lo, hi)


def _star(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_corr(
    r: float | None, p: float | None, n: int, ci: tuple[float | None, float | None] | None = None
) -> str:
    if r is None:
        return f"— (n={n})"
    base = f"{r:+.3f}{_star(p)} (n={n}"
    if ci and ci[0] is not None:
        base += f", 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    base += ")"
    return base


# ---------------------------------------------------------------------------
# Load + canonicalize
# ---------------------------------------------------------------------------


def load_search(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["canonical_model"] = df.apply(lambda r: canonical_search_model(r["agent"], r["model"]), axis=1)
    n_total = len(df)
    df = df[df["canonical_model"].notna() & (df["infra_fail"] == 0)].copy()
    print(
        f"[search] {n_total} rows → {len(df)} after dropping minimax + infra_fail",
        file=sys.stderr,
    )
    df["search_iter"] = df["trial_index"].astype(int)
    return df


def load_e2e(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["canonical_model"] = df["iter_label"].map(canonical_e2e_model)
    df["e2e_iter"] = df["iter_label"].map(e2e_iter_index)
    n_total = len(df)
    df = df[df["canonical_model"].notna() & (df["infra_fail"] == 0)].copy()
    df["f2p_pass_rate"] = df.apply(
        lambda r: (r["f2p_passed"] / r["f2p_total"]) if r["f2p_total"] else float("nan"), axis=1
    )
    print(
        f"[e2e] {n_total} rows → {len(df)} after dropping infra_fail",
        file=sys.stderr,
    )
    return df


# ---------------------------------------------------------------------------
# Join builders
# ---------------------------------------------------------------------------

# Search-side metrics worth correlating against e2e signals.
SEARCH_METRICS = [
    "reward",
    "navigation_score",
    "navigation_recall",
    "navigation_iou",
    "file_f1",
    "function_f1",
    "file_recall",
    "function_recall",
    "assertion_coverage",
]

# E2E-side outcomes.
E2E_METRICS_PRIMARY = [("resolved", "binary"), ("f2p_pass_rate", "continuous")]

# Efficiency columns on both sides (for "is search-good-correlated-with-cheap?" sub-analysis).
EFFICIENCY_COLS = ["agent_steps", "tool_call_count", "input_tokens", "output_tokens", "wall_clock_sec"]


def build_framing_a(search: pd.DataFrame, e2e: pd.DataFrame, search_agg: str) -> pd.DataFrame:
    """Per-(model, parent). Aggregate search variants per parent (mean or max),
    then average across iters; same for e2e (mean across iters per (model, task))."""
    s_agg = (
        search.groupby(["canonical_model", "parent_task_id", "task"])[SEARCH_METRICS + EFFICIENCY_COLS]
        .mean()
        .reset_index()
    )
    if search_agg == "mean":
        s_parent = s_agg.groupby(["canonical_model", "parent_task_id"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].mean()
    elif search_agg == "max":
        s_parent = s_agg.groupby(["canonical_model", "parent_task_id"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].max()
    else:
        raise ValueError(search_agg)
    s_parent = s_parent.reset_index().rename(
        columns={c: f"search_{c}" for c in SEARCH_METRICS + EFFICIENCY_COLS}
    )
    e_parent = (
        e2e.groupby(["canonical_model", "task"])[["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS]
        .mean()
        .reset_index()
        .rename(columns={c: f"e2e_{c}" for c in ["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS})
    )
    return s_parent.merge(
        e_parent, left_on=["canonical_model", "parent_task_id"], right_on=["canonical_model", "task"]
    ).drop(columns=["task"])


def build_framing_b(search: pd.DataFrame, e2e: pd.DataFrame, search_agg: str) -> pd.DataFrame:
    """Per-iter-pair. Pair search iter i with e2e iter i (mod min available)."""
    s_agg = (
        search.groupby(["canonical_model", "parent_task_id", "task", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ]
        .mean()
        .reset_index()
    )
    if search_agg == "mean":
        s_parent_iter = s_agg.groupby(["canonical_model", "parent_task_id", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].mean()
    else:
        s_parent_iter = s_agg.groupby(["canonical_model", "parent_task_id", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].max()
    s_parent_iter = s_parent_iter.reset_index().rename(
        columns={c: f"search_{c}" for c in SEARCH_METRICS + EFFICIENCY_COLS}
    )

    # For pairing, take min(n_iters_search, n_iters_e2e) per model. Map each
    # search iter i to e2e iter i. Drop e2e iters past that.
    rows = []
    for model in s_parent_iter["canonical_model"].unique():
        s_iters = sorted(s_parent_iter[s_parent_iter.canonical_model == model]["search_iter"].unique())
        e_iters = sorted(e2e[e2e.canonical_model == model]["e2e_iter"].unique())
        n = min(len(s_iters), len(e_iters))
        s_take = s_iters[:n]
        e_take = e_iters[:n]
        e_sub = e2e[e2e.canonical_model == model].copy()
        for s_i, e_i in zip(s_take, e_take):
            s_slice = s_parent_iter[
                (s_parent_iter.canonical_model == model) & (s_parent_iter.search_iter == s_i)
            ]
            e_slice = e_sub[e_sub.e2e_iter == e_i].rename(
                columns={c: f"e2e_{c}" for c in ["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS}
            )[
                ["canonical_model", "task", "e2e_resolved", "e2e_f2p_pass_rate"]
                + [f"e2e_{c}" for c in EFFICIENCY_COLS]
            ]
            paired = s_slice.merge(
                e_slice, left_on=["canonical_model", "parent_task_id"], right_on=["canonical_model", "task"]
            ).drop(columns=["task"])
            paired["pair_iter"] = s_i
            rows.append(paired)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_framing_c(search: pd.DataFrame, e2e: pd.DataFrame, search_agg: str) -> pd.DataFrame:
    """Per-parent, model-collapsed. "Is this task intrinsically hard at both?" """
    s_agg = (
        search.groupby(["canonical_model", "parent_task_id", "task"])[SEARCH_METRICS + EFFICIENCY_COLS]
        .mean()
        .reset_index()
    )
    if search_agg == "mean":
        s_parent = s_agg.groupby(["canonical_model", "parent_task_id"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].mean()
    else:
        s_parent = s_agg.groupby(["canonical_model", "parent_task_id"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].max()
    s_collapsed = s_parent.groupby("parent_task_id").mean().reset_index()
    s_collapsed = s_collapsed.rename(columns={c: f"search_{c}" for c in SEARCH_METRICS + EFFICIENCY_COLS})

    e_parent = (
        e2e.groupby(["canonical_model", "task"])[["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS]
        .mean()
        .reset_index()
    )
    e_collapsed = e_parent.groupby("task").mean(numeric_only=True).reset_index()
    e_collapsed = e_collapsed.rename(
        columns={c: f"e2e_{c}" for c in ["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS}
    )
    return s_collapsed.merge(e_collapsed, left_on="parent_task_id", right_on="task").drop(columns=["task"])


def build_framing_d(search: pd.DataFrame, e2e: pd.DataFrame, search_agg: str) -> pd.DataFrame:
    """Per-trial-pair: every search trial × every same-(model, task) e2e trial.

    Dirty (non-independent) but maximizes data points. We keep only one search
    row per (model, parent, search_iter) — agg across variants per the agg rule.
    """
    s_agg = (
        search.groupby(["canonical_model", "parent_task_id", "task", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ]
        .mean()
        .reset_index()
    )
    if search_agg == "mean":
        s_parent_iter = s_agg.groupby(["canonical_model", "parent_task_id", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].mean()
    else:
        s_parent_iter = s_agg.groupby(["canonical_model", "parent_task_id", "search_iter"])[
            SEARCH_METRICS + EFFICIENCY_COLS
        ].max()
    s_parent_iter = s_parent_iter.reset_index().rename(
        columns={c: f"search_{c}" for c in SEARCH_METRICS + EFFICIENCY_COLS}
    )
    e_renamed = e2e.rename(columns={c: f"e2e_{c}" for c in ["resolved", "f2p_pass_rate"] + EFFICIENCY_COLS})
    return s_parent_iter.merge(
        e_renamed, left_on=["canonical_model", "parent_task_id"], right_on=["canonical_model", "task"]
    ).drop(columns=["task"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_correlations(name: str, df: pd.DataFrame, by_model: bool = True) -> list[dict]:
    """Print a markdown table of all (search_metric, e2e_metric) correlations.
    Returns the raw rows for downstream CSV writing."""
    print(f"\n### {name}\n")
    if df.empty:
        print("(no data)")
        return []
    print(f"_n_pairs (overall): {len(df)}_")
    print()
    print("| Search metric | E2E metric | Pearson r | Spearman ρ |")
    print("|---|---|---:|---:|")
    rows = []
    for sm in SEARCH_METRICS:
        sx = f"search_{sm}"
        for em, _ in E2E_METRICS_PRIMARY:
            ey = f"e2e_{em}"
            if sx not in df.columns or ey not in df.columns:
                continue
            pr, pp, pn = _corr(df[sx], df[ey], "pearson")
            sr, sp, sn = _corr(df[sx], df[ey], "spearman")
            ci = _fisher_ci(pr, pn)
            print(f"| `{sm}` | `{em}` | {_fmt_corr(pr, pp, pn, ci)} | {_fmt_corr(sr, sp, sn)} |")
            rows.append(
                {
                    "framing": name,
                    "search_metric": sm,
                    "e2e_metric": em,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "pearson_ci_lo": ci[0],
                    "pearson_ci_hi": ci[1],
                    "spearman_r": sr,
                    "spearman_p": sp,
                    "n": pn,
                    "model": "(all)",
                }
            )
    if not by_model or "canonical_model" not in df.columns:
        return rows

    print()
    print("**Per-model breakdown** (Pearson / Spearman against `resolved`):")
    print()
    print("| Model | n | Pearson r | Spearman ρ |")
    print("|---|---:|---:|---:|")
    for model in sorted(df["canonical_model"].unique()):
        sub = df[df.canonical_model == model]
        if "search_navigation_score" not in sub.columns or "e2e_resolved" not in sub.columns:
            continue
        pr, pp, pn = _corr(sub["search_navigation_score"], sub["e2e_resolved"], "pearson")
        sr, sp, _ = _corr(sub["search_navigation_score"], sub["e2e_resolved"], "spearman")
        print(f"| `{model}` | {pn} | {_fmt_corr(pr, pp, pn)} | {_fmt_corr(sr, sp, pn)} |")
        rows.append(
            {
                "framing": f"{name} / per-model",
                "search_metric": "navigation_score",
                "e2e_metric": "resolved",
                "pearson_r": pr,
                "pearson_p": pp,
                "pearson_ci_lo": None,
                "pearson_ci_hi": None,
                "spearman_r": sr,
                "spearman_p": sp,
                "n": pn,
                "model": model,
            }
        )
    return rows


def report_model_rankings(framing_a: pd.DataFrame) -> None:
    """Compare search ranking vs e2e ranking across the 4 models. Spearman of model means."""
    print("\n## Model ranking stability\n")
    print("Model means across all parents (framing A, search-agg=mean):\n")
    if framing_a.empty:
        print("(no data)")
        return
    means = framing_a.groupby("canonical_model")[
        ["search_navigation_score", "search_reward", "e2e_resolved", "e2e_f2p_pass_rate"]
    ].mean()
    # Hand-rolled markdown to avoid the `tabulate` optional dep that
    # pandas.DataFrame.to_markdown() pulls in.
    cols = list(means.columns)
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for model, row in means.round(3).iterrows():
        print(f"| `{model}` | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |")

    print()
    rs, ps, ns = _corr(means["search_navigation_score"], means["e2e_resolved"], "spearman")
    print(
        f"_Spearman ρ between search nav-score model-rank and e2e resolved model-rank:_ "
        f"{_fmt_corr(rs, ps, ns)}"
    )


def report_efficiency_correlations(search: pd.DataFrame, e2e: pd.DataFrame) -> None:
    """Within-side: do agents that use more turns/tokens get better rewards?"""
    print("\n## Efficiency vs reward (within each side)\n")

    print("**Search side** (per-trial, all models pooled):")
    print()
    print("| Efficiency col | Pearson vs reward | Spearman vs reward |")
    print("|---|---:|---:|")
    for col in EFFICIENCY_COLS:
        if col not in search.columns:
            continue
        pr, pp, n = _corr(search[col], search["reward"], "pearson")
        sr, sp, _ = _corr(search[col], search["reward"], "spearman")
        print(f"| `{col}` | {_fmt_corr(pr, pp, n)} | {_fmt_corr(sr, sp, n)} |")

    print()
    print("**E2E side** (per-trial, all models pooled):")
    print()
    print("| Efficiency col | Pearson vs resolved | Spearman vs f2p_pass_rate |")
    print("|---|---:|---:|")
    for col in EFFICIENCY_COLS:
        if col not in e2e.columns:
            continue
        pr, pp, n = _corr(e2e[col], e2e["resolved"], "pearson")
        sr, sp, _ = _corr(e2e[col], e2e["f2p_pass_rate"], "spearman")
        print(f"| `{col}` | {_fmt_corr(pr, pp, n)} | {_fmt_corr(sr, sp, n)} |")


def report_recall_vs_precision_profile(framing_a: pd.DataFrame) -> None:
    """Are 'broad-trace' (high-recall, low-precision) agents better fixers than
    'narrow-root-cause' (high-precision, low-recall) agents?"""
    print("\n## Recall- vs precision-style and e2e outcome\n")
    if framing_a.empty:
        return
    df = framing_a.copy()
    if "search_function_recall" not in df.columns or "search_function_precision" not in df.columns:
        print("(precision/recall columns missing — older trial corpus)")
        return
    df["recall_minus_precision"] = df["search_function_recall"] - df["search_function_precision"]
    pr, pp, n = _corr(df["recall_minus_precision"], df["e2e_resolved"], "pearson")
    sr, sp, _ = _corr(df["recall_minus_precision"], df["e2e_f2p_pass_rate"], "spearman")
    print(
        f"_recall − precision_ vs _resolved_: {_fmt_corr(pr, pp, n)}  \n"
        f"_recall − precision_ vs _f2p pass-rate_: {_fmt_corr(sr, sp, n)}"
    )
    print()
    print(
        "**Interpretation cheatsheet:** A *positive* correlation means broad-trace "
        "(high-recall, low-precision) agents have higher e2e success. *Negative* "
        "means narrow-root-cause agents fix more reliably."
    )


def report_variance_correlation(search: pd.DataFrame, e2e: pd.DataFrame) -> None:
    """Per-task within-model stdev: hard tasks should be noisy on both sides."""
    print("\n## Per-task variance correlation (search vs e2e within model)\n")

    s_var = (
        search.groupby(["canonical_model", "parent_task_id"])["reward"]
        .agg(lambda x: x.std(ddof=1) if len(x) > 1 else 0.0)
        .reset_index()
        .rename(columns={"reward": "search_reward_std"})
    )
    s_var = s_var.groupby(["canonical_model", "parent_task_id"]).mean().reset_index()

    e_var = (
        e2e.groupby(["canonical_model", "task"])["resolved"]
        .agg(lambda x: x.std(ddof=1) if len(x) > 1 else 0.0)
        .reset_index()
        .rename(columns={"resolved": "e2e_resolved_std", "task": "parent_task_id"})
    )
    merged = s_var.merge(e_var, on=["canonical_model", "parent_task_id"])
    if merged.empty:
        print("(no overlap)")
        return
    pr, pp, n = _corr(merged["search_reward_std"], merged["e2e_resolved_std"], "pearson")
    sr, sp, _ = _corr(merged["search_reward_std"], merged["e2e_resolved_std"], "spearman")
    print(
        f"Pearson(σ_search, σ_e2e) across (model, task): {_fmt_corr(pr, pp, n)}  \n"
        f"Spearman: {_fmt_corr(sr, sp, n)}"
    )
    print()
    print(
        "Positive means tasks that are noisy at search are also noisy at e2e — supports "
        "a 'task difficulty' interpretation. Null means the two noise sources are "
        "different (e.g. e2e noise dominated by patch correctness rather than localization)."
    )


# ---------------------------------------------------------------------------
# Anecdote mining
# ---------------------------------------------------------------------------


def report_anecdotes(framing_a: pd.DataFrame, e2e: pd.DataFrame, out_dir: Path | None) -> None:
    """Surface revealing per-task patterns for the paper."""
    print("\n## Anecdotes\n")
    if framing_a.empty:
        return
    df = framing_a.copy()

    # 1. Strong-search, weak-e2e — "found the area, missed the fix"
    print("### Search-strong-but-e2e-weak (per model)\n")
    print(
        "Tasks where a model nailed search (nav≥0.7) but failed e2e (resolved≤0.25). "
        "These point at an 'I knew where, I just couldn't fix it' pattern.\n"
    )
    rows_sw = df[(df["search_navigation_score"] >= 0.7) & (df["e2e_resolved"] <= 0.25)].sort_values(
        ["canonical_model", "search_navigation_score"], ascending=[True, False]
    )
    if rows_sw.empty:
        print("_(none)_")
    else:
        for model, sub in rows_sw.groupby("canonical_model"):
            print(f"**`{model}`** ({len(sub)} task{'s' if len(sub) != 1 else ''}):")
            for _, r in sub.head(5).iterrows():
                print(
                    f"- `{r['parent_task_id']}` — "
                    f"nav={r['search_navigation_score']:.2f}, resolved={r['e2e_resolved']:.2f}, "
                    f"f2p={r['e2e_f2p_pass_rate']:.2f}"
                )
            if len(sub) > 5:
                print(f"- _… and {len(sub) - 5} more_")
            print()

    # 2. Weak-search, strong-e2e — "fixed without localizing"
    print("### Search-weak-but-e2e-strong (per model)\n")
    print(
        "Tasks where the model failed search (nav≤0.3) but solved e2e (resolved≥0.75). "
        "These suggest the patch can be discovered without explicit localization "
        "(maybe via grep, traceback, or test-driven exploration).\n"
    )
    rows_ws = df[(df["search_navigation_score"] <= 0.3) & (df["e2e_resolved"] >= 0.75)].sort_values(
        ["canonical_model", "search_navigation_score"], ascending=[True, True]
    )
    if rows_ws.empty:
        print("_(none)_")
    else:
        for model, sub in rows_ws.groupby("canonical_model"):
            print(f"**`{model}`** ({len(sub)} task{'s' if len(sub) != 1 else ''}):")
            for _, r in sub.head(5).iterrows():
                print(
                    f"- `{r['parent_task_id']}` — "
                    f"nav={r['search_navigation_score']:.2f}, resolved={r['e2e_resolved']:.2f}"
                )
            if len(sub) > 5:
                print(f"- _… and {len(sub) - 5} more_")
            print()

    # 3. Model-disagreement tasks
    print("### Model-disagreement tasks\n")
    print("Top-10 tasks where models split most sharply on resolved (these discriminate capability):\n")
    pivot = df.pivot_table(index="parent_task_id", columns="canonical_model", values="e2e_resolved")
    pivot["std"] = pivot.std(axis=1)
    pivot["mean"] = pivot.drop(columns=["std"]).mean(axis=1)
    pivot = pivot.sort_values("std", ascending=False)
    print("| Parent task | mean | std | " + " | ".join(f"`{m}`" for m in CANONICAL_MODELS) + " |")
    print("|---|---:|---:|" + "---:|" * len(CANONICAL_MODELS))
    for parent, row in pivot.head(10).iterrows():
        cells = [f"{row[m]:.2f}" if pd.notna(row.get(m)) else "—" for m in CANONICAL_MODELS]
        print(f"| `{parent}` | {row['mean']:.2f} | {row['std']:.2f} | " + " | ".join(cells) + " |")

    # 4. Floor (uniform fail) and ceiling (uniform pass)
    print("\n### Floor / ceiling tasks\n")
    floor = pivot[(pivot.drop(columns=["std", "mean"]) <= 0.1).all(axis=1)]
    ceiling = pivot[(pivot.drop(columns=["std", "mean"]) >= 0.9).all(axis=1)]
    print(f"**Floor** (every model resolved ≤ 0.1): {len(floor)} tasks")
    for parent in floor.index[:10]:
        print(f"  - `{parent}`")
    if len(floor) > 10:
        print(f"  - _… and {len(floor) - 10} more_")
    print()
    print(f"**Ceiling** (every model resolved ≥ 0.9): {len(ceiling)} tasks")
    for parent in ceiling.index[:10]:
        print(f"  - `{parent}`")
    if len(ceiling) > 10:
        print(f"  - _… and {len(ceiling) - 10} more_")

    # 5. High-variance within model (rerun-noisy tasks)
    print("\n### High-variance within model (e2e rerun-noisy)\n")
    print("Top 10 (model, task) pairs with largest `resolved` stdev across e2e iterations:\n")
    iv = e2e.groupby(["canonical_model", "task"])["resolved"].agg(["mean", "std", "count"]).reset_index()
    iv = iv[iv["count"] >= 2].sort_values("std", ascending=False)
    print("| Model | Parent task | n_iters | mean | std |")
    print("|---|---|---:|---:|---:|")
    for _, r in iv.head(10).iterrows():
        cells = (
            f"`{r['canonical_model']}` | `{r['task']}` | {int(r['count'])} | {r['mean']:.2f} | {r['std']:.2f}"
        )
        print(f"| {cells} |")

    if out_dir:
        # Save the per-task pivot for plotting in a notebook
        pivot.reset_index().to_csv(out_dir / "anecdotes-pivot.csv", index=False)
        print(f"\n_(per-task pivot written to {out_dir / 'anecdotes-pivot.csv'})_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--search-csv", type=Path, required=True)
    ap.add_argument("--e2e-csv", type=Path, required=True)
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="Write paired-observations CSVs and per-cut data here."
    )
    args = ap.parse_args()

    out_dir: Path | None = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    search = load_search(args.search_csv)
    e2e = load_e2e(args.e2e_csv)

    print("# Search vs E2E correlation analysis")
    print()
    print(f"- Search CSV: `{args.search_csv}` ({len(search)} usable trial rows)")
    print(f"- E2E CSV:    `{args.e2e_csv}` ({len(e2e)} usable trial rows)")
    print(f"- Models (canonical): {', '.join(sorted(search['canonical_model'].unique()))}")
    print(
        f"- Parents in search: {search['parent_task_id'].nunique()}; "
        f"e2e tasks: {e2e['task'].nunique()}; "
        f"overlap: {len(set(search['parent_task_id']) & set(e2e['task']))}"
    )
    print()
    print(
        "**Reading the tables:** `r` is shown ± 95% CI (Fisher-z) where defined; "
        "stars: *<0.05, **<0.01, ***<0.001. n = number of paired observations."
    )

    all_rows: list[dict] = []

    print("\n## Framing A — per-(model, parent), aggregated across iters and variants\n")
    for agg in ("mean", "max"):
        df = build_framing_a(search, e2e, search_agg=agg)
        rows = report_correlations(f"Framing A (search-agg={agg})", df, by_model=True)
        all_rows.extend(rows)
        if out_dir:
            df.to_csv(out_dir / f"framing-a-{agg}.csv", index=False)

    print("\n## Framing B — per-iter-pair (search iter i ↔ e2e iter i)\n")
    for agg in ("mean", "max"):
        df = build_framing_b(search, e2e, search_agg=agg)
        rows = report_correlations(f"Framing B (search-agg={agg})", df, by_model=True)
        all_rows.extend(rows)
        if out_dir:
            df.to_csv(out_dir / f"framing-b-{agg}.csv", index=False)

    print("\n## Framing C — per-parent (model-collapsed)\n")
    for agg in ("mean", "max"):
        df = build_framing_c(search, e2e, search_agg=agg)
        rows = report_correlations(f"Framing C (search-agg={agg})", df, by_model=False)
        all_rows.extend(rows)
        if out_dir:
            df.to_csv(out_dir / f"framing-c-{agg}.csv", index=False)

    print("\n## Framing D — per-trial-pair (search × e2e cross-product)\n")
    for agg in ("mean", "max"):
        df = build_framing_d(search, e2e, search_agg=agg)
        rows = report_correlations(f"Framing D (search-agg={agg})", df, by_model=True)
        all_rows.extend(rows)

    # Side analyses
    framing_a_mean = build_framing_a(search, e2e, search_agg="mean")
    report_model_rankings(framing_a_mean)
    report_efficiency_correlations(search, e2e)
    report_recall_vs_precision_profile(framing_a_mean)
    report_variance_correlation(search, e2e)
    report_anecdotes(framing_a_mean, e2e, out_dir)

    if out_dir:
        pd.DataFrame(all_rows).to_csv(out_dir / "all-correlations.csv", index=False)
        print(f"\n_(All correlations written to {out_dir / 'all-correlations.csv'})_")

    return 0


if __name__ == "__main__":
    sys.exit(main())
