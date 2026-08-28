# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""regress_e2e_resolution.py — multivariate logistic regression on e2e success.

Models:
  e2e_resolved ~ behavioral predictors + trial-length controls + task-complexity
                 controls + per-model fixed effects

Predictors are normalized to be comparable across models (rates, fractions,
recalls). Model fixed effects absorb the codex-vs-others tool-call-volume
difference (codex's exec_command runs rg/grep, inflating examined-file
counts; claude-code's Read is one file each).

Reports for each predictor:
  - β coefficient (log-odds change per unit; SD-units for continuous)
  - SE (Hessian-based)
  - 95% CI
  - z-statistic
  - approximate two-sided p-value (normal approximation; n=364 makes this
    fine)

Three model variants:
  1. POOLED — single regression with model fixed effects
  2. PER-MODEL — one regression per model (no FE), to diagnose heterogeneity
  3. FIRST-HALF — same as pooled but using only first-half-of-trial signals,
     to control for "success ends the trial early" reverse-causality

Logistic regression is fit by IRLS (Newton-Raphson on the log-likelihood)
with ridge regularization (λ=0.01) for numerical stability. No statsmodels
dep — implementation is ~30 lines and is correct for our scale (n=364,
p≤15).

Usage:
  uv run python scripts/regress_e2e_resolution.py \\
      --per-trial-csv docs/data/v2b-deep-dive-per-trial.csv \\
      --first-half-csv docs/data/v2b-first-half-per-trial.csv \\
      --output-md docs/data/v2b-regression-results.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logistic regression via IRLS with L2 regularization
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


@dataclass
class LogitResult:
    feature_names: list[str]
    coef: np.ndarray  # shape (p+1,) — intercept first
    se: np.ndarray  # standard errors
    z: np.ndarray
    p: np.ndarray
    ci_lo: np.ndarray  # 95% CI lower for log-odds
    ci_hi: np.ndarray
    n: int
    n_resolved: int
    log_likelihood: float
    null_log_likelihood: float  # intercept-only
    pseudo_r2: float  # McFadden's
    accuracy: float  # in-sample classification accuracy at 0.5 threshold


def fit_logistic(
    X: np.ndarray,  # shape (n, p) — without intercept
    y: np.ndarray,  # shape (n,) — 0/1
    feature_names: list[str],
    ridge: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogitResult:
    """L2-regularized logistic regression by IRLS.

    Loss: sum_i [-y_i log p_i - (1-y_i) log(1-p_i)] + 0.5 * ridge * ||β[1:]||²
    (intercept is not regularized).
    """
    n, p = X.shape
    # Add intercept column
    X1 = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)

    # Build regularization mask (no penalty on intercept)
    reg = np.full(p + 1, ridge)
    reg[0] = 0.0

    prev_ll = -np.inf
    for it in range(max_iter):
        z = X1 @ beta
        pi = _sigmoid(z)
        # Gradient: X^T (y - pi) - reg * beta
        grad = X1.T @ (y - pi) - reg * beta
        # Hessian: -X^T diag(pi*(1-pi)) X - diag(reg)
        W = pi * (1 - pi)
        # Avoid singularity
        W = np.clip(W, 1e-9, None)
        H = -(X1.T * W) @ X1 - np.diag(reg)
        # Newton step: beta -= H^{-1} grad
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        # Log-likelihood for convergence check
        z_new = X1 @ beta_new
        log_p = -np.logaddexp(0.0, -z_new)
        log_1_p = -np.logaddexp(0.0, z_new)
        ll = float(np.sum(y * log_p + (1 - y) * log_1_p)) - 0.5 * float(reg @ (beta_new**2))
        if abs(ll - prev_ll) < tol:
            beta = beta_new
            break
        prev_ll = ll
        beta = beta_new

    # Standard errors from the (negative) Hessian inverse
    z = X1 @ beta
    pi = _sigmoid(z)
    W = pi * (1 - pi)
    W = np.clip(W, 1e-9, None)
    info = (X1.T * W) @ X1 + np.diag(reg)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    z_stat = np.where(se > 0, beta / se, 0.0)
    # two-sided p via normal CDF
    from math import erf as _erf
    from math import sqrt as _sqrt

    p_vals = np.array([2 * (1 - 0.5 * (1 + _erf(abs(zz) / _sqrt(2)))) for zz in z_stat])
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se

    # Final log-likelihood (without ridge term — for pseudo R²)
    log_p = -np.logaddexp(0.0, -z)
    log_1_p = -np.logaddexp(0.0, z)
    ll_full = float(np.sum(y * log_p + (1 - y) * log_1_p))
    # Null model: intercept only
    p_null = float(np.mean(y))
    ll_null = (
        float(np.sum(y) * math.log(p_null) + np.sum(1 - y) * math.log(1 - p_null)) if 0 < p_null < 1 else 0.0
    )
    pseudo_r2 = 1 - ll_full / ll_null if ll_null < 0 else 0.0
    pred = (pi >= 0.5).astype(int)
    accuracy = float(np.mean(pred == y))

    return LogitResult(
        feature_names=["(intercept)"] + feature_names,
        coef=beta,
        se=se,
        z=z_stat,
        p=p_vals,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n=n,
        n_resolved=int(np.sum(y)),
        log_likelihood=ll_full,
        null_log_likelihood=ll_null,
        pseudo_r2=pseudo_r2,
        accuracy=accuracy,
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "codex": "codex55",
    "claude-code": "opus47",
    "haiku": "haiku45",
    "qwen": "qwen36",
}


def _short_model(model_str: str) -> str:
    s = model_str.lower()
    if "gpt-5.5" in s:
        return "codex55"
    if "opus-4-7" in s:
        return "opus47"
    if "haiku-4-5" in s:
        return "haiku45"
    if "qwen" in s or "vllm/model" in s:
        return "qwen36"
    return "unknown"


def _build_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the design matrix.

    Predictors (continuous, standardized to mean=0 sd=1 across the pooled
    population so coefficients are interpretable in SD-units):
        tests_per_edit
        tests_per_step
        examined_file_recall
        comm_file_recall
        n_examined_files (log-transformed)
        frac_test_files
        agent_diff_size_ratio (clipped at 10x)
        probe_fraction
        thrash_rate
        pre_edit_mention_rate
        n_steps (log-transformed)
        gold_diff_lines (log-transformed)

    Plus 3 model dummy variables (codex55 is reference; opus/haiku/qwen
    each get a dummy).
    """
    feats: list[list[float]] = []
    y_list: list[int] = []
    skipped = 0
    for r in rows:
        model = _short_model(r["model"])
        if model == "unknown":
            skipped += 1
            continue
        try:
            n_steps = max(1, int(float(r["n_steps"])))
            n_edits = max(0, int(float(r["n_edit_calls"])))
            n_tests = max(0, int(float(r["n_test_invocations"])))
            n_exam = max(0, int(float(r["n_examined_files"])))
            n_test_files = max(0, int(float(r["n_examined_test_files"])))
            files_e1 = max(0, int(float(r["files_edited_once"])))
            files_e2 = max(0, int(float(r["files_edited_2plus"])))
            agent_diff = max(0, int(float(r["agent_diff_lines"])))
            gold_diff = max(0, int(float(r["gold_diff_lines"])))
            first_edit = max(0, int(float(r["first_edit_step"])))
            pre_mention = max(0, int(float(r["pre_edit_mention_count"])))
            gold_n_funcs = max(1, int(float(r["gold_n_functions"])))
            exam_recall = float(r["exam_file_recall"])
            comm_recall = float(r["comm_file_recall"])
        except (KeyError, ValueError):
            skipped += 1
            continue

        # Skip degenerate trials (no edits AND no test invocations)
        if n_edits == 0 and n_tests == 0:
            skipped += 1
            continue

        tests_per_edit = n_tests / max(1, n_edits)
        tests_per_step = n_tests / n_steps
        frac_test = n_test_files / max(1, n_exam)
        diff_ratio = min(agent_diff / max(1, gold_diff), 10.0)
        probe_frac = first_edit / n_steps if first_edit else 1.0  # if no edit ever, "all probe"
        thrash = files_e2 / max(1, files_e1 + files_e2)
        mention_rate = min(pre_mention / gold_n_funcs, 1.0)
        log_exam = math.log1p(n_exam)
        log_steps = math.log1p(n_steps)
        log_gold_diff = math.log1p(gold_diff)

        feats.append(
            [
                tests_per_edit,
                tests_per_step,
                exam_recall,
                comm_recall,
                log_exam,
                frac_test,
                diff_ratio,
                probe_frac,
                thrash,
                mention_rate,
                log_steps,
                log_gold_diff,
                # Model dummies (codex55 reference)
                1.0 if model == "opus47" else 0.0,
                1.0 if model == "haiku45" else 0.0,
                1.0 if model == "qwen36" else 0.0,
            ]
        )
        y_list.append(1 if r["e2e_resolved"] == "1" else 0)

    feature_names = [
        "tests_per_edit",
        "tests_per_step",
        "exam_file_recall",
        "comm_file_recall",
        "log_n_examined_files",
        "frac_test_files",
        "agent_diff_ratio",
        "probe_fraction",
        "thrash_rate",
        "pre_edit_mention_rate",
        "log_n_steps",
        "log_gold_diff_lines",
        "is_opus47",
        "is_haiku45",
        "is_qwen36",
    ]

    X = np.array(feats, dtype=float)
    y = np.array(y_list, dtype=float)
    if skipped:
        print(f"[features] skipped {skipped} rows (degenerate or unknown model)", file=sys.stderr)

    # Standardize the continuous features (first 12) for interpretability
    cont_idx = list(range(12))
    standardized_names = list(feature_names)
    means = np.zeros(X.shape[1])
    sds = np.ones(X.shape[1])
    for i in cont_idx:
        mu = X[:, i].mean()
        sd = X[:, i].std()
        means[i] = mu
        sds[i] = sd if sd > 0 else 1.0
        X[:, i] = (X[:, i] - mu) / sds[i]
        standardized_names[i] = feature_names[i] + " (sd)"

    return X, y, feature_names, standardized_names


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_result(result: LogitResult, title: str) -> list[str]:
    out: list[str] = []
    out.append(f"### {title}")
    out.append("")
    out.append(
        f"_n = {result.n}, n_resolved = {result.n_resolved} "
        f"({result.n_resolved / result.n * 100:.1f}%); "
        f"in-sample accuracy at 0.5 = {result.accuracy:.3f}; "
        f"McFadden pseudo-R² = {result.pseudo_r2:.3f}_"
    )
    out.append("")
    out.append("| Feature | β | SE | z | p | 95% CI |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for i, name in enumerate(result.feature_names):
        b = result.coef[i]
        se = result.se[i]
        z = result.z[i]
        p = result.p[i]
        lo = result.ci_lo[i]
        hi = result.ci_hi[i]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        out.append(f"| `{name}` | {b:+.3f}{stars} | {se:.3f} | {z:+.2f} | {p:.4f} | [{lo:+.3f}, {hi:+.3f}] |")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--per-trial-csv",
        type=Path,
        required=True,
        help="docs/data/v2b-deep-dive-per-trial.csv",
    )
    ap.add_argument(
        "--first-half-csv",
        type=Path,
        default=None,
        help="Optional first-half-trials CSV (for reverse-causality check). If absent, skipped.",
    )
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.per_trial_csv.open()))
    print(f"Loaded {len(rows)} trial rows", file=sys.stderr)

    X, y, feature_names, std_names = _build_features(rows)
    print(f"Design matrix: n={X.shape[0]}, p={X.shape[1]}, n_resolved={int(y.sum())}", file=sys.stderr)

    # MODEL 1: pooled
    pooled = fit_logistic(X, y, std_names)

    # MODEL 2: per-model (drop the model dummies; refit on each subset)
    # We need un-standardized model assignment to subset
    by_model: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        m = _short_model(r["model"])
        if m != "unknown":
            by_model[m].append(i)
    # But X has been built including dummies and standardized using the pooled
    # population. Refit per-model on that same X (without dummies) using only
    # the subset of rows.
    per_model_results: dict[str, LogitResult] = {}
    n_dummies = 3  # last 3 columns
    # Re-derive subset mapping on the post-skip X. Since _build_features
    # skips some rows, the indices don't match. Easier: rebuild per-model.
    for m in ("codex55", "opus47", "haiku45", "qwen36"):
        sub_rows = [r for r in rows if _short_model(r["model"]) == m]
        if len(sub_rows) < 30:
            continue
        Xm, ym, _, std_m = _build_features(sub_rows)
        # Drop the 3 dummy columns (all 0 for this single-model fit)
        Xm_nd = Xm[:, :-n_dummies]
        cont_m = std_m[:-n_dummies]
        if int(ym.sum()) < 5 or int((1 - ym).sum()) < 5:
            print(f"[per-model {m}] insufficient class balance, skipping", file=sys.stderr)
            continue
        per_model_results[m] = fit_logistic(Xm_nd, ym, cont_m)

    # MODEL 3: first-half
    first_half_result: LogitResult | None = None
    if args.first_half_csv and args.first_half_csv.is_file():
        rows_h = list(csv.DictReader(args.first_half_csv.open()))
        print(f"Loaded {len(rows_h)} first-half trial rows", file=sys.stderr)
        Xh, yh, _, std_h = _build_features(rows_h)
        first_half_result = fit_logistic(Xh, yh, std_h)

    # Write markdown
    out: list[str] = []
    out.append("# E2E resolution: multivariate logistic regression")
    out.append("")
    out.append(
        "Logistic regression of `e2e_resolved ∈ {0,1}` on behavioral predictors. "
        "Continuous features standardized to SD=1 so coefficients are comparable. "
        "Model dummies in the pooled regression (`codex55` is the reference category). "
        "L2 regularization with λ=0.01."
    )
    out.append("")
    out.append("## How to read this")
    out.append("")
    out.append(
        "- `β` is the change in log-odds of resolution per 1 SD increase (for continuous) "
        "or per category change (for dummies). e.g. β=+0.3 means +1 SD raises log-odds by 0.3, "
        "or odds-ratio = exp(0.3) ≈ 1.35.\n"
        "- The pooled model is the headline. Per-model regressions diagnose heterogeneity.\n"
        "- The first-half regression controls for 'success ends the trial early': it uses only "
        "  signals from the first half of each trial.\n"
        "- Stars: *** p<0.001, ** p<0.01, * p<0.05.\n"
        "- McFadden pseudo-R² is interpretable as 'fraction of null deviance explained'. Values "
        "  above ~0.2 indicate strong fit for binary outcomes."
    )
    out.append("")

    out.append("## Pooled regression with model fixed effects")
    out.append("")
    out.extend(_format_result(pooled, "Pooled (all 4 models, n=" + str(pooled.n) + ")"))
    out.append("")

    out.append("## Per-model regressions")
    out.append("")
    for m in ("codex55", "opus47", "haiku45", "qwen36"):
        if m in per_model_results:
            out.extend(_format_result(per_model_results[m], m))
            out.append("")

    if first_half_result is not None:
        out.append("## First-half-only regression (reverse-causality control)")
        out.append("")
        out.append(
            "Same predictors, but using only signals from steps before the median step of each trial. "
            "Controls for 'successful trials end early' confounding."
        )
        out.append("")
        out.extend(_format_result(first_half_result, "First-half only (n=" + str(first_half_result.n) + ")"))
        out.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(out) + "\n")
    print(f"Wrote regression results → {args.output_md}", file=sys.stderr)

    # Also dump as JSON for downstream tooling
    json_out = args.output_md.with_suffix(".json")
    json_data = {
        "pooled": {
            "feature_names": pooled.feature_names,
            "coef": pooled.coef.tolist(),
            "se": pooled.se.tolist(),
            "z": pooled.z.tolist(),
            "p": pooled.p.tolist(),
            "n": pooled.n,
            "n_resolved": pooled.n_resolved,
            "pseudo_r2": pooled.pseudo_r2,
            "accuracy": pooled.accuracy,
        },
        "per_model": {
            m: {
                "feature_names": r.feature_names,
                "coef": r.coef.tolist(),
                "p": r.p.tolist(),
                "n": r.n,
                "n_resolved": r.n_resolved,
                "pseudo_r2": r.pseudo_r2,
            }
            for m, r in per_model_results.items()
        },
    }
    if first_half_result is not None:
        json_data["first_half"] = {
            "feature_names": first_half_result.feature_names,
            "coef": first_half_result.coef.tolist(),
            "p": first_half_result.p.tolist(),
            "n": first_half_result.n,
            "pseudo_r2": first_half_result.pseudo_r2,
        }
    json_out.write_text(json.dumps(json_data, indent=2))
    print(f"Wrote regression JSON → {json_out}", file=sys.stderr)

    # Stdout summary
    print()
    print("=== Top-5 features by |β| in pooled regression ===")
    pairs = sorted(zip(pooled.feature_names, pooled.coef, pooled.p), key=lambda x: abs(x[1]), reverse=True)
    for name, b, p in pairs[:8]:
        if name == "(intercept)":
            continue
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<28} β = {b:+.3f}{stars}  (p = {p:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
