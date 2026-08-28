# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""rescore_search_against_patch_gold.py — re-score search trials against patch-derived gold.

Each search trial's `verifier/reward.json` already contains the agent's
submission (`agent_files`, `agent_functions`, `agent_explanation`). This
script reads those alongside the patch-derived gold (from
`extract_v2b_patch_gold.py`) and recomputes file/function recall, precision,
F1, and IoU against the patch-gold sets — then emits a per-trial CSV in the
same shape as `summarize_search_baseline.py --csv`, with the new scores
substituted.

The point: produce an apples-to-apples gold set so we can correlate the
search agent's localization (against the patch-gold) with the e2e agent's
implicit-search localization (also against the patch-gold).

The scoring math mirrors `src/craft_taskgen/search/templates/test_runner.py`
(strict recall + lenient precision F1, exact-or-tail-match one-to-one).

Usage:
  uv run python scripts/rescore_search_against_patch_gold.py \\
      --search-roots /tmp/search-rescored/20260501-from-v2-rescored/iter*-* \\
      --patch-gold references/v2b-patch-gold.json \\
      --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-search-from-v2/harbor-tasks \\
      --output /tmp/search-baseline-patchgold.csv

`--tasks-dir` is the search-side dataset (used to look up parent_task_id per
search task via provenance.json). `--patch-gold` is keyed by parent_task_id,
so search tasks whose parent isn't in the patch-gold map (e.g. parents
without a v2b solution patch) are emitted with blank scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Scoring primitives (copy of test_runner.py logic — kept inline so this
# script has no project-internal imports and stays runnable from anywhere)
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = ("/repo/", "repo/", "/code/", "code/", "./")


def _normalize_file(path: str) -> str:
    p = path.strip().lower()
    for prefix in _STRIP_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.rstrip("/")


def _normalize_function_tail(name: str) -> str:
    parts = name.strip().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name.strip()


def _exact_or_tail_match_count(predicted: set[str], gold: set[str]) -> int:
    matched_exact = predicted & gold
    rem_pred = predicted - matched_exact
    rem_gold = gold - matched_exact
    pred_tails = Counter(_normalize_function_tail(p) for p in rem_pred)
    gold_tails = Counter(_normalize_function_tail(g) for g in rem_gold)
    return len(matched_exact) + sum((pred_tails & gold_tails).values())


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _score_against_gold(
    agent_files: list[str], agent_functions: list[str], gold_files: list[str], gold_functions: list[str]
) -> dict:
    """Compute the same scoring shape as the search verifier writes to reward.json,
    against the supplied gold (no alts in patch-gold mode)."""
    af = {_normalize_file(f) for f in agent_files if f.strip()}
    gf = {_normalize_file(f) for f in gold_files if f.strip()}
    afn = {f.strip() for f in agent_functions if f.strip()}
    gfn = {f.strip() for f in gold_functions if f.strip()}

    file_hits_strict = len(af & gf)
    # No alts in patch-gold by construction; lenient == strict here.
    file_recall = file_hits_strict / len(gf) if gf else 1.0
    file_precision = file_hits_strict / len(af) if af else 0.0
    file_f1 = _f1(file_precision, file_recall)
    file_iou_denom = len(af | gf)
    file_iou = file_hits_strict / file_iou_denom if file_iou_denom > 0 else 1.0

    func_recall_hits = _exact_or_tail_match_count(afn, gfn)
    func_recall = func_recall_hits / len(gfn) if gfn else 1.0
    func_precision_hits = _exact_or_tail_match_count(gfn, afn)
    func_precision = func_precision_hits / len(afn) if afn else 0.0
    func_f1 = _f1(func_precision, func_recall)
    afn_tails = {_normalize_function_tail(f) for f in afn}
    gfn_tails = {_normalize_function_tail(f) for f in gfn}
    func_iou_num = len(afn_tails & gfn_tails)
    func_iou_denom = len(afn_tails | gfn_tails)
    func_iou = func_iou_num / func_iou_denom if func_iou_denom > 0 else 1.0

    has_files = bool(gf)
    has_funcs = bool(gfn)
    if has_files and has_funcs:
        nav_score = 0.5 * file_f1 + 0.5 * func_f1
        nav_recall = 0.5 * file_recall + 0.5 * func_recall
        nav_iou = 0.5 * file_iou + 0.5 * func_iou
    elif has_files:
        nav_score = file_f1
        nav_recall = file_recall
        nav_iou = file_iou
    elif has_funcs:
        nav_score = func_f1
        nav_recall = func_recall
        nav_iou = func_iou
    else:
        nav_score = 1.0
        nav_recall = 1.0
        nav_iou = 1.0

    return {
        "navigation_score": round(nav_score, 4),
        "navigation_recall": round(nav_recall, 4),
        "navigation_iou": round(nav_iou, 4),
        "file_recall": round(file_recall, 4),
        "file_precision": round(file_precision, 4),
        "file_f1": round(file_f1, 4),
        "file_iou": round(file_iou, 4),
        "function_recall": round(func_recall, 4),
        "function_precision": round(func_precision, 4),
        "function_f1": round(func_f1, 4),
        "function_iou": round(func_iou, 4),
    }


# ---------------------------------------------------------------------------
# Trial discovery + identity
# ---------------------------------------------------------------------------


def _find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Yield trial dirs (dirs containing verifier/reward.json + result.json)."""
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"WARN: missing root {root}", file=sys.stderr)
            continue
        for entry in root.rglob("verifier/reward.json"):
            trial = entry.parent.parent
            if (trial / "result.json").is_file():
                out.append(trial)
    return out


def _identity_from_trial(trial_dir: Path) -> tuple[str, str]:
    """Return (agent, model) by reading result.json::config.agent."""
    try:
        r = json.loads((trial_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ("unknown", "unknown")
    ac = (r.get("config") or {}).get("agent") or {}
    name = ac.get("name") or "unknown"
    model = ac.get("model_name") or "unknown"
    effort = (ac.get("kwargs") or {}).get("reasoning_effort")
    if effort and model != "unknown":
        model = f"{model} / effort={effort}"
    return (name, model)


def _task_name_from_trial(trial_dir: Path) -> str:
    try:
        r = json.loads((trial_dir / "result.json").read_text())
        return r.get("task_name") or trial_dir.name.rsplit("-", 1)[0]
    except (OSError, json.JSONDecodeError):
        return trial_dir.name.rsplit("-", 1)[0]


def _load_provenance_map(tasks_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not tasks_dir.is_dir():
        return out
    for td in tasks_dir.iterdir():
        if not td.is_dir():
            continue
        prov = td / "provenance.json"
        if not prov.is_file():
            out[td.name] = ""
            continue
        try:
            data = json.loads(prov.read_text())
        except (OSError, json.JSONDecodeError):
            out[td.name] = ""
            continue
        out[td.name] = data.get("parent_t2_task") or ""
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "agent",
    "model",
    "task",
    "parent_task_id",
    "trial_index",  # synthesized: order of discovery within (agent, model, task)
    "infra_fail",
    "patchgold_file_recall",
    "patchgold_file_precision",
    "patchgold_file_f1",
    "patchgold_file_iou",
    "patchgold_function_recall",
    "patchgold_function_precision",
    "patchgold_function_f1",
    "patchgold_function_iou",
    "patchgold_navigation_score",
    "patchgold_navigation_recall",
    "patchgold_navigation_iou",
    # Original (gold-review-curated) scores carried alongside for direct comparison.
    "original_navigation_score",
    "original_file_f1",
    "original_function_f1",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--search-roots",
        nargs="+",
        type=Path,
        required=True,
        help="One or more dirs containing search trial subdirs (each holds verifier/reward.json).",
    )
    ap.add_argument(
        "--patch-gold",
        type=Path,
        required=True,
        help="Patch-derived gold JSON (output of extract_v2b_patch_gold.py).",
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Search-side dataset dir (for provenance.json → parent_task_id lookup).",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    patch_gold: dict[str, dict] = json.loads(args.patch_gold.read_text())
    print(f"Loaded patch-gold for {len(patch_gold)} tasks", file=sys.stderr)

    parent_map = _load_provenance_map(args.tasks_dir)
    print(f"Loaded provenance for {len(parent_map)} search tasks", file=sys.stderr)

    trials = _find_trial_dirs(args.search_roots)
    print(f"Found {len(trials)} search trials", file=sys.stderr)

    # Track trial_index per (agent, model, task) deterministically — by sorted
    # trial dir path so multiple invocations produce the same indices.
    trials.sort(key=lambda p: str(p))
    seen: dict[tuple[str, str, str], int] = {}

    n_scored = 0
    n_no_parent = 0
    n_no_gold = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for td in trials:
            agent, model = _identity_from_trial(td)
            task = _task_name_from_trial(td)
            parent = parent_map.get(task, "")
            key = (agent, model, task)
            tidx = seen.get(key, 0)
            seen[key] = tidx + 1

            try:
                rj = json.loads((td / "verifier" / "reward.json").read_text())
            except (OSError, json.JSONDecodeError):
                w.writerow([agent, model, task, parent, tidx, 1] + [""] * (len(CSV_HEADER) - 6))
                continue

            agent_files = rj.get("agent_files") or []
            agent_funcs = rj.get("agent_functions") or []
            original_nav = rj.get("navigation_score")
            original_file_f1 = rj.get("file_f1")
            original_func_f1 = rj.get("function_f1")

            if not parent:
                n_no_parent += 1
                w.writerow(
                    [agent, model, task, parent, tidx, 0]
                    + [""] * 11  # blank patchgold columns
                    + [
                        "" if original_nav is None else original_nav,
                        "" if original_file_f1 is None else original_file_f1,
                        "" if original_func_f1 is None else original_func_f1,
                    ]
                )
                continue

            gold = patch_gold.get(parent)
            if gold is None:
                n_no_gold += 1
                w.writerow(
                    [agent, model, task, parent, tidx, 0]
                    + [""] * 11
                    + [
                        "" if original_nav is None else original_nav,
                        "" if original_file_f1 is None else original_file_f1,
                        "" if original_func_f1 is None else original_func_f1,
                    ]
                )
                continue

            scored = _score_against_gold(
                agent_files=agent_files,
                agent_functions=agent_funcs,
                gold_files=gold.get("files") or [],
                gold_functions=gold.get("functions") or [],
            )
            n_scored += 1
            w.writerow(
                [
                    agent,
                    model,
                    task,
                    parent,
                    tidx,
                    0,
                    scored["file_recall"],
                    scored["file_precision"],
                    scored["file_f1"],
                    scored["file_iou"],
                    scored["function_recall"],
                    scored["function_precision"],
                    scored["function_f1"],
                    scored["function_iou"],
                    scored["navigation_score"],
                    scored["navigation_recall"],
                    scored["navigation_iou"],
                    "" if original_nav is None else original_nav,
                    "" if original_file_f1 is None else original_file_f1,
                    "" if original_func_f1 is None else original_func_f1,
                ]
            )

    print(
        f"\nWrote {len(trials)} trial rows → {args.output}\n"
        f"  scored against patch-gold: {n_scored}\n"
        f"  trials with no parent (search task without provenance): {n_no_parent}\n"
        f"  trials with no patch-gold for parent: {n_no_gold}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
