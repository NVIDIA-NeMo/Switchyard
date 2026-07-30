# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""summarize_search_baseline.py — search baseline summary from Harbor trial dirs.

Walks one or more Harbor baseline output dirs (the dirs `scripts/run-baselines.sh`
writes against a search-task suite, e.g. the extracted `search-v2-accflag-*`
tarballs) and emits markdown tables per (agent, model):

  1. Reward — headline reward + 4 search subscores. Missing trials are
     counted as `reward=0` across all metrics (same infra-fail accounting
     as `summarize_baseline.py`).
  2. Score variants — per-rule comparison (strict F1 / lenient F1 / IoU /
     recall flavors) for the same trial corpus.
  3. Efficiency — mean turns / tools / tokens / wall-clock per task. Means
     are taken across non-missing trials only (zero-filling efficiency
     metrics for infra-fails would be misleading); the `n` column shows
     how many trials contributed.

Multi-trial input: when a single (agent, model) ran the same task more than
once (e.g. variance-study runs), all trials are kept and aggregated. Per-task
mean is taken first, then aggregated to the cohort. The `±σ` column on the
reward shows mean within-task stdev (= "if you reran the same model on the
same cohort, how much would the headline number bounce?"). Cells where every
task has only one trial (k=1) report `±0.000`.

Default cohort = `references/search-tasks.txt` (resolved relative to this
script's parent directory). Denominator is fixed by the cohort size; any task
not seen for a model counts as an infra fail.

Override with `--cohort-tasks-file PATH`, `--exclude TASKID,...`, or
`--no-cohort-check` to disable cohort filling and use trials-seen as the
denominator.

Multi-trial input: when more than one trial per (agent, model, task) is found
(e.g. a model rerun against the same cohort for variance analysis), all trials
are aggregated. Per-task mean is computed first, then aggregated to the
cohort. The reward column shows `mean ± σ` where σ is mean within-task
stdev — the rerun-jitter you'd expect from a fresh run.

Runs split across multiple job-dirs that don't overlap on tasks
(e.g. opus47-high + opus47-high-resume) merge cleanly with σ=0.

Model identity comes from `run_manifest.json` adjacent to each job-dir
(`agent.name` + `agent.model` + `reasoning.effort`). Falls back to a trial's
`result.json::config.agent.{name,model_name}` if no manifest is present.

Usage:
  uv run python scripts/summarize_search_baseline.py <root>... \\
      [--alias JOB_DIR=LABEL]... [--cohort-tasks-file PATH] \\
      [--exclude TASKID,...] [--no-cohort-check]

Examples:
  # Default — pin to canonical cohort, missing tasks counted as infra-fail
  uv run python scripts/summarize_search_baseline.py ~/Downloads/search-v2-accflag-*/

  # Merge a split run (opus47 ran in two batches)
  uv run python scripts/summarize_search_baseline.py \\
      ~/Downloads/search-v2-accflag-opus47-high \\
      ~/Downloads/search-v2-accflag-opus47-high-resume

  # Drop a flaky task ad-hoc
  uv run python scripts/summarize_search_baseline.py <root>... \\
      --exclude craft-tunix-c-066ae850

  # No cohort filling — denominator = trials seen
  uv run python scripts/summarize_search_baseline.py <root>... --no-cohort-check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_COHORT_FILE = Path(__file__).resolve().parent.parent / "references" / "search-tasks.txt"

# Search reward.json subscores reported alongside the headline reward.
#
# As of the Apr 2026 verifier with strict-recall + lenient-precision +
# Jaccard IoU support:
#   navigation_score        = 0.5·file_f1 + 0.5·func_f1, where each F1 is
#                             strict-recall + lenient-precision.
#   navigation_score_lenient = same, but with alts crediting recall (capped).
#   navigation_recall       = pure 0.5·file_recall + 0.5·func_recall (strict).
#   navigation_recall_lenient = lenient variant.
#   navigation_iou          = 0.5·file_iou + 0.5·func_iou (Option B Jaccard).
#
# Older trial corpora carry recall-only `nav_score` in `navigation_score` and
# don't populate the precision/F1/IoU fields — those render as `—` in tables.
SUBSCORE_KEYS = (
    "navigation_score",
    "navigation_score_lenient",
    "navigation_recall",
    "navigation_recall_lenient",
    "navigation_iou",
    "assertion_coverage",
    "file_recall",
    "file_recall_lenient",
    "file_precision",
    "file_f1",
    "file_f1_lenient",
    "file_iou",
    "function_recall",
    "function_recall_lenient",
    "function_precision",
    "function_f1",
    "function_f1_lenient",
    "function_iou",
)

# Efficiency metrics pulled from reward.json::process_metrics + result.json.
EFFICIENCY_KEYS = (
    "agent_steps",  # turns
    "tool_call_count",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "wall_clock_sec",
)


def _mean(xs: list[float]) -> float:
    """Arithmetic mean. Returns 0 on empty list (caller is responsible for that case)."""
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    """Sample stdev (Bessel-corrected). Returns 0 for n<2."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Harbor writes ISO-8601 with trailing 'Z' for UTC; datetime.fromisoformat
        # handles 'Z' from Python 3.11+.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_clock_sec(result: dict, process_metrics: dict) -> float | None:
    """Wall-clock for the agent's solve attempt.

    Prefer `result.json::agent_execution.{started_at,finished_at}` because it
    excludes environment_setup + verifier overhead and is present on every
    agent. Falls back to `process_metrics.execution_time_sec` (opencode/codex
    populate it; claude-code does not).
    """
    ax = result.get("agent_execution") or {}
    start = _parse_iso(ax.get("started_at"))
    end = _parse_iso(ax.get("finished_at"))
    if start and end:
        return (end - start).total_seconds()
    raw = process_metrics.get("execution_time_sec")
    return float(raw) if raw is not None else None


def find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Yield every trial dir (containing `verifier/reward.json` + `result.json`)."""
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"WARN: missing root {root}", file=sys.stderr)
            continue
        if (root / "verifier" / "reward.json").exists() and (root / "result.json").exists():
            out.append(root)
            continue
        for entry in root.rglob("verifier/reward.json"):
            trial = entry.parent.parent
            if (trial / "result.json").exists():
                out.append(trial)
    return out


def find_job_dirs(roots: list[Path]) -> list[Path]:
    """Yield every job-dir (containing `run_manifest.json`).

    Used to seed model rows even when a run produced zero readable trials —
    otherwise the cohort-mode infra-fail accounting silently drops the failed
    run instead of reporting it as 100% infra-fail.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if (root / "run_manifest.json").is_file():
            if root not in seen:
                out.append(root)
                seen.add(root)
            continue
        for entry in root.rglob("run_manifest.json"):
            job_dir = entry.parent
            if job_dir not in seen:
                out.append(job_dir)
                seen.add(job_dir)
    return out


def _job_dir_identity(job_dir: Path) -> tuple[str, str] | None:
    """Return (agent, model_with_effort) from the job-dir's run_manifest.json.

    The model field is suffixed with `effort=X` when reasoning effort is set,
    so two runs of the same model at different efforts get distinct rows.
    Returns None if the manifest is missing or unparseable.
    """
    manifest_path = job_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    agent = manifest.get("agent") or {}
    name = agent.get("name") or ""
    model = agent.get("model") or ""
    effort = (manifest.get("reasoning") or {}).get("effort")
    if not name and not model:
        return None
    if effort:
        model = f"{model} / effort={effort}" if model else f"effort={effort}"
    return (name or "unknown", model or "unknown")


def _result_identity(result: dict) -> tuple[str, str]:
    """Fallback (agent, model) from `result.json::config.agent` when no manifest exists.

    Effort is appended to the model when present in `config.agent.kwargs.reasoning_effort`,
    matching the `_job_dir_identity` shape so manifest-fed and result-fed rows merge.
    """
    agent_cfg = ((result.get("config") or {}).get("agent")) or {}
    name = agent_cfg.get("name") or "unknown"
    model = agent_cfg.get("model_name") or "unknown"
    effort = (agent_cfg.get("kwargs") or {}).get("reasoning_effort")
    if effort and model != "unknown":
        model = f"{model} / effort={effort}"
    elif effort:
        model = f"effort={effort}"
    return (name, model)


def _apply_alias(value: str, base: tuple[str, str]) -> tuple[str, str]:
    """Apply an --alias value to a base (agent, model) identity.

    Alias forms:
      "agent / model"  -> overrides both
      "model"          -> overrides model only (keeps base agent)
    """
    if " / " in value:
        agent, model = value.split(" / ", 1)
        return (agent.strip(), model.strip())
    return (base[0], value.strip())


def load_trial(trial_dir: Path, aliases: dict[str, str]) -> dict | None:
    """Read one trial. Returns None if reward.json or result.json is unparseable.

    Supported trial-dir layouts (the alias key is the matching ancestor name):
      - `<root>/<job-dir>/<trial>/...`              run-baselines.sh + Harbor (k=1 baselines)
      - `<root>/<trial>/...`                        flat harbor-jobs dirs (k=N variance runs)
      - `<root>/<run-group>/<job-dir>/<trial>/...`  tools-side `summarize_baseline.py` shape

    Identity resolution order:
      1. nearest ancestor with `run_manifest.json` (canonical for run-baselines.sh)
         else `result.json::config.agent.{name,model_name,kwargs.reasoning_effort}`
      2. `--alias <ancestor-name>=LABEL` override — tries each ancestor name from
         `trial.parent` upward, first match wins. Lets aliases work across all
         layouts without requiring callers to know the depth.
    """
    try:
        reward = json.loads((trial_dir / "verifier" / "reward.json").read_text())
        result = json.loads((trial_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None

    # Trial dir convention: `<task_name>-<7-char-shortuuid>`. Prefer
    # result.json's `task_name` field; fall back to stripping the suffix.
    task_name = result.get("task_name") or trial_dir.name.rsplit("-", 1)[0]

    # Walk up looking for the nearest run_manifest.json. Bound the walk at 4
    # levels to avoid hitting the filesystem root on weird inputs. The trial
    # itself is excluded — manifests live alongside or above the trial.
    manifest_dir: Path | None = None
    for ancestor in list(trial_dir.parents)[:4]:
        if (ancestor / "run_manifest.json").is_file():
            manifest_dir = ancestor
            break

    base = (_job_dir_identity(manifest_dir) if manifest_dir else None) or _result_identity(result)

    # Try aliases against each ancestor name from closest-to-trial outward.
    agent, model = base
    for ancestor in list(trial_dir.parents)[:4]:
        if ancestor.name in aliases:
            agent, model = _apply_alias(aliases[ancestor.name], base)
            break

    rec: dict = {
        "task": task_name,
        "agent": agent,
        "model": model,
        "reward": float(reward.get("reward") or 0.0),
    }
    # Use None for absent fields (older trial corpora pre-date the F1 verifier
    # change and lack file_precision/file_f1/function_precision/function_f1).
    # Aggregation skips None so old runs report `—` instead of a fake 0.0.
    for key in SUBSCORE_KEYS:
        v = reward.get(key)
        rec[key] = float(v) if v is not None else None

    process_metrics = reward.get("process_metrics") or {}
    rec["agent_steps"] = process_metrics.get("agent_steps")
    rec["tool_call_count"] = process_metrics.get("tool_call_count")
    rec["input_tokens"] = process_metrics.get("input_tokens")
    rec["cached_tokens"] = process_metrics.get("cached_tokens")
    rec["output_tokens"] = process_metrics.get("output_tokens")
    rec["wall_clock_sec"] = _wall_clock_sec(result, process_metrics)
    return rec


def _read_cohort_file(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def _load_provenance_map(tasks_dir: Path) -> dict[str, str]:
    """Walk `<tasks_dir>/<task_name>/provenance.json` and return task_name → parent_t2_task.

    Tasks without a provenance file or without a `parent_t2_task` field map to
    the empty string (so the CSV column is still populated, just blank). Used
    for the search-side CSV's parent_task_id column, enabling cross-summarizer
    correlation against the tools-side per-task CSV.
    """
    out: dict[str, str] = {}
    if not tasks_dir.is_dir():
        return out
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        prov = task_dir / "provenance.json"
        if not prov.is_file():
            out[task_dir.name] = ""
            continue
        try:
            data = json.loads(prov.read_text())
        except (OSError, json.JSONDecodeError):
            out[task_dir.name] = ""
            continue
        out[task_dir.name] = data.get("parent_t2_task") or ""
    return out


# CSV column order. Stable so downstream scripts can rely on positions.
# Scoring columns mirror the SUBSCORE_KEYS the verifier writes; efficiency
# columns mirror the markdown table.
_CSV_HEADER = [
    "agent",
    "model",
    "task",
    "parent_task_id",
    "trial_index",
    "infra_fail",
    "reward",
    "navigation_score",
    "navigation_score_lenient",
    "navigation_recall",
    "navigation_recall_lenient",
    "navigation_iou",
    "assertion_coverage",
    "file_recall",
    "file_recall_lenient",
    "file_precision",
    "file_f1",
    "file_f1_lenient",
    "file_iou",
    "function_recall",
    "function_recall_lenient",
    "function_precision",
    "function_f1",
    "function_f1_lenient",
    "function_iou",
    "agent_steps",
    "tool_call_count",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "wall_clock_sec",
]


def _csv_infra_row(agent: str, model: str, task: str, parent: str) -> list:
    """Build a CSV row for an infra-fail (no trial seen). Reward zero-filled
    per the cohort-mode convention; subscores and efficiency left blank."""
    row = [agent, model, task, parent, 0, 1, 0]
    # All metric columns blank for infra-fails — zero-filling subscores and
    # efficiency would bias downstream means and Pearson correlations.
    row.extend([""] * (len(_CSV_HEADER) - len(row)))
    return row


def _csv_trial_row(agent: str, model: str, task: str, parent: str, trial_index: int, rec: dict) -> list:
    """Build a CSV row for one observed trial. Missing optional fields blank."""

    def _v(key: str) -> str | float | int:
        v = rec.get(key)
        return "" if v is None else v

    return [
        agent,
        model,
        task,
        parent,
        trial_index,
        0,
        rec["reward"],
        _v("navigation_score"),
        _v("navigation_score_lenient"),
        _v("navigation_recall"),
        _v("navigation_recall_lenient"),
        _v("navigation_iou"),
        _v("assertion_coverage"),
        _v("file_recall"),
        _v("file_recall_lenient"),
        _v("file_precision"),
        _v("file_f1"),
        _v("file_f1_lenient"),
        _v("file_iou"),
        _v("function_recall"),
        _v("function_recall_lenient"),
        _v("function_precision"),
        _v("function_f1"),
        _v("function_f1_lenient"),
        _v("function_iou"),
        _v("agent_steps"),
        _v("tool_call_count"),
        _v("input_tokens"),
        _v("cached_tokens"),
        _v("output_tokens"),
        _v("wall_clock_sec"),
    ]


def _parse_excludes(values: list[str]) -> set[str]:
    out: set[str] = set()
    for v in values:
        for tok in v.split(","):
            tok = tok.strip()
            if tok:
                out.add(tok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("roots", nargs="+", type=Path, help="Trial dirs OR parent dirs containing trials.")
    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="JOB_DIR=LABEL",
        help="Override (agent, model) for trials whose job-dir basename matches "
        "JOB_DIR. LABEL forms: 'agent / model' (overrides both) or 'model' "
        "(overrides model only). Useful for vllm-served runs whose manifest "
        "carries model='model'. Repeatable.",
    )
    ap.add_argument(
        "--cohort-tasks-file",
        type=Path,
        default=None,
        help="Cohort definition file (one task name per line; '#' comments OK). "
        f"Default: references/{DEFAULT_COHORT_FILE.name}",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TASKID[,TASKID...]",
        help="Drop these task IDs from the cohort for this run. Repeatable.",
    )
    ap.add_argument(
        "--no-cohort-check",
        action="store_true",
        help="Disable cohort filling. Denominator becomes trials seen per model.",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-trial CSV with one row per (agent, model, task, trial_index). "
        "Includes infra-fail rows (zero-filled, trial_index=0) so every cohort "
        "task is represented per (agent, model) cell. Use with --tasks-dir to "
        "populate the parent_task_id column for cross-summarizer correlation.",
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the harbor-tasks directory of the search dataset (e.g. "
        "~/projects/craft-bench/harbor-tasks/craft-search-from-v2/harbor-tasks). "
        "Used to look up each task's `provenance.json::parent_t2_task` for the "
        "CSV's parent_task_id column. Required when --csv is used; otherwise ignored.",
    )
    args = ap.parse_args()

    aliases: dict[str, str] = {}
    for a in args.alias:
        if "=" not in a:
            print(f"WARN: malformed --alias {a!r}; need JOB_DIR=LABEL", file=sys.stderr)
            continue
        k, v = a.split("=", 1)
        aliases[k.strip()] = v.strip()

    # CSV mode requires --tasks-dir for the parent_task_id lookup. Validating
    # upfront so we don't run the whole pipeline only to fail at the writer.
    parent_map: dict[str, str] = {}
    if args.csv:
        if args.tasks_dir is None:
            print(
                "ERROR: --csv requires --tasks-dir (path to the search dataset's harbor-tasks/)",
                file=sys.stderr,
            )
            return 2
        if not args.tasks_dir.is_dir():
            print(f"ERROR: --tasks-dir {args.tasks_dir} not found", file=sys.stderr)
            return 2
        parent_map = _load_provenance_map(args.tasks_dir)
        n_with_parent = sum(1 for v in parent_map.values() if v)
        print(
            f"Loaded provenance for {len(parent_map)} tasks "
            f"({n_with_parent} with parent_t2_task) from {args.tasks_dir}",
            file=sys.stderr,
        )

    excludes = _parse_excludes(args.exclude)

    cohort: set[str] | None = None
    cohort_label = ""
    if args.no_cohort_check:
        if args.cohort_tasks_file:
            print(
                "ERROR: --no-cohort-check and --cohort-tasks-file are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        if excludes:
            print(
                "ERROR: --no-cohort-check and --exclude are mutually exclusive",
                file=sys.stderr,
            )
            return 2
    else:
        cohort_path = args.cohort_tasks_file or DEFAULT_COHORT_FILE
        if not cohort_path.is_file():
            print(f"ERROR: cohort file {cohort_path} not found", file=sys.stderr)
            print(
                "Hint: either commit a cohort file or pass --no-cohort-check.",
                file=sys.stderr,
            )
            return 2
        cohort = _read_cohort_file(cohort_path)
        if excludes:
            unknown = excludes - cohort
            if unknown:
                print(
                    f"WARN: --exclude tasks not present in cohort: {sorted(unknown)}",
                    file=sys.stderr,
                )
            applied = excludes & cohort
            cohort -= applied
            cohort_label = f"{cohort_path} ({len(cohort)} tasks, excludes: {len(applied)})"
        else:
            cohort_label = f"{cohort_path} ({len(cohort)} tasks)"

    # Seed `by_identity` from job-dir manifests first so a run that produced
    # zero readable trials still appears as a row of all infra-fails (cohort
    # mode) rather than vanishing from the report.
    #
    # Per (identity, task) we keep a LIST of trials, not a single record.
    # Repeat runs of the same model on the same task are kept for variance.
    by_identity: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for job_dir in find_job_dirs(args.roots):
        base = _job_dir_identity(job_dir)
        if base is None:
            continue
        if job_dir.name in aliases:
            identity = _apply_alias(aliases[job_dir.name], base)
        else:
            identity = base
        # touch the entry so manifest-only runs (zero readable trials) still appear
        _ = by_identity[identity]

    trial_dirs = find_trial_dirs(args.roots)
    print(f"Found {len(trial_dirs)} trial dirs across {len(args.roots)} root(s)", file=sys.stderr)

    skipped = 0
    for td in trial_dirs:
        rec = load_trial(td, aliases)
        if rec is None:
            skipped += 1
            continue
        identity = (rec["agent"], rec["model"])
        by_identity[identity][rec["task"]].append(rec)
    if skipped:
        print(f"Skipped {skipped} unreadable trials", file=sys.stderr)

    if not by_identity:
        print("ERROR: no manifests or trials found", file=sys.stderr)
        return 2

    rows = []
    for identity in sorted(by_identity):
        per_task = by_identity[identity]
        seen = set(per_task)
        if cohort is not None:
            target = cohort
            missing = sorted(target - seen)
            extra = sorted(seen - target)
        else:
            target = seen
            missing = []
            extra = []

        n = len(target) if target else 0
        infra = 0
        # Per-cohort-task means + within-task stdevs, then aggregated. Reward
        # is cohort-divided (zeros for infra-fail tasks). Subscores skip Nones
        # so older trial corpora that lack F1/IoU fields don't fake-zero them.
        reward_task_means: list[float] = []
        reward_task_stdevs: list[float] = []
        sub_task_values: dict[str, list[tuple[float, float]]] = {k: [] for k in SUBSCORE_KEYS}
        # Efficiency stays simple-mean across all trials (no per-task fold) —
        # variance there is mostly noise of-no-interest; we just want a
        # representative cost figure.
        eff_sums: dict[str, float] = {k: 0.0 for k in EFFICIENCY_KEYS}
        eff_counts: dict[str, int] = {k: 0 for k in EFFICIENCY_KEYS}
        # k_per_task tracks trial-count-per-cohort-task so the table can show
        # whether the dataset is k=1 (single-shot) or k=multi (variance run).
        k_per_task: list[int] = []

        for t in target:
            recs = per_task.get(t) or []
            if not recs:
                infra += 1
                continue
            k_per_task.append(len(recs))
            # Reward: per-task mean + stdev across trials.
            r_vals = [r["reward"] for r in recs]
            reward_task_means.append(_mean(r_vals))
            reward_task_stdevs.append(_stdev(r_vals))
            # Subscores: per-task (mean, stdev) over non-None values.
            for k in SUBSCORE_KEYS:
                vals = [r.get(k) for r in recs]
                vals = [float(v) for v in vals if v is not None]
                if vals:
                    sub_task_values[k].append((_mean(vals), _stdev(vals)))
            for k in EFFICIENCY_KEYS:
                for r in recs:
                    v = r.get(k)
                    if v is None:
                        continue
                    eff_sums[k] += float(v)
                    eff_counts[k] += 1

        # Cohort-mode reward stays cohort-divided (zeros for infra-fail tasks).
        # mean reward = average of per-task means, with infra-fails zero-filled
        # so the headline matches the existing convention.
        reward_mean = (sum(reward_task_means)) / n if n else 0.0
        # Within-task stdev, averaged across observed (non-infra) tasks. This is
        # the per-rerun-jitter on the headline. Multiplying by 0 for infra-fails
        # would distort the signal, so we average across observed tasks only.
        reward_stdev = _mean(reward_task_stdevs) if reward_task_stdevs else 0.0

        sub_means: dict[str, float | None] = {}
        sub_stdevs: dict[str, float | None] = {}
        for k, pairs in sub_task_values.items():
            if not pairs:
                sub_means[k] = None
                sub_stdevs[k] = None
                continue
            sub_means[k] = _mean([m for m, _ in pairs])
            sub_stdevs[k] = _mean([s for _, s in pairs])

        eff_means = {k: (eff_sums[k] / eff_counts[k] if eff_counts[k] else None) for k in EFFICIENCY_KEYS}
        eff_n = max(eff_counts.values()) if eff_counts else 0

        k_min = min(k_per_task) if k_per_task else 0
        k_max = max(k_per_task) if k_per_task else 0
        rows.append(
            {
                "agent": identity[0],
                "model": identity[1],
                "n": n,
                "infra": infra,
                "trials_seen": len(seen),
                "missing": len(missing),
                "extra": len(extra),
                "eff_n": eff_n,
                "eff": eff_means,
                "reward": reward_mean,
                "reward_stdev": reward_stdev,
                "sub": sub_means,
                "sub_stdev": sub_stdevs,
                "k_min": k_min,
                "k_max": k_max,
                "trial_count": sum(k_per_task),
            }
        )

    print()
    if cohort_label:
        print(f"# Search baseline — cohort: **{cohort_label}**")
    else:
        print("# Search baseline — `--no-cohort-check` (denominator = trials seen per model)")
    print()

    def _f(v: float | None, decimals: int = 3) -> str:
        return "—" if v is None else f"{v:.{decimals}f}"

    def _ms(mean: float | None, stdev: float | None, decimals: int = 3) -> str:
        """Render `mean ± stdev`. `—` if mean is None; drops `± 0.000` for k=1 cells."""
        if mean is None:
            return "—"
        if not stdev:
            return f"{mean:.{decimals}f}"
        return f"{mean:.{decimals}f} ± {stdev:.{decimals}f}"

    def _trio(sub: dict, key_f1: str, key_p: str, key_r: str, bold_f1: bool = False) -> str:
        f1 = sub.get(key_f1)
        p = sub.get(key_p)
        r = sub.get(key_r)
        f1_str = _f(f1)
        if bold_f1 and f1 is not None:
            f1_str = f"**{f1_str}**"
        return f"{f1_str} ({_f(p, 2)} / {_f(r, 2)})"

    def _kstr(r: dict) -> str:
        """Render trial-count column: `k=N` if uniform, `k=min..max` otherwise."""
        if r["k_min"] == r["k_max"]:
            return f"k={r['k_min']}"
        return f"k={r['k_min']}..{r['k_max']}"

    def _winners(values: list[float | None], higher_is_better: bool) -> set[int]:
        """Return row indices that hold the best value for this column.

        Skips None and NaN. Ties bold all tied rows. Returns empty set if no
        comparable values exist (column entirely `—`) — caller renders normally.
        """
        comparable = [(i, v) for i, v in enumerate(values) if v is not None]
        if not comparable:
            return set()
        best = (
            max(comparable, key=lambda iv: iv[1])[1]
            if higher_is_better
            else min(comparable, key=lambda iv: iv[1])[1]
        )
        # Float equality is fine here — winners are derived from the same
        # cohort-mean computation, no rounding-vs-comparison drift.
        return {i for i, v in comparable if v == best}

    def _bold(s: str, mark: bool) -> str:
        # Bold the value but leave any trailing `± stdev` alone; adding the
        # markers around the whole string would bold the σ too which reads as
        # noisy in the table.
        if not mark or s == "—":
            return s
        if " ± " in s:
            mean_part, stdev_part = s.split(" ± ", 1)
            return f"**{mean_part}** ± {stdev_part}"
        return f"**{s}**"

    # Per-column winner sets. Higher-is-better for reward + nav + assert + F1s.
    win_reward = _winners([r["reward"] for r in rows], higher_is_better=True)
    win_nav = _winners([r["sub"].get("navigation_score") for r in rows], higher_is_better=True)
    win_assert = _winners([r["sub"].get("assertion_coverage") for r in rows], higher_is_better=True)
    win_file_f1 = _winners([r["sub"].get("file_f1") for r in rows], higher_is_better=True)
    win_func_f1 = _winners([r["sub"].get("function_f1") for r in rows], higher_is_better=True)

    print("| Agent | Model | Reward | Nav | Assert | File F1 (P / R) | Func F1 (P / R) | n | k |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows):
        scored = r["n"] - r["infra"]
        sub = r["sub"]
        sd = r["sub_stdev"]
        reward = _bold(_ms(r["reward"], r["reward_stdev"]), i in win_reward)
        nav = _bold(_ms(sub.get("navigation_score"), sd.get("navigation_score")), i in win_nav)
        assertc = _bold(_ms(sub.get("assertion_coverage"), sd.get("assertion_coverage")), i in win_assert)
        file_cell = _trio(sub, "file_f1", "file_precision", "file_recall", bold_f1=i in win_file_f1)
        func_cell = _trio(
            sub, "function_f1", "function_precision", "function_recall", bold_f1=i in win_func_f1
        )
        print(
            f"| `{r['agent']}` | `{r['model']}` | {reward} | {nav} | {assertc} "
            f"| {file_cell} | {func_cell} | {scored} | {_kstr(r)} |"
        )
    print()

    print("## Score variants (mean ± within-task stdev)")
    print()
    print("Strict = strict-recall + lenient-precision F1 mix (current `navigation_score`).  ")
    print("Lenient = alts credit recall too (capped at 1.0).  ")
    print("Recall = pure recall mean (legacy pre-F1 metric).  ")
    print("IoU = Jaccard with alts in numerator only, primary in denominator (Option B).  ")
    print("σ = within-task stdev across repeat trials, averaged over the cohort. `0.000` for k=1 runs.")
    print()
    win_strict = win_nav  # already computed
    win_lenient = _winners([r["sub"].get("navigation_score_lenient") for r in rows], higher_is_better=True)
    win_recall = _winners([r["sub"].get("navigation_recall") for r in rows], higher_is_better=True)
    win_recall_l = _winners([r["sub"].get("navigation_recall_lenient") for r in rows], higher_is_better=True)
    win_iou = _winners([r["sub"].get("navigation_iou") for r in rows], higher_is_better=True)

    print("| Agent | Model | Strict F1 | Lenient F1 | Recall (strict) | Recall (lenient) | IoU | n | k |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows):
        scored = r["n"] - r["infra"]
        sub = r["sub"]
        sd = r["sub_stdev"]
        strict = _bold(_ms(sub.get("navigation_score"), sd.get("navigation_score")), i in win_strict)
        lenient = _bold(
            _ms(sub.get("navigation_score_lenient"), sd.get("navigation_score_lenient")),
            i in win_lenient,
        )
        recall = _bold(_ms(sub.get("navigation_recall"), sd.get("navigation_recall")), i in win_recall)
        recall_l = _bold(
            _ms(sub.get("navigation_recall_lenient"), sd.get("navigation_recall_lenient")),
            i in win_recall_l,
        )
        iou = _bold(_ms(sub.get("navigation_iou"), sd.get("navigation_iou")), i in win_iou)
        print(
            f"| `{r['agent']}` | `{r['model']}` | {strict} | {lenient} | {recall} "
            f"| {recall_l} | {iou} | {scored} | {_kstr(r)} |"
        )
    print()

    print("## Efficiency (mean per task, non-missing trials only)")
    print()
    print("| Agent | Model | Turns | Tools | Input tok | Cached tok | Output tok | Wall (s) | n |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    # Efficiency is lower-is-better — bold the min in each column.
    eff_winners = {k: _winners([r["eff"][k] for r in rows], higher_is_better=False) for k in EFFICIENCY_KEYS}

    def _fmt_eff(v: float | None, decimals: int) -> str:
        if v is None:
            return "—"
        if decimals == 0:
            return f"{int(round(v)):,}"
        return f"{v:,.{decimals}f}"

    eff_specs = [
        ("agent_steps", 1),
        ("tool_call_count", 1),
        ("input_tokens", 0),
        ("cached_tokens", 0),
        ("output_tokens", 0),
        ("wall_clock_sec", 1),
    ]
    for i, r in enumerate(rows):
        eff = r["eff"]
        cells = [_bold(_fmt_eff(eff[key], dec), i in eff_winners[key]) for key, dec in eff_specs]
        print(
            f"| `{r['agent']}` "
            f"| `{r['model']}` "
            f"| {cells[0]} "
            f"| {cells[1]} "
            f"| {cells[2]} "
            f"| {cells[3]} "
            f"| {cells[4]} "
            f"| {cells[5]} "
            f"| {r['eff_n']} |"
        )
    print()

    for r in rows:
        identity = f"{r['agent']} / {r['model']}"
        if cohort and r["missing"]:
            print(
                f"⚠ `{identity}` is missing {r['missing']} of {len(cohort)} cohort tasks "
                f"(counted as infra-fail with reward=0).",
                file=sys.stderr,
            )
        if cohort and r["extra"]:
            print(
                f"ℹ `{identity}` has {r['extra']} trials outside the cohort "
                f"(ignored in headline; pass --no-cohort-check to include).",
                file=sys.stderr,
            )

    # Per-trial CSV. One row per (agent, model, task, trial_index) over the
    # cohort (or trials-seen if --no-cohort-check). Infra-fail rows are
    # emitted with trial_index=0 and zero-filled scoring fields so every
    # cohort task is represented per (agent, model) cell — matches the
    # tools-side convention. Efficiency cells stay blank for infra-fails
    # (zero-filling them would bias downstream means).
    if args.csv:
        target_tasks = cohort if cohort is not None else None
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(_CSV_HEADER)
            for identity in sorted(by_identity):
                agent, model = identity
                per_task = by_identity[identity]
                tasks_to_emit = target_tasks if target_tasks is not None else set(per_task)
                for task in sorted(tasks_to_emit):
                    parent = parent_map.get(task, "")
                    recs = per_task.get(task) or []
                    if not recs:
                        # Infra-fail: one zero-filled row.
                        w.writerow(_csv_infra_row(agent, model, task, parent))
                        continue
                    for trial_index, rec in enumerate(recs):
                        w.writerow(_csv_trial_row(agent, model, task, parent, trial_index, rec))
        print(f"Wrote per-trial CSV → {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
