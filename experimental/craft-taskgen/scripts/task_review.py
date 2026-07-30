# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""task_review.py — review-artifact generator for multi-trial Harbor task results.

Reads Harbor trial directories (each with `verifier/reward.json` + `result.json`),
computes per-task pass@k and universal-fail rescue analysis, and emits two
review artifacts:

  * `review_index.csv` — one row per task; CSV intended for triage in a
    spreadsheet. Sort by `flagged_for_review`, then scan `best_f2p` /
    `n_universal_fails` to prioritize.

  * `review_md/<task_id>.md` — per-task self-contained markdown for tasks
    that no model+trial resolved (pass@k == 0). Inlines the instruction,
    enumerates F2P/P2P tests with per-test pass counts, lists every trial
    outcome. Designed to be fed wholesale to a coding assistant for guided
    inspection without needing additional context.

## Rescue analysis (informational; not in the CSV columns)

For tasks with no pass@k=True trial, we still compute whether the task
*could* be rescued by adding universally-failing F2P tests to `f2p_skip.txt`.
Rules:

  * skip set = universally-failing F2P tests (intersection of failed test
    name sets across all trials with complete failed-test listings)
  * constraints: `|skip| <= 4`, `2*|skip| < f2p_total`,
    `f2p_total - |skip| >= 2`
  * rescue eligible iff some trial achieves F2P=100% on the remaining tests
    AND that same trial has clean P2P (no regression)
  * NOT rescued if every (model, trial) tuple passes after skip — that
    would make the task trivial

The skip set (if non-empty) is included in the per-task markdown for
human consideration; the CSV does NOT include a categorical "rescue
verdict" because the labels are session jargon and would confuse external
reviewers.

## Inputs

  <root>...                         Trial dir parents (recursively walked)
  --alias RUN_DIR=LABEL             Repeatable: relabel a run-group dir
  --tasks-file FILE                 Optional: restrict analysis to listed tasks
  --harbor-tasks-root <path>        Path to harbor task dirs (e.g.
                                    `craft-bench/harbor-tasks/craft-taskgen-v2b`).
                                    Required for full markdown generation —
                                    provides instruction text, repo URL,
                                    commit SHA, and full F2P/P2P test lists
                                    needed for per-test annotations.
  --output-dir <dir>                Output dir for review_index.csv +
                                    review_md/ subdirectory.
                                    Defaults to ./review_artifacts.

## Example

Run on the v2b 92-task cohort using the 10 single-trial baseline runs
(one trial per (model, task)):

    uv run python scripts/task_review.py \\
      data/extracted/jobs/v2-opus47-claude/baseline-claude-code-craft-taskgen-v2-20260427-151228 \\
      data/extracted/jobs/v2-codex-gpt55/baseline-codex-craft-taskgen-v2-20260427-151235 \\
      data/extracted/jobs/v2-haiku-opencode/baseline-opencode-craft-taskgen-v2-20260427-151243 \\
      data/extracted/oc-minimax-m2.7-v2-20260427T234745Z/baseline-opencode-craft-taskgen-v2-20260427-184745 \\
      data/extracted/jobs/v2-qwen36-opencode/baseline-opencode-craft-taskgen-v2-20260428-064419 \\
      data/extracted/jobs/v2b-opus47-claude-high \\
      data/extracted/jobs/v2b-codex-gpt55-xhigh \\
      data/extracted/jobs/v2b-sonnet46-claude-high-rerun \\
      data/extracted/jobs/v2b-qwen35-397b-opencode-rerun \\
      data/extracted/jobs/v2b-nemotron3-super-120b-opencode \\
      --alias 'v2-opus47-claude=opus-4.7-xhigh' \\
      --alias 'v2b-opus47-claude-high=opus-4.7-high' \\
      --alias 'v2-codex-gpt55=gpt-5.5-high' \\
      --alias 'v2b-codex-gpt55-xhigh=gpt-5.5-xhigh' \\
      --alias 'v2-qwen36-opencode=qwen3.6-35b-a3b' \\
      --alias 'v2b-qwen35-397b-opencode-rerun=qwen3.5-397b' \\
      --alias 'v2b-nemotron3-super-120b-opencode=nemotron-3-super-120b' \\
      --tasks-file audit/v2b_92.txt \\
      --harbor-tasks-root ../craft-bench/harbor-tasks/craft-taskgen-v2b \\
      --output-dir audit/review_artifacts

Add multi-trial roots (e.g. `v2b30-*-4iters/`, `v2b24-*-iter1/`) as additional
positional arguments to tighten the universal-fail intersection — they
contribute extra trials per (model, task) and exclude any "fluke" universal-
fails seen in only single-trial baselines. Not required for the basic
10-baseline view above.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# vllm-served models report `model_info.name = "model"` and don't disambiguate
# distinct deployments; fall back to the run-group dir name in that case.
GENERIC_MODEL_NAMES = {"", "model", "vllm/model", "default", "unknown"}

# Rescue-skip constraints (the four rules we landed on)
MAX_SKIP_TESTS = 4
MIN_TESTS_REMAINING = 2
# (also enforced: 2*|skip| < f2p_total)


# ─── trial loading utilities ─────────────────────────────────────────────────

def find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Yield trial directories (containing verifier/reward.json + result.json)."""
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


def load_trial(trial_dir: Path, aliases: dict[str, str]) -> dict | None:
    """Read one trial. Returns None if reward.json or result.json is unparseable.

    Sets `failed_listing_complete` when the failed-test name list matches the
    expected count. Some Harbor trials report `f2p_failed=[]` even when tests
    failed (pytest crashed before listing); those trials must NOT contribute
    to the universal-failing-tests intersection.
    """
    try:
        reward = json.loads((trial_dir / "verifier" / "reward.json").read_text())
        result = json.loads((trial_dir / "result.json").read_text())
        # f2p_failed/p2p_failed moved out of reward.json (harbor>=0.13.1
        # pydantic rejects list values). Merge from reward-details.json if
        # the trial was written by the newer SCORE_PY_TEMPLATE.
        details_path = trial_dir / "verifier" / "reward-details.json"
        if details_path.exists():
            try:
                details = json.loads(details_path.read_text())
                for key in ("f2p_failed", "p2p_failed"):
                    if key in details and key not in reward:
                        reward[key] = details[key]
            except (OSError, json.JSONDecodeError):
                pass
    except (OSError, json.JSONDecodeError):
        return None
    task_name = result.get("task_name") or trial_dir.name.rsplit("-", 1)[0]
    agent_info = result.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    raw_model = model_info.get("name") or agent_info.get("name") or ""

    try:
        run_group = trial_dir.parent.parent.name
    except (AttributeError, IndexError):
        run_group = ""

    if run_group in aliases:
        model = aliases[run_group]
    elif raw_model in GENERIC_MODEL_NAMES:
        model = run_group or "unknown"
    else:
        model = raw_model

    f2p_total = int(reward.get("f2p_total") or 0)
    f2p_passed = int(reward.get("f2p_passed") or 0)
    f2p_failed_list = list(reward.get("f2p_failed") or [])
    p2p_total = int(reward.get("p2p_total") or 0)
    p2p_passed = int(reward.get("p2p_passed") or 0)
    p2p_failed_list = list(reward.get("p2p_failed") or [])

    return {
        "task": task_name,
        "model": model,
        "trial_dir": str(trial_dir),
        "trial_name": trial_dir.name,
        "resolved": bool(reward.get("resolved")),
        "f2p_total": f2p_total,
        "f2p_passed": f2p_passed,
        "f2p_failed": f2p_failed_list,
        "p2p_total": p2p_total,
        "p2p_passed": p2p_passed,
        "p2p_failed": p2p_failed_list,
        "failed_listing_complete": (len(f2p_failed_list) + f2p_passed == f2p_total),
        "p2p_listing_complete": (len(p2p_failed_list) + p2p_passed == p2p_total),
    }


# ─── harbor task metadata extraction ─────────────────────────────────────────

RE_GIT_CLONE = re.compile(
    r"git\s+clone\s+(?:--[^\s]+\s+)*https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?(?=\s|$)"
)
RE_DIAG_COMMIT = re.compile(r"^\*\*Commit:\*\*\s+([0-9a-f]{7,40})", re.MULTILINE)


def parse_task_meta(harbor_task_dir: Path) -> dict:
    """Extract repo URL, commit SHA, instruction, F2P/P2P test lists, and patch info."""
    meta: dict = {
        "repo": "",
        "commit_sha": "",
        "instruction": "",
        "f2p_tests": [],
        "p2p_tests": [],
        "patch_files": [],
    }
    if not harbor_task_dir.is_dir():
        return meta

    dockerfile = harbor_task_dir / "environment" / "Dockerfile"
    if dockerfile.is_file():
        m = RE_GIT_CLONE.search(dockerfile.read_text(encoding="utf-8", errors="replace"))
        if m:
            meta["repo"] = m.group(1)

    diag_dir = harbor_task_dir / "diagnostics"
    if diag_dir.is_dir():
        for md in sorted(diag_dir.glob("*docker_classify.md")):
            m = RE_DIAG_COMMIT.search(md.read_text(encoding="utf-8", errors="replace"))
            if m:
                meta["commit_sha"] = m.group(1)
                break

    instruction_path = harbor_task_dir / "instruction.md"
    if instruction_path.is_file():
        meta["instruction"] = instruction_path.read_text(encoding="utf-8", errors="replace").strip()

    f2p_path = harbor_task_dir / "tests" / "fail_to_pass.txt"
    if f2p_path.is_file():
        meta["f2p_tests"] = [ln.strip() for ln in f2p_path.read_text().splitlines() if ln.strip()]
    p2p_path = harbor_task_dir / "tests" / "pass_to_pass.txt"
    if p2p_path.is_file():
        meta["p2p_tests"] = [ln.strip() for ln in p2p_path.read_text().splitlines() if ln.strip()]

    patch_path = harbor_task_dir / "solution" / "changes.patch"
    if patch_path.is_file():
        files = set()
        for line in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("+++ b/"):
                files.add(line[6:].strip())
            elif line.startswith("--- a/"):
                files.add(line[6:].strip())
        meta["patch_files"] = sorted(f for f in files if f != "/dev/null")

    diag_dir = harbor_task_dir / "diagnostics"
    if diag_dir.is_dir():
        meta["diagnostics_files"] = sorted(
            str(p) for p in diag_dir.iterdir()
            if p.is_file() and p.suffix == ".md"
        )

    return meta


# ─── per-task analysis ───────────────────────────────────────────────────────

def analyze_task(task_id: str, trials: list[dict], harbor_meta: dict | None = None) -> dict:
    """Compute pass@k, per-test pass/fail counts, and skip-rescue verdict."""
    n_trials = len(trials)
    models = sorted({t["model"] for t in trials})

    # f2p_total: take modal value (in case any trial reports something weird)
    totals = [t["f2p_total"] for t in trials if t["f2p_total"] > 0]
    f2p_total = max(set(totals), key=totals.count) if totals else 0
    p2p_totals = [t["p2p_total"] for t in trials if t["p2p_total"] > 0]
    p2p_total = max(set(p2p_totals), key=p2p_totals.count) if p2p_totals else 0

    # Pass@k: trials where resolved=True
    resolved_trials = [t for t in trials if t["resolved"]]
    pass1_models = sorted({t["model"] for t in resolved_trials})

    # Best F2P score across trials
    best_f2p = 0.0
    best_f2p_passed = 0
    best_f2p_trial: dict | None = None
    for t in trials:
        score = t["f2p_passed"] / t["f2p_total"] if t["f2p_total"] else 0.0
        if score > best_f2p or (score == best_f2p and best_f2p_trial is None):
            best_f2p = score
            best_f2p_passed = t["f2p_passed"]
            best_f2p_trial = t

    # Universal-fail F2P tests (intersection of failed_set across complete trials)
    complete_trials = [t for t in trials if t["failed_listing_complete"]]
    if complete_trials:
        universally_failing = set.intersection(*(set(t["f2p_failed"]) for t in complete_trials))
    else:
        universally_failing = set()

    # Per-test pass counts (only computable if we have the full F2P test list)
    full_f2p = list(harbor_meta["f2p_tests"]) if harbor_meta and harbor_meta.get("f2p_tests") else []
    full_p2p = list(harbor_meta["p2p_tests"]) if harbor_meta and harbor_meta.get("p2p_tests") else []

    f2p_test_pass_counts: dict[str, dict] = {}
    if full_f2p:
        for test in full_f2p:
            f2p_test_pass_counts[test] = {
                "passed_count": 0,
                "total_complete_trials": 0,
                "passed_models": set(),
            }
        for t in complete_trials:
            failed_set = set(t["f2p_failed"])
            for test in full_f2p:
                f2p_test_pass_counts[test]["total_complete_trials"] += 1
                if test not in failed_set:
                    f2p_test_pass_counts[test]["passed_count"] += 1
                    f2p_test_pass_counts[test]["passed_models"].add(t["model"])

    # P2P regression detection: any trial with clean F2P=100% but a P2P regression
    # (similar to F2P, count regressions per test)
    p2p_test_regression_counts: dict[str, dict] = {}
    if full_p2p:
        for test in full_p2p:
            p2p_test_regression_counts[test] = {
                "regressed_count": 0,
                "regressed_models": set(),
            }
        for t in trials:
            if not t["p2p_listing_complete"]:
                continue
            for test in t["p2p_failed"]:
                if test in p2p_test_regression_counts:
                    p2p_test_regression_counts[test]["regressed_count"] += 1
                    p2p_test_regression_counts[test]["regressed_models"].add(t["model"])

    # Skip-rescue feasibility (informational, surfaced in per-task markdown)
    skip = universally_failing
    skip_count = len(skip)
    skip_constraint_violations: list[str] = []
    if skip_count > MAX_SKIP_TESTS:
        skip_constraint_violations.append(f"|skip|={skip_count} > {MAX_SKIP_TESTS}")
    if 2 * skip_count >= f2p_total:
        skip_constraint_violations.append(f"|skip|={skip_count} >= half_total ({f2p_total}/2)")
    if f2p_total - skip_count < MIN_TESTS_REMAINING:
        skip_constraint_violations.append(f"remaining={f2p_total - skip_count} < {MIN_TESTS_REMAINING}")

    # Trials that would pass after skip = trials with f2p_passed == f2p_total - skip_count
    # AND no P2P regression
    def trial_passes_after_skip(t: dict) -> bool:
        f2p_ok = t["f2p_passed"] == t["f2p_total"] - skip_count
        p2p_ok = t["p2p_total"] == 0 or t["p2p_passed"] == t["p2p_total"]
        return f2p_ok and p2p_ok

    n_trials_passing_after_skip = sum(1 for t in trials if trial_passes_after_skip(t))
    rescue_trial_models = sorted({t["model"] for t in trials if trial_passes_after_skip(t)})

    return {
        "task_id": task_id,
        "n_trials": n_trials,
        "models": models,
        "n_models": len(models),
        "f2p_total": f2p_total,
        "p2p_total": p2p_total,
        "n_pass1_trials": len(resolved_trials),
        "n_pass1_models": len(pass1_models),
        "pass1_models": pass1_models,
        "best_f2p": best_f2p,
        "best_f2p_passed": best_f2p_passed,
        "best_f2p_model": best_f2p_trial["model"] if best_f2p_trial else "",
        "best_f2p_trial_dir": best_f2p_trial["trial_dir"] if best_f2p_trial else "",
        "universally_failing": sorted(universally_failing),
        "n_universal_fails": len(universally_failing),
        "f2p_test_pass_counts": f2p_test_pass_counts,
        "p2p_test_regression_counts": p2p_test_regression_counts,
        "skip_constraint_violations": skip_constraint_violations,
        "n_trials_passing_after_skip": n_trials_passing_after_skip,
        "rescue_trial_models": rescue_trial_models,
        "trials": trials,
        "complete_trials_count": len(complete_trials),
    }


# ─── markdown rendering ──────────────────────────────────────────────────────

def render_review_markdown(analysis: dict, harbor_meta: dict, task_id: str) -> str:
    """Render the per-task review markdown for a flagged task."""
    out: list[str] = []
    a = analysis
    repo = harbor_meta.get("repo", "")
    sha = harbor_meta.get("commit_sha", "")
    instruction = harbor_meta.get("instruction", "")
    patch_files = harbor_meta.get("patch_files", [])

    out.append(f"# `{task_id}`\n")

    # Setup note — what the reviewer needs locally
    out.append("> **Setup note**: file paths in this doc are relative to (a) the "
               "craft-bench repo root (`harbor-tasks/<cohort>/<task_id>/...`) and (b) "
               "the trial dirs that were passed as input when this artifact was generated. "
               "To work with this task you'll need the craft-bench repo cloned locally "
               "AND the trial dir data accessible at the same paths.\n")

    # Source
    out.append("## Source\n")
    if repo:
        out.append(f"- Repo: https://github.com/{repo}")
    if repo and sha:
        out.append(f"- Commit: https://github.com/{repo}/commit/{sha}")
    elif sha:
        out.append(f"- Commit SHA: `{sha}`")
    out.append("")

    # Instruction
    out.append("## Instruction\n")
    if instruction:
        out.append(instruction)
    else:
        out.append("_(no instruction.md found at harbor task dir; pass `--harbor-tasks-root` to inline it)_")
    out.append("")

    # Reference solution
    if patch_files:
        out.append("## Reference solution — files touched by `solution/changes.patch`\n")
        for f in patch_files:
            out.append(f"- `{f}`")
        out.append("")

    # If the harbor task dir has a diagnostics/ subdir, list its files. These
    # are prior audit reports from earlier pipeline steps; reviewers can read
    # them for additional context but shouldn't treat them as authoritative.
    diag_files = harbor_meta.get("diagnostics_files", [])
    if diag_files:
        out.append("## Prior pipeline-audit reports (additional context)\n")
        for f in diag_files:
            out.append(f"- `{f}`")
        out.append("")

    # F2P tests
    out.append(f"## F2P tests ({a['f2p_total']} total)\n")
    f2p_counts = a["f2p_test_pass_counts"]
    if f2p_counts:
        # Three buckets
        universal = []
        sometimes = []
        always = []
        for test, info in f2p_counts.items():
            if info["passed_count"] == 0:
                universal.append(test)
            elif info["passed_count"] == info["total_complete_trials"]:
                always.append(test)
            else:
                sometimes.append((test, info))

        if universal:
            out.append(f"### Universally failing ({len(universal)} of {a['f2p_total']})")
            out.append("")
            for t in sorted(universal):
                out.append(f"- `{t}`")
            out.append("")

        if sometimes:
            out.append(f"### Sometimes passing ({len(sometimes)} of {a['f2p_total']})")
            out.append("")
            for test, info in sorted(sometimes, key=lambda x: x[0]):
                models_disp = "|".join(sorted(info["passed_models"]))
                out.append(f"- `{test}` — passed by {info['passed_count']}/{info['total_complete_trials']} trials "
                           f"(models: {models_disp})")
            out.append("")

        if always:
            out.append(f"### Always passing ({len(always)} of {a['f2p_total']})")
            out.append("")
            for t in sorted(always):
                out.append(f"- `{t}`")
            out.append("")
    else:
        # No harbor task dir for full F2P test list — show only universally-failing
        if a["universally_failing"]:
            out.append(f"### Universally failing ({a['n_universal_fails']} of {a['f2p_total']})")
            out.append("_(full F2P test list not available — pass `--harbor-tasks-root` for "
                       "complete sometimes-passing / always-passing breakdown.)_")
            out.append("")
            for t in a["universally_failing"]:
                out.append(f"- `{t}`")
            out.append("")
        else:
            out.append("_(no universally-failing F2P tests detected; full test list not available.)_")
            out.append("")

    # P2P tests (only show if any regress)
    p2p_regressed = [(t, info) for t, info in a["p2p_test_regression_counts"].items()
                      if info["regressed_count"] > 0]
    if p2p_regressed:
        out.append(f"## P2P regressions ({len(p2p_regressed)} of {a['p2p_total']})\n")
        for test, info in sorted(p2p_regressed, key=lambda x: x[0]):
            models_disp = "|".join(sorted(info["regressed_models"]))
            out.append(f"- `{test}` — regressed by {info['regressed_count']} trial(s) "
                       f"(models: {models_disp})")
        out.append("")
    else:
        out.append(f"## P2P tests ({a['p2p_total']} total)\n")
        out.append("All P2P tests pass on all observed trials.")
        out.append("")

    # Trial outcomes — one row per trial. Includes full trial_dir path so a
    # reviewer / coding assistant can navigate directly to the agent transcripts
    # and per-trial outputs without guessing which root dir each trial lives under.
    out.append(f"## Trial outcomes ({a['n_trials']} trials)\n")
    out.append("| Model | Resolved | F2P | P2P | Trial dir (full path) |")
    out.append("|---|---|---:|---:|---|")
    for t in sorted(a["trials"], key=lambda x: (x["model"], x["trial_name"])):
        f2p_disp = f"{t['f2p_passed']}/{t['f2p_total']}" if t["f2p_total"] else "—"
        p2p_disp = f"{t['p2p_passed']}/{t['p2p_total']}" if t["p2p_total"] else "—"
        out.append(f"| `{t['model']}` | {'✓' if t['resolved'] else '✗'} | "
                   f"{f2p_disp} | {p2p_disp} | `{t['trial_dir']}` |")
    out.append("")

    # Findings — facts only
    out.append("## Findings (facts only)\n")
    out.append(f"- **F2P universal-fail count**: {a['n_universal_fails']} / {a['f2p_total']}")
    out.append(f"- **F2P best score observed**: {a['best_f2p_passed']}/{a['f2p_total']} "
               f"({a['best_f2p']*100:.1f}%) — by `{a['best_f2p_model']}`")
    out.append(f"  - Trial dir: `{a['best_f2p_trial_dir']}`")
    out.append(f"- **Pass@k**: {a['n_pass1_trials']}/{a['n_trials']} trials achieved `resolved=True` "
               f"across {a['n_models']} models")
    if p2p_regressed:
        out.append(f"- **P2P regressions observed**: {len(p2p_regressed)} test(s)")
    if a["skip_constraint_violations"]:
        out.append(f"- **Universal-fail skip-rescue**: blocked — {'; '.join(a['skip_constraint_violations'])}")
    elif a["universally_failing"]:
        if a["n_trials_passing_after_skip"] > 0:
            out.append(f"- **Universal-fail skip-rescue**: feasible — skipping {a['n_universal_fails']} "
                       f"universally-failing test(s) would let "
                       f"{a['n_trials_passing_after_skip']} trial(s) achieve F2P=100% "
                       f"on remaining tests with clean P2P (models: {'|'.join(a['rescue_trial_models'])})")
        else:
            out.append(f"- **Universal-fail skip-rescue**: not feasible — even with "
                       f"{a['n_universal_fails']} universally-failing test(s) skipped, no trial "
                       f"achieves F2P=100% on the remaining tests with clean P2P")
    out.append("")

    return "\n".join(out)


# ─── CSV rendering ───────────────────────────────────────────────────────────

CSV_FIELDS = [
    "task_id", "repo", "f2p_total", "p2p_total",
    "n_trials", "n_models", "n_pass1_trials", "n_pass1_models", "pass1_models",
    "best_f2p", "best_f2p_passed", "best_f2p_model",
    "n_universal_fails", "universal_fail_tests",
    "flagged_for_review", "review_md",
]


def render_csv_row(analysis: dict, harbor_meta: dict, review_md_relpath: str) -> dict:
    """Render one CSV row dict."""
    flagged = analysis["n_pass1_trials"] == 0
    return {
        "task_id": analysis["task_id"],
        "repo": harbor_meta.get("repo", ""),
        "f2p_total": analysis["f2p_total"],
        "p2p_total": analysis["p2p_total"],
        "n_trials": analysis["n_trials"],
        "n_models": analysis["n_models"],
        "n_pass1_trials": analysis["n_pass1_trials"],
        "n_pass1_models": analysis["n_pass1_models"],
        "pass1_models": "|".join(analysis["pass1_models"]),
        "best_f2p": f"{analysis['best_f2p']:.4f}",
        "best_f2p_passed": analysis["best_f2p_passed"],
        "best_f2p_model": analysis["best_f2p_model"],
        "n_universal_fails": analysis["n_universal_fails"],
        "universal_fail_tests": "|".join(analysis["universally_failing"]),
        "flagged_for_review": "T" if flagged else "F",
        "review_md": review_md_relpath if flagged else "",
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("roots", nargs="+", type=Path,
                    help="Trial dir parents (recursively walked).")
    ap.add_argument("--alias", action="append", default=[], metavar="RUN_DIR=LABEL",
                    help="Repeatable: relabel a run-group dir.")
    ap.add_argument("--tasks-file", type=Path, default=None,
                    help="Optional file with one task name per line; restrict analysis.")
    ap.add_argument("--harbor-tasks-root", type=Path, default=None,
                    help="Path to harbor task dirs (e.g. craft-bench/harbor-tasks/"
                         "craft-taskgen-v2b). Required for full markdown generation.")
    ap.add_argument("--output-dir", type=Path, default=Path("review_artifacts"),
                    help="Where to write review_index.csv + review_md/.")
    args = ap.parse_args()

    aliases: dict[str, str] = {}
    for a in args.alias:
        if "=" in a:
            k, v = a.split("=", 1)
            aliases[k.strip()] = v.strip()

    task_filter: set[str] | None = None
    if args.tasks_file:
        if not args.tasks_file.is_file():
            print(f"ERROR: --tasks-file {args.tasks_file} not found", file=sys.stderr)
            return 2
        task_filter = {ln.strip() for ln in args.tasks_file.read_text().splitlines() if ln.strip()}
        print(f"Filtering to {len(task_filter)} tasks from {args.tasks_file}", file=sys.stderr)

    if args.harbor_tasks_root and not args.harbor_tasks_root.is_dir():
        print(f"ERROR: --harbor-tasks-root {args.harbor_tasks_root} not a directory", file=sys.stderr)
        return 2

    trial_dirs = find_trial_dirs(args.roots)
    print(f"Found {len(trial_dirs)} trial dirs across {len(args.roots)} root(s)", file=sys.stderr)

    by_task: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for td in trial_dirs:
        rec = load_trial(td, aliases)
        if rec is None:
            skipped += 1
            continue
        if task_filter and rec["task"] not in task_filter:
            continue
        by_task[rec["task"]].append(rec)
    if skipped:
        print(f"Skipped {skipped} unreadable trials", file=sys.stderr)
    print(f"Tasks with at least one valid trial: {len(by_task)}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_dir = args.output_dir / "review_md"
    md_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale markdown files
    for old in md_dir.glob("*.md"):
        old.unlink()

    csv_path = args.output_dir / "review_index.csv"
    n_flagged = 0
    n_md_written = 0
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for task_id in sorted(by_task):
            trials = by_task[task_id]
            harbor_meta: dict = {}
            if args.harbor_tasks_root:
                harbor_meta = parse_task_meta(args.harbor_tasks_root / task_id)
            analysis = analyze_task(task_id, trials, harbor_meta=harbor_meta)
            review_md_relpath = ""
            if analysis["n_pass1_trials"] == 0:
                n_flagged += 1
                md = render_review_markdown(analysis, harbor_meta, task_id)
                md_path = md_dir / f"{task_id}.md"
                md_path.write_text(md)
                review_md_relpath = f"review_md/{task_id}.md"
                n_md_written += 1
            w.writerow(render_csv_row(analysis, harbor_meta, review_md_relpath))

    print(f"\nWrote {csv_path}", file=sys.stderr)
    print(f"Wrote {n_md_written} per-task markdowns to {md_dir} "
          f"(flagged={n_flagged}/{len(by_task)})", file=sys.stderr)

    # Sample prompt — paste-ready for a Claude Code / coding-assistant session
    print("\n" + "=" * 78, file=sys.stderr)
    print("Sample prompt (copy/paste into a coding-assistant session):", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    sample_md = next(iter(sorted(md_dir.glob("*.md"))), None)
    sample_md_path = str(sample_md) if sample_md else f"{md_dir}/<task_id>.md"
    harbor_root_disp = str(args.harbor_tasks_root) if args.harbor_tasks_root else "<path-to-craft-bench>/harbor-tasks/<cohort>"
    print(f"""
You're helping me audit a CRAFT benchmark task. Goal: decide whether the task is
a genuine capability test or whether the instruction/tests are unfair (test scope
mismatch, ambiguous instruction, brittle assertions, P2P regression coupled to
the F2P fix, etc.).

Local paths on this machine:
  - Review markdown for one task: {sample_md_path}
  - Harbor task dir (instruction, reference patch, verifier):
        {harbor_root_disp}/<task_id>/
  - Trial dirs (agent transcripts, per-trial outputs):
        Each trial's full path is listed in the markdown's "Trial outcomes" table —
        use those paths directly; trials are spread across multiple parent dirs.

Steps:
1. Read the markdown end-to-end. Note the full trial_dir paths in the
   "Trial outcomes" table.
2. Open the harbor task dir and inspect:
     - instruction.md (also inlined in the markdown — verify completeness)
     - solution/changes.patch (the reference fix the verifier expects)
     - tests/test.sh and tests/score.py (verifier mechanics)
3. Pick the highest-F2P trial from the markdown's trial table (the row's
   trial_dir column has the full path); open `<trial_dir>/agent/` to see the
   agent's transcript and `<trial_dir>/verifier/` for per-test results.
4. For each universally-failing F2P test: open the test source, decide whether
   it tests behavior the instruction asks for or unrelated functionality.
5. If P2P regressions are listed: open the regressed test, decide whether the
   assertion is overly tight given the instruction's scope.

Final output: a short verdict (keep / revise instruction / revise tests / drop)
with concrete evidence from each step.
""", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
