# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the triage judges over existing harbor trajectories — no Docker, no smoke.

The full pipeline runs smoke (Harbor + Docker) before triage, which is slow and
needs the task's container. When we already have harbor trial trajectories from a
prior evaluation (e.g. the codex + GPT-5.5 run over the v2b tasks), we can feed
those trials straight into the two triage *judges* — the Opus per-test deep dive
(skip/keep on each failing reference test) and the cross-family GPT fairness
review (task-level severity) — and collect their verdicts in bulk.

This mirrors the judge calls in ``steps.py::_run_triage_one`` (the reward<1 path)
exactly, but runs them **side-effect-free**: it does NOT write ``f2p_skip.txt``,
rescore the trial, regenerate the instruction, or re-smoke. It only reads the
existing trial dir + the task dir and records what the judges say.

Primary use: re-running the *bumped* judges (Opus 4.8 deep dive + GPT-5.5 fairness
review) over tasks that previously passed fairness, to measure how the model bump
shifts verdicts (see the GPT-5.5-stricter calibration note).

Usage:
  uv run python scripts/triage-replay.py \
      --job-dir   <harbor job dir with per-task trial subdirs> \
      --task-root <dir containing the built task dirs, e.g. craft-bench/harbor-tasks/craft-taskgen-v2> \
      --tasks-file references/v2b-tasks.txt \
      --out triage-replay.csv

Each trial subdir is matched to its task dir via ``task_name`` in result.json.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import craft_taskgen.config as _cfg
from craft_taskgen import llm_judge
from craft_taskgen.config import PipelineProfile, Stage, TaskState
from craft_taskgen.prompts import DEEP_DIVE_SCHEMA, deep_dive_prompt
from craft_taskgen.steps import (
    _fetch_deep_dive_context,
    _load_actually_failed_tests,
    _load_skipped_tests,
    _run_fairness_review_one,
)


def _read_task_name(trial_dir: Path) -> str | None:
    """Read the harbor task_name from a trial dir's result.json."""
    result = trial_dir / "result.json"
    if not result.is_file():
        return None
    try:
        return json.loads(result.read_text()).get("task_name")
    except (json.JSONDecodeError, OSError):
        return None


def _reward_of(trial_dir: Path) -> float | None:
    """Best-effort reward read for filtering/labelling (handles both reward.json eras)."""
    for fname in ("reward-details.json", "reward.json"):
        p = trial_dir / "verifier" / fname
        if p.is_file():
            try:
                return float(json.loads(p.read_text()).get("reward", 0.0))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                continue
    return None


def _discover(job_dir: Path, task_root: Path, allow: set[str] | None) -> list[dict]:
    """Map each trial subdir to its built task dir. Returns work items."""
    items: list[dict] = []
    unmatched: list[str] = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        name = _read_task_name(trial_dir)
        if not name:
            continue
        if allow is not None and name not in allow:
            continue
        # Built task dir is named exactly task_name somewhere under task_root.
        matches = [d for d in task_root.rglob(name) if (d / "task.toml").is_file()]
        if not matches:
            unmatched.append(name)
            continue
        items.append({"task_name": name, "trial_dir": str(trial_dir), "task_dir": str(matches[0])})
    if unmatched:
        print(f"  [warn] {len(unmatched)} trial(s) had no matching task dir under {task_root}:")
        for n in unmatched[:10]:
            print(f"           {n}")
    return items


async def _replay_one(item: dict, sem: asyncio.Semaphore) -> dict:
    """Run the two triage judges on one trial; record verdicts only."""
    task_dir, trial_dir, name = item["task_dir"], item["trial_dir"], item["task_name"]
    async with sem:
        row: dict = {"task_name": name, "reward": _reward_of(Path(trial_dir))}
        try:
            ctx = await _fetch_deep_dive_context(task_dir, trial_dir)
            dd_prompt = deep_dive_prompt(
                instruction_md=ctx["instruction_md"],
                reward_json=ctx["reward_json"],
                verify_output_tail=ctx["verify_output_tail"],
                postmerge_test_bodies=ctx["postmerge_test_bodies"],
                harbor_lab_errors=ctx["harbor_lab_errors"],
                harbor_lab_edits=ctx["harbor_lab_edits"],
                harbor_lab_tool_sequence=ctx["harbor_lab_tool_sequence"],
                harbor_lab_metrics=ctx["harbor_lab_metrics"],
                f2p_tests=ctx["f2p_tests"],
                p2p_tests=ctx["p2p_tests"],
                f2p_skip=ctx["f2p_skip"],
                p2p_skip=ctx["p2p_skip"],
                triage_history="",
            )
            # Minimal task object — _run_fairness_review_one only uses it for usage logging.
            stub = TaskState(
                task_id=name,
                repo="replay",
                commit_sha="replay",
                description="triage replay",
                base_sha="replay",
                merge_base_sha="replay",
                stage=Stage.OPUS_SMOKE_TESTED,
                task_dir=task_dir,
            )
            dd_judge, reviewer = await asyncio.gather(
                llm_judge.judge(prompt=dd_prompt, schema=DEEP_DIVE_SCHEMA, model=_cfg.LLM_STEP_MODEL),
                _run_fairness_review_one(stub, ctx, "Opus"),
            )
        except Exception as e:  # noqa: BLE001 — record per-task failure, keep the batch going
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  [err] {name}: {row['error']}")
            return row

        # Deep dive — mirror steps.py::_run_triage_one filtering so counts are
        # production-faithful:
        #   (1) drop verdicts on tests ALREADY in f2p_skip.txt/p2p_skip.txt — they
        #       still run and still report FAILED, so the judge re-nominates them
        #       every pass; production drops them. Without this, dd_skip is inflated
        #       by re-nominations of already-excluded tests.
        #   (2) drop verdicts on tests that didn't actually fail in this trial.
        raw_failures = list(dd_judge.result.get("failures", []))
        row["dd_raw_skip"] = sum(1 for f in raw_failures if f.get("classification") == "skip")
        already_skipped = _load_skipped_tests(task_dir)
        failures = [f for f in raw_failures if f.get("test_name", "") not in already_skipped]
        row["dd_already_skipped"] = len(already_skipped)
        actually_failed = _load_actually_failed_tests(trial_dir)
        if actually_failed is not None:
            failures = [f for f in failures if f.get("test_name", "") in actually_failed]
        classes = [f.get("classification", "") for f in failures]
        row["dd_failures"] = len(failures)
        row["dd_skip"] = sum(1 for c in classes if c == "skip")  # NEW skips (not already excluded)
        row["dd_keep"] = sum(1 for c in classes if c == "keep")
        row["dd_assessment"] = (dd_judge.result.get("overall_assessment", "") or "")[:200]

        # Fairness review
        sev = (reviewer or {}).get("severity", "") or ""
        quote = ((reviewer or {}).get("evidence_quote", "") or "").strip()
        test = ((reviewer or {}).get("evidence_test", "") or "").strip()
        row["reviewer_severity"] = sev
        row["reviewer_flag"] = sev not in ("", "none")
        row["reviewer_triggers_regen"] = sev == "major" and bool(quote) and bool(test)
        row["reviewer_evidence_test"] = test
        row["reviewer_evidence_quote"] = quote[:300]
        row["reviewer_reason"] = ((reviewer or {}).get("reason", "") or "")[:300]
        print(
            f"  [ok] {name}: reward={row['reward']} | DD {row['dd_keep']}keep/{row['dd_skip']}skip "
            f"of {row['dd_failures']} | fairness={sev or 'none'}"
            + (" [MAJOR->regen]" if row["reviewer_triggers_regen"] else "")
        )
        return row


async def _run(items: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*[_replay_one(it, sem) for it in items])


def _summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    print("\n" + "=" * 60)
    print("TRIAGE REPLAY SUMMARY")
    print("=" * 60)
    print(f"  tasks: {len(rows)}  ({len(ok)} judged, {len(errs)} errored)")
    sev_dist: dict[str, int] = {}
    for r in ok:
        sev_dist[r["reviewer_severity"] or "none"] = sev_dist.get(r["reviewer_severity"] or "none", 0) + 1
    print(f"  fairness severity: {dict(sorted(sev_dist.items()))}")
    regen = [r for r in ok if r.get("reviewer_triggers_regen")]
    print(f"  would trigger Build regen (major + quote + test): {len(regen)}")
    for r in regen:
        print(f"      {r['task_name']}: {r['reviewer_evidence_test']}")
    any_skip = [r for r in ok if r.get("dd_skip", 0) > 0]
    print(f"  tasks with >=1 deep-dive 'skip' verdict (unfair test): {len(any_skip)}")
    for r in any_skip:
        print(f"      {r['task_name']}: {r['dd_skip']} skip / {r['dd_keep']} keep")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-dir", required=True, help="Harbor job dir with per-task trial subdirs")
    p.add_argument("--task-root", required=True, help="Dir containing built task dirs (searched recursively)")
    p.add_argument("--tasks-file", help="Optional newline-delimited task-name allowlist (e.g. v2b-tasks.txt)")
    p.add_argument("--profile", help="TOML profile (else config.py defaults — Opus 4.8 + GPT-5.5)")
    p.add_argument("--alignment-model", help="Override LLM_ALIGNMENT_MODEL (fairness judge)")
    p.add_argument("--deep-dive-model", help="Override LLM_STEP_MODEL (deep-dive judge)")
    p.add_argument("--concurrency", type=int, default=6, help="Parallel judge pairs (default 6)")
    p.add_argument("--limit", type=int, help="Process only the first N matched trials")
    p.add_argument("--out", default="triage-replay.csv", help="CSV output path")
    args = p.parse_args()

    _cfg._load_env()
    if args.profile:
        PipelineProfile.from_toml(args.profile).apply()
    if args.alignment_model:
        _cfg.LLM_ALIGNMENT_MODEL = args.alignment_model
    if args.deep_dive_model:
        _cfg.LLM_STEP_MODEL = args.deep_dive_model

    job_dir = Path(args.job_dir)
    task_root = Path(args.task_root)
    if not job_dir.is_dir():
        sys.exit(f"ERROR: --job-dir not a directory: {job_dir}")
    if not task_root.is_dir():
        sys.exit(f"ERROR: --task-root not a directory: {task_root}")

    allow: set[str] | None = None
    if args.tasks_file:
        allow = {ln.strip() for ln in Path(args.tasks_file).read_text().splitlines() if ln.strip()}

    print(f"deep-dive model: {_cfg.LLM_STEP_MODEL}")
    print(f"fairness model:  {_cfg.LLM_ALIGNMENT_MODEL}")
    items = _discover(job_dir, task_root, allow)
    if args.limit:
        items = items[: args.limit]
    print(f"matched {len(items)} trials to task dirs; running judges (concurrency={args.concurrency})...\n")
    if not items:
        sys.exit("ERROR: no trials matched — check --job-dir / --task-root / --tasks-file")

    rows = asyncio.run(_run(items, args.concurrency))

    fields = [
        "task_name",
        "reward",
        "dd_failures",
        "dd_keep",
        "dd_skip",
        "dd_raw_skip",
        "dd_already_skipped",
        "reviewer_severity",
        "reviewer_flag",
        "reviewer_triggers_regen",
        "reviewer_evidence_test",
        "reviewer_evidence_quote",
        "reviewer_reason",
        "dd_assessment",
        "error",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _summarize(rows)
    print(f"\nCSV written to {args.out}")


if __name__ == "__main__":
    main()
