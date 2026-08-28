#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRAFT task generation pipeline: evaluate candidates -> build tasks -> validate -> triage.

Uses `claude -p --permission-mode auto` for LLM steps, subprocess for Docker/Harbor,
and a JSON manifest to track each task's state across the pipeline.

Usage:
    craft-taskgen --candidates candidates/*.json --concurrency 4
    craft-taskgen --resume runs/2026-04-09/state.json --from-step smoke
    craft-taskgen-dashboard runs/2026-04-09/state.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import socket
import sys
from datetime import datetime

import craft_taskgen.config as _cfg
from craft_taskgen.config import (
    STEP_ORDER,
    PipelineProfile,
    PipelineState,
    TaskState,
    _load_env,
)
from craft_taskgen.steps import (
    run_task_pipeline,
    select_candidates,
    step_evaluate,
    step_report,
)


def _log_input_stats(candidate_files: list[str]) -> None:
    """Print a per-file summary of the input. Useful when splitting work across machines."""
    if not candidate_files:
        print("Input: no candidate files (using --resume, likely)")
        print()
        return

    print(f"Input: {len(candidate_files)} candidate file(s)")
    total_cands = 0
    total_has_test = 0
    per_file: list[tuple[str, int, int, str | None]] = []
    for fpath in candidate_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  {fpath}: ERROR reading — {e}")
            continue
        cands = data.get("candidates", [])
        has_test = sum(1 for c in cands if c.get("has_test_patch"))
        after = data.get("after")
        per_file.append((fpath, len(cands), has_test, after))
        total_cands += len(cands)
        total_has_test += has_test

    # Sort by candidate count so the heavy-hitters show first
    per_file.sort(key=lambda x: x[1], reverse=True)
    for fpath, n_cands, n_has_test, after in per_file[:30]:
        after_str = f" after={after}" if after else ""
        print(f"  {fpath}: {n_cands} cands ({n_has_test} w/tests){after_str}")
    if len(per_file) > 30:
        print(f"  ...and {len(per_file) - 30} more files")
    print(f"Total: {total_cands} raw candidates, {total_has_test} with test_patch")
    print()


async def async_main():
    # Unbuffered stdout so background runs show progress in real time
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dimension",
        choices=["tools", "search"],
        default="tools",
        help="Pipeline dimension: tools (default) or search (derived from tools-track tasks)",
    )
    parser.add_argument("--candidates", nargs="*", help="Mining candidate JSON files (glob pattern)")
    parser.add_argument(
        "--top-per-repo",
        type=int,
        default=5,
        help="Keep top N candidates per repo (0 = no cap)",
    )
    parser.add_argument(
        "--skip-per-repo",
        type=int,
        default=0,
        help="Skip first N candidates per repo (for subsequent runs)",
    )
    parser.add_argument(
        "--max-evaluate",
        type=int,
        default=30,
        help="Maximum total candidates to select (0 = no cap)",
    )
    parser.add_argument("--resume", type=str, help="Resume from existing pipeline state JSON")
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        help="Start from this step (tools: select/evaluate/build/..., search: extract/synthesize/...)",
    )
    parser.add_argument(
        "--stop-after-step",
        type=str,
        default=None,
        help="Stop after this step (tools only; supported: select, evaluate).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run fully unattended (no human checkpoint after evaluate)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max parallel invocations per step (overrides profile, default: 4)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="TOML profile for model names, thresholds, tuning parameters",
    )
    # Search-specific args
    parser.add_argument("--tasks-dir", type=str, help="Input tasks dir for search dimension")
    parser.add_argument("--repos-dir", type=str, default="repos", help="Local repos dir (search dimension)")
    parser.add_argument("--output-dir", type=str, help="Output dir for search gold data")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of input tasks to process (0 = all, useful for testing)",
    )
    args = parser.parse_args()

    # Load profile (if provided) and apply to module constants
    if args.profile:
        profile = PipelineProfile.from_toml(args.profile)
        print(f"Loaded profile: {args.profile}")
    else:
        profile = PipelineProfile()
    profile.apply()

    # Route to search pipeline if --dimension search
    if args.dimension == "search":
        from craft_taskgen.search.config import SEARCH_STEPS
        from craft_taskgen.search.pipeline import run_search_pipeline

        if args.stop_after_step:
            parser.error("--stop-after-step is only supported for tools dimension")
        if not args.tasks_dir and not args.resume:
            parser.error("--tasks-dir is required for search dimension (or --resume)")
        if not args.output_dir and not args.resume:
            parser.error("--output-dir is required for search dimension (or --resume)")
        concurrency = args.concurrency if args.concurrency is not None else profile.default_concurrency
        from_step = args.from_step or "extract"
        if from_step not in SEARCH_STEPS:
            valid = ", ".join(SEARCH_STEPS)
            parser.error(f"--from-step '{from_step}' invalid for search. Options: {valid}")
        run_search_pipeline(
            tasks_dir=args.tasks_dir or "",
            repos_dir=args.repos_dir,
            output_dir=args.output_dir or "",
            concurrency=concurrency,
            limit=args.limit,
            from_step=from_step,
            resume_path=args.resume,
            profile_data=profile.to_dict(),
        )
        return

    # --- Tools pipeline below ---

    from_step = args.from_step or "select"
    if from_step not in STEP_ORDER:
        parser.error(f"--from-step '{from_step}' invalid for tools. Options: {', '.join(STEP_ORDER)}")
    if args.stop_after_step:
        if args.stop_after_step not in {"select", "evaluate"}:
            parser.error("--stop-after-step invalid for tools. Options: select, evaluate")
        if STEP_ORDER.index(args.stop_after_step) < STEP_ORDER.index(from_step):
            parser.error("--stop-after-step must not be earlier than --from-step")

    # Load or create state
    if args.resume and os.path.exists(args.resume):
        state = PipelineState.load(args.resume)
        # Clear stale in_progress flags from crashed runs
        for t in state.tasks.values():
            t.in_progress_step = ""
        run_dir = state.run_dir
        state_file = args.resume  # use the resumed file path
        print(f"Resumed pipeline with {len(state.tasks)} tasks")
        print(f"  Run directory: {run_dir}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_dir = os.path.join(profile.task_suite_dir, "runs", ts)
        os.makedirs(run_dir, exist_ok=True)
        state = PipelineState(
            created=datetime.now().isoformat(),
            run_dir=run_dir,
            profile_data=profile.to_dict(),
        )
        state_file = os.path.join(run_dir, "state.json")

    # Expand globs for candidate files
    candidate_files = []
    if args.candidates:
        for pattern in args.candidates:
            candidate_files.extend(glob.glob(pattern))
    candidate_files = sorted(set(candidate_files))

    concurrency = args.concurrency if args.concurrency is not None else profile.default_concurrency

    _load_env()

    # Record host/input metadata for provenance (useful when splitting across machines)
    if not args.resume:
        state.run_info = {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER", ""),
            "cli_args": sys.argv[1:],
            "candidate_files": candidate_files,
            "candidate_patterns": list(args.candidates or []),
            "concurrency": concurrency,
            "from_step": from_step,
            "stop_after_step": args.stop_after_step,
        }
        state.save(state_file)

    print("CRAFT Task Pipeline")
    print(f"  Host:          {socket.gethostname()}")
    print(f"  User:          {os.environ.get('USER', '?')}")
    print(f"  Run directory: {run_dir}")
    print(f"  State file:    {state_file}")
    print(f"  Starting from: {from_step}")
    print(f"  Concurrency:   {concurrency}")
    print()
    _log_input_stats(candidate_files)

    start_idx = STEP_ORDER.index(from_step)

    # Step 1: Select
    if start_idx <= 0 and candidate_files:
        candidates = select_candidates(
            candidate_files, args.top_per_repo, args.skip_per_repo, args.max_evaluate
        )
        print(f"Selected {len(candidates)} candidates from {len(candidate_files)} files")
        for c in candidates:
            tid = f"{c['repo']}-{c['sha'][:8]}"
            if tid not in state.tasks:
                state.tasks[tid] = TaskState(
                    task_id=tid,
                    repo=c["repo"],
                    commit_sha=c["sha"],
                    base_sha=c["base_sha"],
                    merge_base_sha=c["merge_base_sha"],
                    description=c["subject"],
                    candidate_data=c.get("_raw", {}),
                )
        state.save(state_file)
        if args.stop_after_step == "select":
            print()
            print("=" * 60)
            print("STOPPED AFTER SELECT")
            print(f"State saved to {state_file}")
            print(f"To continue: uv run python {sys.argv[0]} --resume {state_file} --from-step evaluate")
            print("=" * 60)
            return

    # Step 2: Evaluate
    if start_idx <= 1:
        await step_evaluate(state, state_file, concurrency=concurrency)

        if not args.no_checkpoint:
            step_report(state)
            print()
            print("=" * 60)
            print("CHECKPOINT: Review PROMISING candidates above.")
            print(f"State saved to {state_file}")
            print(
                f"To continue: uv run python {sys.argv[0]} "
                f"--resume {state_file} --from-step build --no-checkpoint"
            )
            print("=" * 60)
            return

    # Steps 3+: Each task flows through the full pipeline independently (build → ... → accept).
    from craft_taskgen.config import MAX_PROMISING_PER_REPO, Stage

    # Apply per-repo build cap before dispatching
    repo_counts: dict[str, int] = {}
    for t in state.tasks.values():
        if t.stage == Stage.PROMISING:
            count = repo_counts.get(t.repo, 0)
            if count >= MAX_PROMISING_PER_REPO:
                t.stage = Stage.REJECTED
                t.eval_reason = f"Skipped: repo {t.repo} already has {MAX_PROMISING_PER_REPO} candidates"
                print(f"  Skipping {t.task_id} (repo cap: {t.repo} has {MAX_PROMISING_PER_REPO} already)")
            else:
                repo_counts[t.repo] = count + 1

    # Candidate-level semaphore bounds the per-task build+align fanout
    # (N candidates per task). Sized at 2× LLM_CONCURRENCY so candidates
    # don't head-of-line block downstream tasks. Each candidate still
    # acquires sems["llm"] internally for its judge calls — same gateway
    # flow control as today.
    candidate_sem_size = max(_cfg.LLM_CONCURRENCY, _cfg.LLM_CONCURRENCY * 2)
    if candidate_sem_size < _cfg.BUILD_N_CANDIDATES:
        # Defensive: ensure at least N slots exist or fanout deadlocks.
        candidate_sem_size = _cfg.BUILD_N_CANDIDATES
    sems = {
        "llm": asyncio.Semaphore(_cfg.LLM_CONCURRENCY),
        "docker": asyncio.Semaphore(_cfg.DOCKER_CONCURRENCY),
        "smoke": asyncio.Semaphore(_cfg.SMOKE_CONCURRENCY),
        "candidate": asyncio.Semaphore(candidate_sem_size),
    }
    active_stages = {
        Stage.EVALUATED,  # build failed and fell back; retry via run_task_pipeline
        Stage.PROMISING,  # build is now per-task
        Stage.BUILT,
        Stage.ALIGNMENT_CHECKED,
        Stage.TESTS_DISCOVERED,
        Stage.DOCKERFILE_BUILT,
        Stage.F2P_P2P_CLASSIFIED,
        Stage.ORACLE_CHECKED,
        Stage.OPUS_SMOKE_TESTED,
        Stage.OPUS_TRIAGED,
    }
    active_tasks = [t for t in state.tasks.values() if t.stage in active_stages]

    if active_tasks:
        print(
            f"\nProcessing {len(active_tasks)} tasks through pipeline "
            f"(llm={_cfg.LLM_CONCURRENCY}, docker={_cfg.DOCKER_CONCURRENCY}, "
            f"smoke={_cfg.SMOKE_CONCURRENCY})..."
        )
        # return_exceptions=True prevents one task's crash from killing the whole run.
        # Exceptions come back as values in the result list; we log them and mark the
        # task NEEDS_FIX so the rest of the pipeline can continue.
        import traceback

        results = await asyncio.gather(
            *[run_task_pipeline(t, state, state_file, sems) for t in active_tasks],
            return_exceptions=True,
        )
        for task, result in zip(active_tasks, results):
            if isinstance(result, BaseException):
                tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
                print(f"  !! CRASH in task {task.task_id}: {type(result).__name__}: {result}")
                print(f"     (traceback truncated; first line shown) {tb.splitlines()[0] if tb else ''}")
                task.stage = Stage.NEEDS_FIX
                task.needs_human_review = True
                task.human_review_reason = f"Crashed: {type(result).__name__}: {str(result)[:200]}"
                state.save(state_file)

    # Report
    step_report(state)
    state.save(state_file)
    print(f"\nPipeline complete. State saved to {state_file}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
