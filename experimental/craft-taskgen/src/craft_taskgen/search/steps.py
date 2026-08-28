# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search pipeline step implementations.

Each step is a standalone function that takes the pipeline state and modifies it.
Steps are designed to be resumable — they check what's already done before acting.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

from craft_taskgen.search.config import SearchPipelineState


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


HEARTBEAT_INTERVAL = 60


def run_cmd(
    cmd: list[str], desc: str, *, timeout: int = 43200, cwd: str | None = None
) -> subprocess.CompletedProcess:
    """Run a command with timestamps and heartbeat logging."""
    print(f"\n{'=' * 60}")
    print(f"[{_ts()}] STEP: {desc}")
    print(f"[{_ts()}] CMD: {' '.join(cmd)}")
    if cwd:
        print(f"[{_ts()}] CWD: {cwd}")
    print(f"{'=' * 60}\n", flush=True)

    start = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            print(f"[{_ts()}] ... {desc} still running ({time.time() - start:.0f}s)", flush=True)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        result = subprocess.run(cmd, timeout=timeout, cwd=cwd)
    finally:
        stop.set()

    elapsed = time.time() - start
    print(f"\n[{_ts()}] [{desc}] finished in {elapsed:.0f}s (exit={result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: non-zero exit code {result.returncode}", file=sys.stderr)
    return result


def find_job_dir(jobs_dir: str, after_ts: float) -> str | None:
    """Find the job directory created after the given timestamp."""
    if not os.path.isdir(jobs_dir):
        return None
    best = None
    best_mtime = 0.0
    for d in os.listdir(jobs_dir):
        full = os.path.join(jobs_dir, d)
        if not os.path.isdir(full):
            continue
        mtime = os.path.getmtime(full)
        if mtime > after_ts and mtime > best_mtime:
            best = full
            best_mtime = mtime
    return best


def _load_rewards(job_dir: str, agent_key: str, state: SearchPipelineState) -> None:
    """Load reward.json files from a job dir into task_statuses."""
    from craft_taskgen.search.config import SearchTaskStatus

    for trial in os.listdir(job_dir):
        rj = os.path.join(job_dir, trial, "verifier", "reward.json")
        if not os.path.exists(rj):
            continue
        # harbor-agent-patches.diff replaces `__<uuid>` with `-<uuid>` in trial
        # names (Docker Compose v5 rejects `__` in image tags), so strip the
        # 7-char ShortUUID suffix to recover the task id.
        tid = trial.rsplit("-", 1)[0]
        with open(rj) as f:
            r = json.load(f)
        ts = state.task_statuses.setdefault(tid, SearchTaskStatus())
        setattr(ts, f"{agent_key}_reward", r.get("reward", 0))
        setattr(ts, f"{agent_key}_nav", r.get("navigation_score", 0))
        setattr(ts, f"{agent_key}_assert", r.get("assertion_coverage", 0))
        setattr(ts, f"{agent_key}_file_recall", r.get("file_recall", 0))
        setattr(ts, f"{agent_key}_func_recall", r.get("function_recall", 0))


# Single lock shared by all in-flight smoke agents so reward loading into
# state.task_statuses never races when step_smoke_all runs agents in parallel.
_SMOKE_STATE_LOCK = threading.Lock()


def _run_harbor_agent(
    state: SearchPipelineState,
    *,
    agent: str,
    state_key: str,
    label: str,
    model: str | None = None,
) -> None:
    """Run a Harbor agent on the search tasks and load results into state."""
    harbor_dir = state.harbor_dir
    agent_jobs_dir = os.path.join("jobs", state_key)
    os.makedirs(agent_jobs_dir, exist_ok=True)

    cmd = ["uv", "run", "harbor", "run", "--agent", agent]
    if model:
        cmd += ["--model", model]
    cmd += [
        "--path",
        f"{harbor_dir}/",
        "--n-concurrent",
        str(state.concurrency),
        "--env",
        "docker",
        "-o",
        f"{agent_jobs_dir}/",
    ]

    before_ts = time.time()
    run_cmd(cmd, label)

    job_dir = find_job_dir(agent_jobs_dir, before_ts)
    if job_dir:
        with _SMOKE_STATE_LOCK:
            state.job_dirs[state_key] = job_dir
            _load_rewards(job_dir, state_key, state)
            vals = [
                getattr(s, f"{state_key}_reward")
                for s in state.task_statuses.values()
                if getattr(s, f"{state_key}_reward", None) is not None
            ]
            mean = sum(vals) / len(vals) if vals else 0.0
            print(f"\n  {label} mean reward: {mean:.3f} (n={len(vals)})")
    else:
        print(f"  WARNING: Could not find job directory for {label}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step_extract(state: SearchPipelineState) -> None:
    """Parse input tasks and mine repo maps at pre-change commits.

    Uses native craft_taskgen.mining (no external craft-bench dependency).
    """
    from craft_taskgen.search.extract import run_extract

    output_dir = state.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tasks_input_dir = os.path.abspath(state.tasks_dir)
    repos_dir = os.path.abspath(state.repos_dir)
    abs_output = os.path.abspath(output_dir)

    # Apply --limit: symlink a subset of T2 tasks into a temp dir
    effective_tasks_dir = tasks_input_dir
    if state.limit > 0:
        task_dirs = sorted(
            d for d in os.listdir(tasks_input_dir) if os.path.isdir(os.path.join(tasks_input_dir, d))
        )
        task_dirs = task_dirs[: state.limit]
        limited_dir = os.path.join(abs_output, "_limited_tasks")
        if os.path.isdir(limited_dir):
            shutil.rmtree(limited_dir)
        os.makedirs(limited_dir)
        for d in task_dirs:
            src = os.path.join(tasks_input_dir, d)
            dst = os.path.join(limited_dir, d)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        effective_tasks_dir = limited_dir
        print(f"  --limit {state.limit}: using {len(task_dirs)} of {len(os.listdir(tasks_input_dir))} tasks")

    print(f"\n{'=' * 60}")
    print(f"[{_ts()}] STEP: extract: parse tasks + mine repo maps")
    print(f"{'=' * 60}\n", flush=True)

    run_extract(
        tasks_dir=effective_tasks_dir,
        output_dir=abs_output,
        repos_dir=repos_dir,
    )

    # Verify output
    combined = os.path.join(abs_output, "_all_contexts.json")
    if not os.path.exists(combined):
        raise RuntimeError(f"Extract step did not produce {combined}")
    with open(combined) as f:
        contexts = json.load(f)
    print(f"  Extracted {len(contexts)} task contexts -> {combined}")


def step_synthesize(state: SearchPipelineState) -> None:
    """3-model LLM synthesis with cross-judging (3 seed strategies).

    Calls the synthesis logic directly (no subprocess needed -- uses litellm).
    """
    from craft_taskgen.search.config import SYNTHESIS_CONCURRENCY
    from craft_taskgen.search.synthesize import run_synthesis

    run_synthesis(
        contexts_dir=state.output_dir,
        output_dir=state.output_dir,
        approaches=["A", "B", "C"],
        concurrency=SYNTHESIS_CONCURRENCY,
    )


def step_validate(state: SearchPipelineState) -> None:
    """Validate gold answers against repo AST, expand alt_functions.

    Uses native craft_taskgen.mining (no external craft-bench dependency).
    """
    from craft_taskgen.search.validate import run_validate

    abs_output = os.path.abspath(state.output_dir)
    repos_dir = os.path.abspath(state.repos_dir)

    print(f"\n{'=' * 60}")
    print(f"[{_ts()}] STEP: validate: check gold against repo AST")
    print(f"{'=' * 60}\n", flush=True)

    run_validate(
        contexts_dir=abs_output,
        repos_dir=repos_dir,
        fix_alt_funcs=True,
    )


def step_dedup(state: SearchPipelineState) -> None:
    """Deduplicate tasks by embedding cosine similarity across all approaches."""
    from craft_taskgen.search.config import DEDUP_EMBEDDING_MODEL, DEDUP_THRESHOLD
    from craft_taskgen.search.dedup import run_dedup

    run_dedup(
        state.output_dir,
        threshold=DEDUP_THRESHOLD,
        embedding_model=DEDUP_EMBEDDING_MODEL,
    )


def step_harbor(state: SearchPipelineState) -> None:
    """Convert search tasks to Harbor task directories."""
    from craft_taskgen.search.harbor import run_harbor_convert

    harbor_dir = os.path.join(state.output_dir, "harbor-tasks")
    run_harbor_convert(
        output_dir=state.output_dir,
        tasks_dir=state.tasks_dir,
        harbor_dir=harbor_dir,
    )
    state.harbor_dir = harbor_dir


def step_smoke_opus(state: SearchPipelineState) -> None:
    """Run Claude Code Opus on all search tasks."""
    from craft_taskgen.config import OPUS_MODEL

    _run_harbor_agent(state, agent="claude-code", model=OPUS_MODEL, state_key="opus", label="smoke-opus")


def step_smoke_codex(state: SearchPipelineState) -> None:
    """Run Codex GPT-5.3 on all search tasks."""
    from craft_taskgen.search.config import CODEX_MODEL

    _run_harbor_agent(state, agent="codex", model=CODEX_MODEL, state_key="codex", label="smoke-codex")


def step_smoke_haiku(state: SearchPipelineState) -> None:
    """Run Haiku 4.5 weak baseline (measurement only)."""
    from craft_taskgen.config import HAIKU_MODEL

    _run_harbor_agent(state, agent="claude-code", model=HAIKU_MODEL, state_key="haiku", label="smoke-haiku")


def step_smoke_all(state: SearchPipelineState) -> None:
    """Run Opus, Codex, and Haiku smoke tests concurrently.

    Each harbor subprocess is independent (separate jobs/<agent>/ output
    dirs) so parallelism is bounded only by Docker resources on the host.
    Reward collection into shared state is serialized via _SMOKE_STATE_LOCK.
    """
    from craft_taskgen.config import HAIKU_MODEL, OPUS_MODEL
    from craft_taskgen.search.config import CODEX_MODEL

    configs = [
        {"agent": "claude-code", "model": OPUS_MODEL, "state_key": "opus", "label": "smoke-opus"},
        {"agent": "codex", "model": CODEX_MODEL, "state_key": "codex", "label": "smoke-codex"},
        {"agent": "claude-code", "model": HAIKU_MODEL, "state_key": "haiku", "label": "smoke-haiku"},
    ]

    threads: list[threading.Thread] = []
    for cfg in configs:
        t = threading.Thread(target=_run_harbor_agent, kwargs={"state": state, **cfg}, daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def step_gold_review(state: SearchPipelineState) -> None:
    """Run automated gold review via Harbor + /gold-review skill."""
    from craft_taskgen.search.gold_review import run_gold_review

    run_gold_review(state)


def step_filter(state: SearchPipelineState) -> None:
    """Auto-accept/reject based on 3-model agent scores + gold review."""
    from craft_taskgen.search.config import (
        HAIKU_INVERSION_FLOOR,
        HAIKU_INVERSION_MARGIN,
        REJECT_THRESHOLD,
        SearchTaskStatus,
        load_approach_tasks,
    )

    all_tasks = load_approach_tasks(state.output_dir)

    accepted, rejected, flagged = 0, 0, 0

    for task in all_tasks:
        tid = task["id"]
        ts = state.task_statuses.setdefault(tid, SearchTaskStatus())

        status = "accepted"
        flags: list[str] = []

        # Both strong models score low → reject
        if ts.opus_reward is not None and ts.codex_reward is not None:
            if ts.opus_reward <= REJECT_THRESHOLD and ts.codex_reward <= REJECT_THRESHOLD:
                status = "rejected"
                flags.append(f"both_low: opus={ts.opus_reward:.2f}, codex={ts.codex_reward:.2f}")

        # No gold functions AND no alt functions → reject. Gold-review's DEMOTE
        # moves functions to alt_functions rather than discarding them, so the
        # task is still scorable as long as either list is non-empty.
        gold = task.get("gold_answer", {})
        if not gold.get("functions") and not gold.get("alt_functions"):
            status = "rejected"
            flags.append("no_gold_functions_or_alt")

        # Haiku meaningfully outperforms Opus → gold is likely broken.
        # Require both a non-noise margin AND haiku actually performing well, so
        # single-tick variance or both-models-failed near-zeros don't reject.
        if (
            ts.haiku_reward is not None
            and ts.opus_reward is not None
            and ts.haiku_reward >= HAIKU_INVERSION_FLOOR
            and ts.haiku_reward > ts.opus_reward + HAIKU_INVERSION_MARGIN
        ):
            status = "rejected"
            flags.append(
                f"haiku_inversion: haiku={ts.haiku_reward:.2f} > "
                f"opus={ts.opus_reward:.2f}+{HAIKU_INVERSION_MARGIN}"
            )

        # All 3 models ace it → reject (trivial)
        if (
            ts.opus_reward is not None
            and ts.codex_reward is not None
            and ts.haiku_reward is not None
            and ts.opus_reward >= 0.9
            and ts.codex_reward >= 0.9
            and ts.haiku_reward >= 0.9
        ):
            status = "rejected"
            flags.append("flat_easy")

        # Gold review flagged → reject
        if ts.review_recommendation == "REJECT":
            status = "rejected"
            flags.append("gold_review_reject")
        elif ts.review_recommendation == "FLAG" and status == "accepted":
            status = "flagged"
            flags.append("gold_review_flag")

        ts.status = status
        ts.flags = flags

        if status == "accepted":
            accepted += 1
        elif status == "rejected":
            rejected += 1
        else:
            flagged += 1

    print(f"\n  Filter: {accepted} accepted, {rejected} rejected, {flagged} flagged")


def step_report(state: SearchPipelineState) -> None:
    """Print summary report."""
    print("\n  === SEARCH PIPELINE SUMMARY ===\n")
    ts_map = state.task_statuses

    for status in ["accepted", "rejected", "flagged"]:
        tasks = [tid for tid, s in ts_map.items() if s.status == status]
        if tasks:
            print(f"  {status.upper()}: {len(tasks)}")
            for tid in sorted(tasks)[:5]:
                s = ts_map[tid]
                o = f"{s.opus_reward:.2f}" if s.opus_reward is not None else "?"
                c = f"{s.codex_reward:.2f}" if s.codex_reward is not None else "?"
                h = f"{s.haiku_reward:.2f}" if s.haiku_reward is not None else "?"
                flags_str = ", ".join(s.flags)
                print(f"    {tid}: opus={o} codex={c} haiku={h} {flags_str}")
            if len(tasks) > 5:
                print(f"    ... and {len(tasks) - 5} more")

    for agent in ["opus", "codex", "haiku"]:
        vals = [
            getattr(s, f"{agent}_reward")
            for s in ts_map.values()
            if getattr(s, f"{agent}_reward", None) is not None
        ]
        if vals:
            print(f"\n  {agent.capitalize():6s}: mean={sum(vals) / len(vals):.3f} (n={len(vals)})")


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------

SEARCH_STEP_FUNCS = {
    "extract": step_extract,
    "synthesize": step_synthesize,
    "validate": step_validate,
    "dedup": step_dedup,
    "harbor": step_harbor,
    "smoke-all": step_smoke_all,
    # Legacy per-agent smoke steps retained so resumes from older state files
    # still dispatch. New runs use the combined smoke-all step.
    "smoke-opus": step_smoke_opus,
    "smoke-codex": step_smoke_codex,
    "smoke-haiku": step_smoke_haiku,
    "gold-review": step_gold_review,
    "filter": step_filter,
    "report": step_report,
}
