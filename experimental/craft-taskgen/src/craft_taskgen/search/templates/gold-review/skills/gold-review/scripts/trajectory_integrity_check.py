# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory integrity checks for CRAFT Search gold data review.

Examines agent trajectories for cheating, contamination, and validity issues.
Based on the agentic-benchmark-eval checklists (verifier_audit.md, task_design.md).

Four checks:
  1. Gold Contamination — agent reads gold_answer.json or solve.sh
  2. Memorization — high score with minimal/zero tool calls
  3. Exploration Coverage — agent claims files it never read
  4. Answer Leakage — instruction text contains gold file/function names

Usage::

    uv run python .claude/skills/gold-review/scripts/trajectory_integrity_check.py <task-id>
    uv run python .claude/skills/gold-review/scripts/trajectory_integrity_check.py <task-id> \\
        --job-dirs jobs/2026-04-13__07-36-40 jobs/2026-04-13__07-37-50
    uv run python .claude/skills/gold-review/scripts/trajectory_integrity_check.py --all \\
        --job-dirs jobs/opus-run jobs/codex-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Paths that agents must never read (verifier/oracle internals)
CONTAMINATION_PATHS = [
    "/tests/gold_answer.json",
    "/solution/solve.sh",
    "gold_answer.json",
    "solve.sh",
    "/tests/test_runner.py",
    "/tests/test.sh",
]

# Minimum tool calls for a non-memorized answer
MIN_TOOL_CALLS_FOR_HIGH_REWARD = 3
HIGH_REWARD_THRESHOLD = 0.5

# Maximum fraction of claimed files that can be unverified
MAX_UNVERIFIED_FRACTION = 0.5


def load_review_data() -> dict[str, dict]:
    """Load review_data.json and return task_id → task mapping."""
    review_path = os.path.join("tools", "search", "review_data.json")
    if not os.path.exists(review_path):
        return {}
    with open(review_path) as f:
        data = json.load(f)
    return {t["task_id"]: t for t in data.get("tasks", [])}


def load_task_from_gold(task_id: str, gold_dirs: list[str] | None = None) -> dict | None:
    """Load task from gold directories (search-from-t2 tasks)."""
    search_dirs = gold_dirs or ["gold/craft-search-from-t2"]
    for gold_dir in search_dirs:
        for approach in ["a", "b", "c"]:
            path = os.path.join(gold_dir, f"approach-{approach}", "search_tasks.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                tasks = json.load(f)
            for t in tasks:
                if t["id"] == task_id:
                    return t
    return None


def find_trials(task_id: str, job_dirs: list[str]) -> list[dict]:
    """Find all trial directories for a task across job dirs.

    Returns list of {agent, trial_dir, trajectory_path, reward_path}.
    """
    trials = []
    for job_dir in job_dirs:
        if not os.path.isdir(job_dir):
            continue
        # Infer agent from job config
        config_path = os.path.join(job_dir, "config.json")
        agent_name = "unknown"
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            agent_name = config.get("agent", {}).get("name", "unknown")
            model = config.get("agent", {}).get("model_name", "")
            if "opus" in model.lower():
                agent_name = "opus"
            elif "codex" in model.lower() or "gpt-5" in model.lower():
                agent_name = "codex"
            elif "haiku" in model.lower():
                agent_name = "haiku"

        for trial_name in os.listdir(job_dir):
            if not trial_name.startswith(task_id):
                continue
            trial_dir = os.path.join(job_dir, trial_name)
            traj = os.path.join(trial_dir, "agent", "trajectory.json")
            reward = os.path.join(trial_dir, "verifier", "reward.json")
            if os.path.exists(traj) and os.path.exists(reward):
                trials.append(
                    {
                        "agent": agent_name,
                        "trial_dir": trial_dir,
                        "trajectory_path": traj,
                        "reward_path": reward,
                    }
                )
    return trials


def extract_tool_calls(trajectory: dict) -> list[dict]:
    """Extract all tool calls from a trajectory."""
    calls = []
    for step in trajectory.get("steps", []):
        for tc in step.get("tool_calls", []):
            calls.append(
                {
                    "function": tc.get("function_name", tc.get("name", "")),
                    "arguments": tc.get("arguments", tc.get("input", {})),
                }
            )
    return calls


def extract_files_read(tool_calls: list[dict]) -> set[str]:
    """Extract file paths from Read/Glob tool calls."""
    files = set()
    for tc in tool_calls:
        fn = tc["function"]
        args = tc["arguments"]
        if not isinstance(args, dict):
            continue
        if fn in ("Read", "read_file", "View"):
            fp = args.get("file_path", args.get("path", ""))
            if fp:
                files.add(fp)
        elif fn in ("Glob", "glob"):
            # Glob returns matched files in the result, not args
            pass
    return files


_STRIP_PREFIXES = ("/repo/", "repo/", "/code/", "code/", "./")


def normalize_path(p: str) -> str:
    """Strip repo-root prefix for comparison."""
    p = p.strip()
    for prefix in _STRIP_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix) :]
    return p


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_gold_contamination(tool_calls: list[dict]) -> list[str]:
    """Check if agent read gold/solution files."""
    findings = []
    for tc in tool_calls:
        args = tc["arguments"]
        if not isinstance(args, dict):
            continue
        for arg_val in args.values():
            if not isinstance(arg_val, str):
                continue
            for bad_path in CONTAMINATION_PATHS:
                if bad_path in arg_val:
                    findings.append(f"read '{bad_path}' via {tc['function']}")
    return findings


def check_memorization(tool_calls: list[dict], reward: float) -> list[str]:
    """Check for high reward with minimal tool usage."""
    findings = []
    n_calls = len(tool_calls)
    n_reads = sum(1 for tc in tool_calls if tc["function"] in ("Read", "read_file", "View"))
    n_searches = sum(1 for tc in tool_calls if tc["function"] in ("Grep", "Glob", "grep", "glob"))

    if reward > HIGH_REWARD_THRESHOLD and n_calls < MIN_TOOL_CALLS_FOR_HIGH_REWARD:
        findings.append(
            f"reward={reward:.2f} with only {n_calls} tool calls ({n_reads} reads, {n_searches} searches)"
        )
    return findings


def check_exploration_coverage(tool_calls: list[dict], agent_files: list[str]) -> list[str]:
    """Check if agent actually read the files it claims in answer.json."""
    findings = []
    files_read = {normalize_path(f) for f in extract_files_read(tool_calls)}
    claimed = {normalize_path(f) for f in agent_files}

    if not claimed:
        return findings

    unverified = set()
    for f in claimed:
        # Check if agent read this file or any file containing it
        if f not in files_read and not any(f in r for r in files_read):
            unverified.add(f)

    if unverified and len(unverified) > len(claimed) * MAX_UNVERIFIED_FRACTION:
        findings.append(
            f"claimed {len(claimed)} files, read {len(claimed) - len(unverified)}, "
            f"unverified: {sorted(unverified)[:5]}"
        )
    return findings


def check_answer_leakage(instruction: str, gold: dict) -> list[str]:
    """Check if instruction text contains gold file/function names."""
    findings = []
    instr_lower = instruction.lower()

    # Check gold files
    for gf in gold.get("files", []):
        if gf.lower() in instr_lower:
            findings.append(f"gold file '{gf}' appears in instruction")
        # Check module-style reference
        module = gf.lower().replace("/", ".").replace(".py", "")
        if module in instr_lower and len(module) > 10:
            findings.append(f"gold module path '{module}' appears in instruction")

    # Check private gold functions
    for gfn in gold.get("functions", []):
        leaf = gfn.split(".")[-1]
        if leaf.startswith("_") and leaf.lower() in instr_lower:
            findings.append(f"private function '{leaf}' appears in instruction")
        # Check full qualified name
        if gfn.lower() in instr_lower:
            findings.append(f"full function '{gfn}' appears in instruction")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_checks(
    task_id: str,
    instruction: str,
    gold: dict,
    trials: list[dict],
) -> dict:
    """Run all integrity checks for a task. Returns results dict."""
    results: dict[str, Any] = {"task_id": task_id, "checks": {}}

    # Per-agent checks
    for trial in trials:
        agent = trial["agent"]
        with open(trial["trajectory_path"]) as f:
            trajectory = json.load(f)
        with open(trial["reward_path"]) as f:
            reward_data = json.load(f)

        tool_calls = extract_tool_calls(trajectory)
        reward = reward_data.get("reward", 0)
        agent_files = reward_data.get("agent_files", [])

        agent_results = {}

        # Check 1: Gold contamination
        contamination = check_gold_contamination(tool_calls)
        agent_results["gold_contamination"] = {
            "status": "FAIL" if contamination else "PASS",
            "findings": contamination,
        }

        # Check 2: Memorization
        memorization = check_memorization(tool_calls, reward)
        agent_results["memorization"] = {
            "status": "FAIL" if memorization else "PASS",
            "findings": memorization,
            "tool_calls": len(tool_calls),
            "reward": reward,
        }

        # Check 3: Exploration coverage
        coverage = check_exploration_coverage(tool_calls, agent_files)
        agent_results["exploration_coverage"] = {
            "status": "WARN" if coverage else "PASS",
            "findings": coverage,
        }

        results["checks"][agent] = agent_results

    # Check 4: Answer leakage (task-level, not per-agent)
    leakage = check_answer_leakage(instruction, gold)
    results["checks"]["answer_leakage"] = {
        "status": "FAIL" if leakage else "PASS",
        "findings": leakage,
    }

    return results


def print_results(results: dict) -> None:
    """Pretty-print integrity check results."""
    tid = results["task_id"]
    print(f"\n=== Trajectory Integrity: {tid} ===\n")

    # Per-agent checks
    agents = [k for k in results["checks"] if k != "answer_leakage"]
    for check_name in ["gold_contamination", "memorization", "exploration_coverage"]:
        label = check_name.upper().replace("_", " ")
        print(f"{label}")
        for agent in agents:
            agent_check = results["checks"].get(agent, {}).get(check_name, {})
            status = agent_check.get("status", "?")
            findings = agent_check.get("findings", [])
            extra = ""
            if check_name == "memorization":
                n = agent_check.get("tool_calls", 0)
                r = agent_check.get("reward", 0)
                extra = f" ({n} tool calls, reward={r:.2f})"
            color = "\033[32m" if status == "PASS" else "\033[33m" if status == "WARN" else "\033[31m"
            print(f"  {agent:>8s}: {color}{status}\033[0m{extra}")
            for finding in findings:
                print(f"           {finding}")
        print()

    # Answer leakage
    leakage = results["checks"].get("answer_leakage", {})
    status = leakage.get("status", "?")
    color = "\033[32m" if status == "PASS" else "\033[31m"
    print("ANSWER LEAKAGE")
    print(f"  {color}{status}\033[0m")
    for finding in leakage.get("findings", []):
        print(f"  {finding}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task_id", nargs="?", help="Task ID to check (e.g., craft-uvicorn-c-5cd79476)")
    parser.add_argument("--all", action="store_true", help="Check all tasks found in job dirs")
    parser.add_argument(
        "--job-dirs",
        nargs="+",
        default=[],
        help="Job directories to scan for trajectories",
    )
    parser.add_argument(
        "--tier-mapping",
        default="validation/tier_job_dirs.json",
        help="Tier mapping file to auto-discover job dirs",
    )
    parser.add_argument(
        "--gold-dir",
        nargs="+",
        default=[],
        help="Additional gold directories to search for tasks (e.g., /tmp/pipeline/output)",
    )
    args = parser.parse_args()

    # Auto-discover job dirs from tier mapping if none provided
    job_dirs = args.job_dirs
    if not job_dirs and os.path.exists(args.tier_mapping):
        with open(args.tier_mapping) as f:
            mapping = json.load(f)
        job_dirs = list(mapping.values())
        print(f"Auto-discovered {len(job_dirs)} job dirs from {args.tier_mapping}", file=sys.stderr)

    if not job_dirs:
        print("ERROR: No job directories. Provide --job-dirs or --tier-mapping.", file=sys.stderr)
        sys.exit(1)

    # Load task data
    review_data = load_review_data()
    gold_dirs = args.gold_dir or None

    if args.all:
        # Check all tasks found in job dirs
        task_ids = set()
        for jd in job_dirs:
            if not os.path.isdir(jd):
                continue
            for d in os.listdir(jd):
                if d.startswith("craft-"):
                    # harbor-agent-patches.diff uses `-<uuid>` (not `__<uuid>`) as trial suffix
                    task_ids.add(d.rsplit("-", 1)[0])
        task_ids_list = sorted(task_ids)
    elif args.task_id:
        task_ids_list = [args.task_id]
    else:
        parser.error("Provide a task_id or --all")
        return

    all_results = []
    for task_id in task_ids_list:
        # Load task instruction + gold
        task = review_data.get(task_id)
        if task:
            instruction = task.get("instruction", "")
            gold = task.get("gold", {})
        else:
            # Try gold/ directory (search-from-t2 tasks)
            t2_task = load_task_from_gold(task_id, gold_dirs=gold_dirs)
            if t2_task:
                instruction = t2_task.get("instruction", "")
                gold = t2_task.get("gold_answer", {})
            else:
                print(f"  Skipping {task_id}: not found in review_data or gold/", file=sys.stderr)
                continue

        trials = find_trials(task_id, job_dirs)
        if not trials:
            print(f"  Skipping {task_id}: no trials found in job dirs", file=sys.stderr)
            continue

        results = run_checks(task_id, instruction, gold, trials)
        all_results.append(results)
        print_results(results)

    # Summary
    if len(all_results) > 1:
        n_contaminated = sum(
            1
            for r in all_results
            if any(
                r["checks"].get(a, {}).get("gold_contamination", {}).get("status") == "FAIL"
                for a in r["checks"]
                if a != "answer_leakage"
            )
        )
        n_memorized = sum(
            1
            for r in all_results
            if any(
                r["checks"].get(a, {}).get("memorization", {}).get("status") == "FAIL"
                for a in r["checks"]
                if a != "answer_leakage"
            )
        )
        n_coverage_warn = sum(
            1
            for r in all_results
            if any(
                r["checks"].get(a, {}).get("exploration_coverage", {}).get("status") == "WARN"
                for a in r["checks"]
                if a != "answer_leakage"
            )
        )
        n_leaked = sum(
            1 for r in all_results if r["checks"].get("answer_leakage", {}).get("status") == "FAIL"
        )
        print(f"\n{'=' * 50}")
        print(f"SUMMARY: {len(all_results)} tasks checked")
        print(f"  Gold contamination: {n_contaminated} FAIL")
        print(f"  Memorization:       {n_memorized} FAIL")
        print(f"  Coverage warnings:  {n_coverage_warn} WARN")
        print(f"  Answer leakage:     {n_leaked} FAIL")


# Need Any for type annotation
from typing import Any  # noqa: E402

if __name__ == "__main__":
    main()
