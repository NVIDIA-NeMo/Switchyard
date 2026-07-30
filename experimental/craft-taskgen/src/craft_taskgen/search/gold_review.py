# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold-review step: generate Harbor tasks, run Opus, collect and apply verdicts.

Creates one review task per T2 parent (per repo). Opus uses the /gold-review skill
to review all search tasks derived from that T2 problem. Volume mounts repos/
for code access.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any

from craft_taskgen.search.config import SearchPipelineState

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "gold-review"
_BUNDLED_SKILL = _TEMPLATES_DIR / "skills" / "gold-review"

# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------

_INSTRUCTION_TEMPLATE = """\
# Batch Gold Data Review

Use your /gold-review skill to review the gold answer quality for {n_tasks} search \
tasks derived from the **{repo}** repository.

The repository source code is mounted at `/repos/{repo}/`.
You should read source code from there to verify gold answers.
Pre-change commit: `{commit}` (the code state agents see in their Docker containers).

## Tasks to Review

The file `/app/data/tasks_to_review.json` contains all {n_tasks} tasks with their:
- instruction (the developer question)
- gold_answer (files, functions, assertions, explanation)

For **each** task, follow the /gold-review skill workflow:

1. Read each gold function's source code at `/repos/{repo}/`
2. Classify each function:
   - **KEEP**: Contains substantive logic relevant to the question
   - **DEMOTE**: Trivial delegation, one-liner, abstract stub, or generic plumbing
   - **REMOVE**: Doesn't exist in the repo, or completely irrelevant
3. Verify each assertion is factually correct by reading the code
4. Check if the instruction leaks gold file paths or private function names
5. Assess the explanation accuracy

{trajectory_section}

## Output Format

Write your verdicts to `/app/verdicts.json` as a JSON array -- one object per task:

```json
[
  {{
    "task_id": "craft-{repo}-c-xyz",
    "file_actions": [
      {{"file": "path/to/file.py", "action": "KEEP", "reason": "core logic for the question"}},
      {{"file": "conftest.py", "action": "REMOVE", "reason": "test fixture, not relevant"}}
    ],
    "function_actions": [
      {{"function": "module.Class.method", "action": "KEEP", "reason": "contains branching logic for X"}},
      {{"function": "module.helper", "action": "DEMOTE", "reason": "one-liner delegation"}}
    ],
    "assertion_verdicts": [
      {{"assertion": "X does Y", "verdict": "CORRECT", "reason": "confirmed in source"}},
      {{"assertion": "Z handles W", "verdict": "INCORRECT", "reason": "code shows it handles V instead"}}
    ],
    "explanation_assessment": "ACCURATE",
    "integrity_flags": [],
    "overall_recommendation": "ACCEPT",
    "notes": ""
  }}
]
```

Valid file/function actions: KEEP, DEMOTE, REMOVE
Valid assertion verdicts: CORRECT, INCORRECT, UNVERIFIABLE
Valid recommendations: ACCEPT, FLAG, REJECT

For trajectory integrity, check the files in `/app/data/opus_trajectories/` and \
`/app/data/codex_trajectories/` for: gold contamination (reads of gold_answer.json), \
memorization (high reward with <3 tool calls), and exploration coverage (claimed files \
not actually read).

**You MUST review ALL {n_tasks} tasks. Write one verdict per task.**
"""


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _write_review_task_toml(task_dir: str, tid: str, n_tasks: int) -> None:
    """Write task.toml with generous timeouts for batch review."""
    timeout = max(600, n_tasks * 180)  # ~3 min per task
    content = textwrap.dedent(f"""\
        version = "1.0"

        [metadata]
        name = "{tid}"
        difficulty = "hard"

        [verifier]
        timeout_sec = 120

        [agent]
        timeout_sec = {timeout}

        [environment]
        build_timeout_sec = 300.0
        cpus = 2
        memory_mb = 4096
        storage_mb = 10240
        gpus = 0
        allow_internet = true
        mcp_servers = []

        [environment.env]
        ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
        ANTHROPIC_BASE_URL = "${{ANTHROPIC_BASE_URL}}"
        OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
        OPENAI_BASE_URL = "${{OPENAI_BASE_URL}}"

        [solution.env]
    """)
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write(content)


def _copy_trajectories(
    data_dir: str,
    task_ids: list[str] | set[str],
    job_dirs: dict[str, str],
) -> str:
    """Copy trajectory excerpts for the tasks. Returns trajectory section for instruction."""
    task_ids = set(task_ids)  # O(1) membership test
    has_trajectories = False
    for agent_key, job_dir in job_dirs.items():
        if not job_dir or not os.path.isdir(job_dir):
            continue
        traj_dir = os.path.join(data_dir, f"{agent_key}_trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        for trial_name in os.listdir(job_dir):
            # harbor-agent-patches.diff uses `-<uuid>` (not `__<uuid>`) as the trial suffix
            tid = trial_name.rsplit("-", 1)[0]
            if tid not in task_ids:
                continue
            traj_src = os.path.join(job_dir, trial_name, "agent", "trajectory.json")
            reward_src = os.path.join(job_dir, trial_name, "verifier", "reward.json")
            if os.path.exists(traj_src):
                shutil.copy2(traj_src, os.path.join(traj_dir, f"{tid}.json"))
                has_trajectories = True
            if os.path.exists(reward_src):
                shutil.copy2(reward_src, os.path.join(traj_dir, f"{tid}_reward.json"))

    if has_trajectories:
        return (
            "## Agent Trajectories\n\n"
            "Agent trajectory files are available in `/app/data/opus_trajectories/` and "
            "`/app/data/codex_trajectories/`. Use these for trajectory integrity checks "
            "(gold contamination, memorization, exploration coverage)."
        )
    return ""


def generate_review_tasks(
    output_dir: str,
    review_dir: str,
    *,
    contexts_path: str = "",
    job_dirs: dict[str, str] | None = None,
) -> list[str]:
    """Generate gold-review Harbor tasks, one per T2 parent (per repo).

    Args:
        output_dir: Directory containing approach-{a,b,c}/search_tasks.json.
        review_dir: Output directory for review Harbor task directories.
        contexts_path: Path to _all_contexts.json (for repo/commit info).
        job_dirs: Map of agent key -> job dir for trajectory copying.

    Returns:
        List of generated review task IDs.
    """
    # Load all search tasks
    all_tasks: list[dict] = []
    for approach in ["a", "b", "c"]:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            all_tasks.extend(json.load(f))
    print(f"Loaded {len(all_tasks)} search tasks for gold review")

    # Load contexts for repo/commit info
    contexts: dict[str, dict] = {}
    if contexts_path and os.path.exists(contexts_path):
        with open(contexts_path) as f:
            contexts = {c["task_id"]: c for c in json.load(f)}

    # Group by parent T2 task (= by repo)
    by_parent: dict[str, list[dict]] = {}
    for task in all_tasks:
        parent = task.get("parent_t2_task", "unknown")
        by_parent.setdefault(parent, []).append(task)

    # Read templates
    with open(_TEMPLATES_DIR / "Dockerfile") as f:
        dockerfile = f.read()
    with open(_TEMPLATES_DIR / "test.sh", "rb") as f:
        test_sh = f.read()
    with open(_TEMPLATES_DIR / "test_runner.py", "rb") as f:
        test_runner = f.read()

    os.makedirs(review_dir, exist_ok=True)
    task_ids_out: list[str] = []
    agent_job_dirs = job_dirs or {}

    for parent_id, tasks in sorted(by_parent.items()):
        ctx = contexts.get(parent_id, {})
        url = ctx.get("solve_info", {}).get("upstream_url", "")
        repo = url.rstrip("/").removesuffix(".git").split("/")[-1] if url else "unknown"
        commit = ctx.get("solve_info", {}).get("commit_hash", "")[:12]

        tid = f"review-{repo}-{parent_id[-8:]}"
        task_dir = os.path.join(review_dir, tid)
        env_dir = os.path.join(task_dir, "environment")
        tests_dir = os.path.join(task_dir, "tests")
        sol_dir = os.path.join(task_dir, "solution")

        for d in (task_dir, env_dir, tests_dir, sol_dir):
            os.makedirs(d, exist_ok=True)

        # task.toml
        _write_review_task_toml(task_dir, tid, len(tasks))

        # Dockerfile + bundled gold-review skill
        skill_dest = os.path.join(env_dir, "skills", "gold-review")
        shutil.copytree(str(_BUNDLED_SKILL), skill_dest, dirs_exist_ok=True)
        with open(os.path.join(env_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile)

        # Put review data inside Docker build context so agent can access it
        data_dir = os.path.join(env_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "tasks_to_review.json"), "w") as f:
            json.dump(tasks, f, indent=2)

        # Also put in tests/ for the verifier (verifier has separate mount)
        with open(os.path.join(tests_dir, "tasks_to_review.json"), "w") as f:
            json.dump(tasks, f, indent=2)

        # Copy trajectories into Docker context
        task_id_list = [t["id"] for t in tasks]
        traj_section = _copy_trajectories(data_dir, task_id_list, agent_job_dirs)

        # instruction.md
        instruction = _INSTRUCTION_TEMPLATE.format(
            n_tasks=len(tasks),
            repo=repo,
            commit=commit or "HEAD",
            trajectory_section=traj_section,
        )
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write(instruction)

        # test.sh + test_runner.py
        with open(os.path.join(tests_dir, "test.sh"), "wb") as f:
            f.write(test_sh)
        os.chmod(os.path.join(tests_dir, "test.sh"), 0o755)
        with open(os.path.join(tests_dir, "test_runner.py"), "wb") as f:
            f.write(test_runner)

        # solution/solve.sh (trivial -- no oracle for reviews)
        with open(os.path.join(sol_dir, "solve.sh"), "w") as f:
            f.write("#!/bin/bash\necho 'No oracle solution for gold review tasks'\n")
        os.chmod(os.path.join(sol_dir, "solve.sh"), 0o755)

        task_ids_out.append(tid)
        print(f"  {tid}: {len(tasks)} tasks ({repo} @ {commit})")

    # Registry
    registry = [
        {
            "name": "craft-gold-review",
            "version": "1.0",
            "metrics": [{"type": "mean"}],
            "tasks": task_ids_out,
        }
    ]
    with open(os.path.join(review_dir, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nGenerated {len(task_ids_out)} review tasks -> {review_dir}")
    return task_ids_out


# ---------------------------------------------------------------------------
# Verdict collection and application
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"KEEP", "DEMOTE", "REMOVE"}
VALID_RECOMMENDATIONS = {"ACCEPT", "FLAG", "REJECT"}


def collect_verdicts(job_dir: str) -> dict[str, list[dict]]:
    """Collect verdicts.json from gold-review trial results.

    Returns:
        Map of task_id -> list of verdict dicts.
    """
    verdicts_by_task: dict[str, list[dict]] = {}
    if not os.path.isdir(job_dir):
        print(f"  WARNING: gold-review job dir not found: {job_dir}")
        return verdicts_by_task

    for trial_name in os.listdir(job_dir):
        reward_path = os.path.join(job_dir, trial_name, "verifier", "reward.json")
        if not os.path.exists(reward_path):
            continue
        with open(reward_path) as f:
            reward_data = json.load(f)
        for v in reward_data.get("verdicts", []):
            tid = v.get("task_id", "")
            if tid:
                verdicts_by_task.setdefault(tid, []).append(v)

    print(f"  Collected verdicts for {len(verdicts_by_task)} tasks from {job_dir}")
    return verdicts_by_task


def apply_verdicts(
    output_dir: str,
    verdicts: dict[str, list[dict]],
) -> dict[str, Any]:
    """Apply gold-review verdicts to search_tasks.json files.

    For each task with a verdict:
    - REMOVE functions get moved to gold_answer.removed_functions
    - DEMOTE functions get moved to gold_answer.alt_functions
    - INCORRECT assertions get flagged
    - REJECT recommendation -> task is removed from the file

    Returns audit log dict.
    """
    audit: dict[str, Any] = {"applied": [], "rejected_tasks": [], "no_verdict": []}

    for approach in ["a", "b", "c"]:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tasks = json.load(f)

        kept_tasks = []
        for task in tasks:
            tid = task["id"]
            task_verdicts = verdicts.get(tid, [])
            if not task_verdicts:
                audit["no_verdict"].append(tid)
                kept_tasks.append(task)
                continue

            # Use the first verdict (single-run review)
            v = task_verdicts[0]
            rec = v.get("overall_recommendation", "ACCEPT")

            if rec == "REJECT":
                audit["rejected_tasks"].append({"task_id": tid, "verdict": v})
                continue

            # Apply function actions with an aggregate floor: preserve at least
            # max(1, original_count // 3) functions in gold["functions"], so a
            # verdict that REMOVE/DEMOTEs every function doesn't leave the task
            # un-scorable under filter rule `no_gold_functions`.
            gold = task.get("gold_answer", {})
            actions_applied = []

            original_functions = list(gold.get("functions", []))
            original_n = len(original_functions)
            floor = max(1, original_n // 3) if original_n else 0

            reducing = [
                fa
                for fa in v.get("function_actions", [])
                if fa.get("action") in ("REMOVE", "DEMOTE") and fa.get("function") in original_functions
            ]
            reduced_names = {fa["function"] for fa in reducing}
            survivors = [f for f in original_functions if f not in reduced_names]

            clamped: set[str] = set()
            if original_n and len(survivors) < floor:
                needed = floor - len(survivors)
                # Prefer clamping REMOVEs first (more destructive) then DEMOTEs.
                for fa in sorted(reducing, key=lambda x: 0 if x["action"] == "REMOVE" else 1):
                    if len(clamped) >= needed:
                        break
                    clamped.add(fa["function"])

            for fa in v.get("function_actions", []):
                func_name = fa.get("function", "")
                action = fa.get("action", "KEEP")
                if func_name in clamped:
                    actions_applied.append(
                        {
                            "function": func_name,
                            "action": "KEEP_CLAMPED",
                            "original_action": action,
                        }
                    )
                    continue
                if action == "REMOVE" and func_name in gold.get("functions", []):
                    gold["functions"].remove(func_name)
                    gold.setdefault("removed_functions", []).append(func_name)
                    actions_applied.append({"function": func_name, "action": "REMOVE"})
                elif action == "DEMOTE" and func_name in gold.get("functions", []):
                    gold["functions"].remove(func_name)
                    gold.setdefault("alt_functions", []).append(func_name)
                    actions_applied.append({"function": func_name, "action": "DEMOTE"})

            for fa in v.get("file_actions", []):
                file_path = fa.get("file", "")
                action = fa.get("action", "KEEP")
                if action == "REMOVE" and file_path in gold.get("files", []):
                    gold["files"].remove(file_path)
                    gold.setdefault("removed_files", []).append(file_path)
                    actions_applied.append({"file": file_path, "action": "REMOVE"})
                elif action == "DEMOTE" and file_path in gold.get("files", []):
                    gold["files"].remove(file_path)
                    gold.setdefault("alt_files", []).append(file_path)
                    actions_applied.append({"file": file_path, "action": "DEMOTE"})

            if actions_applied:
                audit["applied"].append({"task_id": tid, "actions": actions_applied})

            kept_tasks.append(task)

        with open(path, "w") as f:
            json.dump(kept_tasks, f, indent=2)
        n_removed = len(tasks) - len(kept_tasks)
        print(f"  Approach {approach}: {len(tasks)} -> {len(kept_tasks)} (-{n_removed})")

    return audit


def run_gold_review(
    state: SearchPipelineState,
) -> None:
    """Full gold-review step: generate tasks, run Harbor, collect and apply verdicts.

    Uses run_cmd and find_job_dir from the steps module for Harbor execution.
    """
    from craft_taskgen.config import OPUS_MODEL
    from craft_taskgen.search.steps import find_job_dir, run_cmd

    review_dir = os.path.join(state.output_dir, "harbor-gold-review")
    contexts_path = os.path.join(state.output_dir, "_all_contexts.json")

    # 1. Generate review Harbor tasks
    review_task_ids = generate_review_tasks(
        output_dir=state.output_dir,
        review_dir=review_dir,
        contexts_path=contexts_path,
        job_dirs={k: v for k, v in state.job_dirs.items() if k in ("opus", "codex")},
    )

    if not review_task_ids:
        print("  No review tasks generated -- skipping gold review")
        return

    # 2. Run Harbor with repo mounts
    repos_abs = os.path.abspath(state.repos_dir)
    mounts_json = json.dumps([{"type": "bind", "source": repos_abs, "target": "/repos", "read_only": True}])

    review_jobs_dir = os.path.join("jobs", "gold-review")
    os.makedirs(review_jobs_dir, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "harbor",
        "run",
        "--agent",
        "claude-code",
        "--model",
        OPUS_MODEL,
        "--path",
        f"{review_dir}/",
        "--mounts-json",
        mounts_json,
        "--n-concurrent",
        str(state.concurrency),
        "--env",
        "docker",
        "-o",
        f"{review_jobs_dir}/",
    ]

    before_ts = time.time()
    run_cmd(cmd, "gold-review: Opus reviewing gold answers")

    job_dir = find_job_dir(review_jobs_dir, before_ts)
    if not job_dir:
        print("  WARNING: Could not find gold-review job directory")
        return

    state.job_dirs["gold_review"] = job_dir

    # 3. Collect verdicts
    all_verdicts = collect_verdicts(job_dir)

    # 4. Apply verdicts to search tasks
    audit = apply_verdicts(state.output_dir, all_verdicts)

    # 5. Update task statuses with review recommendations
    from craft_taskgen.search.config import SearchTaskStatus

    for tid, verdict_list in all_verdicts.items():
        if not verdict_list:
            continue
        v = verdict_list[0]
        ts = state.task_statuses.setdefault(tid, SearchTaskStatus())
        ts.review_recommendation = v.get("overall_recommendation", "")
        ts.review_flags = v.get("integrity_flags", [])

    # 6. Write audit log
    audit_path = os.path.join(state.output_dir, "gold_review_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Audit log: {audit_path}")
    print(f"  Applied: {len(audit['applied'])} tasks modified")
    print(f"  Rejected: {len(audit['rejected_tasks'])} tasks removed")
    print(f"  No verdict: {len(audit['no_verdict'])} tasks")
