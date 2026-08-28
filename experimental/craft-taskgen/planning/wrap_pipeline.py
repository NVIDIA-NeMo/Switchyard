#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transform a craft-bench dataset into planner or implementer mode.

Takes any craft-bench dataset directory and generates a derived dataset with
modified instructions and scoring for the iterative planning pipeline.

Planner mode: agent explores repo, writes plan.md + plan.json, scored by
file/symbol recall against gold files extracted from solve.sh.

Implementer mode: agent receives a plan from a prior planner run, implements
the changes, scored by existing F2P tests.

Usage:
    python scripts/planning/craft_pipeline.py \\
        --source harbor-tasks/iterative-planning \\
        --mode planner \\
        --output /tmp/planner-dataset

    python scripts/planning/craft_pipeline.py \\
        --source harbor-tasks/iterative-planning \\
        --mode implementer \\
        --plans-dir jobs/planner-opus/ \\
        --output /tmp/impl-dataset

    python scripts/planning/craft_pipeline.py \\
        --source harbor-tasks/iterative-planning \\
        --mode planner \\
        --filter 'scrapy__*' \\
        --output /tmp/planning-only
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import shutil
import stat
from pathlib import Path

logger = logging.getLogger(__name__)


PLANNER_INSTRUCTION_TEMPLATE = """\
You are working in the {repo_hint} repository. The source code is in `/repo/`.

Produce an implementation plan for the task below. Another engineer will \
implement your plan. Explore the codebase with your tools (read, grep, glob) \
but do not modify any files.

Write two files when done:

1. `/logs/agent/plan.md` - the implementation plan.
2. `/logs/agent/plan.json` - a structured summary:
```json
{{
  "files_to_touch": ["path/to/file.py", "..."],
  "symbols_to_modify": ["ClassName.method_name", "..."]
}}
```

## Task

{spec}
"""

IMPLEMENTER_INSTRUCTION_TEMPLATE = """\
Implement the following plan. The codebase is at `/repo/`.

# Plan

{plan_content}

---

# Original Task

{original_instruction}
"""

PLANNER_TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/score.py
"""

PLANNER_SCORE_PY = r"""#!/usr/bin/env python3
import json
from pathlib import Path

log_dir = Path("/logs/verifier")
log_dir.mkdir(parents=True, exist_ok=True)
agent_dir = Path("/logs/agent")

plan_md = agent_dir / "plan.md"
plan_json = agent_dir / "plan.json"
metadata = json.load(open("/tests/metadata.json"))
gold_files = set(metadata.get("src_files", []))

results = {"plan_md_exists": plan_md.exists(), "plan_json_exists": plan_json.exists()}

if not plan_md.exists():
    print("No plan.md found")
    results["reward"] = 0.0
    results["error"] = "no_plan_md"
elif not plan_json.exists():
    print("No plan.json found")
    results["reward"] = 0.0
    results["error"] = "no_plan_json"
else:
    try:
        plan = json.loads(plan_json.read_text())
        planned_files = set(plan.get("files_to_touch", []))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Invalid plan.json: {e}")
        results["reward"] = 0.0
        results["error"] = f"invalid_json: {e}"
        planned_files = set()

    if planned_files and gold_files:
        file_matched = planned_files & gold_files
        file_recall = len(file_matched) / len(gold_files)
        results["planned_files"] = sorted(planned_files)
        results["gold_files"] = sorted(gold_files)
        results["matched_files"] = sorted(file_matched)
        results["file_recall"] = round(file_recall, 3)

    SKIP_SYMBOLS = {"__init__", "__repr__", "__str__", "__eq__", "__hash__",
                    "__new__", "__del__", "__enter__", "__exit__"}
    raw_planned = plan.get("symbols_to_modify", [])
    raw_gold = metadata.get("gold_symbols", [])
    planned_sym = set(s.split(".")[-1].lstrip("_") for s in raw_planned) - SKIP_SYMBOLS
    gold_sym = set(s.lstrip("_") for s in raw_gold) - SKIP_SYMBOLS
    if planned_sym and gold_sym:
        sym_matched = planned_sym & gold_sym
        sym_recall = len(sym_matched) / len(gold_sym)
        results["planned_symbols"] = sorted(raw_planned)
        results["gold_symbols"] = sorted(raw_gold)
        results["matched_symbols"] = sorted(sym_matched)
        results["symbol_recall"] = round(sym_recall, 3)

    results["reward"] = results.get("file_recall", 0.0)
    results["plan_text_length"] = len(plan_md.read_text()) if plan_md.exists() else 0

    print(f"File recall:   {results.get('file_recall', 0):.3f}")
    print(f"Symbol recall: {results.get('symbol_recall', 'N/A')}")
    print(f"Reward:        {results['reward']:.3f}")

(log_dir / "reward.txt").write_text(f"{results['reward']:.3f}")
json.dump(results, open(log_dir / "results.json", "w"), indent=2)
"""


def extract_gold_files_from_solve(solve_sh_path):
    """Parse solve.sh to extract source files the oracle modifies."""
    files = []
    text = solve_sh_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"git checkout FETCH_HEAD -- (.+)", line)
        if m:
            files.extend(m.group(1).split())
        m = re.match(r"rm -f (.+)", line)
        if m:
            files.extend(m.group(1).split())
    return [f for f in files if not _is_test_path(f)]


def _is_test_path(path):
    lower = path.lower()
    return "test" in lower or "fixture" in lower


def extract_spec_from_instruction(instruction_path, strip_test_hints=False):
    """Extract the spec text from instruction.md, stripping the ## Environment footer."""
    text = instruction_path.read_text()
    env_marker = "## Environment"
    if env_marker in text:
        text = text[: text.index(env_marker)].rstrip()
    if strip_test_hints:
        lines = text.splitlines()
        lines = [ln for ln in lines if not re.match(r"^Run (?:tests|the test)", ln.strip())]
        text = "\n".join(lines).rstrip()
    return text


def guess_repo_from_dockerfile(dockerfile_path):
    """Try to extract the repo name from a git clone line in the Dockerfile."""
    text = dockerfile_path.read_text()
    m = re.search(r"git clone.*github\.com/([^/]+/[^/\s.]+)", text)
    if m:
        repo = m.group(1)
        if repo.endswith(".git"):
            repo = repo[:-4]
        return repo
    return "unknown"


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def generate_planner_dataset(source_dir, output_dir, task_filter=None):
    """Transform a craft-bench dataset into planner mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_generated = []

    for task_dir in sorted(source_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        if task_filter and not fnmatch.fnmatch(task_dir.name, task_filter):
            continue

        out_task = output_dir / task_dir.name
        out_task.mkdir(parents=True, exist_ok=True)

        env_src = task_dir / "environment"
        sol_src = task_dir / "solution"
        env_dst = out_task / "environment"
        sol_dst = out_task / "solution"

        if env_dst.exists():
            if env_dst.is_symlink():
                env_dst.unlink()
            else:
                shutil.rmtree(env_dst)
        if sol_dst.exists():
            if sol_dst.is_symlink():
                sol_dst.unlink()
            else:
                shutil.rmtree(sol_dst)

        env_dst.symlink_to(env_src.resolve())
        sol_dst.symlink_to(sol_src.resolve())

        shutil.copy2(task_dir / "task.toml", out_task / "task.toml")

        spec = extract_spec_from_instruction(task_dir / "instruction.md", strip_test_hints=True)
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        repo_hint = guess_repo_from_dockerfile(dockerfile_path)

        (out_task / "instruction.md").write_text(
            PLANNER_INSTRUCTION_TEMPLATE.format(
                repo_hint=repo_hint,
                spec=spec,
            )
        )

        tests_dir = out_task / "tests"
        if tests_dir.exists():
            shutil.rmtree(tests_dir)
        tests_dir.mkdir()

        gold_files = extract_gold_files_from_solve(task_dir / "solution" / "solve.sh")
        gold_symbols = []

        for candidate in ["planning_tasks.json", "planning_tasks_bootstrapped.json"]:
            p = Path(__file__).parent / candidate
            if p.exists():
                planning_tasks_json = p
                break
        else:
            planning_tasks_json = None
        if planning_tasks_json:
            ptasks = json.loads(planning_tasks_json.read_text())
            for ptask in ptasks.values():
                if ptask["task_name"] == task_dir.name:
                    gold_files = ptask.get("src_files", gold_files)
                    gold_symbols = ptask.get("gold_symbols", [])
                    break

        metadata = {"src_files": gold_files, "gold_symbols": gold_symbols}
        (tests_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        test_sh = tests_dir / "test.sh"
        test_sh.write_text(PLANNER_TEST_SH)
        make_executable(test_sh)

        (tests_dir / "score.py").write_text(PLANNER_SCORE_PY)

        tasks_generated.append(task_dir.name)
        logger.info("Planner: %s (%d gold files)", task_dir.name, len(gold_files))

    registry = [
        {
            "name": output_dir.name,
            "version": "1.0",
            "description": f"Planner mode ({len(tasks_generated)} tasks)",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": t, "path": t} for t in tasks_generated],
        }
    ]
    (output_dir / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    return tasks_generated


def generate_implementer_dataset(source_dir, plans_dir, output_dir, task_filter=None):
    """Transform a craft-bench dataset into implementer mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_dir = Path(plans_dir)
    tasks_generated = []
    tasks_skipped = []

    for task_dir in sorted(source_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        if task_filter and not fnmatch.fnmatch(task_dir.name, task_filter):
            continue

        plan_path = None
        # Harbor appends a random suffix to task dirs (e.g., hugapi__hug-651__dH4hXAg)
        # Search for directories starting with the task name
        if plans_dir.is_dir():
            for candidate_dir in sorted(plans_dir.iterdir()):
                if not candidate_dir.is_dir():
                    continue
                if not candidate_dir.name.startswith(task_dir.name):
                    continue
                for subpath in ["agent/plan.md", "plan.md"]:
                    candidate = candidate_dir / subpath
                    if candidate.exists():
                        plan_path = candidate
                        break
                if plan_path:
                    break

        if plan_path is None:
            logger.warning("Skipping %s: no plan found in %s", task_dir.name, plans_dir)
            tasks_skipped.append(task_dir.name)
            continue

        plan_content = plan_path.read_text()

        out_task = output_dir / task_dir.name
        out_task.mkdir(parents=True, exist_ok=True)

        for subdir in ("environment", "solution", "tests"):
            src = task_dir / subdir
            dst = out_task / subdir
            if dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.symlink_to(src.resolve())

        shutil.copy2(task_dir / "task.toml", out_task / "task.toml")

        original_instruction = (task_dir / "instruction.md").read_text()
        (out_task / "instruction.md").write_text(
            IMPLEMENTER_INSTRUCTION_TEMPLATE.format(
                plan_content=plan_content,
                original_instruction=original_instruction,
            )
        )

        tasks_generated.append(task_dir.name)
        logger.info("Implementer: %s (plan: %d chars)", task_dir.name, len(plan_content))

    registry = [
        {
            "name": output_dir.name,
            "version": "1.0",
            "description": f"Implementer mode ({len(tasks_generated)} tasks)",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": t, "path": t} for t in tasks_generated],
        }
    ]
    (output_dir / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    if tasks_skipped:
        logger.warning("Skipped %d tasks (no plan): %s", len(tasks_skipped), tasks_skipped)

    return tasks_generated


def main():
    parser = argparse.ArgumentParser(description="Generate planner/implementer datasets.")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--mode", required=True, choices=["planner", "implementer"])
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--plans-dir", help="Plans directory (implementer mode)")
    parser.add_argument("--filter", help="Glob filter for task names (e.g. 'scrapy__*')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if args.mode == "planner":
        tasks = generate_planner_dataset(source_dir, output_dir, args.filter)
    elif args.mode == "implementer":
        if not args.plans_dir:
            logger.error("--plans-dir required for implementer mode")
            return 1
        tasks = generate_implementer_dataset(source_dir, args.plans_dir, output_dir, args.filter)

    print(f"\n{'=' * 60}")
    print(f"{len(tasks)} tasks generated in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
