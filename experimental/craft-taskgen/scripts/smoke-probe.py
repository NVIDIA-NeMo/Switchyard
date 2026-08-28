# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast smoke-step iteration harness — run the Harbor agent trial in isolation.

The full pipeline (select → eval → build → docker → oracle → smoke → triage)
takes 2+ hours per task. When iterating on the *smoke* step alone — the Harbor
agent trial that actually solves the task and produces the reward — that round
trip is far too slow. This script runs only `runner._run_smoke_async` against an
already-built task directory, so you can validate a new agent/model (e.g. codex
+ GPT-5.5) end-to-end in minutes instead of hours.

Layered testing for the smoke step, fast → slow:
  1. `uv run pytest tests/test_smoke_runner.py`  — instant; argv/env construction
  2. `… smoke-probe.py --dry-run -t <task_dir>`  — instant; prints resolved
                                                    harbor argv + env overrides
                                                    (catalog path, reasoning) for
                                                    real config, no Docker
  3. `… smoke-probe.py -t <task_dir>`            — ~5-15 min; real harbor+codex
                                                    trial on one built task
  4. full pipeline run                            — 2+ hr; before merge

Examples:
  # Verify the codex + GPT-5.5 default wiring without launching Docker
  uv run python scripts/smoke-probe.py --dry-run \
      -t templates/t2v3-CE0266-celery-retry-unification

  # Actually run the trial against a *built* task (needs Docker + .env + harbor).
  # The target must be a runnable task dir with a tests/ verifier — a bare
  # templates/ dir has no tests/ and harbor rejects it (surfaces as NO_TRIAL).
  uv run python scripts/smoke-probe.py \
      -t harbor-tasks/craft-tools-v4/<some-task-dir>

  # Compare agents on the same task
  uv run python scripts/smoke-probe.py -t <dir> --agent claude-code \
      --model azure/anthropic/claude-opus-4-6 --reasoning-effort high

A "task dir" is any built Harbor task: a directory with task.toml, environment/,
instruction.md, AND a tests/ verifier (harbor's TaskPaths.is_valid requires the
test path). Pass `-t` multiple times, or `--suite-dir` to probe every task.toml
under a tree.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import craft_taskgen.config as _cfg
from craft_taskgen.config import PipelineProfile, Stage, TaskState
from craft_taskgen.runner import _build_smoke_cmd, _run_smoke_async


def _discover_task_dirs(args: argparse.Namespace) -> list[str]:
    dirs: list[str] = list(args.task_dir or [])
    if args.suite_dir:
        for toml in sorted(Path(args.suite_dir).glob("*/task.toml")):
            dirs.append(str(toml.parent))
    deduped = sorted(set(dirs))
    bad = [d for d in deduped if not (Path(d) / "task.toml").is_file()]
    if bad:
        sys.exit(f"ERROR: not a built task dir (no task.toml): {', '.join(bad)}")
    if not deduped:
        sys.exit("ERROR: no task dirs — pass -t/--task-dir or --suite-dir")
    return deduped


def _make_task(task_dir: str) -> TaskState:
    name = Path(task_dir).name
    return TaskState(
        task_id=name,
        repo="probe",
        commit_sha="probe",
        description="smoke probe",
        base_sha="probe",
        merge_base_sha="probe",
        stage=Stage.ORACLE_CHECKED,
        task_dir=task_dir,
    )


def _dry_run(task_dirs: list[str], agent: str, model: str, effort: str) -> None:
    for task_dir in task_dirs:
        cmd, env_overrides = _build_smoke_cmd(
            task_dir, agent, model, effort, f"smoke-probe-{Path(task_dir).name}"
        )
        print(f"\n=== {task_dir} ===")
        print("harbor argv:")
        print("  " + " ".join(cmd))
        print("env overrides:", env_overrides or "(none — keys read from os.environ/.env)")


async def _run(task_dirs: list[str], agent: str, model: str, effort: str, concurrency: int) -> int:
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    async def _one(task_dir: str) -> None:
        async with sem:
            diag = await _run_smoke_async(
                _make_task(task_dir), model, "probe", agent=agent, reasoning_effort=effort
            )
            results[task_dir] = diag

    await asyncio.gather(*[_one(d) for d in task_dirs])

    print("\n" + "=" * 60)
    print("SMOKE PROBE RESULTS")
    print("=" * 60)
    failed = 0
    for task_dir, diag in results.items():
        if diag.get("infra_failure") or diag.get("no_trial") or diag.get("timeout"):
            failed += 1
            kind = (
                "TIMEOUT"
                if diag.get("timeout")
                else ("INFRA_FAILURE" if diag.get("infra_failure") else "NO_TRIAL")
            )
            print(f"  {Path(task_dir).name}: {kind}")
            continue
        print(
            f"  {Path(task_dir).name}: {diag.get('score_detail', '?')} "
            f"(reward={diag.get('reward', '?')}, model={diag.get('model', '?')}) "
            f"trial={diag.get('trial_dir', '?')}"
        )
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--task-dir", action="append", help="Built task dir (repeatable)")
    p.add_argument("--suite-dir", help="Probe every */task.toml under this directory")
    p.add_argument("--profile", help="TOML profile (sets smoke_agent/smoke_model defaults)")
    p.add_argument("--agent", help="Harbor agent (default: profile/config SMOKE_AGENT)")
    p.add_argument("--model", help="Model slug (default: profile/config SMOKE_MODEL)")
    p.add_argument(
        "--reasoning-effort",
        help="Reasoning effort (default: profile/config; empty falls back to reasoning_defaults)",
    )
    p.add_argument("--concurrency", type=int, default=2, help="Parallel trials (default: 2)")
    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved harbor argv + env, don't run Docker"
    )
    args = p.parse_args()

    _cfg._load_env()
    if args.profile:
        PipelineProfile.from_toml(args.profile).apply()

    agent = args.agent or _cfg.SMOKE_AGENT
    model = args.model or _cfg.SMOKE_MODEL
    effort = args.reasoning_effort if args.reasoning_effort is not None else _cfg.SMOKE_REASONING_EFFORT
    task_dirs = _discover_task_dirs(args)

    print(f"agent={agent} model={model} reasoning_effort={effort or '(reasoning_defaults)'}")
    print(f"tasks: {len(task_dirs)}")

    if args.dry_run:
        _dry_run(task_dirs, agent, model, effort)
        return

    rc = asyncio.run(_run(task_dirs, agent, model, effort, args.concurrency))
    sys.exit(rc)


if __name__ == "__main__":
    main()
