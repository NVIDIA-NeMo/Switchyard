# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor smoke test runner and trial diagnostics."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from craft_taskgen.config import TaskState

# Repo root (…/craft-taskgen) — sibling resolution for the codex model catalog.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Filtered single-model codex catalogs are cached here (cwd-relative, same as
# the jobs/ dir harbor writes to). Reused across runs; cheap to regenerate.
_CODEX_CATALOG_DIR = Path("jobs") / ".codex-catalogs"


def _agent_version(agent: str) -> str | None:
    """Pinned CLI version for a Harbor agent, or None if not version-pinned."""
    from craft_taskgen.adapters._docker import CLAUDE_CODE_VERSION, CODEX_VERSION, OPENCODE_VERSION

    return {
        "claude-code": CLAUDE_CODE_VERSION,
        "codex": CODEX_VERSION,
        "opencode": OPENCODE_VERSION,
    }.get(agent)


def _filtered_codex_catalog(model: str) -> str | None:
    """Write a single-model codex catalog for `model`; return its path (or None).

    Codex only emits reasoning tokens for slugs it recognizes as
    reasoning-capable; gateway slugs aren't in its built-in catalog, so harbor's
    codex patch reads CODEX_MODEL_CATALOG_JSON and injects the catalog. Harbor
    inlines the file content into a docker-exec argv, so we filter to one model
    to stay under Linux's per-arg MAX_ARG_STRLEN (the full catalog is ~150 KB).
    Returns None when the source catalog has no row for `model` — codex then
    runs without reasoning (matches scripts/run-baselines.sh behavior).
    """
    src = _REPO_ROOT / "patches" / "codex-model-catalog.json"
    if not src.is_file():
        return None
    data = json.loads(src.read_text())
    matches = [m for m in data.get("models", []) if m.get("slug") == model]
    if not matches:
        return None
    _CODEX_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out = _CODEX_CATALOG_DIR / f"{model.replace('/', '-')}.json"
    out.write_text(json.dumps({"models": matches}, indent=2))
    return str(out)


def _build_smoke_cmd(
    task_dir: str,
    agent: str,
    model: str,
    reasoning_effort: str,
    job_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Build the ``harbor run`` argv + env overrides for an agent smoke trial.

    Per-agent wiring mirrors scripts/run-baselines.sh (the canonical launcher):
    version pin, codex reasoning catalog, reasoning_effort kwarg, and
    claude-code's plan-mode tool restriction. API keys/base URLs are read by
    harbor's agents from the host environment (loaded from .env), so they are
    not passed here. Returns (cmd, env_overrides); env_overrides is layered on
    top of os.environ by the caller.
    """
    cmd = [
        ".venv/bin/harbor",
        "run",
        # Auto-confirm Harbor's host-env-access prompt (added in the 331dcba3
        # pin). The smoke step runs harbor as a non-interactive subprocess, so
        # without this the prompt receives no input and harbor aborts before
        # creating the job dir (surfacing as a context-free NO_TRIAL). Mirrors
        # the --yes that scripts/run-baselines.sh already passes.
        "--yes",
        "-p",
        task_dir,
        "-a",
        agent,
        "-m",
        model,
        "--job-name",
        job_name,
    ]
    env_overrides: dict[str, str] = {}

    version = _agent_version(agent)
    if version:
        cmd += ["--agent-kwarg", f"version={version}"]

    if agent == "claude-code":
        cmd += ["--agent-kwarg", "disallowed_tools=EnterPlanMode,ExitPlanMode"]

    if agent == "codex":
        catalog = _filtered_codex_catalog(model)
        if catalog:
            env_overrides["CODEX_MODEL_CATALOG_JSON"] = catalog
        else:
            print(f"    -> WARNING: no codex catalog row for {model}; reasoning will be absent")

    effort = reasoning_effort
    if not effort:
        from craft_taskgen.baselines.reasoning_defaults import effort_for

        effort = effort_for(agent, model) or ""
    if effort:
        cmd += ["--agent-kwarg", f"reasoning_effort={effort}"]

    return cmd, env_overrides


def _find_trial_in_job(job_dir: str) -> str | None:
    """Find the trial subdirectory within a job directory."""
    if not os.path.isdir(job_dir):
        return None
    for entry in os.listdir(job_dir):
        trial_path = os.path.join(job_dir, entry)
        if os.path.isdir(trial_path):
            return trial_path
    return None


def _read_trial_diagnostics(trial_dir: str) -> dict:
    """Read diagnostic info from a trial directory. Returns dict with key metrics."""
    diag: dict = {}

    # reward.json — score and test counts
    reward_path = os.path.join(trial_dir, "verifier", "reward.json")
    if os.path.isfile(reward_path):
        with open(reward_path) as f:
            reward_data = json.load(f)
        diag["reward"] = reward_data.get("reward", 0.0)
        f2p_passed = reward_data.get("f2p_passed", 0)
        p2p_passed = reward_data.get("p2p_passed", 0)
        f2p_total = reward_data.get("f2p_total", 0)
        p2p_total = reward_data.get("p2p_total", 0)
        diag["ref_passed"] = f2p_passed + p2p_passed
        diag["ref_total"] = f2p_total + p2p_total
        diag["ref_failed"] = diag["ref_total"] - diag["ref_passed"]
        diag["f2p_passed"] = f2p_passed
        diag["f2p_total"] = f2p_total
        diag["p2p_passed"] = p2p_passed
        diag["p2p_total"] = p2p_total

    # result.json — agent execution metadata (Harbor ATIF format)
    result_path = os.path.join(trial_dir, "result.json")
    if os.path.isfile(result_path):
        with open(result_path) as f:
            result_data = json.load(f)
        agent_result = result_data.get("agent_result") or {}
        diag["input_tokens"] = agent_result.get("n_input_tokens") or 0
        diag["output_tokens"] = agent_result.get("n_output_tokens") or 0
        diag["cache_tokens"] = agent_result.get("n_cache_tokens") or 0
        # Duration from timestamps
        agent_exec = result_data.get("agent_execution") or {}
        started = agent_exec.get("started_at", "")
        finished = agent_exec.get("finished_at", "")
        if started and finished:
            from datetime import datetime as _dt

            try:
                t0 = _dt.fromisoformat(started.replace("Z", "+00:00"))
                t1 = _dt.fromisoformat(finished.replace("Z", "+00:00"))
                diag["duration_s"] = (t1 - t0).total_seconds()
            except ValueError:
                diag["duration_s"] = 0
        config = result_data.get("config") or {}
        agent_cfg = config.get("agent") or {}
        diag["model"] = agent_cfg.get("model_name", "")
        diag["exception_info"] = result_data.get("exception_info")

    # Detect infra failure: agent produced no output tokens
    diag["infra_failure"] = diag.get("output_tokens", 0) == 0 and diag.get("exception_info") is not None

    # verify_full_output.txt — first 500 chars for diagnostic
    full_output_path = os.path.join(trial_dir, "verifier", "verify_full_output.txt")
    if os.path.isfile(full_output_path):
        with open(full_output_path) as f:
            diag["verify_output_head"] = f.read()[:500]

    # exception.txt — agent crash info
    exception_path = os.path.join(trial_dir, "exception.txt")
    if os.path.isfile(exception_path):
        with open(exception_path) as f:
            diag["exception"] = f.read()[:500]
        diag["infra_failure"] = True

    return diag


async def _run_smoke_async(
    task: TaskState,
    model: str,
    label: str,
    *,
    agent: str = "claude-code",
    reasoning_effort: str = "",
) -> dict:
    """Async smoke test with deterministic --job-name to avoid race conditions."""
    task_name = Path(task.task_dir).name if task.task_dir else task.task_id
    # Always unique job name — prevents Harbor from reusing stale job dirs
    # and ensures harbor-lab only sees one trial per analysis
    from datetime import datetime as _dt

    ts = _dt.now().strftime("%H%M%S")
    job_name = f"smoke-{label.lower()}-{task_name}-{ts}"

    cmd, env_overrides = _build_smoke_cmd(task.task_dir, agent, model, reasoning_effort, job_name)
    print(f"    -> {label} smoke: agent={agent} model={model}")
    env = {**os.environ, **env_overrides} if env_overrides else None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await asyncio.wait_for(proc.communicate(), timeout=1800)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        print(f"    -> {label} TIMEOUT (>30 min)")
        return {"timeout": True}

    # Look up by job name instead of "latest"
    job_dir = os.path.join("jobs", job_name)
    trial_dir = _find_trial_in_job(job_dir) if os.path.isdir(job_dir) else None

    if not trial_dir:
        print(f"    -> {label}: Could not find trial directory in {job_dir}")
        return {"no_trial": True}

    diag = _read_trial_diagnostics(trial_dir)

    score = f"{diag.get('ref_passed', '?')}/{diag.get('ref_total', '?')}"
    f2p_str = f"F2P {diag.get('f2p_passed', '?')}/{diag.get('f2p_total', '?')}"
    p2p_str = f"P2P {diag.get('p2p_passed', '?')}/{diag.get('p2p_total', '?')}"
    score_detail = f"{f2p_str}, {p2p_str}"
    tokens = f"{diag.get('input_tokens') or 0:,}in/{diag.get('output_tokens') or 0:,}out"
    duration = f"{diag.get('duration_s') or 0:.0f}s"
    print(f"    -> {label}: {score_detail} (reward={diag.get('reward', '?')}) | {tokens} | {duration}")

    if diag.get("infra_failure"):
        print("    -> WARNING: Infra failure detected")
        if diag.get("exception"):
            print(f"       Exception: {diag['exception'][:150]}")

    diag["trial_dir"] = trial_dir
    diag["score"] = score
    diag["score_detail"] = score_detail
    return diag
