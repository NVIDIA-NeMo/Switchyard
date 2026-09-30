# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Harbor outcomes and the ATIF input visible before the first agent step."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .models import LoadIssue, Run, Trial
from .trajectory import _initial_messages, _task_messages


def load_harbor(
    path: str | Path,
    *,
    target: str,
    dataset: str | None = None,
    reward_key: str = "reward",
    on_error: Literal["raise", "record"] = "raise",
    task_inputs: Mapping[str, str] | None = None,
    cost_source: Literal["result", "trajectory"] = "result",
) -> Run:
    """Load a trial, a Harbor job, or a downloaded run containing ``jobs/``.

    ``target`` names the recorded model/agent configuration. An explicit dataset
    namespaces task names as ``dataset/task_name``; temporary task paths are never
    used as identities. Repeats remain separate trials.

    Only system/user messages before the first ATIF agent step become routing
    input. Missing trajectories or inputs require an explicit ``task_inputs``
    fallback keyed by task ID or task name. Raw session logs are not parsed.

    Reported cost comes from the preferred ``cost_source``, falling back to the
    other source when absent. These values can differ because the producers use
    different prices. No token pricing is inferred. Duration measures only the
    agent execution phase. Missing measurements remain unknown.

    By default any invalid trial raises. ``on_error='record'`` keeps rejected
    trial paths and reasons in ``issues`` so callers can account for exclusions.
    Processes one trajectory file at a time.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string")
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        raise ValueError("dataset must be a non-empty string")
    if not isinstance(reward_key, str) or not reward_key:
        raise ValueError("reward_key must be a non-empty string")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if cost_source not in ("result", "trajectory"):
        raise ValueError("cost_source must be 'result' or 'trajectory'")

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Harbor directory does not exist: {root}")
    paths, issues = _trial_paths(root)
    if issues and on_error == "raise":
        raise ValueError(f"{issues[0].source}: {issues[0].message}")
    if not paths and not issues:
        raise ValueError(f"No Harbor trials found under {root}")

    trials: list[Trial] = []
    trial_ids: set[str] = set()
    for result_path in paths:
        task_id = None
        try:
            result = _read_object(result_path)
            name = _string(result.get("task_name"), "task_name")
            task_id = f"{dataset}/{name}" if dataset else name
            trial = _trial(
                result_path, result, task_id, target, reward_key, task_inputs, cost_source
            )
            if trial.trial_id in trial_ids:
                raise ValueError("duplicate trial ID")
            trial_ids.add(trial.trial_id)
            trials.append(trial)
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else "cannot read trial artifact"
            if on_error == "raise":
                raise ValueError(f"{result_path}: {message}") from None
            issues.append(LoadIssue(source=str(result_path), message=message, task_id=task_id))
    return Run(tuple(trials), tuple(issues))


def _trial_paths(root: Path) -> tuple[list[Path], list[LoadIssue]]:
    """Inspect known Harbor layouts, excluding nested copies of trial metadata."""
    result_path = root / "result.json"
    if (root / "agent").is_dir():
        return [result_path], []
    unreadable_result = False
    summary = None
    if result_path.is_file():
        try:
            summary = _read_object(result_path)
            if "task_name" in summary:
                return [result_path], []
        except (OSError, ValueError):
            unreadable_result = True
    jobs = root / "jobs"
    roots = sorted(p for p in jobs.iterdir() if p.is_dir()) if jobs.is_dir() else [root]
    paths: list[Path] = []
    issues: list[LoadIssue] = []
    for job in roots:
        candidates = [
            trial / "result.json"
            for trial in sorted(job.iterdir())
            if trial.is_dir()
            and (
                (trial / "result.json").is_file()
                or (trial / "agent").is_dir()
                or (trial / "config.json").is_file()
            )
        ]
        paths.extend(candidates)
        job_result = job / "result.json"
        job_summary = summary if job == root else None
        if job != root and job_result.is_file():
            try:
                job_summary = _read_object(job_result)
            except (OSError, ValueError):
                pass
        expected = job_summary.get("n_total_trials") if job_summary is not None else None
        if type(expected) is int and expected >= 0 and expected != len(candidates):
            issues.append(
                LoadIssue(
                    str(job_result),
                    f"job trial count mismatch: expected {expected}, found {len(candidates)}",
                )
            )
    # A damaged job summary must not hide valid trial directories. Without
    # children, keep the unreadable artifact visible as a standalone trial issue.
    return paths or ([result_path] if unreadable_result else []), issues


def _read_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError(f"invalid JSON in {path.name}") from None
    return _object(value, path.name)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _optional_object(value: Any, field: str) -> dict[str, Any]:
    return {} if value is None else _object(value, field)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _trial(
    path: Path,
    result: dict[str, Any],
    task_id: str,
    target: str,
    reward_key: str,
    task_inputs: Mapping[str, str] | None,
    cost_source: str,
) -> Trial:
    trajectory_path = path.parent / "agent" / "trajectory.json"
    trajectory = _read_object(trajectory_path) if trajectory_path.is_file() else None
    fallback = None
    if task_inputs is not None:
        fallback = task_inputs.get(task_id, task_inputs.get(result["task_name"]))
    messages = (
        _initial_messages(trajectory, fallback)
        if trajectory is not None
        else _task_messages(fallback)
    )
    agent_result = _optional_object(result.get("agent_result"), "agent_result")
    metrics = _optional_object(
        trajectory.get("final_metrics") if trajectory is not None else None, "final_metrics"
    )
    verifier = _optional_object(result.get("verifier_result"), "verifier_result")
    rewards = _optional_object(verifier.get("rewards"), "rewards")
    agent = _optional_object(result.get("agent_info"), "agent_info")
    model = _optional_object(agent.get("model_info"), "model_info")
    error = _optional_object(result.get("exception_info"), "exception_info")
    checksum = result.get("task_checksum")
    if checksum is not None:
        checksum = _string(checksum, "task_checksum")

    costs = {
        "result": agent_result.get("cost_usd"),
        "trajectory": metrics.get("total_cost_usd"),
    }
    cost = costs[cost_source]
    if cost is None:
        cost_source = "trajectory" if cost_source == "result" else "result"
        cost = costs[cost_source]

    usage: dict[str, int | None] = {}
    for source, keys in (
        (agent_result, ("n_input_tokens", "n_output_tokens", "n_cache_tokens")),
        (metrics, ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens")),
    ):
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{key} must be a non-negative integer or null")
            usage[key] = value

    name = model.get("name")
    provider = model.get("provider")
    if name is not None:
        name = _string(name, "model_info.name")
        if provider:
            name = f"{_string(provider, 'model_info.provider')}/{name}"
    return Trial(
        task_id=task_id,
        trial_id=_string(
            result["id"] if result.get("id") is not None else path.parent.name, "trial ID"
        ),
        target=target,
        messages=messages,
        reward=rewards.get(reward_key),
        cost_usd=cost,
        cost_source=cost_source if cost is not None else None,
        duration_seconds=_duration(result),
        task_checksum=checksum,
        model=name,
        source=str(path),
        usage=usage,
        error=_string(error["exception_type"], "exception_type")
        if error.get("exception_type")
        else None,
    )


def _duration(result: dict[str, Any]) -> float | None:
    timing = _optional_object(result.get("agent_execution"), "agent_execution")
    start, finish = timing.get("started_at"), timing.get("finished_at")
    if start is None or finish is None:
        return None
    if not isinstance(start, str) or not isinstance(finish, str):
        raise ValueError("agent_execution timestamps must be strings")
    try:
        # Python 3.10's fromisoformat does not accept the UTC Z suffix.
        return (
            datetime.fromisoformat(finish.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except (ValueError, TypeError):
        raise ValueError("invalid or incompatible agent_execution timestamps") from None
