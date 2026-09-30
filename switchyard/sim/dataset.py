# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match recorded targets without choosing favorable trials or hiding exclusions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .models import Outcome, Run, Task, Trial


@dataclass(frozen=True)
class Dataset:
    """A validated cohort with one decision per task and equal weight per task."""

    tasks: tuple[Task, ...]
    targets: tuple[str, ...]
    coverage: Mapping[str, object]

    @classmethod
    def from_runs(
        cls,
        runs: Mapping[str, Run],
        *,
        input_target: str,
        intersection: bool = False,
    ) -> Dataset:
        """Pair targets by task ID and checksum, averaging all recorded repeats.

        ``input_target`` supplies the routing input because agent scaffolding may
        differ between baseline runs. ``intersection`` explicitly excludes tasks
        missing or invalid in any run; exclusions remain in ``coverage``.
        """
        if not runs or input_target not in runs:
            raise ValueError("input_target must name one of the supplied runs")
        grouped: dict[str, dict[str, list[Trial]]] = {}
        invalid: set[str] = set()
        issues = []
        for target, run in runs.items():
            groups: dict[str, list[Trial]] = defaultdict(list)
            seen: set[tuple[str, str]] = set()
            for trial in run.trials:
                if trial.target != target:
                    raise ValueError(f"trial target {trial.target!r} does not match run {target!r}")
                key = trial.task_id, trial.trial_id
                if key in seen:
                    raise ValueError(f"duplicate trial {trial.trial_id!r} for {trial.task_id!r}")
                seen.add(key)
                groups[trial.task_id].append(trial)
            grouped[target] = groups
            for issue in run.issues:
                if issue.task_id is None:
                    raise ValueError(
                        f"input issue has no task identity: {issue.source}; "
                        "repair or identify the rejected trial before pairing"
                    )
                issues.append(
                    {
                        "target": target,
                        "source": issue.source,
                        "task_id": issue.task_id,
                        "message": issue.message,
                    }
                )
                if issue.task_id is not None:
                    invalid.add(issue.task_id)
        task_sets = [set(groups) for groups in grouped.values()]
        union = set.union(*task_sets) | invalid
        common = set.intersection(*task_sets) - invalid
        excluded = sorted(union - common)
        if not intersection and (excluded or issues):
            raise ValueError(
                f"incomplete cohort: {len(excluded)} tasks missing or invalid, "
                f"{len(issues)} input issues; use intersection=True to explicitly exclude them"
            )
        if not common:
            raise ValueError("no tasks have valid trials for every target")
        tasks = []
        for task_id in sorted(common):
            trials = {
                target: tuple(sorted(groups[task_id], key=lambda trial: trial.trial_id))
                for target, groups in grouped.items()
            }
            all_trials = [trial for values in trials.values() for trial in values]
            checksums = {trial.task_checksum for trial in all_trials if trial.task_checksum}
            if len(checksums) > 1:
                raise ValueError(f"conflicting task checksums for {task_id!r}")
            if not all(trial.task_checksum for trial in all_trials):
                instructions = {_instruction(trial) for trial in all_trials}
                if len(instructions) != 1:
                    raise ValueError(
                        f"task {task_id!r} has conflicting inputs without complete checksums"
                    )
            messages = trials[input_target][0].messages
            if not messages or not any(message.get("role") == "user" for message in messages):
                raise ValueError(f"task {task_id!r} has no user input")
            tasks.append(
                Task(
                    task_id,
                    messages,
                    {target: Outcome.from_trials(values) for target, values in trials.items()},
                    trials,
                )
            )
        return cls(
            tuple(tasks),
            tuple(runs),
            {
                "input_target": input_target,
                "intersection": intersection,
                "tasks_seen": len(union),
                "tasks_included": len(tasks),
                "tasks_by_target": {target: len(groups) for target, groups in grouped.items()},
                "excluded_task_ids": excluded,
                "input_issues": issues,
                "cost_sources_by_target": {
                    target: dict(Counter(trial.cost_source or "unknown" for trial in run.trials))
                    for target, run in runs.items()
                },
            },
        )


def _instruction(trial: Trial) -> str:
    """Require the same user input when a task checksum is unavailable."""
    contents = [
        message.get("content") for message in trial.messages if message.get("role") == "user"
    ]
    if not contents:
        raise ValueError(f"task {trial.task_id!r} has no user input")
    return json.dumps(contents, sort_keys=True, allow_nan=False)
