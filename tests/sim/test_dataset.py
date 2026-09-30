# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohort and repeat validation for recorded routing evaluation."""

from dataclasses import replace

import pytest

from switchyard.sim.dataset import Dataset
from switchyard.sim.models import HarborRun, LoadIssue, Outcome, Trial


def trial(target="fast", task="task", attempt="1", **kwargs):
    return Trial(
        task,
        attempt,
        target,
        ({"role": "user", "content": [{"type": "text", "text": task}]},),
        **kwargs,
    )


def test_repeats_are_task_means_and_never_arbitrarily_paired():
    dataset = Dataset.from_runs(
        {
            "fast": HarborRun(
                (trial(reward=1, cost_usd=2), trial(attempt="2", reward=0, cost_usd=4))
            ),
            "strong": HarborRun(
                (trial("strong", attempt="unrelated-id", reward=0.8, cost_usd=10),)
            ),
        },
        input_target="fast",
    )
    task = dataset.tasks[0]
    assert task.outcomes["fast"].reward == 0.5
    assert task.outcomes["fast"].cost_usd == 3
    assert task.outcomes["strong"].reward == 0.8
    assert task.outcomes["fast"].trials == 2


def test_incomplete_repeat_measurements_stay_unknown():
    outcome = Outcome.from_trials((trial(reward=0, cost_usd=0), trial(attempt="2", reward=None)))
    assert outcome.reward is None
    assert outcome.cost_usd is None
    assert outcome.reward_trials == outcome.cost_trials == 1
    assert outcome.trials == 2


def test_missing_target_coverage_requires_explicit_intersection():
    runs = {
        "fast": HarborRun((trial(), trial(task="extra"))),
        "strong": HarborRun((trial("strong"),)),
    }
    with pytest.raises(ValueError, match="incomplete cohort"):
        Dataset.from_runs(runs, input_target="fast")
    dataset = Dataset.from_runs(runs, input_target="fast", intersection=True)
    assert len(dataset.tasks) == 1
    assert dataset.coverage["tasks_seen"] == 2
    assert dataset.coverage["excluded_task_ids"] == ["extra"]


def test_invalid_repeat_excludes_whole_task_instead_of_improving_its_mean():
    runs = {
        "fast": HarborRun(
            (trial(), trial(task="safe")), (LoadIssue("broken", "bad input", "task"),)
        )
    }
    dataset = Dataset.from_runs(runs, input_target="fast", intersection=True)
    assert [task.task_id for task in dataset.tasks] == ["safe"]
    assert dataset.coverage["excluded_task_ids"] == ["task"]
    assert len(dataset.coverage["input_issues"]) == 1


def test_unidentified_rejected_repeat_cannot_leave_a_favorable_mean():
    run = HarborRun((trial(reward=1),), (LoadIssue("missing/result.json", "cannot read"),))
    with pytest.raises(ValueError, match="no task identity"):
        Dataset.from_runs({"fast": run}, input_target="fast", intersection=True)


def test_cost_source_mixture_is_visible_in_coverage():
    run = HarborRun(
        (
            trial(cost_usd=1, cost_source="result"),
            trial(task="b", cost_usd=2, cost_source="trajectory"),
            trial(task="c"),
        )
    )
    cohort = Dataset.from_runs({"fast": run}, input_target="fast")
    assert cohort.coverage["cost_sources_by_target"] == {
        "fast": {"result": 1, "trajectory": 1, "unknown": 1}
    }


def test_huge_json_numbers_are_rejected_as_invalid_measurements():
    with pytest.raises(ValueError, match="reward must be finite"):
        trial(reward=10**400)


def test_checksums_reject_changed_tasks_even_when_names_match():
    with pytest.raises(ValueError, match="conflicting task checksums"):
        Dataset.from_runs(
            {
                "a": HarborRun((trial("a", task_checksum="old"),)),
                "b": HarborRun((trial("b", task_checksum="new"),)),
            },
            input_target="a",
        )


@pytest.mark.parametrize("checksum", ["", " \t\n", 1, True, [], {}])
def test_invalid_task_checksums_are_rejected(checksum):
    with pytest.raises(ValueError, match="task_checksum must be a non-empty string or None"):
        trial(task_checksum=checksum)


def test_agent_wrappers_may_differ_and_input_target_is_explicit():
    fast = trial(task_checksum="same")
    strong = replace(
        trial("strong", task_checksum="same"),
        messages=(
            {"role": "system", "content": [{"type": "text", "text": "different agent"}]},
            *fast.messages,
        ),
    )
    dataset = Dataset.from_runs(
        {"fast": HarborRun((fast,)), "strong": HarborRun((strong,))}, input_target="strong"
    )
    assert dataset.tasks[0].messages == strong.messages


def test_missing_checksums_require_matching_task_instructions():
    other = replace(trial("strong"), messages=trial(task="different").messages)
    with pytest.raises(ValueError, match="conflicting inputs"):
        Dataset.from_runs(
            {"fast": HarborRun((trial(),)), "strong": HarborRun((other,))}, input_target="fast"
        )


def test_missing_checksums_compare_all_initial_user_constraints():
    first = trial()
    second = replace(
        trial("strong"),
        messages=(
            {"role": "user", "content": [{"type": "text", "text": "additional constraint"}]},
            *first.messages,
        ),
    )
    with pytest.raises(ValueError, match="conflicting inputs"):
        Dataset.from_runs(
            {"fast": HarborRun((first,)), "strong": HarborRun((second,))}, input_target="fast"
        )


def test_duplicate_trial_and_incorrect_target_are_rejected():
    with pytest.raises(ValueError, match="duplicate trial"):
        Dataset.from_runs({"fast": HarborRun((trial(), trial()))}, input_target="fast")
    with pytest.raises(ValueError, match="does not match"):
        Dataset.from_runs({"strong": HarborRun((trial(),))}, input_target="strong")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": float("nan")},
        {"cost_usd": -1},
        {"duration_seconds": float("inf")},
        {"reward": True},
    ],
)
def test_invalid_measurements_are_rejected(kwargs):
    with pytest.raises(ValueError):
        trial(**kwargs)
