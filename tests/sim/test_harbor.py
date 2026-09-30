# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized Harbor fixtures covering native Claude and Codex ATIF layouts."""

import json
from pathlib import Path

import pytest

from switchyard.sim import Dataset
from switchyard.sim.harbor import load_harbor


def write_trial(
    root: Path,
    name: str = "trial-a",
    *,
    task: str = "benchmark/task-a",
    version: str = "ATIF-v1.7",
    codex: bool = False,
) -> Path:
    trial = root / name
    (trial / "agent").mkdir(parents=True)
    result = {
        "id": name,
        "task_name": task,
        "task_checksum": "checksum-a",
        "task_id": {"path": "/tmp/ephemeral-task"},
        "agent_info": {
            "name": "codex" if codex else "claude-code",
            "version": "0.154.0" if codex else "2.1.216",
            "model_info": {"name": "model-a", "provider": "provider"},
        },
        "agent_result": {
            "n_input_tokens": 100,
            "n_output_tokens": 20,
            "n_cache_tokens": 80,
            "cost_usd": 0.25,
        },
        "verifier_result": {"rewards": {"reward": 0.75, "custom": 0.5}},
        "started_at": "2026-09-01T01:00:00Z",
        "finished_at": "2026-09-01T02:00:00Z",
        "agent_execution": {
            "started_at": "2026-09-01T01:10:00Z",
            "finished_at": "2026-09-01T01:10:12.500Z",
        },
    }
    steps = []
    if codex:
        steps.extend(
            [
                {"source": "system", "message": "Agent instructions."},
                {"source": "user", "message": "Working directory: /workspace"},
            ]
        )
    steps.extend(
        [
            {"source": "user", "message": "Implement the requested feature."},
            {"source": "agent", "message": "FUTURE_ASSISTANT_LEAK"},
            {"source": "user", "message": "FUTURE_USER_LEAK"},
        ]
    )
    trajectory = {
        "schema_version": version,
        "session_id": name,
        "agent": {"name": "codex" if codex else "claude-code", "version": "fixture"},
        "steps": [dict(step_id=i, **step) for i, step in enumerate(steps, start=1)],
        "final_metrics": {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "total_cached_tokens": 80,
            "total_cost_usd": 0.1,
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    return trial


def update(path: Path, **values: object) -> None:
    row = json.loads(path.read_text())
    row.update(values)
    path.write_text(json.dumps(row))


@pytest.mark.parametrize(
    "version,codex", [("ATIF-v1.7", False), ("ATIF-v1.7", True), ("ATIF-v1.5", True)]
)
def test_initial_messages_include_codex_context_without_future_leakage(
    tmp_path: Path, version: str, codex: bool
) -> None:
    write_trial(tmp_path, version=version, codex=codex)
    run = load_harbor(tmp_path, target="baseline", dataset="suite")
    trial = run.trials[0]
    assert run.issues == ()
    assert trial.task_id == "suite/benchmark/task-a"
    assert trial.task_checksum == "checksum-a"
    assert [message["role"] for message in trial.messages] == (
        ["system", "user", "user"] if codex else ["user"]
    )
    assert trial.messages[-1]["content"] == [
        {"type": "text", "text": "Implement the requested feature."}
    ]
    assert "FUTURE" not in json.dumps(trial.messages)
    assert trial.reward == 0.75
    assert trial.cost_usd == 0.25
    assert trial.cost_source == "result"
    assert trial.duration_seconds == 12.5
    assert trial.model == "provider/model-a"
    assert trial.usage["total_prompt_tokens"] == 100
    assert trial.usage["n_cache_tokens"] == 80


def test_discovery_ignores_nested_metadata_and_accepts_download_layout(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "run-a"
    trial = write_trial(job)
    (trial / "_scaled_evals").mkdir()
    (trial / "_scaled_evals" / "result.json").write_text((trial / "result.json").read_text())
    (job / "result.json").write_text(json.dumps({"n_total_trials": 1}))
    for root in (tmp_path, job, trial):
        assert len(load_harbor(root, target="baseline").trials) == 1


@pytest.mark.parametrize("unreadable", [False, True])
def test_invalid_job_summary_does_not_hide_valid_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unreadable: bool
) -> None:
    write_trial(tmp_path)
    summary = tmp_path / "result.json"
    summary.write_text("{")
    if unreadable:
        original_open = Path.open

        def open_artifact(path: Path, *args, **kwargs):
            if path == summary:
                raise PermissionError("cannot read summary")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_artifact)

    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == 1
    assert run.issues == ()


@pytest.mark.parametrize("has_agent_directory", [False, True])
def test_standalone_malformed_result_is_still_a_trial_issue(
    tmp_path: Path, has_agent_directory: bool
) -> None:
    if has_agent_directory:
        (tmp_path / "agent").mkdir()
    result_path = tmp_path / "result.json"
    result_path.write_text("{")

    with pytest.raises(ValueError, match="invalid JSON in result.json"):
        load_harbor(tmp_path, target="baseline")
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert run.trials == ()
    assert len(run.issues) == 1
    assert run.issues[0].source == str(result_path)
    assert run.issues[0].task_id is None


def test_repeated_tasks_are_preserved_as_distinct_trials(tmp_path: Path) -> None:
    write_trial(tmp_path, "repeat-1")
    write_trial(tmp_path, "repeat-2")
    trials = load_harbor(tmp_path, target="baseline").trials
    assert len(trials) == 2
    assert trials[0].task_id == trials[1].task_id
    assert trials[0].trial_id != trials[1].trial_id


def test_distinct_trial_directories_can_share_result_metadata(tmp_path: Path) -> None:
    first = write_trial(tmp_path, "first")
    second = write_trial(tmp_path, "second")
    update(first / "result.json", id=None)
    (second / "result.json").unlink()
    (second / "result.json").symlink_to(first / "result.json")
    run = load_harbor(tmp_path, target="baseline")
    assert run.issues == ()
    assert [trial.trial_id for trial in run.trials] == ["first", "second"]


@pytest.mark.parametrize("layout", ["job", "download"])
@pytest.mark.parametrize("id_mode", ["absent", "null"])
def test_trial_directory_alias_cannot_bias_repeat_outcomes(
    tmp_path: Path, layout: str, id_mode: str
) -> None:
    job = tmp_path / "jobs" / "run" if layout == "download" else tmp_path
    for name, reward in (("repeat-1", 1), ("repeat-2", 0)):
        path = write_trial(job, name) / "result.json"
        result = json.loads(path.read_text())
        result.pop("id")
        if id_mode == "null":
            result["id"] = None
        result["verifier_result"]["rewards"]["reward"] = reward
        path.write_text(json.dumps(result))
    write_trial(job, "clean", task="unaffected")
    run = load_harbor(tmp_path, target="baseline", dataset="suite")
    dataset = Dataset.from_runs({"baseline": run}, input_target="baseline")
    assert dataset.tasks[0].outcomes["baseline"].reward == 0.5
    alias = job / "repeat-1-alias"
    alias.symlink_to(job / "repeat-1", target_is_directory=True)

    with pytest.raises(ValueError, match="duplicate trial directory"):
        load_harbor(tmp_path, target="baseline", dataset="suite")
    run = load_harbor(tmp_path, target="baseline", dataset="suite", on_error="record")
    assert len(run.trials) == 3
    (issue,) = run.issues
    assert issue.source == str(alias / "result.json")
    assert issue.task_id == "suite/benchmark/task-a"
    with pytest.raises(ValueError, match="incomplete cohort"):
        Dataset.from_runs({"baseline": run}, input_target="baseline")
    dataset = Dataset.from_runs({"baseline": run}, input_target="baseline", intersection=True)
    assert [task.task_id for task in dataset.tasks] == ["suite/unaffected"]
    assert dataset.coverage["excluded_task_ids"] == [issue.task_id]


@pytest.mark.parametrize("layout", ["job", "download"])
@pytest.mark.parametrize("remaining", [0, 1])
def test_missing_repeat_directory_is_not_silently_dropped(
    tmp_path: Path, layout: str, remaining: int
) -> None:
    job = tmp_path / "jobs" / "run" if layout == "download" else tmp_path
    job.mkdir(parents=True, exist_ok=True)
    if remaining:
        write_trial(job, "retained")
    summary = job / "result.json"
    summary.write_text(json.dumps({"n_total_trials": 2}))

    with pytest.raises(ValueError, match=f"expected 2, found {remaining}"):
        load_harbor(tmp_path, target="baseline")
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == remaining
    assert len(run.issues) == 1
    assert run.issues[0].source == str(summary)
    assert run.issues[0].task_id is None
    with pytest.raises(ValueError, match="no task identity"):
        Dataset.from_runs({"baseline": run}, input_target="baseline", intersection=True)


def test_job_counts_are_checked_separately_including_empty_jobs(tmp_path: Path) -> None:
    for name, count in (("retained", 0), ("empty", 1)):
        job = tmp_path / "jobs" / name
        job.mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({"n_total_trials": count}))
    write_trial(tmp_path / "jobs" / "retained")

    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == 1
    assert {issue.message for issue in run.issues} == {
        "job trial count mismatch: expected 0, found 1",
        "job trial count mismatch: expected 1, found 0",
    }


def test_job_count_includes_rejected_trial_directories(tmp_path: Path) -> None:
    invalid = write_trial(tmp_path, "invalid")
    (invalid / "result.json").unlink()
    write_trial(tmp_path, "valid")
    (tmp_path / "result.json").write_text(json.dumps({"n_total_trials": 2}))

    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == len(run.issues) == 1
    assert run.issues[0].source == str(invalid / "result.json")


@pytest.mark.parametrize("count", [None, True, -1, "2"])
def test_unusable_job_count_does_not_hide_valid_trials(tmp_path: Path, count: object) -> None:
    write_trial(tmp_path)
    (tmp_path / "result.json").write_text(json.dumps({"n_total_trials": count}))
    run = load_harbor(tmp_path, target="baseline")
    assert len(run.trials) == 1
    assert run.issues == ()


def test_missing_measurements_remain_unknown_and_preserve_exception_type(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    update(
        trial / "result.json",
        agent_result=None,
        verifier_result=None,
        agent_execution=None,
        exception_info={
            "exception_type": "VerifierTimeoutError",
            "exception_message": "PRIVATE_TRACE",
        },
    )
    update(trial / "agent" / "trajectory.json", final_metrics=None)
    row = load_harbor(tmp_path, target="baseline").trials[0]
    assert row.reward is None
    assert row.cost_usd is None
    assert row.cost_source is None
    assert row.duration_seconds is None
    assert row.usage == {}
    assert row.error == "VerifierTimeoutError"
    assert "PRIVATE_TRACE" not in repr(row)


def test_explicit_zero_measurements_are_not_replaced_by_fallbacks(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    update(
        trial / "result.json",
        agent_result={"cost_usd": 0},
        verifier_result={"rewards": {"reward": 0}},
    )
    row = load_harbor(tmp_path, target="baseline").trials[0]
    assert row.reward == 0
    assert row.cost_usd == 0
    assert row.cost_source == "result"


def test_reward_key_and_cost_preference_are_explicit(tmp_path: Path) -> None:
    write_trial(tmp_path)
    row = load_harbor(
        tmp_path, target="baseline", reward_key="custom", cost_source="trajectory"
    ).trials[0]
    assert row.reward == 0.5
    assert row.cost_usd == 0.1
    assert row.cost_source == "trajectory"


def test_absent_preferred_cost_falls_back_with_provenance(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    update(trial / "result.json", agent_result={"cost_usd": None})
    row = load_harbor(tmp_path, target="baseline").trials[0]
    assert row.cost_usd == 0.1
    assert row.cost_source == "trajectory"


@pytest.mark.parametrize("key", ["benchmark/task-a", "suite/benchmark/task-a"])
def test_missing_trajectory_requires_explicit_task_input(tmp_path: Path, key: str) -> None:
    trial = write_trial(tmp_path)
    (trial / "agent" / "trajectory.json").unlink()
    with pytest.raises(ValueError, match="provide task_inputs explicitly"):
        load_harbor(tmp_path, target="baseline", dataset="suite")
    row = load_harbor(
        tmp_path, target="baseline", dataset="suite", task_inputs={key: "Canonical task."}
    ).trials[0]
    assert row.messages == (
        {"role": "user", "content": [{"type": "text", "text": "Canonical task."}]},
    )


def test_malformed_trajectory_records_safe_error_and_task_identity(tmp_path: Path) -> None:
    invalid = write_trial(tmp_path, "invalid")
    write_trial(tmp_path, "valid", task="benchmark/task-b")
    (invalid / "agent" / "trajectory.json").write_text('{"PRIVATE_TRACE":broken')
    with pytest.raises(ValueError, match="invalid JSON in trajectory.json") as error:
        load_harbor(tmp_path, target="baseline")
    assert "PRIVATE_TRACE" not in str(error.value)
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == 1
    assert len(run.issues) == 1
    assert run.issues[0].task_id == "benchmark/task-a"
    assert run.issues[0].source.endswith("invalid/result.json")
    assert "PRIVATE_TRACE" not in run.issues[0].message


@pytest.mark.parametrize("artifact", ["result.json", "agent/trajectory.json"])
def test_excessive_json_nesting_records_issue_without_losing_valid_trials(
    tmp_path: Path, artifact: str
) -> None:
    invalid = write_trial(tmp_path, "invalid")
    write_trial(tmp_path, "valid", task="benchmark/task-b")
    path = invalid / artifact
    nested = "[" * 10_000 + '"PRIVATE_TRACE"' + "]" * 10_000
    path.write_text(path.read_text()[:-1] + ', "unused":' + nested + "}")

    with pytest.raises(ValueError, match=f"invalid JSON in {path.name}"):
        load_harbor(tmp_path, target="baseline", dataset="suite")
    run = load_harbor(tmp_path, target="baseline", dataset="suite", on_error="record")
    assert [trial.task_id for trial in run.trials] == ["suite/benchmark/task-b"]
    assert len(run.issues) == 1
    issue = run.issues[0]
    assert issue.message == f"invalid JSON in {path.name}"
    assert issue.source == str(invalid / "result.json")
    if artifact == "result.json":
        assert issue.task_id is None
        with pytest.raises(ValueError, match="no task identity"):
            Dataset.from_runs({"baseline": run}, input_target="baseline", intersection=True)
    else:
        assert issue.task_id == "suite/benchmark/task-a"
        dataset = Dataset.from_runs({"baseline": run}, input_target="baseline", intersection=True)
        assert dataset.coverage["excluded_task_ids"] == [issue.task_id]


@pytest.mark.parametrize("layout", ["job", "download"])
def test_excessive_json_nesting_in_summary_keeps_discoverable_trials(
    tmp_path: Path, layout: str
) -> None:
    job = tmp_path / "jobs" / "run" if layout == "download" else tmp_path
    write_trial(job)
    nested = "[" * 10_000 + "0" + "]" * 10_000
    (job / "result.json").write_text('{"unused":' + nested + "}")
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert [trial.task_id for trial in run.trials] == ["benchmark/task-a"]
    assert run.issues == ()


def test_trial_without_result_is_an_issue_not_an_invisible_exclusion(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    (trial / "result.json").unlink()
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert not run.trials
    assert len(run.issues) == 1
    assert run.issues[0].task_id is None


def test_duplicate_trial_ids_are_rejected(tmp_path: Path) -> None:
    write_trial(tmp_path, "first")
    second = write_trial(tmp_path, "second")
    update(second / "result.json", id="first")
    with pytest.raises(ValueError, match="duplicate trial ID"):
        load_harbor(tmp_path, target="baseline")


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, "1"])
def test_invalid_reward_is_rejected(tmp_path: Path, reward: object) -> None:
    trial = write_trial(tmp_path)
    update(trial / "result.json", verifier_result={"rewards": {"reward": reward}})
    with pytest.raises(ValueError, match="reward must be finite"):
        load_harbor(tmp_path, target="baseline")


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_usd", -1),
        ("cost_usd", float("nan")),
        ("n_input_tokens", -1),
        ("n_cache_tokens", True),
    ],
)
def test_invalid_accounting_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    trial = write_trial(tmp_path)
    update(trial / "result.json", agent_result={field: value})
    with pytest.raises(ValueError, match="must be"):
        load_harbor(tmp_path, target="baseline")


@pytest.mark.parametrize("finish", ["2026-09-01T00:00:00Z", "invalid", "2026-09-01T01:10:13"])
def test_invalid_or_negative_agent_duration_is_rejected(tmp_path: Path, finish: str) -> None:
    trial = write_trial(tmp_path)
    update(
        trial / "result.json",
        agent_execution={"started_at": "2026-09-01T01:10:00Z", "finished_at": finish},
    )
    with pytest.raises(ValueError, match="duration_seconds|timestamps"):
        load_harbor(tmp_path, target="baseline")


def test_text_content_blocks_are_normalized_without_private_metadata(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    update(
        trial / "agent" / "trajectory.json",
        steps=[
            {
                "source": "user",
                "message": [{"type": "text", "text": "Task.", "private": "PRIVATE_TRACE"}],
            }
        ],
    )
    row = load_harbor(tmp_path, target="baseline").trials[0]
    assert row.messages == ({"role": "user", "content": [{"type": "text", "text": "Task."}]},)


@pytest.mark.parametrize(
    "steps",
    [
        [{"source": "user", "message": [{"type": "image", "source": "PRIVATE_IMAGE"}]}],
        [{"source": "tool", "message": "Tool output."}],
        [
            {"source": "agent", "message": "First answer."},
            {"source": "user", "message": "Late task."},
        ],
        [{"source": "user", "message": "   "}],
    ],
)
def test_unsupported_or_absent_initial_input_is_rejected(
    tmp_path: Path, steps: list[dict[str, object]]
) -> None:
    trial = write_trial(tmp_path)
    update(trial / "agent" / "trajectory.json", steps=steps)
    with pytest.raises(ValueError):
        load_harbor(tmp_path, target="baseline")


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    write_trial(tmp_path, version="ATIF-v2.0")
    with pytest.raises(ValueError, match="schema_version"):
        load_harbor(tmp_path, target="baseline")


@pytest.mark.parametrize("version", [[], {}])
def test_invalid_schema_type_records_issue_without_aborting_other_trials(
    tmp_path: Path, version: object
) -> None:
    invalid = write_trial(tmp_path, "invalid")
    write_trial(tmp_path, "valid", task="benchmark/task-b")
    update(invalid / "agent" / "trajectory.json", schema_version=version)

    with pytest.raises(ValueError, match="schema_version"):
        load_harbor(tmp_path, target="baseline")
    run = load_harbor(tmp_path, target="baseline", on_error="record")
    assert len(run.trials) == 1
    assert run.trials[0].task_id == "benchmark/task-b"
    assert len(run.issues) == 1
    assert run.issues[0].task_id == "benchmark/task-a"
    assert "schema_version" in run.issues[0].message
