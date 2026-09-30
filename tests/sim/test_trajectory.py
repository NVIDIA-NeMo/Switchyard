# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ATIF conversion preserves evidence while isolating task-routing input."""

import json
from pathlib import Path

import pytest

from switchyard.sim import Dataset, HarborRun, Run, Trajectory, load_harbor
from tests.sim.test_harbor import update, write_trial


def atif() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-a",
        "agent": {"name": "custom-agent", "version": "1", "model_name": "default-model"},
        "steps": [
            {"step_id": 1, "source": "system", "message": "Instructions."},
            {"step_id": 2, "source": "user", "message": "Original task."},
            {
                "step_id": 3,
                "source": "agent",
                "model_name": "override-model",
                "message": "LATER_ANSWER",
                "tool_calls": [
                    {"tool_call_id": "call-a", "function_name": "shell", "arguments": {"cmd": "ls"}}
                ],
                "observation": {"results": [{"source_call_id": "call-a", "content": "LATER_TOOL"}]},
            },
            {
                "step_id": 4,
                "source": "user",
                "message": [
                    {"type": "image", "source": {"media_type": "image/png", "path": "a.png"}}
                ],
            },
        ],
        "final_metrics": {"total_prompt_tokens": 100, "total_cost_usd": 2.5},
        "continued_trajectory_ref": None,
        "extra": {"extension": {"nullable": None, "scores": [0, 0.5]}},
    }


def test_full_atif_roundtrip_and_projection_have_independent_ownership() -> None:
    source = atif()
    expected = json.loads(json.dumps(source))
    trajectory = Trajectory.from_dict(source)
    source["steps"][1]["message"] = "MUTATED_SOURCE"
    exported = trajectory.to_dict()
    assert exported == expected
    exported["steps"][2]["tool_calls"][0]["arguments"]["cmd"] = "MUTATED_EXPORT"
    exported["extra"]["extension"]["scores"].append(1)
    assert trajectory.to_dict() == expected

    messages = trajectory.initial_messages()
    assert messages == (
        {"role": "system", "content": [{"type": "text", "text": "Instructions."}]},
        {"role": "user", "content": [{"type": "text", "text": "Original task."}]},
    )
    assert "LATER_" not in json.dumps(messages)
    messages[0]["content"][0]["text"] = "MUTATED_PROJECTION"
    assert trajectory.to_dict() == expected
    assert "LATER_" not in repr(trajectory)


def test_projection_requires_explicit_outcomes_and_whole_run_model_identity() -> None:
    trajectory = Trajectory.from_dict(atif())
    unknown = trajectory.to_trial(task_id="task-a", trial_id="trial-a", target="fast")
    assert unknown.reward is None
    assert unknown.cost_usd is None
    assert unknown.usage == {}
    assert unknown.model is None
    assert unknown.task_checksum is None

    usage = {"total_prompt_tokens": 100}
    known = trajectory.to_trial(
        task_id="task-a",
        trial_id="trial-a",
        target="fast",
        reward=0,
        cost_usd=0,
        cost_source="custom",
        duration_seconds=0,
        model="recorded-model",
        task_checksum="task-v1",
        usage=usage,
    )
    usage["total_prompt_tokens"] = 999
    assert known.reward == known.cost_usd == known.duration_seconds == 0
    assert known.cost_source == "custom"
    assert known.model == "recorded-model"
    assert known.usage == {"total_prompt_tokens": 100}
    assert "LATER_" not in repr(known)
    with pytest.raises(ValueError, match="reward must be finite"):
        trajectory.to_trial(
            task_id="task-a", trial_id="trial-a", target="fast", reward=float("nan")
        )


def test_custom_converter_feeds_the_same_dataset_and_repeat_accounting() -> None:
    def convert(record: dict) -> Trajectory:
        return Trajectory.from_dict(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": record["id"],
                "agent": {"name": "my-agent", "version": "1"},
                "steps": [{"step_id": 1, "source": "user", "message": record["instruction"]}],
            }
        )

    records = [
        {"id": "first", "instruction": "Task.", "reward": 0},
        {"id": "second", "instruction": "Task.", "reward": 1},
    ]
    custom = Run(
        tuple(
            convert(record).to_trial(
                task_id="suite/task-a",
                trial_id=record["id"],
                target="custom",
                reward=record["reward"],
            )
            for record in records
        )
    )
    dataset = Dataset.from_runs({"custom": custom}, input_target="custom")
    assert dataset.tasks[0].outcomes["custom"].reward == 0.5
    assert dataset.tasks[0].outcomes["custom"].trials == 2
    assert isinstance(custom, HarborRun)
    assert HarborRun is Run


@pytest.mark.parametrize("checksum", [" \t\n", 1])
def test_custom_projection_rejects_invalid_task_checksums(checksum) -> None:
    with pytest.raises(ValueError, match="task_checksum must be a non-empty string or None"):
        Trajectory.from_dict(atif()).to_trial(
            task_id="task", trial_id="attempt", target="fast", task_checksum=checksum
        )


@pytest.mark.parametrize("version", [[], {}, None, "ATIF-v2.0"])
def test_invalid_version_is_rejected_without_printing_recording(version: object) -> None:
    with pytest.raises(ValueError, match="schema_version") as error:
        Trajectory.from_dict({"schema_version": version, "steps": [], "private": "PRIVATE_TRACE"})
    assert "PRIVATE_TRACE" not in str(error.value)


@pytest.mark.parametrize("steps", [None, {}, "PRIVATE_TRACE"])
def test_invalid_steps_container_is_rejected(steps: object) -> None:
    with pytest.raises(ValueError, match="steps must be an array"):
        Trajectory.from_dict({"schema_version": "ATIF-v1.7", "steps": steps})


def test_multimodal_input_is_preserved_but_cannot_be_scored_as_text() -> None:
    data = atif()
    data["steps"] = [data["steps"][-1]]
    trajectory = Trajectory.from_dict(data)
    assert trajectory.to_dict() == data
    with pytest.raises(ValueError, match="only text"):
        trajectory.initial_messages()


def test_copied_continuation_context_requires_original_task_input() -> None:
    data = atif()
    data["steps"][1].update(is_copied_context=True, message="SUMMARY_OF_COMPLETED_WORK")
    trajectory = Trajectory.from_dict(data)
    with pytest.raises(ValueError, match="copied ATIF context"):
        trajectory.initial_messages()
    messages = trajectory.initial_messages(task_input="Canonical original task.")
    assert messages == (
        {"role": "user", "content": [{"type": "text", "text": "Canonical original task."}]},
    )
    assert "SUMMARY" not in json.dumps(messages)
    assert trajectory.to_dict() == data


@pytest.mark.parametrize("copied", ["false", 1, []])
def test_malformed_context_flag_is_rejected(copied: object) -> None:
    data = atif()
    data["steps"][1]["is_copied_context"] = copied
    with pytest.raises(ValueError, match="is_copied_context"):
        Trajectory.from_dict(data).initial_messages()


def test_harbor_uses_the_shared_continuation_guard(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    update(
        trial / "agent" / "trajectory.json",
        steps=[{"source": "user", "message": "LATER_SUMMARY", "is_copied_context": True}],
    )
    with pytest.raises(ValueError, match="copied ATIF context"):
        load_harbor(tmp_path, target="fast")
    run = load_harbor(
        tmp_path, target="fast", task_inputs={"benchmark/task-a": "Canonical original task."}
    )
    assert run.trials[0].messages[0]["content"][0]["text"] == "Canonical original task."


def test_present_empty_atif_is_not_treated_as_a_missing_file(tmp_path: Path) -> None:
    trial = write_trial(tmp_path)
    (trial / "agent" / "trajectory.json").write_text("{}")
    with pytest.raises(ValueError, match="schema_version"):
        load_harbor(tmp_path, target="fast", task_inputs={"benchmark/task-a": "Task."})
