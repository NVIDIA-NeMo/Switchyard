# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-free configured algorithms and trusted target requirements."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal

import pytest

from switchyard.libsy import (
    AlgorithmFactory,
    ConfiguredAlgorithm,
    LibsyError,
    LlmResponse,
    RoutingOutcome,
    Step,
)


async def test_factory_builds_a_usable_python_handle() -> None:
    """Configured handles run without a separate model map."""
    source = deployment().split("[routes.coding.eligibility]")[0]
    route = AlgorithmFactory.from_toml(source).build()["switchyard/coding"]
    assert isinstance(route, ConfiguredAlgorithm)
    with pytest.raises(LibsyError, match="build_configured"):
        route.target_models()
    with pytest.raises(LibsyError, match="build_configured"):
        route.run_stream(request_body(), required_labels=[])
    outcome = await finish(route.run_stream(request_body()))
    assert outcome.selected_model_ids == ["fast", "capable"]


def deployment(*, scope: str = "request", judge: Literal["llm", "decision"] | None = None) -> str:
    targets = "\n".join(
        f'{name} = {{ id = "provider/{name}", llm_client = "{client}", labels = {json.dumps(labels)} }}'
        for name, client, labels in [
            ("fast", "standard", []),
            ("capable", "standard", []),
            ("private_fast", "approved", ["private"]),
            ("private_capable", "approved", ["private"]),
            ("both_fast", "approved", ["private", "customer_data"]),
            ("both_capable", "approved", ["private", "customer_data"]),
            ("judge", "standard", []),
            ("approved_judge", "approved", ["private", "customer_data"]),
        ]
    )
    algorithm = (
        'type = "stage_router"\nefficient_target = "fast"\ncapable_target = "capable"\n'
        'picker = "efficient_first"\nconfidence_threshold = 0.6'
    )
    alternative = ""
    if judge is not None:
        is_decision = judge == "decision"
        judge_target = "judge_decision" if is_decision else "judge"
        settings = "[routes.coding.decision]\ncutoff = 0.7" if is_decision else "base_threshold = 0.5"
        algorithm = (
            f'type = "llm_classifier"\nclassifier_target = "{judge_target}"\n'
            f'strong_target = "capable"\nweak_target = "fast"\nfail_open = false\n{settings}'
        )
        replacement = "approved_decision" if is_decision else "approved_judge"
        alternative = f'{judge_target} = ["{replacement}"]'
    return f"""
schema_version = 1
[llm_clients.standard]
format = "openai_chat"
base_url = "https://standard.invalid/v1"
api_key_env = "SWITCHYARD_PYTHON_FACTORY_UNUSED_KEY"
[llm_clients.approved]
format = "openai_chat"
base_url = "https://approved.invalid/v1"
api_key_env = "SWITCHYARD_PYTHON_FACTORY_UNUSED_KEY"
[targets]
{targets}
[decision_clients.scoring]
format = "openai_decisions"
endpoint = "https://host.invalid/v1/decisions"
api_key_env = "SWITCHYARD_PYTHON_FACTORY_UNUSED_KEY"
timeout_ms = 1000
[decision_targets.judge_decision]
id = "decision/standard"
decision_client = "scoring"
[decision_targets.approved_decision]
id = "decision/approved"
decision_client = "scoring"
labels = ["private", "customer_data"]
[routes.coding]
id = "switchyard/coding"
{algorithm}
[routes.coding.eligibility]
require_explicit_requirements = true
scope = "{scope}"
[routes.coding.eligibility.target_alternatives]
fast = ["private_fast", "both_fast"]
capable = ["private_capable", "both_capable"]
{alternative}
"""


def request_body() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}


async def finish(
    stream: AsyncIterator[Step.CallModel | Step.CallDecision | Step.Done],
) -> RoutingOutcome:
    step = await anext(stream)
    assert isinstance(step, Step.Done)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    return step.outcome


def test_requirements_are_rejected_before_returning_routing_work() -> None:
    factory = AlgorithmFactory.from_toml(deployment())
    with pytest.raises(ValueError, match="Runner admission"):
        factory.build()
    route = factory.build_configured()["switchyard/coding"]
    with pytest.raises(LibsyError, match="explicit target requirements"):
        route.run_stream(request_body())
    with pytest.raises(LibsyError, match="required target"):
        route.run_stream(request_body(), required_labels=["unapproved"])


async def test_alternatives_and_task_requirements_use_the_trusted_task_id() -> None:
    route = AlgorithmFactory.from_toml(deployment(scope="task")).build_configured()[
        "switchyard/coding"
    ]
    targets = route.target_models()
    with pytest.raises(LibsyError, match="valid task id"):
        route.run_stream(request_body(), required_labels=[])
    for task, labels, expected in [
        ("shared", ["private"], "private_fast"),
        ("shared", ["customer_data"], "both_fast"),
        ("shared", [], "both_fast"),
        ("other", [], "fast"),
    ]:
        outcome = await finish(
            route.run_stream(
                request_body(),
                required_labels=labels,
                task_id=task,
                headers={"x-switchyard-task-id": "caller-controlled"},
            )
        )
        assert outcome.selected_model_ids[0] == expected
        assert targets[outcome.selected_model_ids[0]] == {
            "client": "standard" if expected == "fast" else "approved",
            "id": f"provider/{expected}",
        }


@pytest.mark.parametrize("judge", ["llm", "decision"])
async def test_approved_judge_calls_are_executed_by_the_python_host(
    judge: Literal["llm", "decision"],
) -> None:
    route = AlgorithmFactory.from_toml(deployment(judge=judge)).build_configured()[
        "switchyard/coding"
    ]
    targets = route.target_models()
    stream = route.run_stream(request_body(), required_labels=["private", "customer_data"])
    step = await anext(stream)
    if judge == "decision":
        assert isinstance(step, Step.CallDecision)
        assert step.call.model == step.call.request["model"] == "decision/approved"
        assert targets[step.call.model] == {"client": "scoring", "id": "decision/approved"}
        choice = {
            "type": "choice",
            "data": {
                "selected": "efficient",
                "probabilities": {"efficient": 0.9, "capable": 0.1},
            },
        }
        response = {"answers": {"route": {"value": choice}}}
    else:
        assert isinstance(step, Step.CallModel)
        assert step.call.models == ["approved_judge"]
        assert step.call.request["model"] == "approved_judge"
        target = targets[step.call.models[0]]
        assert target == {"client": "approved", "id": "provider/approved_judge"}
        verdict = {
            "crux": "bounded task",
            "primary_rule": "SUP-1",
            "capability_boundary": "supported",
            "p_solve": 0.9,
        }
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(verdict)}],
            "stop_reason": "end_turn",
        }
        response = LlmResponse.Agg({"model": target["id"], "outputs": [message]})

    async def reply() -> object:
        return response

    await step.call.respond(reply())
    outcome = await finish(stream)
    assert outcome.selected_model_ids == ["both_fast", "both_capable"]
    assert all(targets[key]["client"] == "approved" for key in outcome.selected_model_ids)
