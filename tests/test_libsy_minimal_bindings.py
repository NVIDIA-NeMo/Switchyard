# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dictionary-based libsy Python API."""

from typing import Any

import pytest

from switchyard.libsy import LibsyError, LlmTarget, TaskClassifierConfig, algorithms


def request_body() -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
    }


class EchoClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "model": self.model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.model}],
                    "stop_reason": "end_turn",
                }
            ],
        }


async def test_random_runs_with_a_python_client() -> None:
    client = EchoClient("fast")
    algorithm = algorithms.random([LlmTarget("fast", client)])

    decisions, response = await algorithm.run(request_body())

    assert decisions == [
        {
            "selected_model": "fast",
            "reasoning": "random routing selected target 'fast'",
        }
    ]
    assert client.calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert response["model"] == "fast"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "fast"}]


async def test_classifier_config_accepts_a_prompt_override() -> None:
    """Verify that a configured classifier prompt is rendered for the judge."""

    class JudgeClient(EchoClient):
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(request)
            return {
                "model": self.model,
                "outputs": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    '{"recommended_route":"efficient","p_solve":0.9,'
                                    '"confidence":0.9,"abstain":false,'
                                    '"capability_boundary":"supported",'
                                    '"primary_rule":"SUP-1","crux":"bounded task"}'
                                ),
                            }
                        ],
                        "stop_reason": "end_turn",
                    }
                ],
            }

    judge = JudgeClient("judge")
    weak = EchoClient("weak")
    algorithm = algorithms.llm_task_classifier(
        LlmTarget("judge", judge),
        LlmTarget("weak", weak),
        LlmTarget("strong", EchoClient("strong")),
        config=TaskClassifierConfig(
            0.5,
            prompt="Custom capability rubric:\n{{RESPONSE_SCHEMA}}",
        ),
    )

    _, response = await algorithm.run(request_body())

    prompt = judge.calls[0]["messages"][0]["content"][0]["text"]
    assert prompt.startswith("Custom capability rubric:")
    assert '"recommended_route"' in prompt
    assert "{{RESPONSE_SCHEMA}}" not in prompt
    assert response["model"] == "weak"


async def test_random_weights_and_seed_are_reproducible() -> None:
    def algorithm():
        return algorithms.random(
            [
                LlmTarget("fast", EchoClient("fast")),
                LlmTarget("capable", EchoClient("capable")),
            ],
            weights=[1, 3],
            seed=42,
        )

    first_router = algorithm()
    second_router = algorithm()
    first = [(await first_router.run(request_body()))[1]["model"] for _ in range(100)]
    second = [(await second_router.run(request_body()))[1]["model"] for _ in range(100)]

    assert first == second
    assert 65 <= second.count("capable") <= 85


def test_random_rejects_invalid_weights() -> None:
    targets = [
        LlmTarget("fast", EchoClient("fast")),
        LlmTarget("capable", EchoClient("capable")),
    ]

    with pytest.raises(ValueError, match="expected 2 weights, got 1"):
        algorithms.random(targets, weights=[1])


async def test_noop_needs_no_client() -> None:
    decisions, response = await algorithms.noop().run(request_body())

    assert decisions[0]["selected_model"] == "auto"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "OK"}]


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"invalid header": "value"}, "invalid HTTP header name"),
        ({"x-valid": "invalid\nvalue"}, "failed to parse header value"),
    ],
)
def test_algorithm_rejects_invalid_headers(headers: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        algorithms.noop().run(request_body(), headers=headers)


async def test_algorithm_accepts_case_insensitive_duplicate_names() -> None:
    decisions, _ = await algorithms.noop().run(
        request_body(), headers={"X-Unused": "first", "x-unused": "second"}
    )

    assert decisions[0]["selected_model"] == "auto"


def test_algorithm_rejects_header_map_capacity_overflow() -> None:
    headers = {f"x-header-{index}": "value" for index in range(32_769)}

    with pytest.raises(ValueError, match="max size reached"):
        algorithms.noop().run(request_body(), headers=headers)


def test_algorithm_exposes_only_managed_execution() -> None:
    algorithm = algorithms.noop()

    assert callable(algorithm.run)
    assert not hasattr(algorithm, "run_stream")


def test_target_requires_a_callable_client() -> None:
    with pytest.raises(TypeError, match="client must define async call"):
        LlmTarget("fast", object())

    with pytest.raises(TypeError, match="client.call must be callable"):
        LlmTarget("fast", type("Client", (), {"call": None})())


def test_random_requires_a_target() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        algorithms.random([])


async def test_invalid_request_is_rejected_at_the_boundary() -> None:
    algorithm = algorithms.random([LlmTarget("fast", EchoClient("fast"))])

    with pytest.raises(ValueError, match="unknown variant"):
        await algorithm.run(
            {
                "model": "auto",
                "messages": [{"role": "invalid", "content": []}],
            }
        )


async def test_client_failure_becomes_libsy_error() -> None:
    class FailingClient:
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("client failed")

    algorithm = algorithms.random([LlmTarget("broken", FailingClient())])

    with pytest.raises(LibsyError, match="client failed"):
        await algorithm.run(request_body())
