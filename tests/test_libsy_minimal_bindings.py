# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dictionary-based libsy Python API."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from switchyard.libsy import (
    Algorithm,
    ContextWindowExceededError,
    CustomClassifierConfig,
    LlmClassifierConfig,
    LlmResponse,
    RoutingOutcome,
    Step,
    TaskClassifierConfig,
    algorithms,
)


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


async def run_algorithm(
    algorithm: Algorithm,
    clients: dict[str, Any] | None = None,
    *,
    request: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    async for step in algorithm.run_stream(request or request_body(), headers=headers):
        match step:
            case Step.CallModel(call):
                for index, target in enumerate(call.models):
                    candidate_request = {**call.request, "model": target}
                    client = (clients or {})[target]
                    try:
                        response = await client.call(candidate_request)
                    except ContextWindowExceededError as error:
                        if index + 1 == len(call.models):
                            call.fail(error)
                    except Exception as error:
                        call.fail(error)
                        break
                    else:
                        call.respond(LlmResponse.Agg(response))
                        break
            case Step.Done(outcome):
                if outcome.response is not None:
                    match outcome.response:
                        case LlmResponse.Agg(response):
                            return outcome.selected_model_ids[0], response
                        case LlmResponse.Stream(_):
                            raise AssertionError("test helper expected an aggregate response")
                candidates = outcome.selected_model_ids
                for index, target in enumerate(candidates):
                    candidate_request = {**outcome.request, "model": target}
                    client = (clients or {})[target]
                    try:
                        response = await client.call(candidate_request)
                    except ContextWindowExceededError:
                        if index + 1 == len(candidates):
                            raise
                    else:
                        return outcome.selected_model_ids[0], response
    raise AssertionError("algorithm stream ended without an outcome")


async def test_random_streams_complex_steps_and_accepts_a_dictionary_response() -> None:
    client = EchoClient("fast")
    algorithm = algorithms.random(["fast"])
    outcome: RoutingOutcome | None = None
    variants: list[str] = []

    async for step in algorithm.run_stream(request_body()):
        match step:
            case Step.Done(done):
                variants.append("done")
                outcome = done

    assert variants == ["done"]
    assert outcome is not None
    assert outcome.selected_model_ids == ["fast"]
    assert outcome.response is None
    response = await client.call(outcome.request)
    assert client.calls[0]["model"] == "fast"
    assert client.calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert response["model"] == "fast"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "fast"}]


async def test_routing_call_accepts_a_streamed_response() -> None:
    async def events() -> AsyncIterator[dict[str, object]]:
        for chunk in [
            {"MessageStart": {"id": "response-1", "model": "judge"}},
            {"TextDelta": {"index": 0, "text": '{"target":"balanced"}'}},
            {"MessageStop": {"reason": "end_turn"}},
        ]:
            yield {"preservation": None, "normalized": [chunk]}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target"],
        "properties": {"target": {"type": "string", "enum": ["fast", "balanced"]}},
    }
    algorithm = algorithms.llm_classifier(
        LlmClassifierConfig.custom(
            "judge",
            [("fast", "model-a"), ("balanced", "model-b")],
            default_target="fast",
            config=CustomClassifierConfig("Choose a target.", schema, "/target"),
        )
    )
    outcome: RoutingOutcome | None = None

    async for step in algorithm.run_stream(request_body()):
        match step:
            case Step.CallModel(call):
                call.respond(LlmResponse.Stream(events()))
            case Step.Done(done):
                outcome = done

    assert outcome is not None
    assert outcome.selected_model_ids == ["model-b", "model-a"]


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
                                    '{"crux":"bounded task","primary_rule":"SUP-1",'
                                    '"capability_boundary":"supported","p_solve":0.9}'
                                ),
                            }
                        ],
                        "stop_reason": "end_turn",
                    }
                ],
            }

    judge = JudgeClient("judge")
    weak = EchoClient("weak")
    algorithm = algorithms.llm_classifier(
        LlmClassifierConfig.capability(
            "judge",
            "weak",
            "strong",
            config=TaskClassifierConfig(
                0.5,
                threshold_step=0.1,
                prompt="Custom capability rubric.",
            ),
        ),
    )

    _, response = await run_algorithm(
        algorithm,
        {
            "judge": judge,
            "weak": weak,
            "strong": EchoClient("strong"),
        },
    )

    prompt = judge.calls[0]["instructions"][0]["content"][0]["text"]
    assert prompt == "Custom capability rubric."
    assert judge.calls[0]["output"]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["p_solve"]
    assert response["model"] == "weak"


async def test_custom_classifier_routes_across_named_targets() -> None:
    class JudgeClient(EchoClient):
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(request)
            return {
                "model": self.model,
                "outputs": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": '{"target":"balanced"}'}],
                        "stop_reason": "end_turn",
                    }
                ],
            }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target"],
        "properties": {"target": {"type": "string", "enum": ["fast", "balanced", "best"]}},
    }
    algorithm = algorithms.llm_classifier(
        LlmClassifierConfig.custom(
            "judge",
            [("fast", "model-a"), ("balanced", "model-b"), ("best", "model-c")],
            default_target="fast",
            config=CustomClassifierConfig("Choose a target.", schema, "/target"),
        )
    )

    _, response = await run_algorithm(
        algorithm,
        {
            "judge": JudgeClient("judge"),
            "model-a": EchoClient("model-a"),
            "model-b": EchoClient("model-b"),
            "model-c": EchoClient("model-c"),
        },
    )

    assert response["model"] == "model-b"


async def test_classifier_config_accepts_json_object_output() -> None:
    """Verify that Python can select JSON Object mode for a classifier judge."""

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
                                    '{"crux":"bounded task","primary_rule":"SUP-1",'
                                    '"capability_boundary":"supported","p_solve":0.9}'
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
        "judge",
        "weak",
        "strong",
        config=TaskClassifierConfig(0.5, response_format_type="json_object"),
    )

    _, response = await run_algorithm(
        algorithm,
        {
            "judge": judge,
            "weak": weak,
            "strong": EchoClient("strong"),
        },
    )

    assert judge.calls[0]["output"]["response_format"] == {"type": "json_object"}
    prompt = judge.calls[0]["instructions"][0]["content"][0]["text"]
    assert "JSON Schema" in prompt
    assert '"p_solve"' in prompt
    assert response["model"] == "weak"


def test_classifier_config_rejects_unknown_response_format() -> None:
    invalid_response_format: Any = "yaml"

    with pytest.raises(
        ValueError,
        match="response_format_type must be 'json_schema' or 'json_object'",
    ):
        TaskClassifierConfig(0.5, response_format_type=invalid_response_format)


async def test_random_weights_and_seed_are_reproducible() -> None:
    def algorithm():
        return algorithms.random(
            ["fast", "capable"],
            weights=[1, 3],
            seed=42,
        )

    first_router = algorithm()
    second_router = algorithm()
    clients = {"fast": EchoClient("fast"), "capable": EchoClient("capable")}
    first = [(await run_algorithm(first_router, clients))[1]["model"] for _ in range(100)]
    second = [(await run_algorithm(second_router, clients))[1]["model"] for _ in range(100)]

    assert first == second
    assert 65 <= second.count("capable") <= 85


def test_random_rejects_invalid_weights() -> None:
    targets = ["fast", "capable"]

    with pytest.raises(ValueError, match="expected 2 weights, got 1"):
        algorithms.random(targets, weights=[1])


async def test_noop_needs_no_client() -> None:
    selected_model, response = await run_algorithm(algorithms.noop())

    assert selected_model == "auto"
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
        algorithms.noop().run_stream(request_body(), headers=headers)


async def test_algorithm_accepts_case_insensitive_duplicate_names() -> None:
    selected_model, _ = await run_algorithm(
        algorithms.noop(), headers={"X-Unused": "first", "x-unused": "second"}
    )

    assert selected_model == "auto"


def test_algorithm_rejects_header_map_capacity_overflow() -> None:
    headers = {f"x-header-{index}": "value" for index in range(32_769)}

    with pytest.raises(ValueError, match="max size reached"):
        algorithms.noop().run_stream(request_body(), headers=headers)


def test_algorithm_exposes_only_streaming_execution() -> None:
    algorithm = algorithms.noop()

    assert callable(algorithm.run_stream)
    assert not hasattr(algorithm, "run")


def test_random_requires_a_target() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        algorithms.random([])


def test_invalid_request_is_rejected_at_the_boundary() -> None:
    algorithm = algorithms.random(["fast"])

    with pytest.raises(ValueError, match="unknown variant"):
        algorithm.run_stream(
            {
                "model": "auto",
                "messages": [{"role": "invalid", "content": []}],
            }
        )


async def test_context_window_failure_falls_back_to_the_next_model() -> None:
    class OverflowClient:
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            raise ContextWindowExceededError("request exceeds context window")

    algorithm = algorithms.stage_router(
        "strong",
        "fast",
        picker="efficient_first",
        confidence_threshold=0.5,
    )
    selected_model, response = await run_algorithm(
        algorithm,
        {"fast": OverflowClient(), "strong": EchoClient("strong")},
    )

    assert selected_model == "fast"
    assert response["model"] == "strong"


async def test_stage_router_applies_additive_tool_semantics() -> None:
    """Verify that configured mutation semantics reach stage-router scoring."""

    algorithm = algorithms.stage_router(
        "strong",
        "fast",
        picker="capable_first",
        confidence_threshold=0.3,
        tool_semantics={
            "observe": ["KB_search"],
            "mutate": ["send_payment_request"],
            "plan": ["create_research_plan"],
            "new": ["send_message_to_user"],
        },
    )

    selected_model, _ = await run_algorithm(
        algorithm,
        {"strong": EchoClient("strong"), "fast": EchoClient("fast")},
        request={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "pay the balance"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "call_1",
                            "name": "send_payment_request",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "content": [{"type": "text", "text": "payment sent"}],
                            "is_error": None,
                        }
                    ],
                },
            ],
        },
    )

    assert selected_model == "fast"


def test_stage_router_rejects_unknown_tool_semantics_category() -> None:
    with pytest.raises(ValueError, match="unknown tool_semantics category"):
        algorithms.stage_router(
            "strong",
            "fast",
            picker="efficient_first",
            confidence_threshold=0.5,
            tool_semantics={"complete": ["end_conversation"]},
        )
