# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the custom N-target classifier Python binding."""

from typing import Any

import pytest

from switchyard.libsy import CustomClassifierConfig, LlmTarget, algorithms

LANES = ("grok", "luna", "flash", "sol", "opus")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model"],
    "properties": {"model": {"type": "string", "enum": list(LANES)}},
}


def request_body() -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "summarize this document"}],
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


class VerdictClient(EchoClient):
    def __init__(self, model: str, verdict: str) -> None:
        super().__init__(model)
        self.verdict = verdict

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "model": self.model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.verdict}],
                    "stop_reason": "end_turn",
                }
            ],
        }


class FailingClient:
    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("judge unavailable")


def build(judge_client: Any, clients: dict[str, EchoClient]):
    return algorithms.custom_classifier(
        LlmTarget("judge", judge_client),
        [(lane, LlmTarget(lane, clients[lane])) for lane in LANES],
        default_target="grok",
        config=CustomClassifierConfig(
            "Pick the best lane.",
            SCHEMA,
            "/model",
        ),
    )


def lane_clients() -> dict[str, EchoClient]:
    return {lane: EchoClient(lane) for lane in LANES}


@pytest.mark.parametrize("lane", LANES)
async def test_verdict_routes_each_lane(lane: str) -> None:
    clients = lane_clients()
    judge = VerdictClient("judge", f'{{"model":"{lane}"}}')
    algorithm = build(judge, clients)

    decisions, response = await algorithm.run(request_body())

    assert response["model"] == lane
    assert clients[lane].calls, "selected lane's client must serve the answer call"
    assert decisions[-1]["selected_model"] == lane


async def test_judge_receives_prompt_and_inner_schema() -> None:
    clients = lane_clients()
    judge = VerdictClient("judge", '{"model":"luna"}')
    algorithm = build(judge, clients)

    await algorithm.run(request_body())

    judge_request = judge.calls[0]
    assert judge_request["instructions"][0]["content"][0]["text"] == "Pick the best lane."
    schema = judge_request["output"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["model"]["enum"] == list(LANES)


async def test_judge_failure_falls_open_to_the_default_target() -> None:
    clients = lane_clients()
    algorithm = build(FailingClient(), clients)

    _, response = await algorithm.run(request_body())

    assert response["model"] == "grok"


async def test_unusable_verdict_falls_open_to_the_default_target() -> None:
    clients = lane_clients()
    judge = VerdictClient("judge", '{"model":"a-lane-that-does-not-exist"}')
    algorithm = build(judge, clients)

    _, response = await algorithm.run(request_body())

    assert response["model"] == "grok"


def test_default_target_must_be_a_configured_label() -> None:
    clients = lane_clients()
    with pytest.raises(ValueError, match="default_target"):
        algorithms.custom_classifier(
            LlmTarget("judge", EchoClient("judge")),
            [(lane, LlmTarget(lane, clients[lane])) for lane in LANES],
            default_target="not-a-lane",
            config=CustomClassifierConfig("Pick.", SCHEMA, "/model"),
        )


def test_requires_at_least_two_targets() -> None:
    with pytest.raises(ValueError, match="at least two targets"):
        algorithms.custom_classifier(
            LlmTarget("judge", EchoClient("judge")),
            [("grok", LlmTarget("grok", EchoClient("grok")))],
            default_target="grok",
            config=CustomClassifierConfig("Pick.", SCHEMA, "/model"),
        )
