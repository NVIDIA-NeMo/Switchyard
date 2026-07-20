# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dictionary-based libsy Python API."""

from typing import Any

import pytest

from switchyard.libsy import LibsyError, LlmTarget, algorithms


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


async def test_noop_needs_no_client() -> None:
    decisions, response = await algorithms.noop().run(request_body())

    assert decisions[0]["selected_model"] == "auto"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "OK"}]


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
