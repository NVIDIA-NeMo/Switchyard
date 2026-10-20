# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TOML routing decisions leave the final completion with the Python host."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any

import pytest

from switchyard.libsy import LibsyError, LlmResponse, RoutingOutcome
from switchyard.runner import Runner


@pytest.fixture
async def deployment() -> AsyncIterator[tuple[Runner, list[tuple[dict[str, Any], str]]]]:
    calls: list[tuple[dict[str, Any], str]] = []

    async def judge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            headers = {
                name.lower(): value
                for name, value in (line.split(": ", 1) for line in head.split("\r\n")[1:-2])
            }
            body = await reader.readexactly(int(headers["content-length"]))
            calls.append((json.loads(body), headers.get("authorization", "")))
            content = json.dumps(
                {
                    "crux": "bounded task",
                    "primary_rule": "SUP-1",
                    "capability_boundary": "supported",
                    "p_solve": 0.9,
                }
            )
            response = json.dumps(
                {
                    "id": "judge-1",
                    "model": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()
                + response
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(judge, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        yield (
            Runner.from_toml(f"""
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:{port}/v1"
forward_auth = true
max_retries = 0
[targets.capable]
id = "capable"
llm_client = "local"
[targets.efficient]
id = "efficient"
llm_client = "local"
system_prompt = "Efficient target prompt."
[targets.judge]
id = "judge"
llm_client = "local"
[routes.classifier]
id = "classifier"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "efficient"
base_threshold = 0.5
classify_trigger = "new_session"
[routes.noop]
id = "noop"
type = "noop"
"""),
            calls,
        )


async def test_decide_calls_only_the_judge_and_preserves_sessions(deployment: Any) -> None:
    """Judge calls retain headers and session state without sending the final completion."""
    runner, calls = deployment
    headers = MappingProxyType(
        {"authorization": "Bearer test-caller", "x-switchyard-session-id": "task-a"}
    )
    request = {
        "model": "classifier",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Fix the test."}]}],
    }
    outcome = await runner.decide(request, headers=headers)
    assert isinstance(outcome, RoutingOutcome)
    assert outcome.selected_model_ids == ["efficient", "capable"]
    assert outcome.request["model"] == "efficient"
    assert outcome.request["instructions"][0]["content"][0]["text"] == "Efficient target prompt."
    assert outcome.request["messages"][0]["content"] == request["messages"][0]["content"]
    assert outcome.metadata.algorithm == "llm_task_classifier"
    assert outcome.metadata.evidence["score"] == pytest.approx(0.9)
    assert len(calls) == 1

    resumed = await runner.decide(request, headers=headers)
    assert resumed.selected_model_ids[0] == "efficient"
    assert len(calls) == 1
    fresh = await runner.decide(request, headers={**headers, "x-switchyard-session-id": "task-b"})
    assert fresh.selected_model_ids == ["efficient", "capable"]
    assert all(result.response is None for result in (outcome, resumed, fresh))
    assert [(body["model"], auth) for body, auth in calls] == [
        ("judge", "Bearer test-caller"),
        ("judge", "Bearer test-caller"),
    ]
    assert request["model"] == "classifier" and "instructions" not in request


@pytest.mark.parametrize("stream", [False, True])
async def test_decide_preserves_an_algorithm_answer(deployment: Any, stream: bool) -> None:
    """An answer produced during routing must not trigger another completion."""
    runner, calls = deployment
    outcome = await runner.decide({"model": "noop", "messages": [], "stream": stream})
    assert outcome.selected_model_ids == ["noop"]
    if stream:
        assert isinstance(outcome.response, LlmResponse.Stream)
        events = [event async for event in outcome.response.stream]
        assert any(
            chunk.get("TextDelta", {}).get("text") == "OK"
            for event in events
            for chunk in event["normalized"]
        )
    else:
        assert isinstance(outcome.response, LlmResponse.Agg)
        assert outcome.response.response["outputs"][0]["content"][0]["text"] == "OK"
    assert calls == []


@pytest.mark.parametrize("body,error", [({}, ValueError), ({"model": "missing"}, LibsyError)])
async def test_decide_rejects_missing_or_unknown_models(
    deployment: Any, body: dict[str, object], error: type[Exception]
) -> None:
    """Reject missing or unknown routes before any provider call."""
    runner, calls = deployment
    with pytest.raises(error):
        await runner.decide(body)
    assert calls == []
