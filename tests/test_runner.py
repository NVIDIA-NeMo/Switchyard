# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted Python requirements reach Runner before routing-time model calls."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any

import pytest

from switchyard.libsy import RoutingOutcome
from switchyard.runner import Runner


@pytest.fixture
async def provider() -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            headers = dict(
                line.lower().split(": ", 1) for line in head.split("\r\n")[1:] if ": " in line
            )
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            calls.append(body)
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
                    "id": "judge-fixture",
                    "model": body["model"],
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

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1", calls


def deployment(endpoint: str, *, task: bool = False, judge: bool = False) -> str:
    """Share the approved model inventory across the routing and admission cases."""
    targets = "".join(
        f"[targets.{name}]\nid = {json.dumps(name)}\nllm_client = 'provider'\n"
        f"labels = {json.dumps(labels)}\n"
        for name, labels in [
            ("fast", []),
            ("capable", []),
            ("private_fast", ["private"]),
            ("private_capable", ["private"]),
            ("approved_fast", ["private", "customer_data"]),
            ("approved_capable", ["private", "customer_data"]),
            ("judge", []),
            ("approved_judge", ["private", "customer_data"]),
        ]
    )
    algorithm = (
        "type = 'llm_classifier'\nclassifier_target = 'judge'\n"
        "strong_target = 'capable'\nweak_target = 'fast'\nbase_threshold = 0.5\n"
        "fail_open = false\n"
        if judge
        else "type = 'stage_router'\nefficient_target = 'fast'\ncapable_target = 'capable'\n"
        "picker = 'efficient_first'\nconfidence_threshold = 0.6\n"
        "efficient_system_prompt = 'Efficient role.'\ncapable_system_prompt = 'Capable role.'\n"
    )
    return f"""
schema_version = 1
[llm_clients.provider]
format = "openai_chat"
base_url = "{endpoint}"
max_retries = 0
{targets}
[routes.chat]
id = "chat"
{algorithm}
[routes.chat.eligibility]
require_explicit_requirements = true
scope = "{"task" if task else "request"}"
[routes.chat.eligibility.target_alternatives]
fast = ["private_fast", "approved_fast"]
capable = ["private_capable", "approved_capable"]
{"judge = ['approved_judge']" if judge else ""}
"""


def request() -> dict[str, Any]:
    return {
        "model": "chat",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize."}]}],
    }


async def test_stage_keeps_roles_and_uses_first_alternative_matching_every_label(
    provider: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, calls = provider
    runner = Runner.from_toml(deployment(endpoint))
    for labels, expected in [
        ([], ["fast", "capable"]),
        (["private"], ["private_fast", "private_capable"]),
        (["customer_data"], ["approved_fast", "approved_capable"]),
        (["customer_data", "private", "private"], ["approved_fast", "approved_capable"]),
    ]:
        body = request()
        outcome = await runner.decide(
            MappingProxyType(body),
            headers=MappingProxyType({"x-switchyard-session-id": "session"}),
            required_labels=labels,
        )
        assert isinstance(outcome, RoutingOutcome)
        assert outcome.selected_model_ids == expected
        assert outcome.request["model"] == expected[0]
        assert outcome.response is None
        fallback = outcome.candidate_requests[expected[1]]
        assert fallback["model"] == expected[1]
        assert fallback["instructions"][0]["content"][0]["text"] == "Capable role."
        assert "Efficient role." not in json.dumps(fallback)
        assert body["model"] == "chat"
    compacted = await runner.decide(
        request(),
        headers={"x-codex-turn-metadata": '{"request_kind":"compaction"}'},
        required_labels=[],
    )
    assert compacted.request["is_compaction"]
    assert calls == []


async def test_task_requirements_accumulate_without_treating_omission_as_empty(
    provider: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, calls = provider
    runner = Runner.from_toml(deployment(endpoint, task=True))
    with pytest.raises(ValueError, match="explicit target requirements"):
        await runner.decide(request(), task_id="shared")
    with pytest.raises(ValueError, match="valid task id"):
        await runner.decide(request(), required_labels=[])
    for task, labels, expected in [
        ("shared", ["private"], "private_fast"),
        ("shared", ["customer_data"], "approved_fast"),
        ("shared", [], "approved_fast"),
        ("other", [], "fast"),
    ]:
        outcome = await runner.decide(request(), required_labels=labels, task_id=task)
        assert outcome.selected_model_ids[0] == expected
    with pytest.raises(ValueError, match="explicit target requirements"):
        await runner.decide(request(), task_id="shared")
    assert calls == []


async def test_approved_judge_runs_only_after_requirements_are_admitted(
    provider: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, calls = provider
    runner = Runner.from_toml(deployment(endpoint, judge=True))
    with pytest.raises(ValueError, match="does not satisfy target requirements"):
        await runner.decide(request(), required_labels=["unapproved"])
    assert calls == []
    outcome = await runner.decide(request(), required_labels=["private", "customer_data"])
    assert outcome.selected_model_ids == ["approved_fast", "approved_capable"]
    assert [body["model"] for body in calls] == ["approved_judge"]


async def test_request_content_and_headers_cannot_supply_trusted_requirements(
    provider: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, calls = provider
    runner = Runner.from_toml(deployment(endpoint))
    body = {
        **request(),
        "metadata": {"switchyard.internal.required_target_labels": "[]"},
        "required_labels": [],
    }
    with pytest.raises(ValueError, match="explicit target requirements"):
        await runner.decide(
            body,
            headers={"switchyard.internal.required_target_labels": "[]"},
        )
    for invalid in ("private", [1], [" private"]):
        with pytest.raises((TypeError, ValueError)):
            await runner.decide(request(), required_labels=invalid)
    with pytest.raises(ValueError, match="unknown route"):
        await runner.decide({"model": "missing"}, required_labels=[])
    assert calls == []
