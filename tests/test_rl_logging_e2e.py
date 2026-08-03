# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process end-to-end test for ``--enable-rl-logging`` on the serve chain.

Drives a real HTTP request through the production serve path — route-bundle
table → ``build_switchyard_app`` → endpoint → chain → backend — with the
RL-logging processors wired in exactly as ``switchyard serve
--enable-rl-logging`` wires them, and asserts a ``message_history`` trace file
is written. The upstream is a real loopback HTTP server (``_OpenAICompatStub``)
because the Rust backend uses ``reqwest``, which ``respx`` cannot intercept.

No API key, no ``claude`` binary, no outbound network — safe for CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from switchyard.cli.route_bundle import build_route_bundle_table
from switchyard.lib.processors.rl_logging_response_processor import build_rl_logging_processors
from switchyard.server.switchyard_app import build_switchyard_app
from tests._chain_test_helpers import _backend_payload, _OpenAICompatStub, _sse_body, _stream_chunk


def _build_app(stub: _OpenAICompatStub, log_dir: Path):
    """Build the serve chain pointed at ``stub`` with RL-logging attached.

    Mirrors ``_cmd_serve``: ``build_rl_logging_processors`` produces the paired
    snapshot + writer processors, which feed the route-bundle table builder.
    """
    rl_request, rl_response = build_rl_logging_processors(log_dir)
    table = build_route_bundle_table(
        {
            "defaults": {
                "api_key": "dummy",
                "base_url": stub.base_url,
                "format": "openai",
            },
            "routes": {
                "mock-model": {"type": "passthrough", "target": "mock-model"}
            },
        },
        pre_routing_request_processors=rl_request,
        extra_response_processors=rl_response,
    )
    return build_switchyard_app(table)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


def _only_trace(log_dir: Path) -> dict:
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1, f"expected one trace file, got {files}"
    return json.loads(files[0].read_text())


async def test_serve_chain_writes_trace_non_streaming(tmp_path: Path) -> None:
    with _OpenAICompatStub() as stub:
        stub.respond_json(_backend_payload(content="hello world", model="mock-model"))
        app = _build_app(stub, tmp_path)
        async with _client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
                headers={"authorization": "Bearer test"},
            )
        assert resp.status_code == 200
        assert stub.called

    entry = _only_trace(tmp_path)
    assert entry["is_valid"] is True
    assert [m["role"] for m in entry["messages"]] == ["user", "assistant"]
    assert entry["messages"][-1]["content"] == "hello world"
    assert entry["token_count"] == {
        "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12,
    }


async def test_serve_chain_writes_trace_streaming(tmp_path: Path) -> None:
    usage_chunk = {
        "id": "chatcmpl-backend-stream",
        "object": "chat.completion.chunk",
        "created": 1700000002,
        "model": "mock-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    with _OpenAICompatStub() as stub:
        stub.respond_sse(_sse_body([_stream_chunk(content="hello world"), usage_chunk]))
        app = _build_app(stub, tmp_path)
        async with _client(app) as client:
            # The trace is written when the stream drains, so the body must be
            # fully consumed before asserting.
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"authorization": "Bearer test"},
            ) as resp:
                assert resp.status_code == 200
                async for _ in resp.aiter_bytes():
                    pass

    entry = _only_trace(tmp_path)
    assert entry["is_valid"] is True
    assert [m["role"] for m in entry["messages"]] == ["user", "assistant"]
    assert entry["messages"][-1]["content"] == "hello world"
    assert entry["token_count"]["total_tokens"] == 12
