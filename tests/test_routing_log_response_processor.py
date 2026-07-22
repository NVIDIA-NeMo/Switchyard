# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-request JSONL routing log (`serve --routing-log-file`)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from switchyard.cli.switchyard_cli import _build_parser
from switchyard.lib.chat_response import ResponseStream
from switchyard.lib.processors.routing_log_response_processor import (
    RoutingLogResponseProcessor,
)
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL
from switchyard.lib.request_metadata import attach_request_metadata
from switchyard_rust.components import RequestMetadata
from switchyard_rust.core import ChatResponse, ProxyContext

TASK_HEADERS = {
    "x-switchyard-intake-task": "hello-world-abc1",
    "proxy_x_session_id": "trial-session-1",
}


def _ctx(*, headers: dict[str, str] | None = None, model: str = "gpt-test") -> ProxyContext:
    ctx = ProxyContext()
    if headers is not None:
        attach_request_metadata(ctx, RequestMetadata.from_headers(headers), headers)
    ctx.metadata[CTX_PROXY_ACTUAL_MODEL] = model
    return ctx


def _openai_completion() -> ChatResponse:
    return ChatResponse.openai_completion({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })


def _anthropic_completion() -> ChatResponse:
    return ChatResponse.anthropic_completion({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": "claude-test",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 7,
            "output_tokens": 3,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 4,
        },
    })


def _responses_completion() -> ChatResponse:
    return ChatResponse.openai_responses_completion({
        "id": "resp_test",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "codex-test",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi"}],
        }],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 6, "output_tokens": 2, "total_tokens": 8},
    })


def _records(log_file: Path) -> list[dict]:
    return [json.loads(line) for line in log_file.read_text().splitlines()]


async def test_openai_completion_record_carries_task_and_usage(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    ctx = _ctx(headers=TASK_HEADERS)
    ctx.metadata["_random_routing_tier"] = "weak"
    await RoutingLogResponseProcessor(log_file).process(ctx, _openai_completion())

    (record,) = _records(log_file)
    assert record["task"] == "hello-world-abc1"
    assert record["session_id"] == "trial-session-1"
    assert record["model"] == "gpt-test"
    assert record["tier"] == "weak"
    assert record["prompt_tokens"] == 5
    assert record["completion_tokens"] == 3
    assert record["total_tokens"] == 8


async def test_anthropic_usage_sums_cache_siblings(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    await RoutingLogResponseProcessor(log_file).process(
        _ctx(headers=TASK_HEADERS, model="claude-test"), _anthropic_completion(),
    )

    (record,) = _records(log_file)
    assert record["prompt_tokens"] == 13  # input + cache_creation + cache_read
    assert record["completion_tokens"] == 3


async def test_responses_completion_uses_input_output_tokens(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    await RoutingLogResponseProcessor(log_file).process(
        _ctx(headers=TASK_HEADERS, model="codex-test"), _responses_completion(),
    )

    (record,) = _records(log_file)
    assert record["prompt_tokens"] == 6
    assert record["completion_tokens"] == 2


async def test_missing_headers_log_null_task_and_session(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    await RoutingLogResponseProcessor(log_file).process(_ctx(), _openai_completion())

    (record,) = _records(log_file)
    assert record["task"] is None
    assert record["session_id"] is None


async def test_streaming_appends_after_drain(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    content_chunk = ChatCompletionChunk(
        id="chatcmpl-test", object="chat.completion.chunk", created=1700000000,
        model="gpt-test",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason="stop")],
    )
    usage_chunk = ChatCompletionChunk(
        id="chatcmpl-test", object="chat.completion.chunk", created=1700000000,
        model="gpt-test", choices=[],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )

    async def _iter() -> AsyncIterator[ChatCompletionChunk]:
        yield content_chunk
        yield usage_chunk

    response = ChatResponse.openai_stream(ResponseStream(_iter()))
    out = await RoutingLogResponseProcessor(log_file).process(
        _ctx(headers=TASK_HEADERS), response,
    )
    assert not log_file.exists()  # nothing until the stream drains

    forwarded = [chunk async for chunk in out.stream]
    assert len(forwarded) == 2

    (record,) = _records(log_file)
    assert record["task"] == "hello-world-abc1"
    assert record["total_tokens"] == 8


async def test_appends_one_line_per_request(tmp_path: Path) -> None:
    log_file = tmp_path / "routing_requests.jsonl"
    processor = RoutingLogResponseProcessor(log_file)
    for _ in range(3):
        await processor.process(_ctx(headers=TASK_HEADERS), _openai_completion())
    assert len(_records(log_file)) == 3


def test_serve_parser_accepts_routing_log_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve", "--routing-log-file", "/tmp/routing.jsonl"])
    assert args.routing_log_file == "/tmp/routing.jsonl"
