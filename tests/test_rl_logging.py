# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local RL trace logging (`--enable-rl-logging`)."""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from switchyard.cli.launchers.launch_intake_config import build_launch_capture_processors
from switchyard.cli.switchyard_cli import _build_parser
from switchyard.lib.chat_response import ResponseStream
from switchyard.lib.processors.rl_logging_request_processor import (
    CTX_RL_LOGGING_REQUEST,
    RlLoggingRequestProcessor,
)
from switchyard.lib.processors.rl_logging_response_processor import (
    RlLoggingResponseProcessor,
    build_rl_logging_processors,
)
from switchyard.server.server_util import resolve_rl_log_dir
from switchyard_rust.core import ChatRequest, ChatResponse, ProxyContext


def _request() -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "tool_choice": "auto",
    })


def _completion(*, content: str | None = "hi there", tool_calls: list | None = None,
                choices: list | None = None) -> ChatResponse:
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if choices is None:
        choices = [{"index": 0, "message": message, "finish_reason": "stop"}]
    return ChatResponse.openai_completion({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-test",
        "choices": choices,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })


def _read_only_trace(log_dir: Path) -> dict:
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1, f"expected exactly one trace file, got {files}"
    return json.loads(files[0].read_text())


async def _run(log_dir: Path, response: ChatResponse, *, snapshot: bool = True) -> ProxyContext:
    ctx = ProxyContext()
    if snapshot:
        await RlLoggingRequestProcessor().process(ctx, _request())
    await RlLoggingResponseProcessor(log_dir).process(ctx, response)
    return ctx


async def test_request_processor_snapshots_openai_body() -> None:
    ctx = ProxyContext()
    await RlLoggingRequestProcessor().process(ctx, _request())
    snapshot = ctx.metadata[CTX_RL_LOGGING_REQUEST]
    assert isinstance(snapshot, dict)
    assert [m["role"] for m in snapshot["messages"]] == ["system", "user"]


async def test_non_streaming_writes_message_history_trace(tmp_path: Path) -> None:
    await _run(tmp_path, _completion(content="hi there"))
    entry = _read_only_trace(tmp_path)

    assert entry["is_valid"] is True
    assert entry["tool_choice"] == "auto"
    assert entry["token_count"] == {
        "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
    }
    # Request history + appended assistant turn.
    assert [m["role"] for m in entry["messages"]] == ["system", "user", "assistant"]
    assert entry["messages"][-1]["content"] == "hi there"
    # Tools rewritten to the message_history shape.
    assert entry["tools"] == [{
        "id": "get_weather",
        "description": "Look up weather",
        "inputSchema": {"jsonSchema": {"type": "object", "properties": {}}},
    }]


async def test_assistant_tool_calls_are_logged(tmp_path: Path) -> None:
    tool_calls = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }]
    await _run(tmp_path, _completion(content=None, tool_calls=tool_calls))
    entry = _read_only_trace(tmp_path)
    assistant = entry["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == tool_calls
    assert "content" not in assistant


async def test_empty_string_assistant_content_is_preserved(tmp_path: Path) -> None:
    """An empty-string completion is valid content and must not be dropped."""
    await _run(tmp_path, _completion(content=""))
    entry = _read_only_trace(tmp_path)
    assert entry["messages"][-1] == {"role": "assistant", "content": ""}


async def test_empty_choices_writes_nothing(tmp_path: Path) -> None:
    await _run(tmp_path, _completion(choices=[]))
    assert list(tmp_path.glob("*.json")) == []


async def test_missing_request_snapshot_writes_nothing(tmp_path: Path) -> None:
    await _run(tmp_path, _completion(), snapshot=False)
    assert list(tmp_path.glob("*.json")) == []


def _anthropic_completion(*, content: str, input_tokens: int, output_tokens: int) -> ChatResponse:
    return ChatResponse.anthropic_completion({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-test",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


def _responses_completion(*, content: str, input_tokens: int, output_tokens: int) -> ChatResponse:
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
            "content": [{"type": "output_text", "text": content}],
        }],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })


async def test_anthropic_response_is_translated_and_logged(tmp_path: Path) -> None:
    """claude's backend can answer in Anthropic format; it must translate to message_history."""
    await _run(tmp_path, _anthropic_completion(content="hello world", input_tokens=7, output_tokens=3))
    entry = _read_only_trace(tmp_path)
    assert [m["role"] for m in entry["messages"]] == ["system", "user", "assistant"]
    assert entry["messages"][-1]["content"] == "hello world"
    assert entry["token_count"]["prompt_tokens"] == 7
    assert entry["token_count"]["completion_tokens"] == 3
    assert entry["is_valid"] is True


async def test_openai_responses_response_is_translated_and_logged(tmp_path: Path) -> None:
    """codex talks the Responses API; a Responses completion must translate too."""
    await _run(tmp_path, _responses_completion(content="hello world", input_tokens=6, output_tokens=2))
    entry = _read_only_trace(tmp_path)
    assert entry["messages"][-1] == {"role": "assistant", "content": "hello world"}
    assert entry["token_count"]["prompt_tokens"] == 6
    assert entry["token_count"]["completion_tokens"] == 2
    assert entry["is_valid"] is True


async def test_writes_one_file_per_turn(tmp_path: Path) -> None:
    """A reused processor writes one independent file per completed turn (no overwrite)."""
    request_processor = RlLoggingRequestProcessor()
    response_processor = RlLoggingResponseProcessor(tmp_path)
    for i in range(3):
        ctx = ProxyContext()
        await request_processor.process(ctx, _request())
        await response_processor.process(ctx, _completion(content=f"turn {i}"))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 3  # distinct filename per turn, nothing overwritten
    last_contents = sorted(
        json.loads(f.read_text())["messages"][-1]["content"] for f in files
    )
    assert last_contents == ["turn 0", "turn 1", "turn 2"]


async def test_write_failure_does_not_break_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace-write failure is swallowed — the proxy response must flow untouched."""
    processor = RlLoggingResponseProcessor(tmp_path)
    ctx = ProxyContext()
    await RlLoggingRequestProcessor().process(ctx, _request())

    def _boom(_entry: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(processor, "_write_entry", _boom)
    response = _completion()
    assert await processor.process(ctx, response) is response
    assert list(tmp_path.glob("*.json")) == []


def test_build_launch_capture_processors_toggles_on_rl_log_dir(tmp_path: Path) -> None:
    none_req, none_resp = build_launch_capture_processors(None, None)
    assert none_req == [] and none_resp == []

    req, resp = build_launch_capture_processors(None, tmp_path)
    assert [type(p).__name__ for p in req] == ["RlLoggingRequestProcessor"]
    assert [type(p).__name__ for p in resp] == ["RlLoggingResponseProcessor"]


def test_resolve_rl_log_dir() -> None:
    off = argparse.Namespace(enable_rl_logging=False, rl_log_dir="./rl_data")
    assert resolve_rl_log_dir(off) is None

    on = argparse.Namespace(enable_rl_logging=True, rl_log_dir="/tmp/traces")
    assert resolve_rl_log_dir(on) == Path("/tmp/traces")


def test_build_rl_logging_processors() -> None:
    assert build_rl_logging_processors(None) == ([], [])

    req, resp = build_rl_logging_processors(Path("/tmp/x"))
    assert [type(p).__name__ for p in req] == ["RlLoggingRequestProcessor"]
    assert [type(p).__name__ for p in resp] == ["RlLoggingResponseProcessor"]


async def test_streaming_logs_after_drain(tmp_path: Path) -> None:
    """Streaming turns are captured on stream completion, not before."""
    ctx = ProxyContext()
    await RlLoggingRequestProcessor().process(ctx, _request())

    content_chunk = ChatCompletionChunk(
        id="chatcmpl-test", object="chat.completion.chunk", created=1700000000,
        model="gpt-test",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi there"), finish_reason="stop")],
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
    out = await RlLoggingResponseProcessor(tmp_path).process(ctx, response)
    # Nothing is written until the stream actually drains.
    assert list(tmp_path.glob("*.json")) == []

    forwarded = [chunk async for chunk in out.stream]
    assert len(forwarded) == 2  # stream still forwards every chunk to the client

    entry = _read_only_trace(tmp_path)
    assert entry["messages"][-1] == {"role": "assistant", "content": "hi there"}
    assert entry["token_count"] == {
        "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
    }


def test_global_flag_parses_before_subcommand() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--enable-rl-logging", "--rl-log-dir", "/tmp/x", "launch", "claude"],
    )
    assert args.enable_rl_logging is True
    assert args.rl_log_dir == "/tmp/x"
    assert resolve_rl_log_dir(args) == Path("/tmp/x")


def test_global_flag_parses_before_serve() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--enable-rl-logging", "--rl-log-dir", "/tmp/x", "serve"],
    )
    assert args.command == "serve"
    assert args.enable_rl_logging is True
    assert resolve_rl_log_dir(args) == Path("/tmp/x")


def test_serve_attaches_rl_logging_processors(monkeypatch, tmp_path: Path) -> None:
    """`serve --routing-profiles --enable-rl-logging` wires the trace logger into the chain."""
    import switchyard.cli.switchyard_cli as cli

    captured: dict[str, list] = {}

    class _FakeTable:
        def registered_models(self) -> list[str]:
            return ["m"]

        def default_model(self) -> str | None:
            return None

    def _fake_load(routing_profiles, *, pre_routing_request_processors=(),
                   extra_response_processors=(), **_kwargs):
        captured["request"] = list(pre_routing_request_processors)
        captured["response"] = list(extra_response_processors)
        return _FakeTable()

    monkeypatch.setattr(cli, "load_route_bundle_table", _fake_load)
    monkeypatch.setattr(cli, "build_and_serve", lambda *a, **k: None)

    args = argparse.Namespace(
        config=None, routing_profiles="profiles.yaml",
        enable_rl_logging=True, rl_log_dir=str(tmp_path),
        intake_enabled=False, intake_base_url=None, intake_workspace=None,
        intake_api_key=None, intake_target_url=None,
    )
    cli._cmd_serve(args)

    assert [type(p).__name__ for p in captured["request"]] == ["RlLoggingRequestProcessor"]
    assert [type(p).__name__ for p in captured["response"]] == ["RlLoggingResponseProcessor"]


def test_serve_config_rejects_rl_logging() -> None:
    """The Rust profile-server path has no Python chain, so it must reject the flag."""
    from switchyard.cli.switchyard_cli import _cmd_serve_profile_config

    args = argparse.Namespace(
        config="profiles.yaml", routing_profiles=None, inbound=None,
        reload=False, workers=1,
        intake_enabled=False, intake_base_url=None, intake_workspace=None,
        intake_api_key=None, intake_target_url=None,
        enable_rl_logging=True, rl_log_dir="./rl_data",
    )
    with pytest.raises(SystemExit, match="enable-rl-logging"):
        _cmd_serve_profile_config(args)
