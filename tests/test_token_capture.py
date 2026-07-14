# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for token-level capture (RL logging + ``token_capture_engine`` targets)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from switchyard.cli.route_bundle import route_bundle_declares_token_capture
from switchyard.cli.switchyard_cli import _build_parser
from switchyard.lib.processors.rl_logging_request_processor import RlLoggingRequestProcessor
from switchyard.lib.processors.token_capture_request_processor import (
    CTX_TOKEN_CAPTURE_ORIGINAL_STREAM,
    CTX_TOKEN_CAPTURE_SESSION,
    TokenCaptureRequestProcessor,
)
from switchyard.lib.processors.token_capture_response_processor import (
    TokenCaptureResponseProcessor,
    build_token_capture_processors,
)
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
from switchyard.server.server_util import resolve_rl_log_dir
from switchyard_rust.components import RequestMetadata
from switchyard_rust.core import ChatRequest, ChatResponse, ProxyContext

_SESSION = "claude-1700000000000-abc12345"


def _anthropic_request() -> ChatRequest:
    return ChatRequest.anthropic(
        {
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )


def _vllm_completion(*, choices: list | None = None) -> ChatResponse:
    """Backend body as vLLM emits it with token-capture params enabled."""
    if choices is None:
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "token_ids": [7, 8],
                "logprobs": {
                    "content": [
                        {"token": "hi", "logprob": -0.1},
                        {"token": "!", "logprob": -0.2},
                    ]
                },
            }
        ]
    return ChatResponse.openai_completion(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen/Qwen3-0.6B",
            "prompt_token_ids": [1, 2, 3],
            "choices": choices,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )


def _read_only_record(capture_dir: Path) -> dict:
    files = list(capture_dir.rglob("*.json"))
    assert len(files) == 1, f"expected exactly one record file, got {files}"
    return json.loads(files[0].read_text())


def _session_ctx(session_id: str | None = _SESSION) -> ProxyContext:
    ctx = ProxyContext()
    if session_id is not None:
        ctx.metadata[CTX_REQUEST_METADATA] = RequestMetadata.from_headers(
            {"proxy_x_session_id": session_id}
        )
    return ctx


async def _run(
    capture_dir: Path,
    response: ChatResponse,
    *,
    session_id: str | None = None,
    request: ChatRequest | None = None,
    ctx: ProxyContext | None = None,
) -> ProxyContext:
    """Run the capture pair the way the wiring installs it (rl snapshot first)."""
    if ctx is None:
        ctx = _session_ctx(session_id)
    if request is None:
        request = _anthropic_request()
    await RlLoggingRequestProcessor().process(ctx, request)
    await TokenCaptureRequestProcessor().process(ctx, request)
    await TokenCaptureResponseProcessor(capture_dir).process(ctx, response)
    return ctx


# ---------------------------------------------------------------------------
# Activation: --enable-rl-logging + token_capture_engine target
# ---------------------------------------------------------------------------


def test_capture_gates_on_rl_logging_flag(tmp_path: Path) -> None:
    parser = _build_parser()

    args = parser.parse_args(["serve"])
    assert resolve_rl_log_dir(args) is None

    args = parser.parse_args(["--enable-rl-logging", "--rl-log-dir", str(tmp_path), "serve"])
    assert resolve_rl_log_dir(args) == tmp_path


def test_route_bundle_declares_token_capture_dict() -> None:
    declaring = {
        "routes": {
            "m": {
                "type": "model",
                "target": {"model": "a", "base_url": "http://x/v1", "token_capture_engine": "vllm"},
            },
        },
    }
    plain = {
        "routes": {
            "m": {"type": "model", "target": {"model": "a", "base_url": "http://x/v1"}},
        },
    }
    assert route_bundle_declares_token_capture(declaring) is True
    assert route_bundle_declares_token_capture(plain) is False
    assert route_bundle_declares_token_capture(None) is False


def test_route_bundle_declares_token_capture_path(tmp_path: Path) -> None:
    profile = tmp_path / "profiles.yaml"
    profile.write_text(
        "routes:\n"
        "  m:\n"
        "    type: model\n"
        "    target:\n"
        "      model: a\n"
        "      base_url: http://x/v1\n"
        "      token_capture_engine: vllm\n"
    )
    assert route_bundle_declares_token_capture(str(profile)) is True
    # An unreadable bundle reports False; the real table load raises.
    assert route_bundle_declares_token_capture(str(tmp_path / "missing.yaml")) is False


def test_serve_installs_capture_pair_for_declaring_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """rl-logging on + token_capture_engine target -> capture pair replaces rl pair."""
    import switchyard.cli.switchyard_cli as cli

    profile = tmp_path / "profiles.yaml"
    profile.write_text(
        "routes:\n"
        "  m:\n"
        "    type: model\n"
        "    target:\n"
        "      model: a\n"
        "      base_url: http://x/v1\n"
        "      token_capture_engine: vllm\n"
    )
    captured: dict[str, list] = {}

    class _FakeTable:
        def registered_models(self) -> list[str]:
            return ["m"]

        def default_model(self) -> str | None:
            return None

    def _fake_load(
        routing_profiles,
        *,
        pre_routing_request_processors=(),
        extra_response_processors=(),
        **_kwargs,
    ):
        captured["request"] = list(pre_routing_request_processors)
        captured["response"] = list(extra_response_processors)
        return _FakeTable()

    monkeypatch.setattr(cli, "load_route_bundle_table", _fake_load)
    monkeypatch.setattr(cli, "build_and_serve", lambda *a, **k: None)

    args = argparse.Namespace(
        config=None,
        routing_profiles=str(profile),
        enable_rl_logging=True,
        rl_log_dir=str(tmp_path / "rl_data"),
        intake_enabled=False,
        intake_base_url=None,
        intake_workspace=None,
        intake_api_key=None,
        intake_nvdataflow_project=None,
    )
    cli._cmd_serve(args)

    assert [type(p).__name__ for p in captured["request"]] == [
        "RlLoggingRequestProcessor",
        "TokenCaptureRequestProcessor",
    ]
    assert [type(p).__name__ for p in captured["response"]] == [
        "TokenCaptureResponseProcessor",
    ]


def test_builder_disabled_returns_empty_lists() -> None:
    assert build_token_capture_processors(None) == ([], [])


def test_builder_returns_capture_pair(tmp_path: Path) -> None:
    request, response = build_token_capture_processors(tmp_path)
    assert [type(p).__name__ for p in request] == [
        "RlLoggingRequestProcessor",
        "TokenCaptureRequestProcessor",
    ]
    assert [type(p).__name__ for p in response] == ["TokenCaptureResponseProcessor"]


def test_build_launch_capture_processors_token_capture(tmp_path: Path) -> None:
    from switchyard.cli.launchers.launch_intake_config import build_launch_capture_processors

    request, response = build_launch_capture_processors(None, tmp_path, token_capture=True)
    assert [type(p).__name__ for p in request] == [
        "RlLoggingRequestProcessor",
        "TokenCaptureRequestProcessor",
    ]
    assert [type(p).__name__ for p in response] == ["TokenCaptureResponseProcessor"]


# ---------------------------------------------------------------------------
# Request processor: session resolution, caller-param strip, stream flip
# ---------------------------------------------------------------------------


def _streaming_request() -> ChatRequest:
    return ChatRequest.openai_chat(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )


async def test_streaming_request_flipped_when_session_present() -> None:
    ctx = _session_ctx()
    request = _streaming_request()
    await TokenCaptureRequestProcessor().process(ctx, request)

    body = dict(request.body)
    assert body["stream"] is False
    assert "stream_options" not in body
    assert ctx.metadata[CTX_TOKEN_CAPTURE_ORIGINAL_STREAM] is True
    assert ctx.metadata[CTX_TOKEN_CAPTURE_SESSION] == _SESSION


async def test_no_session_leaves_request_untouched() -> None:
    ctx = ProxyContext()
    request = _streaming_request()
    await TokenCaptureRequestProcessor().process(ctx, request)

    body = dict(request.body)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert CTX_TOKEN_CAPTURE_ORIGINAL_STREAM not in ctx.metadata
    assert CTX_TOKEN_CAPTURE_SESSION not in ctx.metadata


async def test_caller_token_params_stripped_when_session_present() -> None:
    ctx = _session_ctx()
    request = ChatRequest.openai_chat(
        {
            "model": "m",
            "messages": [],
            "logprobs": False,
            "top_logprobs": None,
            "return_token_ids": False,
        }
    )
    await TokenCaptureRequestProcessor().process(ctx, request)

    body = dict(request.body)
    # The target's derived extra_body params must win over caller-supplied ones.
    assert "logprobs" not in body
    assert "top_logprobs" not in body
    assert "return_token_ids" not in body


async def test_caller_token_params_kept_without_session() -> None:
    ctx = ProxyContext()
    request = ChatRequest.openai_chat(
        {"model": "m", "messages": [], "logprobs": False, "top_logprobs": None}
    )
    await TokenCaptureRequestProcessor().process(ctx, request)

    body = dict(request.body)
    assert body["logprobs"] is False
    assert body["top_logprobs"] is None


async def test_non_streaming_request_left_untouched() -> None:
    ctx = _session_ctx()
    request = ChatRequest.openai_chat({"model": "m", "messages": []})
    await TokenCaptureRequestProcessor().process(ctx, request)

    assert "stream" not in dict(request.body)
    assert CTX_TOKEN_CAPTURE_ORIGINAL_STREAM not in ctx.metadata


async def test_session_id_falls_back_to_launcher_shaped_caller_key(tmp_path: Path) -> None:
    from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

    ctx = ProxyContext()
    ctx.metadata[CTX_CALLER_API_KEY] = "openclaw-1783600121000-a1b2c3d4"
    await _run(tmp_path, _vllm_completion(), ctx=ctx)

    record = _read_only_record(tmp_path / "sessions" / "openclaw-1783600121000-a1b2c3d4")
    assert record["session_id"] == "openclaw-1783600121000-a1b2c3d4"


async def test_real_api_key_never_becomes_session_id(tmp_path: Path) -> None:
    from switchyard.lib.proxy_context import CTX_CALLER_API_KEY

    ctx = ProxyContext()
    ctx.metadata[CTX_CALLER_API_KEY] = "sk-real-secret-key-12345"
    await _run(tmp_path, _vllm_completion(), ctx=ctx)

    # Not launcher-shaped -> no session -> nothing captured, nothing leaked.
    assert list(tmp_path.rglob("*.json")) == []


# ---------------------------------------------------------------------------
# Response processor: unified record
# ---------------------------------------------------------------------------


async def test_unified_record_schema(tmp_path: Path) -> None:
    await _run(tmp_path, _vllm_completion(), session_id=_SESSION)
    record = _read_only_record(tmp_path / "sessions" / _SESSION)

    import uuid as uuid_lib
    from datetime import datetime

    assert record["schema_version"] == 1
    uuid_lib.UUID(record["uuid"])  # must parse as a valid UUID
    assert record["session_id"] == _SESSION
    # captured_at must parse as an ISO timestamp (retrieval sorts on it).
    datetime.fromisoformat(record["captured_at"])
    assert record["request_id"] == "chatcmpl-test"
    assert record["model"] == "Qwen/Qwen3-0.6B"
    # Text trace from the translated request snapshot + raw-body assistant turn.
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    assert record["messages"][0]["content"] == "hello"
    assert record["messages"][-1]["content"] == "hi"
    assert record["tools"] == []
    assert record["tool_choice"] == "auto"
    assert record["token_count"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    # Token-level fields from the raw vLLM body.
    assert record["prompt_token_ids"] == [1, 2, 3]
    assert record["generation_token_ids"] == [7, 8]
    assert record["generation_log_probs"] == [-0.1, -0.2]
    assert record["finish_reason"] == "stop"
    assert record["is_valid"] is True


async def test_record_file_is_owner_only(tmp_path: Path) -> None:
    await _run(tmp_path, _vllm_completion(), session_id=_SESSION)
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    assert files[0].stat().st_mode & 0o777 == 0o600


async def test_no_session_is_not_captured(tmp_path: Path) -> None:
    await _run(tmp_path, _vllm_completion())
    assert list(tmp_path.rglob("*.json")) == []


async def test_misaligned_logprobs_marks_record_invalid(tmp_path: Path) -> None:
    choices = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "token_ids": [7, 8],
            "logprobs": {"content": [{"token": "hi", "logprob": -0.1}]},
        }
    ]
    await _run(tmp_path, _vllm_completion(choices=choices), session_id=_SESSION)
    record = _read_only_record(tmp_path / "sessions" / _SESSION)

    assert record["is_valid"] is False
    assert record["generation_token_ids"] == [7, 8]
    assert record["generation_log_probs"] == [-0.1]


async def test_multiple_choices_marks_record_invalid(tmp_path: Path) -> None:
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop",
        "token_ids": [7, 8],
        "logprobs": {
            "content": [
                {"token": "hi", "logprob": -0.1},
                {"token": "!", "logprob": -0.2},
            ]
        },
    }
    await _run(
        tmp_path,
        _vllm_completion(choices=[choice, {**choice, "index": 1}]),
        session_id=_SESSION,
    )
    record = _read_only_record(tmp_path / "sessions" / _SESSION)
    assert record["is_valid"] is False


async def test_missing_token_fields_mark_record_invalid(tmp_path: Path) -> None:
    """A non-engine backend (no token fields) still yields a stored, invalid record."""
    choices = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ]
    await _run(tmp_path, _vllm_completion(choices=choices), session_id=_SESSION)
    record = _read_only_record(tmp_path / "sessions" / _SESSION)
    assert record["is_valid"] is False
    assert record["messages"][-1]["content"] == "hi"


async def test_non_finite_logprobs_mark_record_invalid(tmp_path: Path) -> None:
    choices = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "token_ids": [7, 8],
            "logprobs": {
                "content": [
                    {"token": "hi", "logprob": float("nan")},
                    {"token": "!", "logprob": -0.2},
                ]
            },
        }
    ]
    await _run(tmp_path, _vllm_completion(choices=choices), session_id=_SESSION)
    record = _read_only_record(tmp_path / "sessions" / _SESSION)
    assert record["is_valid"] is False


async def test_non_int_token_ids_mark_record_invalid(tmp_path: Path) -> None:
    choices = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "token_ids": [7.5, "8"],
            "logprobs": {
                "content": [
                    {"token": "hi", "logprob": -0.1},
                    {"token": "!", "logprob": -0.2},
                ]
            },
        }
    ]
    await _run(tmp_path, _vllm_completion(choices=choices), session_id=_SESSION)
    record = _read_only_record(tmp_path / "sessions" / _SESSION)
    assert record["is_valid"] is False


async def test_synthetic_stream_preserves_tool_calls(tmp_path: Path) -> None:
    from switchyard_rust.core import ChatResponseType, response_type_matches

    choices = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": '{"path": "hello.py"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "token_ids": [7, 8],
            "logprobs": {
                "content": [
                    {"token": "a", "logprob": -0.1},
                    {"token": "b", "logprob": -0.2},
                ]
            },
        }
    ]
    ctx = _session_ctx()
    request = _streaming_request()
    await RlLoggingRequestProcessor().process(ctx, request)
    await TokenCaptureRequestProcessor().process(ctx, request)
    out = await TokenCaptureResponseProcessor(tmp_path).process(
        ctx, _vllm_completion(choices=choices)
    )

    assert response_type_matches(out, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk async for chunk in out.stream]
    delta_calls = chunks[0].choices[0].delta.tool_calls
    assert delta_calls is not None and len(delta_calls) == 1
    assert delta_calls[0].id == "call_abc123"
    assert delta_calls[0].index == 0
    assert delta_calls[0].function.name == "write_file"
    assert delta_calls[0].function.arguments == '{"path": "hello.py"}'
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


async def test_streaming_response_is_skipped(tmp_path: Path) -> None:
    async def _iter():
        return
        yield  # pragma: no cover

    from switchyard.lib.chat_response import ResponseStream

    ctx = ProxyContext()
    ctx.metadata[CTX_TOKEN_CAPTURE_SESSION] = _SESSION
    response = ChatResponse.openai_stream(ResponseStream(_iter()))
    out = await TokenCaptureResponseProcessor(tmp_path).process(ctx, response)

    assert out is response
    assert list(tmp_path.rglob("*.json")) == []


async def test_synthetic_stream_replaces_buffered_response(tmp_path: Path) -> None:
    ctx = await _run(
        tmp_path,
        _vllm_completion(),
        session_id=_SESSION,
        request=_streaming_request(),
    )
    assert ctx.metadata[CTX_TOKEN_CAPTURE_ORIGINAL_STREAM] is True

    # The record was still captured from the buffered body.
    record = _read_only_record(tmp_path / "sessions" / _SESSION)
    assert record["generation_token_ids"] == [7, 8]


async def test_synthetic_stream_chunks(tmp_path: Path) -> None:
    from switchyard_rust.core import ChatResponseType, response_type_matches

    ctx = _session_ctx()
    request = _streaming_request()
    await RlLoggingRequestProcessor().process(ctx, request)
    await TokenCaptureRequestProcessor().process(ctx, request)
    out = await TokenCaptureResponseProcessor(tmp_path).process(ctx, _vllm_completion())

    # The client gets an OpenAI chunk stream reproducing the completion.
    assert response_type_matches(out, ChatResponseType.OPENAI_STREAM)
    chunks = [chunk async for chunk in out.stream]
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "hi"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[1].usage.total_tokens == 5


async def test_buffered_request_gets_buffered_response(tmp_path: Path) -> None:
    from switchyard_rust.core import ChatResponseType, response_type_matches

    response = _vllm_completion()
    ctx = _session_ctx()
    await RlLoggingRequestProcessor().process(ctx, _anthropic_request())
    await TokenCaptureRequestProcessor().process(ctx, _anthropic_request())
    result = await TokenCaptureResponseProcessor(tmp_path).process(ctx, response)

    assert result is response
    assert response_type_matches(result, ChatResponseType.OPENAI_COMPLETION)


# ---------------------------------------------------------------------------
# Retrieval endpoint
# ---------------------------------------------------------------------------


def _retrieval_client(capture_dir: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    processor = TokenCaptureResponseProcessor(capture_dir)
    app = FastAPI()
    processor.get_endpoint().register(app)
    return TestClient(app, raise_server_exceptions=False)


async def test_retrieval_endpoint_serves_session_records(tmp_path: Path) -> None:
    await _run(tmp_path, _vllm_completion(), session_id=_SESSION)
    await _run(tmp_path, _vllm_completion(), session_id=_SESSION)
    client = _retrieval_client(tmp_path)

    result = client.get(f"/v1/sessions/{_SESSION}/completions")
    assert result.status_code == 200
    payload = result.json()
    assert payload["schema_version"] == 1
    assert payload["session_id"] == _SESSION
    assert len(payload["completions"]) == 2
    assert payload["completions"][0]["generation_token_ids"] == [7, 8]


def test_retrieval_completions_sorted_by_captured_at_then_uuid(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / _SESSION
    session_dir.parent.mkdir(exist_ok=True)
    session_dir.mkdir()
    # Filenames deliberately reverse the capture order.
    (session_dir / "z.json").write_text(
        json.dumps({"captured_at": "2026-01-01T00:00:00+00:00", "uuid": "aaa"})
    )
    (session_dir / "a.json").write_text(
        json.dumps({"captured_at": "2026-01-02T00:00:00+00:00", "uuid": "bbb"})
    )
    (session_dir / "m.json").write_text(
        json.dumps({"captured_at": "2026-01-01T00:00:00+00:00", "uuid": "zzz"})
    )
    client = _retrieval_client(tmp_path)

    payload = client.get(f"/v1/sessions/{_SESSION}/completions").json()
    assert [(r["captured_at"], r["uuid"]) for r in payload["completions"]] == [
        ("2026-01-01T00:00:00+00:00", "aaa"),
        ("2026-01-01T00:00:00+00:00", "zzz"),
        ("2026-01-02T00:00:00+00:00", "bbb"),
    ]


async def test_retrieval_skips_torn_record_and_serves_the_rest(tmp_path: Path) -> None:
    await _run(tmp_path, _vllm_completion(), session_id=_SESSION)
    # Simulate a torn/corrupt record alongside the good one.
    (tmp_path / "sessions" / _SESSION / "0000-torn.json").write_text('{"schema_version": 1, "uuid": ')
    client = _retrieval_client(tmp_path)

    payload = client.get(f"/v1/sessions/{_SESSION}/completions").json()
    assert len(payload["completions"]) == 1
    assert payload["completions"][0]["generation_token_ids"] == [7, 8]


def test_retrieval_endpoint_unknown_session_is_404(tmp_path: Path) -> None:
    client = _retrieval_client(tmp_path)
    assert client.get("/v1/sessions/nope/completions").status_code == 404


def test_retrieval_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    # A sentinel record one level above sessions/ — a vulnerable join would
    # surface it. Encoded ".." reaches the handler un-normalized (plain "../"
    # is collapsed by the client before routing, so it only proves routing).
    (tmp_path / "secret.json").write_text('{"leaked": true}')
    client = _retrieval_client(tmp_path)

    resp = client.get("/v1/sessions/%2E%2E/completions")
    assert resp.status_code == 404
    assert "leaked" not in resp.text


def test_retrieval_endpoint_registers_once() -> None:
    from switchyard.lib.endpoints.token_capture_endpoint import (
        TokenCaptureSessionsEndpoint,
    )

    assert TokenCaptureSessionsEndpoint.register_once is True


# ---------------------------------------------------------------------------
# Route bundle + launcher wiring
# ---------------------------------------------------------------------------


def test_token_capture_engine_stripped_when_capture_disabled() -> None:
    from switchyard.cli.route_bundle import _strip_token_capture_engine

    raw = {
        "routes": {
            "m1": {
                "type": "model",
                "target": {"model": "a", "base_url": "http://x/v1", "token_capture_engine": "vllm"},
            },
        },
    }
    stripped = _strip_token_capture_engine(raw)
    assert "token_capture_engine" not in stripped["routes"]["m1"]["target"]
    # Original mapping untouched; other keys intact.
    assert raw["routes"]["m1"]["target"]["token_capture_engine"] == "vllm"
    assert stripped["routes"]["m1"]["target"]["model"] == "a"


def test_capture_session_headers() -> None:
    from switchyard.cli.launchers.launch_intake_config import (
        LaunchIntakeConfig,
        capture_session_headers,
    )

    # Capture off -> no headers.
    assert capture_session_headers(None, False, "claude") == {}
    # Intake on -> its own headers already carry the session id.
    intake = LaunchIntakeConfig.from_resolved(
        base_url=None,
        workspace=None,
        api_key=None,
        app="a",
        task="t",
        session_id="s1",
        target="claude",
    )
    assert capture_session_headers(intake, True, "claude") == {}
    # Capture on, intake off -> per-launch session header.
    headers = capture_session_headers(None, True, "claude")
    assert set(headers) == {"proxy_x_session_id"}
    assert headers["proxy_x_session_id"].startswith("claude-")


def test_capture_session_id_for_api_key_covers_intake_on() -> None:
    from switchyard.cli.launchers.launch_intake_config import (
        LaunchIntakeConfig,
        capture_session_id_for_api_key,
    )

    intake = LaunchIntakeConfig.from_resolved(
        base_url=None,
        workspace=None,
        api_key=None,
        app="a",
        task="t",
        session_id="openclaw-1700000000000-abcd1234",
        target="openclaw",
    )
    # Capture off -> no session id regardless of intake.
    assert capture_session_id_for_api_key(intake, False, "openclaw") is None
    # Capture on + intake on -> intake's session id rides the key (headers
    # can't reach OpenClaw).
    assert (
        capture_session_id_for_api_key(intake, True, "openclaw")
        == "openclaw-1700000000000-abcd1234"
    )
    # Capture on, intake off -> fresh launcher-shaped id.
    generated = capture_session_id_for_api_key(None, True, "openclaw")
    assert generated is not None and generated.startswith("openclaw-")


def test_openclaw_env_carries_capture_session_id_as_api_key() -> None:
    from switchyard.cli.launchers.openclaw_launcher import _openclaw_env

    env = _openclaw_env(
        workspace="/tmp/ws",
        capture_session_id="openclaw-1783600121000-a1b2c3d4",
    )
    assert env["SWITCHYARD_API_KEY"] == "openclaw-1783600121000-a1b2c3d4"

    # Without capture, the opaque placeholder is unchanged.
    assert _openclaw_env(workspace="/tmp/ws")["SWITCHYARD_API_KEY"] == "switchyard"


def test_claude_env_carries_capture_session_header() -> None:
    from switchyard.cli.launchers.claude_code_launcher import _claude_env

    env = _claude_env(1234, "m", capture_headers={"proxy_x_session_id": "claude-1-ab"})
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "proxy_x_session_id: claude-1-ab"
    assert env["SWITCHYARD_SESSION_ID"] == "claude-1-ab"

    # Without capture headers the env is unchanged from the intake-off default.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in _claude_env(1234, "m")
