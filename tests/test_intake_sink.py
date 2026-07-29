# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the intake sink usage case."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.config import IntakeSinkConfig
from switchyard.lib.processors import (
    IntakeClient,
    IntakePayloadBuilder,
    IntakeRequestProcessor,
    IntakeResponseProcessor,
)
from switchyard.lib.processors.intake_payload_builder import (
    INTAKE_ENDED_AT_MS_KEY,
    INTAKE_INBOUND_FORMAT_KEY,
    INTAKE_SESSION_ID_KEY,
    INTAKE_STARTED_AT_MS_KEY,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import (
    CTX_REQUEST_METADATA,
    IntakeRequestMetadata,
    RequestMetadata,
    attach_request_metadata,
)
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import (
    ChatRequest,
    ChatResponse,
)
from switchyard_rust.translation import TranslationEngine


def _assert_chat_completions_ingest_shape(payload: dict[str, Any]) -> None:
    assert "request" in payload
    assert "response" in payload
    assert payload["provider"] == "switchyard"

    # nemo-platform chat-completions ingest forbids legacy entry-envelope fields.
    for key in (
        "data",
        "context",
        "external_id",
        "usage",
    ):
        assert key not in payload


def _completion(
    *,
    response_id: str = "chatcmpl-test",
    model: str = "openai/openai/gpt-5.2",
    content: str = "hello",
    include_usage: bool = True,
) -> ChatCompletion:
    return ChatCompletion(
        id=response_id,
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=(
            CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            if include_usage
            else None
        ),
    )


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: CompletionUsage | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-stream",
        object="chat.completion.chunk",
        created=1700000000,
        model="openai/openai/gpt-5.2",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _responses_response_dict(
    *,
    response_id: str = "resp_123",
    model: str = "gpt-4o",
    text: str = "hello",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class _FakeIntakeClient:
    def __init__(
        self,
        effective_config: IntakeSinkConfig | None = None,
    ) -> None:
        self.enqueue = AsyncMock()
        self.enqueue_background = Mock(side_effect=self._enqueue_background)
        self.aclose = AsyncMock()
        self.background_payloads: list[dict[str, object]] = []
        self.background_errors: list[Exception] = []
        self.effective_config = effective_config or IntakeSinkConfig(
            intake_base_url="http://localhost:8080",
            workspace="default",
            user_id="brian",
        )

    def _enqueue_background(self, payload_factory):
        try:
            self.background_payloads.append(payload_factory())
        except Exception as exc:
            self.background_errors.append(exc)


class _IntakeHttpStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []

    def __enter__(self) -> _IntakeHttpStub:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                with owner._lock:
                    owner._requests.append({
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "body": json.loads(raw.decode("utf-8")),
                    })
                response = b"{}"
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(response)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)


class _StaticCompletionBackend(LLMBackend):
    async def call(self, _ctx: ProxyContext, _request: ChatRequest) -> ChatResponse:
        return ChatResponse.openai_completion(_completion(content="endpoint-ok"))


def _stub_sdk_client(monkeypatch) -> AsyncMock:
    """Replace ``_build_sdk_client`` so IntakeClient tests don't need the SDK installed."""
    fake = AsyncMock()
    fake.close = AsyncMock()
    fake.workspace = "default"
    monkeypatch.setattr(
        "switchyard.lib.processors.intake_client._build_sdk_client",
        lambda config: fake,
    )
    return fake


class TestRequestMetadata:
    def test_from_headers_extracts_only_explicit_fields(self):
        metadata = RequestMetadata.from_headers({
            "proxy_x_session_id": "sess-1",
            "x-switchyard-intake-enabled": "true",
            "x-switchyard-intake-app": "log2/codex",
            "x-switchyard-intake-task": "developer-session",
            "authorization": "Bearer secret",
        })

        assert metadata.session_id == "sess-1"
        assert metadata.intake.enabled is True
        assert metadata.intake.app == "log2/codex"
        assert metadata.intake.task == "developer-session"


class TestIntakePayloadBuilder:
    def test_responses_request_is_normalized_to_openai_shape(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
                capture_content=True,
            )
        )
        request = ChatRequest.openai_responses({
            "model": "openai/openai/gpt-5.2",
            "input": "say hi",
        })
        ctx = ProxyContext(metadata={
            CTX_REQUEST_METADATA: RequestMetadata(
                intake=IntakeRequestMetadata(
                    app="codex",
                    task="developer-session",
                ),
            ),
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
            "_proxy_actual_model": "openai/openai/gpt-5.2",
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion()),
            stream=False,
        )

        assert payload["request"]["messages"][0]["content"] == "say hi"
        assert "app" not in payload["request"]["switchyard"]
        assert "task" not in payload["request"]["switchyard"]
        assert payload["response"]["id"] == "chatcmpl-test"
        assert payload["provider"] == "switchyard"
        assert "user_id" not in payload
        assert "switchyard" not in payload["response"]
        _assert_chat_completions_ingest_shape(payload)

    def test_metadata_only_by_default_redacts_content(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(intake_base_url="http://localhost:8080", workspace="default")
        )
        request = ChatRequest.openai_chat({
            "model": "openai/openai/gpt-5.2",
            "messages": [{"role": "user", "content": "SENTINEL_PROMPT"}],
            "tools": [{"type": "function", "function": {"name": "SENTINEL_TOOL"}}],
        })
        ctx = ProxyContext(metadata={})

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion(content="SENTINEL_RESPONSE")),
            stream=False,
        )

        serialized = json.dumps(payload)
        for sentinel in ("SENTINEL_PROMPT", "SENTINEL_TOOL", "SENTINEL_RESPONSE"):
            assert sentinel not in serialized, f"leaked content: {sentinel}"
        assert "messages" not in payload["request"]
        assert payload["response"]["choices"][0]["message"].get("content") is None
        assert payload["response"]["model"] == "openai/openai/gpt-5.2"
        assert payload["response"]["usage"]["prompt_tokens"] == 10

    def test_synthetic_stream_id_is_stripped_before_ingest(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        request = ChatRequest.openai_chat({
            "model": "openai/openai/gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext(metadata={
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
            "_proxy_actual_model": "openai/openai/gpt-5.2",
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(
                _completion(response_id="chatcmpl-switchyard-stream"),
            ),
            stream=True,
        )

        assert "id" not in payload["response"]

    def test_missing_served_model_yields_null_usage_model(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        request = ChatRequest.openai_chat({
            "model": "openai/openai/gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext(metadata={
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion()),
            stream=False,
        )

        assert "served_model" not in payload["request"]["switchyard"]
        assert "switchyard" not in payload["response"]

    def test_response_usage_carries_tokens_and_request_metadata_carries_timing(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext(metadata={
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
            "_proxy_actual_model": "gpt-4o",
            INTAKE_STARTED_AT_MS_KEY: 1_700_000_000_000,
            INTAKE_ENDED_AT_MS_KEY: 1_700_000_001_840,
            INTAKE_SESSION_ID_KEY: "session-123",
            CTX_REQUEST_METADATA: RequestMetadata(
                intake=IntakeRequestMetadata(
                    app="codex",
                    task="developer-session",
                ),
            ),
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion()),
            stream=False,
        )

        assert payload["response"]["usage"]["prompt_tokens"] == 10
        assert payload["response"]["usage"]["completion_tokens"] == 5
        assert payload["cost_usd"] == pytest.approx(0.000088)
        assert payload["cost_input_usd"] == pytest.approx(0.000018)
        assert payload["cost_output_usd"] == pytest.approx(0.00007)
        assert payload["cost_details"]["base_input"] == pytest.approx(0.000018)
        assert payload["cost_details"]["cached_input"] == pytest.approx(0.0)
        assert payload["cost_details"]["cache_write"] == pytest.approx(0.0)

        request_switchyard = payload["request"]["switchyard"]
        assert payload["evaluation_context"] == {
            "evaluation_run_id": "session-123",
            "test_case_id": "developer-session",
        }
        assert "served_model" not in request_switchyard
        assert "started_at_ms" not in request_switchyard
        assert "ended_at_ms" not in request_switchyard
        assert "duration_ms" not in request_switchyard
        assert request_switchyard["latency_ms"] == 1840
        assert request_switchyard["stream"] is False
        assert request_switchyard["inbound_format"] == request.request_type

        assert "switchyard" not in payload["response"]
        _assert_chat_completions_ingest_shape(payload)

    def test_chat_completions_payload_omits_cost_for_unknown_model(self):
        """Unknown model aliases do not invent chat-completions cost fields."""
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        request = ChatRequest.openai_chat({
            "model": "made-up-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext(metadata={
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
            "_proxy_actual_model": "made-up-model",
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion(model="made-up-model")),
            stream=False,
        )

        assert payload["response"]["model"] == "made-up-model"
        assert payload["response"]["usage"]["prompt_tokens"] == 10
        assert payload["response"]["usage"]["completion_tokens"] == 5
        assert "cost_usd" not in payload
        assert "cost_input_usd" not in payload
        assert "cost_output_usd" not in payload
        assert "cost_details" not in payload
        _assert_chat_completions_ingest_shape(payload)

    def test_chat_completions_payload_omits_cost_when_tokens_are_missing_for_known_model(self):
        builder = IntakePayloadBuilder(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        request = ChatRequest.openai_chat({
            "model": "openai/openai/gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext(metadata={
            INTAKE_INBOUND_FORMAT_KEY: request.request_type,
            "_proxy_actual_model": "openai/openai/gpt-5.2",
        })

        payload = builder.build(
            ctx=ctx,
            request_snapshot=request,
            response=ChatResponse.openai_completion(_completion(include_usage=False)),
            stream=False,
        )

        assert payload["response"]["model"] == "openai/openai/gpt-5.2"
        assert "usage" not in payload["response"] or payload["response"]["usage"] is None
        _assert_chat_completions_ingest_shape(payload)


class TestIntakeRequestProcessor:
    async def test_returns_request_unchanged_without_client_opt_in(self):
        processor = IntakeRequestProcessor()
        ctx = ProxyContext(metadata={
            CTX_REQUEST_METADATA: RequestMetadata(
                session_id="session-123",
            ),
        })
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        })

        returned = await processor.process(ctx, request)

        assert returned.body == request.body

    async def test_store_true_request_still_round_trips_through_rust_processor(self):
        processor = IntakeRequestProcessor()
        ctx = ProxyContext(metadata={
            CTX_REQUEST_METADATA: RequestMetadata(),
        })
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "store": True,
        })

        returned = await processor.process(ctx, request)

        assert returned.body == request.body
        assert returned.body["store"] is True

    async def test_header_false_request_still_round_trips_through_rust_processor(self):
        processor = IntakeRequestProcessor()
        ctx = ProxyContext(metadata={
            CTX_REQUEST_METADATA: RequestMetadata(
                intake=IntakeRequestMetadata(enabled=False),
            ),
        })
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "store": True,
        })

        returned = await processor.process(ctx, request)

        assert returned.body == request.body

    async def test_constructor_metadata_opt_out_overrides_store_true_in_rust_context(self):
        request_processor = IntakeRequestProcessor()
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "store": True,
        })
        ctx = ProxyContext(metadata={
            CTX_REQUEST_METADATA: RequestMetadata(
                intake=IntakeRequestMetadata(enabled=False),
            ),
        })

        with _IntakeHttpStub() as server:
            response_processor = IntakeResponseProcessor(
                IntakeSinkConfig(
                    intake_base_url=server.base_url,
                    workspace="default",
                    max_retries=0,
                )
            )
            await request_processor.process(ctx, request)
            await response_processor.process(ctx, ChatResponse.openai_completion(_completion()))
            await response_processor.shutdown()

        assert server.requests == []

    async def test_header_opt_in_attached_to_rust_context_posts_payload(self):
        request_processor = IntakeRequestProcessor()
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        })
        ctx = ProxyContext()
        attach_request_metadata(
            ctx,
            RequestMetadata.from_headers({
                "proxy_x_session_id": "session-123",
                "x-switchyard-intake-enabled": "true",
                "x-switchyard-intake-app": "codex",
                "x-switchyard-intake-task": "developer-session",
            }),
        )

        with _IntakeHttpStub() as server:
            response_processor = IntakeResponseProcessor(
                IntakeSinkConfig(
                    intake_base_url=server.base_url,
                    workspace="default",
                    max_retries=0,
                )
            )
            await request_processor.process(ctx, request)
            await response_processor.process(ctx, ChatResponse.openai_completion(_completion()))
            await response_processor.shutdown()

        assert len(server.requests) == 1
        payload = server.requests[0]["body"]
        assert "app" not in payload["request"]["switchyard"]
        assert "task" not in payload["request"]["switchyard"]
        assert payload["session_id"] == "session-123"

    async def test_store_true_posts_routed_turn_and_submodel_records_to_flat_target(self):
        """`store: true` alone (no header) opts in, so the routed turn and every
        recorded routing call post to the flat-document target as distinct
        documents. This is the opt-in the response-side Python processor could
        never see."""
        request_processor = IntakeRequestProcessor()
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "store": True,
        })
        ctx = ProxyContext()
        attach_request_metadata(
            ctx,
            RequestMetadata.from_headers({"proxy_x_session_id": "sess-xyz"}),
        )
        # A router records its own model call, exactly as the LLM classifier does.
        ctx.record_submodel_call(
            model="router-clf",
            prompt_tokens=120,
            completion_tokens=8,
            cached_tokens=0,
            router_type="deterministic",
            routed_to="classifier",
        )

        with _IntakeHttpStub() as server:
            response_processor = IntakeResponseProcessor(
                IntakeSinkConfig(
                    target_url=f"{server.base_url}/lake/switchyard/posting",
                    max_retries=0,
                )
            )
            await request_processor.process(ctx, request)
            await response_processor.process(
                ctx, ChatResponse.openai_completion(_completion())
            )
            await response_processor.shutdown()

        # One routed-turn document plus one per routing call, all flat documents.
        posted = server.requests
        assert len(posted) == 2
        for entry in posted:
            assert entry["path"] == "/lake/switchyard/posting"
        # Distinct _id so the routing record never overwrites the routed turn.
        assert len({entry["body"]["_id"] for entry in posted}) == 2
        routed = {entry["body"].get("s_switchyard_routed_to") for entry in posted}
        assert "classifier" in routed

    async def test_header_opt_out_overrides_store_true_in_rust_context(self):
        request_processor = IntakeRequestProcessor()
        request = ChatRequest.openai_chat({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "store": True,
        })
        ctx = ProxyContext()
        attach_request_metadata(
            ctx,
            RequestMetadata.from_headers({
                "x-switchyard-intake-enabled": "false",
            }),
        )

        with _IntakeHttpStub() as server:
            response_processor = IntakeResponseProcessor(
                IntakeSinkConfig(
                    intake_base_url=server.base_url,
                    workspace="default",
                    max_retries=0,
                )
            )
            await request_processor.process(ctx, request)
            await response_processor.process(ctx, ChatResponse.openai_completion(_completion()))
            await response_processor.shutdown()

        assert server.requests == []

    async def test_http_endpoint_attaches_headers_to_rust_context(self):
        request_processor = IntakeRequestProcessor()

        with _IntakeHttpStub() as server:
            response_processor = IntakeResponseProcessor(
                IntakeSinkConfig(
                    intake_base_url=server.base_url,
                    workspace="default",
                    max_retries=0,
                    capture_content=True,
                )
            )
            switchyard = Switchyard(
                request_processors=[request_processor],
                backend=_StaticCompletionBackend(),
                response_processors=[response_processor],
                translator=TranslationEngine(),
            )
            app = build_switchyard_app(switchyard)

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "x-switchyard-intake-enabled": "true",
                        "x-switchyard-intake-app": "codex",
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            await response_processor.shutdown()

        assert response.status_code == 200, response.text
        assert len(server.requests) == 1
        payload = server.requests[0]["body"]
        assert "app" not in payload["request"]["switchyard"]
        assert payload["request"]["messages"][0]["content"] == "hi"


class TestIntakeResponseProcessor:
    def test_requires_http_sink_base_url(self) -> None:
        with pytest.raises(RuntimeError, match="intake_base_url"):
            IntakeResponseProcessor(IntakeSinkConfig())

    async def test_missing_request_state_leaves_response_untouched(self):
        processor = IntakeResponseProcessor(
            IntakeSinkConfig(
                intake_base_url="http://127.0.0.1:9",
                workspace="default",
            )
        )
        response = ChatResponse.openai_completion(_completion(content="world"))
        ctx = ProxyContext()

        returned = await processor.process(ctx, response)

        assert returned.body == response.body


class TestIntakeClient:
    async def test_aclose_flushes_background_payloads(self, monkeypatch):
        _stub_sdk_client(monkeypatch)
        client = IntakeClient(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="default",
            )
        )
        posted: list[dict[str, object]] = []

        async def fake_post(payload: dict[str, object]) -> None:
            posted.append(payload)

        monkeypatch.setattr(client, "_post_payload", fake_post)

        client.enqueue_background(lambda: {"ok": True})
        await client.aclose()

        assert posted == [{"ok": True}]

    async def test_post_payload_uses_sdk_generic_post(self, monkeypatch):
        sdk_client = _stub_sdk_client(monkeypatch)
        client = IntakeClient(
            IntakeSinkConfig(
                intake_base_url="http://localhost:8080",
                workspace="team space",
            )
        )
        payload = {"request": {"model": "m", "messages": []}, "response": {"choices": []}}

        await client._post_payload(payload)

        sdk_client.post.assert_awaited_once_with(
            "/apis/intake/v2/workspaces/team%20space/ingest/chat-completions",
            cast_to=object,
            body=payload,
        )

    def test_missing_sdk_raises_clear_error(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "nemo_platform", None)
        with pytest.raises(RuntimeError, match="NeMo Platform SDK"):
            IntakeClient(
                IntakeSinkConfig(
                    intake_base_url="http://localhost:8080",
                    workspace="default",
                )
            )
