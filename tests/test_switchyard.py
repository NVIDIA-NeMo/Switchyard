# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Switchyard executor."""

import asyncio

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard_rust.components import (
    IntakeRequestProcessor,
)
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    SwitchyardContextWindowExceededError,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_completion(*, model: str = "gpt-4o", content: str = "hello") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
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
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_request(*, model: str = "gpt-4o") -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })


# ---------------------------------------------------------------------------
# Test helpers — concrete processors for testing
# ---------------------------------------------------------------------------


class ModelOverrideProcessor:
    """Mutates the model field on the request body."""

    def __init__(self, new_model: str) -> None:
        self._new_model = new_model

    async def process(self, ctx, request):
        request.set_model(self._new_model)
        return request


class MetadataTagProcessor:
    """Writes a tag into ctx.metadata to prove it ran."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    async def process(self, ctx, request):
        ctx.metadata[self._tag] = True
        return request


class MetadataAssertResponseProcessor:
    """Asserts request metadata remains visible after Rust-native components run."""

    def __init__(self, expected: dict[str, object]) -> None:
        self._expected = expected

    async def process(self, ctx, response):
        for key, expected in self._expected.items():
            assert ctx.metadata[key] == expected
        return response


class ModelMetadataProcessor:
    async def process(self, ctx, request):
        ctx.metadata["seen_model"] = request.model
        return request


class ModelMetadataAssertProcessor:
    async def process(self, ctx, response):
        assert ctx.metadata["seen_model"] == response.body["model"]
        return response


class FailingRequestProcessor:
    async def process(self, ctx, request):
        ctx.metadata["request_started"] = request.model
        raise RuntimeError("request processor exploded")


class InvalidResponseProcessor:
    async def process(self, ctx, response):
        ctx.metadata["response_started"] = True
        return {"not": "a ChatResponse"}


class ContextReadingStreamTapProcessor:
    def __init__(self) -> None:
        self.seen_models: list[str | None] = []

    async def process(self, ctx, response):
        ctx.selected_model = "stream-selected"

        async def tap(_event):
            self.seen_models.append(ctx.selected_model)

        response.stream.tap(tap)
        return response


class ContentUpperProcessor:
    """Uppercases the first choice's content (non-streaming only)."""

    async def process(self, ctx, response):
        if response_type_matches(response, ChatResponseType.OPENAI_COMPLETION):
            body = response.body
            body["choices"][0]["message"]["content"] = (
                body["choices"][0]["message"].get("content") or ""
            ).upper()
            response.replace_body(body)
        return response


class RecordingBackend(LLMBackend):
    """Returns a canned OpenAI response and records processed requests."""

    def __init__(self, *, content: str = "base") -> None:
        self._content = content
        self.requests: list[ChatRequest] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.model:
            ctx.selected_model = request.model
        return ChatResponse.openai_completion(
            make_completion(model=request.model or "test-model", content=self._content)
        )


class LegacyBackendWithoutSupportedTypes(LLMBackend):
    """Compatibility backend that relies on the legacy Python executor surface."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.model:
            ctx.selected_model = request.model
        return ChatResponse.openai_completion(
            make_completion(model=request.model or "test-model", content="legacy")
        )


class FailingBackend(LLMBackend):
    """Raises after mutating context so restoration-on-error is observable."""

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        ctx.metadata["backend_started"] = request.model
        raise RuntimeError("backend exploded")


class RecordingTranslator:
    """Records the request the executor hands to the terminal translator."""

    def __init__(self) -> None:
        self.request_model: str | None = None

    async def translate(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        response: ChatResponse,
    ):
        _ = ctx
        self.request_model = request.model
        return response.body


async def _single_chunk_stream():
    yield {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ],
    }


class StreamingBackend(LLMBackend):
    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        _ = ctx, request
        return ChatResponse.openai_stream(_single_chunk_stream())


class PickWeakProcessor:
    """Routes the first attempt to the weak target."""

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.selected_target = "weak"
        return request


class OverflowWeakBackend(LLMBackend):
    """Overflows on weak and succeeds once the compatibility chain rewrites to strong."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.calls.append(ctx.selected_target)
        if ctx.selected_target == "weak":
            error = SwitchyardContextWindowExceededError("weak target overflowed")
            error.target_id = "weak"
            error.model = "weak-model"
            raise error
        ctx.selected_model = "strong-model"
        return ChatResponse.openai_completion(
            make_completion(model=request.model or "strong-model", content="fallback")
        )


class ExceptionOnlyOverflowBackend(LLMBackend):
    """Overflows with only an exception target id, then succeeds on fallback."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.calls.append(ctx.selected_target)
        if len(self.calls) == 1:
            error = SwitchyardContextWindowExceededError("weak target overflowed")
            error.target_id = "weak"
            error.model = "weak-model"
            raise error
        ctx.selected_model = "strong-model"
        return ChatResponse.openai_completion(
            make_completion(model=request.model or "strong-model", content="fallback")
        )


# ---------------------------------------------------------------------------
# Switchyard executor
# ---------------------------------------------------------------------------


class TestSwitchyard:
    def _make_chain(self, **overrides):
        defaults = {
            "backend": RecordingBackend(content="base"),
            "translator": TranslationEngine(),
        }
        defaults.update(overrides)
        return Switchyard(**defaults)

    def test_public_chain_classes_are_switchyard_rust_compatibility_exports(self):
        from switchyard_rust import core as rust_core

        assert Switchyard is rust_core.Switchyard

    async def test_minimal_chain(self):
        """Backend + translator only, no processors."""
        chain = self._make_chain()
        result = await chain.call(make_request())
        assert result["choices"][0]["message"]["content"] == "base"

    async def test_python_backend_without_supported_request_types_keeps_old_switchyard_behavior(
        self,
    ):
        backend = LegacyBackendWithoutSupportedTypes()
        chain = self._make_chain(backend=backend)

        result = await chain.call(make_request(model="gpt-4o"))

        assert result["choices"][0]["message"]["content"] == "legacy"
        assert backend.requests[-1].request_type == ChatRequestType.OPENAI_CHAT

    async def test_python_backend_without_supported_request_types_accepts_legacy_anthropic_request(
        self,
    ):
        """The compatibility fallback is intentionally all formats, matching old Python."""
        backend = LegacyBackendWithoutSupportedTypes()
        chain = self._make_chain(
            backend=backend,
            translator=RecordingTranslator(),
        )
        request = ChatRequest.anthropic({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
        })

        result = await chain.call(request)

        assert result["choices"][0]["message"]["content"] == "legacy"
        assert backend.requests[-1].request_type == ChatRequestType.ANTHROPIC

    async def test_request_processor_runs(self):
        chain = self._make_chain(
            request_processors=[ModelOverrideProcessor("gpt-4o-mini")],
        )
        await chain.call(make_request(model="gpt-4o"))
        backend = chain._backend
        assert isinstance(backend, RecordingBackend)
        assert backend.requests[-1].model == "gpt-4o-mini"

    async def test_multiple_request_processors_chain(self):
        chain = self._make_chain(
            request_processors=[
                MetadataTagProcessor("first"),
                MetadataTagProcessor("second"),
            ],
        )
        await chain.call(make_request())

    async def test_response_processor_runs(self):
        chain = self._make_chain(
            response_processors=[ContentUpperProcessor()],
        )
        result = await chain.call(make_request())
        assert result["choices"][0]["message"]["content"] == "BASE"

    async def test_full_chain(self):
        """Request processors → backend → response processors → translator."""
        chain = self._make_chain(
            request_processors=[
                MetadataTagProcessor("tagged"),
                ModelOverrideProcessor("gpt-4o-mini"),
            ],
            response_processors=[ContentUpperProcessor()],
        )
        result = await chain.call(make_request())
        assert result["choices"][0]["message"]["content"] == "BASE"

    async def test_translator_receives_processed_request_from_compatibility_executor(self):
        """Terminal translation uses the post-request-pipeline request."""
        translator = RecordingTranslator()
        chain = self._make_chain(
            request_processors=[ModelOverrideProcessor("gpt-4o-mini")],
            translator=translator,
        )

        await chain.call(make_request(model="gpt-4o"))

        assert translator.request_model == "gpt-4o-mini"

    async def test_python_metadata_survives_across_mixed_rust_and_python_components(self):
        """Python metadata is carried through Rust context across mixed components."""
        chain = self._make_chain(
            request_processors=[
                MetadataTagProcessor("before_native"),
                IntakeRequestProcessor(),
                MetadataTagProcessor("after_native"),
            ],
            response_processors=[
                MetadataAssertResponseProcessor(
                    {
                        "before_native": True,
                        "after_native": True,
                    }
                )
            ],
        )

        result = await chain.call(make_request())

        assert result["choices"][0]["message"]["content"] == "base"

    async def test_backend_adapter_restores_context_after_error(self):
        """Context mutations made before Python backend failure are not lost."""
        chain = self._make_chain(backend=FailingBackend())
        ctx = ProxyContext()

        with pytest.raises(RuntimeError, match="backend exploded"):
            await chain.call(make_request(model="gpt-4o"), ctx=ctx)

        assert ctx.metadata["backend_started"] == "gpt-4o"

    async def test_request_processor_adapter_restores_context_after_error(self):
        chain = self._make_chain(request_processors=[FailingRequestProcessor()])
        ctx = ProxyContext()

        with pytest.raises(RuntimeError, match="request processor exploded"):
            await chain.call(make_request(model="gpt-4o"), ctx=ctx)

        assert ctx.metadata["request_started"] == "gpt-4o"

    async def test_response_processor_invalid_return_restores_context(self):
        chain = self._make_chain(response_processors=[InvalidResponseProcessor()])
        ctx = ProxyContext()

        with pytest.raises(RuntimeError, match="ChatResponse"):
            await chain.call(make_request(model="gpt-4o"), ctx=ctx)

        assert ctx.metadata["response_started"] is True

    async def test_concurrent_calls_do_not_share_context_metadata(self):
        chain = self._make_chain(
            request_processors=[ModelMetadataProcessor()],
            response_processors=[ModelMetadataAssertProcessor()],
        )

        results = await asyncio.gather(
            *(chain.call(make_request(model=f"model-{index}")) for index in range(20))
        )

        assert {result["model"] for result in results} == {
            f"model-{index}" for index in range(20)
        }

    async def test_streaming_response_survives_compatibility_executor_boundary(self):
        chain = self._make_chain(backend=StreamingBackend())

        stream = await chain.call(make_request())
        events = [event async for event in stream]

        assert events[0]["choices"][0]["delta"]["content"] == "hello"

    async def test_streaming_callbacks_can_read_context_after_executor_returns(self):
        processor = ContextReadingStreamTapProcessor()
        chain = self._make_chain(
            backend=StreamingBackend(),
            response_processors=[processor],
        )

        stream = await chain.call(make_request())
        _ = [event async for event in stream]

        assert processor.seen_models == ["stream-selected"]

    async def test_context_overflow_evicts_target_and_retries_fallback(self):
        backend = OverflowWeakBackend()
        chain = self._make_chain(
            request_processors=[PickWeakProcessor()],
            backend=backend,
            fallback_target_on_evict="strong",
        )
        ctx = ProxyContext()

        result = await chain.call(make_request(model="client-model"), ctx=ctx)

        assert result["choices"][0]["message"]["content"] == "fallback"
        assert backend.calls == ["weak", "strong"]
        assert ctx.selected_target == "strong"
        assert ctx.evicted_targets == ["weak"]

    async def test_context_overflow_exception_target_retries_fallback_when_context_unset(self):
        backend = ExceptionOnlyOverflowBackend()
        chain = self._make_chain(
            backend=backend,
            fallback_target_on_evict="strong",
        )
        ctx = ProxyContext()

        result = await chain.call(make_request(model="client-model"), ctx=ctx)

        assert result["choices"][0]["message"]["content"] == "fallback"
        assert backend.calls == [None, "strong"]
        assert ctx.selected_target == "strong"
        assert ctx.evicted_targets == ["weak"]

    async def test_empty_processors_ok(self):
        chain = self._make_chain(
            request_processors=[],
            response_processors=[],
        )
        result = await chain.call(make_request())
        assert result["choices"][0]["message"]["content"] == "base"
