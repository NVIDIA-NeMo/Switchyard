# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-backed no-op runtime for proxy overhead benchmarking."""

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.profiles.chain import _context_from_input
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardProcessorError,
)
from switchyard_rust.profiles import ProfileInput
from switchyard_rust.translation import TranslationEngine

_NOOP_SUPPORTED_TYPES = [ChatRequestType.OPENAI_CHAT]
_NOOP_CONTENT = "pong"
_NOOP_MODEL = "noop"


def _make_completion(*, request_id: str, created: int) -> ChatCompletion:
    """Build the fixed no-op non-streaming completion."""
    return ChatCompletion(
        id=f"chatcmpl-{request_id}",
        object="chat.completion",
        created=created,
        model=_NOOP_MODEL,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=_NOOP_CONTENT),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


async def _noop_stream(
    *,
    request_id: str,
    created: int,
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield the fixed no-op streaming completion chunks."""
    yield ChatCompletionChunk(
        id=f"chatcmpl-{request_id}",
        object="chat.completion.chunk",
        created=created,
        model=_NOOP_MODEL,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=_NOOP_CONTENT),
                finish_reason=None,
            )
        ],
    )
    yield ChatCompletionChunk(
        id=f"chatcmpl-{request_id}",
        object="chat.completion.chunk",
        created=created,
        model=_NOOP_MODEL,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
        ],
    )


@profile_config("noop")
class NoopProfileConfig:
    """Dataclass config for the no-op profile."""

    def build(self) -> "NoopProfile":
        """Build a no-op runtime profile."""
        return NoopProfile()


@dataclass(frozen=True, slots=True)
class NoopProcessedRequest:
    """Request-side state carried from no-op ``process`` into ``rprocess``."""

    input: ProfileInput
    ctx: ProxyContext
    request: ChatRequest


class NoopProfile:
    """Profile that immediately returns a fixed minimal response.

    No network call is made. The response is always ``"pong"`` for both
    streaming and non-streaming requests. Non-OpenAI inbound requests are
    translated to OpenAI Chat before the profile reads the streaming flag.
    """

    def __init__(
        self,
        *,
        request_processors: tuple[Any, ...] = (),
        response_processors: tuple[Any, ...] = (),
        stats_accumulator: StatsAccumulator | None = None,
    ) -> None:
        """Create a no-op profile with a local translation helper."""
        self._translation = TranslationEngine()
        self._request_processors = request_processors
        self._response_processors = response_processors
        self._stats_accumulator = stats_accumulator

    def iter_components(self) -> list[object]:
        """Return lifecycle components in startup order."""
        return [
            *self._request_processors,
            self,
            *self._response_processors,
        ]

    def with_runtime_components(
        self,
        stats_accumulator: StatsAccumulator | None = None,
        enable_stats: bool = True,
        pre_request_processors: Sequence[Any] = (),
        post_request_processors: Sequence[Any] = (),
        response_processors: Sequence[Any] = (),
    ) -> "NoopProfile":
        """Return a no-op profile with route-table processors attached."""
        from switchyard.lib.processors.stats_request_processor import (
            StatsRequestProcessor,
        )
        from switchyard.lib.processors.stats_response_processor_accumulator import (
            StatsResponseProcessor,
        )

        request_chain: list[Any] = []
        response_chain: list[Any] = list(self._response_processors)
        if enable_stats:
            stats = stats_accumulator or StatsAccumulator()
            request_chain.append(StatsRequestProcessor())
            response_chain.append(StatsResponseProcessor(stats))

        request_chain.extend(pre_request_processors)
        request_chain.extend(self._request_processors)
        request_chain.extend(post_request_processors)
        response_chain.extend(response_processors)
        return NoopProfile(
            request_processors=tuple(request_chain),
            response_processors=tuple(response_chain),
            stats_accumulator=stats if enable_stats else None,
        )

    async def process(self, input: ProfileInput) -> NoopProcessedRequest:
        """Run request-side processors with a context derived from metadata."""
        return await self.process_with_context(input, _context_from_input(input))

    async def process_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> NoopProcessedRequest:
        """Run request-side processors against the caller-supplied context."""
        current: Any = input.request
        for processor in self._request_processors:
            try:
                current = await processor.process(ctx, current)
            except Exception as error:
                raise SwitchyardProcessorError(str(error)) from error
            if not isinstance(current, ChatRequest):
                actual = type(current).__name__
                raise SwitchyardProcessorError(
                    f"Request processor returned {actual}, expected ChatRequest"
                )
        return NoopProcessedRequest(input=input, ctx=ctx, request=current)

    async def rprocess(
        self,
        processed: NoopProcessedRequest,
        response: ChatResponse,
    ) -> ChatResponse:
        """Run response-side processors for the fixed no-op response."""
        current: Any = response
        for processor in self._response_processors:
            try:
                current = await processor.process(processed.ctx, current)
            except Exception as error:
                raise SwitchyardProcessorError(str(error)) from error
            if not isinstance(current, ChatResponse):
                actual = type(current).__name__
                raise SwitchyardProcessorError(
                    f"Response processor returned {actual}, expected ChatResponse"
                )
        return cast(ChatResponse, current)

    async def run(self, input: ProfileInput) -> ChatResponse:
        """Execute no-op response generation with a derived context."""
        return await self.run_with_context(input, _context_from_input(input))

    async def run_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ChatResponse:
        """Execute no-op response generation with an existing context."""
        processed = await self.process_with_context(input, ctx)
        # Record the call under the "noop" model. Stamping ctx.selected_model
        # lets the stats response processor attribute the response tokens to
        # "noop" instead of "<unknown>"; record_success counts the request so
        # the noop route shows up in /v1/routing/stats.
        ctx.selected_model = _NOOP_MODEL
        if self._stats_accumulator is not None:
            await self._stats_accumulator.record_success(_NOOP_MODEL, None, None)
        openai_request = self._translation.request_to_any_of(
            processed.request,
            _NOOP_SUPPORTED_TYPES,
        )
        body: dict[str, object] = dict(openai_request.body)
        is_streaming = bool(body.get("stream", False))

        request_id = uuid.uuid4().hex[:16]
        created = int(time.time())

        if is_streaming:
            response = ChatResponse.openai_stream(
                ResponseStream(_noop_stream(request_id=request_id, created=created))
            )
        else:
            response = ChatResponse.openai_completion(
                _make_completion(request_id=request_id, created=created)
            )
        return await self.rprocess(processed, response)


__all__ = ["NoopProcessedRequest", "NoopProfile", "NoopProfileConfig"]
