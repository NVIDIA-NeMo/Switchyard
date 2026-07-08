# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-side processor that records per-model token usage into a LiveStatsCollector.

This is the single stats-recording mechanism for the chain when using
LiveStatsCollector (vs StatsAccumulator). It is format-agnostic (handles
Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses API)
and routing-policy-agnostic (works with any LLMBackend).

Model attribution uses Rust-owned ``ProxyContext.selected_model`` first,
falling back to the legacy ``ctx.metadata["_proxy_actual_model"]`` key.
Tier labelling uses ``ctx.metadata["_random_routing_tier"]`` when present,
or Rust-owned ``ProxyContext.selected_target`` for strong/weak routing.

The processor exposes ``get_routing_stats()`` / ``reset_routing_stats()``
and ``get_endpoint()`` so the chain's app factory can mount a live stats
endpoint at ``GET /v1/routing/stats`` with no extra wiring.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from anthropic.types import RawMessageStreamEvent

from switchyard.lib.live_stats_collector import LiveStatsCollector
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL, ProxyContext
from switchyard_rust.core import (
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)
from switchyard_rust.translation import sse_frame_payloads

if TYPE_CHECKING:
    from switchyard.lib.endpoints.base import Endpoint


class StatsResponseProcessor:
    """Records token usage after each LLM call into a :class:`LiveStatsCollector`.

    Non-streaming responses: reads usage directly from the response body.

    Streaming responses: attaches a lightweight tap to the stream so
    usage-carrying events update the collector as they flow through —
    no buffering, zero impact on streaming latency.

    Supported response shapes are identified by Rust-owned ``ChatResponseType``.
    """

    def __init__(
        self,
        collector: LiveStatsCollector,
        *,
        expose_endpoint: bool = True,
    ) -> None:
        self._collector = collector
        self._expose_endpoint = expose_endpoint

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        model: str = ctx.selected_model or ctx.metadata.get(CTX_PROXY_ACTUAL_MODEL, "unknown")
        tier = _stats_tier(ctx)

        if response_type_matches(response, ChatResponseType.ANTHROPIC_COMPLETION):
            _record_anthropic(response, model, tier, self._collector)
        elif response_type_matches(response, ChatResponseType.ANTHROPIC_STREAM):
            _attach_anthropic_tap(response, model, tier, self._collector)
        elif response_type_matches(response, ChatResponseType.OPENAI_COMPLETION):
            _record_openai_chat(response, model, tier, self._collector)
        elif response_type_matches(response, ChatResponseType.OPENAI_STREAM):
            _attach_openai_chat_tap(response, model, tier, self._collector)
        elif response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_COMPLETION):
            _record_openai_responses(response, model, tier, self._collector)
        elif response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_STREAM):
            _attach_openai_responses_tap(response, model, tier, self._collector)
        return response

    # ------------------------------------------------------------------
    # HTTP stats endpoint integration
    # ------------------------------------------------------------------

    def get_routing_stats(self) -> dict[str, object]:
        """Return a snapshot of accumulated per-model statistics."""
        return self._collector.to_dict()

    def reset_routing_stats(self) -> None:
        """Reset all accumulated statistics to zero."""
        self._collector.reset()

    def get_endpoint(self) -> Endpoint | None:
        """Contribute ``GET /v1/routing/stats`` to the server."""
        if not self._expose_endpoint:
            return None
        from switchyard.lib.endpoints.stats_endpoint import (
            StatsEndpoint,
        )

        return StatsEndpoint(self._collector)


def _stats_tier(ctx: ProxyContext) -> str:
    tier: str = ctx.metadata.get("_random_routing_tier", "")
    if tier:
        return tier
    selected_target = ctx.selected_target
    return selected_target if selected_target in {"strong", "weak"} else ""


# ---------------------------------------------------------------------------
# Non-streaming extractors
# ---------------------------------------------------------------------------


def _record_anthropic(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    u = _field(response.body, "usage")
    if not u:
        return
    input_tok = _int_field(u, "input_tokens")
    cache_create = _int_field(u, "cache_creation_input_tokens")
    cache_read = _int_field(u, "cache_read_input_tokens")
    completion = _int_field(u, "output_tokens")
    collector.record(
        model, tier,
        prompt_tokens=input_tok + cache_create + cache_read,
        completion_tokens=completion,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_create,
    )


def _record_openai_chat(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    u = _field(response.body, "usage")
    if u is None:
        return
    prompt = _int_field(u, "prompt_tokens")
    completion = _int_field(u, "completion_tokens")
    reasoning = 0
    cached = 0
    if (details := _field(u, "completion_tokens_details")) is not None:
        reasoning = _int_field(details, "reasoning_tokens")
    if (prompt_details := _field(u, "prompt_tokens_details")) is not None:
        cached = _int_field(prompt_details, "cached_tokens")
    collector.record(
        model, tier,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cache_read_tokens=cached,
    )


def _record_openai_responses(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    u = _field(response.body, "usage")
    if u is None:
        return
    prompt = _int_field(u, "input_tokens")
    completion = _int_field(u, "output_tokens")
    reasoning = 0
    cached = 0
    if (out_details := _field(u, "output_tokens_details")) is not None:
        reasoning = _int_field(out_details, "reasoning_tokens")
    if (in_details := _field(u, "input_tokens_details")) is not None:
        cached = _int_field(in_details, "cached_tokens")
    collector.record(
        model, tier,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cache_read_tokens=cached,
    )


# ---------------------------------------------------------------------------
# Streaming tap installers
# ---------------------------------------------------------------------------


def _attach_anthropic_tap(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    """Register a tap that accumulates Anthropic SSE usage."""
    acc: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    committed = False

    def _merge(usage: object) -> None:
        for key in acc:
            value = _field(usage, key)
            if isinstance(value, int):
                acc[key] = value

    async def _tap(event: RawMessageStreamEvent) -> None:
        nonlocal committed
        if committed:
            return
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                usage = _field(msg, "usage")
                if usage is not None:
                    _merge(usage)
        elif event_type == "message_delta":
            usage = _field(event, "usage")
            if usage is not None:
                _merge(usage)
        elif event_type == "message_stop":
            input_tok = acc["input_tokens"]
            cache_create = acc["cache_creation_input_tokens"]
            cache_read = acc["cache_read_input_tokens"]
            completion = acc["output_tokens"]
            collector.record(
                model, tier,
                prompt_tokens=input_tok + cache_create + cache_read,
                completion_tokens=completion,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_create,
            )
            committed = True

    response.stream.tap(_tap)


def _attach_openai_chat_tap(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    """Register a tap for OpenAI Chat streaming usage."""
    seen = False

    async def _tap(chunk) -> None:  # type: ignore[no-untyped-def]
        nonlocal seen
        if seen:
            return
        u = _field(chunk, "usage")
        if u is None:
            return
        prompt = _int_field(u, "prompt_tokens")
        completion = _int_field(u, "completion_tokens")
        reasoning = 0
        cached = 0
        if (details := _field(u, "completion_tokens_details")) is not None:
            reasoning = _int_field(details, "reasoning_tokens")
        if (prompt_details := _field(u, "prompt_tokens_details")) is not None:
            cached = _int_field(prompt_details, "cached_tokens")
        collector.record(
            model, tier,
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            cache_read_tokens=cached,
        )
        seen = True

    response.stream.tap(_tap)


def _attach_openai_responses_tap(
    response: ChatResponse,
    model: str,
    tier: str,
    collector: LiveStatsCollector,
) -> None:
    """Register a tap for OpenAI Responses API streaming usage."""
    seen = False

    async def _tap(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal seen
        if seen:
            return
        # Verbatim-passthrough backends yield raw SSE frame strings; parse the
        # frame's data payload(s) to find the usage-carrying event.
        candidates = sse_frame_payloads(event) if isinstance(event, str) else [event]
        inner = next(
            (found for c in candidates if (found := _field(c, "response")) is not None),
            None,
        )
        if inner is None:
            return
        u = _field(inner, "usage")
        if u is None:
            return
        prompt = _int_field(u, "input_tokens")
        completion = _int_field(u, "output_tokens")
        reasoning = 0
        cached = 0
        if (out_details := _field(u, "output_tokens_details")) is not None:
            reasoning = _int_field(out_details, "reasoning_tokens")
        if (in_details := _field(u, "input_tokens_details")) is not None:
            cached = _int_field(in_details, "cached_tokens")
        collector.record(
            model, tier,
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            cache_read_tokens=cached,
        )
        seen = True

    response.stream.tap(_tap)


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int_field(value: object, name: str) -> int:
    field = _field(value, name)
    return field if isinstance(field, int) else 0
