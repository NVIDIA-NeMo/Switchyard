# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record token, cost, and TTFT metrics from a backend response.

Called from the chain executor after the backend returns. Non-streaming responses
are recorded immediately; streaming responses install a stream tap that records
final token usage at end-of-stream and time-to-first-token on the first chunk.

Token-usage extraction mirrors the historical stats response processor across the
three wire formats (OpenAI Chat, OpenAI Responses, Anthropic). Cost is derived from
the retained pricing table in :mod:`switchyard.lib.cost_estimator`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from switchyard.lib import metrics, spans
from switchyard.lib.cost_estimator import estimate_model_cost
from switchyard_rust.core import ChatResponse, ChatResponseType, response_type_matches

#: ``ctx.metadata`` key holding the monotonic request-start timestamp, stamped by
#: the chain executor so TTFT can be computed on the first streamed chunk.
CTX_REQUEST_START = "_otel_request_start"


def record_response_usage(
    ctx: Any,
    response: ChatResponse,
    model: str,
    tier: str | None,
    role: str = "routed",
) -> None:
    """Record token/cost (and TTFT for streams) for one backend response."""
    start = ctx.metadata.get(CTX_REQUEST_START) if hasattr(ctx, "metadata") else None

    if response_type_matches(response, ChatResponseType.ANTHROPIC_COMPLETION):
        _emit_anthropic(response.body, model, tier, role)
    elif response_type_matches(response, ChatResponseType.ANTHROPIC_STREAM):
        _attach_tap(response, model, tier, role, start, _make_anthropic_extractor())
    elif response_type_matches(response, ChatResponseType.OPENAI_COMPLETION):
        _emit_openai_chat(response.body, model, tier, role)
    elif response_type_matches(response, ChatResponseType.OPENAI_STREAM):
        _attach_tap(response, model, tier, role, start, _openai_chat_stream_usage)
    elif response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_COMPLETION):
        _emit_openai_responses(response.body, model, tier, role)
    elif response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_STREAM):
        _attach_tap(response, model, tier, role, start, _openai_responses_stream_usage)


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------


def _emit(
    model: str,
    tier: str | None,
    role: str,
    *,
    prompt: int,
    completion: int,
    cached: int = 0,
    cache_creation: int = 0,
    reasoning: int = 0,
) -> None:
    metrics.record_tokens(
        model=model,
        tier=tier,
        prompt=prompt,
        completion=completion,
        cached=cached,
        cache_creation=cache_creation,
        reasoning=reasoning,
    )
    costs = estimate_model_cost(model, prompt, completion, cached, cache_creation)
    for kind, key in (
        ("input", "base_input_cost"),
        ("cached", "cached_input_cost"),
        ("cache_write", "cache_write_cost"),
        ("output", "output_cost"),
    ):
        value = costs.get(key, 0.0)
        if value:
            metrics.record_cost(model=model, tier=tier, role=role, kind=kind, cost_usd=value)


def _emit_anthropic(body: object, model: str, tier: str | None, role: str) -> None:
    u = _field(body, "usage")
    if not u:
        return
    input_tok = _int(u, "input_tokens")
    cache_create = _int(u, "cache_creation_input_tokens")
    cache_read = _int(u, "cache_read_input_tokens")
    _emit(
        model, tier, role,
        prompt=input_tok + cache_create + cache_read,
        completion=_int(u, "output_tokens"),
        cached=cache_read,
        cache_creation=cache_create,
    )


def _emit_openai_chat(body: object, model: str, tier: str | None, role: str) -> None:
    u = _field(body, "usage")
    if u is None:
        return
    reasoning = cached = 0
    if (details := _field(u, "completion_tokens_details")) is not None:
        reasoning = _int(details, "reasoning_tokens")
    if (prompt_details := _field(u, "prompt_tokens_details")) is not None:
        cached = _int(prompt_details, "cached_tokens")
    _emit(
        model, tier, role,
        prompt=_int(u, "prompt_tokens"),
        completion=_int(u, "completion_tokens"),
        cached=cached,
        reasoning=reasoning,
    )


def _emit_openai_responses(body: object, model: str, tier: str | None, role: str) -> None:
    u = _field(body, "usage")
    if u is None:
        return
    reasoning = cached = 0
    if (out_details := _field(u, "output_tokens_details")) is not None:
        reasoning = _int(out_details, "reasoning_tokens")
    if (in_details := _field(u, "input_tokens_details")) is not None:
        cached = _int(in_details, "cached_tokens")
    _emit(
        model, tier, role,
        prompt=_int(u, "input_tokens"),
        completion=_int(u, "output_tokens"),
        cached=cached,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Streaming taps — usage at end-of-stream, TTFT on first chunk
# ---------------------------------------------------------------------------


def _attach_tap(
    response: ChatResponse,
    model: str,
    tier: str | None,
    role: str,
    start: float | None,
    extract: Any,
) -> None:
    """Install a tap that records TTFT on the first event and usage when present."""
    state = {"ttft_done": False, "usage_done": False}

    async def _tap(event: object) -> None:
        if not state["ttft_done"]:
            state["ttft_done"] = True
            if start is not None:
                ttft_ms = (time.monotonic() - start) * 1000.0
                metrics.record_ttft(model=model, tier=tier, ttft_ms=ttft_ms)
                spans.add_span_event("first_token", {"model": model})
        if state["usage_done"]:
            return
        commit = extract(event)
        if commit is not None:
            state["usage_done"] = True
            commit(model, tier, role)

    response.stream.tap(_tap)


# Anthropic needs cross-event accumulation; provide a stateful extractor factory.
# OpenAI Chat / Responses carry final usage on a single event, so their extractors
# are stateless module functions.
def _make_anthropic_extractor() -> Any:
    acc = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    def _extract(event: object) -> Any:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                _merge(acc, _field(msg, "usage"))
        elif event_type == "message_delta":
            _merge(acc, _field(event, "usage"))
        elif event_type == "message_stop":
            input_tok = acc["input_tokens"]
            cache_create = acc["cache_creation_input_tokens"]
            cache_read = acc["cache_read_input_tokens"]
            completion = acc["output_tokens"]

            def _commit(model: str, tier: str | None, role: str) -> None:
                _emit(
                    model, tier, role,
                    prompt=input_tok + cache_create + cache_read,
                    completion=completion,
                    cached=cache_read,
                    cache_creation=cache_create,
                )

            return _commit
        return None

    return _extract


def _openai_chat_stream_usage(event: object) -> Any:
    u = _field(event, "usage")
    if u is None:
        return None
    reasoning = cached = 0
    if (details := _field(u, "completion_tokens_details")) is not None:
        reasoning = _int(details, "reasoning_tokens")
    if (prompt_details := _field(u, "prompt_tokens_details")) is not None:
        cached = _int(prompt_details, "cached_tokens")
    prompt = _int(u, "prompt_tokens")
    completion = _int(u, "completion_tokens")

    def _commit(model: str, tier: str | None, role: str) -> None:
        _emit(model, tier, role, prompt=prompt, completion=completion, cached=cached,
              reasoning=reasoning)

    return _commit


def _openai_responses_stream_usage(event: object) -> Any:
    inner = _field(event, "response")
    if inner is None:
        return None
    u = _field(inner, "usage")
    if u is None:
        return None
    reasoning = cached = 0
    if (out_details := _field(u, "output_tokens_details")) is not None:
        reasoning = _int(out_details, "reasoning_tokens")
    if (in_details := _field(u, "input_tokens_details")) is not None:
        cached = _int(in_details, "cached_tokens")
    prompt = _int(u, "input_tokens")
    completion = _int(u, "output_tokens")

    def _commit(model: str, tier: str | None, role: str) -> None:
        _emit(model, tier, role, prompt=prompt, completion=completion, cached=cached,
              reasoning=reasoning)

    return _commit


def _merge(acc: dict[str, int], usage: object) -> None:
    if usage is None:
        return
    for key in acc:
        value = _field(usage, key)
        if isinstance(value, int):
            acc[key] = value


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int(value: object, name: str) -> int:
    field = _field(value, name)
    return field if isinstance(field, int) else 0
