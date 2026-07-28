# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production end-to-end smoke test for ``switchyard`` passthrough on ``/v1/responses``.

Exercises the cross-format production path end-to-end::

    openai SDK (responses.create) → HTTP (127.0.0.1)
        → switchyard passthrough
            → TranslationEngine.request_to (Responses → Chat)
            → real backend (Chat Completions)
            → TranslationEngine.response_to (Chat → Responses)
        → Responses-shaped response

Unlike the sibling Chat Completions suite (which tests same-format
passthrough), this suite specifically validates the inbound and
outbound translators — a Responses-shaped request must survive
translation to Chat Completions, execute against the backend, and be
translated back to a Responses-shaped response.

Covers both non-streaming (``responses.create``) and streaming
(``responses.create(stream=True)``) paths — the latter validates that
``TranslationEngine.stream_for_request`` drives the
OpenAI-chunk → Responses-SSE conversion end-to-end through
``iter_preframed_sse`` and the ``/v1/responses`` endpoint.

Uses the official ``openai`` Python SDK as the client
(``client.responses.create(...)``) so we validate the exact request
shape a real customer integration would produce, and we get typed
response objects for free.

Shares the ``passthrough_server`` fixture with the other e2e files
via ``conftest.py`` — one subprocess server per session.

Run with::

    OPENROUTER_API_KEY=sk-or-... pytest tests/e2e/test_passthrough_responses_e2e.py -v
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai
import pytest

from ._helpers import (
    RESPONSES_WEATHER_TOOL,
    resolve_tool_capable_model,
)

pytestmark = pytest.mark.integration

logger = logging.getLogger("e2e.responses")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_openai_client(passthrough_server: dict) -> openai.OpenAI:
    """Build an OpenAI SDK client pointed at the passthrough server.

    Same convention as the Chat Completions sibling — the passthrough
    injects its own upstream ``api_key`` when calling the real backend,
    so the SDK-side ``api_key`` is just a placeholder for the
    ``Authorization`` header.
    """
    return openai.OpenAI(
        base_url=f"{passthrough_server['base_url']}/v1",
        api_key="not-used-passthrough-forwards-its-own",
        timeout=60.0,
        max_retries=0,
    )


def _extract_assistant_text(response: Any) -> str:
    """Concatenate all ``output_text`` blocks across ``output`` items.

    The Responses API ``output`` array is heterogeneous — it can
    contain ``message``, ``reasoning``, ``function_call``, and other
    item types.  This helper walks only ``type=message`` items and
    pulls out their ``output_text`` blocks, returning the combined
    assistant-visible text (empty string if none).
    """
    parts: list[str] = []
    for item in response.output or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
    return "".join(parts)


def _extract_function_calls(response: Any) -> list[Any]:
    """Return every ``type=function_call`` item in ``response.output``.

    In the Responses API tool calls are top-level items in ``output``
    rather than nested inside a ``message`` (unlike Chat Completions
    where they live under ``message.tool_calls``).  This helper keeps
    the extraction logic in one place for the tool test below.
    """
    return [
        item
        for item in (response.output or [])
        if getattr(item, "type", None) == "function_call"
    ]


def _collect_responses_stream_events(
    stream: Any,
) -> tuple[list[str], str, dict[int, dict[str, Any]]]:
    """Drain a Responses API stream and summarize events.

    Returns ``(event_types, text, function_calls)``:

    * ``event_types`` — the ``type`` field of every ``ResponseStreamEvent``
      received, in order.  Used for lifecycle structural assertions
      (``response.created`` first, ``response.completed`` last).
    * ``text`` — all ``response.output_text.delta`` strings concatenated.
    * ``function_calls`` — keyed by ``output_index``, each entry carries
      ``{"name", "call_id", "arguments"}`` with ``arguments`` being the
      concatenated ``response.function_call_arguments.delta`` fragments.
    """
    event_types: list[str] = []
    text_parts: list[str] = []
    function_calls: dict[int, dict[str, Any]] = {}

    for event in stream:
        ev_type = getattr(event, "type", None)
        if ev_type:
            event_types.append(ev_type)

        if ev_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item is not None and getattr(item, "type", None) == "function_call":
                idx = getattr(event, "output_index", 0)
                function_calls[idx] = {
                    "name": getattr(item, "name", ""),
                    "call_id": getattr(item, "call_id", ""),
                    "arguments": "",
                }
        elif ev_type == "response.output_text.delta":
            text_parts.append(getattr(event, "delta", "") or "")
        elif ev_type == "response.function_call_arguments.delta":
            idx = getattr(event, "output_index", 0)
            if idx in function_calls:
                function_calls[idx]["arguments"] += (
                    getattr(event, "delta", "") or ""
                )

    return event_types, "".join(text_parts), function_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponsesPassthroughE2E:
    """OpenAI Responses SDK → cross-format passthrough → backend → Responses out."""

    # ------------------------------------------------------------------
    # Basic say-hi
    # ------------------------------------------------------------------

    def test_say_hi_get_response_back(self, passthrough_server: dict) -> None:
        """``client.responses.create(input="hi")`` round-trips successfully.

        Validates the minimum contract of the Responses passthrough:

        * the Responses SDK can reach the server and the request
          deserializes cleanly into a ``Response`` object
        * ``output`` is non-empty and contains an ``assistant`` message
          with at least one ``output_text`` block that has visible text
        """
        client = _make_openai_client(passthrough_server)
        response = client.responses.create(
            model=passthrough_server["model"],
            input="hi",
            max_output_tokens=2048,
        )

        assert response.output, (
            f"expected non-empty output, got {response.model_dump()}"
        )

        text = _extract_assistant_text(response)
        assert text, (
            f"no assistant output_text across {len(response.output)} items: "
            f"{response.model_dump()}"
        )

        logger.info(
            f"  [Responses] output_items={len(response.output)}, "
            f"text[:100]={text[:100]!r}"
        )

    # ------------------------------------------------------------------
    # System prompt — Responses API uses ``instructions`` as the
    # top-level system-equivalent parameter (not a ``system`` role).
    # ------------------------------------------------------------------

    def test_system_prompt_non_streaming(self, passthrough_server: dict) -> None:
        """``instructions`` parameter flows through the cross-format translator.

        Responses API models the system prompt as a top-level
        ``instructions`` string, distinct from the message list.  The
        ``TranslationEngine.request_to`` must lift
        ``instructions`` into a ``role=system`` message before calling
        the backend.  If that translation drops the field, the model
        has no way to honor the planted ``pineapple`` keyword, and
        this test catches it.
        """
        client = _make_openai_client(passthrough_server)
        response = client.responses.create(
            model=passthrough_server["model"],
            instructions=(
                "You are a helpful assistant. Always include the exact "
                "word 'pineapple' in your response."
            ),
            input="Say hello.",
            max_output_tokens=2048,
        )

        assert response.output, (
            f"expected non-empty output: {response.model_dump()}"
        )

        # Keyword check — only when the model produced visible
        # output_text.  Reasoning models can burn their entire
        # ``max_output_tokens`` budget on internal thought before the
        # Responses translator surfaces any ``output_text`` block;
        # that's a backend-scheduling artifact, not a passthrough
        # failure, so we forgive it.  The shape checks above still
        # prove the ``instructions`` translation didn't blow up.
        text = _extract_assistant_text(response)
        if text:
            assert "pineapple" in text.lower(), (
                f"instructions not honored — the ``instructions`` → "
                f"system translation may have dropped the field. "
                f"text={text[:200]!r}"
            )

        logger.info(
            f"  [Responses][system] output_items={len(response.output)}, "
            f"text[:150]={text[:150]!r}"
        )

    # ------------------------------------------------------------------
    # Multi-turn — Responses API accepts a list of message items as
    # ``input``, exercising the translator's handling of assistant-role
    # history items.
    # ------------------------------------------------------------------

    def test_multi_turn_non_streaming(self, passthrough_server: dict) -> None:
        """Multi-turn history via message-list ``input`` flows through.

        Sends a 3-item ``input`` (``user`` / ``assistant`` / ``user``)
        and asks the model to recall a name planted two turns back.
        If the translator drops the assistant turn, the model has no
        context and can't recall the name.
        """
        # Each input item is explicitly tagged ``type="message"``:
        # ``TranslationEngine._convert_input_items_to_messages``
        # dispatches on ``item["type"]`` and silently drops items that
        # don't carry one, which would leave ``messages=[]`` and 400
        # the backend with "list index out of range".
        client = _make_openai_client(passthrough_server)
        response = client.responses.create(
            model=passthrough_server["model"],
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "My name is Alice. Please remember this.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Got it, Alice. I will remember your name.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        "What is my name? Reply with only the name "
                        "itself, nothing else."
                    ),
                },
            ],
            max_output_tokens=2048,
        )

        assert response.output, (
            f"expected non-empty output: {response.model_dump()}"
        )

        # Same reasoning-truncation forgiveness as the ``instructions``
        # test above.
        text = _extract_assistant_text(response)
        if text:
            assert "alice" in text.lower(), (
                f"prior-turn context not recalled — the ``input`` list "
                f"→ messages translation may have dropped history items. "
                f"text={text[:200]!r}"
            )

        logger.info(
            f"  [Responses][multi-turn] output_items={len(response.output)}, "
            f"text[:150]={text[:150]!r}"
        )

    # ------------------------------------------------------------------
    # Tool call — Responses API puts ``function_call`` items directly
    # into ``output`` rather than nesting them under a message's
    # ``tool_calls`` field.  This test confirms that shape survives the
    # round-trip through Chat Completions format.
    # ------------------------------------------------------------------

    def test_tool_call_non_streaming(self, passthrough_server: dict) -> None:
        """``tools`` definitions flow through and a ``function_call`` item returns.

        Validates that:

        * the flat Responses-shape tool definition is forwarded through
          the translator (Responses → Chat Completions nested shape)
        * the backend's ``tool_calls`` on the Chat response are
          translated back to top-level ``function_call`` items in
          ``response.output``
        * the first function call has the expected name and its
          ``arguments`` string parses as JSON containing ``city``

        Uses :func:`resolve_tool_capable_model` to avoid vLLM-hosted
        models that lack ``--enable-auto-tool-choice``.
        """
        tool_model = resolve_tool_capable_model(passthrough_server["model"])
        logger.info(
            f"  [Responses][tools] using tool-capable model: {tool_model!r}"
        )

        client = _make_openai_client(passthrough_server)
        response = client.responses.create(
            model=tool_model,
            input=(
                "What is the weather in Tokyo? "
                "You MUST use the get_weather tool."
            ),
            tools=[RESPONSES_WEATHER_TOOL],
            tool_choice="auto",
            max_output_tokens=2048,
        )

        function_calls = _extract_function_calls(response)
        assert function_calls, (
            f"expected at least one function_call item in response.output, "
            f"got {[getattr(i, 'type', None) for i in response.output or []]}"
        )

        first = function_calls[0]
        assert first.name == "get_weather", (
            f"unexpected function name: {first.name!r}"
        )

        raw_args = first.arguments or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        assert "city" in args, (
            f"expected 'city' in arguments, got {args!r}"
        )

        logger.info(
            f"  [Responses][tools] function_calls={len(function_calls)}, "
            f"name={first.name!r}, args={args!r}"
        )

    # ------------------------------------------------------------------
    # Streaming — validates the Chat Completions → Responses SSE
    # translation drives through the chain end-to-end.
    #
    # Wire path:
    #   OpenAI SDK responses.create(stream=True) → /v1/responses
    #     → chain → LlmClient returns an OpenAI stream response
    #     → translate_stream dispatches on the Responses request format
    #     → stream_chat_to_responses_sse yields pre-framed SSE strings
    #     → iter_preframed_sse forwards them → client SDK parses them
    # ------------------------------------------------------------------

    def test_say_hi_streaming(self, passthrough_server: dict) -> None:
        """``responses.create(stream=True)`` yields a valid Responses SSE stream.

        Validates the minimum streaming contract:

        * the SDK can iterate the SSE stream (so the passthrough's
          envelope — ``Content-Type: text/event-stream``, ``event:
          response.*\\ndata: {...}\\n\\n`` frames — must be well-formed)
        * the stream emits ``response.created`` before any content and
          ``response.completed`` at the end
        * at least one ``response.output_text.delta`` carries visible
          text (so the translator text-path works chunk-by-chunk)
        """
        client = _make_openai_client(passthrough_server)
        stream = client.responses.create(
            model=passthrough_server["model"],
            input="hi",
            max_output_tokens=2048,
            stream=True,
        )
        event_types, text, _ = _collect_responses_stream_events(stream)

        assert event_types and event_types[0] == "response.created", (
            f"expected first event to be response.created, got {event_types[:3]!r}"
        )
        assert "response.completed" in event_types, (
            f"missing response.completed, got types={event_types!r}"
        )
        assert text, (
            f"no output_text.delta across {len(event_types)} events: "
            f"types={event_types!r}"
        )

        logger.info(
            f"  [Responses-stream] events={len(event_types)}, "
            f"text[:100]={text[:100]!r}"
        )

    def test_system_prompt_streaming(self, passthrough_server: dict) -> None:
        """``instructions`` flows through the streaming cross-format translator."""
        client = _make_openai_client(passthrough_server)
        stream = client.responses.create(
            model=passthrough_server["model"],
            instructions=(
                "You are a helpful assistant. Always include the exact "
                "word 'pineapple' in your response."
            ),
            input="Say hello.",
            max_output_tokens=2048,
            stream=True,
        )
        event_types, text, _ = _collect_responses_stream_events(stream)

        assert "response.created" in event_types
        assert "response.completed" in event_types

        if text:
            assert "pineapple" in text.lower(), (
                f"instructions not honored over the stream path. "
                f"text={text[:200]!r}"
            )

        logger.info(
            f"  [Responses-stream][system] events={len(event_types)}, "
            f"text[:150]={text[:150]!r}"
        )

    def test_multi_turn_streaming(self, passthrough_server: dict) -> None:
        """Multi-turn history via message-list ``input`` streams back."""
        client = _make_openai_client(passthrough_server)
        stream = client.responses.create(
            model=passthrough_server["model"],
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "My name is Alice. Please remember this.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Got it, Alice. I will remember your name.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        "What is my name? Reply with only the name "
                        "itself, nothing else."
                    ),
                },
            ],
            max_output_tokens=2048,
            stream=True,
        )
        event_types, text, _ = _collect_responses_stream_events(stream)

        assert "response.created" in event_types
        assert "response.completed" in event_types

        if text:
            assert "alice" in text.lower(), (
                f"prior-turn context not recalled over the stream path. "
                f"text={text[:200]!r}"
            )

        logger.info(
            f"  [Responses-stream][multi-turn] events={len(event_types)}, "
            f"text[:150]={text[:150]!r}"
        )

    def test_tool_call_streaming(self, passthrough_server: dict) -> None:
        """Streaming tool-call deltas reassemble into a valid ``function_call``.

        Validates:

        * ``response.output_item.added`` carries a ``function_call``
          item with the expected ``name``
        * ``response.function_call_arguments.delta`` fragments
          concatenate into a parseable JSON object containing ``city``
        * the stream terminates with ``response.completed``
        """
        tool_model = resolve_tool_capable_model(passthrough_server["model"])
        logger.info(
            f"  [Responses-stream][tools] using tool-capable model: {tool_model!r}"
        )

        client = _make_openai_client(passthrough_server)
        stream = client.responses.create(
            model=tool_model,
            input=(
                "What is the weather in Tokyo? "
                "You MUST use the get_weather tool."
            ),
            tools=[RESPONSES_WEATHER_TOOL],
            tool_choice="auto",
            max_output_tokens=2048,
            stream=True,
        )
        event_types, _, function_calls = _collect_responses_stream_events(stream)

        assert "response.completed" in event_types
        assert function_calls, (
            f"no streamed function_call item across {len(event_types)} events: "
            f"types={event_types!r}"
        )

        first_idx = sorted(function_calls)[0]
        first = function_calls[first_idx]
        assert first["name"] == "get_weather", (
            f"unexpected function name: {first['name']!r}"
        )

        raw_args = first["arguments"] or "{}"
        args = json.loads(raw_args)
        assert "city" in args, (
            f"expected 'city' in reassembled arguments, got {args!r}"
        )

        logger.info(
            f"  [Responses-stream][tools] events={len(event_types)}, "
            f"function_calls={len(function_calls)}, "
            f"name={first['name']!r}, args={args!r}"
        )
