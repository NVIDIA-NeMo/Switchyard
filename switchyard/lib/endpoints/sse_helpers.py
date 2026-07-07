# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SSE serialization helpers for HTTP endpoints.

Pure async generators that turn typed response streams into the raw
SSE frames FastAPI's ``StreamingResponse`` expects.  Each wire format
(OpenAI Chat Completions, Anthropic Messages, Responses API) has its
own helper with the format-specific framing contract.

Error contract (shared across all helpers):
(:mod:`switchyard.server.endpoints.openai_endpoint`):

- The endpoint awaits ``switchyard.call()`` *before* handing the stream
  to ``StreamingResponse``, so upstream auth / connection / rate-limit
  failures raise exceptions that the global handler turns into proper
  HTTP error responses.
- Only failures during chunk iteration land in ``except`` here.  At
  that point HTTP 200 has already been committed, so the best we can
  do is emit a final SSE error frame (format-specific shape).
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping

from anthropic.types import RawMessageStreamEvent
from openai.types.chat import ChatCompletionChunk

log = logging.getLogger(__name__)


async def _aclose_stream(stream: object) -> None:
    """Best-effort close of an upstream response stream.

    On every exit path — normal completion, mid-stream error, or the client
    disconnecting (which makes the ASGI server ``aclose()`` the SSE generator,
    raising ``GeneratorExit`` at the suspended ``yield``) — the upstream stream
    must be closed so its underlying httpx connection is returned to the pool
    instead of leaking. ``ChatResponseStream`` and async generators expose
    ``aclose``; SDK ``AsyncStream`` objects expose ``close``; either may be a
    coroutine. Closing must never mask the original control flow, so failures
    are logged and swallowed.
    """
    closer = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        log.debug("Failed to close upstream stream: %s: %s", type(exc).__name__, exc)


async def iter_chat_completion_sse(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncGenerator[str, None]:
    """Serialize a Chat Completions chunk stream to OpenAI-style SSE frames.

    Accepts any async iterator of objects with ``model_dump()`` (OpenAI
    SDK ``ChatCompletionChunk``) or ``to_dict()``; falls back to
    ``dict(chunk)`` for dict-likes.  This is deliberately duck-typed —
    the same helper serves the backend's raw ``ResponseStream`` and
    any future transformed stream that still yields chunk-like objects.

    Emits ``data: [DONE]\\n\\n`` after successful completion, matching
    the OpenAI streaming contract.

    Args:
        stream: Async iterator of ``ChatCompletionChunk`` (or compatible).

    Yields:
        SSE-framed strings suitable for ``StreamingResponse``.
    """
    try:
        async for chunk in stream:
            if hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump(exclude_none=True)
            elif hasattr(chunk, "to_dict"):
                chunk_dict = chunk.to_dict()
            else:
                chunk_dict = (
                    dict(chunk) if hasattr(chunk, "__iter__") else {"data": str(chunk)}
                )
            yield f"data: {json.dumps(chunk_dict)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        log.error("Error during chat completions streaming: %s: %s", type(e).__name__, e)
        error_data = {
            "error": {
                "message": repr(e)[:200],
                "type": "internal_error",
                "code": "internal_chain_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        await _aclose_stream(stream)


async def iter_anthropic_sse(
    events: AsyncIterator[RawMessageStreamEvent | Mapping[str, object]],
) -> AsyncGenerator[str, None]:
    """Frame Anthropic events into ``event: <type>\\ndata: <json>\\n\\n``.

    Anthropic's SSE contract carries a named event per frame (unlike
    OpenAI Chat Completions, which only uses ``data:`` lines).  The
    event name comes from each event's ``"type"`` field —
    ``message_start``, ``content_block_delta``, ``message_stop``, etc.

    Accepts two producer shapes:

    * Plain ``dict`` events (from ``stream_openai_to_anthropic`` when
      the chain translates OpenAI → Anthropic on the fly).
    * Pydantic ``RawMessageStreamEvent`` models (from the Anthropic
      SDK's ``AsyncStream`` when the backend speaks Anthropic natively
      — see :class:`AnthropicNativeBackend`). Serialized via
      ``model_dump(exclude_none=True)``.

    No ``[DONE]`` terminator: Anthropic signals end-of-stream with a
    ``message_stop`` event, not a sentinel frame.

    Mid-stream iteration failures emit a final ``event: error`` frame
    with an ``{"error": {...}}`` payload and terminate — same error
    quarantine pattern as :func:`iter_chat_completion_sse`.

    Args:
        events: Async iterator of Anthropic events — dicts or pydantic
            models with ``model_dump``.

    Yields:
        SSE-framed strings suitable for ``StreamingResponse``.
    """
    try:
        async for event in events:
            if isinstance(event, Mapping):
                event_dict = dict(event)
            else:
                event_dict = event.model_dump(exclude_none=True)
            event_type = event_dict.get("type", "message")
            yield f"event: {event_type}\ndata: {json.dumps(event_dict)}\n\n"
    except Exception as e:
        log.error("Error during anthropic streaming: %s: %s", type(e).__name__, e)
        error_data = {
            "type": "error",
            "error": {
                "message": repr(e)[:200],
                "type": "internal_error",
            },
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
    finally:
        await _aclose_stream(events)


async def iter_gemini_sse(
    chunks: AsyncIterator[Mapping[str, object]],
) -> AsyncGenerator[str, None]:
    """Frame Gemini stream chunks into unnamed ``data: <json>\\n\\n`` lines.

    Gemini's ``streamGenerateContent?alt=sse`` contract carries one complete
    ``GenerateContentResponse`` JSON object per ``data:`` line, with no named
    events and no ``[DONE]`` terminator — the final chunk is recognizable by
    its ``finishReason`` and ``usageMetadata``.

    Mid-stream iteration failures emit a final ``data:`` frame carrying a
    Gemini-shaped ``{"error": {...}}`` payload and terminate — same error
    quarantine pattern as :func:`iter_chat_completion_sse`.

    Args:
        chunks: Async iterator of Gemini chunk dicts.

    Yields:
        SSE-framed strings suitable for ``StreamingResponse``.
    """
    try:
        async for chunk in chunks:
            chunk_dict = dict(chunk) if isinstance(chunk, Mapping) else {"data": str(chunk)}
            yield f"data: {json.dumps(chunk_dict)}\n\n"
    except Exception as e:
        log.error("Error during gemini streaming: %s: %s", type(e).__name__, e)
        error_data = {
            "error": {
                "code": 500,
                "message": repr(e)[:200],
                "status": "INTERNAL",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        await _aclose_stream(chunks)


async def iter_preframed_sse(
    frames: AsyncIterator[object],
) -> AsyncGenerator[str, None]:
    """Forward Responses SSE strings or frame native Responses events.

    The Responses API translator (:func:`stream_chat_to_responses_sse`)
    already yields fully-formatted SSE frames (``"event: ...\\ndata: ...\\n\\n"``)
    so the endpoint just needs a thin wrapper that preserves the same
    mid-stream error contract as the other helpers. Native Responses upstreams
    yield JSON events instead, so this helper frames those mappings using each
    event's ``type`` field.

    Args:
        frames: Async iterator of pre-formatted SSE strings or Responses events.

    Yields:
        SSE frames; on exception, a final ``error`` frame.
    """
    try:
        async for frame in frames:
            if isinstance(frame, str):
                yield frame
                continue
            if isinstance(frame, Mapping):
                event_dict = dict(frame)
            elif hasattr(frame, "model_dump"):
                event_dict = frame.model_dump(exclude_none=True)
            elif hasattr(frame, "to_dict"):
                event_dict = frame.to_dict()
            else:
                event_dict = {"type": "message", "data": str(frame)}
            event_type = event_dict.get("type", "message")
            yield f"event: {event_type}\ndata: {json.dumps(event_dict)}\n\n"
    except Exception as e:
        log.error("Error during responses streaming: %s: %s", type(e).__name__, e)
        error_data = {
            "type": "error",
            "error": {
                "message": repr(e)[:200],
                "type": "internal_error",
                "code": "internal_chain_error",
            },
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
    finally:
        await _aclose_stream(frames)
