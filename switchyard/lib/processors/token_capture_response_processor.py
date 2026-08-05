# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-side processor that writes per-call token-level completion records."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid as uuid_lib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard, cast

from switchyard.lib.processors.rl_logging_request_processor import (
    CTX_RL_LOGGING_REQUEST,
    RlLoggingRequestProcessor,
)
from switchyard.lib.processors.rl_logging_response_processor import (
    build_trace_messages,
    build_trace_token_count,
    format_trace_tool_choice,
    format_trace_tools,
)
from switchyard.lib.processors.token_capture_request_processor import (
    CTX_TOKEN_CAPTURE_ORIGINAL_STREAM,
    CTX_TOKEN_CAPTURE_PARENT_SESSION,
    CTX_TOKEN_CAPTURE_SESSION,
    TokenCaptureRequestProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatResponse, ChatResponseType, response_type_matches

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]

#: Records live under ``<rl-log-dir>/sessions/<hashed-key>/<uuid>.json`` so token
#: capture never collides with flat text traces in the same log directory.
_SESSIONS_SUBDIR = "sessions"

#: Schema version stamped on every stored record and retrieval envelope.
SCHEMA_VERSION = 1


class TokenCaptureResponseProcessor:
    """Write one token-level completion record per captured turn to ``capture_dir``.

    Runs before the terminal ``TranslationEngine``, so the raw backend body
    still carries the engine token fields (``prompt_token_ids``,
    ``choice.token_ids``, ``logprobs.content[]``) that translation would drop.
    Each record unifies the RL text trace (messages/tools built from the
    ``CTX_RL_LOGGING_REQUEST`` snapshot) with those token-level fields. The
    live response is returned unchanged — this processor only observes.

    Calls without a resolved capture session (``CTX_TOKEN_CAPTURE_SESSION``)
    are skipped. Records that fail token-field validation are still stored,
    flagged ``is_valid: false`` — capture never affects the harness response.
    """

    def __init__(self, capture_dir: Path | str) -> None:
        self._capture_dir = Path(capture_dir)
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._warned_streaming = False

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        """Record the completed turn's token-level data.

        Returns the response unchanged, except when the request processor
        flipped a streaming request to non-streaming: the buffered completion
        is then re-emitted as a synthetic OpenAI chunk stream so the terminal
        ``TranslationEngine`` can stream to the client in its native format.
        """
        session_id = ctx.metadata.get(CTX_TOKEN_CAPTURE_SESSION)
        if not isinstance(session_id, str) or not session_id:
            logger.debug("token capture: no capture session on context; skipping record")
            return response
        if not response_type_matches(response, ChatResponseType.OPENAI_COMPLETION):
            self._note_skip(response)
            return response

        record = self._build_record(ctx, session_id, response)
        try:
            self._write_record(session_id, record)
        except OSError as exc:
            logger.warning(
                "token capture: failed to write record to %s: %s",
                self._capture_dir,
                exc,
            )
        if ctx.metadata.get(CTX_TOKEN_CAPTURE_ORIGINAL_STREAM):
            return _synthesize_stream(dict(response.body))
        return response

    def _note_skip(self, response: ChatResponse) -> None:
        streaming = response_type_matches(response, ChatResponseType.OPENAI_STREAM)
        if streaming and not self._warned_streaming:
            self._warned_streaming = True
            logger.warning(
                "token capture: a streaming response reached the capture "
                "processor; record skipped (unexpected — capture forces "
                "non-streaming upstream)",
            )

    def _build_record(
        self, ctx: ProxyContext, session_id: str, response: ChatResponse
    ) -> JsonObject:
        body = dict(response.body)
        request = ctx.metadata.get(CTX_RL_LOGGING_REQUEST)
        request = request if isinstance(request, dict) else {}
        choices = body.get("choices")
        choices = choices if isinstance(choices, list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}

        request_id = body.get("id")
        model = body.get("model")
        prompt_token_ids = body.get("prompt_token_ids")
        generation_token_ids = choice.get("token_ids")
        generation_log_probs = _extract_log_probs(choice.get("logprobs"))
        problem = _validate_token_fields(
            request_id, model, prompt_token_ids, generation_token_ids, generation_log_probs, len(choices)
        )
        if problem is not None:
            logger.warning("token capture: storing record with is_valid=false: %s", problem)

        return {
            "schema_version": SCHEMA_VERSION,
            "uuid": str(uuid_lib.uuid4()),
            "session_id": session_id,
            "parent_session_id": ctx.metadata.get(CTX_TOKEN_CAPTURE_PARENT_SESSION),
            "captured_at": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "model": model,
            "messages": build_trace_messages(request, message),
            "tools": format_trace_tools(request.get("tools", [])),
            "tool_choice": format_trace_tool_choice(request.get("tool_choice")),
            "token_count": build_trace_token_count(body.get("usage")),
            "prompt_token_ids": prompt_token_ids,
            "generation_token_ids": generation_token_ids,
            "generation_log_probs": generation_log_probs,
            "finish_reason": choice.get("finish_reason"),
            "is_valid": problem is None,
        }

    def _write_record(self, session_id: str, record: JsonObject) -> None:
        session_dir = sessions_root(self._capture_dir) / session_dir_name(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{record['uuid']}.json"
        # Write-then-rename so the retrieval endpoint never reads a torn record
        # (the endpoint globs *.json; the .tmp suffix keeps partials invisible).
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w") as handle:
            json.dump(record, handle, indent=2)
        os.replace(tmp_path, path)
        # Records carry full conversation text; keep them owner-only.
        os.chmod(path, 0o600)

    def get_endpoint(self) -> Any:
        """Contribute ``GET /v1/sessions`` record retrieval to the server."""
        from switchyard.lib.endpoints.token_capture_endpoint import (
            TokenCaptureSessionsEndpoint,
        )

        return TokenCaptureSessionsEndpoint(self._capture_dir)


def build_token_capture_processors(
    capture_dir: Path | None,
) -> tuple[list[Any], list[Any]]:
    """Request/response processor lists for token-level capture.

    The request side pairs :class:`RlLoggingRequestProcessor` — whose
    translated snapshot supplies the record's ``messages``/``tools`` — with
    :class:`TokenCaptureRequestProcessor`. Returns ``([], [])`` when
    ``capture_dir`` is ``None`` (capture disabled). Shared by the ``launch``
    and ``serve`` wiring.
    """
    if capture_dir is None:
        return [], []
    return (
        [RlLoggingRequestProcessor(), TokenCaptureRequestProcessor()],
        [TokenCaptureResponseProcessor(capture_dir)],
    )


def _extract_log_probs(logprobs: object) -> list[Any] | None:
    """``[entry.logprob for entry in logprobs.content]``, or ``None`` off-shape."""
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list):
        return None
    return [entry.get("logprob") if isinstance(entry, dict) else None for entry in content]


def _validate_token_fields(
    request_id: object,
    model: object,
    prompt_token_ids: object,
    generation_token_ids: object,
    generation_log_probs: object,
    choice_count: int,
) -> str | None:
    """Reason the record is unusable for training, or ``None`` if valid."""
    if choice_count != 1:
        return f"response has {choice_count} choices; multiple-choice capture is unsupported"
    # Provenance: the design requires request id and model identity. ``model``
    # is load-bearing — token ids are meaningless without the tokenizer that
    # produced them — and real vLLM always emits both, so absence is anomalous.
    if not isinstance(request_id, str) or not request_id:
        return "request_id is missing or not a non-empty string"
    if not isinstance(model, str) or not model:
        return "model is missing or not a non-empty string"
    if not _is_token_id_list(prompt_token_ids):
        return "prompt_token_ids is not a non-empty list of ints"
    if not _is_token_id_list(generation_token_ids):
        return "generation_token_ids is not a non-empty list of ints"
    if not isinstance(generation_log_probs, list) or not all(
        _is_finite_number(value) for value in generation_log_probs
    ):
        return "generation_log_probs is not a list of finite floats"
    if len(generation_token_ids) != len(generation_log_probs):
        return (
            f"generation_token_ids ({len(generation_token_ids)}) and "
            f"generation_log_probs ({len(generation_log_probs)}) are misaligned"
        )
    return None


def _is_token_id_list(value: object) -> TypeGuard[list[int]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _synthesize_stream(body: JsonObject) -> ChatResponse:
    """Re-emit a buffered completion as a synthetic OpenAI chunk stream.

    The harness asked for streaming, but the upstream call was forced
    non-streaming for token capture. Two chunks reproduce the completion:
    the first carries the assistant delta (content + tool calls), the second
    carries ``finish_reason`` and usage. First choice only — training
    harnesses run ``n=1``. Token-level fields stay in the stored record;
    chunks carry only the standard wire fields.
    """
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
    )
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from pydantic import ValidationError

    from switchyard.lib.chat_response.openai_chat import ResponseStream

    completion_id = str(body.get("id") or "chatcmpl-token-capture")
    created = int(body.get("created") or 0)
    model = str(body.get("model") or "unknown")

    choices = body.get("choices")
    choice: JsonObject = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
    message_obj = choice.get("message")
    message: JsonObject = message_obj if isinstance(message_obj, dict) else {}

    delta_tool_calls: list[ChoiceDeltaToolCall] | None = None
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list) and raw_tool_calls:
        delta_tool_calls = [
            ChoiceDeltaToolCall(
                index=index,
                id=call.get("id"),
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=(call.get("function") or {}).get("name"),
                    arguments=(call.get("function") or {}).get("arguments"),
                ),
            )
            for index, call in enumerate(raw_tool_calls)
            if isinstance(call, dict)
        ]

    usage = body.get("usage")
    final_usage = None
    if isinstance(usage, dict):
        # A malformed usage block must not abort the client-facing stream —
        # drop it rather than raise from model_validate.
        try:
            final_usage = CompletionUsage.model_validate(usage)
        except ValidationError as exc:
            logger.warning("token capture: dropping malformed usage in synthesized stream: %s", exc)
    finish_reason = cast(Any, choice.get("finish_reason") or "stop")

    async def _chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content=message.get("content"),
                        tool_calls=delta_tool_calls,
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason)],
            usage=final_usage,
        )

    return ChatResponse.openai_stream(ResponseStream(_chunks()))


def sessions_root(capture_dir: Path) -> Path:
    """Root under which per-session record directories live (writer + endpoint)."""
    return capture_dir / _SESSIONS_SUBDIR


def session_dir_name(session_id: str) -> str:
    """Collision-resistant directory name for a session id (writer + endpoint).

    A hash, not a sanitized form: sanitizing distinct opaque ids to the same
    safe string (e.g. ``tenant:a`` and ``tenant*a`` → ``tenant_a``) would let
    one session's records surface under another. The fixed-length hex output is
    also inherently path-safe, so no traversal guard is needed.
    """
    return hashlib.sha256(session_id.encode()).hexdigest()
