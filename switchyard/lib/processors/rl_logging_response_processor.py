# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-side processor that writes per-turn RL training traces to local JSON files."""

from __future__ import annotations

import json
import logging
import uuid as uuid_lib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchyard.lib.chat_response.streaming_response_accumulator import (
    attach_final_response_callback,
)
from switchyard.lib.processors.rl_logging_request_processor import (
    CTX_RL_LOGGING_REQUEST,
    RlLoggingRequestProcessor,
)
from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL, ProxyContext
from switchyard_rust.core import (
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)
from switchyard_rust.translation import TranslationEngine

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]


class RlLoggingResponseProcessor:
    """Write one ``message_history`` JSON trace per completed turn to ``log_dir``.

    Restores the pre-1.0 ``--enable-rl-logging`` local trace format: each
    request/response pair is written to its own file as
    ``{uuid, messages, tools, tool_choice, token_count, is_valid}``.

    Streaming responses are captured via
    :func:`attach_final_response_callback`, which accumulates the native stream
    and fires once it drains; non-streaming responses are logged inline. The
    response is always returned unchanged — this processor only observes.
    """

    def __init__(self, log_dir: Path | str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._translation = TranslationEngine()

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        """Log one ``message_history`` trace for the completed turn; return the response.

        Streaming responses are captured when the stream drains; non-streaming
        responses log inline. Write failures are swallowed (logged, never
        raised) so trace logging can never break the proxied response.
        """
        served_model: str = ctx.selected_model or ctx.metadata.get(
            CTX_PROXY_ACTUAL_MODEL, "unknown",
        )

        async def _emit(final: ChatResponse) -> None:
            self._write_trace(ctx, final)

        # Streaming responses log on stream completion; everything else is
        # already complete and logs inline.
        attached = attach_final_response_callback(
            response, served_model=served_model, callback=_emit,
        )
        if not attached:
            await _emit(response)
        return response

    def _write_trace(self, ctx: ProxyContext, response: ChatResponse) -> None:
        request = ctx.metadata.get(CTX_RL_LOGGING_REQUEST)
        if not isinstance(request, dict):
            return
        entry = self._build_entry(request, response)
        if entry is None:
            return
        try:
            self._write_entry(entry)
        except OSError as exc:
            logger.warning("RL logging: failed to write trace to %s: %s", self._log_dir, exc)

    def _build_entry(self, request: JsonObject, response: ChatResponse) -> JsonObject | None:
        translated = self._translation.response_to(ChatRequestType.OPENAI_CHAT, response)
        if not response_type_matches(translated, ChatResponseType.OPENAI_COMPLETION):
            return None
        body = dict(translated.body)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None

        return {
            "uuid": str(uuid_lib.uuid4()),
            "messages": build_trace_messages(request, message),
            "tools": format_trace_tools(request.get("tools", [])),
            "tool_choice": format_trace_tool_choice(request.get("tool_choice")),
            "token_count": build_trace_token_count(body.get("usage")),
            "is_valid": True,
        }

    def _write_entry(self, entry: JsonObject) -> None:
        path = self._log_dir / _trace_filename()
        with open(path, "w") as handle:
            json.dump(entry, handle, indent=2)


def build_rl_logging_processors(
    rl_log_dir: Path | None,
) -> tuple[list[Any], list[Any]]:
    """Request/response processor lists for local RL trace logging.

    Returns ``([], [])`` when ``rl_log_dir`` is ``None`` (logging disabled), or
    the paired snapshot + writer processors otherwise. Shared by the ``launch``
    and ``serve`` wiring.
    """
    if rl_log_dir is None:
        return [], []
    return [RlLoggingRequestProcessor()], [RlLoggingResponseProcessor(rl_log_dir)]


def _trace_filename() -> str:
    """File-safe ``{timestamp}_trace_{trace_id}_{suffix}.json`` (one file per turn)."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
    trace_id = uuid_lib.uuid4().hex[:8]
    suffix_id = uuid_lib.uuid4().hex[:8]
    return f"{timestamp}_trace_{trace_id}_{suffix_id}.json"


def build_trace_messages(request: JsonObject, message: JsonObject) -> list[JsonObject]:
    """Request-snapshot history plus the appended assistant turn.

    Shared by the RL trace logger and the token-capture record writer so the
    text-trace shape stays identical across both outputs.
    """
    messages = [dict(m) for m in request.get("messages", []) if isinstance(m, dict)]
    assistant: JsonObject = {"role": "assistant"}
    content = message.get("content")
    if content is not None:
        assistant["content"] = content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    messages.append(assistant)
    return messages


def build_trace_token_count(usage: object) -> JsonObject:
    """Normalize a response ``usage`` block into the trace ``token_count`` shape."""
    usage = usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def format_trace_tools(raw_tools: object) -> list[JsonObject]:
    """Port the V1 message_history tool shape: ``{id, description, inputSchema}``."""
    if not isinstance(raw_tools, list):
        return []
    tools: list[JsonObject] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        entry: JsonObject = {}
        function = tool.get("function")
        if isinstance(function, dict):
            entry["id"] = function.get("name", "")
            entry["description"] = function.get("description", "")
            if "parameters" in function:
                entry["inputSchema"] = {"jsonSchema": function["parameters"]}
        else:
            entry["id"] = tool.get("name", tool.get("id", ""))
            entry["description"] = tool.get("description", "")
            if "input_schema" in tool:
                entry["inputSchema"] = {"jsonSchema": tool["input_schema"]}
            elif "parameters" in tool:
                entry["inputSchema"] = {"jsonSchema": tool["parameters"]}
        tools.append(entry)
    return tools


def format_trace_tool_choice(tool_choice: object) -> str:
    """Extract the tool-choice ``type`` string, defaulting to ``"auto"``."""
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if isinstance(choice_type, str):
            return choice_type
    return "auto"
