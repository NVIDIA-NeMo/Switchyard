# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn counting helpers for conversation-scoped routing decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from switchyard_rust.core import ChatRequest


def conversation_turn_number(request: ChatRequest) -> int:
    """Return the 1-indexed LLM invocation number for ``request``.

    OpenAI Chat and Anthropic Messages count prior assistant messages. OpenAI
    Responses uses a coarse acknowledgement count because responses can emit
    multiple model-side items per turn. Unknown or malformed request bodies are
    treated as turn 1 so routing falls back to first-turn behavior.
    """
    from switchyard_rust.core import ChatRequestType

    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return 1

    match request.request_type:
        case ChatRequestType.OPENAI_CHAT | ChatRequestType.ANTHROPIC:
            return _count_assistant_messages(body.get("messages")) + 1
        case ChatRequestType.OPENAI_RESPONSES:
            return _count_responses_turn(body)


def _count_assistant_messages(messages: Any) -> int:
    """Count prior assistant messages in chat-style request bodies."""
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )


def _count_responses_turn(body: dict[str, Any]) -> int:
    """Approximate the turn number for an OpenAI Responses request body."""
    input_val = body.get("input")
    if isinstance(input_val, str):
        return 1
    if not isinstance(input_val, list):
        return 1

    acks = 0
    for item in input_val:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            acks += 1
        elif item.get("type") == "function_call_output":
            acks += 1
    return max(acks, 1)


__all__ = ["conversation_turn_number"]
