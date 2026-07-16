# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for condensing chat requests into routing-side LLM prompts.

The LLM classifier, the escalation judge, and the plan-execute planner all
show an inner LLM a compact view of the live request: anchors (system + first
user message) plus a recent window, with bulk trimmed. These helpers are the
common plumbing — message trimming, content flattening, truncation, and
output-fence stripping — shared so the routing processors condense requests
consistently instead of each keeping a private copy.
"""

from __future__ import annotations

import json
from typing import Any


def trim_messages(messages: list[Any], *, recent_turn_window: int = 0) -> list[Any]:
    """Keep system + first-user anchor + a trailing window of messages.

    Anchors retained unconditionally:

    * system / developer messages — global framing the inner LLM
      always needs.
    * the **first** user message — agent frameworks like terminus-2
      bundle task framing into ``role="user"`` rather than ``system``;
      losing it leaves the inner LLM blind to what the agent is
      working on.

    Trailing window controlled by ``recent_turn_window``:

    * ``0`` (default) — keep only the last user message. Smallest
      prompt, but blind to recent assistant tool calls and
      tool results, so signal estimation (``tool_call_count_estimate``,
      DEBUG-vs-EXPLORATION turn type) must guess from a terse
      "Continue" echo. Tends to over-escalate on pessimistic
      classifiers.
    * ``N >= 1`` — keep the last ``N`` non-anchor messages
      (assistant / tool / non-first user) in original order. Gives
      the inner LLM visibility into recent agent activity. Each new
      turn appends to a stable prefix the upstream can prompt-cache,
      so the extra tokens are nearly free on cache-discounted backends
      (DeepSeek V4 Flash ~98% cache discount).
    """
    system_msgs: list[Any] = []
    first_user: Any = None
    first_user_idx: int | None = None
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            system_msgs.append(m)
        elif role == "user" and first_user is None:
            first_user = m
            first_user_idx = idx

    if first_user is None:
        return system_msgs

    # Candidate tail = everything after the first user message that
    # isn't a system/developer anchor (those are already included).
    tail_candidates = [
        m
        for idx, m in enumerate(messages)
        if idx > (first_user_idx or 0)
        and isinstance(m, dict)
        and m.get("role") not in ("system", "developer")
    ]

    if recent_turn_window <= 0:
        # Historical behavior: keep only the last user message.
        last_user: Any = None
        for m in tail_candidates:
            if m.get("role") == "user":
                last_user = m
        if last_user is None:
            return [*system_msgs, first_user]
        if last_user is first_user:
            return [*system_msgs, first_user]
        return [*system_msgs, first_user, last_user]

    window = tail_candidates[-recent_turn_window:]
    # Filter out the first user (already pinned) to avoid duplicating
    # it if the window reaches back that far on short conversations.
    window = [m for m in window if m is not first_user]
    return [*system_msgs, first_user, *window]


def message_text(message: dict[str, Any]) -> str:
    """Flatten a chat message (or Responses item) to plain text, tool calls included."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str | list):
        text = content_text(content)
        if text:
            parts.append(text)
    # OpenAI chat tool calls: the command shapes an inner LLM needs for loop detection.
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if isinstance(fn, dict):
                parts.append(f"tool_call {fn.get('name')}({fn.get('arguments', '')})")
    # OpenAI Responses items carry their payloads in type-specific fields.
    if message.get("type") == "function_call":
        parts.append(f"tool_call {message.get('name')}({message.get('arguments', '')})")
    output = message.get("output")
    if isinstance(output, str | list):
        text = content_text(output)
        if text:
            parts.append(text)
    return " ".join(parts)


def content_text(content: str | list[Any]) -> str:
    """Flatten string-or-content-block message content to text.

    Handles plain strings, OpenAI/Anthropic text blocks, nested
    ``tool_result`` content, and Anthropic ``tool_use`` blocks — the last
    carry their payload in ``name``/``input`` rather than ``text``/``content``
    and would otherwise flatten to nothing, erasing exactly the
    repeated-command signal a trajectory judge relies on.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            arguments = json.dumps(
                block.get("input", {}), ensure_ascii=False, sort_keys=True, default=str
            )
            parts.append(f"tool_call {block.get('name')}({arguments})")
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        inner = block.get("content")
        if isinstance(inner, str | list):
            parts.append(content_text(inner))
    return " ".join(p for p in parts if p)


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` within ``limit`` chars."""
    if len(text) <= limit:
        return text
    marker = " ...[trimmed] "
    keep = max(limit - len(marker), 20)
    head = (keep * 2) // 3
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def strip_markdown_fence(raw: str) -> str:
    """Drop a wrapping ``` fence from an inner LLM's JSON output, if present."""
    stripped = raw.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = [
    "content_text",
    "message_text",
    "strip_markdown_fence",
    "trim_messages",
    "truncate_middle",
]
