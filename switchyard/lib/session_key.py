# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable per-conversation key derived from the request prefix."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, overload


@overload
def session_key_from_body(body: Any, depth: Literal[0] = 0) -> str: ...


@overload
def session_key_from_body(body: Any, depth: int) -> str | None: ...


def session_key_from_body(body: Any, depth: int = 0) -> str | None:
    """Hash stable conversation anchors and an optional early-response prefix."""
    if depth < 0:
        raise ValueError("depth must be non-negative")

    parts: list[str] = [_flatten_text(_field(body, "system"))]
    anchored = False
    tail_taken = 0

    for sequence_name in ("messages", "input"):
        items = _field(body, sequence_name)
        if not isinstance(items, list):
            continue
        for item in items:
            role = _field(item, "role")
            if role in {"system", "developer"}:
                parts.append(_flatten_text(_field(item, "content")))
            elif role == "user" and not anchored:
                parts.append(_flatten_text(_field(item, "content")))
                anchored = True
            elif anchored and tail_taken < depth:
                parts.append(_flatten_message(item))
                tail_taken += 1
            if anchored and tail_taken >= depth:
                break
        if anchored:
            break

    if depth > 0 and (not anchored or tail_taken < depth):
        return None

    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _flatten_message(item: Any) -> str:
    parts: list[str] = []
    content = _flatten_text(_field(item, "content"))
    if content:
        parts.append(content)

    blocks = _field(item, "content")
    if isinstance(blocks, list):
        for block in blocks:
            if _field(block, "type") == "tool_use":
                name = _field(block, "name") or ""
                raw_input = _field(block, "input")
                payload = _json_text(raw_input) if raw_input is not None else ""
                parts.append(f"tool_call {name}({payload})")

    calls = _field(item, "tool_calls")
    if isinstance(calls, list):
        for call in calls:
            function = _field(call, "function")
            if function is not None:
                raw_name = _field(function, "name")
                raw_arguments = _field(function, "arguments")
                name = raw_name if isinstance(raw_name, str) else ""
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, str)
                    else _json_text(raw_arguments)
                    if raw_arguments is not None
                    else ""
                )
                parts.append(f"tool_call {name}({arguments})")

    if _field(item, "type") == "function_call":
        raw_name = _field(item, "name")
        raw_arguments = _field(item, "arguments")
        name = raw_name if isinstance(raw_name, str) else ""
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, str)
            else _json_text(raw_arguments)
            if raw_arguments is not None
            else ""
        )
        parts.append(f"tool_call {name}({arguments})")

    output = _flatten_text(_field(item, "output"))
    if output:
        parts.append(output)
    return " ".join(parts)


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if not isinstance(text, str) or not text:
                    text = block.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
            else:
                parts.append(_json_text(block))
        return "".join(parts)
    if content is None:
        return ""
    return _json_text(content)


def _json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


__all__ = ["session_key_from_body"]
