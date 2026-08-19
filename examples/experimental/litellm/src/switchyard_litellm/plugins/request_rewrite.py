# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate Switchyard request deltas into LiteLLM completion arguments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

_REQUEST_FIELDS = frozenset(
    {
        "model",
        "instructions",
        "messages",
        "tools",
        "tool_choice",
        "sampling",
        "output",
        "reasoning",
        "stream",
        "extensions",
        "preservation",
    }
)
_EXTENSION_FIELDS = {
    "parallel_tool_calls": "parallel_tool_calls",
    "prompt_cache_key": "prompt_cache_key",
    "prompt_cache_retention": "prompt_cache_retention",
    "safety_identifier": "safety_identifier",
    "service_tier": "service_tier",
    "store": "store",
    "stream_options": "stream_options",
    "top_logprobs": "top_logprobs",
    "user": "user",
    "stop_sequences": "stop",
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _text_content(value: object, path: str, separator: str) -> str:
    parts: list[str] = []
    for index, raw_block in enumerate(_sequence(value, path)):
        block_path = f"{path}[{index}]"
        block = _mapping(raw_block, block_path)
        text = block.get("text")
        if block.get("type") != "text" or not isinstance(text, str):
            raise ValueError(f"{block_path} cannot safely apply non-text content to LiteLLM")
        parts.append(text)
    return separator.join(parts)


def _instructions(value: object) -> list[dict[str, object]]:
    instructions: list[dict[str, object]] = []
    for index, raw_instruction in enumerate(_sequence(value, "request.instructions")):
        path = f"request.instructions[{index}]"
        instruction = _mapping(raw_instruction, path)
        role = instruction.get("role")
        if role not in {"system", "developer"}:
            raise ValueError(f"{path} cannot safely apply role {role!r} to LiteLLM")
        instructions.append(
            {
                "role": role,
                "content": _text_content(instruction.get("content"), f"{path}.content", "\n\n"),
            }
        )
    return instructions


def _tool_call(block: Mapping[str, object], path: str) -> dict[str, object]:
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"{path}.id cannot safely apply an empty tool-call ID to LiteLLM")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}.name cannot safely apply an empty tool name to LiteLLM")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(block.get("arguments"), separators=(",", ":")),
        },
    }


def _tool_result(block: Mapping[str, object], path: str) -> dict[str, object]:
    tool_call_id = block.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError(f"{path}.tool_call_id cannot safely apply an empty ID to LiteLLM")
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _text_content(block.get("content"), f"{path}.content", " "),
    }


def _messages(value: object) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for index, raw_message in enumerate(_sequence(value, "request.messages")):
        path = f"request.messages[{index}]"
        message = _mapping(raw_message, path)
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"{path} cannot safely apply role {role!r} to LiteLLM")

        text_blocks: list[dict[str, object]] = []
        tool_calls: list[dict[str, object]] = []
        tool_results: list[dict[str, object]] = []
        for block_index, raw_block in enumerate(_sequence(message.get("content"), f"{path}.content")):
            block_path = f"{path}.content[{block_index}]"
            block = _mapping(raw_block, block_path)
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                text_blocks.append(dict(block))
            elif block_type == "tool_call" and role == "assistant":
                tool_calls.append(_tool_call(block, block_path))
            elif block_type == "tool_result" and role in {"tool", "user"}:
                tool_results.append(_tool_result(block, block_path))
            else:
                raise ValueError(
                    f"{block_path} cannot safely apply content type {block_type!r} to LiteLLM"
                )

        messages.extend(tool_results)
        if text_blocks or tool_calls or not tool_results:
            converted: dict[str, object] = {
                "role": role,
                "content": "\n".join(cast(str, block["text"]) for block in text_blocks),
            }
            if tool_calls:
                converted["tool_calls"] = tool_calls
                if not text_blocks:
                    converted["content"] = None
            messages.append(converted)
    return messages


def _tools(value: object) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    for index, raw_tool in enumerate(_sequence(value, "request.tools")):
        path = f"request.tools[{index}]"
        tool = _mapping(raw_tool, path)
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name cannot safely apply an empty tool name to LiteLLM")
        function: dict[str, object] = {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters", {}),
        }
        strict = tool.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError(f"{path}.strict cannot safely apply a non-Boolean value")
            function["strict"] = strict
        tools.append({"type": "function", "function": function})
    return tools


def _tool_choice(value: object) -> object:
    if value is None:
        return None
    choice = _mapping(value, "request.tool_choice")
    choice_type = choice.get("type")
    if choice_type in {"auto", "required", "none"}:
        return choice_type
    if choice_type == "tool":
        data = _mapping(choice.get("data"), "request.tool_choice.data")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("request.tool_choice cannot safely apply an empty tool name")
        return {"type": "function", "function": {"name": name}}
    if choice_type == "raw":
        return choice.get("data")
    raise ValueError(f"request.tool_choice cannot safely apply type {choice_type!r}")


def _changed_fields(
    original: Mapping[str, object],
    rewritten: Mapping[str, object],
    fields: Sequence[str],
) -> list[str]:
    return [field for field in fields if original.get(field) != rewritten.get(field)]


def build_request_patch(
    original_request: Mapping[str, object],
    rewritten_request: Mapping[str, object],
    *,
    selected_model_id: str,
) -> dict[str, object]:
    """Build a LiteLLM argument patch for fields explicitly changed by Switchyard."""
    if set(original_request) != _REQUEST_FIELDS or set(rewritten_request) != _REQUEST_FIELDS:
        raise ValueError("Switchyard request shape changed and cannot safely apply to LiteLLM")
    if rewritten_request.get("model") != selected_model_id:
        raise ValueError("Switchyard request model cannot safely apply to the selected deployment")
    if original_request.get("preservation") != rewritten_request.get("preservation"):
        raise ValueError("Switchyard preservation rewrite cannot safely apply to LiteLLM")

    set_values: dict[str, object] = {}
    remove_values: list[str] = []
    if _changed_fields(original_request, rewritten_request, ["instructions", "messages"]):
        set_values["messages"] = [
            *_instructions(rewritten_request.get("instructions")),
            *_messages(rewritten_request.get("messages")),
        ]
    if original_request.get("tools") != rewritten_request.get("tools"):
        set_values["tools"] = _tools(rewritten_request.get("tools"))
    if original_request.get("tool_choice") != rewritten_request.get("tool_choice"):
        set_values["tool_choice"] = _tool_choice(rewritten_request.get("tool_choice"))

    original_sampling = _mapping(original_request.get("sampling"), "original.sampling")
    rewritten_sampling = _mapping(rewritten_request.get("sampling"), "request.sampling")
    for field in ("temperature", "top_p", "top_k"):
        if original_sampling.get(field) != rewritten_sampling.get(field):
            set_values[field] = rewritten_sampling.get(field)

    original_output = _mapping(original_request.get("output"), "original.output")
    rewritten_output = _mapping(rewritten_request.get("output"), "request.output")
    if original_output.get("max_output_tokens") != rewritten_output.get("max_output_tokens"):
        set_values["max_completion_tokens"] = rewritten_output.get("max_output_tokens")
        remove_values.append("max_tokens")
    if original_output.get("response_format") != rewritten_output.get("response_format"):
        set_values["response_format"] = rewritten_output.get("response_format")

    original_reasoning = _mapping(original_request.get("reasoning"), "original.reasoning")
    rewritten_reasoning = _mapping(rewritten_request.get("reasoning"), "request.reasoning")
    if original_reasoning.get("raw") != rewritten_reasoning.get("raw"):
        raise ValueError("Switchyard raw reasoning rewrite cannot safely apply to LiteLLM")
    if original_reasoning.get("effort") != rewritten_reasoning.get("effort"):
        set_values["reasoning_effort"] = rewritten_reasoning.get("effort")

    if original_request.get("stream") != rewritten_request.get("stream"):
        stream = rewritten_request.get("stream")
        if not isinstance(stream, bool):
            raise ValueError("Switchyard stream rewrite cannot safely apply to LiteLLM")
        set_values["stream"] = stream

    original_extensions = _mapping(original_request.get("extensions"), "original.extensions")
    rewritten_extensions = _mapping(rewritten_request.get("extensions"), "request.extensions")
    original_fields = _mapping(original_extensions.get("fields"), "original.extensions.fields")
    rewritten_fields = _mapping(rewritten_extensions.get("fields"), "request.extensions.fields")
    changed_extension_fields = set(original_fields) | set(rewritten_fields)
    for field in sorted(changed_extension_fields):
        if original_fields.get(field) == rewritten_fields.get(field):
            continue
        litellm_field = _EXTENSION_FIELDS.get(field)
        if litellm_field is None:
            raise ValueError(
                f"Switchyard extension {field!r} cannot safely apply to LiteLLM"
            )
        if field in rewritten_fields:
            set_values[litellm_field] = rewritten_fields[field]
        else:
            remove_values.append(litellm_field)

    patch: dict[str, object] = {"set": set_values}
    if remove_values:
        patch["remove"] = remove_values
    return patch


__all__ = ["build_request_patch"]
