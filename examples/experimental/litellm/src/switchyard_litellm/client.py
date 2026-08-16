# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM gateway adapter for normalized libsy requests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from litellm import ModelResponse, acompletion
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)

from switchyard.libsy import ContextWindowExceededError

_TEXT_ROLES = {"system", "developer", "user"}
_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _reject_sequence_payload(
    request: Mapping[str, object],
    field: str,
) -> None:
    value = request.get(field)
    if value is None:
        return
    if _sequence(value, field):
        raise ValueError(f"{field} is not supported")


def _reject_extensions_payload(request: Mapping[str, object]) -> None:
    value = request.get("extensions")
    if value is None:
        return
    extensions = _mapping(value, "extensions")
    if set(extensions) - {"fields"}:
        raise ValueError("extensions is not supported")
    if "fields" in extensions and _mapping(extensions["fields"], "extensions.fields"):
        raise ValueError("extensions is not supported")


def _reject_preservation_payload(request: Mapping[str, object]) -> None:
    value = request.get("preservation")
    if value is None:
        return
    preservation = _mapping(value, "preservation")
    if set(preservation) - {"requests", "responses"}:
        raise ValueError("preservation is not supported")
    for field in ("requests", "responses"):
        if field in preservation and _mapping(
            preservation[field], f"preservation.{field}"
        ):
            raise ValueError("preservation is not supported")


def _text_block(raw_block: object, path: str) -> dict[str, str]:
    block = _mapping(raw_block, path)
    if block.get("type") != "text" or not isinstance(block.get("text"), str):
        raise ValueError(f"{path} must be a text block")
    return {"type": "text", "text": cast(str, block["text"])}


def _tool_call(raw_block: object, path: str) -> dict[str, object]:
    block = _mapping(raw_block, path)
    call_id = block.get("id")
    name = block.get("name")
    arguments = block.get("arguments")
    if block.get("type") != "tool_call" or not isinstance(call_id, str):
        raise ValueError(f"{path} must be a tool-call block")
    if not isinstance(name, str):
        raise ValueError(f"{path}.name must be a string")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{path}.arguments must be a mapping")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _tool_result(raw_block: object, path: str) -> dict[str, object]:
    block = _mapping(raw_block, path)
    call_id = block.get("tool_call_id")
    if block.get("type") != "tool_result" or not isinstance(call_id, str):
        raise ValueError(f"{path} must be a tool-result block")
    content = _sequence(block.get("content"), f"{path}.content")
    text = [
        _text_block(raw_content, f"{path}.content[{index}]")["text"]
        for index, raw_content in enumerate(content)
    ]
    return {"role": "tool", "tool_call_id": call_id, "content": "\n".join(text)}


def _messages(request: Mapping[str, object]) -> list[dict[str, object]]:
    messages = _sequence(request.get("messages"), "messages")
    converted: list[dict[str, object]] = []
    for message_index, raw_message in enumerate(messages):
        message_path = f"messages[{message_index}]"
        message = _mapping(raw_message, message_path)
        role = message.get("role")
        blocks = _sequence(message.get("content"), f"{message_path}.content")
        if role in _TEXT_ROLES:
            content = [
                _text_block(raw_block, f"{message_path}.content[{block_index}]")
                for block_index, raw_block in enumerate(blocks)
            ]
            converted.append({"role": role, "content": content})
        elif role == "assistant":
            content: list[dict[str, str]] = []
            tool_calls: list[dict[str, object]] = []
            for block_index, raw_block in enumerate(blocks):
                block_path = f"{message_path}.content[{block_index}]"
                block = _mapping(raw_block, block_path)
                if block.get("type") == "text":
                    content.append(_text_block(block, block_path))
                elif block.get("type") == "tool_call":
                    tool_calls.append(_tool_call(block, block_path))
                else:
                    raise ValueError(f"{block_path} is not supported")
            converted_message: dict[str, object] = {
                "role": "assistant",
                "content": content or None,
            }
            if tool_calls:
                converted_message["tool_calls"] = tool_calls
            converted.append(converted_message)
        elif role == "tool":
            converted.extend(
                _tool_result(raw_block, f"{message_path}.content[{block_index}]")
                for block_index, raw_block in enumerate(blocks)
            )
        else:
            raise ValueError(f"{message_path}.role is not supported")
    if not converted:
        raise ValueError("messages must not be empty")
    return converted


def _tools(request: Mapping[str, object]) -> list[dict[str, object]]:
    tools = _sequence(request.get("tools", []), "tools")
    converted: list[dict[str, object]] = []
    for index, raw_tool in enumerate(tools):
        path = f"tools[{index}]"
        tool = _mapping(raw_tool, path)
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("parameters")
        strict = tool.get("strict")
        if not isinstance(name, str):
            raise ValueError(f"{path}.name must be a string")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{path}.description must be a string or null")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{path}.parameters must be a mapping")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError(f"{path}.strict must be a boolean or null")
        function: dict[str, object] = {"name": name, "parameters": dict(parameters)}
        if description is not None:
            function["description"] = description
        if strict is not None:
            function["strict"] = strict
        converted.append({"type": "function", "function": function})
    return converted


def _tool_choice(request: Mapping[str, object]) -> object | None:
    value = request.get("tool_choice")
    if value is None:
        return None
    choice = _mapping(value, "tool_choice")
    choice_type = choice.get("type")
    if choice_type in {"auto", "required", "none"}:
        return choice_type
    if choice_type == "tool":
        data = _mapping(choice.get("data"), "tool_choice.data")
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError("tool_choice.data.name must be a string")
        return {"type": "function", "function": {"name": name}}
    raise ValueError("tool_choice is not supported")


def _optional_mapping(
    request: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    value = request.get(field)
    if value is None:
        return {}
    return _mapping(value, field)


async def _payload(request: Mapping[str, object]) -> dict[str, Any]:
    model = request.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if request.get("stream") is True:
        raise ValueError("stream=True is not supported")
    _reject_sequence_payload(request, "instructions")
    _reject_extensions_payload(request)
    _reject_preservation_payload(request)

    sampling = _optional_mapping(request, "sampling")
    output = _optional_mapping(request, "output")
    reasoning = _optional_mapping(request, "reasoning")
    if sampling.get("top_k") is not None:
        raise ValueError("sampling.top_k is not supported")
    if output.get("response_format") is not None:
        raise ValueError("output.response_format is not supported")
    if reasoning.get("raw") is not None:
        raise ValueError("reasoning.raw is not supported")

    payload: dict[str, Any] = {
        # LiteLLM needs this prefix to select its OpenAI-compatible transport,
        # then strips it before sending the selected alias to the gateway.
        "model": f"openai/{model}",
        "messages": _messages(request),
        "stream": False,
    }
    tools = _tools(request)
    if tools:
        payload["tools"] = tools
    tool_choice = _tool_choice(request)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    for source, target in (("temperature", "temperature"), ("top_p", "top_p")):
        value = sampling.get(source)
        if value is not None:
            payload[target] = value
    max_tokens = output.get("max_output_tokens")
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    effort = reasoning.get("effort")
    if effort is not None:
        payload["reasoning_effort"] = effort
    return payload


def _usage(response: ModelResponse) -> dict[str, int]:
    usage = response.usage
    if usage is None:
        return {}
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cached_value = prompt_details.cached_tokens if prompt_details is not None else None
    cached = cached_value if cached_value is not None else 0
    normalized = {
        "input_tokens": max(usage.prompt_tokens - cached, 0),
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    if cached_value is not None:
        normalized["cached_input_tokens"] = cached
    if completion_details is not None and completion_details.reasoning_tokens is not None:
        normalized["reasoning_tokens"] = completion_details.reasoning_tokens
    return normalized


def _response(response: ModelResponse) -> dict[str, object]:
    if not response.choices:
        raise ValueError("LiteLLM returned no choices")
    choice = response.choices[0]
    text = choice.message.content
    content: list[dict[str, object]] = []
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for tool_call in choice.message.tool_calls or []:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("LiteLLM returned invalid tool-call arguments") from error
        if not isinstance(arguments, Mapping):
            raise ValueError("LiteLLM returned invalid tool-call arguments")
        content.append({
            "type": "tool_call",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": arguments,
        })
    if not content:
        raise ValueError("LiteLLM returned no text content")
    return {
        "id": response.id,
        "model": response.model,
        "outputs": [
            {
                "role": "assistant",
                "content": content,
                "stop_reason": _STOP_REASONS.get(choice.finish_reason, "unknown"),
            }
        ],
        "usage": _usage(response),
    }


class LiteLLMSyClient:
    """Call the LiteLLM gateway alias selected in a libsy request."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:4000/v1",
        api_key: str = "not-needed",
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key

    async def call(
        self,
        sy_request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Send one normalized, buffered text request through LiteLLM."""
        payload = await _payload(sy_request)
        try:
            response = await acompletion(
                **payload,
                api_base=self._base_url,
                api_key=self._api_key,
                num_retries=0,
                # LiteLLM otherwise imports its optional proxy MCP stack for ordinary
                # OpenAI function tools before it determines that they are not MCP tools.
                _skip_mcp_handler=True,
                allowed_openai_params=["reasoning_effort"],
                extra_body={"allowed_openai_params": ["reasoning_effort"]},
            )
        except LiteLLMContextWindowExceededError as error:
            raise ContextWindowExceededError(str(error)) from error
        return _response(cast(ModelResponse, response))

    async def aclose(self) -> None:
        """Retain asynchronous lifecycle compatibility without owning a transport."""
