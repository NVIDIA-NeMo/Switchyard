# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM gateway adapter for normalized libsy requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from litellm import ModelResponse, acompletion

_ROLES = {"system", "developer", "user", "assistant"}
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


def _messages(request: Mapping[str, object]) -> list[dict[str, object]]:
    messages = _sequence(request.get("messages"), "messages")
    converted: list[dict[str, object]] = []
    for message_index, raw_message in enumerate(messages):
        message_path = f"messages[{message_index}]"
        message = _mapping(raw_message, message_path)
        role = message.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError(f"{message_path}.role is not supported")
        blocks = _sequence(message.get("content"), f"{message_path}.content")
        content: list[dict[str, str]] = []
        for block_index, raw_block in enumerate(blocks):
            block_path = f"{message_path}.content[{block_index}]"
            block = _mapping(raw_block, block_path)
            if block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise ValueError(f"{block_path} must be a text block")
            content.append({"type": "text", "text": cast(str, block["text"])})
        converted.append({"role": role, "content": content})
    if not converted:
        raise ValueError("messages must not be empty")
    return converted


def _optional_mapping(
    request: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    value = request.get(field)
    if value is None:
        return {}
    return _mapping(value, field)


def _payload(request: Mapping[str, object], model: str) -> dict[str, Any]:
    if request.get("stream") is True:
        raise ValueError("stream=True is not supported")
    for field in ("instructions", "tools"):
        _reject_sequence_payload(request, field)
    if request.get("tool_choice") is not None:
        raise ValueError("tool_choice is not supported")
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
        "model": model,
        "messages": _messages(request),
        "stream": False,
    }
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
    if not isinstance(text, str) or not text:
        raise ValueError("LiteLLM returned no text content")
    return {
        "id": response.id,
        "model": response.model,
        "outputs": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "stop_reason": _STOP_REASONS.get(choice.finish_reason, "unknown"),
            }
        ],
        "usage": _usage(response),
    }


class LiteLLMSyClient:
    """Call a LiteLLM gateway alias for a libsy target."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:4000/v1",
        api_key: str = "not-needed",
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self._base_url = base_url
        self._api_key = api_key

    async def call(
        self,
        sy_request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Send one normalized, buffered text request through LiteLLM."""
        response = await acompletion(
            **_payload(sy_request, f"openai/{self.model}"),
            api_base=self._base_url,
            api_key=self._api_key,
            num_retries=0,
            allowed_openai_params=["reasoning_effort"],
            extra_body={"allowed_openai_params": ["reasoning_effort"]},
        )
        return _response(cast(ModelResponse, response))

    async def aclose(self) -> None:
        """Retain asynchronous lifecycle compatibility without owning a transport."""
