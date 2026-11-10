# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use a compatible Switchyard algorithm to narrow LiteLLM routing candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from litellm.types.router import RoutingContext

from switchyard.libsy import Algorithm, Step

from .judge import JudgeClient, is_judge_call
from .lite_llm_request_rewriter import LiteLLMRequestRewriter
from .request_rewrite import build_request_patch


def _mapping(value: object, path: str) -> Mapping[str, object]:
    """Require a mapping and preserve its message path in validation errors."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    """Require a non-string sequence at the given message path."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _text_blocks(value: object, path: str, *, allow_none: bool = False) -> list[dict[str, object]]:
    """Normalize LiteLLM string or text-block content for Switchyard."""
    if value is None and allow_none:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]

    blocks: list[dict[str, object]] = []
    for index, raw_block in enumerate(_sequence(value, path)):
        block_path = f"{path}[{index}]"
        block = _mapping(raw_block, block_path)
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ValueError(f"{block_path} content type is not supported")
        blocks.append({"type": "text", "text": cast(str, block["text"])})
    return blocks


def _tool_calls(value: object, path: str) -> list[dict[str, object]]:
    """Normalize OpenAI function calls and parse their JSON arguments."""
    calls: list[dict[str, object]] = []
    for index, raw_call in enumerate(_sequence(value, path)):
        call_path = f"{path}[{index}]"
        call = _mapping(raw_call, call_path)
        call_id = call.get("id")
        if call.get("type") != "function":
            raise ValueError(f"{call_path}.type is not supported")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"{call_path}.id must be a non-empty string")
        function = _mapping(call.get("function"), f"{call_path}.function")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{call_path}.function.name must be a non-empty string")
        if not isinstance(arguments, str):
            raise ValueError(f"{call_path}.function.arguments must be a JSON string")
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{call_path}.function.arguments must be a valid JSON object"
            ) from error
        if not isinstance(parsed_arguments, Mapping):
            raise ValueError(f"{call_path}.function.arguments must be a valid JSON object")
        calls.append(
            {
                "type": "tool_call",
                "id": call_id,
                "name": name,
                "arguments": dict(parsed_arguments),
            }
        )
    return calls


def _messages(structured_messages: Sequence[object]) -> list[dict[str, object]]:
    """Normalize LiteLLM structured chat history for Switchyard algorithms."""
    if not structured_messages:
        raise ValueError("structured_messages must not be empty")

    converted: list[dict[str, object]] = []
    for index, raw_message in enumerate(structured_messages):
        path = f"structured_messages[{index}]"
        message = _mapping(raw_message, path)
        role = message.get("role")
        if role in {"system", "developer", "user"}:
            converted.append(
                {
                    "role": role,
                    "content": _text_blocks(message.get("content"), f"{path}.content"),
                }
            )
        elif role == "assistant":
            content: list[dict[str, object]] = list(
                _text_blocks(message.get("content"), f"{path}.content", allow_none=True)
            )
            raw_calls = message.get("tool_calls")
            if raw_calls is not None:
                content.extend(_tool_calls(raw_calls, f"{path}.tool_calls"))
            if not content:
                raise ValueError(f"{path} must contain text or tool calls")
            converted.append({"role": "assistant", "content": content})
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"{path}.tool_call_id must be a non-empty string")
            converted.append(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": call_id,
                            "content": _text_blocks(message.get("content"), f"{path}.content"),
                            "is_error": None,
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"{path}.role is not supported")
    return converted


def _request(structured_messages: Sequence[object]) -> dict[str, object]:
    """Build the canonical normalized request used to detect unsupported rewrites."""
    return {
        "model": "auto",
        "instructions": [],
        "messages": _messages(structured_messages),
        "tools": [],
        "tool_choice": None,
        "sampling": {"temperature": None, "top_p": None, "top_k": None},
        "output": {"max_output_tokens": None, "response_format": None},
        "reasoning": {"effort": None, "raw": None},
        "stream": False,
        "extensions": {"fields": {}},
        "preservation": {"requests": {}, "responses": {}},
    }


class SwitchyardRoutingPlugin(LiteLLMRequestRewriter):
    """Narrow LiteLLM candidates with a Switchyard algorithm.

    An algorithm may ask for intermediate model calls (a capability or
    escalation judge) only when a `judge` client is configured; the plugin
    serves each call through it and returns the normalized response to
    Switchyard. Without a judge, an intermediate call is rejected. An algorithm
    must never produce the final response itself. Supported request rewrites
    are carried to the selected deployment by the object's LiteLLM callback
    role.

    While `is_judge_call()` is true, the request being routed is a judge call
    this integration issued through LiteLLM; the plugin passes it through
    unchanged so routing never re-enters itself.

    `models` maps a category name (`any`, `capable`, `efficient`, `judge`, or a
    deployment-defined group) to its LiteLLM model names, ordered best-first.
    The algorithm picks the first name it can use from a category, so the order
    within each list is the preference order. Every name must be a LiteLLM
    deployment that can appear in the routing candidate pool: the plugin keeps
    only the selected candidate, and a selection outside the pool LiteLLM
    offered for this request raises `ValueError`.

    When `models` is omitted, each of the four categories is filled with the
    request's full candidate pool, so any selection is valid by construction.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        models: Mapping[str, Sequence[str]] | None = None,
        *,
        judge: JudgeClient | None = None,
    ) -> None:
        """Copy the category map so later changes by the caller do not affect routing."""
        super().__init__()
        self._algorithm = algorithm
        self._judge = judge
        self._models = (
            {category: list(names) for category, names in models.items()}
            if models is not None
            else None
        )

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Run Switchyard and retain only its selected LiteLLM candidate."""
        if is_judge_call():
            return context
        candidates = list(context.candidate_models)
        request = _request(context.structured_messages)

        models = (
            self._models
            if self._models is not None
            else {
                "any": candidates,
                "judge": candidates,
                "capable": candidates,
                "efficient": candidates,
            }
        )
        async for step in self._algorithm.run_stream(request, models):
            match step:
                case Step.CallModel(call):
                    if self._judge is None:
                        raise ValueError(
                            "Switchyard algorithm requested an intermediate model call; "
                            "configure a judge client to serve it through LiteLLM"
                        )
                    # A failing judge is reported to Switchyard through the same reply
                    # path, so the algorithm's own fail-open policy decides the tier.
                    await call.respond(self._judge(call.request, call.models[0]))
                case Step.Done(outcome):
                    if outcome.response is not None:
                        raise ValueError(
                            "Switchyard algorithm produced a response while routing, "
                            "which a LiteLLM routing plugin cannot return"
                        )
                    selected = outcome.selected_model_ids[0]
                    if selected not in candidates:
                        raise ValueError(
                            f"Switchyard selected {selected!r}, which is not in LiteLLM's "
                            "candidate pool"
                        )
                    request_patch = build_request_patch(
                        request,
                        outcome.request,
                        selected_model_id=selected,
                    )
                    context.candidate_models = [
                        candidate for candidate in candidates if candidate == selected
                    ]
                    # Every algorithm reports how it decided: the classifier's score and
                    # threshold, the stage router's fall-open or override. Recorded as given,
                    # so a gateway can show the reason beside the selection.
                    context.signals["switchyard"] = {
                        "selected_model_id": selected,
                        "fallback_models": outcome.selected_model_ids[1:],
                        "algorithm": outcome.metadata.algorithm,
                        "evidence": dict(outcome.metadata.evidence or {}),
                    }
                    if request_patch["set"] or request_patch.get("remove"):
                        context.signals["switchyard"]["request_patch"] = request_patch
                    return context

        raise ValueError("Switchyard algorithm stream ended without a routing outcome")


__all__ = ["SwitchyardRoutingPlugin"]
