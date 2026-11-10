# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Switchyard's routing-time judge calls through LiteLLM.

A classifier-backed algorithm asks for one intermediate model call: the judge
that estimates whether the efficient tier can complete the task. A LiteLLM
routing plugin runs before deployment selection, so it cannot hand that call
back to the request it is routing. It can, however, make the call itself. The
client here does that through LiteLLM, against a deployment the deployer names
for the judge role, so the judge is subject to the same credentials, retries
and spend tracking as every other call the gateway makes.

The judge request carries :data:`JUDGE_MARKER` in its metadata. LiteLLM hands
the request's metadata to the routing context, so a plugin that sees the marker
knows it is looking at a judge call it issued and must pass it through without
classifying it again.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, cast

from switchyard.libsy import LlmResponse

#: Metadata key set on every judge request issued by :class:`LiteLLMJudge`.
JUDGE_MARKER = "switchyard_judge"

_MAX_TOKENS_FIELDS = frozenset({"max_completion_tokens", "max_tokens"})
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


class JudgeClient(Protocol):
    """Serve one normalized Switchyard model call and return its normalized response."""

    def __call__(self, request: Mapping[str, object], model: str) -> Awaitable[LlmResponse.Agg]: ...


def _text(value: object, path: str) -> str:
    """Join the text blocks of one normalized message; reject anything else."""
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{path} content must be a sequence of text blocks")
    parts: list[str] = []
    for index, raw_block in enumerate(value):
        if not isinstance(raw_block, Mapping) or raw_block.get("type") != "text":
            raise ValueError(
                f"{path}[{index}] is not a text block; the judge call carries text only"
            )
        parts.append(cast(str, raw_block["text"]))
    return "\n".join(parts)


def judge_messages(request: Mapping[str, object]) -> list[dict[str, str]]:
    """Flatten a normalized judge request into OpenAI chat messages."""
    messages: list[dict[str, str]] = []
    for source in ("instructions", "messages"):
        for index, raw_message in enumerate(cast(Sequence[object], request.get(source) or [])):
            path = f"{source}[{index}]"
            if not isinstance(raw_message, Mapping):
                raise ValueError(f"{path} must be a mapping")
            role = raw_message.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(f"{path}.role {role!r} is not supported in a judge call")
            messages.append(
                {"role": cast(str, role), "content": _text(raw_message.get("content"), path)}
            )
    if not messages:
        raise ValueError("judge request carries no messages")
    return messages


def judge_completion_kwargs(
    request: Mapping[str, object],
    model: str,
    *,
    max_tokens_field: str = "max_completion_tokens",
) -> dict[str, Any]:
    """Translate a normalized judge request into ``litellm.acompletion`` keyword arguments.

    ``max_tokens_field`` names the output-limit parameter the judge deployment
    accepts: OpenAI's reasoning models reject ``max_tokens`` and take
    ``max_completion_tokens``; some OpenAI-compatible servers only know
    ``max_tokens``.
    """
    if max_tokens_field not in _MAX_TOKENS_FIELDS:
        raise ValueError("max_tokens_field must be 'max_completion_tokens' or 'max_tokens'")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": judge_messages(request),
        "metadata": {JUDGE_MARKER: True},
    }
    output = cast(Mapping[str, object], request.get("output") or {})
    limit = output.get("max_output_tokens")
    if limit is not None:
        kwargs[max_tokens_field] = int(cast(int, limit))
    response_format = output.get("response_format")
    if response_format is not None:
        kwargs["response_format"] = json.loads(json.dumps(response_format))
    sampling = cast(Mapping[str, object], request.get("sampling") or {})
    for field in ("temperature", "top_p"):
        value = sampling.get(field)
        if value is not None:
            kwargs[field] = value
    return kwargs


def judge_response(response: Any, model: str) -> LlmResponse.Agg:
    """Wrap a LiteLLM chat completion as the normalized response Switchyard expects.

    An empty or missing ``content`` is passed through as empty text: Switchyard
    treats an unparseable verdict conservatively, and that is the right outcome
    for a judge that answered without a verdict.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("judge response carries no choices")
    choice = choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    text = content if isinstance(content, str) else ""
    stop_reason = _STOP_REASONS.get(getattr(choice, "finish_reason", None) or "stop", "end_turn")
    return LlmResponse.Agg(
        {
            "model": getattr(response, "model", None) or model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": stop_reason,
                }
            ],
        }
    )


def _proxy_router() -> Any | None:
    """The LiteLLM proxy's router when this process is the proxy, else ``None``."""
    try:
        from litellm.proxy import proxy_server
    except Exception:  # pragma: no cover - the proxy package is optional at runtime
        return None
    return getattr(proxy_server, "llm_router", None)


class LiteLLMJudge:
    """Serve judge calls through LiteLLM against one named deployment.

    Args:
        model: The LiteLLM model name of the judge deployment. Give it its own
            ``model_name`` in the deployer's ``model_list``; it must not be one
            of the routing candidates.
        router: The ``litellm.Router`` to call through. SDK users who build the
            plugin before the Router can assign :attr:`router` afterwards. When
            unset, the proxy's router is used if this process is the LiteLLM
            proxy; otherwise the call goes through ``litellm.acompletion`` with
            ``completion_params`` (for example ``api_base`` and ``api_key``).
        max_tokens_field: See :func:`judge_completion_kwargs`.
        completion_params: Extra keyword arguments for every judge call, such as
            ``timeout`` or provider credentials in the direct mode.
    """

    def __init__(
        self,
        model: str,
        *,
        router: Any | None = None,
        max_tokens_field: str = "max_completion_tokens",
        **completion_params: Any,
    ) -> None:
        if not model:
            raise ValueError("LiteLLMJudge requires a judge model name")
        if max_tokens_field not in _MAX_TOKENS_FIELDS:
            raise ValueError("max_tokens_field must be 'max_completion_tokens' or 'max_tokens'")
        self.model = model
        #: The Router judge calls go through; assignable after construction.
        self.router: Any | None = router
        self._max_tokens_field = max_tokens_field
        self._completion_params = dict(completion_params)

    async def __call__(self, request: Mapping[str, object], model: str) -> LlmResponse.Agg:
        kwargs = judge_completion_kwargs(request, model, max_tokens_field=self._max_tokens_field)
        kwargs.update(self._completion_params)
        router = self.router if self.router is not None else _proxy_router()
        if router is not None:
            response = await router.acompletion(**kwargs)
        else:
            import litellm

            response = await litellm.acompletion(**kwargs)
        return judge_response(response, model)


__all__ = [
    "JUDGE_MARKER",
    "JudgeClient",
    "LiteLLMJudge",
    "judge_completion_kwargs",
    "judge_messages",
    "judge_response",
]
