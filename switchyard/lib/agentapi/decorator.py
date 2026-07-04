# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The @route decorator: drives an async model-call function through a router.

Per call it mints one optimizer and runs the round-agnostic optimizer loop,
applying each emitted ChatRequest (model + prompt) to the wrapped function and
feeding the response back until the optimizer returns control. A single-round
router (rand) and a multi-round router (llm_class) drive through identical code.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData
from switchyard.lib.agentapi.optimizer import (
    AgentApiOptAlgorithm,
    ModelInference,
    RequestInput,
    ResponseInput,
    Return,
)


def _default_get_prompt(kwargs: Mapping[str, Any]) -> str:
    """litellm-shaped: last user message content, else a `prompt` kwarg."""
    messages = kwargs.get("messages")
    if messages is not None:
        return str(messages[-1]["content"])
    return str(kwargs["prompt"])


def _default_apply_request(kwargs: dict[str, Any], req: ChatRequest) -> None:
    """Apply an emitted request: set model, and set the prompt for this round.

    With `messages`, replaces the last user message's content (preserving system
    messages / history); otherwise sets a `prompt` kwarg.
    """
    kwargs["model"] = req.model
    messages = kwargs.get("messages")
    if messages is not None:
        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].get("role") == "user":
                new_messages[i] = {**new_messages[i], "content": req.prompt}
                break
        else:
            new_messages.append({"role": "user", "content": req.prompt})
        kwargs["messages"] = new_messages
    else:
        kwargs["prompt"] = req.prompt


def _default_get_completion(resp: Any) -> str:
    """litellm-shaped: resp.choices[0].message.content, else a str response."""
    if isinstance(resp, str):
        return resp
    return str(resp.choices[0].message.content)


def route(
    algorithm: AgentApiOptAlgorithm,
    *,
    get_prompt: Callable[[Mapping[str, Any]], str] = _default_get_prompt,
    apply_request: Callable[[dict[str, Any], ChatRequest], None] = _default_apply_request,
    get_completion: Callable[[Any], str] = _default_get_completion,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an async model-call function so `algorithm` routes each call.

    The wrapped function must be a coroutine function with a keyword-compatible
    signature. Returns the wrapped function's return value from the last model
    call the optimizer requested (for a classifier, the routed call's response).
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@route requires an async function, got {fn!r}")
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call_kwargs = bound.arguments

            optimizer = algorithm.optimizer()
            prompt = get_prompt(call_kwargs)
            # Lenient: both routers overwrite the inbound model, and custom
            # adapters may use a non-`model` parameter name (e.g. `engine`).
            model = call_kwargs.get("model", "")
            await optimizer.feed(
                RequestInput(ChatRequest(prompt=prompt, model=model)), EnrichmentData()
            )

            last_resp: Any = None
            while True:
                decision = await optimizer.optimize()
                if isinstance(decision, Return):
                    break
                assert isinstance(decision, ModelInference)
                for req in decision.response.requests:
                    apply_request(call_kwargs, req)
                    last_resp = await fn(**call_kwargs)
                    completion = get_completion(last_resp)
                    await optimizer.feed(
                        ResponseInput(ChatResponse(completion)), EnrichmentData()
                    )
            return last_resp

        return wrapper

    return decorate
