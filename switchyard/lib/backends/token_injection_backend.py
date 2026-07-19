# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful vLLM backend wrapper that preserves token continuity by prompt injection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from switchyard.lib.backends.vllm_parsers import (
    VllmParserError,
    parse_generation,
    parse_guided_tool_array,
    tool_choice_json_schema,
)
from switchyard.lib.processors.token_capture_request_processor import CTX_TOKEN_CAPTURE_SESSION
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]

#: finish_reasons where the model emitted its natural end-of-turn token itself,
#: making the generation's final token usable as the EOT split marker.
_NATURAL_STOP_REASONS = frozenset({"stop", "tool_calls", "stop_sequence"})

#: Chat request params forwarded verbatim onto the completions request.
_FORWARDED_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
)


@dataclass
class _SessionState:
    """Canonical token history for one capture session.

    ``prefix`` is the no-generation-prompt template tokenization of the last
    call's messages — the stable split offset for the next call's env-suffix
    extraction (the generation-prompt suffix differs between ``/tokenize`` and
    chat completions when a reasoning parser is active, so full-prompt offsets
    are not comparable).
    """

    accumulated: list[int]
    prefix: list[int]
    last_generation: list[int]
    eot_token_id: int | None
    fallen_back: bool = False


class TokenInjectionBackend(LLMBackend):
    """Guarantee per-session token continuity by injecting raw prompt token IDs.

    Wraps the normal chat backend. The first call of a session delegates to it
    and seeds the session's canonical token history from the captured engine
    fields; every later call bypasses chat-template re-rendering — which
    strips historical reasoning for reasoning models — by sending the
    accumulated history plus the newly tokenized environment suffix directly
    to vLLM's ``/v1/completions`` as token IDs. The raw generation is parsed
    with vLLM's own parsers and returned as a chat-shaped response carrying
    the same engine token fields the capture processor already reads.

    Any failure (prefix mismatch from history rewrites, missing token fields,
    parser errors, transport errors) falls back to the wrapped backend and
    pins the session to the fallback path — behavior then equals capture-only
    mode and Gym's validation masks the sample. Session state is in-process
    only: one proxy instance must own all calls of a session.
    """

    def __init__(
        self,
        inner: LLMBackend,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        timeout_secs: float = 600.0,
    ) -> None:
        self._inner = inner
        # /tokenize lives at the server root, not under /v1.
        self._v1_url = base_url.rstrip("/")
        self._root_url = self._v1_url.removesuffix("/v1")
        self._model = model
        self._api_key = api_key
        self._tool_parser = tool_parser
        self._reasoning_parser = reasoning_parser
        self._timeout = timeout_secs
        self._sessions: dict[str, _SessionState] = {}

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        """Serve one model call, injecting the canonical history when possible."""
        session_id = ctx.metadata.get(CTX_TOKEN_CAPTURE_SESSION)
        if not isinstance(session_id, str) or not session_id:
            return await self._inner.call(ctx, request)

        state = self._sessions.get(session_id)
        if state is not None and state.fallen_back:
            return await self._inner.call(ctx, request)

        if state is None:
            return await self._first_call(ctx, request, session_id)

        body = dict(request.body)
        if body.get("n", 1) not in (None, 1):
            self._fall_back(session_id, "n>1 is unsupported for injection")
            return await self._inner.call(ctx, request)
        try:
            return await self._injected_call(session_id, state, body)
        except (httpx.HTTPError, VllmParserError, KeyError, ValueError) as exc:
            self._fall_back(session_id, f"injection failed: {exc}")
            return await self._inner.call(ctx, request)

    async def _first_call(
        self, ctx: ProxyContext, request: ChatRequest, session_id: str
    ) -> ChatResponse:
        """Delegate the session's first call and seed its token history."""
        response = await self._inner.call(ctx, request)
        try:
            body = dict(response.body)
        except (TypeError, ValueError):
            self._fall_back(session_id, "first response body is not readable")
            return response
        harvested = _harvest_token_fields(body)
        if harvested is None:
            self._fall_back(session_id, "first response carries no engine token fields")
            return response
        prompt_ids, generation_ids, finish_reason = harvested
        try:
            prefix = await self._tokenize(dict(request.body), add_generation_prompt=False)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self._fall_back(session_id, f"prefix tokenization failed: {exc}")
            return response
        if prompt_ids[: len(prefix)] != prefix:
            self._fall_back(session_id, "no-generation-prompt prefix does not match the prompt")
            return response
        self._sessions[session_id] = _SessionState(
            accumulated=prompt_ids + generation_ids,
            prefix=prefix,
            last_generation=generation_ids,
            eot_token_id=(
                generation_ids[-1] if finish_reason in _NATURAL_STOP_REASONS else None
            ),
        )
        return response

    async def _injected_call(
        self, session_id: str, state: _SessionState, body: JsonObject
    ) -> ChatResponse:
        """Generate through ``/v1/completions`` with the canonical token history."""
        if state.eot_token_id is None:
            raise ValueError("no end-of-turn token observed (prior turn did not stop naturally)")
        rendered = await self._tokenize(body, add_generation_prompt=True)
        if rendered[: len(state.prefix)] != state.prefix:
            raise ValueError("templated prompt no longer extends the session prefix")
        env_tokens = _slice_env_suffix(
            rendered[len(state.prefix):], state.last_generation, state.eot_token_id
        )
        injected_prompt = state.accumulated + env_tokens

        completion = await self._completions(body, injected_prompt)
        choice = completion["choices"][0]
        prompt_ids = choice.get("prompt_token_ids") or completion.get("prompt_token_ids")
        generation_ids = choice.get("token_ids")
        if prompt_ids != injected_prompt:
            raise ValueError("completions endpoint did not echo the injected prompt")
        if not isinstance(generation_ids, list) or not generation_ids:
            raise ValueError("completions response carries no generation token ids")

        chat_body = self._synthesize_chat_body(body, completion, injected_prompt)

        # State advances only after a fully successful call, so a failed call
        # falls back without corrupting the history.
        state.accumulated = injected_prompt + generation_ids
        state.prefix = await self._tokenize(body, add_generation_prompt=False)
        state.last_generation = generation_ids
        finish_reason = choice.get("finish_reason")
        if finish_reason in _NATURAL_STOP_REASONS:
            state.eot_token_id = generation_ids[-1]
        return ChatResponse.openai_completion(chat_body)

    def _synthesize_chat_body(
        self, request_body: JsonObject, completion: JsonObject, injected_prompt: list[int]
    ) -> JsonObject:
        """Chat-completions-shaped body carrying the engine token fields.

        The shape matches what vLLM's chat endpoint returns with the capture
        params on, so the capture response processor and the stream
        synthesizer consume it unchanged.
        """
        choice = completion["choices"][0]
        text = choice.get("text") or ""
        tools = request_body.get("tools") if isinstance(request_body.get("tools"), list) else None
        tool_choice = request_body.get("tool_choice")

        if tool_choice_json_schema(tools or [], tool_choice) is not None:
            # Guided decoding: the generation is the schema-constrained JSON
            # array itself; the tool parser never applies.
            tool_calls = parse_guided_tool_array(text)
            reasoning_content: str | None = None
            content: str | None = None
        else:
            parsed = parse_generation(
                text,
                model=self._model,
                tools=tools,
                tool_parser=self._tool_parser,
                reasoning_parser=self._reasoning_parser,
            )
            tool_calls = parsed.tool_calls
            reasoning_content = parsed.reasoning_content
            content = parsed.content

        message: JsonObject = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = choice.get("finish_reason") or "stop"
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        generation_ids = choice.get("token_ids") or []
        return {
            "id": completion.get("id") or "chatcmpl-token-injection",
            "object": "chat.completion",
            "created": completion.get("created") or int(time.time()),
            "model": completion.get("model") or self._model,
            "prompt_token_ids": injected_prompt,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "token_ids": generation_ids,
                    "logprobs": _chat_shaped_logprobs(choice.get("logprobs"), generation_ids),
                }
            ],
            "usage": completion.get("usage"),
        }

    async def _tokenize(self, body: JsonObject, *, add_generation_prompt: bool) -> list[int]:
        """Template-render and tokenize the request's messages without generating."""
        payload: JsonObject = {
            "model": self._model,
            "messages": body.get("messages") or [],
            "add_generation_prompt": add_generation_prompt,
        }
        if isinstance(body.get("tools"), list):
            payload["tools"] = body["tools"]
        if isinstance(body.get("chat_template_kwargs"), dict):
            payload["chat_template_kwargs"] = body["chat_template_kwargs"]
        data = await self._post(f"{self._root_url}/tokenize", payload)
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise ValueError("/tokenize response has no integer token list")
        return tokens

    async def _completions(self, body: JsonObject, prompt: list[int]) -> JsonObject:
        """One ``/v1/completions`` generation with the injected token-ID prompt."""
        payload: JsonObject = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "return_token_ids": True,
            "logprobs": 0,
        }
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        for key in _FORWARDED_SAMPLING_KEYS:
            if body.get(key) is not None:
                payload[key] = body[key]
        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        schema = tool_choice_json_schema(tools or [], body.get("tool_choice"))
        if schema is not None:
            payload["structured_outputs"] = {"json": schema}
        return await self._post(f"{self._v1_url}/completions", payload)

    async def _post(self, url: str, payload: JsonObject) -> JsonObject:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"{url} returned a non-object JSON body")
        return data

    def _fall_back(self, session_id: str, reason: str) -> None:
        """Pin the session to the wrapped chat path for all remaining calls."""
        logger.warning(
            "token injection: session %s falls back to the chat path: %s", session_id, reason
        )
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionState(
                accumulated=[], prefix=[], last_generation=[], eot_token_id=None
            )
            self._sessions[session_id] = state
        state.fallen_back = True


def _harvest_token_fields(body: JsonObject) -> tuple[list[int], list[int], Any] | None:
    """``(prompt_token_ids, generation_token_ids, finish_reason)`` from a raw chat body."""
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    prompt_ids = body.get("prompt_token_ids")
    generation_ids = choice.get("token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(generation_ids, list):
        return None
    if not _is_token_list(prompt_ids) or not _is_token_list(generation_ids):
        return None
    return list(prompt_ids), list(generation_ids), choice.get("finish_reason")


def _is_token_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _slice_env_suffix(
    canonical_tail: list[int], last_generation: list[int], eot_token_id: int
) -> list[int]:
    """Environment tokens added since the last call (tool results, user turns, glue).

    ``canonical_tail`` re-renders the previous assistant turn (with reasoning
    stripped by the template) followed by the new environment messages; the
    first end-of-turn token marks where the re-rendered turn ends. When the
    raw generation already ended with the end-of-turn token it is skipped in
    the tail to avoid duplication; otherwise it is kept so the stream still
    closes the assistant turn.
    """
    try:
        index = canonical_tail.index(eot_token_id)
    except ValueError:
        raise ValueError("end-of-turn token not found in the re-rendered prompt tail") from None
    if last_generation and last_generation[-1] == eot_token_id:
        return canonical_tail[index + 1:]
    return canonical_tail[index:]


def _chat_shaped_logprobs(logprobs: object, generation_ids: list[int]) -> JsonObject | None:
    """Convert completions-endpoint logprobs to the chat shape capture reads.

    Completions returns ``{token_logprobs: [...], tokens: [...]}``; the capture
    processor reads ``{content: [{token_id, logprob}, ...]}`` aligned with the
    generation token ids.
    """
    if not isinstance(logprobs, dict):
        return None
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list) or len(token_logprobs) != len(generation_ids):
        return None
    return {
        "content": [
            {"token_id": token_id, "logprob": logprob}
            for token_id, logprob in zip(generation_ids, token_logprobs, strict=True)
        ]
    }
