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

#: Clamp headroom: the chat endpoint can render the prompt a few tokens longer
#: than /tokenize does (measured on vLLM 0.24.0 + Qwen3: the chat path's
#: disabled-thinking render pre-fills an empty think block, exactly 4 tokens).
#: The pre-fill size is a template property, so keep headroom beyond the
#: observed value rather than pinning to it.
_CONTEXT_MARGIN_TOKENS = 8

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
class _Chain:
    """Canonical token history of one conversation within a capture session.

    Harnesses interleave several conversations under one session id (e.g.
    opencode's title-generator side-call rides the main agent's session), so a
    session holds a list of chains and each call is routed to the chain it
    extends. ``prefix`` is the no-generation-prompt template tokenization of
    the chain's last call — the stable split offset for env-suffix extraction
    (the generation-prompt suffix differs between ``/tokenize`` and chat
    completions when a reasoning parser is active, so full-prompt offsets are
    not comparable).
    """

    accumulated: list[int]
    prefix: list[int]
    #: The chain's last raw generation; its final token on a natural stop is
    #: the end-of-turn marker used to slice the env suffix. Verified: vLLM
    #: includes the stop token in token_ids on both 0.11 and 0.24.
    last_generation: list[int]
    #: End-of-turn token id, observed from a naturally stopped generation.
    #: None (e.g. after a length-stop) means the chain cannot be extended and
    #: is reseeded by the next matching call instead.
    eot_token_id: int | None


class TokenInjectionBackend(LLMBackend):
    """Guarantee per-session token continuity by injecting raw prompt token IDs.

    Wraps the normal chat backend. Each call is matched to the session chain
    whose history it extends; matched calls bypass chat-template re-rendering
    — which strips historical reasoning for reasoning models — by sending the
    chain's accumulated history plus the newly tokenized environment suffix
    directly to vLLM's ``/v1/completions`` as token IDs. The raw generation is
    parsed with vLLM's own parsers and returned as a chat-shaped response
    carrying the same engine token fields the capture processor already reads.

    A call that matches no chain is served via the wrapped backend and seeds a
    new chain from its response, so interleaved side-calls and arrival order
    never disable injection for the real conversation. Any failure (missing
    token fields, parser errors, transport errors) also falls back to the
    wrapped backend — behavior then equals capture-only mode and Gym's
    validation masks the affected sample. Chain state is in-process only: one
    proxy instance must own all calls of a session.
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
        self._sessions: dict[str, list[_Chain]] = {}
        # Model context length, learned from /tokenize responses; used to clamp
        # the completion budget the way older vLLM chat endpoints did implicitly
        # (newer versions reject over-budget requests outright, on both paths).
        self._max_model_len: int | None = None

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        """Serve one model call, injecting the matching chain's history when possible."""
        session_id = ctx.metadata.get(CTX_TOKEN_CAPTURE_SESSION)
        if not isinstance(session_id, str) or not session_id:
            return await self._inner.call(ctx, request)
        body = dict(request.body)
        if body.get("n", 1) not in (None, 1):
            return await self._inner.call(ctx, request)

        chains = self._sessions.setdefault(session_id, [])
        rendered: list[int] | None = None
        try:
            chain = None
            if chains:
                rendered = await self._tokenize(body, add_generation_prompt=True)
                chain = _longest_matching_chain(chains, rendered)
            if rendered is not None and chain is not None and chain.eot_token_id is not None:
                return await self._injected_call(chain, body, rendered)

            # No chain extends this call (first call, side-call, rewritten
            # history) or the matched chain has no end-of-turn marker: serve
            # via the chat path and (re)seed a chain from the response.
            prefix = await self._tokenize(body, add_generation_prompt=False)
            response = await self._serve_inner(ctx, request, body, prompt_len=len(prefix))
            self._seed_chain(chains, chain, response, prefix)
            return response
        except (httpx.HTTPError, VllmParserError, KeyError, ValueError) as exc:
            logger.warning(
                "token injection: session %s: serving via the chat path: %s", session_id, exc
            )
            if rendered is not None:
                # The budget clamp must apply on this path too, or the call
                # trades an injection error for a context-window rejection.
                return await self._serve_inner(ctx, request, body, prompt_len=len(rendered))
            return await self._inner.call(ctx, request)

    async def _serve_inner(
        self, ctx: ProxyContext, request: ChatRequest, body: JsonObject, prompt_len: int
    ) -> ChatResponse:
        """Delegate to the chat backend with the completion budget clamped.

        The clamp applies on this path too: newer vLLM rejects
        ``prompt + max_tokens > max_model_len`` on chat completions as well.
        """
        max_tokens = self._resolve_max_tokens(body, prompt_len)
        if max_tokens is not None and max_tokens != body.get("max_tokens"):
            body = dict(body)
            body["max_tokens"] = max_tokens
            body.pop("max_completion_tokens", None)
            request.replace_body(body)
        return await self._inner.call(ctx, request)

    def _seed_chain(
        self,
        chains: list[_Chain],
        chain: _Chain | None,
        response: ChatResponse,
        prefix: list[int],
    ) -> None:
        """Start (or refresh) a chain from a chat-path response's token fields.

        Responses without engine token fields seed nothing — the call stays
        capture-only and a later call may seed the chain instead.
        """
        try:
            body = dict(response.body)
        except (TypeError, ValueError):
            return
        harvested = _harvest_token_fields(body)
        if harvested is None:
            return
        prompt_ids, generation_ids, finish_reason = harvested
        if prompt_ids[: len(prefix)] != prefix:
            return
        seeded = _Chain(
            accumulated=prompt_ids + generation_ids,
            prefix=prefix,
            last_generation=generation_ids,
            eot_token_id=(
                generation_ids[-1] if finish_reason in _NATURAL_STOP_REASONS else None
            ),
        )
        if chain is not None:
            chains[chains.index(chain)] = seeded
        else:
            chains.append(seeded)

    async def _injected_call(
        self, chain: _Chain, body: JsonObject, rendered: list[int]
    ) -> ChatResponse:
        """Generate through ``/v1/completions`` with the chain's token history."""
        if chain.eot_token_id is None:  # caller guarantees; narrows the type
            raise ValueError("chain has no end-of-turn marker")
        env_tokens = _slice_env_suffix(
            rendered[len(chain.prefix):], chain.last_generation, chain.eot_token_id
        )
        injected_prompt = chain.accumulated + env_tokens

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
        chain.accumulated = injected_prompt + generation_ids
        chain.prefix = await self._tokenize(body, add_generation_prompt=False)
        chain.last_generation = generation_ids
        if choice.get("finish_reason") in _NATURAL_STOP_REASONS:
            chain.eot_token_id = generation_ids[-1]
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
            # Both spellings: vLLM chat responses renamed reasoning_content ->
            # reasoning around 0.24; harnesses may read either.
            message["reasoning_content"] = reasoning_content
            message["reasoning"] = reasoning_content
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

    def _resolve_max_tokens(self, body: JsonObject, prompt_len: int) -> int | None:
        """The effective completion budget for a call with a *prompt_len*-token prompt.

        Honors the smaller of ``max_tokens`` / ``max_completion_tokens`` when
        both are present, clamped to the remaining model context when the
        context length is known (learned from ``/tokenize`` responses). When
        no budget fits at all, the requested value is returned unchanged — a
        genuinely exhausted context should surface as the engine's own error.
        """
        requested = [
            value
            for value in (body.get("max_tokens"), body.get("max_completion_tokens"))
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        max_tokens = min(requested) if requested else None
        if max_tokens is None or self._max_model_len is None:
            return max_tokens
        remaining = self._max_model_len - prompt_len - _CONTEXT_MARGIN_TOKENS
        if remaining <= 0:
            return max_tokens
        return min(max_tokens, remaining)

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
        max_model_len = data.get("max_model_len")
        if isinstance(max_model_len, int) and not isinstance(max_model_len, bool):
            self._max_model_len = max_model_len
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
        max_tokens = self._resolve_max_tokens(body, len(prompt))
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


def _longest_matching_chain(chains: list[_Chain], rendered: list[int]) -> _Chain | None:
    """The chain whose prefix the rendered prompt extends; longest match wins."""
    best: _Chain | None = None
    for chain in chains:
        n = len(chain.prefix)
        if (best is None or n > len(best.prefix)) and rendered[:n] == chain.prefix:
            best = chain
    return best


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
    """Environment tokens added since the chain's last call (tool results, user turns, glue).

    ``canonical_tail`` re-renders the previous assistant turn (reasoning
    stripped by the template) followed by the new environment messages; the
    first end-of-turn token marks where the re-rendered turn ends. This never
    renders a conversation ENDING with an assistant message, so thinking
    templates' special-casing of a final assistant turn cannot affect it.
    When the raw generation already ended with the end-of-turn token it is
    skipped in the tail to avoid duplication; otherwise it is kept so the
    stream still closes the assistant turn.
    """
    try:
        index = canonical_tail.index(eot_token_id)
    except ValueError:
        raise ValueError(
            "end-of-turn token not found in the re-rendered prompt tail "
            f"(eot={eot_token_id}, tail starts {canonical_tail[:8]})"
        ) from None
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
