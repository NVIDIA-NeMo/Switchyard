# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LLMBackend`` that re-creates Anthropic's advisor tool proxy-side, on any Chat wire.

Anthropic's native ``advisor_20260301`` server tool runs the advisor
sub-inference server-side, so it is unavailable on gateways that only proxy
model traffic (e.g. NVIDIA Inference Hub) — and it only exists for Anthropic
models at all. This backend reproduces the *executor-triggered* behavior in
the proxy, for **any** Chat-shaped executor:

1. Offer the executor a real, parameterless ``advisor`` tool (empty schema —
   like the native tool, the executor signals timing and the proxy supplies
   context).
2. Call the executor. If its turn contains ``advisor`` tool calls, intercept
   them **before they reach the client**, consult the advisor model on the
   transcript (including the text the executor produced so far in the turn,
   mirroring the native tool's context), append the advice as the tool
   result, and call the executor again.
3. Loop until the executor returns an advisor-free turn (a real tool call for
   the client, or a final answer), then return that turn.

Once ``max_uses`` consultations have happened in a request, further advisor
calls receive a ``max_uses exceeded`` tool result without a consult — the
executor sees the error and continues, mirroring the native tool's
``max_uses_exceeded`` error result.

The executor's wire format is selected by ``config.executor.format`` and
handled by a private dialect object:

* ``anthropic`` — the executor call is delegated verbatim to an
  :class:`AnthropicNativeBackend` (``/v1/messages``), so the client's
  ``cache_control`` breakpoints reach the upstream unchanged and prompt
  caching is honored. Thinking blocks round-trip verbatim for upstreams that
  verify signatures.
* ``openai`` — the executor call is delegated verbatim to an
  :class:`OpenAiNativeBackend` (``/chat/completions``), covering OSS models
  (Qwen, DeepSeek on vLLM/NIM) and OpenAI. Vendor-specific assistant fields
  (e.g. ``reasoning_content``) are dropped from the rebuilt advisor turns so
  strict endpoints accept the replayed history.

The advisor tier is likewise format-dispatched (see
``advisor_loop_backend._build_advisor_caller``); tiers mix freely.
``responses`` targets are rejected — the loop is Chat-shaped.

Steering is injected cache-stably: the executor steering is prepended to the
system prompt and the advisor length line is appended to the **first** user
message (both constant across a session's turns; the native doc suggests the
latest user message, but re-injecting there would shift the cached prefix on
every turn because the client never sees the injection). The same placement
holds on the OpenAI wire, where vLLM/OpenAI automatic prefix caching also
wants a stable prefix.

A turn that mixes advisor and client tool calls is regenerated: the appended
assistant turn keeps only the advisor calls (plus text/thinking), and the
sibling client calls are re-issued advice-informed on the next iteration. The
native API instead pauses with the client calls pending; a proxy cannot,
because it can neither execute the client's tools nor hand the client a turn
containing a tool it was never offered.

Chain integration::

    [RequestProcessor*] → AdvisorToolCallBackend → [ResponseProcessor*] → TranslationEngine

Declares ``supported_request_types`` for the executor's wire so the
TranslationEngine normalizes any inbound format to it once. The outer chain's
``StatsResponseProcessor`` records the terminal turn's token usage; this
backend additionally records the advisor consults **and the intermediate
executor turns** (which the client never sees) into the planner bucket, so
the run's cost output prices the full loop, and stamps ``ctx.selected_model``.

Streaming is single-pass: each executor turn is streamed and buffered while
its content is reassembled to detect advisor calls; the terminal turn's
buffered events are replayed verbatim, so the turn the client pays for is
generated exactly once. Advice is not persisted across client turns (the
client cannot carry advisor exchanges back); its effect lives in the terminal
turn the client keeps.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from switchyard.lib.backends.advisor_loop_backend import (
    AdvisorCaller,
    _blocks_text,
    _build_advisor_caller,
    _consume_openai_stream,
    _ev,
    _replay_events,
    _seed_advice_for,
    _session_key,
    _usage_tokens,
    _with_length_line,
)
from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.backends.multi_llm_backend import (
    build_native_backend,
    resolve_llm_target,
)
from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    request_type_enum,
    request_type_matches,
    request_with_type,
)
from switchyard_rust.translation import TranslationEngine

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.stats_accumulator import StatsAccumulator
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: Tool result handed to the executor for advisor calls past the ``max_uses``
#: cap (mirrors the native tool's ``max_uses_exceeded`` error result).
_MAX_USES_RESULT = "[advisor unavailable: max_uses exceeded]"

#: Per-tool cap on the description text included in the advisor's transcript.
_TOOL_SUMMARY_DESC_CHARS = 200


@dataclass
class _ToolCallTurn:
    """One executor turn, normalized across wire formats and delivery modes.

    ``advisor_calls`` are the turn's advisor invocations in the executor's
    native shape (Anthropic ``tool_use`` blocks / OpenAI ``tool_calls``
    entries); ``message`` is the format-native assistant payload the dialect
    rebuilds the feedback turn from (Anthropic content-block list / OpenAI
    assistant message dict). Exactly one of ``completion_body`` /
    ``stream_events`` carries the payload for verbatim replay if the turn is
    terminal.
    """

    advisor_calls: list[dict[str, Any]]
    text: str
    message: Any
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    completion_body: Any | None = None
    stream_events: list[Any] | None = None


class _WireDialect(Protocol):
    """Wire-format-specific operations of the advisor tool-call loop."""

    #: ``request_with_type`` discriminator for the executor's wire
    #: (``"anthropic"`` or ``"openai_chat"``).
    request_type: str

    def advisor_tool_def(self, *, name: str, description: str) -> dict[str, Any]:
        """Build the synthetic, parameterless ``advisor`` tool definition."""
        ...

    def inject_steering(
        self,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        steering: str,
        length_line: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return (body, messages) with executor steering injected cache-stably."""
        ...

    def tool_summaries(self, tools: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Return (name, description) pairs for the client's tools."""
        ...

    async def parse_turn(
        self,
        response: ChatResponse,
        *,
        advisor_tool_name: str,
        latency_ms: float,
    ) -> _ToolCallTurn:
        """Buffer and reassemble an executor response into a ``_ToolCallTurn``."""
        ...

    def feedback_messages(
        self, turn: _ToolCallTurn, advice: str,
    ) -> list[dict[str, Any]]:
        """Messages appending the advisor turn + its tool result(s)."""
        ...

    def replay(self, turn: _ToolCallTurn) -> ChatResponse:
        """Rebuild the buffered terminal turn as a verbatim response."""
        ...


class _AnthropicDialect:
    """Anthropic Messages wire (``tool_use`` blocks, ``tool_result`` turns)."""

    request_type = "anthropic"

    def advisor_tool_def(self, *, name: str, description: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

    def inject_steering(
        self,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        steering: str,
        length_line: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        body = {**body, "system": _prepend_system(body.get("system"), steering)}
        return body, _with_length_line(messages, length_line)

    def tool_summaries(self, tools: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(str(t.get("name", "?")), str(t.get("description", ""))) for t in tools]

    async def parse_turn(
        self,
        response: ChatResponse,
        *,
        advisor_tool_name: str,
        latency_ms: float,
    ) -> _ToolCallTurn:
        if response.response_type == ChatResponseType.ANTHROPIC_STREAM:
            events, blocks, usage = await _consume_stream(response.stream)
            return self._turn(
                blocks, advisor_tool_name, latency_ms, stream_events=events, **usage,
            )
        completion: Any = response.to_body()
        blocks = [b for b in (completion.get("content") or []) if isinstance(b, dict)]
        prompt_tokens, completion_tokens = _usage_tokens(completion.get("usage"))
        cached = (completion.get("usage") or {}).get("cache_read_input_tokens") or 0
        return self._turn(
            blocks,
            advisor_tool_name,
            latency_ms,
            input_tokens=prompt_tokens or 0,
            output_tokens=completion_tokens or 0,
            cached_tokens=int(cached),
            completion_body=completion,
        )

    def _turn(
        self,
        blocks: list[dict[str, Any]],
        advisor_tool_name: str,
        latency_ms: float,
        **kwargs: Any,
    ) -> _ToolCallTurn:
        advisor_calls = [
            b for b in blocks
            if b.get("type") == "tool_use" and b.get("name") == advisor_tool_name
        ]
        return _ToolCallTurn(
            advisor_calls=advisor_calls,
            text=_blocks_text(blocks),
            message=blocks,
            latency_ms=latency_ms,
            **kwargs,
        )

    def feedback_messages(
        self, turn: _ToolCallTurn, advice: str,
    ) -> list[dict[str, Any]]:
        # Thinking blocks must round-trip verbatim for upstreams that verify
        # signatures; sibling client calls are dropped (regenerated next turn).
        return [
            {
                "role": "assistant",
                "content": _advisor_only_content(turn.message, turn.advisor_calls),
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.get("id"), "content": advice}
                    for call in turn.advisor_calls
                ],
            },
        ]

    def replay(self, turn: _ToolCallTurn) -> ChatResponse:
        if turn.stream_events is not None:
            return ChatResponse.anthropic_stream(
                AnthropicResponseStream(_replay_events(turn.stream_events))
            )
        return ChatResponse.anthropic_completion(turn.completion_body)


class _OpenAiDialect:
    """OpenAI Chat Completions wire (``tool_calls``, ``role: tool`` results)."""

    request_type = "openai_chat"

    def advisor_tool_def(self, *, name: str, description: str) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }

    def inject_steering(
        self,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        steering: str,
        length_line: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # OpenAI carries the system prompt as a leading message, not a body
        # field. _with_length_line's list branch already emits the OpenAI
        # content-part shape, so it is shared verbatim.
        messages = _prepend_system_message(messages, steering)
        return body, _with_length_line(messages, length_line)

    def tool_summaries(self, tools: list[dict[str, Any]]) -> list[tuple[str, str]]:
        summaries: list[tuple[str, str]] = []
        for tool in tools:
            raw_function = tool.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            summaries.append(
                (str(function.get("name", "?")), str(function.get("description", "")))
            )
        return summaries

    async def parse_turn(
        self,
        response: ChatResponse,
        *,
        advisor_tool_name: str,
        latency_ms: float,
    ) -> _ToolCallTurn:
        if response.response_type == ChatResponseType.OPENAI_STREAM:
            events, message, usage = await _consume_openai_stream(response.stream)
            return self._turn(
                message, advisor_tool_name, latency_ms, stream_events=events, **usage,
            )
        completion: Any = response.to_body()
        choices = completion.get("choices") or [{}]
        message = dict(choices[0].get("message") or {})
        usage_value = completion.get("usage")
        prompt_tokens, completion_tokens = _usage_tokens(usage_value)
        cached = ((usage_value or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ) or 0
        return self._turn(
            message,
            advisor_tool_name,
            latency_ms,
            input_tokens=prompt_tokens or 0,
            output_tokens=completion_tokens or 0,
            cached_tokens=int(cached),
            completion_body=completion,
        )

    def _turn(
        self,
        message: dict[str, Any],
        advisor_tool_name: str,
        latency_ms: float,
        **kwargs: Any,
    ) -> _ToolCallTurn:
        return _ToolCallTurn(
            advisor_calls=_openai_advisor_calls(message, advisor_tool_name),
            # reasoning_content is intentionally excluded, matching the
            # Anthropic dialect excluding thinking blocks from the text.
            text=str(message.get("content") or ""),
            message=message,
            latency_ms=latency_ms,
            **kwargs,
        )

    def feedback_messages(
        self, turn: _ToolCallTurn, advice: str,
    ) -> list[dict[str, Any]]:
        kept_ids = {call.get("id") for call in turn.advisor_calls}
        # OpenAI requires exactly one role:"tool" message per tool_call id in
        # the preceding assistant message; the same advice is fanned out.
        return [
            _advisor_only_message(turn.message, kept_ids),
            *(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": advice,
                }
                for call in turn.advisor_calls
            ),
        ]

    def replay(self, turn: _ToolCallTurn) -> ChatResponse:
        if turn.stream_events is not None:
            return ChatResponse.openai_stream(
                ResponseStream(_replay_events(turn.stream_events))
            )
        return ChatResponse.openai_completion(turn.completion_body)


def _dialect_for_format(fmt: BackendFormat) -> _WireDialect:
    """Return the wire dialect for a resolved executor format."""
    if fmt == BackendFormat.ANTHROPIC:
        return _AnthropicDialect()
    if fmt == BackendFormat.OPENAI:
        return _OpenAiDialect()
    if fmt == BackendFormat.RESPONSES:
        # Backstop for format: auto resolving to a Responses endpoint; the
        # config validator rejects an explicit responses format earlier.
        raise ValueError(
            "the advisor tool-call loop is Chat-shaped and does not support "
            "Responses executors; use format 'openai' or 'anthropic'"
        )
    raise ValueError(
        f"advisor executor format {fmt!r} must be resolved before constructing "
        "AdvisorToolCallBackend (pin format: 'openai' or 'anthropic' when "
        "supplying executor_backend)"
    )


class AdvisorToolCallBackend(LLMBackend):
    """Executor backend offering a proxy-intercepted ``advisor`` tool (any Chat wire)."""

    #: Absolute backstop on executor calls per request. ``max_uses`` already
    #: bounds consults; this bounds an executor that keeps calling the advisor
    #: through ``max_uses exceeded`` results.
    _HARD_ITERATION_CAP = 8

    def __init__(
        self,
        config: AdvisorConfig,
        *,
        stats_accumulator: StatsAccumulator | None = None,
        executor_backend: LLMBackend | None = None,
        advisor_caller: AdvisorCaller | None = None,
    ) -> None:
        self._config = config
        self._stats = stats_accumulator if config.enable_stats else None
        self._translation = TranslationEngine()
        # Resolve format: auto before dialect selection; injected fakes must
        # pin a concrete format (probing a fake's endpoint makes no sense).
        executor_target = (
            config.executor if executor_backend is not None
            else resolve_llm_target(config.executor)
        )
        self._dialect = _dialect_for_format(executor_target.format)
        self._request_type = cast(
            "ChatRequestType", request_type_enum(self._dialect.request_type),
        )
        # The executor is delegated to verbatim so caching survives
        # (cache_control breakpoints on Anthropic; prefix stability on OpenAI).
        self._executor_backend = executor_backend or build_native_backend(executor_target)
        self._advisor_caller = advisor_caller or _build_advisor_caller(config)
        # Per-session seed advice cache for seed_plan_advice ("" = unseeded).
        self._seed_advice: dict[str, str] = {}

    async def startup(self) -> None:
        await self._executor_backend.startup()

    async def shutdown(self) -> None:
        await self._executor_backend.shutdown()

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        """The executor's native wire; inbound formats are normalized to it."""
        return [self._request_type]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        normalized = self._translation.request_to_any_of(
            request, self.supported_request_types,
        )
        if not request_type_matches(normalized, self._request_type):
            raise TypeError(
                "AdvisorToolCallBackend expected a "
                f"{self._dialect.request_type} request after translation"
            )

        body = dict(normalized.body)
        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        base_tools: list[dict[str, Any]] = list(body.get("tools") or [])
        # Seed the session with upfront advisor advice (consulted once at the
        # session-opening request, cached, and re-injected identically on every
        # later turn so the upstream cache prefix stays stable). Applied before
        # steering so the advice directly follows the task text.
        if self._config.seed_plan_advice:
            session = _session_key(body.get("system"), messages)
            advice = await _seed_advice_for(
                self._seed_advice, session, messages,
                caller=self._advisor_caller, config=self._config, stats=self._stats,
            )
            if advice:
                messages = _with_length_line(
                    messages, self._config.seed_advice_prefix + advice,
                )
        if self._config.inject_steering:
            body, messages = self._dialect.inject_steering(
                body,
                messages,
                steering=self._config.executor_steering,
                length_line=self._config.advisor_length_line,
            )
        tools = [
            *base_tools,
            self._dialect.advisor_tool_def(
                name=self._config.advisor_tool_name,
                description=self._config.advisor_tool_description,
            ),
        ]
        tool_summaries = self._dialect.tool_summaries(base_tools)

        advisor_uses = 0
        turn: _ToolCallTurn | None = None
        for _ in range(self._HARD_ITERATION_CAP):
            turn_body = {**body, "messages": messages, "tools": tools}
            turn = await self._run_executor(
                ctx, request_with_type(self._dialect.request_type, turn_body),
            )

            if not turn.advisor_calls:
                # Real tool call(s) for the client, or a final answer.
                return await self._finish(ctx, turn)

            # This turn stays proxy-internal — price it into the planner bucket
            # (under the executor model) so the run's cost output sees it.
            await self._record_internal_turn(turn)

            if advisor_uses < self._config.max_uses:
                advisor_uses += 1
                advice = await self._consult_advisor(messages, turn.text, tool_summaries)
            else:
                advice = _MAX_USES_RESULT

            messages = [*messages, *self._dialect.feedback_messages(turn, advice)]

        log.warning(
            "AdvisorToolCallBackend: hit hard iteration cap (%d) without a terminal "
            "executor turn; returning the last result.",
            self._HARD_ITERATION_CAP,
        )
        if turn is None:  # unreachable: _HARD_ITERATION_CAP >= 1
            raise RuntimeError("AdvisorToolCallBackend produced no executor turn")
        return await self._finish(ctx, turn)

    # ------------------------------------------------------------------
    # Executor turn
    # ------------------------------------------------------------------

    async def _run_executor(self, ctx: ProxyContext, request: ChatRequest) -> _ToolCallTurn:
        """Call the executor, buffering its response and reassembling the turn."""
        started = time.monotonic()
        try:
            response = await self._executor_backend.call(ctx, request)
        except Exception:
            # Includes ContextWindowExceeded (the chain uses it for evict-and-retry).
            if self._stats is not None:
                await self._stats.record_error(self._config.executor.model)
            raise

        latency_ms = (time.monotonic() - started) * 1000.0
        return await self._dialect.parse_turn(
            response,
            advisor_tool_name=self._config.advisor_tool_name,
            latency_ms=latency_ms,
        )

    async def _finish(self, ctx: ProxyContext, turn: _ToolCallTurn) -> ChatResponse:
        """Record stats, stamp ctx, and rebuild the terminal turn as a response."""
        ctx.selected_model = self._config.executor.model
        ctx.backend_call_latency_ms = turn.latency_ms
        if self._stats is not None:
            await self._stats.record_success(self._config.executor.model, turn.latency_ms)
        return self._dialect.replay(turn)

    async def _record_internal_turn(self, turn: _ToolCallTurn) -> None:
        """Price a proxy-internal executor turn into the planner bucket."""
        if self._stats is None:
            return
        await self._stats.record_planner_usage(
            model=self._config.executor.model,
            prompt_tokens=turn.input_tokens,
            completion_tokens=turn.output_tokens,
            cached_tokens=turn.cached_tokens,
            latency_ms=turn.latency_ms,
        )

    # ------------------------------------------------------------------
    # Advisor consultation
    # ------------------------------------------------------------------

    async def _consult_advisor(
        self,
        messages: list[dict[str, Any]],
        current_turn_text: str,
        tool_summaries: list[tuple[str, str]],
    ) -> str:
        """Consult the advisor on the transcript and return its guidance.

        On failure with ``fail_open`` set, returns a short "unavailable" marker
        so the executor can proceed; the failed call still counts toward
        ``max_uses`` at the call site, bounding retries against a down advisor.
        """
        transcript = self._serialize_transcript(messages, current_turn_text, tool_summaries)
        started = time.monotonic()
        try:
            advice, usage = await self._advisor_caller.advise(
                system=self._config.advisor_system_prompt, transcript=transcript,
            )
        except Exception as exc:
            if not self._config.fail_open:
                raise
            log.warning(
                "AdvisorToolCallBackend: advisor call failed; continuing unadvised: %s", exc,
            )
            if self._stats is not None:
                await self._stats.record_planner_error(self._config.advisor.model)
            _audit_advisor(error=str(exc), usage=None,
                           latency_ms=(time.monotonic() - started) * 1000.0)
            return f"[advisor unavailable: {type(exc).__name__}]"

        latency_ms = (time.monotonic() - started) * 1000.0
        if self._stats is not None:
            prompt_tokens, completion_tokens = _usage_tokens(usage)
            await self._stats.record_planner_usage(
                model=self._config.advisor.model,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                cached_tokens=0,
                latency_ms=latency_ms,
            )
        _audit_advisor(error=None, usage=usage, latency_ms=latency_ms)
        return advice

    def _serialize_transcript(
        self,
        messages: list[dict[str, Any]],
        current_turn_text: str,
        tool_summaries: list[tuple[str, str]],
    ) -> str:
        """Serialize the conversation for the advisor, newest turns kept first.

        Mirrors the native tool's context: the executor's tools (as a compact
        name — description summary; full schemas would swamp the char budget),
        the conversation, and the text the executor has produced so far in the
        turn that called the advisor. When over ``transcript_max_chars``, the
        **oldest** messages are dropped — the newest turns are the ones the
        consult is about.
        """
        sections: list[str] = []
        if tool_summaries:
            summary = "\n".join(
                f"- {name}: {desc[:_TOOL_SUMMARY_DESC_CHARS]}"
                for name, desc in tool_summaries
            )
            sections.append(f"Tools available to the executor:\n{summary}")

        parts = [json.dumps(m, default=str, ensure_ascii=False) for m in messages]
        cap = self._config.transcript_max_chars
        kept: list[str] = []
        total = 0
        for part in reversed(parts):
            if kept and total + len(part) > cap:
                break
            if not kept and len(part) > cap:
                part = "...<truncated>" + part[-(cap - 16):]
            kept.append(part)
            total += len(part)
        kept.reverse()
        omitted = len(parts) - len(kept)
        header = "Conversation so far (JSON, oldest first"
        if omitted:
            header += f"; {omitted} earlier messages omitted"
        sections.append(header + "):\n[" + ",\n".join(kept) + "]")

        sections.append(
            "The executor's turn so far (it is consulting you now):\n"
            + (current_turn_text or "(no text)")
        )
        return "\n\n".join(sections)


# ----------------------------------------------------------------------
# Anthropic-wire helpers
# ----------------------------------------------------------------------


async def _consume_stream(
    stream: Any,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, int]]:
    """Buffer an Anthropic stream; reassemble its content blocks and usage.

    Events are the dicts the native backend's SSE parser yields. Returns
    ``(events, blocks, usage)`` where ``usage`` carries ``input_tokens`` /
    ``output_tokens`` / ``cached_tokens`` keyword-ready for ``_ToolCallTurn``.
    """
    events: list[Any] = []
    blocks: dict[int, dict[str, Any]] = {}
    json_parts: dict[int, list[str]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    async for event in stream:
        events.append(event)
        etype = _ev(event, "type")
        if etype == "message_start":
            start_usage = _ev(_ev(event, "message"), "usage") or {}
            usage["input_tokens"] = int(start_usage.get("input_tokens") or 0)
            usage["cached_tokens"] = int(start_usage.get("cache_read_input_tokens") or 0)
        elif etype == "content_block_start":
            index = int(_ev(event, "index") or 0)
            block = _ev(event, "content_block")
            blocks[index] = dict(block) if isinstance(block, dict) else {}
        elif etype == "content_block_delta":
            index = int(_ev(event, "index") or 0)
            block = blocks.setdefault(index, {})
            delta = _ev(event, "delta")
            dtype = _ev(delta, "type")
            if dtype == "text_delta":
                block["text"] = str(block.get("text") or "") + str(_ev(delta, "text") or "")
            elif dtype == "input_json_delta":
                json_parts.setdefault(index, []).append(str(_ev(delta, "partial_json") or ""))
            elif dtype == "thinking_delta":
                block["thinking"] = (
                    str(block.get("thinking") or "") + str(_ev(delta, "thinking") or "")
                )
            elif dtype == "signature_delta":
                block["signature"] = _ev(delta, "signature")
        elif etype == "message_delta":
            delta_usage = _ev(event, "usage") or {}
            usage["output_tokens"] = int(delta_usage.get("output_tokens") or 0)
    for index, parts in json_parts.items():
        joined = "".join(parts).strip()
        try:
            blocks[index]["input"] = json.loads(joined) if joined else {}
        except json.JSONDecodeError:
            blocks[index]["input"] = {}
    ordered = [blocks[i] for i in sorted(blocks)]
    return events, ordered, usage


def _advisor_only_content(
    blocks: list[dict[str, Any]], advisor_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The assistant turn's blocks with sibling (non-advisor) tool calls dropped."""
    kept_ids = {call.get("id") for call in advisor_calls}
    return [
        b for b in blocks
        if b.get("type") != "tool_use" or b.get("id") in kept_ids
    ]


def _prepend_system(system: Any, prefix: str) -> Any:
    """Prepend steering to an Anthropic ``system`` field (string or block list)."""
    if system is None or system == "":
        return prefix
    if isinstance(system, str):
        return f"{prefix}\n\n{system}"
    if isinstance(system, list):
        return [{"type": "text", "text": prefix}, *system]
    return f"{prefix}\n\n{system}"


# ----------------------------------------------------------------------
# OpenAI-wire helpers
# ----------------------------------------------------------------------


def _openai_advisor_calls(
    message: dict[str, Any], tool_name: str,
) -> list[dict[str, Any]]:
    """The message's advisor ``tool_calls``, detected by function name.

    Detection is by tool-call presence, never ``finish_reason`` — some OSS
    servers mislabel tool-call turns as ``stop``.
    """
    return [
        tc for tc in (message.get("tool_calls") or [])
        if isinstance(tc, dict)
        and (tc.get("function") or {}).get("name") == tool_name
    ]


def _advisor_only_message(
    message: dict[str, Any], kept_ids: set[Any],
) -> dict[str, Any]:
    """The assistant message with sibling (non-advisor) tool calls dropped.

    Key-whitelisted on purpose: vendor fields such as ``reasoning_content``
    are dropped so strict endpoints (OpenAI proper) accept the round-trip;
    the advisor already saw the turn's text via the transcript.
    """
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": [
            tc for tc in (message.get("tool_calls") or [])
            if isinstance(tc, dict) and tc.get("id") in kept_ids
        ],
    }


def _prepend_system_message(
    messages: list[dict[str, Any]], prefix: str,
) -> list[dict[str, Any]]:
    """Prepend steering to the first ``system``/``developer`` message.

    When the request carries no system-role message at all, one is inserted at
    index 0 so the steering still leads the prompt (and the prefix cache stays
    stable across the session's turns).
    """
    msgs = [dict(m) for m in messages]
    for msg in msgs:
        if msg.get("role") not in ("system", "developer"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [{"type": "text", "text": prefix}, *content]
        else:
            msg["content"] = f"{prefix}\n\n{content or ''}".rstrip()
        return msgs
    return [{"role": "system", "content": prefix}, *msgs]


# ----------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------


def _audit_advisor(*, error: str | None, usage: Any, latency_ms: float) -> None:
    """Emit a one-line ``advisor_call=...`` audit record to stderr."""
    payload: dict[str, Any] = {
        "advisor_call": True,
        "error": error,
        "latency_ms": round(latency_ms, 1),
    }
    prompt_tokens, completion_tokens = _usage_tokens(usage)
    if prompt_tokens is not None:
        payload["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        payload["completion_tokens"] = completion_tokens
    sys.stderr.write(f"advisor_call={json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


__all__ = ["AdvisorToolCallBackend"]
