# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LLMBackend`` that gates the executor with a once-per-session advisor review.

This is the ``review_gate`` strategy of :class:`AdvisorConfig`; the
executor-triggered tool-call strategy lives in
``switchyard/lib/backends/advisor_tool_call_backend.py``.

An earlier design offered the executor an ``advisor`` tool it could call
mid-generation. Trace analysis showed that front-loading the advisor's plan
*suppressed the executor's own test-and-iterate loop* — it trusted the plan,
one-shot it, and declared "done" prematurely (e.g. solving a concurrency task in
4 turns vs the 17 the unadvised baseline needed to catch the bug). Net effect
was within noise, with real losses on tasks the baseline solved by iterating.

This backend instead uses the advisor as a **once-per-session review gate**:

1. The executor works the task with its **own** tools (no advisor tool injected,
   no upfront advice) — its iteration loop is untouched.
2. The first time the executor produces a turn with **no tool calls** — either a
   plan it is about to execute, or a claim that the task is complete — the
   backend consults the advisor **once** to review the full transcript:
   - ``APPROVE`` → the executor's turn is returned unchanged (sound plan / done).
   - ``REDO`` → the advisor's optimized plan is fed back as a user turn and the
     executor is re-invoked to **keep working** (it produces tool calls again).
3. Subsequent turns in the same session pass through unreviewed
   (once-per-session), so the gate can force at most one extra round of work.

This is a near-superset of solo behavior — identical to the bare executor until
"done", plus one quality gate — so it is downside-protected (≈ baseline if the
advisor always approves) while catching premature convergence.

The executor is **native Anthropic** (``/v1/messages``, Bearer auth): it is
called by delegating verbatim to an :class:`AnthropicNativeBackend`, so the
client's ``cache_control`` breakpoints reach the upstream unchanged and prompt
caching is honored (``AdvisorConfig`` validation rejects non-Anthropic
executors for this strategy). The gate reads the executor's tool use from the
Anthropic ``stop_reason``/``tool_use`` content blocks. The advisor tier is
format-dispatched (``_build_advisor_caller``): Anthropic Messages or OpenAI
Chat Completions.

Chain integration::

    [RequestProcessor*] → AdvisorLoopBackend → [ResponseProcessor*] → TranslationEngine

Declares ``supported_request_types = [ANTHROPIC]`` and normalizes inbound
OpenAI / Responses via the TranslationEngine, mirroring
:class:`LatencyServiceLLMBackend`. The outer chain's ``StatsResponseProcessor``
records executor token usage (including cache reads) from the returned response;
this backend additionally records the advisor review's usage into the planner
bucket and stamps ``ctx.selected_model``.

Streaming is single-pass: until a session is reviewed, each executor turn is
streamed and buffered while detecting whether it has tool calls; a passed-through
/ approved turn's buffered events are replayed verbatim, so the turn is generated
once. After the review fires, the session is pure passthrough (the upstream
stream is returned directly — true streaming, full caching, zero overhead).
Once-per-session is tracked in-process by a hash of the conversation's stable
prefix (system + first user message); all of a task's turns hit the same per-run
switchyard pod. A pod restart mid-session could allow a second review (rare,
harmless).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from switchyard.lib.backends.multi_llm_backend import build_native_backend
from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.profiles.advisor_prompts import REDO_FEEDBACK_PREFIX
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    request_type_matches,
    request_with_type,
)
from switchyard_rust.translation import TranslationEngine

if TYPE_CHECKING:
    from switchyard.lib.backends.llm_target import LlmTarget
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.stats_accumulator import StatsAccumulator
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"


class AdvisorCaller(Protocol):
    """Consults the advisor model and returns ``(text, usage)``."""

    async def advise(self, *, system: str, transcript: str) -> tuple[str, Any]:
        ...


@dataclass
class _ExecTurn:
    """One executor turn, normalized across the buffered streaming / completion paths."""

    has_tool_use: bool
    content: str | None
    latency_ms: float
    completion_body: Any | None = None
    stream_events: list[Any] | None = None


class AdvisorLoopBackend(LLMBackend):
    """Executor backend gated by a once-per-session advisor review (native Anthropic)."""

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
        # Sessions already reviewed (once-per-session), keyed by conversation
        # prefix hash. In-process; a task's turns share one switchyard pod.
        self._reviewed: set[str] = set()
        # The executor is delegated to verbatim so cache_control passes through.
        self._executor_backend = executor_backend or build_native_backend(config.executor)
        self._advisor_caller = advisor_caller or _build_advisor_caller(config)

    async def startup(self) -> None:
        await self._executor_backend.startup()

    async def shutdown(self) -> None:
        await self._executor_backend.shutdown()

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        """Executor + gate are native Anthropic Messages."""
        return [ChatRequestType.ANTHROPIC]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        normalized = self._translation.request_to_any_of(
            request, self.supported_request_types,
        )
        if not request_type_matches(normalized, ChatRequestType.ANTHROPIC):
            raise TypeError(
                "AdvisorLoopBackend expected an Anthropic request after translation"
            )

        body = dict(normalized.body)
        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        session = _session_key(body.get("system"), messages)

        # After the gate has fired for this session, every turn is pure
        # passthrough — return the upstream stream directly (true streaming,
        # caching intact, no buffering).
        if session in self._reviewed:
            return await self._passthrough(ctx, normalized)

        # Not yet reviewed: run the executor and inspect its turn for tool use.
        turn = await self._run_executor(ctx, normalized)

        # Tool calls → executor is working; never gate mid-work.
        if turn.has_tool_use:
            return await self._finish(ctx, turn)

        # No tool calls = a plan or a "done". Gate it (once per session).
        self._reviewed.add(session)
        verdict, plan = await self._review(messages, turn.content)
        if verdict != "REDO":
            return await self._finish(ctx, turn)

        # REDO: feed the optimized plan back and re-invoke so the executor keeps
        # working instead of stopping. The session is now reviewed, so the redo
        # turn (and everything after it) is plain passthrough.
        redo_messages = [
            *messages,
            {"role": "assistant", "content": turn.content or ""},
            {"role": "user", "content": REDO_FEEDBACK_PREFIX + plan},
        ]
        redo_body = {**body, "messages": redo_messages}
        redo_request = request_with_type("anthropic", redo_body)
        return await self._passthrough(ctx, redo_request)

    # ------------------------------------------------------------------
    # Executor turn
    # ------------------------------------------------------------------

    async def _run_executor(self, ctx: ProxyContext, request: ChatRequest) -> _ExecTurn:
        """Call the executor, buffering its response to detect tool use."""
        started = time.monotonic()
        try:
            response = await self._executor_backend.call(ctx, request)
        except Exception:
            # Includes ContextWindowExceeded (the chain uses it for evict-and-retry).
            if self._stats is not None:
                await self._stats.record_error(self._config.executor.model)
            raise

        latency_ms = (time.monotonic() - started) * 1000.0
        if response.response_type == ChatResponseType.ANTHROPIC_STREAM:
            events, has_tool_use, content = await _consume_anthropic_stream(response.stream)
            return _ExecTurn(
                has_tool_use=has_tool_use,
                content=content,
                latency_ms=latency_ms,
                stream_events=events,
            )
        body = response.to_body()
        has_tool_use, content = _completion_tool_use(body)
        return _ExecTurn(
            has_tool_use=has_tool_use,
            content=content,
            latency_ms=latency_ms,
            completion_body=body,
        )

    async def _passthrough(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        """Call the executor and return its response verbatim (no buffering)."""
        started = time.monotonic()
        try:
            response = await self._executor_backend.call(ctx, request)
        except Exception:
            if self._stats is not None:
                await self._stats.record_error(self._config.executor.model)
            raise
        await self._stamp(ctx, (time.monotonic() - started) * 1000.0)
        return response

    async def _finish(self, ctx: ProxyContext, turn: _ExecTurn) -> ChatResponse:
        """Record stats, stamp ctx, and rebuild the buffered turn as a response."""
        await self._stamp(ctx, turn.latency_ms)
        if turn.stream_events is not None:
            return ChatResponse.anthropic_stream(
                AnthropicResponseStream(_replay_events(turn.stream_events))
            )
        return ChatResponse.anthropic_completion(turn.completion_body)

    async def _stamp(self, ctx: ProxyContext, latency_ms: float) -> None:
        ctx.selected_model = self._config.executor.model
        ctx.backend_call_latency_ms = latency_ms
        if self._stats is not None:
            await self._stats.record_success(self._config.executor.model, latency_ms)

    # ------------------------------------------------------------------
    # Advisor review
    # ------------------------------------------------------------------

    async def _review(
        self, messages: list[dict[str, Any]], terminal_content: str | None,
    ) -> tuple[str, str]:
        """Consult the advisor once; return ``(verdict, plan)``.

        ``verdict`` is ``"APPROVE"`` or ``"REDO"``. On a fail-open advisor error
        or an unparseable reply, defaults to ``APPROVE`` (do not disrupt a
        possibly-correct turn).
        """
        transcript = self._serialize_transcript(messages, terminal_content)
        started = time.monotonic()
        try:
            text, usage = await self._advisor_caller.advise(
                system=self._config.reviewer_system_prompt, transcript=transcript,
            )
        except Exception as exc:
            if not self._config.fail_open:
                raise
            log.warning("AdvisorLoopBackend: review failed; approving (fail-open): %s", exc)
            if self._stats is not None:
                await self._stats.record_planner_error(self._config.advisor.model)
            _audit_review(verdict="APPROVE", error=str(exc), usage=None,
                          latency_ms=(time.monotonic() - started) * 1000.0)
            return "APPROVE", ""
        latency_ms = (time.monotonic() - started) * 1000.0
        verdict, plan = _parse_verdict(text)
        # Record the advisor review's token usage so the run's own cost output
        # (``routing_stats_final.json``) accounts for the advisor, not just the
        # executor. Recorded into the planner bucket — the advisor review is a
        # secondary-model consult, like a planner — so its Opus-4.8 cost rolls
        # into ``cost_estimate.total_cost``.
        if self._stats is not None:
            prompt_tokens, completion_tokens = _usage_tokens(usage)
            await self._stats.record_planner_usage(
                model=self._config.advisor.model,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                cached_tokens=0,
                latency_ms=latency_ms,
            )
        _audit_review(verdict=verdict, error=None, usage=usage, latency_ms=latency_ms)
        return verdict, plan

    def _serialize_transcript(
        self, messages: list[dict[str, Any]], terminal_content: str | None,
    ) -> str:
        """Serialize the conversation + the executor's terminal turn for review."""
        text = json.dumps(messages, default=str, ensure_ascii=False)
        cap = self._config.transcript_max_chars
        if len(text) > cap:
            text = text[: cap - 16] + "...<truncated>"
        tail = terminal_content or "(no text)"
        return (
            f"Conversation so far (JSON):\n\n{text}\n\n"
            f"The executor's latest turn (a plan, or its claim the task is done):\n{tail}"
        )


# ----------------------------------------------------------------------
# Advisor callers
# ----------------------------------------------------------------------


def _build_advisor_caller(config: AdvisorConfig) -> AdvisorCaller:
    """Build the advisor caller for ``config.advisor``, dispatched on its format."""
    from switchyard.lib.backends.llm_target import BackendFormat
    from switchyard.lib.backends.multi_llm_backend import resolve_llm_target

    target = resolve_llm_target(config.advisor)
    if target.format == BackendFormat.ANTHROPIC:
        return _AnthropicAdvisorCaller(
            api_key=target.endpoint.api_key,
            base_url=target.endpoint.base_url,
            model=target.model,
            max_tokens=config.advisor_max_tokens,
            temperature=config.advisor_temperature,
            timeout=target.endpoint.timeout_secs,
        )
    if target.format == BackendFormat.OPENAI:
        return _OpenAiAdvisorCaller(
            target=target,
            max_tokens=config.advisor_max_tokens,
            temperature=config.advisor_temperature,
        )
    raise ValueError(
        f"advisor tier does not support format {target.format!r}; "
        "use 'openai' or 'anthropic'"
    )


class _AnthropicAdvisorCaller:
    """Reviews via an Anthropic-Messages advisor (``/v1/messages``, Bearer auth)."""

    def __init__(
        self, *, api_key: str | None, base_url: str | None, model: str,
        max_tokens: int, temperature: float | None, timeout: float | None,
    ) -> None:
        self._url = _messages_url(base_url)
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    async def advise(self, *, system: str, transcript: str) -> tuple[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "messages": [{"role": "user", "content": transcript}],
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
        return _anthropic_text(data), data.get("usage")


class _OpenAiAdvisorCaller:
    """Consults an OpenAI-Chat advisor (``/chat/completions`` via the SDK).

    Covers OSS advisors (DeepSeek, Qwen on vLLM/NIM) and OpenAI. Built with
    ``max_retries=0`` so a slow or down advisor falls through to the backend's
    own ``fail_open`` handling at the configured timeout instead of
    compounding via SDK exponential backoff (same rationale as the LLM
    classifier's client).
    """

    def __init__(
        self, *, target: LlmTarget, max_tokens: int, temperature: float | None,
    ) -> None:
        from switchyard.lib.llm_client import OpenAILLMClient

        self._client = OpenAILLMClient(
            api_key=target.endpoint.api_key,
            base_url=target.endpoint.base_url,
            timeout=target.endpoint.timeout_secs,
            max_retries=0,
        )
        self._model = target.model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Forward target-level overrides so gateway auth headers and vLLM
        # chat-template hints configured on the route work here too.
        self._extra_body = dict(target.extra_body) if target.extra_body else None
        self._extra_headers = dict(target.extra_headers) if target.extra_headers else None

    async def advise(self, *, system: str, transcript: str) -> tuple[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ],
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._extra_body is not None:
            kwargs["extra_body"] = self._extra_body
        if self._extra_headers is not None:
            kwargs["extra_headers"] = self._extra_headers
        result = await self._client.acompletion(**kwargs)
        choices = getattr(result, "choices", None) or []
        content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        return (content or "").strip(), getattr(result, "usage", None)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


async def _consume_anthropic_stream(stream: Any) -> tuple[list[Any], bool, str | None]:
    """Buffer an Anthropic stream; return (events, has_tool_use, assistant_text)."""
    events: list[Any] = []
    has_tool_use = False
    text_parts: list[str] = []
    async for event in stream:
        events.append(event)
        etype = _ev(event, "type")
        if etype == "content_block_start":
            if _ev(_ev(event, "content_block"), "type") == "tool_use":
                has_tool_use = True
        elif etype == "content_block_delta":
            delta = _ev(event, "delta")
            if _ev(delta, "type") == "text_delta":
                piece = _ev(delta, "text")
                if isinstance(piece, str):
                    text_parts.append(piece)
        elif etype == "message_delta":
            if _ev(_ev(event, "delta"), "stop_reason") == "tool_use":
                has_tool_use = True
    return events, has_tool_use, ("".join(text_parts) or None)


async def _replay_events(events: list[Any]) -> Any:
    """Replay buffered stream events verbatim as a fresh async stream."""
    for event in events:
        yield event


def _completion_tool_use(body: Any) -> tuple[bool, str | None]:
    """Read (has_tool_use, assistant_text) from an Anthropic completion body."""
    if not isinstance(body, dict):
        return False, None
    content = body.get("content") or []
    has_tool_use = body.get("stop_reason") == "tool_use" or any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    )
    return has_tool_use, (_blocks_text(content) or None)


def _ev(event: Any, key: str) -> Any:
    """Read a field from a stream event (dict from Rust, or an SDK object)."""
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _session_key(system: Any, messages: list[dict[str, Any]]) -> str:
    """Stable per-session key: hash of system prompt + first user message."""
    parts: list[str] = ["S:" + _blocks_text(system)]
    for m in messages:
        if m.get("role") == "user":
            parts.append("U:" + _blocks_text(m.get("content")))
            break
    return hashlib.sha256("\n".join(parts).encode("utf-8", "ignore")).hexdigest()


def _blocks_text(content: Any) -> str:
    """Flatten Anthropic content (string, or a list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse the reviewer reply into (verdict, plan). Unclear → APPROVE."""
    stripped = (text or "").strip()
    head = stripped[:16].upper()
    if head.startswith("APPROVE"):
        return "APPROVE", ""
    if head.startswith("REDO"):
        plan = stripped[4:].lstrip(" :\n-").strip()
        return "REDO", plan or stripped
    return "APPROVE", ""


def _messages_url(base_url: str | None) -> str:
    """Resolve the Anthropic Messages URL from a target base URL."""
    base = (base_url or "https://api.anthropic.com").rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _anthropic_text(data: dict[str, Any]) -> str:
    """Join the ``text`` content blocks of an Anthropic Messages response."""
    content = data.get("content") or []
    return "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    """Read (input, output) token counts from Anthropic- or OpenAI-shaped usage."""
    if usage is None:
        return None, None

    def get(*names: str) -> int | None:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if value is not None:
                return int(value)
        return None

    return get("input_tokens", "prompt_tokens"), get("output_tokens", "completion_tokens")


def _audit_review(*, verdict: str, error: str | None, usage: Any, latency_ms: float) -> None:
    """Emit a one-line ``advisor_review=...`` audit record to stderr."""
    payload: dict[str, Any] = {
        "advisor_review": True,
        "verdict": verdict,
        "error": error,
        "latency_ms": round(latency_ms, 1),
    }
    prompt_tokens, completion_tokens = _usage_tokens(usage)
    if prompt_tokens is not None:
        payload["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        payload["completion_tokens"] = completion_tokens
    sys.stderr.write(f"advisor_review={json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


__all__ = ["AdvisorCaller", "AdvisorLoopBackend"]
