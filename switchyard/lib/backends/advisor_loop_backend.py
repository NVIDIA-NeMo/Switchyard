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

The executor's wire is selected by ``config.executor.format``: ``anthropic``
executors are delegated verbatim to an :class:`AnthropicNativeBackend`
(``/v1/messages`` — the client's ``cache_control`` breakpoints reach the
upstream unchanged, so prompt caching is honored); ``openai`` executors
(Qwen, DeepSeek, vLLM/NIM, OpenAI) are delegated verbatim to an
:class:`OpenAiNativeBackend` (``/chat/completions``). Tool use is read from
the wire's native shape (Anthropic ``stop_reason``/``tool_use`` blocks, or
OpenAI ``tool_calls``/``finish_reason``); the REDO feedback is plain-string
assistant/user turns, valid on both wires, with a config-tunable prefix
(``redo_feedback_prefix``). The advisor tier is likewise format-dispatched
(``_build_advisor_caller``): Anthropic Messages or OpenAI Chat Completions.
Because the gate's trigger is proxy-side (the first no-tool-call turn), it
fires regardless of the executor's own tool-use discipline — the property
that distinguishes it from the executor-triggered tool-call strategy on weak
executors.

Chain integration::

    [RequestProcessor*] → AdvisorLoopBackend → [ResponseProcessor*] → TranslationEngine

Declares ``supported_request_types`` for the executor's wire so the
TranslationEngine normalizes any inbound format to it once. The outer chain's
``StatsResponseProcessor`` records executor token usage (including cache reads)
from the returned response; this backend additionally records the advisor
review's usage into the classifier bucket and stamps ``ctx.selected_model``.

Streaming is single-pass: until a session is reviewed, each executor turn is
streamed and buffered while detecting whether it has tool calls; a passed-through
/ approved turn's buffered events are replayed verbatim, so the turn is generated
once. After the review fires, the session is pure passthrough (the upstream
stream is returned directly — true streaming, full caching, zero overhead).
The review budget (``max_reviews``) is keyed by the caller-declared session
identity: the ``proxy_x_session_id`` header (parsed into
``RequestMetadata.session_id`` by the endpoints) when present, else a single
instance-wide scope. Benchmark harnesses stamp that header with the evaluation
id on every request — including sub-agent conversations — so on a gateway
shared by many tasks each task gets its own budget, exactly the "reviews for
*this* task" semantics ``max_reviews`` is meant to have. The content hash of
the conversation prefix (``_session_key``) is NOT used for budgeting — it is
not reliably stable (harnesses compact history, spawn sub-conversations, and
re-render system context) — and only keys the seed-advice cache and the stall
checkpoint. Failed advisor consults do not consume budget; after
``_MAX_FAILED_CONSULTS_PER_SCOPE`` failures a scope stops consulting entirely,
bounding latency against a down advisor. A pod restart mid-run resets the
budget (rare, harmless).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.backends.multi_llm_backend import (
    build_native_backend,
    resolve_llm_target,
)
from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
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
    from switchyard.lib.backends.llm_target import LlmTarget
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.stats_accumulator import StatsAccumulator
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
#: Distinct session keys on one backend instance past which the hashed
#: conversation prefix is assumed unstable (see ``_session_key``). A single
#: agent run legitimately opens a handful of conversations (the main thread plus
#: any sub-agents); dozens means the key is churning per turn.
_SESSION_CHURN_WARN_AT = 12
#: Budget scope for callers that send no ``proxy_x_session_id`` header.
_INSTANCE_SCOPE = "__instance__"
#: Failed (fail-open) consults tolerated per budget scope before the gate stops
#: consulting. Failures refund the review budget — a transient advisor error
#: must not silently exhaust ``max_reviews`` with zero real reviews — so this
#: separate cap is what bounds per-turn consult latency against a down advisor.
_MAX_FAILED_CONSULTS_PER_SCOPE = 3


class AdvisorCaller(Protocol):
    """Consults the advisor model and returns ``(text, usage)``."""

    async def advise(self, *, system: str, transcript: str) -> tuple[str, Any]:
        ...


@dataclass
class _ExecTurn:
    """One executor turn, normalized across the buffered streaming / completion paths.

    Token counts let a REDO-discarded turn (which the client never sees, so the
    outer stats processor never prices it) be recorded explicitly.
    """

    has_tool_use: bool
    content: str | None
    latency_ms: float
    completion_body: Any | None = None
    stream_events: list[Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    #: Reasoning-model internal text (OpenAI ``reasoning_content``); the
    #: fallback evidence when a gated turn has no visible content at all.
    reasoning_text: str | None = None


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
        # Sessions whose stall checkpoint (gate_stall_turns) already fired.
        self._stall_fired: set[str] = set()
        # Per-session seed advice cache for seed_plan_advice ("" = unseeded).
        self._seed_advice: dict[str, str] = {}
        # Session-key churn observability: the hashed conversation prefix keys
        # the seed cache and stall checkpoint, so instability there degrades
        # those features (it no longer affects the review budget, which keys on
        # the caller's session header). Tracked over *every* request so churn
        # stays observable even when the gate never fires.
        self._sessions_seen: set[str] = set()
        self._session_churn_warned = False
        # Review budget, keyed by ``_budget_scope``: the caller's
        # ``proxy_x_session_id`` header when present (one evaluation/task on
        # benchmark harnesses, shared by its sub-agents), else one
        # instance-wide scope. On a gateway shared by many tasks this gives
        # each task its own ``max_reviews`` budget. Failed consults refund the
        # budget and count separately (``_MAX_FAILED_CONSULTS_PER_SCOPE``).
        self._reviews_by_scope: dict[str, int] = {}
        self._failed_consults_by_scope: dict[str, int] = {}
        self._budget_logged_scopes: set[str] = set()
        # Resolve format: auto before wire selection; injected fakes must pin
        # a concrete format (probing a fake's endpoint makes no sense).
        executor_target = (
            config.executor if executor_backend is not None
            else resolve_llm_target(config.executor)
        )
        self._request_type_name = _gate_request_type(executor_target.format)
        self._request_type = cast(
            "ChatRequestType", request_type_enum(self._request_type_name),
        )
        self._is_openai = self._request_type_name == "openai_chat"
        # Pre-compiled pattern trigger; None selects the no_tool_call trigger.
        self._trigger_pattern = (
            re.compile(config.gate_trigger_pattern)
            if config.gate_trigger == "pattern"
            else None
        )
        # The executor is delegated to verbatim so caching survives
        # (cache_control breakpoints on Anthropic; prefix stability on OpenAI).
        self._executor_backend = executor_backend or build_native_backend(executor_target)
        self._advisor_caller = advisor_caller or _build_advisor_caller(config)

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
                "AdvisorLoopBackend expected a "
                f"{self._request_type_name} request after translation"
            )

        body = dict(normalized.body)
        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        session = _session_key(body.get("system"), messages)
        if session not in self._sessions_seen:
            self._sessions_seen.add(session)
            log.info(
                "AdvisorLoopBackend: new session key (%d distinct seen)",
                len(self._sessions_seen),
            )
            if (
                not self._session_churn_warned
                and len(self._sessions_seen) >= _SESSION_CHURN_WARN_AT
            ):
                self._session_churn_warned = True
                log.warning(
                    "AdvisorLoopBackend: %d distinct session keys seen on one backend "
                    "instance; the hashed conversation prefix is unstable for this "
                    "client, so seed-advice caching and stall checkpoints may misfire "
                    "(the review budget is unaffected — it keys on the caller's "
                    "session header)",
                    len(self._sessions_seen),
                )

        # Seed the session with upfront advisor advice (consulted once at the
        # session-opening request, cached, and re-injected identically on every
        # later turn so the upstream cache prefix stays stable).
        if self._config.seed_plan_advice:
            advice = await _seed_advice_for(
                self._seed_advice, session, messages,
                caller=self._advisor_caller, config=self._config, stats=self._stats,
                ctx=ctx,
            )
            if advice:
                messages = _with_length_line(
                    messages, self._config.seed_advice_prefix + advice,
                )
                body = {**body, "messages": messages}
                normalized = request_with_type(self._request_type_name, body)

        # Once the review budget is spent, every turn is pure passthrough —
        # return the upstream stream directly (true streaming, caching intact,
        # no buffering).
        #
        # The budget keys on the caller's declared session identity
        # (``proxy_x_session_id`` header), NOT on the conversation content hash:
        # content hashes are unstable (harnesses compact history, spawn
        # sub-conversations, re-render system context — measured on
        # Terminal-Bench, one task minted up to 194 keys and drew 107 reviews
        # against a configured ``max_reviews`` of 2), and an instance-wide cap
        # breaks the other way on a gateway shared by many tasks, where it
        # would bound reviews for the whole run instead of per task. The header
        # is stamped per evaluation by benchmark harnesses (sub-agents
        # included), so it expresses exactly "reviews for *this* task"; callers
        # that send no header fall back to one instance-wide scope.
        scope = self._budget_scope(ctx)
        if self._scope_exhausted(scope):
            if scope not in self._budget_logged_scopes:
                self._budget_logged_scopes.add(scope)
                log.info(
                    "AdvisorLoopBackend: review budget spent for scope %s "
                    "(max_reviews=%d, reviews=%d, failed consults=%d); "
                    "remaining turns pass through",
                    scope,
                    self._config.max_reviews,
                    self._reviews_by_scope.get(scope, 0),
                    self._failed_consults_by_scope.get(scope, 0),
                )
            return await self._passthrough(ctx, normalized)

        # Budget remains: run the executor and inspect its turn.
        turn = await self._run_executor(ctx, normalized)

        # Stall checkpoint (once per session): the conversation has grown past
        # ``gate_stall_turns`` assistant turns without the main trigger having
        # fired — review mid-task regardless of the turn's shape.
        stall = (
            self._config.gate_stall_turns > 0
            and session not in self._stall_fired
            and sum(1 for m in messages if m.get("role") == "assistant")
            >= self._config.gate_stall_turns
        )

        # Trigger check. "no_tool_call" gates the first turn without tool
        # calls (function-calling harnesses; ``gate_min_tool_results`` skips
        # early commentary turns before any real work exists); "pattern"
        # gates the first turn whose text matches the configured marker
        # (text-protocol harnesses, e.g. terminus's ``task_complete: true``
        # declaration).
        if self._trigger_pattern is not None:
            triggered = bool(self._trigger_pattern.search(turn.content or ""))
        else:
            triggered = not turn.has_tool_use and (
                _count_tool_results(messages) >= self._config.gate_min_tool_results
            )
        if not (triggered or stall):
            return await self._finish(ctx, turn)

        # Trigger fired: a plan, a "done", the marker, or a stall checkpoint.
        # Gate it. Budget is reserved before the consult (so concurrent
        # requests in one scope cannot overdraw across the await) and refunded
        # if the consult itself failed — a fail-open error is not a review.
        if stall and not triggered:
            self._stall_fired.add(session)
        # A reasoning-only turn (no visible text) still triggers; hand the
        # advisor the reasoning as labeled evidence instead of "(no text)".
        review_tail = turn.content
        if review_tail is None and turn.reasoning_text:
            review_tail = (
                "(the executor produced no visible text this turn; its "
                "internal reasoning follows)\n" + turn.reasoning_text
            )
        self._reviews_by_scope[scope] = self._reviews_by_scope.get(scope, 0) + 1
        verdict, plan, consulted = await self._review(messages, review_tail, ctx)
        if not consulted:
            self._reviews_by_scope[scope] -= 1
            self._failed_consults_by_scope[scope] = (
                self._failed_consults_by_scope.get(scope, 0) + 1
            )
        if verdict != "REDO":
            return await self._finish(ctx, turn)

        # REDO: feed the optimized plan back and re-invoke so the executor keeps
        # working instead of stopping. The session is now reviewed, so the redo
        # turn (and everything after it) is plain passthrough.
        # Plain-string assistant/user turns are valid on both wires, so the
        # feedback shape needs no dialect. The prefix is config-tunable
        # (``redo_feedback_prefix``) for per-executor-family steering.
        # The gated turn is discarded (the client never sees it) — record its
        # usage into the classifier bucket and the routing log so the run's
        # cost output prices it, mirroring the tool_call strategy's handling
        # of proxy-internal turns.
        await self._record_discarded_turn(ctx, turn)
        # The assistant echo prefers visible text, then the model's own
        # reasoning (its generated tokens, upstream-only), then "" — an empty
        # echo both risks strict-endpoint rejection and gives the executor a
        # void to continue from.
        redo_messages = [
            *messages,
            {"role": "assistant", "content": turn.content or turn.reasoning_text or ""},
            {"role": "user", "content": self._config.redo_feedback_prefix + plan},
        ]
        redo_body = {**body, "messages": redo_messages}
        redo_request = request_with_type(self._request_type_name, redo_body)
        return await self._passthrough(ctx, redo_request)

    def _budget_scope(self, ctx: ProxyContext) -> str:
        """Review-budget key: the caller's session header, else instance-wide."""
        metadata = ctx.metadata.get(CTX_REQUEST_METADATA)
        session_id = getattr(metadata, "session_id", None)
        return f"client:{session_id}" if session_id else _INSTANCE_SCOPE

    def _scope_exhausted(self, scope: str) -> bool:
        """True when a scope has no review budget or too many failed consults."""
        return (
            self._reviews_by_scope.get(scope, 0) >= self._config.max_reviews
            or self._failed_consults_by_scope.get(scope, 0)
            >= _MAX_FAILED_CONSULTS_PER_SCOPE
        )

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
            events, has_tool_use, content, usage = await _consume_anthropic_stream(
                response.stream
            )
            return _ExecTurn(
                has_tool_use=has_tool_use,
                content=content,
                latency_ms=latency_ms,
                stream_events=events,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
            )
        if response.response_type == ChatResponseType.OPENAI_STREAM:
            events, message, usage = await _consume_openai_stream(response.stream)
            return _ExecTurn(
                has_tool_use=bool(message.get("tool_calls")),
                content=message.get("content") or None,
                latency_ms=latency_ms,
                stream_events=events,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_text=message.get("reasoning_content") or None,
            )
        body = response.to_body()
        reasoning_text = None
        if self._is_openai:
            has_tool_use, content = _openai_completion_tool_use(body)
            reasoning_text = _openai_completion_reasoning(body)
        else:
            has_tool_use, content = _completion_tool_use(body)
        usage = _completion_usage(body, is_openai=self._is_openai)
        return _ExecTurn(
            has_tool_use=has_tool_use,
            content=content,
            latency_ms=latency_ms,
            completion_body=body,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cached_tokens=usage["cached_tokens"],
            reasoning_text=reasoning_text,
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
            if self._is_openai:
                return ChatResponse.openai_stream(
                    ResponseStream(_replay_events(turn.stream_events))
                )
            return ChatResponse.anthropic_stream(
                AnthropicResponseStream(_replay_events(turn.stream_events))
            )
        if self._is_openai:
            return ChatResponse.openai_completion(turn.completion_body)
        return ChatResponse.anthropic_completion(turn.completion_body)

    async def _stamp(self, ctx: ProxyContext, latency_ms: float) -> None:
        ctx.selected_model = self._config.executor.model
        ctx.backend_call_latency_ms = latency_ms
        if self._stats is not None:
            await self._stats.record_success(self._config.executor.model, latency_ms)

    async def _record_discarded_turn(self, ctx: ProxyContext, turn: _ExecTurn) -> None:
        """Price a REDO-discarded executor turn into the classifier bucket + routing log."""
        if self._stats is not None:
            await self._stats.record_classifier_usage(
                model=self._config.executor.model,
                prompt_tokens=turn.input_tokens,
                completion_tokens=turn.output_tokens,
                cached_tokens=turn.cached_tokens,
                latency_ms=turn.latency_ms,
            )
        _emit_routing_usage(
            ctx,
            model=self._config.executor.model,
            tier="review_gate_discarded",
            prompt_tokens=turn.input_tokens,
            completion_tokens=turn.output_tokens,
            cached_tokens=turn.cached_tokens,
        )

    # ------------------------------------------------------------------
    # Advisor review
    # ------------------------------------------------------------------

    async def _review(
        self,
        messages: list[dict[str, Any]],
        terminal_content: str | None,
        ctx: ProxyContext,
    ) -> tuple[str, str, bool]:
        """Consult the advisor once; return ``(verdict, plan, consulted)``.

        ``verdict`` is ``"APPROVE"`` or ``"REDO"``. On a fail-open advisor error
        or an unparseable reply, defaults to ``APPROVE`` (do not disrupt a
        possibly-correct turn). ``consulted`` is False when the advisor call
        itself failed, so the caller can refund the review budget.
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
                await self._stats.record_classifier_error(self._config.advisor.model)
            _audit_review(verdict="APPROVE", error=str(exc), usage=None,
                          latency_ms=(time.monotonic() - started) * 1000.0)
            return "APPROVE", "", False
        latency_ms = (time.monotonic() - started) * 1000.0
        verdict, plan = _parse_verdict(text)
        # Record the advisor review's token usage so the run's own cost output
        # accounts for the advisor, not just the executor: into the classifier
        # bucket (the advisor review is a secondary-model consult, like the
        # escalation judge — its cost rolls into ``cost_estimate.total_cost``)
        # AND into the routing log, so per-session stats
        # (``/v1/routing/session-stats``) attribute it to the caller's session.
        # Recorded even for an unparseable reply — the tokens were spent.
        tokens = _advisor_usage(usage)
        if self._stats is not None:
            await self._stats.record_classifier_usage(
                model=self._config.advisor.model,
                prompt_tokens=tokens["prompt_tokens"],
                completion_tokens=tokens["completion_tokens"],
                cached_tokens=tokens["cached_tokens"],
                latency_ms=latency_ms,
            )
        _emit_routing_usage(
            ctx,
            model=self._config.advisor.model,
            tier="advisor_review",
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            cached_tokens=tokens["cached_tokens"],
            cache_creation_tokens=tokens["cache_creation_tokens"],
        )
        if verdict == "":
            # No leading APPROVE/REDO in the reply: treat as a failed consult
            # (caller refunds the budget) and pass the turn through unchanged.
            _audit_review(
                verdict="UNPARSEABLE", error=None, usage=usage,
                latency_ms=latency_ms, reply_head=text,
            )
            return "APPROVE", "", False
        _audit_review(
            verdict=verdict, error=None, usage=usage, latency_ms=latency_ms,
            reply_head=text,
        )
        return verdict, plan, True

    def _serialize_transcript(
        self, messages: list[dict[str, Any]], terminal_content: str | None,
    ) -> str:
        """Serialize the conversation + the executor's terminal turn for review.

        Over ``transcript_max_chars``, the MIDDLE is dropped: the head keeps
        the task statement, the tail keeps the executor's most recent work —
        the part a completeness review is actually about. (Head-only
        truncation left the reviewer judging "genuinely done?" without ever
        seeing the recent evidence.)
        """
        text = json.dumps(messages, default=str, ensure_ascii=False)
        cap = self._config.transcript_max_chars
        if len(text) > cap:
            head = cap // 4
            tail = cap - head
            text = (
                text[:head]
                + "\n...<middle of the conversation truncated>...\n"
                + text[-tail:]
            )
        tail_turn = terminal_content or "(no text)"
        return (
            f"Conversation so far (JSON):\n\n{text}\n\n"
            f"The executor's latest turn (a plan, or its claim the task is done):\n{tail_turn}"
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


async def _consume_anthropic_stream(
    stream: Any,
) -> tuple[list[Any], bool, str | None, dict[str, int]]:
    """Buffer an Anthropic stream; return (events, has_tool_use, assistant_text, usage)."""
    events: list[Any] = []
    has_tool_use = False
    text_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    async for event in stream:
        events.append(event)
        etype = _ev(event, "type")
        if etype == "message_start":
            start_usage = _ev(_ev(event, "message"), "usage") or {}
            usage["input_tokens"] = int(start_usage.get("input_tokens") or 0)
            usage["cached_tokens"] = int(start_usage.get("cache_read_input_tokens") or 0)
        elif etype == "content_block_start":
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
            delta_usage = _ev(event, "usage") or {}
            usage["output_tokens"] = int(delta_usage.get("output_tokens") or 0)
    return events, has_tool_use, ("".join(text_parts) or None), usage


def _completion_usage(body: Any, *, is_openai: bool) -> dict[str, int]:
    """Read ``_ExecTurn`` token counts from a non-streamed completion body."""
    usage = body.get("usage") if isinstance(body, dict) else None
    prompt_tokens, completion_tokens = _usage_tokens(usage)
    details = usage if isinstance(usage, dict) else {}
    if is_openai:
        cached = (details.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    else:
        cached = details.get("cache_read_input_tokens") or 0
    return {
        "input_tokens": prompt_tokens or 0,
        "output_tokens": completion_tokens or 0,
        "cached_tokens": int(cached),
    }


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


def _openai_completion_tool_use(body: Any) -> tuple[bool, str | None]:
    """Read (has_tool_use, assistant_text) from an OpenAI chat.completion body.

    Detection is by ``tool_calls`` presence with ``finish_reason`` as a
    fallback — some OSS servers mislabel tool-call turns as ``stop``.
    """
    if not isinstance(body, dict):
        return False, None
    choices = body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    has_tool_use = bool(message.get("tool_calls")) or choice.get("finish_reason") == "tool_calls"
    return has_tool_use, (message.get("content") or None)


def _openai_completion_reasoning(body: Any) -> str | None:
    """Read ``reasoning_content`` from an OpenAI chat.completion body, if any."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    return message.get("reasoning_content") or None


def _gate_request_type(fmt: BackendFormat) -> str:
    """Map a resolved executor format to its ``request_with_type`` discriminator."""
    if fmt == BackendFormat.ANTHROPIC:
        return "anthropic"
    if fmt == BackendFormat.OPENAI:
        return "openai_chat"
    if fmt == BackendFormat.RESPONSES:
        # Backstop for format: auto resolving to a Responses endpoint; the
        # config validator rejects an explicit responses format earlier.
        raise ValueError(
            "the advisor strategies are Chat-shaped and do not support "
            "Responses executors; use format 'openai' or 'anthropic'"
        )
    raise ValueError(
        f"advisor executor format {fmt!r} must be resolved before constructing "
        "the backend (pin format: 'openai' or 'anthropic' when supplying "
        "executor_backend)"
    )


async def _consume_openai_stream(
    stream: Any,
) -> tuple[list[Any], dict[str, Any], dict[str, int]]:
    """Buffer an OpenAI Chat stream; reassemble the assistant message and usage.

    Events are the ``chat.completion.chunk`` dicts the native backend's SSE
    parser yields (``[DONE]`` is consumed upstream and never appears; the
    backend force-injects ``stream_options.include_usage`` so a final usage
    chunk normally arrives). ``delta.tool_calls`` fragments merge by ``index``:
    non-empty ``id``/``name`` replace, ``arguments`` fragments concatenate.
    Shared by both advisor strategies.
    """
    events: list[Any] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    slots: dict[int, dict[str, str]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    async for event in stream:
        events.append(event)
        chunk_usage = _ev(event, "usage")
        if isinstance(chunk_usage, dict):
            usage["input_tokens"] = int(chunk_usage.get("prompt_tokens") or 0)
            usage["output_tokens"] = int(chunk_usage.get("completion_tokens") or 0)
            details = chunk_usage.get("prompt_tokens_details") or {}
            usage["cached_tokens"] = int(details.get("cached_tokens") or 0)
        choices = _ev(event, "choices") or []
        delta = _ev(choices[0], "delta") if choices else None
        if delta is None:
            continue
        piece = _ev(delta, "content")
        if isinstance(piece, str):
            text_parts.append(piece)
        # Reasoning models (nemotron on vLLM/NIM) can emit turns whose ONLY
        # output is reasoning_content; keep it so the review gate has
        # something to show the advisor when visible content is empty.
        reasoning_piece = _ev(delta, "reasoning_content")
        if isinstance(reasoning_piece, str):
            reasoning_parts.append(reasoning_piece)
        for fragment in _ev(delta, "tool_calls") or []:
            index = int(_ev(fragment, "index") or 0)
            slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
            fragment_id = _ev(fragment, "id")
            if isinstance(fragment_id, str) and fragment_id:
                slot["id"] = fragment_id
            function = _ev(fragment, "function") or {}
            name = _ev(function, "name")
            if isinstance(name, str) and name:
                slot["name"] = name
            arguments = _ev(function, "arguments")
            if isinstance(arguments, str):
                slot["arguments"] += arguments

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if slots:
        message["tool_calls"] = [
            {
                # A missing id (some OSS servers omit it in deltas) gets a
                # synthesized one so the tool result can reference it.
                "id": slot["id"] or f"call_switchyard_{index}",
                "type": "function",
                "function": {
                    "name": slot["name"],
                    # Empty arguments become "{}" so strict endpoints accept
                    # the replayed history.
                    "arguments": slot["arguments"] or "{}",
                },
            }
            for index, slot in sorted(slots.items())
        ]
    return events, message, usage


def _ev(event: Any, key: str) -> Any:
    """Read a field from a stream event (dict from Rust, or an SDK object)."""
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _with_length_line(
    messages: list[dict[str, Any]], line: str,
) -> list[dict[str, Any]]:
    """Append a line of text to the **first** user message.

    The doc suggests the latest user message, but the client never sees this
    injection, so re-injecting into each turn's newest message would shift the
    upstream cache prefix every turn. The first user message is constant across
    a session, keeping the prefix stable; the advisor still reads the line via
    the forwarded transcript. The list branch emits ``{"type": "text", ...}``
    parts, valid on both the Anthropic and OpenAI wires. Shared by the advisor
    length line, and by ``seed_plan_advice`` for the seeded upfront plan.
    """
    msgs = [dict(m) for m in messages]
    for msg in msgs:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [*content, {"type": "text", "text": line}]
        elif isinstance(content, str) or content is None:
            msg["content"] = f"{content or ''}\n\n{line}".lstrip()
        break
    return msgs


def _seed_transcript(messages: list[dict[str, Any]], cap: int) -> str:
    """Serialize the session-opening messages for the seed consult."""
    text = json.dumps(messages, default=str, ensure_ascii=False)
    if len(text) > cap:
        text = text[: cap - 16] + "...<truncated>"
    return (
        f"The task the executor is about to start (JSON):\n\n{text}\n\n"
        "The executor has not begun yet. Review the task and give your best "
        "upfront plan: the approach, the pitfalls to avoid, and the first "
        "concrete steps."
    )


async def _seed_advice_for(
    cache: dict[str, str],
    session: str,
    messages: list[dict[str, Any]],
    *,
    caller: AdvisorCaller,
    config: AdvisorConfig,
    stats: StatsAccumulator | None,
    ctx: ProxyContext | None = None,
) -> str:
    """Per-session seed advice for ``seed_plan_advice`` (both strategies).

    The advisor is consulted once per session, at a request that opens the
    conversation (no assistant turns yet); the advice is cached so every later
    turn of the session re-injects the identical text (stable cache prefix).
    A session first seen mid-conversation (e.g. after a proxy restart) is
    cached as unseeded — injecting new advice mid-session would shift the
    upstream prefix. Fail-open: a failed consult caches "" (no retry storm).
    """
    cached = cache.get(session)
    if cached is not None:
        return cached
    if any(m.get("role") == "assistant" for m in messages):
        cache[session] = ""
        return ""
    advice = await _fetch_seed_advice(
        caller=caller, config=config, messages=messages, stats=stats, ctx=ctx,
    )
    cache[session] = advice
    return advice


async def _fetch_seed_advice(
    *,
    caller: AdvisorCaller,
    config: AdvisorConfig,
    messages: list[dict[str, Any]],
    stats: StatsAccumulator | None,
    ctx: ProxyContext | None = None,
) -> str:
    """Consult the advisor for an upfront plan; "" on fail-open failure."""
    transcript = _seed_transcript(messages, config.transcript_max_chars)
    started = time.monotonic()
    try:
        advice, usage = await caller.advise(
            system=config.advisor_system_prompt, transcript=transcript,
        )
    except Exception as exc:
        if not config.fail_open:
            raise
        log.warning("seed_plan_advice: advisor call failed; proceeding unseeded: %s", exc)
        if stats is not None:
            await stats.record_classifier_error(config.advisor.model)
        _audit_seed(error=str(exc), usage=None,
                    latency_ms=(time.monotonic() - started) * 1000.0)
        return ""
    latency_ms = (time.monotonic() - started) * 1000.0
    tokens = _advisor_usage(usage)
    if stats is not None:
        await stats.record_classifier_usage(
            model=config.advisor.model,
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            cached_tokens=tokens["cached_tokens"],
            latency_ms=latency_ms,
        )
    if ctx is not None:
        _emit_routing_usage(
            ctx,
            model=config.advisor.model,
            tier="advisor_seed",
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            cached_tokens=tokens["cached_tokens"],
            cache_creation_tokens=tokens["cache_creation_tokens"],
        )
    _audit_seed(error=None, usage=usage, latency_ms=latency_ms)
    return advice.strip()


def _audit_seed(*, error: str | None, usage: Any, latency_ms: float) -> None:
    """Emit a one-line ``advisor_seed=...`` audit record to stderr."""
    payload: dict[str, Any] = {
        "advisor_seed": True,
        "error": error,
        "latency_ms": round(latency_ms, 1),
    }
    _merge_audit_tokens(payload, usage)
    sys.stderr.write(f"advisor_seed={json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def _count_tool_results(messages: list[dict[str, Any]]) -> int:
    """Count tool results across both wires (OpenAI ``role: tool`` messages,
    Anthropic ``tool_result`` blocks in user messages)."""
    n = 0
    for m in messages:
        if m.get("role") == "tool":
            n += 1
        elif m.get("role") == "user" and isinstance(m.get("content"), list):
            n += sum(
                1 for b in m["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            )
    return n


def _session_key(system: Any, messages: list[dict[str, Any]]) -> str:
    """Stable per-session key: hash of the cache-stable system prefix + first user message.

    The system prompt is *not* constant across a session on real agent harnesses:
    Claude Code re-renders volatile context (reminders, todo state, environment)
    into it on every request. Hashing the whole thing minted a fresh key per turn,
    silently resetting the ``max_reviews`` budget — observed as 87 reviews on a
    single Terminal-Bench task instead of the configured 2.

    Only the portion up to and including the client's last ``cache_control``
    breakpoint is used: that prefix is stable by construction, because the client
    is asserting it is byte-identical across turns for prompt caching. Anything
    after the final breakpoint is volatile by definition and must not affect
    session identity. Clients that set no breakpoint fall back to the first user
    message alone, which is the stable task statement.
    """
    parts: list[str] = ["S:" + _cache_stable_system_text(system)]
    for m in messages:
        if m.get("role") == "user":
            parts.append("U:" + _blocks_text(m.get("content")))
            break
    return hashlib.sha256("\n".join(parts).encode("utf-8", "ignore")).hexdigest()


def _cache_stable_system_text(system: Any) -> str:
    """System text through the last ``cache_control`` breakpoint (stable prefix).

    Returns "" when the client marks no breakpoint, so session identity then rests
    on the first user message rather than on volatile per-turn system content.
    """
    if isinstance(system, str):
        # A bare string carries no breakpoint information; it is echoed verbatim
        # by clients that do not use structured system blocks.
        return system
    if not isinstance(system, list):
        return ""
    last_breakpoint = -1
    for index, block in enumerate(system):
        if isinstance(block, dict) and block.get("cache_control"):
            last_breakpoint = index
    if last_breakpoint < 0:
        return ""
    return _blocks_text(system[: last_breakpoint + 1])


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


#: Anchored verdict match: markdown/quote wrappers and a "Verdict:"-style
#: label may precede the verdict word, but PROSE may not — an unanchored
#: window turned "I cannot approve this — REDO: run the tests" into APPROVE
#: (first case-insensitive token wins). Tolerated prefixes: whitespace,
#: ``*_#>"'([`` characters, and a short label ending in ``:``.
_VERDICT_RE = re.compile(
    r"^[\s*_#>\"'\(\[`]*(?:(?:final\s+)?verdict\s*:\s*[\s*_#>\"'\(\[`]*)?(APPROVE|REDO)\b",
    re.IGNORECASE,
)


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse the reviewer reply into (verdict, plan).

    Returns ``("", "")`` when no leading verdict is found — the caller treats
    that as a failed consult (budget refunded) rather than a silent APPROVE,
    so a hedged or malformed reply cannot burn the review budget.
    """
    stripped = (text or "").strip()
    match = _VERDICT_RE.match(stripped)
    if match is None:
        return "", ""
    if match.group(1).upper() == "APPROVE":
        return "APPROVE", ""
    plan = stripped[match.end():].lstrip(" *_:\n-").strip()
    return "REDO", plan or stripped


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


def _advisor_usage(usage: Any) -> dict[str, int]:
    """Token fields for an advisor consult, with cache buckets folded in.

    Anthropic-shaped usage reports cache reads/writes as SIBLINGS of
    ``input_tokens`` — and some gateways (NVIDIA Inference Hub's bedrock
    routes) auto-cache large prompts server-side even when the caller set no
    ``cache_control``, so a consult's real input lands almost entirely in
    ``cache_creation_input_tokens`` while ``input_tokens`` reads as ~2.
    ``prompt_tokens`` here is the inclusive total, matching the routing-log
    processor's accounting; OpenAI ``prompt_tokens`` are already inclusive.
    """

    def get(container: Any, name: str) -> int:
        value = (
            container.get(name) if isinstance(container, dict)
            else getattr(container, name, None)
        )
        return int(value) if value is not None else 0

    input_tokens, output_tokens = _usage_tokens(usage)
    cache_read = get(usage, "cache_read_input_tokens")
    cache_creation = get(usage, "cache_creation_input_tokens")
    if cache_read or cache_creation:
        prompt = (input_tokens or 0) + cache_read + cache_creation
        cached = cache_read
    else:
        prompt = input_tokens or 0
        details = (
            usage.get("prompt_tokens_details") if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        cached = get(details, "cached_tokens") if details is not None else 0
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_creation_tokens": cache_creation,
        "completion_tokens": output_tokens or 0,
    }


def _audit_review(
    *,
    verdict: str,
    error: str | None,
    usage: Any,
    latency_ms: float,
    reply_head: str | None = None,
) -> None:
    """Emit a one-line ``advisor_review=...`` audit record to stderr.

    ``reply_head`` carries the raw reply's first characters so a verdict the
    parser read differently than the reviewer intended is visible in the logs
    (the parsed verdict alone hides misparses).
    """
    payload: dict[str, Any] = {
        "advisor_review": True,
        "verdict": verdict,
        "error": error,
        "latency_ms": round(latency_ms, 1),
    }
    if reply_head is not None:
        payload["reply_head"] = reply_head.strip()[:160]
    _merge_audit_tokens(payload, usage)
    sys.stderr.write(f"advisor_review={json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def _merge_audit_tokens(payload: dict[str, Any], usage: Any) -> None:
    """Add cache-inclusive token fields to an audit payload (no-op when None)."""
    if usage is None:
        return
    tokens = _advisor_usage(usage)
    payload["prompt_tokens"] = tokens["prompt_tokens"]
    payload["completion_tokens"] = tokens["completion_tokens"]
    if tokens["cache_creation_tokens"]:
        payload["cache_creation_tokens"] = tokens["cache_creation_tokens"]
    if tokens["cached_tokens"]:
        payload["cached_tokens"] = tokens["cached_tokens"]


def _emit_routing_usage(
    ctx: ProxyContext,
    *,
    model: str,
    tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Append a proxy-internal usage record to the routing log, if one is active.

    The routing-log response processor only sees the chain's terminal
    response; advisor consults and REDO-discarded executor turns happen inside
    the backend and would otherwise be invisible to
    ``/v1/routing/session-stats`` — and with it, per-model cost attribution.
    """
    from switchyard.lib.processors.routing_log_response_processor import (
        emit_auxiliary_record,
    )

    metadata = ctx.metadata.get(CTX_REQUEST_METADATA)
    emit_auxiliary_record(
        session_id=getattr(metadata, "session_id", None),
        task=getattr(metadata, "task", None),
        model=model,
        tier=tier,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
        completion_tokens=completion_tokens,
    )


__all__ = ["AdvisorCaller", "AdvisorLoopBackend"]
