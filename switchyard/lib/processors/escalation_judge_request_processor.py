# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory judge for the escalation router: weak until in trouble, then strong."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections import OrderedDict
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.conversation_turn import conversation_turn_number
from switchyard.lib.llm_client import OpenAILLMClient
from switchyard.lib.processors.llm_classifier.request_processor import (
    LLMClassifierClient,
    OpenAIChatLLMClassifierClient,
    _strip_markdown_fence,
    _trim_messages,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_affinity import (
    CTX_SESSION_KEY,
    SessionAffinity,
    resolve_session_key,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: ``ProxyContext.metadata`` key holding the audit record of the judge's
#: decision for this turn: ``{"escalate", "reason", "turn", "source",
#: "session", "streak", "confirmed"}`` (the last three only on judged turns).
CTX_ESCALATION_VERDICT = "_escalation_verdict"

#: LRU cap on the consecutive-escalate streak store.
_MAX_STREAK_SESSIONS = 10_000

#: Tier labels stamped into ``CTX_DETERMINISTIC_ROUTING_TIER``; must match the
#: labels the profile registers on ``DeterministicRoutingLLMBackend``.
TIER_STRONG = "strong"
TIER_WEAK = "weak"


def _load_system_prompt() -> str:
    """Read the default judge system prompt from the prompts/ package-data file.

    The file carries the benchmarked line-of-record prompt byte-exact; its
    content is hash-pinned in tests, so edits require deliberate re-validation.
    """
    return (
        files("switchyard.lib.processors.prompts")
        .joinpath("escalation_judge.md")
        .read_text(encoding="utf-8")
    )


ESCALATION_JUDGE_SYSTEM_PROMPT: str = _load_system_prompt()


class EscalationVerdict(BaseModel):
    """Binary judge verdict: escalate to the strong tier or stay on weak."""

    model_config = ConfigDict(frozen=True)

    escalate: bool
    reason: str = ""


class EscalationJudgeConfig(BaseModel):
    """Configuration for :class:`EscalationJudgeRequestProcessor`.

    The judge call is OpenAI-chat-compatible; tests inject any object
    implementing the classifier's ``LLMClassifierClient`` protocol.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = Field(default=5.0, gt=0.0)
    """Judge wall-clock ceiling. The judge sits on the request path of every
    pre-escalation turn; a slow judge taxes the whole session, so keep this
    tight — any failure or timeout fails open to the weak tier."""

    max_completion_tokens: int = Field(default=128, ge=16)
    system_prompt: str = Field(default=ESCALATION_JUDGE_SYSTEM_PROMPT, min_length=1)
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    disable_reasoning: bool = True
    extra_headers: dict[str, str] | None = None

    dump_verdicts_to_stderr: bool = True
    """Emit one ``escalation_verdict={...}`` JSON line to ``sys.stderr`` per
    judge call (verdict or fail-open).

    Written directly (not via the logging module) so it lands in the
    benchmark server's captured log regardless of uvicorn's logger config —
    mirrors :attr:`LLMClassifierConfig.dump_signals_to_stderr`. Grep
    ``escalation_verdict=`` to reconstruct per-turn judge decisions and
    escalation timing for a run. Disable for callers that share stderr
    with an interactive TUI."""

    min_judge_turn: int = Field(default=3, ge=1)
    """First conversation turn on which the judge runs. Earlier turns have no
    trajectory to judge and always route weak."""

    escalate_confirmations: int = Field(default=1, ge=1)
    """Consecutive ``escalate`` verdicts required before the latch fires.

    ``1`` (default) pins on the first escalate verdict. ``2`` requires the
    judge to confirm on the next judged turn, filtering one-shot eager
    verdicts (a single failed command misread as a pattern) while keeping
    recall on real trouble, which by definition persists across turns. A
    non-escalate verdict resets the streak; a fail-open leaves it
    unchanged (no evidence either way)."""

    confirmation_window: int = Field(default=1, ge=1)
    """How many judged turns an escalate verdict stays live for
    confirmation purposes.

    ``1`` (default) keeps the strict-consecutive behaviour: any decline
    resets the streak. ``N > 1`` lets a later escalate verdict confirm an
    earlier one across up to ``N - 1`` intervening declines — trouble that
    keeps *recurring* is trouble, even when a quiet turn separates the
    flare-ups."""

    recent_turn_window: int = Field(default=14, ge=1)
    """Trailing messages shown to the judge on top of the anchors. Wider than
    the classifier's default because loop detection needs to see the repeats:
    a cycle longer than the window is invisible."""

    max_request_chars: int = Field(default=12_000, ge=1_000)
    """Cap on the assembled judge transcript. When exceeded, the *oldest*
    window messages are dropped first — for a trajectory judge the newest
    evidence is strictly the most valuable."""

    system_chars: int = Field(default=1_000, ge=100)
    """Per-message cap for system/developer anchors. Coding-agent harnesses
    (Claude Code) inject very large boilerplate system prompts with no
    trajectory signal; without this cap they would crowd out the window."""

    first_user_chars: int = Field(default=2_000, ge=100)
    """Cap for the first user message — the task statement the judge needs
    for drift detection, so it gets the most generous anchor budget."""

    window_message_chars: int = Field(default=300, ge=50)
    """Per-message cap inside the trailing window. Error signatures and
    command shapes survive this easily; full file dumps do not need to."""


class EscalationJudgeRequestProcessor:
    """Route weak until an LLM judge says the run is in trouble, then latch strong.

    Per-turn policy (the latch): a pinned conversation routes strong with no
    judge call; an unpinned one routes weak, and from ``min_judge_turn`` on the
    judge reads a condensed transcript. An ``escalate`` verdict pins the
    conversation to the strong tier via :class:`SessionAffinity` — one-way for
    the rest of the task. Weak is never pinned, and any judge failure fails
    open to weak without pinning, so an outage costs quality risk, never money.
    """

    def __init__(
        self,
        config: EscalationJudgeConfig,
        *,
        affinity: SessionAffinity,
        client: LLMClassifierClient | None = None,
        session_key_depth: int = 0,
        stats_accumulator: StatsAccumulator | None = None,
    ) -> None:
        if session_key_depth < 0:
            raise ValueError("session_key_depth must be >= 0")
        self._config = config
        self._affinity = affinity
        self._session_key_depth = session_key_depth
        # Also attached post-construction by the profile chain's
        # ``with_runtime_components`` (same hook as the LLM classifier).
        self._stats_accumulator = stats_accumulator
        # Per-conversation (streak, declines_since_last_fire), only consulted
        # when ``escalate_confirmations > 1``. LRU-capped alongside the
        # affinity store so a long-lived server cannot grow it unbounded.
        self._escalate_streaks: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._client = client or OpenAIChatLLMClassifierClient(
            OpenAILLMClient(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_s,
                # Fail-open to weak is the retry surface; let the SDK fail fast.
                max_retries=0,
            ),
            signal_schema=EscalationVerdict,
            structured_output_mode=config.structured_output_mode,
            max_completion_tokens=config.max_completion_tokens,
            disable_reasoning=config.disable_reasoning,
            extra_headers=config.extra_headers,
        )

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        """Stamp the tier for this turn and maybe escalate. Leaves the request unchanged."""
        turn = conversation_turn_number(request)
        # With a deep key, affinity is untouchable until the key prefix is
        # complete — hashing a shorter prefix would produce a key that later
        # turns of the same conversation no longer match.
        affinity_ready = self._seed_session_key(ctx, request)

        if affinity_ready:
            pinned = await self._affinity.pinned(ctx, request)
            if pinned == TIER_STRONG:
                ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = TIER_STRONG
                ctx.metadata[CTX_ESCALATION_VERDICT] = {
                    "escalate": True,
                    "reason": "",
                    "turn": turn,
                    "source": "pinned",
                }
                return request

        ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = TIER_WEAK
        if not affinity_ready or turn < self._config.min_judge_turn:
            return request

        # Present when session_key_depth > 0; lets verdict-dump consumers
        # group per-turn judge decisions by conversation.
        session = ctx.metadata.get(CTX_SESSION_KEY)
        summary = _summarize_for_judge(request, turn=turn, config=self._config)
        started_at = time.perf_counter()
        try:
            completion = await self._client.classify(
                model=self._config.model,
                system_prompt=self._config.system_prompt,
                request_summary=summary,
            )
            verdict = EscalationVerdict.model_validate_json(
                _strip_markdown_fence(completion.content),
            )
        except Exception as exc:
            # Fail open to the current (weak) tier and never pin: a judge
            # outage must not silently burn strong-model tokens.
            log.warning(
                "EscalationJudgeRequestProcessor: judge failed; staying on weak tier: %s",
                exc,
            )
            if self._stats_accumulator is not None:
                # Record the failure into the classifier bucket so
                # ``/v1/routing/stats`` exposes the judge fail-open rate;
                # a silently failing judge reads as "router never
                # escalates" to the benchmark observer.
                await self._stats_accumulator.record_classifier_error(self._config.model)
            ctx.metadata[CTX_ESCALATION_VERDICT] = {
                "escalate": False,
                "reason": f"fail_open: {str(exc)[:200]}",
                "turn": turn,
                "source": "fail_open",
                "session": session,
            }
            self._dump_verdict(
                ctx.metadata[CTX_ESCALATION_VERDICT],
                latency_ms=(time.perf_counter() - started_at) * 1000,
                request=request,
            )
            return request

        if self._stats_accumulator is not None:
            await self._record_judge_call(
                usage=completion.usage,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )

        if verdict.escalate:
            streak = self._bump_streak(ctx, request)
            confirmed = streak >= self._config.escalate_confirmations
        else:
            streak = 0
            confirmed = False
            self._reset_streak(ctx, request)

        ctx.metadata[CTX_ESCALATION_VERDICT] = {
            "escalate": verdict.escalate,
            "reason": verdict.reason,
            "turn": turn,
            "source": "judge",
            "session": session,
            "streak": streak,
            "confirmed": confirmed,
        }
        self._dump_verdict(
            ctx.metadata[CTX_ESCALATION_VERDICT],
            latency_ms=(time.perf_counter() - started_at) * 1000,
            request=request,
        )
        if confirmed:
            log.info(
                "EscalationJudgeRequestProcessor: escalating to strong tier "
                "(turn %d): %s",
                turn,
                verdict.reason,
            )
            await self._affinity.pin(ctx, request, TIER_STRONG)
            ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = TIER_STRONG
        return request

    def _streak_key(self, ctx: ProxyContext, request: ChatRequest) -> str:
        """Conversation key for streak bookkeeping (seeded deep key or Rust default)."""
        return resolve_session_key(ctx, request)

    def _bump_streak(self, ctx: ProxyContext, request: ChatRequest) -> int:
        """Record one escalate verdict for this conversation; return the streak."""
        if self._config.escalate_confirmations <= 1:
            # Single-verdict latch: no bookkeeping needed, streak is per-call.
            return 1
        key = self._streak_key(ctx, request)
        streak, _ = self._escalate_streaks.get(key, (0, 0))
        streak += 1
        self._escalate_streaks[key] = (streak, 0)
        self._escalate_streaks.move_to_end(key)
        while len(self._escalate_streaks) > _MAX_STREAK_SESSIONS:
            self._escalate_streaks.popitem(last=False)
        return streak

    def _reset_streak(self, ctx: ProxyContext, request: ChatRequest) -> None:
        """Register a non-escalate verdict against the streak.

        With ``confirmation_window == 1`` any decline clears the streak
        (strict-consecutive). A larger window tolerates up to
        ``window - 1`` intervening declines before the streak expires, so
        recurring intermittent trouble can still confirm.
        """
        if self._config.escalate_confirmations <= 1:
            return
        key = self._streak_key(ctx, request)
        entry = self._escalate_streaks.get(key)
        if entry is None:
            return
        streak, declines = entry
        declines += 1
        if declines >= self._config.confirmation_window:
            self._escalate_streaks.pop(key, None)
        else:
            self._escalate_streaks[key] = (streak, declines)
            self._escalate_streaks.move_to_end(key)

    def _dump_verdict(
        self,
        verdict: dict[str, Any],
        *,
        latency_ms: float,
        request: ChatRequest | None = None,
    ) -> None:
        """Write one ``escalation_verdict={...}`` JSON line to stderr.

        Direct stderr (not the logging module) so benchmark server logs
        capture it — see :attr:`EscalationJudgeConfig.dump_verdicts_to_stderr`.
        Includes a short first-user-message snippet (``task_hint``) so
        offline analysis can attribute per-turn verdicts to tasks exactly,
        instead of guessing from overlapping trial time windows.
        """
        if not self._config.dump_verdicts_to_stderr:
            return
        payload = dict(verdict)
        payload["latency_ms"] = round(latency_ms, 1)
        if request is not None:
            hint = _first_user_text(request)
            if hint:
                payload["task_hint"] = hint[:48]
        sys.stderr.write(f"escalation_verdict={json.dumps(payload, sort_keys=True)}\n")
        sys.stderr.flush()

    async def _record_judge_call(self, *, usage: Any, latency_ms: float) -> None:
        """Record judge token spend and latency into the classifier stats bucket.

        The judge is a routing-overhead call, exactly like the LLM
        classifier: keeping it in the classifier bucket stops its spend
        from merging with a same-named tier model and keeps
        ``routing_stats_final.json`` cost math honest for the
        escalation-router benchmark.
        """
        assert self._stats_accumulator is not None
        prompt = 0
        completion = 0
        cached = 0
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                cached = getattr(ptd, "cached_tokens", 0) or 0
        await self._stats_accumulator.record_classifier_usage(
            model=self._config.model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            latency_ms=latency_ms,
        )

    def _seed_session_key(self, ctx: ProxyContext, request: ChatRequest) -> bool:
        """Seed the deep session key when configured; return whether affinity is usable.

        ``session_key_depth == 0`` keeps the default Rust key (system + first
        user message) — always usable. Depth ``N`` extends the hash with the
        first ``N`` post-first-user messages so repeated runs of the same task
        diverge via early model responses; until those messages exist, the key
        is not yet stable and affinity must not be read or written.
        """
        if self._session_key_depth == 0:
            return True
        key = _deep_session_key(request, self._session_key_depth)
        if key is None:
            return False
        ctx.metadata[CTX_SESSION_KEY] = key
        return True


def _first_user_text(request: ChatRequest) -> str:
    """Flatten the first user message's task text for verdict dumps.

    Harness wrappers (``<system-reminder>``/``<environment_context>``
    blocks injected ahead of the actual task statement) are skipped so the
    snippet identifies the task rather than repeating boilerplate shared
    by every conversation.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return ""
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = body.get("input")
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            text = _message_text(m).strip()
            while True:
                for tag in ("system-reminder", "environment_context"):
                    if text.startswith(f"<{tag}>"):
                        end = text.find(f"</{tag}>")
                        if end >= 0:
                            text = text[end + len(tag) + 3 :].lstrip()
                            break
                else:
                    return text
    return ""


def _deep_session_key(request: ChatRequest, depth: int) -> str | None:
    """Hash system + first user + first ``depth`` later messages, or ``None`` if short.

    Mirrors the anchor semantics of the Rust ``session_key_from_body`` and
    extends the prefix by ``depth`` messages. The prefix of a conversation
    never changes as it grows, so the key is stable from the moment it exists.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return None
    hasher = hashlib.sha256()
    top_system = body.get("system")
    if isinstance(top_system, str | list):
        hasher.update(_content_text(top_system).encode("utf-8"))
        hasher.update(b"\x00")

    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = body.get("input")
    if not isinstance(messages, list):
        return None

    first_user_seen = False
    tail_taken = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            hasher.update(_message_text(m).encode("utf-8"))
            hasher.update(b"\x00")
        elif not first_user_seen and role == "user":
            first_user_seen = True
            hasher.update(_message_text(m).encode("utf-8"))
            hasher.update(b"\x00")
        elif first_user_seen and tail_taken < depth:
            tail_taken += 1
            hasher.update(_message_text(m).encode("utf-8"))
            hasher.update(b"\x00")
        if first_user_seen and tail_taken >= depth:
            break
    if not first_user_seen or tail_taken < depth:
        return None
    return hasher.hexdigest()[:16]


def _summarize_for_judge(
    request: ChatRequest,
    *,
    turn: int,
    config: EscalationJudgeConfig,
) -> str:
    """Render a compact role-labeled transcript for the judge.

    Anchors (system + first user) are individually capped; the trailing window
    keeps the last ``recent_turn_window`` messages at ``window_message_chars``
    each. A coverage header states how much history is *not* shown so the
    judge can reason about pace. If the global cap still binds, the oldest
    window lines are dropped first.
    """
    body = getattr(request, "body", {})
    messages: list[Any] = []
    if isinstance(body, dict):
        raw = body.get("messages")
        if not isinstance(raw, list):
            raw = body.get("input")
        if isinstance(raw, list):
            messages = raw

    trimmed = _trim_messages(messages, recent_turn_window=config.recent_turn_window)

    anchor_lines: list[str] = []
    window_lines: list[str] = []
    if isinstance(body, dict):
        top_system = body.get("system")
        if isinstance(top_system, str | list):
            anchor_lines.append(
                "[system] " + _truncate(_content_text(top_system), config.system_chars)
            )
    first_user_seen = False
    for m in trimmed:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _message_text(m)
        if role in ("system", "developer"):
            anchor_lines.append(f"[{role}] " + _truncate(text, config.system_chars))
        elif role == "user" and not first_user_seen:
            first_user_seen = True
            anchor_lines.append("[user (task)] " + _truncate(text, config.first_user_chars))
        else:
            label = role or m.get("type") or "message"
            window_lines.append(f"[{label}] " + _truncate(text, config.window_message_chars))

    n_shown = len(window_lines)
    header = (
        f"Conversation turn {turn}; showing the last {n_shown} of "
        f"{len(messages)} messages after the task framing."
    )

    def _assemble() -> str:
        return "\n".join([header, *anchor_lines, *window_lines])

    text = _assemble()
    # Drop oldest window lines first: for a trajectory judge the newest
    # evidence is strictly the most valuable.
    while len(text) > config.max_request_chars and window_lines:
        window_lines.pop(0)
        text = _assemble()
    if len(text) > config.max_request_chars:
        text = text[: config.max_request_chars - 15] + "...<truncated>"
    return text


def _message_text(message: dict[str, Any]) -> str:
    """Flatten a chat message (or Responses item) to plain text, tool calls included."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str | list):
        text = _content_text(content)
        if text:
            parts.append(text)
    # OpenAI chat tool calls: the command shapes the judge needs for loop detection.
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if isinstance(fn, dict):
                parts.append(f"tool_call {fn.get('name')}({fn.get('arguments', '')})")
    # OpenAI Responses items carry their payloads in type-specific fields.
    if message.get("type") == "function_call":
        parts.append(f"tool_call {message.get('name')}({message.get('arguments', '')})")
    output = message.get("output")
    if isinstance(output, str | list):
        text = _content_text(output)
        if text:
            parts.append(text)
    return " ".join(parts)


def _content_text(content: str | list[Any]) -> str:
    """Flatten string-or-content-block message content to text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        inner = block.get("content")
        if isinstance(inner, str | list):
            parts.append(_content_text(inner))
    return " ".join(p for p in parts if p)


def _truncate(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` within ``limit`` chars."""
    if len(text) <= limit:
        return text
    marker = " ...[trimmed] "
    keep = max(limit - len(marker), 20)
    head = (keep * 2) // 3
    tail = keep - head
    return text[:head] + marker + text[-tail:]


__all__ = [
    "CTX_ESCALATION_VERDICT",
    "ESCALATION_JUDGE_SYSTEM_PROMPT",
    "EscalationJudgeConfig",
    "EscalationJudgeRequestProcessor",
    "EscalationVerdict",
]
