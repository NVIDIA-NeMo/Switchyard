# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory judge for the escalation router: weak until in trouble, then strong."""

from __future__ import annotations

import hashlib
import logging
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
from switchyard.lib.session_affinity import CTX_SESSION_KEY, SessionAffinity
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: ``ProxyContext.metadata`` key holding the audit record of the judge's
#: decision for this turn: ``{"escalate", "reason", "turn", "source"}``.
CTX_ESCALATION_VERDICT = "_escalation_verdict"

#: Tier labels stamped into ``CTX_DETERMINISTIC_ROUTING_TIER``; must match the
#: labels the profile registers on ``DeterministicRoutingLLMBackend``.
TIER_STRONG = "strong"
TIER_WEAK = "weak"

ESCALATION_JUDGE_SYSTEM_PROMPT = """\
You are an escalation judge inside an agentic coding router. The session
started on the EFFICIENT tier (a cheap but top-class 2026 model). Your
job is to detect when the run is genuinely in trouble so the router can
escalate the rest of the task to the STRONG tier (frontier, expensive).

You see a condensed view of one session: the task framing (system prompt
+ first user message) and the most recent turns of activity (assistant
messages and tool results). Judge the *trajectory* — is the agent making
real progress toward the stated task — not the difficulty of the task
itself. Return exactly one JSON object:

{"escalate": boolean, "reason": "one short sentence naming the pattern"}

Escalation is one-way for the rest of the task and expensive. Escalate
only on a clear PATTERN of trouble, never on a single failed command.
When the evidence is thin or ambiguous, return {"escalate": false}.

# Trouble patterns — escalate when you see these

Repetition and loops (the most common way agent runs die):
- The same command or edit failing 2+ times with materially the same
  error, especially with unrelated changes in between.
- Near-identical tool calls repeated, or the same files re-read, without
  new information gained — including longer cycles (A -> B -> C -> A).
- Fighting the environment: repeatedly invoking a missing executable,
  retrying installs that fail the same way, or trying variations of a
  command the environment has already rejected, instead of adapting.

False progress (looks like progress, is not):
- Declaring success or moving on while the latest visible evidence
  (test output, exit code, error text) shows failure.
- Finishing without running the verification the task specifies, when
  the task states how success is checked (e.g. "make the provided
  tests pass") and running it was possible.
- A reproduction or test the agent wrote that passes trivially without
  exercising the actual issue, then building on that false signal.
- The agent's stated reading of a tool result contradicts what the
  result actually says (treating an error or empty output as success).

Drift and dead ends:
- Recent activity no longer serves the task in the first user message
  (e.g. polishing style while the required feature is unstarted).
- Violating an explicit task constraint (modifying files the task says
  not to touch, changing the tests instead of the code under test).
- Editing or reasoning about code without ever having opened the files
  the errors point to — acting on guessed file contents.
- Contradicting or re-deriving something already established earlier in
  the session (forgetting its own findings).
- Many turns elapsed with nothing durable produced (no successful
  writes, no passing checks) and no visible narrowing of the problem —
  the run is on pace to exhaust its turn budget.

Desperation:
- Giving up: declaring the task impossible, asking to stop, or drifting
  into restating the problem instead of acting on it.
- Destructive flailing: rm -rf, wholesale reinstalls, chmod -R, or
  reverting everything as a reaction to being stuck rather than a
  reasoned step.

# Expected friction — do NOT escalate on these

Agentic coding is full of failures that are part of healthy work:
- A test written to fail first (TDD) or a bug being reproduced on
  purpose.
- A compile, lint, or test error fixed or meaningfully acted on in the
  immediately following turn.
- Exploration dead-ends early in a session (grep with no matches,
  reading a file that turns out to be irrelevant) while the agent is
  still orienting.
- A missing tool handled adaptively (tries `rg`, falls back to `grep`).
- A long-running command (build, install, test suite) that simply has
  not finished, or the agent waiting on information it asked for.

The distinguishing question: is each failure producing new information
that changes the next action? Failing forward is fine; failing in place
is trouble.

# Worked examples (none drawn from any benchmark task set)

* Turn 3; the agent ran the test suite, 4 tests fail, and it is now
  reading the first failing test. -> {"escalate": false} — reproducing
  failures is the job.
* The agent has run `pytest tests/test_api.py` 4 times with the same
  ImportError, editing an unrelated config file between attempts. ->
  {"escalate": true, "reason": "same ImportError 4 times while editing
  unrelated files"}
* `conda` is not installed; the agent has tried `conda install` five
  ways instead of using the `pip` that earlier output showed present.
  -> {"escalate": true, "reason": "fighting missing executable instead
  of adapting"}
* Task: "make the provided integration tests pass." Recent turns:
  renaming variables and reformatting docstrings; tests not run in 8
  turns. -> {"escalate": true, "reason": "drifted to cosmetic edits,
  verification abandoned"}
* The agent says "All tests pass, task complete" but the last visible
  test output shows "2 failed, 11 passed". -> {"escalate": true,
  "reason": "claims success contradicted by latest test output"}
* The agent wrote a reproduction script that exits 0 without invoking
  the code path the issue describes, concluded "bug not reproducible",
  and is wrapping up. -> {"escalate": true, "reason": "reproduction
  never exercised the reported code path"}
* Two turns of edits, one failed build, then a fixed build and a
  passing test. -> {"escalate": false}
* `npm install` has been running for one turn with no output yet. ->
  {"escalate": false} — slow command, not a stall.

Do not emit markdown, commentary, or chain-of-thought — only the JSON
object.
"""


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

    min_judge_turn: int = Field(default=3, ge=1)
    """First conversation turn on which the judge runs. Earlier turns have no
    trajectory to judge and always route weak."""

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
    ) -> None:
        if session_key_depth < 0:
            raise ValueError("session_key_depth must be >= 0")
        self._config = config
        self._affinity = affinity
        self._session_key_depth = session_key_depth
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

        summary = _summarize_for_judge(request, turn=turn, config=self._config)
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
            ctx.metadata[CTX_ESCALATION_VERDICT] = {
                "escalate": False,
                "reason": f"fail_open: {str(exc)[:200]}",
                "turn": turn,
                "source": "fail_open",
            }
            return request

        ctx.metadata[CTX_ESCALATION_VERDICT] = {
            "escalate": verdict.escalate,
            "reason": verdict.reason,
            "turn": turn,
            "source": "judge",
        }
        if verdict.escalate:
            log.info(
                "EscalationJudgeRequestProcessor: escalating to strong tier "
                "(turn %d): %s",
                turn,
                verdict.reason,
            )
            await self._affinity.pin(ctx, request, TIER_STRONG)
            ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = TIER_STRONG
        return request

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
